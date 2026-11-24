"""把模拟会话交给正式群聊和私聊入口，另存只属于本次仿真的消息档案。"""

from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any, Literal
from urllib.parse import quote

from mahjong_agent_runtime import UserMessage
from mahjong_agent_runtime.customer_visible_contract import customer_visible_text_contract_violations
from mahjong_agent_runtime.group_chat import (
    BOARD_TASK_TYPE, BoardEngine, BoardPublicationError, ChannelIdentity, GroupMessage,
    GroupRoomPolicy, L2IntentRouter,
)
from mahjong_agent_runtime.group_chat.semantic_topic_router import SemanticTopicClassifier
from mahjong_agent_runtime.group_chat.topic_service import TopicRoutingService
from mahjong_agent_runtime.models import CustomerProfile, QuotedMessageRef, ScheduledTaskStatus, using_clock
from mahjong_agent_runtime.scheduled_tasks import ScheduledAgentTaskScheduler, ScheduledTaskDeferred

from .group_transport import SimulationMessenger, process_group_message

AGENT_ID = "simulation-agent"
OWNER_ID = "simulation-owner"
_OUTBOUND_PREFIX = "outbound:"


@dataclass(frozen=True, slots=True)
class MessageEnvelope:
    """渠道公开字段；调用方无法注入模型元数据、客户偏好或运行时身份。"""

    message_id: str
    sender_id: str
    text: str
    sent_at: datetime
    kind: Literal["group", "private"]
    recipient_id: str
    quoted_message_id: str | None = None


class _GroupStore:
    """将群通知的默认客户身份解析限制在当前沙盒渠道。"""

    def __init__(self, store, channel: str, members: dict[str, str]) -> None:
        self.store = store
        self.channel = channel
        self.members = members

    def __getattr__(self, name):
        return getattr(self.store, name)

    def get_channel_identity_for_customer(self, customer_id, channel=None):
        if customer_id in self.members and channel in {None, self.channel}:
            return self.store.get_channel_identity_for_customer(customer_id, channel=self.channel)
        return None


class SandboxChannel:
    """固定模拟群和成员，仅向独立 SQLite 档案交付，没有外部发送器。"""

    def __init__(self, runtime, *, room_id: str, identities: dict[str, str], archive_path: Path, write_guard=None) -> None:
        """用公开昵称建立客户和通道权限，不导入行为标签或既往偏好。"""
        if (not isinstance(room_id, str) or not room_id.strip() or room_id in {AGENT_ID, OWNER_ID} or not identities
                or any(not isinstance(key, str) or not key.strip() or not isinstance(name, str) or not name.strip()
                       or key in {room_id, AGENT_ID, OWNER_ID}
                       for key, name in identities.items())):
            raise ValueError("invalid simulation room or members")
        self.runtime = runtime
        self._write_guard = write_guard
        self.room_id = room_id
        self._identities = dict(identities)
        self._namespace = quote(room_id, safe="")
        self._channel = f"im:{self._namespace}"
        self.group_conversation_id = f"im:group:{self._namespace}"
        self.archive_path = Path(archive_path).resolve()
        runtime_path = getattr(runtime.store, "path", None)
        if runtime_path is not None and self.archive_path == Path(runtime_path).resolve():
            raise ValueError("the channel archive requires its own database")
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        with self._database() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS venue_memberships (
                    room_id TEXT PRIMARY KEY, members_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS venue_messages (
                    sequence INTEGER PRIMARY KEY,
                    room_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    sender_id TEXT NOT NULL,
                    recipient_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    text TEXT NOT NULL,
                    sent_at TEXT NOT NULL,
                    quoted_message_id TEXT,
                    source_message_id TEXT,
                    trace_id TEXT NOT NULL,
                    UNIQUE(room_id, message_id)
                );
                CREATE INDEX IF NOT EXISTS venue_message_order ON venue_messages(room_id, sequence);
                CREATE TABLE IF NOT EXISTS venue_receipts (
                    room_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_json TEXT,
                    PRIMARY KEY (room_id, message_id)
                );
            """)
            members_json = json.dumps(self._identities, ensure_ascii=False, sort_keys=True)
            db.execute("INSERT OR IGNORE INTO venue_memberships VALUES (?, ?)", (room_id, members_json))
            membership = db.execute("SELECT members_json FROM venue_memberships WHERE room_id = ?", (room_id,)).fetchone()
            if membership["members_json"] != members_json:
                raise ValueError("simulation room membership differs from its archive")
        store = runtime.store
        policy = store.get_group_room_policy(room_id)
        if policy is not None and policy.channel != self._channel:
            raise ValueError("the room already belongs to a different channel")
        if policy is None:
            store.upsert_group_room_policy(GroupRoomPolicy(room_id=room_id, channel=self._channel))
        customers = store.customers
        for customer_id, name in self._identities.items():
            if customer_id not in customers:
                store.upsert_customer(CustomerProfile(customer_id=customer_id, display_name=name, public_name=name))
            identity = store.get_channel_identity(self._channel, customer_id)
            conversation_id = self.private_conversation_id(customer_id)
            if identity is None:
                store.upsert_channel_identity(ChannelIdentity(
                    channel=self._channel, external_user_id=customer_id, customer_id=customer_id,
                    public_name=name, private_conversation_id=conversation_id,
                    can_private_message=True, is_friend=True,
                ))
            elif identity.customer_id != customer_id or identity.private_conversation_id != conversation_id:
                raise ValueError("existing simulation identity differs from the configured conversation")

    @contextmanager
    def _database(self):
        with closing(sqlite3.connect(self.archive_path)) as db, db:
            db.row_factory = sqlite3.Row
            yield db

    def private_conversation_id(self, customer_id: str) -> str:
        """返回该会员与正式 Agent 的独立业务会话 ID。"""
        if customer_id not in self._identities:
            raise ValueError("unknown simulation member")
        return f"im:dm:{self._namespace}:{quote(customer_id, safe='')}"

    def _conversation(self, message: MessageEnvelope) -> str:
        if (not isinstance(message.message_id, str) or not message.message_id.strip()
                or message.message_id.startswith(_OUTBOUND_PREFIX)):
            raise ValueError("invalid simulation message id")
        if not isinstance(message.text, str) or not message.text.strip():
            raise ValueError("simulation message text is required")
        if not isinstance(message.sent_at, datetime) or message.sent_at.utcoffset() is None:
            raise ValueError("simulation messages require a timezone-aware timestamp")
        if message.kind == "group" and message.recipient_id == self.room_id:
            return self.group_conversation_id
        if message.kind == "private":
            sender, recipient = message.sender_id, message.recipient_id
            if recipient == AGENT_ID and sender in self._identities:
                return self.private_conversation_id(sender)
            if recipient in self._identities and sender != recipient:
                if sender == OWNER_ID:
                    return self.private_conversation_id(recipient)
                peers = ":".join(quote(member, safe="") for member in sorted([sender, recipient]))
                return f"im:peer:{self._namespace}:{peers}"
        raise ValueError("simulation message recipient is outside its channel")

    def send(self, message: MessageEnvelope) -> dict[str, Any]:
        """先归档用户消息，再执行真实入口；会员之间的私信只供双方查看。"""
        if message.sender_id not in self._identities:
            raise ValueError("unknown simulation sender")
        conversation_id = self._conversation(message)
        with self._receipt_lock(message.message_id):
            routes_to_agent = message.kind == "group" or message.recipient_id == AGENT_ID
            direction = "inbound" if routes_to_agent else "peer"
            trace_id, quoted, cached = self._begin(message, conversation_id, direction)
            if cached is not None:
                return cached
            try:
                if message.kind == "group":
                    response = self._handle_group(message, trace_id)
                elif routes_to_agent:
                    response = self._handle_private(message, conversation_id, trace_id, quoted)
                else:
                    response = {"trace_id": trace_id, "status": "delivered", "final_reply": ""}
                return self._finish(message.message_id, response)
            except Exception as exc:
                self._record_failure(message.message_id, type(exc).__name__)
                raise

    def _receipt_lock(self, message_id: str):
        """同消息并发重投等待首次回执；不同消息仍可进入各自真实会话。"""
        scope = json.dumps([str(self.archive_path), self.room_id, message_id], ensure_ascii=False)
        return self.runtime.coordination_manager.lock(f"venue-delivery:{scope}")

    def publish_owner(
        self, *, kind: Literal["group", "private"], recipient_id: str, text: str,
        sent_at: datetime, message_id: str, source_message_id: str | None = None,
    ) -> dict[str, Any]:
        """记录场景中的店员公告或留言，明确标识来源，供顾客看到。"""
        message = MessageEnvelope(message_id, OWNER_ID, text, sent_at, kind, recipient_id)
        conversation_id = self._conversation(message)
        with self._receipt_lock(message_id):
            trace_id, _, cached = self._begin(message, conversation_id, "owner", source_message_id=source_message_id)
            if cached is not None:
                return cached
            return self._finish(message_id, {"trace_id": trace_id, "status": "delivered", "final_reply": ""})

    def publish_board(self, *, trace_id: str, sent_at: datetime) -> dict[str, Any]:
        """领取本房间到期的持久刷新任务，现场渲染并只向本地档案交付。"""
        if not isinstance(sent_at, datetime) or sent_at.utcoffset() is None:
            raise ValueError("board publication requires a timezone-aware timestamp")
        store = self.runtime.store
        recorder = self.runtime.trace_recorder
        traces: dict[str, str] = {}
        snapshots: dict[str, dict[str, Any]] = {}
        delivered_ids: list[str] = []

        def publish(task, task_trace_id):
            traces[task.task_id] = task_trace_id
            self._check_board_policy()

            def deliver(output):
                self._review_board_output(output, task_trace_id)
                # 审查后重读权限与领取凭证，避免输出旧代看板或让暂停权限失效。
                self._check_board_policy()
                current = store.scheduled_tasks.get(task.task_id)
                if (current is None or task.lease_until is None or task.lease_until <= sent_at
                        or not current.matches_claim(expected_attempts=task.attempts,
                                                     expected_lease_until=task.lease_until)):
                    raise BoardPublicationError("group board publication claim is no longer current")
                message_id, inserted = self._archive_board_output(output, sent_at=sent_at, trace_id=task_trace_id)
                if inserted:
                    delivered_ids.append(message_id)
                return message_id

            messenger = SimulationMessenger(store, self.room_id, channel=self._channel,
                conversation_id=self.group_conversation_id, allowed_members=self._identities, record_output=deliver)
            board = BoardEngine(store=store, messenger=messenger, clock=lambda: sent_at)
            snapshot = board.publish(self.room_id, trace_id=task_trace_id,
                                     publication_id=task.payload["publication_id"], channel=self._channel,
                                     conversation_id=self.group_conversation_id)
            if snapshot is not None:
                with self._database() as db:
                    archived = self._message(db, snapshot.external_message_id)
                if archived is None:
                    raise BoardPublicationError("group board snapshot has no sandbox receipt")
                snapshots[task.task_id] = snapshot.to_dict()
            recorder.record(task_trace_id, "group_board_publish_completed", {
                "task_id": task.task_id, "room_id": self.room_id, "trigger_trace_id": trace_id,
                "snapshot": snapshots.get(task.task_id),
            })

        scheduler = ScheduledAgentTaskScheduler(store, publish, recorder, clock=lambda: sent_at)
        with using_clock(lambda: sent_at):
            candidates = sorted(
                (task for task in store.scheduled_tasks.values()
                 if task.task_type == BOARD_TASK_TYPE and task.aggregate_type == "group_room"
                 and task.aggregate_id == self.room_id and task.due_at <= sent_at
                 and task.status in {ScheduledTaskStatus.PENDING, ScheduledTaskStatus.PROCESSING}),
                key=lambda task: (task.due_at, task.task_id),
            )
            for task in candidates:
                scheduler.run_task_once(task.task_id, at=sent_at)
        tasks = store.scheduled_tasks
        receipts = [
            {"task_id": task_id, "trace_id": actual_trace, "status": tasks[task_id].status.value,
             "snapshot": snapshots.get(task_id)}
            for task_id, actual_trace in traces.items()
        ]
        with self._database() as db:
            outputs = [dict(self._message(db, message_id)) for message_id in delivered_ids]
        return {
            "trace_id": trace_id, "tasks": receipts, "snapshots": list(snapshots.values()), "archive_outputs": outputs,
            "status": ("no_due_task" if not receipts else "completed"
                       if all(receipt["status"] == "completed" for receipt in receipts) else "pending"),
        }

    def _check_board_policy(self):
        policy = self.runtime.store.get_group_room_policy(self.room_id)
        if (policy is None or policy.channel != self._channel or not policy.managed
                or not policy.board_enabled or not policy.outbound_enabled):
            raise ScheduledTaskDeferred("group_board_policy_paused")

    def _review_board_output(self, output, trace_id):
        """复用生产投影审查；离线桩中保留认证文本和末端过滤的证据。"""
        # 只调用该类无渠道 I/O 的审查辅助方法，实际交付仍走本地归档回调。
        from scripts.agent_runtime_app import WechatyGroupMessenger

        reviewer = WechatyGroupMessenger(self.runtime)
        if self.runtime.reply_self_review_enabled:
            reviewer._review_group_board(self.room_id, output["text"], metadata=output["metadata"],
                                         trace_id=trace_id, conversation_id=self.group_conversation_id)
            return
        violations = [*reviewer._board_sensitive_rules(output["text"]),
                      *customer_visible_text_contract_violations(output["text"])]
        self.runtime.trace_recorder.record(trace_id, "group_board_terminal_filter", {
            "approved": not violations, "rule_ids": violations, "scope": "offline_deterministic",
        })
        if violations:
            raise BoardPublicationError("offline group board deterministic filter rejected publication")

    def _archive_board_output(self, output, *, sent_at, trace_id):
        """按不可变快照 ID 去重，传输归档和业务快照之间断点可安全重试。"""
        metadata = output["metadata"]
        snapshot_id = metadata.get("business_ref_id")
        if (output["channel"] != "group" or output["recipient_id"] != self.room_id
                or metadata.get("message_type") != "game_board" or not snapshot_id):
            raise BoardPublicationError("sandbox board receipt requires a public board projection")
        message_id = f"{_OUTBOUND_PREFIX}board:{snapshot_id}"
        with self._database() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = self._message(db, message_id)
            if existing is None:
                self._insert_message(db, {
                    "room_id": self.room_id, "conversation_id": self.group_conversation_id, "message_id": message_id,
                    "sender_id": AGENT_ID, "recipient_id": self.room_id, "kind": "group", "direction": "outbound",
                    "text": output["text"], "sent_at": sent_at.astimezone(timezone.utc).isoformat(),
                    "quoted_message_id": None, "source_message_id": None, "trace_id": trace_id,
                })
            elif existing["text"] != output["text"]:
                raise BoardPublicationError("sandbox board receipt conflicts with the initial delivery")
        return message_id, existing is None

    def _begin(self, message, conversation_id, direction, *, source_message_id=None):
        """同群同消息 ID 只能接受一次；未完成回执保留未知结果，禁止再次执行副作用。"""
        stamp = message.sent_at.astimezone(timezone.utc).isoformat()
        trace_scope = json.dumps([self.room_id, message.message_id], ensure_ascii=False)
        trace_id = "trace_" + sha256(trace_scope.encode()).hexdigest()[:24]
        row = {
            "room_id": self.room_id, "conversation_id": conversation_id, "message_id": message.message_id,
            "sender_id": message.sender_id, "recipient_id": message.recipient_id, "kind": message.kind,
            "direction": direction, "text": message.text, "sent_at": stamp,
            "quoted_message_id": message.quoted_message_id, "source_message_id": source_message_id, "trace_id": trace_id,
        }
        with self._database() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = self._message(db, message.message_id)
            if existing is not None:
                if any(existing[field] != value for field, value in row.items()):
                    raise ValueError("simulation message id conflict")
                receipt = db.execute("SELECT status, response_json FROM venue_receipts WHERE room_id = ? AND message_id = ?",
                                     (self.room_id, message.message_id)).fetchone()
                if receipt["status"] != "completed":
                    raise ValueError(f"simulation receipt is {receipt['status']}; inspect the original trace")
                return trace_id, None, {**json.loads(receipt["response_json"]), "delivery_deduplicated": True}
            quoted = None
            for reference in (message.quoted_message_id, source_message_id):
                if reference is not None:
                    quoted = self._message(db, reference)
                    if quoted is None or quoted["conversation_id"] != conversation_id or quoted["sent_at"] > stamp:
                        raise ValueError("simulation reference is outside the message's visible conversation")
            self._insert_message(db, row)
            db.execute("INSERT INTO venue_receipts VALUES (?, ?, 'processing', NULL)",
                       (self.room_id, message.message_id))
        return trace_id, quoted, None

    def _message(self, db, message_id):
        return db.execute("SELECT * FROM venue_messages WHERE room_id = ? AND message_id = ?",
                          (self.room_id, message_id)).fetchone()

    @staticmethod
    def _insert_message(db, row):
        db.execute("""
            INSERT INTO venue_messages (
                room_id, conversation_id, message_id, sender_id, recipient_id, kind,
                direction, text, sent_at, quoted_message_id, source_message_id, trace_id
            ) VALUES (
                :room_id, :conversation_id, :message_id, :sender_id, :recipient_id, :kind,
                :direction, :text, :sent_at, :quoted_message_id, :source_message_id, :trace_id
            )
        """, row)

    def _handle_group(self, message, trace_id):
        # 显式注入同一生产分类器/话题服务，避免测试继承本机 Redis 环境变量。
        store = _GroupStore(self.runtime.store, self._channel, self._identities)
        classifier = SemanticTopicClassifier(self.runtime.llm_client)
        router = L2IntentRouter(store, topic_service=TopicRoutingService(store, classifier))
        messenger = SimulationMessenger(store, self.room_id, channel=self._channel,
            conversation_id=self.group_conversation_id, allowed_members=self._identities,
            record_output=lambda output: self._record_output(output, message, trace_id))
        group_message = GroupMessage(
            channel=self._channel, room_id=self.room_id, conversation_id=self.group_conversation_id,
            sender_external_id=message.sender_id, sender_name=self._identities[message.sender_id],
            text=message.text, message_id=message.message_id, sent_at=message.sent_at,
            quoted_message_id=message.quoted_message_id,
        )
        return process_group_message(self.runtime, group_message, {}, trace_id=trace_id,
                                     messenger=messenger, intent_router=router, clock=lambda: message.sent_at,
                                     coordinate=True)

    def _handle_private(self, message, conversation_id, trace_id, quoted):
        if self._write_guard is not None:
            history = [row for row in self.events_for(message.sender_id) if row["conversation_id"] == conversation_id]
            self._write_guard.authorize(message, conversation_id, history)
        quoted_message = None
        if quoted is not None:
            quoted_sender = {AGENT_ID: "shop-assistant", OWNER_ID: "shop-operator"}.get(
                quoted["sender_id"], quoted["sender_id"])
            quoted_message = QuotedMessageRef(message_id=quoted["message_id"], sender_id=quoted_sender,
                text=quoted["text"], conversation_id=conversation_id)
        result = self.runtime.handle_user_message(UserMessage(
            conversation_id=conversation_id, sender_id=message.sender_id, sender_name=self._identities[message.sender_id],
            text=message.text, message_id=message.message_id, sent_at=message.sent_at, quoted_message=quoted_message,
        ), trace_id=trace_id)
        response = result.to_dict()
        if result.final_reply:
            if not self.runtime.store.claim_assistant_reply_delivery(conversation_id, trace_id=result.trace_id):
                response.update(final_reply="", status="superseded")
                return response
            self._record_output({"channel": "private", "conversation_id": conversation_id,
                                 "recipient_id": message.sender_id, "text": result.final_reply},
                                message, result.trace_id)
            self.runtime.store.mark_assistant_reply_delivered(conversation_id, trace_id=result.trace_id)
        return response

    def _record_output(self, output, source, trace_id):
        """发送成功的含义是对应收件人和正文已落入独立档案。"""
        kind, recipient_id = output["channel"], output["recipient_id"]
        if not ((kind == "group" and recipient_id == self.room_id)
                or (kind == "private" and recipient_id in self._identities)):
            raise ValueError("simulation output recipient is outside the configured population")
        conversation_id = (self.group_conversation_id if kind == "group"
                           else self.private_conversation_id(recipient_id))
        if output["conversation_id"] != conversation_id:
            raise ValueError("simulation output does not belong to its recipient's conversation")
        with self._database() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM venue_messages WHERE room_id = ? AND source_message_id = ? AND direction = 'outbound'",
                               (self.room_id, source.message_id)).fetchone()[0]
            message_id = f"{_OUTBOUND_PREFIX}{trace_id}:{count + 1}"
            self._insert_message(db, {
                "room_id": self.room_id, "conversation_id": conversation_id, "message_id": message_id,
                "sender_id": AGENT_ID, "recipient_id": recipient_id, "kind": kind, "direction": "outbound",
                "text": output["text"], "sent_at": source.sent_at.astimezone(timezone.utc).isoformat(),
                "quoted_message_id": None, "source_message_id": source.message_id,
                "trace_id": output.get("metadata", {}).get("trace_id") or trace_id,
            })
        return message_id

    def _finish(self, message_id, response):
        with self._database() as db:
            rows = db.execute("""
                SELECT * FROM venue_messages WHERE room_id = ? AND source_message_id = ?
                AND direction = 'outbound' ORDER BY sequence
            """, (self.room_id, message_id)).fetchall()
            response = {**response, "archive_outputs": [dict(row) for row in rows]}
            db.execute("UPDATE venue_receipts SET status = 'completed', response_json = ? WHERE room_id = ? AND message_id = ?",
                       (json.dumps(response, ensure_ascii=False), self.room_id, message_id))
        return response

    def _record_failure(self, message_id, error_type):
        with self._database() as db:
            db.execute("UPDATE venue_receipts SET status = 'failed', response_json = ? WHERE room_id = ? AND message_id = ?",
                       (json.dumps({"error_type": error_type}), self.room_id, message_id))

    def events(self) -> list[dict[str, Any]]:
        """读取裁判可见的全群消息账本，按入档序稳定返回。"""
        with self._database() as db:
            rows = db.execute("SELECT * FROM venue_messages WHERE room_id = ? ORDER BY sequence", (self.room_id,)).fetchall()
        return [dict(row) for row in rows]

    def events_for(self, user_id: str) -> list[dict[str, Any]]:
        """按群成员和私聊收件人投影消息，隐去不可见消息的来源编号。"""
        if user_id not in self._identities and user_id != AGENT_ID:
            raise ValueError("unknown simulation viewer")
        with self._database() as db:
            rows = db.execute("""
                SELECT * FROM venue_messages WHERE room_id = ?
                AND (kind = 'group' OR sender_id = ? OR recipient_id = ?
                     OR (direction = 'owner' AND ? = ?)) ORDER BY sequence
            """, (self.room_id, user_id, user_id, user_id, AGENT_ID)).fetchall()
        visible = {row["message_id"] for row in rows}
        return [{**dict(row), "source_message_id": row["source_message_id"] if row["source_message_id"] in visible else None}
                for row in rows]
