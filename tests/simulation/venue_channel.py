"""把模拟会话交给正式群聊和私聊入口，另存只属于本次仿真的消息档案。"""

from __future__ import annotations

from collections.abc import Collection
from contextlib import closing, contextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any, Literal
from urllib.parse import quote

from mahjong_agent_runtime import UserMessage
from mahjong_agent_runtime.customer_visible_contract import (
    CUSTOMER_EXECUTION_FAILURE_REPLY, customer_visible_text_contract_violations,
)
from mahjong_agent_runtime.group_chat import (
    BOARD_TASK_TYPE, BoardEngine, BoardPublicationError, ChannelIdentity, GroupMessage,
    GroupRoomPolicy, L2IntentRouter,
)
from mahjong_agent_runtime.group_chat.full_table_notice import FULL_TABLE_NOTICE_TASK_TYPE, handle_full_table_notice
from mahjong_agent_runtime.group_chat.late_arrival_delivery import (
    LATE_ARRIVAL_DELIVERY_TYPES, handle_late_arrival_delivery,
)
from mahjong_agent_runtime.group_chat.semantic_topic_router import SemanticTopicClassifier
from mahjong_agent_runtime.group_chat.topic_context import verified_board_claim_reference
from mahjong_agent_runtime.group_chat.topic_recovery import pending_group_topic_recovery
from mahjong_agent_runtime.group_chat.topic_service import TopicRoutingService
from mahjong_agent_runtime.input_aggregation import is_background_input_dispatch
from mahjong_agent_runtime.models import (
    DEFAULT_TZ, ConversationRole, ConversationTurn, CustomerProfile, QuotedMessageRef, ScheduledTaskStatus, using_clock,
)
from mahjong_agent_runtime.scheduled_tasks import ScheduledAgentTaskScheduler, ScheduledTaskDeferred

from .group_transport import SimulationMessenger, process_group_message

AGENT_ID = "simulation-agent"
OWNER_ID = "simulation-owner"
_OUTBOUND_PREFIX = "outbound:"
_PRIVATE_NOTICE_TYPES = LATE_ARRIVAL_DELIVERY_TYPES | {FULL_TABLE_NOTICE_TASK_TYPE}
_DEDUPLICATED_PRIVATE_TYPES = _PRIVATE_NOTICE_TYPES | {"input_gate_wait"}


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
    """在独立档案中保存模拟群、店家好友以及各自的消息交付。"""

    def __init__(
        self, runtime, *, room_id: str, identities: dict[str, str], archive_path: Path, write_guard=None,
        owner_friend_ids: Collection[str] | None = None, show_verified_write_scope: bool = False,
        production_private_routing: bool = False, production_private_aggregation: bool = False,
    ) -> None:
        """用公开昵称建立客户和通道权限，不导入行为标签或既往偏好。"""
        if (not isinstance(room_id, str) or not room_id.strip() or room_id in {AGENT_ID, OWNER_ID} or not identities
                or any(not isinstance(key, str) or not key.strip() or not isinstance(name, str) or not name.strip()
                       or key in {room_id, AGENT_ID, OWNER_ID}
                       for key, name in identities.items())):
            raise ValueError("invalid simulation room or members")
        configured_friends = frozenset(owner_friend_ids) if owner_friend_ids is not None else None
        if configured_friends is not None and not configured_friends <= identities.keys():
            raise ValueError("simulation owner friends must be known members")
        if show_verified_write_scope and write_guard is None:
            raise ValueError("simulation planner write scope requires the independent write guard")
        if production_private_aggregation and not production_private_routing:
            raise ValueError("simulation private aggregation requires production private routing")
        self.runtime = runtime
        self._write_guard = write_guard
        self._show_verified_write_scope = show_verified_write_scope
        self._production_private_routing = production_private_routing
        self._production_private_aggregation = production_private_aggregation
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
                CREATE TABLE IF NOT EXISTS venue_owner_friendships (
                    room_id TEXT PRIMARY KEY, friend_ids_json TEXT NOT NULL
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
            initial_friends = self._identities if configured_friends is None else configured_friends
            db.execute("INSERT OR IGNORE INTO venue_owner_friendships VALUES (?, ?)",
                       (room_id, json.dumps(sorted(initial_friends), ensure_ascii=False)))
            friendship = db.execute("SELECT friend_ids_json FROM venue_owner_friendships WHERE room_id = ?", (room_id,)).fetchone()
            self._owner_friend_ids = frozenset(json.loads(friendship["friend_ids_json"]))
            if (not self._owner_friend_ids <= self._identities.keys()
                    or configured_friends is not None and configured_friends != self._owner_friend_ids):
                raise ValueError("simulation owner friendships differ from the archive")
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
            is_friend = customer_id in self._owner_friend_ids
            if identity is None:
                store.upsert_channel_identity(ChannelIdentity(
                    channel=self._channel, external_user_id=customer_id, customer_id=customer_id,
                    public_name=name, private_conversation_id=conversation_id,
                    can_private_message=is_friend, is_friend=is_friend,
                ))
            elif identity.customer_id != customer_id or identity.private_conversation_id != conversation_id:
                raise ValueError("existing simulation identity differs from the configured conversation")
            elif identity.can_private_message != is_friend or identity.is_friend != is_friend:
                raise ValueError("existing simulation identity differs from the configured owner friendship")

    @property
    def owner_friends(self) -> dict[str, str]:
        """返回老板可见的好友昵称，使用稳定账号句柄区分同名顾客。"""
        return {customer_id: name for customer_id, name in self._identities.items()
                if customer_id in self._owner_friend_ids}

    @property
    def group_member_ids(self) -> frozenset[str]:
        """返回本房间账本确认的成员名册，供角色调度使用。"""
        return frozenset(self._identities)

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
            if recipient in {AGENT_ID, OWNER_ID} and sender in self._owner_friend_ids:
                return self.private_conversation_id(sender)
            if recipient in self._identities and sender != recipient:
                if sender == OWNER_ID:
                    if recipient in self._owner_friend_ids:
                        return self.private_conversation_id(recipient)
                    raise ValueError("simulation owner private recipient is not a friend")
                peers = ":".join(quote(member, safe="") for member in sorted([sender, recipient]))
                return f"im:peer:{self._namespace}:{peers}"
        raise ValueError("simulation message recipient is outside its channel")

    def send(self, message: MessageEnvelope) -> dict[str, Any]:
        """先归档用户消息，再执行真实入口；会员之间的私信只供双方查看。"""
        if message.sender_id not in self._identities:
            raise ValueError("unknown simulation sender")
        # 对顾客是同一个老板账号，账本沿用原有 Agent 端点以兼容回放和去重。
        if message.kind == "private" and message.recipient_id == OWNER_ID:
            message = replace(message, recipient_id=AGENT_ID)
        conversation_id = self._conversation(message)
        with self._receipt_lock(message.message_id):
            routes_to_agent = message.kind == "group" or message.recipient_id == AGENT_ID
            direction = "inbound" if routes_to_agent else "peer"
            trace_id, quoted, cached = self._begin(message, conversation_id, direction)
            if cached is not None:
                return cached
            try:
                delivery_source = message
                if message.kind == "group":
                    response, delivery_source = self._handle_group(message, trace_id)
                elif routes_to_agent:
                    response, delivery_source = self._handle_private(message, conversation_id, trace_id, quoted)
                else:
                    response = {"trace_id": trace_id, "status": "delivered", "final_reply": ""}
                if routes_to_agent:
                    self._deliver_due_private_notices(delivery_source, trace_id)
                return self._finish(message.message_id, response)
            except Exception as exc:
                self._record_failure(message.message_id, type(exc).__name__)
                raise

    def _receipt_lock(self, message_id: str):
        """同消息并发重投等待首次回执；不同消息仍可进入各自真实会话。"""
        scope = json.dumps([str(self.archive_path), self.room_id, message_id], ensure_ascii=False)
        return self.runtime.coordination_manager.lock(f"venue-delivery:{scope}")

    def _deliver_due_private_notices(self, source, trace_id: str) -> None:
        """按虚拟时钟领取本房的通知任务，复用生产处理器和真实归档收据。"""
        store = self.runtime.store
        scheduler = self._private_notice_scheduler(source, trace_id)
        with using_clock(lambda: source.sent_at):
            for task in store.due_scheduled_tasks(at=source.sent_at, limit=50):
                if task.task_type in _PRIVATE_NOTICE_TYPES and task.payload.get("room_id") == self.room_id:
                    scheduler.run_task_once(task.task_id, at=source.sent_at)

    def _private_notice_scheduler(self, source, trace_id: str):
        """真实生产通知处理器复用沙盒的精确私聊收件和持久收据。"""
        store = self.runtime.store
        messenger = SimulationMessenger(store, self.room_id, channel=self._channel,
            conversation_id=self.group_conversation_id, allowed_members=self._identities,
            record_output=lambda output: self._record_output(output, source, trace_id))

        def deliver(task, task_trace_id):
            """按生产任务类型分发，沙盒通道只把实际交付记录给正确收件人。"""
            if task.task_type == FULL_TABLE_NOTICE_TASK_TYPE:
                handle_full_table_notice(self.runtime, messenger, task, task_trace_id)
            else:
                handle_late_arrival_delivery(self.runtime, messenger, task, task_trace_id)

        return ScheduledAgentTaskScheduler(store, deliver, self.runtime.trace_recorder, clock=lambda: source.sent_at)

    def next_late_arrival_delivery(self):
        """无新聊天时也推进当前房间迟到协商的待发以及可重试交付。"""
        return min(
            (((task.lease_until or task.due_at) if task.status == ScheduledTaskStatus.PROCESSING else task.due_at,
              task.task_id)
             for task in self.runtime.store.scheduled_tasks.values()
             if task.task_type in LATE_ARRIVAL_DELIVERY_TYPES and task.payload.get("room_id") == self.room_id
             and task.status in {ScheduledTaskStatus.PENDING, ScheduledTaskStatus.PROCESSING}),
            default=None,
        )

    def deliver_late_arrival_notice(self, task_id: str, *, source_message_id: str, sent_at: datetime):
        """由虚拟时钟重试到期交付并返回真实档案行，未送达不会触发演员。"""
        task = self.runtime.store.scheduled_tasks.get(task_id)
        if task is None or task.task_type not in LATE_ARRIVAL_DELIVERY_TYPES or task.payload.get("room_id") != self.room_id:
            return []
        source = MessageEnvelope(source_message_id, AGENT_ID, "", sent_at, "private", task.subject_id)
        with using_clock(lambda: sent_at):
            scheduler = self._private_notice_scheduler(source, f"venue:{source_message_id}")
            scheduler.run_task_once(task_id, at=sent_at)
        with self._database() as db:
            output = self._message(db, f"{_OUTBOUND_PREFIX}{task.idempotency_key}")
        if output is None or output["source_message_id"] != source_message_id:
            return []
        return [dict(output)]

    def deliver_group_history(self, message: MessageEnvelope, *, synthetic: bool = False) -> dict[str, Any]:
        """投递历史群聊或单条件模拟消息，两种来源分别归档。"""
        if message.kind != "group" or message.sender_id not in self._identities:
            raise ValueError("simulation group history requires a known group member")
        conversation_id = self._conversation(message)
        with self._receipt_lock(message.message_id):
            trace_id, _, cached = self._begin(message, conversation_id, "scenario" if synthetic else "history")
            if cached is not None:
                return cached
            return self._finish(message.message_id, {"trace_id": trace_id, "status": "delivered", "final_reply": ""})

    def deliver_private_history(self, message: MessageEnvelope) -> dict[str, Any]:
        """回填私聊来话及店员回复，复用正式会话历史而不执行旧请求。"""
        from_owner = message.sender_id == OWNER_ID and message.recipient_id in self._owner_friend_ids
        from_customer = message.sender_id in self._owner_friend_ids and message.recipient_id == AGENT_ID
        if message.kind != "private" or not (from_owner or from_customer) or message.quoted_message_id:
            raise ValueError("simulation private history requires the owner and one known friend")
        conversation = self._conversation(message)
        with self._receipt_lock(message.message_id):
            trace_id, _, cached = self._begin(message, conversation, "history")
            if cached is not None:
                return cached
            store = self.runtime.store
            try:
                if from_owner:
                    store.record_owner_reply(conversation, customer_id=message.recipient_id, turn=ConversationTurn(
                        role=ConversationRole.ASSISTANT, content=message.text, trace_id=trace_id, occurred_at=message.sent_at,
                        metadata={"source_message_id": message.message_id, "reply_author": "owner",
                                  "delivery_status": "delivered", "owner_reply_recorded_at": message.sent_at.isoformat()},
                    ))
                else:
                    store.append_user_turn(UserMessage(conversation, message.sender_id, self._identities[message.sender_id],
                        message.text, message_id=message.message_id, sent_at=message.sent_at), trace_id)
            except Exception as exc:
                self._record_failure(message.message_id, type(exc).__name__)
                raise
            return self._finish(message.message_id, {"trace_id": trace_id, "status": "delivered", "final_reply": ""})

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
            # 群里没有新来话时也要先在当前虚拟时间结算到期，再渲染可报名牌局。
            store.active_games()

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
        """规范化原群来话并按虚拟时钟消费共享的分类重试；外发始终绑定该来话。"""
        # 显式注入同一生产分类器/话题服务，避免测试继承本机 Redis 环境变量。
        store = _GroupStore(self.runtime.store, self._channel, self._identities)
        classifier = SemanticTopicClassifier(self.runtime.llm_client, trace_recorder=self.runtime.trace_recorder)
        router = L2IntentRouter(store, topic_service=TopicRoutingService(store, classifier))
        delivery_source = message
        messenger = SimulationMessenger(store, self.room_id, channel=self._channel,
            conversation_id=self.group_conversation_id, allowed_members=self._identities,
            record_output=lambda output: self._record_output(output, delivery_source, trace_id))
        group_message = GroupMessage(
            channel=self._channel, room_id=self.room_id, conversation_id=self.group_conversation_id,
            sender_external_id=message.sender_id, sender_name=self._identities[message.sender_id],
            text=message.text, message_id=message.message_id, sent_at=message.sent_at,
            quoted_message_id=message.quoted_message_id,
        )
        checked_claim = False

        @contextmanager
        def member_write_scope(target):
            # 编号认领走生产原子入口。群转私聊的自然报名需对原话另作独立授权，不能把路由分类当授权。
            nonlocal checked_claim
            if (self._write_guard is None or target.sender_id != message.sender_id
                    or target.message_id != message.message_id
                    or target.conversation_id != self.private_conversation_id(message.sender_id)
                    or not (verified_board_claim_reference(store, target) or {}).get("game_id")):
                yield
                return
            authorization = self._write_guard.authorize(
                message, target.conversation_id, self.events_for(message.sender_id),
                evidence_conversation_id=self.group_conversation_id,
            )
            checked_claim = True
            scope = (self._write_guard.planner_scope(authorization)
                     if self._show_verified_write_scope else nullcontext())
            with scope:
                yield

        def dispatch_group(topic_retry_batch=None):
            """保持源消息时间语义，回复收据使用实际重试时刻。"""
            with using_clock(lambda: delivery_source.sent_at):
                return process_group_message(self.runtime, group_message, {}, trace_id=trace_id,
                    messenger=messenger, intent_router=router, clock=lambda: delivery_source.sent_at,
                    coordinate=True, member_write_scope=member_write_scope, topic_retry_batch=topic_retry_batch)

        response = dispatch_group()
        initial_routing = response["group_processing"]["routing"]
        if initial_routing["action"] == "group_topic_retry_queued":
            batch = pending_group_topic_recovery(store, group_message)
            if batch is not None:
                delivery_source = replace(message, sent_at=max(message.sent_at, batch.quiet_deadline))
                response = dispatch_group(topic_retry_batch=batch)
                response["group_processing"]["topic_recovery"] = initial_routing
        if checked_claim:
            response["group_processing"]["claim_write_preflight"] = True
        return response, delivery_source

    def _handle_private(self, message, conversation_id, trace_id, quoted):
        """把各自完整沙盒文本交给生产分流或显式研究对照，交付只写模拟收据。"""
        scope = nullcontext()
        if self._write_guard is not None:
            history = [row for row in self.events_for(message.sender_id) if row["conversation_id"] == conversation_id]
            authorization = self._write_guard.authorize(message, conversation_id, history)
            if self._show_verified_write_scope:
                scope = self._write_guard.planner_scope(authorization)
        quoted_message = None
        if quoted is not None:
            quoted_sender = {AGENT_ID: "shop-assistant", OWNER_ID: "shop-operator"}.get(
                quoted["sender_id"], quoted["sender_id"])
            quoted_message = QuotedMessageRef(message_id=quoted["message_id"], sender_id=quoted_sender,
                text=quoted["text"], conversation_id=conversation_id)
        business_message = UserMessage(
            conversation_id=conversation_id, sender_id=message.sender_id, sender_name=self._identities[message.sender_id],
            text=message.text.strip() if self._production_private_routing else message.text, message_id=message.message_id,
            sent_at=message.sent_at, quoted_message=quoted_message,
        )
        delivery_source = message
        with scope:
            if self._production_private_routing:
                decision, delivery_source = self._route_private_ingress(business_message, message, trace_id)
                response = self._private_route_response(decision, trace_id)
            else:
                response = self.runtime.handle_user_message(business_message, trace_id=trace_id).to_dict()
        self._deliver_private_reply(response, conversation_id, delivery_source)
        return response, delivery_source

    def _private_route_response(self, decision, trace_id):
        """应用路由与业务结果分别留证，等待补字本身没有顾客可见回复。"""
        response = dict(decision.get("agent_result") or {
            "trace_id": trace_id, "status": decision.get("input_status", "ignored"), "final_reply": "",
        })
        response["channel_routing"] = decision["audit"]
        if self._production_private_aggregation:
            response.update(input_status=decision["input_status"], input_batch=decision["input_batch"])
        return response

    def _deliver_private_reply(self, response, conversation_id, source):
        """即时和到期答复沿同一发送版本门禁进入模拟账本。"""
        if response["final_reply"]:
            reply_trace = response["trace_id"]
            if not self.runtime.store.claim_assistant_reply_delivery(conversation_id, trace_id=reply_trace):
                response.update(final_reply="", status="superseded")
            else:
                self._record_output({"channel": "private", "conversation_id": conversation_id,
                                     "recipient_id": source.sender_id, "text": response["final_reply"]},
                                    source, reply_trace)
                self.runtime.store.mark_assistant_reply_delivered(conversation_id, trace_id=reply_trace)

    def _private_gate_wait_delivery(self, source, conversation_id, trace_id):
        """仅为已归档的同一条私聊回执登记入口等待提示。"""
        def deliver_wait(waiting_message, batch, wait_trace):
            """按原私聊的来话 ID 记一条可复用的真正沙盒等待回执。"""
            if (waiting_message.metadata.get("source_message_id") != source.message_id or batch.sender_id != source.sender_id
                    or batch.conversation_id != conversation_id or wait_trace != trace_id):
                return False
            receipt = self._record_output({
                "channel": "private", "conversation_id": conversation_id, "recipient_id": source.sender_id,
                "text": CUSTOMER_EXECUTION_FAILURE_REPLY, "metadata": {
                    "message_type": "input_gate_wait", "draft_id": f"input-gate-wait:{trace_id}",
                },
            }, source, trace_id)
            return bool(receipt)
        return deliver_wait

    def _route_private_ingress(self, business_message, source, trace_id):
        """复用正式文字分流，使用独立账本和虚拟时钟执行沙盒收发。"""
        from scripts.agent_runtime_app import (
            build_wechaty_message_metadata, dispatch_pending_input_batch, route_normalized_wechat_message,
            route_user_message_with_aggregation,
        )

        business_message.metadata = build_wechaty_message_metadata({
            "message_id": source.message_id, "sender_id": source.sender_id, "message_type": 7,
        }, text=business_message.text, text_source="text")
        conversation_id = business_message.conversation_id
        deliver_wait = self._private_gate_wait_delivery(source, conversation_id, trace_id)
        audit = {"channel": self._channel, "routed_to_agent": False}
        with using_clock(lambda: source.sent_at):
            if self._production_private_aggregation:
                return route_user_message_with_aggregation(
                    self.runtime, business_message, trace_id=trace_id, channel=self._channel,
                    audit=audit, allow_immediate_ack=False, on_gate_wait=deliver_wait,
                ), source
            decision = route_normalized_wechat_message(
                self.runtime, business_message, trace_id=trace_id, channel=self._channel,
                audit=audit, quiet_period_elapsed=True, on_gate_wait=deliver_wait,
            )
        queued = decision.get("input_batch") or {}
        if not (decision.get("background_processing")
                and queued.get("decision", {}).get("gate_recovery", {}).get("route") == "complete_message"):
            return decision, source
        batch = self.runtime.store.pending_input_batch(conversation_id, source.sender_id)
        if batch is None or (batch.batch_id, batch.version) != (queued["batch_id"], queued["version"]):
            return {**decision, "input_status": "superseded"}, source

        # 到期才调用正式分流重试；新来话已覆盖该批时不挪用它的回执。
        delivery_source = replace(source, sent_at=max(source.sent_at, batch.quiet_deadline))
        with using_clock(lambda: delivery_source.sent_at):
            decision = dispatch_pending_input_batch(
                self.runtime, batch, trace_id=trace_id, quiet_period_elapsed=True,
                trigger="input_gate_recovery", audit=audit,
            )
        return decision, delivery_source

    def dispatch_due_private_inputs(self, *, at: datetime) -> list[dict[str, Any]]:
        """推进本房间已到期的私聊批次，实际答复归到批次最后一条来话。"""
        from scripts.agent_runtime_app import dispatch_pending_input_batch

        if not self._production_private_aggregation:
            return []
        if at.utcoffset() is None:
            raise ValueError("simulation time must include a timezone")
        at = at.astimezone(DEFAULT_TZ)
        store = self.runtime.store
        receipts = []
        for batch in store.due_pending_input_batches(at=at, limit=max(100, len(self._owner_friend_ids))):
            if (batch.sender_id not in self._owner_friend_ids
                    or batch.conversation_id != self.private_conversation_id(batch.sender_id)):
                continue
            source_id = batch.fragments[-1]["message_id"]
            with self._receipt_lock(source_id), using_clock(lambda: at):
                current = store.pending_input_batch(batch.conversation_id, batch.sender_id)
                if current is None or (current.batch_id, current.version, current.status) != (
                    batch.batch_id, batch.version, batch.status,
                ):
                    continue
                with self._database() as db:
                    row = self._message(db, source_id)
                if (row is None or row["direction"] != "inbound" or row["conversation_id"] != batch.conversation_id
                        or row["kind"] != "private" or row["sender_id"] != batch.sender_id or row["recipient_id"] != AGENT_ID):
                    raise ValueError("due private input has no authenticated simulation receipt")
                source = MessageEnvelope(source_id, batch.sender_id, row["text"], at, "private", AGENT_ID,
                                         row["quoted_message_id"])
                trace_id = row["trace_id"]
                recovery = batch.decision.get("gate_recovery") or {}
                background = is_background_input_dispatch(batch.decision)
                decision = dispatch_pending_input_batch(
                    self.runtime, batch, trace_id=trace_id, quiet_period_elapsed=not background,
                    trigger="input_gate_recovery" if recovery else batch.decision["dispatch_mode"] if background else "quiet_period_elapsed",
                    audit={"channel": self._channel, "routed_to_agent": False},
                    gate_decision_override=batch.decision if background else None,
                    allow_immediate_ack=False,
                    on_gate_wait=self._private_gate_wait_delivery(source, batch.conversation_id, trace_id),
                )
                response = self._private_route_response(decision, trace_id)
                self._deliver_private_reply(response, batch.conversation_id, source)
                self._deliver_due_private_notices(source, trace_id)
                receipts.append(self._finish(source_id, response))
        return receipts

    def _record_output(self, output, source, trace_id):
        """发送成功的含义是对应收件人和正文已落入独立档案。"""
        kind, recipient_id = output["channel"], output["recipient_id"]
        if not ((kind == "group" and recipient_id == self.room_id)
                or (kind == "private" and recipient_id in self._owner_friend_ids)):
            raise ValueError("simulation output recipient is outside the configured population")
        conversation_id = (self.group_conversation_id if kind == "group"
                           else self.private_conversation_id(recipient_id))
        if output["conversation_id"] != conversation_id:
            raise ValueError("simulation output does not belong to its recipient's conversation")
        with self._database() as db:
            db.execute("BEGIN IMMEDIATE")
            metadata = output.get("metadata") or {}
            draft_id = metadata.get("draft_id", "")
            if metadata.get("message_type") in _DEDUPLICATED_PRIVATE_TYPES and draft_id:
                message_id = f"{_OUTBOUND_PREFIX}{draft_id}"
                existing = self._message(db, message_id)
                if existing is not None:
                    if existing["recipient_id"] != recipient_id or existing["text"] != output["text"]:
                        raise ValueError("private notice receipt conflicts with its first delivery")
                    return message_id
            else:
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

    def completed_peer_message(self, message_id: str) -> dict[str, Any] | None:
        """由调度器核对已完成的搭子回执，崩溃留下的处理中记录不能触发来话。"""
        with self._database() as db:
            row = db.execute("""
                SELECT message.* FROM venue_messages AS message
                JOIN venue_receipts AS receipt
                    ON receipt.room_id = message.room_id AND receipt.message_id = message.message_id
                WHERE message.room_id = ? AND message.message_id = ? AND message.kind = 'private'
                    AND message.direction = 'peer' AND receipt.status = 'completed'
            """, (self.room_id, message_id)).fetchone()
        return dict(row) if row is not None else None

    def events_for(self, user_id: str) -> list[dict[str, Any]]:
        """按群成员和私聊收件人投影消息，隐去不可见消息的来源编号。"""
        if user_id == OWNER_ID:
            user_id = AGENT_ID
        if user_id not in self._identities and user_id != AGENT_ID:
            raise ValueError("unknown simulation viewer")
        with self._database() as db:
            rows = db.execute("""
                SELECT message.* FROM venue_messages AS message WHERE message.room_id = ?
                AND (kind = 'group' OR sender_id = ? OR recipient_id = ?
                     OR (direction = 'owner' AND ? = ?))
                AND (direction != 'peer' OR EXISTS (
                    SELECT 1 FROM venue_receipts AS receipt WHERE receipt.room_id = message.room_id
                    AND receipt.message_id = message.message_id AND receipt.status = 'completed'
                )) ORDER BY sequence
            """, (self.room_id, user_id, user_id, user_id, AGENT_ID)).fetchall()
        visible = {row["message_id"] for row in rows}
        return [{**dict(row), "source_message_id": row["source_message_id"] if row["source_message_id"] in visible else None}
                for row in rows]

    def guarded_agent_input_count(self) -> int:
        """按实际入站及已归档群转私聊预核计数，原有回执没有预核标记时不推定有授权。"""
        with self._database() as db:
            return db.execute("""
                SELECT COUNT(*) FROM venue_messages AS message
                JOIN venue_receipts AS receipt USING (room_id, message_id)
                WHERE message.room_id = ? AND message.direction = 'inbound'
                AND (message.kind = 'private' OR (
                    receipt.status = 'completed'
                    AND json_extract(receipt.response_json, '$.group_processing.claim_write_preflight') = 1
                ))
            """, (self.room_id,)).fetchone()[0]
