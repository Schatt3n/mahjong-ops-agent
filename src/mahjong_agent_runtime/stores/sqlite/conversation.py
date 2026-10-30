"""实现 SQLite 存储层的会话数据访问与事务操作。

"""

from __future__ import annotations

from typing import Any
from copy import deepcopy
from datetime import datetime
from ...models import (
    ConversationCheckpoint,
    ConversationRole,
    ConversationTaskContext,
    ConversationTurn,
    DEFAULT_TZ,
    InviteStatus,
    OutboundDraftStatus,
    ScheduledTaskStatus,
    StateTransition,
    UserMessage,
    new_id,
)
from .serialization import (
    _checkpoint_from_payload,
    _dumps,
    _loads,
    _now_iso,
    _task_context_from_payload,
    _task_memory_from_payload,
    _turn_from_payload,
)

class SQLiteConversationStoreMixin:
    """持久化对话轮次、任务上下文、记忆和检查点。"""

    __slots__ = ()

    def accept_user_input(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        message_key: str,
        preparation: dict[str, Any] | None = None,
    ) -> tuple[int, StateTransition | None, bool, dict[str, int], list[StateTransition]]:
        """以立即写事务原子接收原话和版本，跨连接重复消息也只能接收一次。

        接收轮次是持久化幂等证据；重复返回其原版本。这里不调用会提交连接的公开
        追加/版本接口，确保原话、版本、旧pending废弃及迁移记录一起提交或回滚，也不信任用户任务归属。
        """
        if not isinstance(message_key, str) or not message_key.strip():
            raise ValueError("message_key must be a non-empty string")
        version_key = message.conversation_id or "default"
        with self._write_transaction():
            rows = self._connection.execute(
                """
                SELECT payload FROM runtime_conversation_turns
                WHERE conversation_id = ? AND role = ?
                ORDER BY id DESC
                """,
                (message.conversation_id, ConversationRole.USER.value),
            ).fetchall()
            for row in rows:
                payload = _loads(row["payload"])
                metadata = payload.get("metadata") or {}
                if payload.get("sender_id") == message.sender_id and metadata.get("input_receipt_key") == message_key:
                    return int(metadata["conversation_version"]), None, False, {
                        "invite_drafts": 0, "outbound_message_drafts": 0, "assistant_replies": 0,
                    }, []
            row = self._connection.execute(
                """
                INSERT INTO runtime_conversation_versions(conversation_id, version, updated_at)
                VALUES (?, 1, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    version=runtime_conversation_versions.version + 1,
                    updated_at=excluded.updated_at
                RETURNING version
                """,
                (version_key, _now_iso()),
            ).fetchone()
            if row is None:
                raise RuntimeError("conversation version update returned no row")
            version = int(row["version"])
            task_context = self.current_task_context(message.conversation_id, message.sender_id)
            metadata = dict(message.metadata or {})
            metadata.pop("_trusted_source_task_context_id", None)
            metadata.pop("task_context_id", None)
            metadata.pop("input_receipt_quoted_message", None)
            if message.quoted_message is not None:
                metadata["input_receipt_quoted_message"] = message.quoted_message.to_dict()
            metadata.update({
                "input_receipt_key": message_key,
                "input_receipt_preparation": deepcopy(preparation or {}),
                "input_receipt_hooks_dispatched": False,
                "conversation_version": version,
                "source_message_id": message.message_id or message_key,
            })
            if task_context is not None:
                metadata["task_context_id"] = task_context.task_context_id
            turn = ConversationTurn(
                role=ConversationRole.USER,
                content=message.text,
                trace_id=trace_id,
                sender_id=message.sender_id,
                sender_name=message.sender_name,
                metadata=metadata,
                occurred_at=message.sent_at,
            )
            self._connection.execute(
                """
                INSERT INTO runtime_conversation_turns(
                    conversation_id, trace_id, role, task_context_id,
                    source_message_id, occurred_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.conversation_id, trace_id, turn.role.value,
                    str(metadata.get("task_context_id") or ""), str(metadata["source_message_id"]),
                    turn.occurred_at.isoformat(), _dumps(turn.to_dict()),
                ),
            )
            transition = StateTransition(
                "conversation_version", version_key, str(version - 1), str(version),
                "user_message_received", trace_id,
            )
            self._append_transition(transition)
            counts, superseded = self._supersede_pending_outputs_in_transaction(
                message.conversation_id, sender_id=message.sender_id, trace_id=trace_id,
                reason="new_user_message_superseded_previous_pending_output",
            )
            return version, transition, True, counts, superseded

    def claim_user_input_hooks(self, conversation_id: str, *, sender_id: str, message_key: str) -> dict[str, Any] | None:
        """在事务中声明一次通知，后续 Hook 不持数据库或输入接收锁。"""
        with self._write_transaction():
            rows = self._connection.execute("SELECT id, payload FROM runtime_conversation_turns WHERE conversation_id = ? AND role = ?",
                                            (conversation_id, ConversationRole.USER.value)).fetchall()
            for row in rows:
                turn = _turn_from_payload(_loads(row["payload"]))
                if turn.sender_id != sender_id or turn.metadata.get("input_receipt_key") != message_key:
                    continue
                if turn.metadata.get("input_receipt_hooks_dispatched") is not False:
                    return None
                turn.metadata = {**turn.metadata, "input_receipt_hooks_dispatched": True}
                self._connection.execute("UPDATE runtime_conversation_turns SET payload = ? WHERE id = ?", (_dumps(turn.to_dict()), row["id"]))
                return {"trace_id": turn.trace_id, "preparation": deepcopy(turn.metadata["input_receipt_preparation"])}
        return None

    def claim_assistant_reply_delivery(self, conversation_id: str, *, trace_id: str) -> bool:
        """用事务核对待发版本并声明交付开始，不在事务内执行网络或文件展示操作。"""
        with self._write_transaction():
            rows = self._connection.execute(
                "SELECT id, payload FROM runtime_conversation_turns WHERE conversation_id = ? AND trace_id = ? AND role = ?",
                (conversation_id, trace_id, ConversationRole.ASSISTANT.value),
            ).fetchall()
            for row in rows:
                turn = _turn_from_payload(_loads(row["payload"]))
                status = turn.metadata.get("delivery_status")
                if status == "delivered":
                    return True
                # 重试尚未确认成功的交付仍必须属于当前版本，不能永久凭 dispatching 绕过新输入。
                if status not in {"pending_operator_send", "dispatching"} or turn.metadata.get("conversation_version") != self.conversation_version(conversation_id):
                    return False
                if "input_window" in turn.metadata:
                    source = turn.metadata["input_window"]
                    if (not isinstance(source, dict) or not isinstance(source.get("sender_id"), str)
                            or not source["sender_id"] or not isinstance(source.get("batch_id"), str)
                            or not source["batch_id"] or type(source.get("batch_version")) is not int):
                        return False
                    current = self.pending_input_batch(conversation_id, source["sender_id"])
                    if current is None or current.batch_id != source["batch_id"] or current.version != source["batch_version"]:
                        return False
                if status == "dispatching":
                    return True
                turn.metadata = {**turn.metadata, "delivery_status": "dispatching"}
                self._connection.execute("UPDATE runtime_conversation_turns SET payload = ? WHERE id = ?", (_dumps(turn.to_dict()), row["id"]))
                return True
        return False

    def mark_assistant_reply_delivered(self, conversation_id: str, *, trace_id: str) -> bool:
        """原子确认精确回复的本地交付事实，不改历史正文，也不把已过期记录恢复为可发。"""
        delivered = False
        with self._write_transaction():
            rows = self._connection.execute(
                """
                SELECT id, payload FROM runtime_conversation_turns
                WHERE conversation_id = ? AND trace_id = ? AND role = ?
                """,
                (conversation_id, trace_id, ConversationRole.ASSISTANT.value),
            ).fetchall()
            for row in rows:
                turn = _turn_from_payload(_loads(row["payload"]))
                if turn.metadata.get("delivery_status") in {"pending_operator_send", "dispatching"}:
                    turn.metadata = {**turn.metadata, "delivery_status": "delivered"}
                    self._connection.execute(
                        "UPDATE runtime_conversation_turns SET payload = ? WHERE id = ?",
                        (_dumps(turn.to_dict()), row["id"]),
                    )
                delivered = delivered or turn.metadata.get("delivery_status") == "delivered"
        return delivered

    def append_user_turn(self, message, trace_id: str) -> None:
        """追加用户turn，保留原有时间顺序和审计信息。"""
        task_context = self.current_task_context(message.conversation_id, message.sender_id)
        metadata = dict(getattr(message, "metadata", {}) or {})
        metadata.setdefault("source_message_id", str(getattr(message, "message_id", "") or ""))
        trusted_task_context_id = str(metadata.pop("_trusted_source_task_context_id", "") or "")
        if trusted_task_context_id:
            metadata["task_context_id"] = trusted_task_context_id
        elif task_context is not None:
            metadata["task_context_id"] = task_context.task_context_id
        else:
            metadata.pop("task_context_id", None)
        self.append_turn(
            message.conversation_id,
            ConversationTurn(
                role=ConversationRole.USER,
                content=message.text,
                trace_id=trace_id,
                sender_id=message.sender_id,
                sender_name=message.sender_name,
                metadata=metadata,
                occurred_at=message.sent_at,
            ),
        )

    def append_assistant_turn(
        self,
        conversation_id: str,
        text: str,
        trace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """追加assistantturn，保留原有时间顺序和审计信息。"""
        turn_metadata = dict(metadata or {})
        task_context_id = self._task_context_id_for_trace(conversation_id, trace_id)
        if task_context_id:
            turn_metadata.setdefault("task_context_id", task_context_id)
        else:
            task_context = self.latest_task_context(conversation_id)
            if task_context is not None:
                turn_metadata.setdefault("task_context_id", task_context.task_context_id)
        self.append_turn(
            conversation_id,
            ConversationTurn(
                role=ConversationRole.ASSISTANT,
                content=text,
                trace_id=trace_id,
                metadata=turn_metadata,
            ),
        )

    def append_tool_turn(self, conversation_id: str, text: str, trace_id: str) -> None:
        """追加工具turn，保留原有时间顺序和审计信息。"""
        task_context_id = self._task_context_id_for_trace(conversation_id, trace_id)
        if not task_context_id:
            task_context = self.latest_task_context(conversation_id)
            task_context_id = task_context.task_context_id if task_context else ""
        metadata = {"task_context_id": task_context_id} if task_context_id else {}
        self.append_turn(
            conversation_id,
            ConversationTurn(role=ConversationRole.TOOL, content=text, trace_id=trace_id, metadata=metadata),
        )

    def append_turn(self, conversation_id: str, turn: ConversationTurn) -> None:
        """追加turn，保留原有时间顺序和审计信息。"""
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runtime_conversation_turns(
                    conversation_id, trace_id, role, task_context_id,
                    source_message_id, occurred_at, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    turn.trace_id,
                    turn.role.value,
                    str(turn.metadata.get("task_context_id") or ""),
                    str(turn.metadata.get("source_message_id") or ""),
                    turn.occurred_at.isoformat(),
                    _dumps(turn.to_dict()),
                ),
            )

    def recent_turns(self, conversation_id: str, limit: int = 30) -> list[ConversationTurn]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload
                FROM runtime_conversation_turns
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, int(limit)),
            ).fetchall()
            turns = [_turn_from_payload(_loads(row["payload"])) for row in rows]
            return list(reversed(turns))

    def task_context_turns(
        self,
        conversation_id: str,
        task_context_id: str,
        limit: int = 60,
    ) -> list[ConversationTurn]:
        """在应用历史限制前，先按业务 episode 过滤。"""

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload
                FROM runtime_conversation_turns
                WHERE conversation_id = ? AND task_context_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, task_context_id, int(limit)),
            ).fetchall()
            return list(
                reversed([_turn_from_payload(_loads(row["payload"])) for row in rows])
            )

    def find_conversation_turn(
        self,
        conversation_id: str,
        *,
        message_id: str | None = None,
        text: str | None = None,
        sender_id: str | None = None,
    ) -> ConversationTurn | None:
        """查找会话turn；找不到时按合同返回空值。

"""

        with self._lock:
            if message_id:
                row = self._connection.execute(
                    """
                    SELECT payload
                    FROM runtime_conversation_turns
                    WHERE conversation_id = ? AND source_message_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (conversation_id, message_id),
                ).fetchone()
                if row is not None:
                    return _turn_from_payload(_loads(row["payload"]))

            normalized_text = str(text or "").strip()
            if not normalized_text:
                return None
            rows = self._connection.execute(
                """
                SELECT payload
                FROM runtime_conversation_turns
                WHERE conversation_id = ?
                ORDER BY id DESC
                """,
                (conversation_id,),
            ).fetchall()
            for row in rows:
                turn = _turn_from_payload(_loads(row["payload"]))
                if turn.content != normalized_text:
                    continue
                if sender_id and turn.sender_id != sender_id:
                    continue
                return turn
            return None

    def get_conversation_checkpoint(self, conversation_id: str) -> ConversationCheckpoint | None:
        """读取会话检查点；返回值只反映当前存储或运行快照。"""
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_conversation_checkpoints WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            return _checkpoint_from_payload(_loads(row["payload"]))

    def get_task_context_checkpoint(self, task_context_id: str) -> ConversationCheckpoint | None:
        """读取任务上下文检查点；返回值只反映当前存储或运行快照。"""
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_task_context_checkpoints WHERE task_context_id = ?",
                (task_context_id,),
            ).fetchone()
            return _checkpoint_from_payload(_loads(row["payload"])) if row else None

    def get_task_context(self, task_context_id: str) -> ConversationTaskContext | None:
        """读取任务上下文；返回值只反映当前存储或运行快照。"""
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_task_contexts WHERE task_context_id = ?",
                (task_context_id,),
            ).fetchone()
            return _task_context_from_payload(_loads(row["payload"])) if row else None

    def current_task_context(self, conversation_id: str, customer_id: str) -> ConversationTaskContext | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload
                FROM runtime_task_contexts
                WHERE conversation_id = ? AND customer_id = ? AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (conversation_id, customer_id),
            ).fetchone()
            return _task_context_from_payload(_loads(row["payload"])) if row else None

    def latest_task_context(self, conversation_id: str) -> ConversationTaskContext | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload
                FROM runtime_task_contexts
                WHERE conversation_id = ? AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            return _task_context_from_payload(_loads(row["payload"])) if row else None

    def activate_task_context(
        self,
        *,
        conversation_id: str,
        customer_id: str,
        trace_id: str,
        activity_at: datetime,
        started_at: datetime,
        reason: str,
        force_new: bool,
        archive_previous: bool,
    ) -> tuple[ConversationTaskContext, list[StateTransition]]:
        with self._write_transaction():
            from ...models import new_id

            transitions: list[StateTransition] = []
            previous = self.current_task_context(conversation_id, customer_id)
            if previous is not None and not force_new:
                previous.updated_at = activity_at
                previous.source_trace_id = trace_id
                self._save_task_context(previous)
                return previous, transitions

            if previous is not None:
                previous.status = "closed"
                previous.closed_at = activity_at
                previous.updated_at = activity_at
                self._save_task_context(previous)
                transitions.append(
                    StateTransition(
                        "task_context",
                        previous.task_context_id,
                        "active",
                        "closed",
                        reason,
                        trace_id,
                    )
                )

            if archive_previous:
                rows = self._connection.execute(
                    """
                    SELECT payload
                    FROM runtime_task_memories
                    WHERE conversation_id = ? AND status = 'active'
                      AND (customer_id = ? OR target_customer_id = ?)
                    """,
                    (conversation_id, customer_id, customer_id),
                ).fetchall()
                previous_context_id = previous.task_context_id if previous else None
                for row in rows:
                    memory = _task_memory_from_payload(_loads(row["payload"]))
                    memory_context_id = str(memory.metadata.get("task_context_id") or "")
                    belongs_to_previous = bool(previous_context_id and memory_context_id == previous_context_id)
                    predates_new_context = memory.updated_at < started_at
                    if not belongs_to_previous and not predates_new_context:
                        continue
                    memory.status = "archived"
                    memory.updated_at = activity_at
                    self._save_task_memory(memory)
                    transitions.append(
                        StateTransition(
                            "task_memory",
                            memory.memory_id,
                            "active",
                            "archived",
                            "task_context_reset",
                            trace_id,
                        )
                    )

            context = ConversationTaskContext(
                task_context_id=new_id("task_context"),
                conversation_id=conversation_id,
                customer_id=customer_id,
                reset_reason=reason,
                previous_task_context_id=previous.task_context_id if previous else None,
                source_trace_id=trace_id,
                started_at=started_at,
                updated_at=activity_at,
            )
            self._save_task_context(context)
            transitions.append(
                StateTransition("task_context", context.task_context_id, None, "active", reason, trace_id)
            )
            for transition in transitions:
                self._append_transition(transition)
            return context, transitions

    def conversation_version(self, conversation_id: str) -> int:
        key = conversation_id or "default"
        with self._lock:
            row = self._connection.execute(
                "SELECT version FROM runtime_conversation_versions WHERE conversation_id = ?",
                (key,),
            ).fetchone()
            return int(row["version"]) if row else 0

    def advance_conversation_version(
        self,
        conversation_id: str,
        *,
        trace_id: str,
        reason: str,
    ) -> tuple[int, StateTransition]:
        key = conversation_id or "default"
        with self._lock, self._connection:
            row = self._connection.execute(
                """
                INSERT INTO runtime_conversation_versions(conversation_id, version, updated_at)
                VALUES (?, 1, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    version=runtime_conversation_versions.version + 1,
                    updated_at=excluded.updated_at
                RETURNING version
                """,
                (key, _now_iso()),
            ).fetchone()
            if row is None:
                raise RuntimeError("conversation version update returned no row")
            new = int(row["version"])
            old = new - 1
            transition = StateTransition(
                "conversation_version",
                key,
                str(old),
                str(new),
                reason or "user_message_received",
                trace_id,
            )
            self._append_transition(transition)
            return new, transition

    def supersede_pending_outputs(
        self, conversation_id: str, *, sender_id: str | None = None, trace_id: str, reason: str,
    ) -> tuple[dict[str, int], list[StateTransition]]:
        """独立调用仍先取得写事务；接收入口复用同一个事务内部实现。"""
        with self._write_transaction():
            return self._supersede_pending_outputs_in_transaction(
                conversation_id, sender_id=sender_id, trace_id=trace_id, reason=reason,
            )

    def _supersede_pending_outputs_in_transaction(
        self, conversation_id: str, *, sender_id: str | None, trace_id: str, reason: str,
    ) -> tuple[dict[str, int], list[StateTransition]]:
        """在调用方已持有的写事务中废弃pending，保留已发送、交付中与已交付事实。

        本助手只执行SQL和迁移记录，不另开或提交事务；输入接收可把原话、版本与
        废弃旧输出一起提交，任何写入或审计失败都会由外层事务完整回滚。
        """
        if not self._connection.in_transaction:
            raise RuntimeError("pending output supersession requires an active write transaction")
        key = conversation_id or "default"
        transitions: list[StateTransition] = []
        counts = {
            "invite_drafts": 0,
            "outbound_message_drafts": 0,
            "assistant_replies": 0,
        }
        game_ids = {game.game_id for game in self.games.values() if game.conversation_id == key}
        sender_is_pending_candidate = bool(
            sender_id
            and any(
                draft.game_id in game_ids
                and draft.customer_id == sender_id
                and draft.status == InviteStatus.PENDING_APPROVAL
                for draft in self.invite_drafts.values()
            )
        )
        for draft in self.invite_drafts.values():
            if sender_is_pending_candidate:
                continue
            if draft.game_id not in game_ids or draft.status != InviteStatus.PENDING_APPROVAL:
                continue
            old = draft.status.value
            draft.status = InviteStatus.SUPERSEDED
            draft.updated_at = datetime.now(DEFAULT_TZ)
            draft.metadata = {
                **dict(draft.metadata),
                "superseded_by_trace_id": trace_id,
                "superseded_reason": reason,
            }
            counts["invite_drafts"] += 1
            transitions.append(StateTransition("invite_draft", draft.draft_id, old, draft.status.value, reason, trace_id))
            self._save_invite(draft)
        for draft in self.outbound_message_drafts.values():
            if sender_is_pending_candidate:
                continue
            if draft.conversation_id != key or draft.status != OutboundDraftStatus.PENDING_APPROVAL:
                continue
            old = draft.status.value
            draft.status = OutboundDraftStatus.SUPERSEDED
            draft.updated_at = datetime.now(DEFAULT_TZ)
            draft.metadata = {
                **dict(draft.metadata),
                "superseded_by_trace_id": trace_id,
                "superseded_reason": reason,
            }
            counts["outbound_message_drafts"] += 1
            transitions.append(
                StateTransition("outbound_message_draft", draft.draft_id, old, draft.status.value, reason, trace_id)
            )
            self._save_outbound_message_draft(draft)
        rows = self._connection.execute(
            """
            SELECT id, payload
            FROM runtime_conversation_turns
            WHERE conversation_id = ? AND role = ?
            """,
            (key, ConversationRole.ASSISTANT.value),
        ).fetchall()
        for row in rows:
            payload = _loads(row["payload"])
            turn = _turn_from_payload(payload)
            if turn.metadata.get("delivery_status") != "pending_operator_send":
                continue
            old = str(turn.metadata.get("delivery_status") or "")
            turn.metadata = {
                **dict(turn.metadata),
                "delivery_status": "superseded",
                "superseded_by_trace_id": trace_id,
                "superseded_reason": reason,
            }
            counts["assistant_replies"] += 1
            transitions.append(StateTransition("assistant_reply", turn.trace_id, old, "superseded", reason, trace_id))
            self._connection.execute(
                "UPDATE runtime_conversation_turns SET payload = ? WHERE id = ?",
                (_dumps(turn.to_dict()), row["id"]),
            )
        for transition in transitions:
            self._append_transition(transition)
        return counts, transitions

    def upsert_conversation_checkpoint(
        self,
        *,
        conversation_id: str,
        summary: str,
        facts: dict[str, Any],
        open_questions: list[str],
        trace_id: str,
        task_context_id: str | None = None,
    ) -> tuple[ConversationCheckpoint, StateTransition]:
        """新增或更新会话检查点，使重复写入收敛到同一条记录。"""
        with self._lock, self._connection:
            previous = self.get_conversation_checkpoint(conversation_id)
            task_context = self.latest_task_context(conversation_id)
            resolved_task_context_id = task_context_id or (
                task_context.task_context_id if task_context else None
            )
            checkpoint = ConversationCheckpoint(
                conversation_id=conversation_id,
                summary=summary,
                facts=dict(facts),
                open_questions=list(open_questions),
                task_context_id=resolved_task_context_id,
                source_trace_id=trace_id,
            )
            transition = StateTransition(
                "conversation_checkpoint",
                conversation_id,
                "exists" if previous else None,
                "updated",
                "update_context_checkpoint",
                trace_id,
            )
            self._connection.execute(
                """
                INSERT INTO runtime_conversation_checkpoints(conversation_id, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (conversation_id, _dumps(checkpoint.to_dict()), checkpoint.updated_at.isoformat()),
            )
            if checkpoint.task_context_id:
                self._connection.execute(
                    """
                    INSERT INTO runtime_task_context_checkpoints(
                        task_context_id, conversation_id, payload, updated_at
                    )
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(task_context_id) DO UPDATE SET
                        conversation_id=excluded.conversation_id,
                        payload=excluded.payload,
                        updated_at=excluded.updated_at
                    """,
                    (
                        checkpoint.task_context_id,
                        conversation_id,
                        _dumps(checkpoint.to_dict()),
                        checkpoint.updated_at.isoformat(),
                    ),
                )
            self._append_transition(transition)
            return checkpoint, transition

    def _task_context_id_for_trace(self, conversation_id: str, trace_id: str) -> str:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT task_context_id
                FROM runtime_conversation_turns
                WHERE conversation_id = ? AND trace_id = ? AND role = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (conversation_id, trace_id, ConversationRole.USER.value),
            ).fetchone()
        return str(row["task_context_id"] or "") if row else ""

    def invalidate_message_context(self, conversation_id: str, sender_id: str, message_id: str, *, trace_id: str) -> bool:
        """在同一事务标记撤回来源并失效派生摘要，原始消息和领域状态仍保留。"""
        with self._write_transaction():
            target = self.find_conversation_turn(conversation_id, message_id=message_id)
            if target is None or target.sender_id != sender_id or target.metadata.get("context_invalidated"):
                return False
            rows = self._connection.execute("SELECT id,payload FROM runtime_conversation_turns WHERE conversation_id=? AND trace_id=?", (conversation_id,target.trace_id)).fetchall()
            for row in rows:
                payload = _loads(row["payload"])
                payload.setdefault("metadata", {}).update(context_invalidated=True, recalled_by_trace_id=trace_id)
                self._connection.execute("UPDATE runtime_conversation_turns SET payload=? WHERE id=?", (_dumps(payload),row["id"]))
            self._connection.execute("DELETE FROM runtime_conversation_checkpoints WHERE conversation_id=?",(conversation_id,))
            self._connection.execute("DELETE FROM runtime_task_context_checkpoints WHERE conversation_id=?",(conversation_id,))
            for memory in self.task_memories.values():
                if memory.conversation_id == conversation_id and memory.source_trace_id == target.trace_id:
                    memory.status = "invalidated"
                    self._save_task_memory(memory)
            for candidate in self.pending_memory_candidates.values():
                if candidate.conversation_id == conversation_id and candidate.source_trace_id == target.trace_id:
                    candidate.status = "invalidated"
                    self._save_pending_memory_candidate(candidate)
            for task in self.scheduled_tasks.values():
                if task.conversation_id == conversation_id and task.subject_id == sender_id and task.payload.get("source_message_id") == message_id and task.status in {ScheduledTaskStatus.PENDING, ScheduledTaskStatus.PROCESSING}:
                    old = task.status.value
                    task.status = ScheduledTaskStatus.CANCELLED
                    self._save_scheduled_agent_task(task)
                    self._append_transition(StateTransition("scheduled_agent_task", task.task_id, old, "cancelled", "source_message_recalled", trace_id))
            return True
