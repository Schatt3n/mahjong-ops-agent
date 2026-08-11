"""实现内存存储层的会话数据访问能力，主要用于本地运行和测试。

"""

from __future__ import annotations

from typing import Any
from datetime import datetime
from ...models import (
    ConversationCheckpoint,
    ConversationRole,
    ConversationTaskContext,
    ConversationTurn,
    InviteStatus,
    OutboundDraftStatus,
    StateTransition,
    new_id,
    now,
)

class InMemoryConversationStoreMixin:
    """从兼容性存储中提取的后端特定操作。"""

    __slots__ = ()

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
        with self._lock:
            self.turns.setdefault(conversation_id, []).append(turn)

    def recent_turns(self, conversation_id: str, limit: int = 30) -> list[ConversationTurn]:
        with self._lock:
            return list(self.turns.get(conversation_id, []))[-int(limit):]

    def task_context_turns(
        self,
        conversation_id: str,
        task_context_id: str,
        limit: int = 60,
    ) -> list[ConversationTurn]:
        """返回属于同一业务事件的最新轮次。

                过滤先于限制执行，因此无关的群聊流量无法将未来约会的证据从其自身任务窗口中逐出。
        """

        with self._lock:
            matches = [
                turn
                for turn in self.turns.get(conversation_id, [])
                if str(turn.metadata.get("task_context_id") or "") == task_context_id
            ]
            return matches[-int(limit):]

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
            turns = list(self.turns.get(conversation_id, []))
            if message_id:
                for turn in reversed(turns):
                    if str(turn.metadata.get("source_message_id") or "") == message_id:
                        return turn
            normalized_text = str(text or "").strip()
            if normalized_text:
                for turn in reversed(turns):
                    if turn.content != normalized_text:
                        continue
                    if sender_id and turn.sender_id != sender_id:
                        continue
                    return turn
            return None

    def get_conversation_checkpoint(self, conversation_id: str) -> ConversationCheckpoint | None:
        """读取会话检查点；返回值只反映当前存储或运行快照。"""
        with self._lock:
            return self.conversation_checkpoints.get(conversation_id)

    def get_task_context_checkpoint(self, task_context_id: str) -> ConversationCheckpoint | None:
        """读取任务上下文检查点；返回值只反映当前存储或运行快照。"""
        with self._lock:
            return self.task_context_checkpoints.get(task_context_id)

    def get_task_context(self, task_context_id: str) -> ConversationTaskContext | None:
        """读取任务上下文；返回值只反映当前存储或运行快照。"""
        with self._lock:
            return self.task_contexts.get(task_context_id)

    def current_task_context(self, conversation_id: str, customer_id: str) -> ConversationTaskContext | None:
        with self._lock:
            matches = [
                item
                for item in self.task_contexts.values()
                if item.status == "active"
                and item.conversation_id == conversation_id
                and item.customer_id == customer_id
            ]
            return max(matches, key=lambda item: item.updated_at) if matches else None

    def latest_task_context(self, conversation_id: str) -> ConversationTaskContext | None:
        with self._lock:
            matches = [
                item
                for item in self.task_contexts.values()
                if item.status == "active" and item.conversation_id == conversation_id
            ]
            return max(matches, key=lambda item: item.updated_at) if matches else None

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
        """创建/复用一个业务事件，并在重置时弃用其临时记忆。"""

        with self._lock:
            transitions: list[StateTransition] = []
            previous = self.current_task_context(conversation_id, customer_id)
            if previous is not None and not force_new:
                previous.updated_at = activity_at
                previous.source_trace_id = trace_id
                return previous, transitions

            if previous is not None:
                previous.status = "closed"
                previous.closed_at = activity_at
                previous.updated_at = activity_at
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
                previous_context_id = previous.task_context_id if previous else None
                for memory in self.task_memories.values():
                    if memory.status != "active" or memory.conversation_id != conversation_id:
                        continue
                    if memory.customer_id != customer_id and memory.target_customer_id != customer_id:
                        continue
                    memory_context_id = str(memory.metadata.get("task_context_id") or "")
                    belongs_to_previous = bool(previous_context_id and memory_context_id == previous_context_id)
                    predates_new_context = memory.updated_at < started_at
                    if not belongs_to_previous and not predates_new_context:
                        continue
                    memory.status = "archived"
                    memory.updated_at = activity_at
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
            self.task_contexts[context.task_context_id] = context
            transitions.append(
                StateTransition(
                    "task_context",
                    context.task_context_id,
                    None,
                    "active",
                    reason,
                    trace_id,
                )
            )
            self.transitions.extend(transitions)
            return context, transitions

    def conversation_version(self, conversation_id: str) -> int:
        with self._lock:
            return int(self.conversation_versions.get(conversation_id or "default", 0))

    def advance_conversation_version(
        self,
        conversation_id: str,
        *,
        trace_id: str,
        reason: str,
    ) -> tuple[int, StateTransition]:
        key = conversation_id or "default"
        with self._lock:
            old = int(self.conversation_versions.get(key, 0))
            new = old + 1
            self.conversation_versions[key] = new
            transition = StateTransition(
                entity_type="conversation_version",
                entity_id=key,
                from_status=str(old),
                to_status=str(new),
                reason=reason or "user_message_received",
                trace_id=trace_id,
            )
            self.transitions.append(transition)
            return new, transition

    def supersede_pending_outputs(
        self,
        conversation_id: str,
        *,
        sender_id: str | None = None,
        trace_id: str,
        reason: str,
    ) -> tuple[dict[str, int], list[StateTransition]]:
        key = conversation_id or "default"
        with self._lock:
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
                draft.updated_at = now()
                draft.metadata = {
                    **dict(draft.metadata),
                    "superseded_by_trace_id": trace_id,
                    "superseded_reason": reason,
                }
                counts["invite_drafts"] += 1
                transitions.append(
                    StateTransition("invite_draft", draft.draft_id, old, draft.status.value, reason, trace_id)
                )
            for draft in self.outbound_message_drafts.values():
                if sender_is_pending_candidate:
                    continue
                if draft.conversation_id != key or draft.status != OutboundDraftStatus.PENDING_APPROVAL:
                    continue
                old = draft.status.value
                draft.status = OutboundDraftStatus.SUPERSEDED
                draft.updated_at = now()
                draft.metadata = {
                    **dict(draft.metadata),
                    "superseded_by_trace_id": trace_id,
                    "superseded_reason": reason,
                }
                counts["outbound_message_drafts"] += 1
                transitions.append(
                    StateTransition("outbound_message_draft", draft.draft_id, old, draft.status.value, reason, trace_id)
                )
            for turn in self.turns.get(key, []):
                if turn.role != ConversationRole.ASSISTANT:
                    continue
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
            self.transitions.extend(transitions)
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
        with self._lock:
            previous = self.conversation_checkpoints.get(conversation_id)
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
            self.conversation_checkpoints[conversation_id] = checkpoint
            if checkpoint.task_context_id:
                self.task_context_checkpoints[checkpoint.task_context_id] = checkpoint
            transition = StateTransition(
                entity_type="conversation_checkpoint",
                entity_id=conversation_id,
                from_status="exists" if previous else None,
                to_status="updated",
                reason="update_context_checkpoint",
                trace_id=trace_id,
            )
            self.transitions.append(transition)
            return checkpoint, transition

    def _task_context_id_for_trace(self, conversation_id: str, trace_id: str) -> str:
        for turn in reversed(self.turns.get(conversation_id, [])):
            if turn.trace_id != trace_id or turn.role != ConversationRole.USER:
                continue
            return str(turn.metadata.get("task_context_id") or "")
        return ""
