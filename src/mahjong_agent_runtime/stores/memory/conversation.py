"""实现内存存储层的会话数据访问能力，主要用于本地运行和测试。

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
    InviteStatus,
    OutboundDraftStatus,
    ScheduledTaskStatus,
    StateTransition,
    UserMessage,
    new_id,
    now,
)

class InMemoryConversationStoreMixin:
    """从兼容性存储中提取的后端特定操作。"""

    __slots__ = ()

    def accept_user_input(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        message_key: str,
        preparation: dict[str, Any] | None = None,
    ) -> tuple[int, StateTransition | None, bool, dict[str, int], list[StateTransition]]:
        """在同一锁内接收原话、推进版本并废弃旧pending；准备失败不发布任何半态。

        幂等键属于会话和发送者；重复只返回首次接收版本。接收元数据和任务归属由
        后端覆盖，不能通过用户附带的版本、接收标记或可信任务字段伪造执行事实。
        """
        if not isinstance(message_key, str) or not message_key.strip():
            raise ValueError("message_key must be a non-empty string")
        version_key = message.conversation_id or "default"
        with self._lock:
            for turn in reversed(self.turns.get(message.conversation_id, [])):
                if (
                    turn.role == ConversationRole.USER
                    and turn.sender_id == message.sender_id
                    and turn.metadata.get("input_receipt_key") == message_key
                ):
                    return int(turn.metadata["conversation_version"]), None, False, {
                        "invite_drafts": 0, "outbound_message_drafts": 0, "assistant_replies": 0,
                    }, []
            previous_version = int(self.conversation_versions.get(version_key, 0))
            version = previous_version + 1
            task_context = self.current_task_context(message.conversation_id, message.sender_id)
            metadata = deepcopy(message.metadata or {})
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
            transition = StateTransition(
                "conversation_version", version_key, str(previous_version), str(version),
                "user_message_received", trace_id,
            )
            counts, superseded, invites, outbound, turns = self._prepare_pending_output_supersession(
                message.conversation_id, sender_id=message.sender_id, trace_id=trace_id,
                reason="new_user_message_superseded_previous_pending_output",
            )
            # 版本使用既有 default 键，但原话始终保留输入会话键，与 SQLite 接收合同一致。
            turn_updates = {version_key: turns}
            receipt_turns = turn_updates.setdefault(message.conversation_id, list(self.turns.get(message.conversation_id, [])))
            receipt_turns.append(turn)
            # 所有失败点在准备阶段结束；以下只在同一锁内发布后端状态，不运行回调。
            self.conversation_versions[version_key] = version
            self.turns.update(turn_updates)
            self.invite_drafts.update(invites)
            self.outbound_message_drafts.update(outbound)
            self.transitions.extend([transition, *superseded])
            return version, transition, True, counts, superseded

    def claim_user_input_hooks(self, conversation_id: str, *, sender_id: str, message_key: str) -> dict[str, Any] | None:
        """只领取新 receipt 的首次通知，缺少标记的历史证据不猜测或自动回填。"""
        with self._lock:
            for turn in self.turns.get(conversation_id, []):
                if turn.role != ConversationRole.USER or turn.sender_id != sender_id or turn.metadata.get("input_receipt_key") != message_key:
                    continue
                if turn.metadata.get("input_receipt_hooks_dispatched") is not False:
                    return None
                turn.metadata = {**turn.metadata, "input_receipt_hooks_dispatched": True}
                return {"trace_id": turn.trace_id, "preparation": deepcopy(turn.metadata["input_receipt_preparation"])}
        return None

    def claim_assistant_reply_delivery(self, conversation_id: str, *, trace_id: str) -> bool:
        """在同一锁内核对回复版本并声明交付开始，未来输入不追溯已经开始的交付。"""
        with self._lock:
            for turn in self.turns.get(conversation_id, []):
                if turn.role != ConversationRole.ASSISTANT or turn.trace_id != trace_id:
                    continue
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
                return True
        return False

    def mark_assistant_reply_delivered(self, conversation_id: str, *, trace_id: str) -> bool:
        """按会话和精确 trace 确认已展示回复，已过期轮次不复活，重复确认保持幂等。"""
        delivered = False
        with self._lock:
            for turn in self.turns.get(conversation_id, []):
                if turn.role != ConversationRole.ASSISTANT or turn.trace_id != trace_id:
                    continue
                if turn.metadata.get("delivery_status") in {"pending_operator_send", "dispatching"}:
                    turn.metadata = {**turn.metadata, "delivery_status": "delivered"}
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
        self, conversation_id: str, *, sender_id: str | None = None, trace_id: str, reason: str,
    ) -> tuple[dict[str, int], list[StateTransition]]:
        """在同一锁中准备并发布旧pending的替换，不追溯已开始或完成的投递。"""
        key = conversation_id or "default"
        with self._lock:
            counts, transitions, invites, outbound, turns = self._prepare_pending_output_supersession(
                conversation_id, sender_id=sender_id, trace_id=trace_id, reason=reason,
            )
            self.invite_drafts.update(invites)
            self.outbound_message_drafts.update(outbound)
            self.turns[key] = turns
            self.transitions.extend(transitions)
            return counts, transitions

    def _prepare_pending_output_supersession(
        self, conversation_id: str, *, sender_id: str | None, trace_id: str, reason: str,
    ) -> tuple[dict[str, int], list[StateTransition], dict[str, Any], dict[str, Any], list[ConversationTurn]]:
        """调用方持有Store锁时只准备副本，失败不污染原始草稿、对话或审计。

        内存Store没有数据库回滚，因此先完成所有可能失败的复制与状态准备，再与
        原话和版本一起发布；保持外部持有的旧对象及已交付记录不被原地改写。
        """
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
        invite_updates: dict[str, Any] = {}
        for original in self.invite_drafts.values():
            if sender_is_pending_candidate:
                continue
            if original.game_id not in game_ids or original.status != InviteStatus.PENDING_APPROVAL:
                continue
            draft = deepcopy(original)
            old = draft.status.value
            draft.status = InviteStatus.SUPERSEDED
            draft.updated_at = now()
            draft.metadata = {
                **dict(draft.metadata),
                "superseded_by_trace_id": trace_id,
                "superseded_reason": reason,
            }
            invite_updates[draft.draft_id] = draft
            counts["invite_drafts"] += 1
            transitions.append(
                StateTransition("invite_draft", draft.draft_id, old, draft.status.value, reason, trace_id)
            )
        outbound_updates: dict[str, Any] = {}
        for original in self.outbound_message_drafts.values():
            if sender_is_pending_candidate:
                continue
            if original.conversation_id != key or original.status != OutboundDraftStatus.PENDING_APPROVAL:
                continue
            draft = deepcopy(original)
            old = draft.status.value
            draft.status = OutboundDraftStatus.SUPERSEDED
            draft.updated_at = now()
            draft.metadata = {
                **dict(draft.metadata),
                "superseded_by_trace_id": trace_id,
                "superseded_reason": reason,
            }
            outbound_updates[draft.draft_id] = draft
            counts["outbound_message_drafts"] += 1
            transitions.append(
                StateTransition("outbound_message_draft", draft.draft_id, old, draft.status.value, reason, trace_id)
            )
        updated_turns = list(self.turns.get(key, []))
        for index, turn in enumerate(updated_turns):
            if turn.role != ConversationRole.ASSISTANT:
                continue
            if turn.metadata.get("delivery_status") != "pending_operator_send":
                continue
            turn = deepcopy(turn)
            updated_turns[index] = turn
            old = str(turn.metadata.get("delivery_status") or "")
            turn.metadata = {
                **dict(turn.metadata),
                "delivery_status": "superseded",
                "superseded_by_trace_id": trace_id,
                "superseded_reason": reason,
            }
            counts["assistant_replies"] += 1
            transitions.append(StateTransition("assistant_reply", turn.trace_id, old, "superseded", reason, trace_id))
        return counts, transitions, invite_updates, outbound_updates, updated_turns

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

    def invalidate_message_context(self, conversation_id: str, sender_id: str, message_id: str, *, trace_id: str) -> bool:
        """保留撤回原文审计，只使该来源的派生上下文失效，不回滚已执行业务。"""
        with self._lock:
            target = self.find_conversation_turn(conversation_id, message_id=message_id)
            if target is None or target.sender_id != sender_id or target.metadata.get("context_invalidated"):
                return False
            for turn in self.turns.get(conversation_id, []):
                if turn.trace_id == target.trace_id:
                    turn.metadata.update(context_invalidated=True, recalled_by_trace_id=trace_id)
            self.conversation_checkpoints.pop(conversation_id, None)
            for key, checkpoint in list(self.task_context_checkpoints.items()):
                if checkpoint.conversation_id == conversation_id:
                    self.task_context_checkpoints.pop(key, None)
            for memory in self.task_memories.values():
                if memory.conversation_id == conversation_id and memory.source_trace_id == target.trace_id:
                    memory.status = "invalidated"
            for candidate in self.pending_memory_candidates.values():
                if candidate.conversation_id == conversation_id and candidate.source_trace_id == target.trace_id:
                    candidate.status = "invalidated"
            for task in self.scheduled_tasks.values():
                if task.conversation_id == conversation_id and task.subject_id == sender_id and task.payload.get("source_message_id") == message_id and task.status in {ScheduledTaskStatus.PENDING, ScheduledTaskStatus.PROCESSING}:
                    old = task.status.value
                    task.status = ScheduledTaskStatus.CANCELLED
                    self.transitions.append(StateTransition("scheduled_agent_task", task.task_id, old, "cancelled", "source_message_recalled", trace_id))
            return True
