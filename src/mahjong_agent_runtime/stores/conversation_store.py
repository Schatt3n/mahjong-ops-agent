"""定义会话存储相关的存储接口，隔离领域层与具体持久化实现。

"""

from __future__ import annotations

from typing import Any, Protocol

from ..models import (
    ConversationCheckpoint,
    ConversationTaskContext,
    ConversationTurn,
    MessageReference,
    StateTransition,
    UserMessage,
)


class ConversationStore(Protocol):
    """定义会话存储需要遵守的抽象接口契约。

"""

    def invalidate_message_context(self, conversation_id: str, sender_id: str, message_id: str, *, trace_id: str) -> bool:
        """使撤回来源及其派生上下文失效，保留原始审计记录。"""
        ...

    def append_user_turn(self, message: UserMessage, trace_id: str) -> None:
        """追加用户turn，保留原有时间顺序和审计信息。"""
        ...

    def append_assistant_turn(
        self,
        conversation_id: str,
        text: str,
        trace_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """追加assistantturn，保留原有时间顺序和审计信息。"""
        ...

    def append_tool_turn(self, conversation_id: str, text: str, trace_id: str) -> None:
        """追加工具turn，保留原有时间顺序和审计信息。"""
        ...

    def append_turn(self, conversation_id: str, turn: ConversationTurn) -> None:
        """追加turn，保留原有时间顺序和审计信息。"""
        ...

    def recent_turns(self, conversation_id: str, limit: int = 30) -> list[ConversationTurn]:
        ...

    def task_context_turns(
        self,
        conversation_id: str,
        task_context_id: str,
        limit: int = 60,
    ) -> list[ConversationTurn]:
        ...

    def find_conversation_turn(
        self,
        conversation_id: str,
        *,
        message_id: str | None = None,
        text: str | None = None,
        sender_id: str | None = None,
    ) -> ConversationTurn | None:
        """查找会话turn；找不到时按合同返回空值。"""
        ...

    def get_conversation_checkpoint(self, conversation_id: str) -> ConversationCheckpoint | None:
        """读取会话检查点；返回值只反映当前存储或运行快照。"""
        ...

    def get_task_context_checkpoint(self, task_context_id: str) -> ConversationCheckpoint | None:
        """读取任务上下文检查点；返回值只反映当前存储或运行快照。"""
        ...

    def upsert_conversation_checkpoint(
        self,
        conversation_id: str,
        summary: str,
        facts: dict[str, Any],
        open_questions: list[str],
        trace_id: str,
        *,
        task_context_id: str | None = None,
    ) -> tuple[ConversationCheckpoint, StateTransition]:
        """新增或更新会话检查点，使重复写入收敛到同一条记录。"""
        ...

    def current_task_context(
        self,
        conversation_id: str,
        customer_id: str,
    ) -> ConversationTaskContext | None:
        ...

    def get_task_context(self, task_context_id: str) -> ConversationTaskContext | None:
        """读取任务上下文；返回值只反映当前存储或运行快照。"""
        ...

    def latest_task_context(self, conversation_id: str) -> ConversationTaskContext | None:
        ...

    def activate_task_context(
        self,
        *,
        conversation_id: str,
        customer_id: str,
        trace_id: str,
        reset_reason: str,
        force_new: bool = False,
    ) -> tuple[ConversationTaskContext, list[StateTransition]]:
        ...

    def conversation_version(self, conversation_id: str) -> int:
        ...

    def advance_conversation_version(
        self,
        conversation_id: str,
        *,
        trace_id: str,
        reason: str,
    ) -> tuple[int, StateTransition]:
        ...

    def supersede_pending_outputs(
        self,
        *,
        conversation_id: str,
        trace_id: str,
        reason: str,
        before_version: int | None = None,
    ) -> list[StateTransition]:
        ...

    def register_message_reference(self, reference: MessageReference) -> None:
        """登记消息引用，并建立稳定的后续查找关系。"""
        ...

    def link_message_reference(
        self,
        *,
        message_id: str,
        conversation_id: str,
        business_ref_type: str,
        business_ref_id: str,
        text: str = "",
        channel: str | None = None,
        sender_id: str | None = None,
        sender_name: str | None = None,
        recipient_id: str | None = None,
        recipient_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MessageReference:
        ...

    def resolve_message_reference(
        self,
        *,
        conversation_id: str,
        message_id: str,
    ) -> MessageReference | None:
        """解析并确定消息引用，返回消除歧义后的权威结果。"""
        ...
