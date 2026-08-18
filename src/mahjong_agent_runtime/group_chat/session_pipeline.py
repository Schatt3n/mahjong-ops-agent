"""候选的多人群会话 Pipeline；当前仅用于模拟、评测和测试。"""

from __future__ import annotations

from typing import Any

from .accumulator import MessageAccumulator
from .models import GroupMessage, GroupSessionOutcome
from .owner_parser import OwnerMessageParser
from .quick_filter import QuickFilter
from .session_classifier import GroupSessionClassifier, SessionClassificationContractError
from .session_merger import SessionCrystallizer, SessionMerger
from .session_router import SessionRouter


class GroupSessionPipeline:
    """探索多人消息聚合、Session 分类/合并与结晶的候选流程。

    重要边界：正式 Wechaty 路径当前走 ``GroupMessageHandler``，应用组合根没有构造
    本类。这里的 Accumulator/SessionRouter 状态都只在内存中，不能视为生产持久事实。
    """

    def __init__(
        self,
        *,
        store: Any,
        owner_parser: OwnerMessageParser,
        quick_filter: QuickFilter,
        accumulator: MessageAccumulator,
        session_router: SessionRouter,
        classifier: GroupSessionClassifier,
        session_merger: SessionMerger | None = None,
        crystallizer: SessionCrystallizer | None = None,
    ) -> None:
        self.store = store
        self.owner_parser = owner_parser
        self.quick_filter = quick_filter
        self.accumulator = accumulator
        self.session_router = session_router
        self.classifier = classifier
        self.session_merger = session_merger or SessionMerger(session_router)
        self.crystallizer = crystallizer or SessionCrystallizer()

    def accept(self, message: GroupMessage, *, trace_id: str) -> GroupSessionOutcome:
        """接收一条原始房间消息，不强制立即调用模型。"""

        if str(message.metadata.get("content_type") or "").lower() == "revoke_event":
            target_message_id = str(message.metadata.get("target_message_id") or "")
            if not target_message_id:
                return GroupSessionOutcome(action="revoke_ignored", detail={"reason": "missing_target"})
            buffered = self.accumulator.invalidate(
                room_id=message.room_id,
                source_message_id=target_message_id,
            )
            recorded = self.session_router.invalidate_message(
                room_id=message.room_id,
                source_message_id=target_message_id,
            )
            return GroupSessionOutcome(
                action="message_revoked",
                detail={"target_message_id": target_message_id, "buffered_removed": buffered, "session_removed": recorded},
            )

        if self.owner_parser.is_owner(message):
            current = self.store.get_group_board_state(message.room_id)
            parsed = self.owner_parser.process(message, current)
            if parsed.board_state is not None:
                self.store.upsert_group_board_state(parsed.board_state)
            return GroupSessionOutcome(
                action=parsed.action,
                detail={"changed_item_ids": list(parsed.changed_item_ids)},
            )
        if self.quick_filter.should_filter(message):
            return GroupSessionOutcome(action="filtered")
        self.accumulator.add(message, trace_id=trace_id)
        return GroupSessionOutcome(action="buffered")

    def flush_due(self, *, at) -> list[GroupSessionOutcome]:
        """对静默窗口已过的每个发送者批次进行分类。"""

        outcomes: list[GroupSessionOutcome] = []
        for accumulated in self.accumulator.flush_due(at=at):
            message = accumulated.message
            session = self.session_router.route(message)
            board = self.store.get_group_board_state(message.room_id)
            try:
                classified = self.classifier.classify(
                    board_state=board,
                    session=session,
                    new_message=message,
                    trace_id=accumulated.trace_id,
                )
            except SessionClassificationContractError as exc:
                self.session_router.record_unclassified(session, message)
                outcomes.append(
                    GroupSessionOutcome(
                        action="classification_failed",
                        session_id=session.id,
                        detail={
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "attempts": exc.attempts,
                            "fragment_count": message.metadata.get("fragment_count", 1),
                        },
                    )
                )
                continue
            self.session_router.record(session, message, classified)
            session = self.session_merger.merge_if_related(session)
            crystallized = self.crystallizer.crystallize_if_ready(session)
            outcomes.append(
                GroupSessionOutcome(
                    action="board_update" if crystallized else classified.channel_action,
                    session_id=session.id,
                    classification=classified,
                    detail={
                        "matched_board_no": classified.matched_board_no,
                        "fragment_count": message.metadata.get("fragment_count", 1),
                        "crystallized": crystallized,
                    },
                )
            )
        return outcomes


__all__ = ["GroupSessionPipeline"]
