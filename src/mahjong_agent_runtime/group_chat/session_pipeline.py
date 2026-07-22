"""Owner board parsing and user-session classification pipeline."""

from __future__ import annotations

from typing import Any

from .accumulator import MessageAccumulator
from .models import GroupMessage, GroupSessionOutcome
from .owner_parser import OwnerMessageParser
from .quick_filter import QuickFilter
from .session_classifier import GroupSessionClassifier
from .session_merger import SessionCrystallizer, SessionMerger
from .session_router import SessionRouter


class GroupSessionPipeline:
    """Run deterministic admission before one isolated semantic model task."""

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
        """Accept one raw room message without forcing an immediate model call."""

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
        """Classify every sender batch whose quiet window has elapsed."""

        outcomes: list[GroupSessionOutcome] = []
        for accumulated in self.accumulator.flush_due(at=at):
            message = accumulated.message
            session = self.session_router.route(message)
            board = self.store.get_group_board_state(message.room_id)
            classified = self.classifier.classify(
                board_state=board,
                session=session,
                new_message=message,
                trace_id=accumulated.trace_id,
            )
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
