"""Conservative session merge and stable-formation crystallization."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import timedelta

from .models import ChatSession
from .session_router import SessionRouter
from .session_state import facts_conflict


_CONFIRMATION = re.compile(r"[+＋]1|我也来|我来|我打|可以|\bok\b|\bokk\b|^行$", re.IGNORECASE)


class SessionMerger:
    """Merge only sessions with shared actors, close timing, and no fact conflict."""

    def __init__(self, router: SessionRouter, *, merge_window_minutes: float = 10) -> None:
        self.router = router
        self.merge_window = timedelta(minutes=max(0.0, merge_window_minutes))

    def merge_if_related(self, session: ChatSession) -> ChatSession:
        if session.status != "active" or not session.messages:
            return session
        candidates = [
            item
            for item in self.router.list_sessions(session.room_id)
            if item.id != session.id
            and item.status == "active"
            and bool(item.participants & session.participants)
            and abs(session.created_at - item.last_activity_at) <= self.merge_window
            and self._topics_related(item, session)
        ]
        if not candidates:
            return session
        target = min(candidates, key=lambda item: item.created_at)
        return self.router.merge_sessions(source=session, target=target)

    @staticmethod
    def _topics_related(left: ChatSession, right: ChatSession) -> bool:
        for key in ("game_type", "stakes", "time", "table_id", "smoking"):
            left_value = left.extracted_state.get(key)
            right_value = right.extracted_state.get(key)
            if facts_conflict(key, left_value, right_value):
                return False
        left_initiator = left.messages[0].sender_external_id if left.messages else ""
        right_initiator = right.messages[0].sender_external_id if right.messages else ""
        if left_initiator and left_initiator == right_initiator:
            return True
        return any(
            left.extracted_state.get(key) == right.extracted_state.get(key)
            and left.extracted_state.get(key) not in (None, "")
            for key in ("game_type", "stakes", "time", "table_id", "smoking")
        )


class SessionCrystallizer:
    """Detect when a formation thread contains enough stable public facts to project."""

    def __init__(self, *, on_crystallized: Callable[[ChatSession], None] | None = None) -> None:
        self.on_crystallized = on_crystallized

    def crystallize_if_ready(self, session: ChatSession) -> bool:
        if session.extracted_state.get("crystallized"):
            return False
        if session.topic_type != "formation" or len(session.participants) < 3:
            return False
        if not all(session.extracted_state.get(key) not in (None, "") for key in ("time", "stakes", "smoking")):
            return False
        confirming_senders = {
            message.sender_external_id
            for message in session.messages
            if _CONFIRMATION.search(message.text.strip())
        }
        if len(confirming_senders) < 2:
            return False
        session.extracted_state["crystallized"] = True
        if self.on_crystallized is not None:
            self.on_crystallized(session)
        return True


__all__ = ["SessionCrystallizer", "SessionMerger"]
