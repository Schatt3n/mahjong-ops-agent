"""In-memory routing for bounded public-room interaction sessions."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from datetime import datetime, timedelta

from ..models import new_id, now
from .models import ChatSession, GroupMessage, SessionClassification
from .session_state import merge_session_facts


_FEATURES = re.compile(
    r"红中|川麻换三|川麻|杭麻|财敲|cq|无烟|有烟|人齐开|173|272|371|"
    r"\d{1,2}\s*[.:：点]\s*\d{0,2}|0\.5|\d+块|\d{3}"
)
_SHORT_CONTINUATION = re.compile(r"^\s*(?:[+＋]1|我也来|我来|我打|可以|行|好|ok|okk)\s*[!！。.]?\s*$", re.IGNORECASE)


class SessionRouter:
    """Attach one accumulated unit to one session without exposing other sessions."""

    def __init__(self, *, clock: Callable[[], datetime] = now) -> None:
        self.clock = clock
        self._sessions: dict[str, ChatSession] = {}
        self._message_sessions: dict[tuple[str, str], str] = {}
        self._lock = threading.RLock()

    def route(self, message: GroupMessage) -> ChatSession:
        with self._lock:
            self.expire_stale(at=message.sent_at)
            session = (
                self._by_quote(message)
                or self._recent_sender_session(message)
                or self._short_continuation_session(message)
                or self._feature_session(message)
            )
            if session is None:
                session = ChatSession(
                    id=new_id("group_session"),
                    room_id=message.room_id,
                    created_at=message.sent_at,
                    last_activity_at=message.sent_at,
                )
                self._sessions[session.id] = session
            session.participants.add(message.sender_external_id)
            session.last_activity_at = message.sent_at
            return session

    def record(
        self,
        session: ChatSession,
        message: GroupMessage,
        classification: SessionClassification,
    ) -> None:
        with self._lock:
            session.messages.append(message)
            session.participants.add(message.sender_external_id)
            session.last_activity_at = message.sent_at
            if classification.intent == "claim":
                session.topic_type = "claim"
            elif classification.intent == "new_demand":
                session.topic_type = "formation"
            elif classification.intent == "query":
                session.topic_type = "query"
            elif classification.intent == "thread_update" and session.topic_type in {"unknown", "query"}:
                session.topic_type = "formation"
            elif classification.intent == "chitchat":
                session.topic_type = "unknown"
            session.extracted_state.update(
                {
                    key: value
                    for key, value in classification.extracted_features.items()
                    if value not in (None, "", [])
                }
            )
            if classification.matched_board_no is not None:
                session.related_board_item_id = str(classification.matched_board_no)
            session.topic = classification.reasoning
            if classification.intent == "chitchat":
                session.status = "resolved"
            source_ids = message.metadata.get("accumulated_source_message_ids") or [message.message_id]
            for source_id in source_ids:
                self._message_sessions[(message.room_id, str(source_id))] = session.id

    def list_sessions(self, room_id: str | None = None) -> list[ChatSession]:
        with self._lock:
            sessions = list(self._sessions.values())
            if room_id is not None:
                sessions = [item for item in sessions if item.room_id == room_id]
            return sorted(sessions, key=lambda item: item.created_at)

    def expire_stale(self, *, at: datetime | None = None) -> None:
        stamp = at or self.clock()
        limits = {
            "claim": timedelta(minutes=5),
            "formation": timedelta(minutes=30),
            "query": timedelta(minutes=2),
            "board_update": timedelta(minutes=3),
            "unknown": timedelta(minutes=3),
        }
        for session in self._sessions.values():
            if session.status == "active" and stamp - session.last_activity_at > limits[session.topic_type]:
                session.status = "expired"

    def merge_sessions(self, *, source: ChatSession, target: ChatSession) -> ChatSession:
        """Merge a newly discovered continuation into its older canonical session."""

        with self._lock:
            target.messages = sorted([*target.messages, *source.messages], key=lambda item: item.sent_at)
            target.participants.update(source.participants)
            target.extracted_state = merge_session_facts(target.extracted_state, source.extracted_state)
            target.last_activity_at = max(target.last_activity_at, source.last_activity_at)
            if not target.topic and source.topic:
                target.topic = source.topic
            if target.topic_type == "unknown" and source.topic_type != "unknown":
                target.topic_type = source.topic_type
            if target.related_board_item_id is None:
                target.related_board_item_id = source.related_board_item_id
            source.status = "merged"
            source.merged_into = target.id
            for message in source.messages:
                source_ids = message.metadata.get("accumulated_source_message_ids") or [message.message_id]
                for source_id in source_ids:
                    self._message_sessions[(message.room_id, str(source_id))] = target.id
            return target

    def _by_quote(self, message: GroupMessage) -> ChatSession | None:
        if not message.quoted_message_id:
            return None
        session_id = self._message_sessions.get((message.room_id, message.quoted_message_id))
        session = self._sessions.get(session_id or "")
        return session if session is not None and session.status == "active" else None

    def _recent_sender_session(self, message: GroupMessage) -> ChatSession | None:
        candidates = [
            session
            for session in self._sessions.values()
            if session.room_id == message.room_id
            and session.status == "active"
            and message.sender_external_id in session.participants
            and message.sent_at - session.last_activity_at <= timedelta(minutes=2)
        ]
        return max(candidates, key=lambda item: item.last_activity_at) if candidates else None

    def _short_continuation_session(self, message: GroupMessage) -> ChatSession | None:
        if _SHORT_CONTINUATION.fullmatch(message.text) is None:
            return None
        candidates = [
            session
            for session in self._sessions.values()
            if session.room_id == message.room_id
            and session.status == "active"
            and message.sent_at - session.last_activity_at <= timedelta(minutes=2)
            and session.topic_type in {"claim", "formation", "query", "unknown"}
        ]
        return candidates[0] if len(candidates) == 1 else None

    def _feature_session(self, message: GroupMessage) -> ChatSession | None:
        message_features = set(_FEATURES.findall(message.text))
        if not message_features:
            return None
        candidates: list[tuple[int, datetime, ChatSession]] = []
        for session in self._sessions.values():
            if session.room_id != message.room_id or session.status != "active":
                continue
            values = " ".join(str(value) for value in session.extracted_state.values())
            overlap = len(message_features & set(_FEATURES.findall(values)))
            if overlap:
                candidates.append((overlap, session.last_activity_at, session))
        return max(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else None


__all__ = ["SessionRouter"]
