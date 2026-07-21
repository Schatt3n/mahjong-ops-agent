"""SQLite references store operations."""

from __future__ import annotations

from typing import Any
from ...models import MessageReference
from .serialization import (
    _loads,
    _message_reference_from_payload,
)

class SQLiteReferencesStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

    __slots__ = ()

    def register_message_reference(self, reference: MessageReference) -> None:
        if not reference.message_id:
            return
        with self._lock, self._connection:
            self._save_message_reference(reference)

    def link_message_reference(
        self,
        *,
        conversation_id: str,
        message_id: str,
        source_message_id: str | None = None,
        business_ref_type: str | None = None,
        business_ref_id: str | None = None,
        channel: str | None = None,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MessageReference:
        source = self._find_message_reference_source(
            conversation_id=conversation_id,
            source_message_id=source_message_id,
            business_ref_type=business_ref_type,
            business_ref_id=business_ref_id,
        )
        if source is None:
            raise ValueError("source message reference not found")
        linked = MessageReference(
            message_id=str(message_id or ""),
            conversation_id=str(conversation_id or source.conversation_id),
            business_ref_type=source.business_ref_type,
            business_ref_id=source.business_ref_id,
            text=str(text or source.text or ""),
            channel=str(channel or source.channel or ""),
            sender_id=source.sender_id,
            sender_name=source.sender_name,
            recipient_id=source.recipient_id,
            recipient_name=source.recipient_name,
            metadata={
                **dict(source.metadata),
                **dict(metadata or {}),
                "linked_from_message_id": source.message_id,
                "linked_from_conversation_id": source.conversation_id,
            },
        )
        with self._lock, self._connection:
            self._save_message_reference(linked)
        return linked

    def _find_message_reference_source(
        self,
        *,
        conversation_id: str,
        source_message_id: str | None,
        business_ref_type: str | None,
        business_ref_id: str | None,
    ) -> MessageReference | None:
        if source_message_id:
            source = self.resolve_message_reference(conversation_id=conversation_id, message_id=source_message_id)
            if source is not None:
                return source
        if not business_ref_type or not business_ref_id:
            return None
        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload FROM runtime_message_references
                WHERE conversation_id = ? AND business_ref_type = ? AND business_ref_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (conversation_id, business_ref_type, business_ref_id),
            ).fetchone()
            if row is None:
                row = self._connection.execute(
                    """
                    SELECT payload FROM runtime_message_references
                    WHERE business_ref_type = ? AND business_ref_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (business_ref_type, business_ref_id),
                ).fetchone()
            if row is None:
                return None
            return _message_reference_from_payload(_loads(row["payload"]))

    def resolve_message_reference(
        self,
        *,
        conversation_id: str,
        message_id: str,
    ) -> MessageReference | None:
        if not message_id:
            return None
        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload FROM runtime_message_references
                WHERE conversation_id = ? AND message_id = ?
                """,
                (conversation_id, message_id),
            ).fetchone()
            if row is None:
                row = self._connection.execute(
                    """
                    SELECT payload FROM runtime_message_references
                    WHERE message_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (message_id,),
                ).fetchone()
            if row is None:
                return None
            return _message_reference_from_payload(_loads(row["payload"]))
