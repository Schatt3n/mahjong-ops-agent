"""InMemory references store operations."""

from __future__ import annotations

from typing import Any
from ...models import MessageReference
from ...domains import message_reference_key

class InMemoryReferencesStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

    __slots__ = ()

    def register_message_reference(self, reference: MessageReference) -> None:
        if not reference.message_id:
            return
        with self._lock:
            self.message_references[message_reference_key(reference.conversation_id, reference.message_id)] = reference

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
        self.register_message_reference(linked)
        return linked

    def _find_message_reference_source(
        self,
        *,
        conversation_id: str,
        source_message_id: str | None,
        business_ref_type: str | None,
        business_ref_id: str | None,
    ) -> MessageReference | None:
        with self._lock:
            if source_message_id:
                source = self.resolve_message_reference(conversation_id=conversation_id, message_id=source_message_id)
                if source is not None:
                    return source
            if business_ref_type and business_ref_id:
                same_conversation: MessageReference | None = None
                latest: MessageReference | None = None
                for reference in self.message_references.values():
                    if (
                        reference.business_ref_type != business_ref_type
                        or reference.business_ref_id != business_ref_id
                    ):
                        continue
                    latest = reference
                    if reference.conversation_id == conversation_id:
                        same_conversation = reference
                return same_conversation or latest
            return None

    def resolve_message_reference(
        self,
        *,
        conversation_id: str,
        message_id: str,
    ) -> MessageReference | None:
        if not message_id:
            return None
        with self._lock:
            direct = self.message_references.get(message_reference_key(conversation_id, message_id))
            if direct is not None:
                return direct
            for reference in self.message_references.values():
                if reference.message_id == message_id:
                    return reference
            return None
