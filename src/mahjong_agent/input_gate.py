from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from .models import Message


@dataclass(slots=True)
class InputGateDecision:
    """Decision made before a message enters the controlled workflow."""

    accepted: bool
    scope: str
    source_message_id: str
    tenant_id: str = "default"
    sequence: int | None = None
    expected_sequence: int | None = None
    duplicate: bool = False
    out_of_order: bool = False
    waiting_for_sequence: bool = False
    in_progress: bool = False
    reason: str = ""
    cached_result: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "tenant_id": self.tenant_id,
            "scope": self.scope,
            "source_message_id": self.source_message_id,
            "sequence": self.sequence,
            "expected_sequence": self.expected_sequence,
            "duplicate": self.duplicate,
            "out_of_order": self.out_of_order,
            "waiting_for_sequence": self.waiting_for_sequence,
            "in_progress": self.in_progress,
            "reason": self.reason,
            "has_cached_result": self.cached_result is not None,
        }


class InputGate(Protocol):
    def begin(self, message: Message, *, trace_id: str, now: datetime) -> InputGateDecision:
        ...

    def complete(self, message: Message, result: Any, *, trace_id: str, now: datetime) -> None:
        ...

    def fail(self, message: Message, *, trace_id: str, now: datetime) -> None:
        ...


class InMemoryInputGate:
    """Process-local idempotency and ordering gate for controlled workflows.

    The gate deliberately does not understand Mahjong business semantics. It
    only protects the workflow entrance by source message id and optional
    per-conversation sequence.
    """

    def __init__(self) -> None:
        self._completed_by_source: dict[tuple[str, str], Any] = {}
        self._inflight_by_source: dict[tuple[str, str], InputGateDecision] = {}
        self._last_sequence_by_scope: dict[tuple[str, str], int] = {}
        self._source_scope_sequence: dict[tuple[str, str], tuple[str, int | None]] = {}

    def begin(self, message: Message, *, trace_id: str, now: datetime) -> InputGateDecision:
        tenant_id = _tenant_id(message)
        scope = _scope(message)
        source_message_id = _source_message_id(message)
        sequence = _sequence(message)
        source_key = (tenant_id, source_message_id)
        scope_key = (tenant_id, scope)

        cached_result = self._completed_by_source.get(source_key)
        if cached_result is not None:
            return InputGateDecision(
                accepted=False,
                tenant_id=tenant_id,
                scope=scope,
                source_message_id=source_message_id,
                sequence=sequence,
                duplicate=True,
                reason="source_message_id 已完成，直接复用首轮处理结果。",
                cached_result=cached_result,
            )

        inflight = self._inflight_by_source.get(source_key)
        if inflight is not None:
            return InputGateDecision(
                accepted=False,
                tenant_id=tenant_id,
                scope=scope,
                source_message_id=source_message_id,
                sequence=sequence,
                duplicate=True,
                in_progress=True,
                reason="source_message_id 正在处理中，拒绝重复进入 workflow。",
            )

        if sequence is not None:
            last_sequence = self._last_sequence_by_scope.get(scope_key, 0)
            expected = last_sequence + 1
            if sequence <= last_sequence:
                return InputGateDecision(
                    accepted=False,
                    tenant_id=tenant_id,
                    scope=scope,
                    source_message_id=source_message_id,
                    sequence=sequence,
                    expected_sequence=expected,
                    duplicate=True,
                    out_of_order=True,
                    reason="消息 sequence 已落后于会话已处理进度，拒绝重复或过期消息。",
                )
            if sequence > expected:
                return InputGateDecision(
                    accepted=False,
                    tenant_id=tenant_id,
                    scope=scope,
                    source_message_id=source_message_id,
                    sequence=sequence,
                    expected_sequence=expected,
                    out_of_order=True,
                    waiting_for_sequence=True,
                    reason="消息 sequence 超前，等待前序消息处理完成后再进入 workflow。",
                )

        decision = InputGateDecision(
            accepted=True,
            tenant_id=tenant_id,
            scope=scope,
            source_message_id=source_message_id,
            sequence=sequence,
            expected_sequence=sequence,
            reason="消息通过入口幂等和顺序检查。",
        )
        self._inflight_by_source[source_key] = decision
        self._source_scope_sequence[source_key] = (scope, sequence)
        return decision

    def complete(self, message: Message, result: Any, *, trace_id: str, now: datetime) -> None:
        tenant_id = _tenant_id(message)
        source_message_id = _source_message_id(message)
        source_key = (tenant_id, source_message_id)
        scope, sequence = self._source_scope_sequence.get(source_key, (_scope(message), _sequence(message)))
        self._completed_by_source[source_key] = result
        self._inflight_by_source.pop(source_key, None)
        self._source_scope_sequence.pop(source_key, None)
        if sequence is not None:
            scope_key = (tenant_id, scope)
            previous = self._last_sequence_by_scope.get(scope_key, 0)
            if sequence == previous + 1:
                self._last_sequence_by_scope[scope_key] = sequence

    def fail(self, message: Message, *, trace_id: str, now: datetime) -> None:
        tenant_id = _tenant_id(message)
        source_key = (tenant_id, _source_message_id(message))
        self._inflight_by_source.pop(source_key, None)
        self._source_scope_sequence.pop(source_key, None)


def _tenant_id(message: Message) -> str:
    value = message.metadata.get("tenant_id") or message.metadata.get("store_id") or "default"
    return str(value).strip() or "default"


def _scope(message: Message) -> str:
    value = message.metadata.get("conversation_id") or message.channel_id
    return str(value).strip() or "default"


def _source_message_id(message: Message) -> str:
    value = (
        message.metadata.get("source_message_id")
        or message.metadata.get("message_id")
        or message.metadata.get("platform_message_id")
        or message.id
    )
    return str(value).strip() or message.id


def _sequence(message: Message) -> int | None:
    value = message.metadata.get("sequence")
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
