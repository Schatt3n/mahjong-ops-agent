from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum, StrEnum
from typing import Any, Protocol

from .models import DEFAULT_TZ


class TraceStep(StrEnum):
    USER_INPUT = "user_input"
    CONTEXT_BUILT = "context_built"
    LLM_PROMPT = "llm_prompt"
    LLM_RESPONSE = "llm_response"
    ACTION_PROPOSED = "action_proposed"
    ACTION_VALIDATED = "action_validated"
    TOOL_CALLED = "tool_called"
    STATE_TRANSITION = "state_transition"
    REPLY_DRAFTED = "reply_drafted"
    REPLY_GUARDED = "reply_guarded"
    FINAL_OUTPUT = "final_output"
    MEMORY_WRITTEN = "memory_written"


@dataclass(slots=True)
class TraceEvent:
    trace_id: str
    step: TraceStep | str
    content: dict[str, Any]
    level: str = "INFO"
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        self.step = _coerce_step(self.step)
        self.level = str(self.level or "INFO").upper()
        self.occurred_at = self.occurred_at or datetime.now(DEFAULT_TZ)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "time": self.occurred_at.isoformat() if self.occurred_at else None,
            "step": self.step.value if isinstance(self.step, TraceStep) else str(self.step),
            "level": self.level,
            "content": to_trace_payload(self.content),
        }

    def format_log_line(self) -> str:
        timestamp = (self.occurred_at or datetime.now(DEFAULT_TZ)).strftime("%Y-%m-%d %H:%M:%S")
        content = json.dumps(to_trace_payload(self.content), ensure_ascii=False, sort_keys=True)
        return f"{self.trace_id}-{timestamp}-{self.level}: {content}"


class TraceRecorder(Protocol):
    def record(
        self,
        trace_id: str,
        step: TraceStep | str,
        content: dict[str, Any],
        *,
        level: str = "INFO",
        occurred_at: datetime | None = None,
    ) -> TraceEvent:
        ...

    def get_trace(self, trace_id: str) -> list[TraceEvent]:
        ...


class InMemoryTraceRecorder:
    def __init__(self) -> None:
        self._events: dict[str, list[TraceEvent]] = {}

    def record(
        self,
        trace_id: str,
        step: TraceStep | str,
        content: dict[str, Any],
        *,
        level: str = "INFO",
        occurred_at: datetime | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            trace_id=trace_id,
            step=step,
            content=to_trace_payload(content),
            level=level,
            occurred_at=occurred_at,
        )
        self._events.setdefault(trace_id, []).append(event)
        return event

    def get_trace(self, trace_id: str) -> list[TraceEvent]:
        return list(self._events.get(trace_id, []))

    def clear(self, trace_id: str | None = None) -> int:
        if trace_id is None:
            removed = sum(len(events) for events in self._events.values())
            self._events.clear()
            return removed
        removed = len(self._events.get(trace_id, []))
        self._events.pop(trace_id, None)
        return removed


def to_trace_payload(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): to_trace_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_trace_payload(item) for item in value]
    if is_dataclass(value):
        if hasattr(value, "to_prompt_dict"):
            return to_trace_payload(value.to_prompt_dict())
        try:
            return to_trace_payload(asdict(value))
        except TypeError:
            return {
                field.name: to_trace_payload(getattr(value, field.name))
                for field in fields(value)
            }
    return str(value)


def _coerce_step(step: TraceStep | str) -> TraceStep | str:
    if isinstance(step, TraceStep):
        return step
    try:
        return TraceStep(str(step))
    except ValueError:
        return str(step)
