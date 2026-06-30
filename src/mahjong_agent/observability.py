from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Protocol

from .models import DEFAULT_TZ


CONTROLLED_TRACE_SCHEMA_VERSION = "controlled_trace.v1"


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
    REPLY_APPROVAL = "reply_approval"
    FINAL_OUTPUT = "final_output"
    MEMORY_WRITTEN = "memory_written"


CONTROLLED_WORKFLOW_REQUIRED_TRACE_STEPS: tuple[TraceStep, ...] = (
    TraceStep.USER_INPUT,
    TraceStep.CONTEXT_BUILT,
    TraceStep.LLM_PROMPT,
    TraceStep.LLM_RESPONSE,
    TraceStep.ACTION_PROPOSED,
    TraceStep.ACTION_VALIDATED,
    TraceStep.TOOL_CALLED,
    TraceStep.STATE_TRANSITION,
    TraceStep.REPLY_DRAFTED,
    TraceStep.REPLY_GUARDED,
    TraceStep.REPLY_APPROVAL,
    TraceStep.FINAL_OUTPUT,
)


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
            "schema_version": CONTROLLED_TRACE_SCHEMA_VERSION,
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


class JsonlTraceRecorder:
    """Durable trace recorder for local production-style deployments.

    Each event is appended as one JSON object and also carries the human-readable
    log line format used by the trial console:
    traceId-time(yyyy-mm-dd hh:mm:ss)-loglevel: content
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

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
        payload = event.to_dict()
        payload["log_line"] = event.format_log_line()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        return event

    def get_trace(self, trace_id: str) -> list[TraceEvent]:
        if not self.path.exists():
            return []
        events: list[TraceEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if raw.get("trace_id") != trace_id:
                continue
            events.append(
                TraceEvent(
                    trace_id=str(raw["trace_id"]),
                    step=str(raw["step"]),
                    content=dict(raw.get("content") or {}),
                    level=str(raw.get("level") or "INFO"),
                    occurred_at=datetime.fromisoformat(str(raw["time"])) if raw.get("time") else None,
                )
            )
        return events


@dataclass(slots=True)
class TraceCompletenessReport:
    trace_id: str
    schema_version: str
    complete: bool
    required_steps: list[str]
    present_steps: list[str]
    missing_steps: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "schema_version": self.schema_version,
            "complete": self.complete,
            "required_steps": list(self.required_steps),
            "present_steps": list(self.present_steps),
            "missing_steps": list(self.missing_steps),
        }


def validate_controlled_trace_completeness(
    events: list[TraceEvent],
    *,
    required_steps: tuple[TraceStep, ...] = CONTROLLED_WORKFLOW_REQUIRED_TRACE_STEPS,
) -> TraceCompletenessReport:
    trace_id = events[0].trace_id if events else ""
    present = [_step_value(event.step) for event in events]
    required = [step.value for step in required_steps]
    missing = [step for step in required if step not in present]
    return TraceCompletenessReport(
        trace_id=trace_id,
        schema_version=CONTROLLED_TRACE_SCHEMA_VERSION,
        complete=not missing,
        required_steps=required,
        present_steps=present,
        missing_steps=missing,
    )


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


def _step_value(step: TraceStep | str) -> str:
    return step.value if isinstance(step, TraceStep) else str(step)


def _coerce_step(step: TraceStep | str) -> TraceStep | str:
    if isinstance(step, TraceStep):
        return step
    try:
        return TraceStep(str(step))
    except ValueError:
        return str(step)
