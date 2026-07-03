from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import now_v3


@dataclass(slots=True)
class TraceEventV3:
    trace_id: str
    step: str
    content: dict[str, Any]
    level: str = "INFO"
    occurred_at: str = field(default_factory=lambda: now_v3().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "time": self.occurred_at,
            "level": self.level,
            "step": self.step,
            "content": dict(self.content),
        }

    def to_log_line(self) -> str:
        payload = json.dumps({"step": self.step, **self.content}, ensure_ascii=False, sort_keys=True)
        return f"{self.trace_id}-{self.occurred_at}-{self.level}: {payload}"

    @classmethod
    def from_log_line(cls, line: str) -> "TraceEventV3 | None":
        match = re.match(
            r"^(?P<trace_id>.+)-(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})-(?P<level>[A-Z]+): (?P<payload>\{.*\})$",
            line,
        )
        if not match:
            return None
        try:
            payload = json.loads(match.group("payload"))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        step = payload.pop("step", None)
        if not isinstance(step, str) or not step:
            return None
        return cls(
            trace_id=match.group("trace_id"),
            step=step,
            content=payload,
            level=match.group("level"),
            occurred_at=match.group("time"),
        )


@dataclass(slots=True)
class InMemoryTraceRecorderV3:
    events: list[TraceEventV3] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def record(self, trace_id: str, step: str, content: dict[str, Any], *, level: str = "INFO") -> None:
        with self._lock:
            self.events.append(TraceEventV3(trace_id=trace_id, step=step, content=content, level=level))

    def get_trace(self, trace_id: str) -> list[TraceEventV3]:
        with self._lock:
            return [item for item in self.events if item.trace_id == trace_id]


@dataclass(slots=True)
class JsonlTraceRecorderV3:
    path: str | Path
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, trace_id: str, step: str, content: dict[str, Any], *, level: str = "INFO") -> None:
        event = TraceEventV3(trace_id=trace_id, step=step, content=content, level=level)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(event.to_log_line() + "\n")

    def get_trace(self, trace_id: str) -> list[TraceEventV3]:
        if not Path(self.path).exists():
            return []
        events: list[TraceEventV3] = []
        prefix = f"{trace_id}-"
        with self._lock:
            for line in Path(self.path).read_text(encoding="utf-8").splitlines():
                if not line.startswith(prefix):
                    continue
                event = TraceEventV3.from_log_line(line)
                if event is not None and event.trace_id == trace_id:
                    events.append(event)
        return events


def trace_steps(events: list[TraceEventV3]) -> list[str]:
    return [item.step for item in events]


def validate_trace_v3(events: list[TraceEventV3]) -> dict[str, Any]:
    steps = trace_steps(events)
    budget_denied = any(event.step == "budget_checked" and event.content.get("allowed") is False for event in events)
    llm_failed = "llm_error" in steps
    required = ["user_input", "context_built", "llm_prompt", "budget_checked", "final_output"]
    if not budget_denied and not llm_failed:
        required.append("llm_response")
    missing = [item for item in required if item not in steps]
    if "tool_called" in steps:
        for item in [
            "tool_gateway_received",
            "tool_idempotency_checked",
            "tool_definition_checked",
            "tool_schema_checked",
            "tool_gateway_completed",
            "tool_result",
        ]:
            if item not in steps and item not in missing:
                missing.append(item)
        schema_blocked = any(
            event.step == "tool_schema_checked" and event.content.get("allowed") is False
            for event in events
        )
        definition_blocked = any(
            event.step == "tool_definition_checked" and event.content.get("allowed") is False
            for event in events
        )
        permission_blocked = any(
            event.step == "tool_permission_checked" and event.content.get("allowed") is False
            for event in events
        )
        if (
            not schema_blocked
            and not definition_blocked
            and "tool_permission_checked" not in steps
            and "tool_permission_checked" not in missing
        ):
            missing.append("tool_permission_checked")
        if (
            not schema_blocked
            and not definition_blocked
            and not permission_blocked
            and "tool_idempotency_claimed" not in steps
            and "tool_idempotency_claimed" not in missing
        ):
            missing.append("tool_idempotency_claimed")
    return {"complete": not missing, "missing": missing, "steps": steps}


TraceEvent = TraceEventV3
InMemoryTraceRecorder = InMemoryTraceRecorderV3
JsonlTraceRecorder = JsonlTraceRecorderV3
validate_trace = validate_trace_v3
