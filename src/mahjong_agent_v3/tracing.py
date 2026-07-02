from __future__ import annotations

import json
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
                # Log lines remain human-readable; API trace reads only the raw line.
                events.append(
                    TraceEventV3(
                        trace_id=trace_id,
                        step="raw_log_line",
                        content={"line": line},
                    )
                )
        return events


def trace_steps(events: list[TraceEventV3]) -> list[str]:
    return [item.step for item in events]


def validate_trace_v3(events: list[TraceEventV3]) -> dict[str, Any]:
    steps = trace_steps(events)
    required = ["user_input", "context_built", "llm_prompt", "budget_checked", "llm_response", "final_output"]
    missing = [item for item in required if item not in steps]
    if "tool_called" in steps:
        for item in [
            "tool_gateway_received",
            "tool_idempotency_checked",
            "tool_schema_checked",
            "tool_permission_checked",
            "tool_gateway_completed",
            "tool_result",
        ]:
            if item not in steps and item not in missing:
                missing.append(item)
    return {"complete": not missing, "missing": missing, "steps": steps}
