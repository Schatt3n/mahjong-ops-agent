"""记录、持久化并校验 Agent 全链路追踪事件。"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import now


@dataclass(slots=True)
class TraceEvent:
    """单条结构化追踪事件，使用 trace_id 串联一次端到端处理。"""

    trace_id: str
    step: str
    content: dict[str, Any]
    level: str = "INFO"
    occurred_at: str = field(default_factory=lambda: now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
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
    def from_log_line(cls, line: str) -> "TraceEvent | None":
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
class InMemoryTraceRecorder:
    """将追踪事件保存在内存中，供单元测试和短生命周期运行使用。"""

    events: list[TraceEvent] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def record(self, trace_id: str, step: str, content: dict[str, Any], *, level: str = "INFO") -> None:
        with self._lock:
            self.events.append(TraceEvent(trace_id=trace_id, step=step, content=content, level=level))

    def get_trace(self, trace_id: str) -> list[TraceEvent]:
        """读取链路日志；返回值只反映当前存储或运行快照。"""
        with self._lock:
            return [item for item in self.events if item.trace_id == trace_id]


@dataclass(slots=True)
class JsonlTraceRecorder:
    """以 JSONL 追加写入追踪事件，并用进程内锁保证并发写完整。"""

    path: str | Path
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        """完成数据类初始化后的依赖组装和不变量校验。"""
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, trace_id: str, step: str, content: dict[str, Any], *, level: str = "INFO") -> None:
        event = TraceEvent(trace_id=trace_id, step=step, content=content, level=level)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(event.to_log_line() + "\n")

    def get_trace(self, trace_id: str) -> list[TraceEvent]:
        """读取链路日志；返回值只反映当前存储或运行快照。"""
        if not Path(self.path).exists():
            return []
        events: list[TraceEvent] = []
        prefix = f"{trace_id}-"
        with self._lock:
            for line in Path(self.path).read_text(encoding="utf-8").splitlines():
                if not line.startswith(prefix):
                    continue
                event = TraceEvent.from_log_line(line)
                if event is not None and event.trace_id == trace_id:
                    events.append(event)
        return events


def trace_steps(events: list[TraceEvent]) -> list[str]:
    return [item.step for item in events]


def validate_trace(events: list[TraceEvent]) -> dict[str, Any]:
    """校验链路日志；失败时返回或抛出可审计的原因。"""
    steps = trace_steps(events)
    if "message_deduplicated" in steps:
        required = ["user_input", "message_deduplicated", "final_output"]
        missing = [item for item in required if item not in steps]
        execution_steps = {"context_packed", "context_built", "llm_prompt", "llm_response", "tool_called", "tool_result"}
        if any(item in steps for item in execution_steps):
            missing.append("deduplicated_trace_must_not_execute_llm_or_tools")
        return {"complete": not missing, "missing": missing, "steps": steps}
    budget_denied = any(event.step == "budget_checked" and event.content.get("allowed") is False for event in events)
    llm_failed = "llm_error" in steps
    required = ["user_input", "context_packed", "context_built", "llm_prompt", "budget_checked", "final_output"]
    if not budget_denied and not llm_failed:
        required.append("llm_response")
    if "llm_response" in steps and "action_contract_error" not in steps:
        required.append("action_proposed")
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
    expected_state_transitions = 0
    for event in events:
        if event.step != "tool_result":
            continue
        raw_transitions = event.content.get("state_transitions")
        if isinstance(raw_transitions, list):
            expected_state_transitions += len(raw_transitions)
    if expected_state_transitions:
        observed_state_transitions = sum(1 for item in steps if item in {"state_transition", "state_transition_replayed"})
        if observed_state_transitions < expected_state_transitions and "state_transition" not in missing:
            missing.append("state_transition")
    return {"complete": not missing, "missing": missing, "steps": steps}
