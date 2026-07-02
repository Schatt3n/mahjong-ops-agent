from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import DEFAULT_TZ_V2


AGENT_RUNTIME_V2_TRACE_SCHEMA_VERSION = "agent_runtime_v2.trace.v1"
AGENT_RUNTIME_V2_REQUIRED_TRACE_STEPS: tuple[str, ...] = (
    "user_input",
    "context_packed",
    "context_built",
    "llm_prompt",
    "budget_checked",
    "final_output",
)


@dataclass(slots=True)
class TraceEventV2:
    trace_id: str
    step: str
    content: dict[str, Any]
    level: str = "INFO"
    occurred_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        occurred_at = self.occurred_at or datetime.now(DEFAULT_TZ_V2)
        return {
            "schema_version": AGENT_RUNTIME_V2_TRACE_SCHEMA_VERSION,
            "trace_id": self.trace_id,
            "time": occurred_at.isoformat(),
            "step": self.step,
            "level": self.level.upper(),
            "content": _jsonable(self.content),
            "log_line": self.format_log_line(occurred_at),
        }

    def format_log_line(self, occurred_at: datetime | None = None) -> str:
        actual_time = occurred_at or self.occurred_at or datetime.now(DEFAULT_TZ_V2)
        content = json.dumps(_jsonable(self.content), ensure_ascii=False, sort_keys=True)
        return f"{self.trace_id}-{actual_time.strftime('%Y-%m-%d %H:%M:%S')}-{self.level.upper()}: {content}"


class InMemoryTraceRecorderV2:
    def __init__(self) -> None:
        self.events: dict[str, list[TraceEventV2]] = {}

    def record(self, trace_id: str, step: str, content: dict[str, Any], *, level: str = "INFO") -> TraceEventV2:
        event = TraceEventV2(trace_id=trace_id, step=step, content=_jsonable(content), level=level)
        self.events.setdefault(trace_id, []).append(event)
        return event

    def get_trace(self, trace_id: str) -> list[TraceEventV2]:
        return list(self.events.get(trace_id, []))


class JsonlTraceRecorderV2:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, trace_id: str, step: str, content: dict[str, Any], *, level: str = "INFO") -> TraceEventV2:
        event = TraceEventV2(trace_id=trace_id, step=step, content=_jsonable(content), level=level)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return event

    def get_trace(self, trace_id: str) -> list[TraceEventV2]:
        if not self.path.exists():
            return []
        events: list[TraceEventV2] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("trace_id") != trace_id:
                continue
            events.append(
                TraceEventV2(
                    trace_id=str(raw["trace_id"]),
                    step=str(raw["step"]),
                    content=dict(raw.get("content") or {}),
                    level=str(raw.get("level") or "INFO"),
                    occurred_at=datetime.fromisoformat(str(raw["time"])) if raw.get("time") else None,
                )
            )
        return events


@dataclass(slots=True)
class TraceCompletenessReportV2:
    trace_id: str
    schema_version: str
    complete: bool
    required_steps: list[str]
    present_steps: list[str]
    missing_steps: list[str]
    ordering_errors: list[str]
    pairing_errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "schema_version": self.schema_version,
            "complete": self.complete,
            "required_steps": list(self.required_steps),
            "present_steps": list(self.present_steps),
            "missing_steps": list(self.missing_steps),
            "ordering_errors": list(self.ordering_errors),
            "pairing_errors": list(self.pairing_errors),
        }


def validate_agent_runtime_trace_completeness(
    events: list[TraceEventV2],
    *,
    required_steps: tuple[str, ...] = AGENT_RUNTIME_V2_REQUIRED_TRACE_STEPS,
) -> TraceCompletenessReportV2:
    trace_id = events[0].trace_id if events else ""
    present = [event.step for event in events]
    if "message_deduplicated" in present:
        required = ["message_deduplicated"]
    else:
        required = list(required_steps)
        if "llm_response" in present:
            required.extend(["llm_response", "action_proposed"])
        elif "llm_error" in present:
            required.append("llm_error")
        required.extend(_tool_gateway_required_steps(present))
        required.extend(_reply_review_required_steps(present))
    missing = [step for step in required if step not in present]
    ordering_errors = _trace_ordering_errors(present)
    pairing_errors = _tool_pairing_errors(present)
    return TraceCompletenessReportV2(
        trace_id=trace_id,
        schema_version=AGENT_RUNTIME_V2_TRACE_SCHEMA_VERSION,
        complete=not missing and not ordering_errors and not pairing_errors,
        required_steps=required,
        present_steps=present,
        missing_steps=missing,
        ordering_errors=ordering_errors,
        pairing_errors=pairing_errors,
    )


def _tool_gateway_required_steps(present: list[str]) -> list[str]:
    if "tool_called" not in present:
        return []
    return ["tool_gateway_received", "tool_idempotency_checked", "tool_gateway_completed"]


def _reply_review_required_steps(present: list[str]) -> list[str]:
    review_steps = {
        step
        for step in present
        if step.startswith("reply_review_") or step == "reply_revised"
    }
    if not review_steps:
        return []
    required = ["reply_review_prompt", "reply_review_budget_checked"]
    if "reply_review_skipped" in review_steps:
        required.append("reply_review_skipped")
        return required
    if "reply_review_error" in review_steps:
        required.append("reply_review_error")
        return required
    required.append("reply_review_response")
    if "reply_review_contract_error" in review_steps:
        required.append("reply_review_contract_error")
        return required
    required.append("reply_review_proposed")
    if "reply_revised" in review_steps:
        required.append("reply_revised")
    return required


def _trace_ordering_errors(steps: list[str]) -> list[str]:
    if not steps:
        return ["trace has no events"]
    if "message_deduplicated" in steps:
        return [] if steps[-1] == "message_deduplicated" else ["message_deduplicated trace must end immediately"]
    errors: list[str] = []
    for before, after in [
        ("user_input", "context_packed"),
        ("context_packed", "context_built"),
        ("context_built", "llm_prompt"),
        ("llm_prompt", "budget_checked"),
    ]:
        if before in steps and after in steps and steps.index(before) > steps.index(after):
            errors.append(f"{before} must occur before {after}")
    if "llm_response" in steps and "budget_checked" in steps and steps.index("budget_checked") > steps.index("llm_response"):
        errors.append("budget_checked must occur before llm_response")
    if "llm_error" in steps and "budget_checked" in steps and steps.index("budget_checked") > steps.index("llm_error"):
        errors.append("budget_checked must occur before llm_error")
    if "action_proposed" in steps and "llm_response" in steps and steps.index("llm_response") > steps.index("action_proposed"):
        errors.append("llm_response must occur before action_proposed")
    if "tool_called" in steps and "action_proposed" in steps and steps.index("action_proposed") > steps.index("tool_called"):
        errors.append("action_proposed must occur before tool_called")
    for before, after in [
        ("tool_called", "tool_gateway_received"),
        ("tool_gateway_received", "tool_idempotency_checked"),
        ("tool_idempotency_checked", "tool_definition_checked"),
        ("tool_definition_checked", "tool_schema_checked"),
        ("tool_schema_checked", "tool_permission_checked"),
        ("tool_idempotency_checked", "tool_gateway_completed"),
        ("tool_gateway_completed", "tool_result"),
    ]:
        if before in steps and after in steps and steps.index(before) > steps.index(after):
            errors.append(f"{before} must occur before {after}")
    if "state_transition" in steps and "tool_result" in steps and steps.index("tool_result") > steps.index("state_transition"):
        errors.append("tool_result must occur before state_transition")
    for before, after in [
        ("reply_review_prompt", "reply_review_budget_checked"),
        ("reply_review_budget_checked", "reply_review_response"),
        ("reply_review_budget_checked", "reply_review_skipped"),
        ("reply_review_budget_checked", "reply_review_error"),
        ("reply_review_response", "reply_review_contract_error"),
        ("reply_review_response", "reply_review_proposed"),
        ("reply_review_proposed", "reply_revised"),
        ("reply_review_skipped", "final_output"),
        ("reply_review_error", "final_output"),
        ("reply_review_contract_error", "final_output"),
        ("reply_review_proposed", "final_output"),
        ("reply_revised", "final_output"),
    ]:
        if before in steps and after in steps and steps.index(before) > steps.index(after):
            errors.append(f"{before} must occur before {after}")
    if steps[-1] != "final_output":
        errors.append("trace must end with final_output")
    return errors


def _tool_pairing_errors(steps: list[str]) -> list[str]:
    errors: list[str] = []
    called_indexes = [index for index, step in enumerate(steps) if step == "tool_called"]
    result_indexes = [index for index, step in enumerate(steps) if step == "tool_result"]
    if len(called_indexes) != len(result_indexes):
        errors.append(f"tool_called count {len(called_indexes)} != tool_result count {len(result_indexes)}")
        return errors
    for pair_index, (called_index, result_index) in enumerate(zip(called_indexes, result_indexes), start=1):
        if called_index > result_index:
            errors.append(f"tool pair {pair_index} has tool_result before tool_called")
    return errors


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
