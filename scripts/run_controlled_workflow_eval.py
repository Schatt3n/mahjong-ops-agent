#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mahjong_agent import (  # noqa: E402
    AgentCore,
    ChannelType,
    ControlledWorkflowService,
    CustomerProfile,
    InMemoryShortTermMemoryStore,
    InMemoryTraceRecorder,
    Message,
    PlayPreference,
    SemanticResolver,
    WorkflowContextBuilder,
)
from mahjong_agent.memory import ShortTermMemoryRecord  # noqa: E402
from mahjong_agent.workflow_models import GameRequirement, SlotSource, SlotValue, ToolName, UserMessage  # noqa: E402


TZ = ZoneInfo("Asia/Shanghai")
DEFAULT_NOW = datetime(2026, 6, 30, 16, 0, tzinfo=TZ)
REGRESSION_PATH = ROOT / "eval" / "regression" / "controlled_workflow_regression.jsonl"
REQUIRED_TRACE_STEPS = [
    "user_input",
    "context_built",
    "llm_prompt",
    "llm_response",
    "action_proposed",
    "action_validated",
    "tool_called",
    "state_transition",
    "reply_drafted",
    "reply_guarded",
    "final_output",
]


@dataclass(slots=True)
class ControlledStep:
    name: str
    text: str
    sender_id: str
    sender_name: str
    conversation_id: str
    semantic_output: dict[str, Any]
    expected: dict[str, Any] = field(default_factory=dict)
    now: datetime = DEFAULT_NOW


@dataclass(slots=True)
class ControlledScenario:
    id: str
    name: str
    tags: list[str] = field(default_factory=list)
    setup_memory: list[dict[str, Any]] = field(default_factory=list)
    steps: list[ControlledStep] = field(default_factory=list)


class FixedSemanticLLMClient:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        trace_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "messages": messages,
                "trace_id": trace_id,
                "timeout_seconds": timeout_seconds,
            }
        )
        if not self.outputs:
            raise RuntimeError("controlled workflow eval has no semantic output left")
        return self.outputs.pop(0)


def load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSONL") from exc
    return records


def load_scenarios(path: pathlib.Path) -> list[ControlledScenario]:
    scenarios = [scenario_from_record(record) for record in load_records(path)]
    ids = [scenario.id for scenario in scenarios]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"controlled workflow regression has duplicate ids: {', '.join(duplicates)}")
    return scenarios


def scenario_from_record(record: dict[str, Any]) -> ControlledScenario:
    scenario_id = str(record.get("id") or "").strip()
    if not scenario_id:
        raise ValueError("controlled workflow regression record missing id")
    steps_data = record.get("steps")
    if steps_data is None:
        steps_data = [
            {
                "name": record.get("name") or scenario_id,
                "text": record.get("text"),
                "sender_id": record.get("sender_id"),
                "sender_name": record.get("sender_name"),
                "conversation_id": record.get("conversation_id"),
                "semantic_output": record.get("semantic_output"),
                "expected": record.get("expected") or {},
                "now": record.get("now"),
            }
        ]
    return ControlledScenario(
        id=scenario_id,
        name=str(record.get("name") or scenario_id),
        tags=list(record.get("tags") or []),
        setup_memory=list(record.get("setup_memory") or []),
        steps=[step_from_record(step) for step in steps_data],
    )


def step_from_record(record: dict[str, Any]) -> ControlledStep:
    semantic_output = record.get("semantic_output")
    if not isinstance(semantic_output, dict):
        raise ValueError("controlled workflow step missing semantic_output object")
    return ControlledStep(
        name=str(record.get("name") or "step"),
        text=str(record.get("text") or ""),
        sender_id=str(record.get("sender_id") or "zhang"),
        sender_name=str(record.get("sender_name") or record.get("sender_id") or "张哥"),
        conversation_id=str(record.get("conversation_id") or "controlled_eval"),
        semantic_output=semantic_output,
        expected=dict(record.get("expected") or {}),
        now=parse_dt(record.get("now")) or DEFAULT_NOW,
    )


def parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=TZ)


def seed_core() -> AgentCore:
    core = AgentCore()
    customers = [
        CustomerProfile(
            id="ran",
            display_name="冉姐",
            preferred_levels=["0.5"],
            smoke_free_preference=True,
            play_preferences=[PlayPreference(game_type="hangzhou_mahjong", preferred_levels=["0.5"])],
            usual_start_hours=[16, 18],
        ),
        CustomerProfile(
            id="liu",
            display_name="刘姐",
            preferred_levels=["0.5", "1"],
            smoke_free_preference=False,
            play_preferences=[PlayPreference(game_type="hangzhou_mahjong", preferred_levels=["0.5", "1"])],
            usual_start_hours=[16, 18],
        ),
        CustomerProfile(
            id="pan",
            display_name="潘姐",
            preferred_levels=["1"],
            smoke_free_preference=False,
            play_preferences=[PlayPreference(game_type="hangzhou_mahjong", preferred_levels=["1"])],
            usual_start_hours=[18],
        ),
    ]
    for customer in customers:
        core.upsert_customer(customer)
    return core


def seed_memory(memory: InMemoryShortTermMemoryStore, scenario: ControlledScenario) -> None:
    for item in scenario.setup_memory:
        conversation_id = str(item.get("conversation_id") or "controlled_eval")
        sender_id = str(item.get("sender_id") or "zhang")
        created_at = parse_dt(item.get("created_at")) or DEFAULT_NOW
        user_message = UserMessage(
            text=str(item.get("user_text") or ""),
            sender_id=sender_id,
            sender_name=str(item.get("sender_name") or sender_id),
            conversation_id=conversation_id,
            trace_id=str(item.get("trace_id") or f"setup_{scenario.id}"),
            message_id=str(item.get("message_id") or f"setup_msg_{scenario.id}"),
            sent_at=created_at,
        )
        memory.append(
            ShortTermMemoryRecord(
                conversation_id=conversation_id,
                sender_id=sender_id,
                user_message=user_message,
                system_reply=str(item.get("system_reply") or ""),
                game_requirement=game_requirement_from_slots(item.get("game_requirement_slots") or {}),
                created_at=created_at,
            ),
            now=created_at,
        )


def game_requirement_from_slots(slots: dict[str, Any]) -> GameRequirement:
    requirement = GameRequirement()
    for name, raw in slots.items():
        if isinstance(raw, dict):
            value = raw.get("value")
            source = slot_source(raw.get("source"))
            confidence = float(raw.get("confidence") or 0.9)
            confirmed = bool(raw.get("confirmed", True))
            needs_confirmation = bool(raw.get("needs_confirmation", not confirmed))
        else:
            value = raw
            source = SlotSource.EXPLICIT
            confidence = 0.9
            confirmed = True
            needs_confirmation = False
        requirement.set_slot(
            SlotValue(
                name=str(name),
                value=value,
                source=source,
                confidence=confidence,
                confirmed=confirmed,
                needs_confirmation=needs_confirmation,
            )
        )
    return requirement


def slot_source(value: Any) -> SlotSource:
    try:
        return SlotSource(str(value or SlotSource.EXPLICIT.value))
    except ValueError:
        return SlotSource.UNKNOWN


def make_message(step: ControlledStep, *, trace_id: str) -> Message:
    return Message(
        text=step.text,
        sender_id=step.sender_id,
        sender_name=step.sender_name,
        channel_id=step.conversation_id,
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=step.now,
        id=f"{trace_id}_message",
        metadata={"conversation_id": step.conversation_id, "trace_id": trace_id},
    )


def run_scenario(scenario: ControlledScenario) -> tuple[int, int]:
    core = seed_core()
    memory = InMemoryShortTermMemoryStore()
    seed_memory(memory, scenario)
    llm_client = FixedSemanticLLMClient([step.semantic_output for step in scenario.steps])
    trace_recorder = InMemoryTraceRecorder()
    service = ControlledWorkflowService(
        core=core,
        context_builder=WorkflowContextBuilder(core, memory),
        semantic_resolver=SemanticResolver(llm_client),
        memory_store=memory,
        trace_recorder=trace_recorder,
    )
    passed = 0
    failed = 0
    for step_index, step in enumerate(scenario.steps, start=1):
        trace_id = f"eval_{scenario.id}_{step_index}"
        result = service.handle_message(make_message(step, trace_id=trace_id), now=step.now, trace_id=trace_id)
        errors = validate_result(step.expected, result)
        label = scenario.name if len(scenario.steps) == 1 else f"{scenario.name}/{step.name}"
        if errors:
            failed += 1
            print(f"FAIL {label}: " + "; ".join(errors))
        else:
            passed += 1
            action = result.run.validated_action.effective_action.value if result.run.validated_action else "none"
            print(f"PASS {label}: {action} -> {result.final_text or '<silent>'}")
    return passed, failed


def validate_result(expected: dict[str, Any], result: Any) -> list[str]:
    errors: list[str] = []
    semantic = result.run.semantic_resolution
    validated = result.run.validated_action
    if expected.get("intent") and semantic.intent.value != expected["intent"]:
        errors.append(f"intent={semantic.intent.value}, expected={expected['intent']}")
    if expected.get("effective_action") and validated.effective_action.value != expected["effective_action"]:
        errors.append(f"effective_action={validated.effective_action.value}, expected={expected['effective_action']}")
    if expected.get("required_tools") is not None:
        actual_tools = [tool.value for tool in validated.required_tools]
        if actual_tools != list(expected["required_tools"]):
            errors.append(f"required_tools={actual_tools}, expected={expected['required_tools']}")
    if expected.get("slot_values"):
        for name, value in dict(expected["slot_values"]).items():
            slot = semantic.game_requirement.slot(name)
            if slot is None or slot.value != value:
                errors.append(f"slot {name}={slot.value if slot else None}, expected={value}")
    if expected.get("outbox_count_min") is not None:
        outbox = result.tool_orchestration.result_for(ToolName.CREATE_PENDING_OUTBOX)
        count = int(((outbox.result if outbox else {}) or {}).get("result_count") or 0)
        if count < int(expected["outbox_count_min"]):
            errors.append(f"outbox_count={count}, expected>={expected['outbox_count_min']}")
    if expected.get("final_text_exact") is not None and result.final_text != expected["final_text_exact"]:
        errors.append(f"final_text={result.final_text!r}, expected={expected['final_text_exact']!r}")
    for fragment in expected.get("final_text_contains") or []:
        if fragment not in result.final_text:
            errors.append(f"final_text missing {fragment!r}: {result.final_text!r}")
    if expected.get("state_statuses") is not None:
        statuses = [transition.to_status for transition in result.run.state_transitions]
        if statuses != list(expected["state_statuses"]):
            errors.append(f"state_statuses={statuses}, expected={expected['state_statuses']}")
    if expected.get("trace_steps_contains") is not None:
        steps = [event.step.value if hasattr(event.step, "value") else str(event.step) for event in result.trace_events]
        for step in expected["trace_steps_contains"]:
            if step not in steps:
                errors.append(f"trace missing step {step!r}")
    if expected.get("followup_response_type") is not None:
        followup = result.context_build.followup_context
        actual = followup.get("current_message_response_type")
        if actual != expected["followup_response_type"]:
            errors.append(f"followup_response_type={actual}, expected={expected['followup_response_type']}")
    if expected.get("should_treat_as_followup") is not None:
        actual = result.context_build.followup_context.get("should_treat_current_message_as_followup")
        if bool(actual) != bool(expected["should_treat_as_followup"]):
            errors.append(f"should_treat_as_followup={actual}, expected={expected['should_treat_as_followup']}")
    return errors


def count_checks(scenarios: list[ControlledScenario]) -> int:
    return sum(1 for scenario in scenarios for step in scenario.steps if step.expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(REGRESSION_PATH))
    args = parser.parse_args()
    path = pathlib.Path(args.dataset)
    scenarios = load_scenarios(path)
    passed = 0
    failed = 0
    for scenario in scenarios:
        scenario_passed, scenario_failed = run_scenario(scenario)
        passed += scenario_passed
        failed += scenario_failed
    print(f"\n{passed} passed, {failed} failed")
    print(f"dataset={path}")
    print(f"checks={count_checks(scenarios)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
