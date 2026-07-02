#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mahjong_agent_v2 import (  # noqa: E402
    AgentRuntimeV2,
    CustomerProfileV2,
    InMemoryEvalRecorderV2,
    InMemoryAgentStoreV2,
    ToolGatewayV2,
    UserMessageV2,
)
from mahjong_agent_v2.tracing import InMemoryTraceRecorderV2, validate_agent_runtime_trace_completeness  # noqa: E402


REGRESSION_PATH = ROOT / "eval" / "regression" / "agent_runtime_v2_regression.jsonl"


@dataclass(slots=True)
class AgentRuntimeV2Scenario:
    id: str
    name: str
    input: dict[str, Any]
    llm_outputs: list[dict[str, Any]]
    expected: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    reply_review_enabled: bool = False


class RegressionAgentClientV2:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = [dict(output) for output in outputs]
        self.calls: list[dict[str, Any]] = []
        self.bindings: dict[str, Any] = {}

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        self.calls.append({"messages": messages, "trace_id": trace_id, "timeout_seconds": timeout_seconds})
        self._collect_bindings(messages)
        if not self.outputs:
            raise RuntimeError("regression LLM output exhausted")
        output = replace_placeholders(self.outputs.pop(0), self.bindings)
        return json.dumps(output, ensure_ascii=False)

    def _collect_bindings(self, messages: list[dict[str, str]]) -> None:
        if not messages:
            return
        try:
            payload = json.loads(messages[-1]["content"])
        except (KeyError, json.JSONDecodeError, TypeError):
            return
        for tool_result in payload.get("previous_tool_results") or []:
            result = tool_result.get("result") if isinstance(tool_result, dict) else None
            if not isinstance(result, dict):
                continue
            game = result.get("game")
            if isinstance(game, dict) and game.get("game_id"):
                self.bindings["$last_game_id"] = str(game["game_id"])
            drafts = result.get("drafts")
            if isinstance(drafts, list) and drafts:
                first_draft = drafts[0]
                if isinstance(first_draft, dict) and first_draft.get("game_id"):
                    self.bindings["$last_game_id"] = str(first_draft["game_id"])
            candidates = result.get("candidates")
            if isinstance(candidates, list) and candidates:
                first = candidates[0]
                if isinstance(first, dict):
                    customer = first.get("customer")
                    if isinstance(customer, dict):
                        self.bindings["$first_candidate_customer_id"] = str(customer.get("customer_id") or "")
                        self.bindings["$first_candidate_display_name"] = str(customer.get("display_name") or "")


def replace_placeholders(value: Any, bindings: dict[str, Any]) -> Any:
    if isinstance(value, str):
        replaced = value
        for key, binding in bindings.items():
            replaced = replaced.replace(key, str(binding))
        return replaced
    if isinstance(value, list):
        return [replace_placeholders(item, bindings) for item in value]
    if isinstance(value, dict):
        return {key: replace_placeholders(item, bindings) for key, item in value.items()}
    return value


def load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} invalid JSONL: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} JSON root must be object")
        records.append(record)
    return records


def load_scenarios(path: pathlib.Path) -> list[AgentRuntimeV2Scenario]:
    scenarios = [scenario_from_record(record) for record in load_records(path)]
    ids = [scenario.id for scenario in scenarios]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"agent runtime v2 regression has duplicate ids: {', '.join(duplicates)}")
    return scenarios


def scenario_from_record(record: dict[str, Any]) -> AgentRuntimeV2Scenario:
    scenario_id = str(record.get("id") or "").strip()
    if not scenario_id:
        raise ValueError("agent runtime v2 regression record missing id")
    llm_outputs = record.get("llm_outputs")
    if not isinstance(llm_outputs, list) or not llm_outputs:
        raise ValueError(f"{scenario_id} missing non-empty llm_outputs")
    input_payload = record.get("input")
    if not isinstance(input_payload, dict):
        raise ValueError(f"{scenario_id} missing input object")
    return AgentRuntimeV2Scenario(
        id=scenario_id,
        name=str(record.get("name") or scenario_id),
        tags=list(record.get("tags") or []),
        input=dict(input_payload),
        llm_outputs=[dict(item) for item in llm_outputs if isinstance(item, dict)],
        expected=dict(record.get("expected") or {}),
        reply_review_enabled=bool(record.get("reply_review_enabled", False)),
    )


def seeded_store() -> InMemoryAgentStoreV2:
    store = InMemoryAgentStoreV2()
    profiles = [
        CustomerProfileV2(
            customer_id="zhang",
            display_name="张哥",
            gender="男",
            preferred_games=["hangzhou_mahjong", "sichuan_mahjong"],
            preferred_stakes=["0.5", "1"],
            smoke_preference="any",
            response_score=0.9,
            notes="常客，杭麻和川麻都打。",
        ),
        CustomerProfileV2(
            customer_id="ran",
            display_name="冉姐",
            gender="女",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["0.5", "1"],
            smoke_preference="any",
            response_score=0.9,
        ),
        CustomerProfileV2(
            customer_id="he",
            display_name="何哥",
            gender="男",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["1"],
            smoke_preference="any",
            response_score=0.8,
        ),
    ]
    for profile in profiles:
        store.upsert_customer(profile)
    return store


def message_from_input(payload: dict[str, Any]) -> UserMessageV2:
    return UserMessageV2(
        conversation_id=str(payload.get("conversation_id") or "agent_runtime_v2_eval"),
        sender_id=str(payload.get("sender_id") or "zhang"),
        sender_name=str(payload.get("sender_name") or "张哥"),
        text=str(payload.get("text") or ""),
        message_id=str(payload.get("message_id") or f"{payload.get('conversation_id') or 'eval'}_msg"),
    )


def run_scenario(scenario: AgentRuntimeV2Scenario) -> tuple[int, int, list[str]]:
    store = seeded_store()
    trace = InMemoryTraceRecorderV2()
    eval_recorder = InMemoryEvalRecorderV2()
    client = RegressionAgentClientV2(scenario.llm_outputs)
    runtime = AgentRuntimeV2(
        llm_client=client,
        store=store,
        tool_gateway=ToolGatewayV2(store=store, eval_recorder=eval_recorder),
        trace_recorder=trace,
        reply_review_enabled=scenario.reply_review_enabled,
    )
    message = message_from_input(scenario.input)
    result = runtime.handle_user_message(message, trace_id=f"trace_eval_{scenario.id}")
    errors = validate_result(scenario, result, store, trace, eval_recorder)
    passed = count_checks_for_expected(scenario.expected) - len(errors)
    failed = len(errors)
    return max(0, passed), failed, errors


def validate_result(
    scenario: AgentRuntimeV2Scenario,
    result: Any,
    store: InMemoryAgentStoreV2,
    trace: InMemoryTraceRecorderV2,
    eval_recorder: InMemoryEvalRecorderV2,
) -> list[str]:
    expected = scenario.expected
    errors: list[str] = []
    if "final_reply_exact" in expected and result.final_reply != expected["final_reply_exact"]:
        errors.append(f"final_reply expected {expected['final_reply_exact']!r}, got {result.final_reply!r}")
    for item in expected.get("final_reply_contains") or []:
        if str(item) not in result.final_reply:
            errors.append(f"final_reply missing {item!r}")
    for item in expected.get("final_reply_forbidden") or []:
        if str(item) in result.final_reply:
            errors.append(f"final_reply contains forbidden {item!r}")
    if "tool_names" in expected:
        actual = [tool.name for tool in result.tool_results]
        if actual != list(expected["tool_names"]):
            errors.append(f"tool_names expected {expected['tool_names']!r}, got {actual!r}")
    tool_errors = "\n".join(str(tool.error or "") for tool in result.tool_results)
    for item in expected.get("tool_errors_contains") or []:
        if str(item) not in tool_errors:
            errors.append(f"tool error missing {item!r}")
    if "game_count" in expected and len(store.games) != int(expected["game_count"]):
        errors.append(f"game_count expected {expected['game_count']}, got {len(store.games)}")
    if "invite_draft_count" in expected and len(store.invite_drafts) != int(expected["invite_draft_count"]):
        errors.append(f"invite_draft_count expected {expected['invite_draft_count']}, got {len(store.invite_drafts)}")
    if "state_transition_count" in expected and len(result.state_transitions) != int(expected["state_transition_count"]):
        errors.append(
            f"state_transition_count expected {expected['state_transition_count']}, got {len(result.state_transitions)}"
        )
    if "badcase_count" in expected and len(eval_recorder.records) != int(expected["badcase_count"]):
        errors.append(f"badcase_count expected {expected['badcase_count']}, got {len(eval_recorder.records)}")
    trace_steps = [event.step for event in trace.get_trace(result.trace_id)]
    for step in expected.get("trace_steps_contains") or []:
        if str(step) not in trace_steps:
            errors.append(f"trace missing step {step!r}")
    if expected.get("trace_complete") is not None:
        report = validate_agent_runtime_trace_completeness(trace.get_trace(result.trace_id))
        if bool(report.complete) != bool(expected["trace_complete"]):
            errors.append(
                "trace_complete="
                f"{report.complete}, missing={report.missing_steps}, "
                f"ordering={report.ordering_errors}, pairing={report.pairing_errors}"
            )
    return errors


def count_checks_for_expected(expected: dict[str, Any]) -> int:
    total = 0
    for key in (
        "final_reply_exact",
        "tool_names",
        "game_count",
        "invite_draft_count",
        "state_transition_count",
        "badcase_count",
    ):
        if key in expected:
            total += 1
    if "trace_complete" in expected:
        total += 1
    for key in ("final_reply_contains", "final_reply_forbidden", "tool_errors_contains", "trace_steps_contains"):
        total += len(expected.get(key) or [])
    return total


def count_checks(scenarios: list[AgentRuntimeV2Scenario]) -> int:
    return sum(count_checks_for_expected(scenario.expected) for scenario in scenarios)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Agent Runtime V2 regression evals.")
    parser.add_argument("--dataset", type=pathlib.Path, default=REGRESSION_PATH)
    args = parser.parse_args(argv)
    scenarios = load_scenarios(args.dataset)
    passed_total = 0
    failed_total = 0
    for scenario in scenarios:
        passed, failed, errors = run_scenario(scenario)
        passed_total += passed
        failed_total += failed
        status = "PASS" if failed == 0 else "FAIL"
        print(f"{status} {scenario.id}: {scenario.name} ({passed} checks)")
        for error in errors:
            print(f"  - {error}")
    print(f"\nagent_runtime_v2_regression: passed={passed_total}, failed={failed_total}, scenarios={len(scenarios)}")
    return 1 if failed_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
