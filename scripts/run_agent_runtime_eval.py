#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mahjong_agent_runtime import (  # noqa: E402
    AgentRuntime,
    CustomerProfile,
    InMemoryTraceRecorder,
    SQLiteAgentStore,
    ToolGateway,
    UserMessage,
)
from mahjong_agent_runtime.models import ConversationRole, ConversationTurn  # noqa: E402
from mahjong_agent_runtime.models import now as runtime_now  # noqa: E402
from mahjong_agent_runtime.tracing import validate_trace  # noqa: E402


REGRESSION_PATH = ROOT / "eval" / "regression" / "agent_runtime_regression.jsonl"


@dataclass(slots=True)
class AgentRuntimeScenario:
    id: str
    name: str
    input: dict[str, Any]
    llm_outputs: list[dict[str, Any]]
    expected: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


class RegressionAgentClient:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = [dict(output) for output in outputs]
        self.calls: list[dict[str, Any]] = []
        future_start = (runtime_now() + timedelta(days=1)).replace(hour=17, minute=0, second=0, microsecond=0)
        self.bindings: dict[str, str] = {"$future_17_at": future_start.isoformat()}

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
        except (KeyError, TypeError, json.JSONDecodeError):
            return
        for tool_result in payload.get("previous_tool_results") or []:
            if not isinstance(tool_result, dict):
                continue
            result = tool_result.get("result")
            if not isinstance(result, dict):
                continue
            game = result.get("game")
            if isinstance(game, dict) and game.get("game_id"):
                self.bindings["$last_game_id"] = str(game["game_id"])
            drafts = result.get("drafts")
            if isinstance(drafts, list) and drafts:
                draft = drafts[0]
                if isinstance(draft, dict) and draft.get("game_id"):
                    self.bindings["$last_game_id"] = str(draft["game_id"])
            candidates = result.get("candidates")
            if isinstance(candidates, list) and candidates:
                first = candidates[0]
                if isinstance(first, dict):
                    customer = first.get("customer")
                    if isinstance(customer, dict):
                        self.bindings["$first_candidate_customer_id"] = str(customer.get("customer_id") or "")
                        self.bindings["$first_candidate_display_name"] = str(customer.get("display_name") or "")


def replace_placeholders(value: Any, bindings: dict[str, str]) -> Any:
    if isinstance(value, str):
        replaced = value
        for key, binding in bindings.items():
            replaced = replaced.replace(key, binding)
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


def load_scenarios(path: pathlib.Path) -> list[AgentRuntimeScenario]:
    scenarios = [scenario_from_record(record) for record in load_records(path)]
    ids = [scenario.id for scenario in scenarios]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"agent runtime regression has duplicate ids: {', '.join(duplicates)}")
    return scenarios


def scenario_from_record(record: dict[str, Any]) -> AgentRuntimeScenario:
    scenario_id = str(record.get("id") or "").strip()
    if not scenario_id:
        raise ValueError("agent runtime regression record missing id")
    llm_outputs = record.get("llm_outputs")
    if not isinstance(llm_outputs, list) or not llm_outputs:
        raise ValueError(f"{scenario_id} missing non-empty llm_outputs")
    input_payload = record.get("input")
    if not isinstance(input_payload, dict):
        raise ValueError(f"{scenario_id} missing input object")
    return AgentRuntimeScenario(
        id=scenario_id,
        name=str(record.get("name") or scenario_id),
        tags=list(record.get("tags") or []),
        input=dict(input_payload),
        llm_outputs=[dict(item) for item in llm_outputs if isinstance(item, dict)],
        expected=dict(record.get("expected") or {}),
    )


def seeded_store(path: pathlib.Path) -> SQLiteAgentStore:
    store = SQLiteAgentStore(path)
    profiles = [
        CustomerProfile(
            customer_id="zhang",
            display_name="张哥",
            gender="男",
            preferred_games=["hangzhou_mahjong", "sichuan_mahjong"],
            preferred_stakes=["0.5", "1"],
            smoke_preference="any",
            response_score=0.9,
            notes="常客，杭麻和川麻都打。",
        ),
        CustomerProfile(
            customer_id="ran",
            display_name="冉姐",
            gender="女",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["0.5", "1"],
            smoke_preference="any",
            response_score=0.9,
        ),
        CustomerProfile(
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


def message_from_input(payload: dict[str, Any]) -> UserMessage:
    return UserMessage(
        conversation_id=str(payload.get("conversation_id") or "agent_runtime_eval"),
        sender_id=str(payload.get("sender_id") or "zhang"),
        sender_name=str(payload.get("sender_name") or "张哥"),
        text=str(payload.get("text") or ""),
        message_id=str(payload.get("message_id") or f"{payload.get('conversation_id') or 'eval'}_msg"),
    )


def seed_pre_turns(store: SQLiteAgentStore, payload: dict[str, Any]) -> None:
    conversation_id = str(payload.get("conversation_id") or "agent_runtime_eval")
    for index, raw in enumerate(payload.get("pre_turns") or [], start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"pre_turns[{index}] must be object")
        role_value = str(raw.get("role") or "").strip()
        content = str(raw.get("content") or "")
        trace_id = str(raw.get("trace_id") or f"pre_turn_{index}")
        if role_value == ConversationRole.USER.value:
            store.append_turn(
                conversation_id,
                ConversationTurn(
                    role=ConversationRole.USER,
                    content=content,
                    trace_id=trace_id,
                    sender_id=str(raw.get("sender_id") or payload.get("sender_id") or "zhang"),
                    sender_name=str(raw.get("sender_name") or payload.get("sender_name") or "张哥"),
                ),
            )
        elif role_value == ConversationRole.ASSISTANT.value:
            store.append_assistant_turn(conversation_id, content, trace_id)
        elif role_value == ConversationRole.TOOL.value:
            store.append_tool_turn(conversation_id, content, trace_id)
        else:
            raise ValueError(f"pre_turns[{index}].role must be user, assistant, or tool")


def run_scenario(scenario: AgentRuntimeScenario) -> tuple[int, int, list[str]]:
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = pathlib.Path(temp_dir) / f"{scenario.id}.sqlite3"
        store = seeded_store(db_path)
        seed_pre_turns(store, scenario.input)
        trace = InMemoryTraceRecorder()
        client = RegressionAgentClient(scenario.llm_outputs)
        runtime = AgentRuntime(
            llm_client=client,
            store=store,
            tool_gateway=ToolGateway(store=store),
            trace_recorder=trace,
        )
        message = message_from_input(scenario.input)
        result = runtime.handle_user_message(message, trace_id=f"trace_eval_{scenario.id}")
        reopened = SQLiteAgentStore(db_path)
        errors = validate_result(scenario, result, store, reopened, trace)
    passed = count_checks_for_expected(scenario.expected) - len(errors)
    failed = len(errors)
    return max(0, passed), failed, errors


def validate_result(
    scenario: AgentRuntimeScenario,
    result: Any,
    store: SQLiteAgentStore,
    reopened: SQLiteAgentStore,
    trace: InMemoryTraceRecorder,
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
    business_transitions = [
        transition
        for transition in result.state_transitions
        if transition.entity_type not in {"conversation_version", "assistant_reply", "task_context"}
    ]
    if "state_transition_count" in expected and len(business_transitions) != int(expected["state_transition_count"]):
        errors.append(
            f"state_transition_count expected {expected['state_transition_count']}, got {len(business_transitions)}"
        )
    if "badcase_count" in expected and len(store.badcases) != int(expected["badcase_count"]):
        errors.append(f"badcase_count expected {expected['badcase_count']}, got {len(store.badcases)}")
    if "persisted_game_count" in expected and len(reopened.games) != int(expected["persisted_game_count"]):
        errors.append(f"persisted_game_count expected {expected['persisted_game_count']}, got {len(reopened.games)}")
    if "persisted_invite_draft_count" in expected and len(reopened.invite_drafts) != int(expected["persisted_invite_draft_count"]):
        errors.append(
            "persisted_invite_draft_count expected "
            f"{expected['persisted_invite_draft_count']}, got {len(reopened.invite_drafts)}"
        )
    if "persisted_badcase_count" in expected and len(reopened.badcases) != int(expected["persisted_badcase_count"]):
        errors.append(f"persisted_badcase_count expected {expected['persisted_badcase_count']}, got {len(reopened.badcases)}")
    first_game = next(iter(reopened.games.values()), None)
    if "persisted_first_game_requirement_contains" in expected:
        if first_game is None:
            errors.append("persisted_first_game_requirement_contains expected first game, got none")
        else:
            mismatch = mapping_contains(
                first_game.requirement,
                dict(expected["persisted_first_game_requirement_contains"]),
                path="persisted_first_game.requirement",
            )
            if mismatch:
                errors.append(mismatch)
    if "persisted_first_party_contains" in expected:
        if first_game is None or not first_game.parties:
            errors.append("persisted_first_party_contains expected first game party, got none")
        else:
            mismatch = mapping_contains(
                first_game.parties[0].to_dict(),
                dict(expected["persisted_first_party_contains"]),
                path="persisted_first_party",
            )
            if mismatch:
                errors.append(mismatch)
    trace_events = trace.get_trace(result.trace_id)
    trace_steps = [event.step for event in trace_events]
    for step in expected.get("trace_steps_contains") or []:
        if str(step) not in trace_steps:
            errors.append(f"trace missing step {step!r}")
    if expected.get("trace_complete") is not None:
        report = validate_trace(trace_events)
        if bool(report["complete"]) != bool(expected["trace_complete"]):
            errors.append(f"trace_complete={report['complete']}, missing={report['missing']}")
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
        "persisted_game_count",
        "persisted_invite_draft_count",
        "persisted_badcase_count",
        "persisted_first_game_requirement_contains",
        "persisted_first_party_contains",
    ):
        if key in expected:
            total += 1
    if "trace_complete" in expected:
        total += 1
    for key in ("final_reply_contains", "final_reply_forbidden", "tool_errors_contains", "trace_steps_contains"):
        total += len(expected.get(key) or [])
    return total


def count_checks(scenarios: list[AgentRuntimeScenario]) -> int:
    return sum(count_checks_for_expected(scenario.expected) for scenario in scenarios)


def mapping_contains(actual: dict[str, Any], expected: dict[str, Any], *, path: str) -> str | None:
    for key, expected_value in expected.items():
        item_path = f"{path}.{key}"
        if key not in actual:
            return f"{item_path} missing"
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                return f"{item_path} expected object, got {actual_value!r}"
            mismatch = mapping_contains(actual_value, expected_value, path=item_path)
            if mismatch:
                return mismatch
        elif actual_value != expected_value:
            return f"{item_path} expected {expected_value!r}, got {actual_value!r}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Agent Runtime regression evals.")
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
    print(f"\nagent_runtime_regression: passed={passed_total}, failed={failed_total}, scenarios={len(scenarios)}")
    return 1 if failed_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
