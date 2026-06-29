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

from mahjong_agent import AgentResponder, ChannelType, CustomerProfile, Message, ReplyAction  # noqa: E402


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 6, 16, 12, 0, tzinfo=TZ)
GOLDEN_PATH = ROOT / "eval" / "golden_dataset.jsonl"
BADCASE_PATH = ROOT / "eval" / "badcases.jsonl"


@dataclass(slots=True)
class Expected:
    action: ReplyAction | None = None
    contains: list[str] = field(default_factory=list)
    should_reply: bool | None = None


@dataclass(slots=True)
class ScenarioStep:
    name: str
    text: str
    sender_id: str
    sender_name: str | None = None
    expected: Expected = field(default_factory=Expected)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Scenario:
    id: str
    name: str
    tags: list[str] = field(default_factory=list)
    steps: list[ScenarioStep] = field(default_factory=list)


def seed() -> AgentResponder:
    responder = AgentResponder(invite_limit=3)
    customers = [
        CustomerProfile(
            id="amy",
            display_name="Amy",
            preferred_levels=["0.5"],
            tags=["无烟"],
            smoke_free_preference=True,
            usual_start_hours=[17, 18],
        ),
        CustomerProfile(
            id="chen",
            display_name="陈姐",
            preferred_levels=["0.5", "1"],
            tags=["无烟", "熟人局"],
            smoke_free_preference=True,
            usual_start_hours=[17],
        ),
        CustomerProfile(
            id="ben",
            display_name="Ben",
            preferred_levels=["2"],
            tags=["可吸烟"],
            smoke_free_preference=False,
            usual_start_hours=[21],
        ),
    ]
    for customer in customers:
        responder.core.upsert_customer(customer)
    return responder


def make_message(text: str, sender_id: str, name: str | None = None) -> Message:
    return Message(
        text=text,
        sender_id=sender_id,
        sender_name=name or sender_id,
        channel_id="group_main",
        channel_type=ChannelType.WECHAT_GROUP,
    )


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


def load_scenarios(path: pathlib.Path) -> list[Scenario]:
    if not path.exists():
        raise FileNotFoundError(f"评估集不存在: {path}")
    scenarios = [scenario_from_record(record) for record in load_records(path)]
    ensure_unique_ids(scenarios)
    return scenarios


def ensure_unique_ids(scenarios: list[Scenario]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for scenario in scenarios:
        if scenario.id in seen:
            duplicates.append(scenario.id)
        seen.add(scenario.id)
    if duplicates:
        raise ValueError(f"评估集存在重复 id: {', '.join(sorted(duplicates))}")


def scenario_from_record(record: dict[str, Any]) -> Scenario:
    scenario_id = str(record.get("id") or "").strip()
    if not scenario_id:
        raise ValueError("评估样本缺少 id")
    steps_data = record.get("steps")
    if steps_data is None:
        steps_data = [
            {
                "name": record.get("name") or scenario_id,
                "text": record.get("text"),
                "sender_id": record.get("sender_id"),
                "sender_name": record.get("sender_name"),
                "metadata": record.get("metadata") or {},
                "expected": record.get("expected") or {},
            }
        ]
    return Scenario(
        id=scenario_id,
        name=str(record.get("name") or scenario_id),
        tags=list(record.get("tags") or []),
        steps=[step_from_record(step) for step in steps_data],
    )


def step_from_record(record: dict[str, Any]) -> ScenarioStep:
    expected = expected_from_record(record.get("expected") or {})
    return ScenarioStep(
        name=str(record.get("name") or "step"),
        text=str(record.get("text") or ""),
        sender_id=str(record.get("sender_id") or ""),
        sender_name=record.get("sender_name"),
        expected=expected,
        metadata=dict(record.get("metadata") or {}),
    )


def expected_from_record(record: dict[str, Any]) -> Expected:
    action = record.get("action")
    contains = record.get("contains") or []
    if isinstance(contains, str):
        contains = [contains]
    return Expected(
        action=ReplyAction(action) if action else None,
        contains=list(contains),
        should_reply=record.get("should_reply"),
    )


def has_expectation(expected: Expected) -> bool:
    return expected.action is not None or bool(expected.contains) or expected.should_reply is not None


def run_scenario(scenario: Scenario, *, record_failures: bool) -> tuple[int, int]:
    responder = seed()
    passed = 0
    failed = 0
    for step_index, step in enumerate(scenario.steps, start=1):
        sender_id = resolve_sender_id(step.sender_id, responder)
        message = make_message(step.text, sender_id, step.sender_name)
        message.metadata.update(step.metadata)
        decision = responder.respond(message, now=NOW)
        if not has_expectation(step.expected):
            continue

        errors = validate_decision(step.expected, decision)
        label = scenario.name if len(scenario.steps) == 1 else f"{scenario.name}/{step.name}"
        if errors:
            failed += 1
            print(f"FAIL {label}: " + "; ".join(errors))
            if record_failures:
                append_badcase(scenario, step, step_index, decision, errors)
        else:
            passed += 1
            print(f"PASS {label}: {decision.action.value} -> {decision.reply_text or '<silent>'}")
    return passed, failed


def resolve_sender_id(sender_id: str, responder: AgentResponder) -> str:
    if sender_id != "__first_invited_customer__":
        return sender_id
    invitation = next(iter(responder.core.store.invitations.values()), None)
    if invitation is None:
        raise ValueError("无法解析 __first_invited_customer__，当前场景还没有邀约")
    return invitation.customer_id


def validate_decision(expected: Expected, decision: Any) -> list[str]:
    errors: list[str] = []
    if expected.action is not None and decision.action != expected.action:
        errors.append(f"action={decision.action.value}, expected={expected.action.value}")
    for fragment in expected.contains:
        if fragment not in decision.reply_text:
            errors.append(f"reply does not contain {fragment!r}: {decision.reply_text!r}")
    if expected.should_reply is not None and decision.should_reply != expected.should_reply:
        errors.append(f"should_reply={decision.should_reply}, expected={expected.should_reply}")
    return errors


def append_badcase(
    scenario: Scenario,
    step: ScenarioStep,
    step_index: int,
    decision: Any,
    errors: list[str],
) -> None:
    BADCASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(TZ).strftime("%Y%m%d%H%M%S")
    record = {
        "schema_version": 1,
        "kind": "badcase",
        "id": f"badcase_{stamp}_{scenario.id}_{step_index}",
        "source_scenario_id": scenario.id,
        "source_step_name": step.name,
        "observed_at": datetime.now(TZ).isoformat(),
        "tags": scenario.tags,
        "text": step.text,
        "sender_id": step.sender_id,
        "sender_name": step.sender_name,
        "metadata": step.metadata,
        "expected": expected_to_record(step.expected),
        "actual": {
            "action": decision.action.value,
            "reply_text": decision.reply_text,
            "should_reply": decision.should_reply,
            "needs_human_review": decision.needs_human_review,
            "notes": decision.notes,
        },
        "errors": errors,
        "triage_status": "new",
    }
    with BADCASE_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def expected_to_record(expected: Expected) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if expected.action is not None:
        data["action"] = expected.action.value
    if expected.contains:
        data["contains"] = expected.contains
    if expected.should_reply is not None:
        data["should_reply"] = expected.should_reply
    return data


def count_checks(scenarios: list[Scenario]) -> int:
    return sum(1 for scenario in scenarios for step in scenario.steps if has_expectation(step.expected))


def main() -> int:
    parser = argparse.ArgumentParser(description="运行麻将馆运营 workflow 场景评估集")
    parser.add_argument("--dataset", type=pathlib.Path, default=GOLDEN_PATH, help="JSONL 评估集路径")
    parser.add_argument(
        "--record-failures",
        action="store_true",
        help="把失败样本追加写入 eval/badcases.jsonl，默认不写入",
    )
    args = parser.parse_args()

    scenarios = load_scenarios(args.dataset)
    passed = 0
    failed = 0
    for scenario in scenarios:
        ok_count, failed_count = run_scenario(scenario, record_failures=args.record_failures)
        passed += ok_count
        failed += failed_count

    print(f"\n{passed} passed, {failed} failed")
    print(f"dataset={args.dataset}")
    print(f"checks={count_checks(scenarios)}")
    if args.record_failures:
        print(f"badcases={BADCASE_PATH}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
