#!/usr/bin/env python3
"""Replay anonymized real group messages through executable production components."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mahjong_agent_runtime import (  # noqa: E402
    AgentRuntime,
    InMemoryAgentStore,
    InMemoryTraceRecorder,
    OpenAICompatibleAgentClient,
    UserMessage,
)
from mahjong_agent_runtime.env import load_dotenv_defaults  # noqa: E402
from mahjong_agent_runtime.group_chat import (  # noqa: E402
    BoardItem,
    GroupMessage,
    GroupSessionClassifier,
    GroupSessionPipeline,
    MessageAccumulator,
    OwnerMessageParser,
    QuickFilter,
    SessionCrystallizer,
    SessionMerger,
    SessionRouter,
)


TZ = ZoneInfo("Asia/Shanghai")
BASE_TIME = datetime(2026, 7, 22, 17, 0, tzinfo=TZ)
DEFAULT_DATASET = ROOT / "eval" / "golden" / "real_group_chat_20260722.jsonl"
DEFAULT_REPORT = ROOT / "runtime_data" / "real_group_chat_flow_eval_report.json"


@dataclass(slots=True)
class CountingClient:
    """Count real model calls without changing the provider behavior."""

    delegate: Any
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(self, messages, *, trace_id: str, timeout_seconds: float) -> str:
        started = time.perf_counter()
        try:
            content = self.delegate.complete(
                messages,
                trace_id=trace_id,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            self.calls.append(
                {
                    "trace_id": trace_id,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        self.calls.append(
            {
                "trace_id": trace_id,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": None,
            }
        )
        return content


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_number(value: str | None) -> str | None:
    text = str(value or "").strip().removesuffix("块").removesuffix("元")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else str(number)


def normalize_semantic_value(value: Any, *, path: str) -> Any:
    """Normalize domain-equivalent labels before evaluating model output."""

    field_name = path.rsplit(".", 1)[-1]
    if field_name in {"stake", "stakes"}:
        return normalize_number(value)
    if field_name == "smoking":
        normalized = str(value or "").strip().lower()
        return {
            "no": "无烟",
            "no_smoking": "无烟",
            "non_smoking": "无烟",
            "无烟": "无烟",
            "yes": "有烟",
            "smoking": "有烟",
            "有烟": "有烟",
            "any": "不限",
            "不限": "不限",
            "都可": "不限",
        }.get(normalized, value)
    return value


def project_board_item(item: BoardItem) -> dict[str, Any]:
    return {
        "display_no": item.display_no,
        "game_type": item.game_type or None,
        "ruleset": item.ruleset,
        "participant_code": item.participant_code,
        "current_players": item.current_players,
        "missing_players": item.missing_players,
        "start_time": None if item.time == "人齐开" else item.time,
        "time_mode": "asap_when_full" if item.time == "人齐开" else "scheduled",
        "end_time": item.end_time,
        "duration_hours": item.duration_hours,
        "stake": normalize_number(item.stakes),
        "rule_code": item.rule_code,
        "smoking": item.smoking,
        "special_rules": [item.special_rules] if item.special_rules else [],
        "temporary_constraints": list(item.temporary_constraints),
        "status": item.status,
    }


def subset_errors(expected: Any, actual: Any, *, path: str = "actual") -> list[str]:
    """Compare an expectation as a partial contract, allowing extra actual fields."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected object, got {type(actual).__name__}"]
        errors: list[str] = []
        for key, value in expected.items():
            if key not in actual:
                errors.append(f"{path}.{key}: missing")
                continue
            errors.extend(subset_errors(value, actual[key], path=f"{path}.{key}"))
        return errors
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path}: expected list, got {type(actual).__name__}"]
        if len(actual) < len(expected):
            return [f"{path}: expected at least {len(expected)} items, got {len(actual)}"]
        errors: list[str] = []
        for index, value in enumerate(expected):
            errors.extend(subset_errors(value, actual[index], path=f"{path}[{index}]"))
        return errors
    normalized_expected = normalize_semantic_value(expected, path=path)
    normalized_actual = normalize_semantic_value(actual, path=path)
    return (
        []
        if normalized_expected == normalized_actual
        else [f"{path}: expected {expected!r}, got {actual!r}"]
    )


class RealGroupChatFlowEvaluator:
    """Drive each Gold case through parser, session classifier, and optional main Agent."""

    def __init__(self, *, llm_client: Any | None, quiet_seconds: float = 10.0) -> None:
        self.llm_client = llm_client
        self.quiet_seconds = quiet_seconds

    def evaluate(
        self,
        records: list[dict[str, Any]],
        *,
        dataset_path: str | Path | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        reports = [self.evaluate_case(record) for record in records]
        executed = [item for item in reports if item["status"] != "skipped"]
        passed = [item for item in executed if item["status"] == "passed"]
        return {
            "dataset": str(dataset_path or DEFAULT_DATASET),
            "generated_at": datetime.now(TZ).isoformat(),
            "mode": "live" if self.llm_client is not None else "deterministic",
            "quiet_seconds": self.quiet_seconds,
            "summary": {
                "total": len(reports),
                "executed": len(executed),
                "passed": len(passed),
                "failed": len(executed) - len(passed),
                "skipped": len(reports) - len(executed),
                "pass_rate": round(len(passed) / len(executed), 4) if executed else 0.0,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            },
            "cases": reports,
        }

    def evaluate_case(self, case: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        expected = dict(case["expected"])
        semantic_required = expected.get("model_call_required") is not False and case["case_type"] in {
            "member_query",
            "fragmented_input",
        }
        if semantic_required and self.llm_client is None:
            return {
                "case_id": case["id"],
                "case_type": case["case_type"],
                "status": "skipped",
                "reason": "requires live model",
                "elapsed_ms": 0.0,
            }

        store = InMemoryAgentStore()
        counting = CountingClient(self.llm_client) if self.llm_client is not None else None
        pipeline = self._pipeline(store, counting, case)
        ids: dict[str, str] = {}
        actions: list[str] = []
        outcomes = []
        before_snapshots: list[list[dict[str, Any]]] = []

        self._seed_precondition(case, pipeline, ids)
        self._seed_quoted_source(case, pipeline, ids)

        for index, raw in enumerate(case["messages"]):
            sent_at = BASE_TIME + timedelta(seconds=int(raw["offset_seconds"]))
            outcomes.extend(pipeline.flush_due(at=sent_at))
            message = self._message(case, raw, index=index, sent_at=sent_at, ids=ids)
            before = store.get_group_board_state(message.room_id)
            if before is not None:
                before_snapshots.append([project_board_item(item) for item in before.items])
            result = pipeline.accept(message, trace_id=f"trace_real_group_{case['id']}_{index}")
            actions.append(result.action)
            alias = raw.get("message_alias")
            if alias:
                ids[str(alias)] = message.message_id

        final_at = BASE_TIME + timedelta(
            seconds=max(int(item["offset_seconds"]) for item in case["messages"]) + self.quiet_seconds + 0.1
        )
        outcomes.extend(pipeline.flush_due(at=final_at))
        board = store.get_group_board_state(case["source"]["room_alias"])
        actual = {
            "pipeline_actions": actions,
            "board_items": [project_board_item(item) for item in board.items] if board else [],
            "session_count": len(pipeline.session_router.list_sessions(case["source"]["room_alias"])),
            "session_outcomes": [
                {
                    "action": item.action,
                    "session_id": item.session_id,
                    "classification": item.classification.to_dict() if item.classification else None,
                    "detail": item.detail,
                }
                for item in outcomes
            ],
            "model_call_count": len(counting.calls) if counting is not None else 0,
            "main_agent_invocation_count": 0,
            "main_agent": None,
        }

        if semantic_required and outcomes and counting is not None:
            actual["main_agent"] = self._run_main_agent(
                case=case,
                store=store,
                pipeline=pipeline,
                outcome=outcomes[-1],
                llm=counting,
            )
            actual["main_agent_invocation_count"] = 1
            actual["model_call_count"] = len(counting.calls)

        errors = self._check_case(
            case,
            expected=expected,
            actual=actual,
            before_snapshots=before_snapshots,
            pipeline=pipeline,
        )
        return {
            "case_id": case["id"],
            "case_type": case["case_type"],
            "status": "passed" if not errors else "failed",
            "expected": expected,
            "actual": actual,
            "errors": errors,
            "model_calls": list(counting.calls) if counting is not None else [],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def _pipeline(self, store, llm, case) -> GroupSessionPipeline:
        owners = {
            str(item["actor_alias"])
            for item in case["messages"]
            if item["role"] == "operator"
        }
        classifier = GroupSessionClassifier(llm, timeout_seconds=30) if llm is not None else _NoModelClassifier()
        router = SessionRouter(clock=lambda: BASE_TIME)
        return GroupSessionPipeline(
            store=store,
            owner_parser=OwnerMessageParser(owner_external_ids=owners | {"seed_owner"}),
            quick_filter=QuickFilter(),
            accumulator=MessageAccumulator(
                quiet_seconds=self.quiet_seconds,
                continuation_seconds=120,
            ),
            session_router=router,
            session_merger=SessionMerger(router),
            crystallizer=SessionCrystallizer(),
            classifier=classifier,
        )

    @staticmethod
    def _seed_precondition(case, pipeline, ids: dict[str, str]) -> None:
        items = (case.get("precondition") or {}).get("board_items") or []
        if not items:
            return
        text = "\n".join(str(item["text"]) for item in items)
        message = GroupMessage(
            room_id=case["source"]["room_alias"],
            conversation_id=f"wechaty:room:{case['source']['room_alias']}",
            sender_external_id="seed_owner",
            sender_name="seed_owner",
            text=text,
            message_id=f"seed:{case['id']}",
            sent_at=BASE_TIME - timedelta(seconds=1),
            metadata={"is_owner": True},
        )
        pipeline.accept(message, trace_id=f"trace_seed_{case['id']}")
        ids["precondition"] = message.message_id

    @staticmethod
    def _seed_quoted_source(case, pipeline, ids: dict[str, str]) -> None:
        quoted = next((item for item in case["messages"] if item.get("quoted_text")), None)
        if quoted is None:
            return
        quoted_id = f"quoted:{case['id']}"
        message = GroupMessage(
            room_id=case["source"]["room_alias"],
            conversation_id=f"wechaty:room:{case['source']['room_alias']}",
            sender_external_id="seed_owner",
            sender_name="seed_owner",
            text=str(quoted["quoted_text"]),
            message_id=quoted_id,
            sent_at=BASE_TIME - timedelta(seconds=1),
            metadata={"is_owner": True},
        )
        pipeline.accept(message, trace_id=f"trace_quote_seed_{case['id']}")
        ids["quoted_source"] = quoted_id

    @staticmethod
    def _message(case, raw, *, index: int, sent_at: datetime, ids: dict[str, str]) -> GroupMessage:
        message_id = f"real:{case['id']}:{index}"
        metadata: dict[str, Any] = {"content_type": raw.get("content_type", "text")}
        if raw["role"] == "operator":
            metadata["is_owner"] = True
        if raw.get("quoted_text"):
            metadata["quoted_text"] = raw["quoted_text"]
        target_alias = raw.get("target_message_alias")
        if target_alias:
            metadata["target_message_id"] = ids.get(str(target_alias), str(target_alias))
        quoted_message_id = ids.get("quoted_source") if raw.get("quoted_text") else None
        return GroupMessage(
            room_id=case["source"]["room_alias"],
            conversation_id=f"wechaty:room:{case['source']['room_alias']}",
            sender_external_id=str(raw.get("actor_alias") or raw["role"]),
            sender_name=str(raw.get("actor_alias") or raw["role"]),
            text=str(raw["text"]),
            message_id=message_id,
            sent_at=sent_at,
            quoted_message_id=quoted_message_id,
            metadata=metadata,
        )

    @staticmethod
    def _run_main_agent(*, case, store, pipeline, outcome, llm) -> dict[str, Any]:
        session = next(
            item for item in pipeline.session_router.list_sessions(case["source"]["room_alias"])
            if item.id == outcome.session_id
        )
        source_message = session.messages[-1]
        trace = InMemoryTraceRecorder()
        runtime = AgentRuntime(
            llm_client=llm,
            store=store,
            trace_recorder=trace,
            customer_visible_text_generation_enabled=False,
            reply_self_review_enabled=False,
        )
        result = runtime.handle_user_message(
            UserMessage(
                conversation_id=f"group:{source_message.room_id}:customer:{source_message.sender_external_id}",
                sender_id=source_message.sender_external_id,
                sender_name=source_message.sender_name,
                text=source_message.text,
                message_id=f"agent:{case['id']}",
                sent_at=source_message.sent_at,
                metadata={
                    "source": "group_session_eval",
                    "room_id": source_message.room_id,
                    "group_session_classification": outcome.classification.to_dict(),
                    "group_board_state": (
                        store.get_group_board_state(source_message.room_id).to_dict()
                        if store.get_group_board_state(source_message.room_id)
                        else None
                    ),
                    "reply_constraints": {"max_length": 20, "no_private_info": True},
                },
            ),
            trace_id=f"trace_main_{case['id']}",
        )
        return {
            "status": result.status,
            "final_reply": result.final_reply,
            "tool_calls": [item.name for item in result.tool_results if item.called],
            "tool_results": [item.to_dict() for item in result.tool_results],
            "trace_steps": [item.step for item in trace.events],
        }

    @staticmethod
    def _check_case(case, *, expected, actual, before_snapshots, pipeline) -> list[str]:
        errors: list[str] = []
        case_type = case["case_type"]
        items = actual["board_items"]
        if "items" in expected:
            if case_type == "owner_board_increment":
                by_number = {item["display_no"]: item for item in items}
                for expected_item in expected["items"]:
                    display_no = expected_item.get("display_no")
                    errors.extend(
                        subset_errors(
                            expected_item,
                            by_number.get(display_no),
                            path=f"board_items[display_no={display_no}]",
                        )
                    )
            else:
                errors.extend(subset_errors(expected["items"], items, path="board_items"))
        if case_type == "board_state_diff":
            if len(items) != 1 and expected.get("must_not_create_duplicate_game"):
                errors.append(f"board_items: expected one logical item, got {len(items)}")
            if before_snapshots:
                errors.extend(subset_errors(expected["before"], before_snapshots[-1][0], path="before"))
            if items:
                errors.extend(subset_errors(expected["after"], items[0], path="after"))
        if case_type == "quoted_state_update":
            if len(items) != 1:
                errors.append(f"quoted update expected one item, got {len(items)}")
            elif items[0]["status"] != "full" or items[0]["missing_players"] != 0:
                errors.append("quoted update did not mark the target game full")
            else:
                errors.extend(subset_errors(expected["target_requirement"], items[0], path="target"))
        if case_type == "quoted_requirement_update":
            if len(items) != 1:
                errors.append(f"quoted requirement update expected one logical item, got {len(items)}")
            elif expected.get("must_not_create_duplicate_game"):
                errors.extend(subset_errors(expected["after"], items[0], path="after"))
        if case_type == "quick_filter":
            if actual["model_call_count"] != expected["model_call_count"]:
                errors.append("noise reached the model")
            if actual["session_count"] != expected["session_count_delta"]:
                errors.append("noise created a session")
            if any(action != "filtered" for action in actual["pipeline_actions"]):
                errors.append(f"noise actions were {actual['pipeline_actions']!r}")
        if case_type == "message_revoke":
            sessions = pipeline.session_router.list_sessions(case["source"]["room_alias"])
            leaked = any("人" in message.text for session in sessions for message in session.messages)
            if leaked:
                errors.append("revoked text remains in session state")
        if case_type in {"member_query", "fragmented_input"}:
            classifications = [
                item["classification"] for item in actual["session_outcomes"] if item["classification"] is not None
            ]
            if not classifications:
                errors.append("semantic message produced no session classification")
                return errors
            final = classifications[-1]
            if final["intent"] != expected["intent"]:
                errors.append(f"intent: expected {expected['intent']!r}, got {final['intent']!r}")
            features = final["extracted_features"]
            requirement = expected.get("search_requirement") or {}
            classifier_requirement = {
                ("stakes" if key == "stake" else key): value
                for key, value in requirement.items()
            }
            errors.extend(
                subset_errors(
                    classifier_requirement,
                    features,
                    path="classification.extracted_features",
                )
            )
            for key in ("current_players", "missing_players", "urgency"):
                if key in expected and features.get(key) != expected[key]:
                    errors.append(f"{key}: expected {expected[key]!r}, got {features.get(key)!r}")
            for key in expected.get("must_not_invent_fields") or []:
                feature_key = "stakes" if key == "stake" else key
                if features.get(feature_key) not in (None, "", []):
                    errors.append(f"model invented {key}={features.get(feature_key)!r}")
            if expected.get("must_not_invent_smoking_or_time"):
                for key in ("smoking", "time"):
                    if features.get(key) not in (None, "", []):
                        errors.append(f"model invented {key}={features.get(key)!r}")
            for key in expected.get("missing_fields") or []:
                feature_key = "stakes" if key == "stake" else key
                if features.get(feature_key) not in (None, "", []):
                    errors.append(f"expected {key} to remain missing, got {features.get(feature_key)!r}")
            if expected.get("same_session") and actual["session_count"] != 1:
                errors.append(f"expected one session, got {actual['session_count']}")
            if expected.get("process_after_aggregation") and len(classifications) != 1:
                errors.append(f"expected one post-aggregation classification, got {len(classifications)}")
            if "main_agent_invocation_count" in expected and (
                actual["main_agent_invocation_count"] != expected["main_agent_invocation_count"]
            ):
                errors.append(
                    "main Agent invocation count: "
                    f"expected {expected['main_agent_invocation_count']}, "
                    f"got {actual['main_agent_invocation_count']}"
                )
            if expected.get("preferred_channel_action") and (
                final["channel_action"] != expected["preferred_channel_action"]
            ):
                errors.append(
                    f"channel_action: expected {expected['preferred_channel_action']!r}, "
                    f"got {final['channel_action']!r}"
                )
            main = actual.get("main_agent") or {}
            tool_calls = main.get("tool_calls") or []
            for name in expected.get("required_tool_calls") or []:
                if name not in tool_calls:
                    errors.append(f"main Agent did not call required tool {name}")
            for name in expected.get("forbidden_tool_calls") or []:
                if name in tool_calls:
                    errors.append(f"main Agent called forbidden tool {name}")
        return errors


class _NoModelClassifier:
    def classify(self, **_kwargs):
        raise AssertionError("deterministic case unexpectedly reached the model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--live", action="store_true", help="run semantic cases through the configured model and main Agent")
    parser.add_argument("--strict", action="store_true", help="exit non-zero when any executed case fails")
    parser.add_argument("--quiet-seconds", type=float, default=10.0)
    parser.add_argument("--env-file", type=Path, default=Path.home() / ".local/share/mahjong-ops-agent/.env")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    llm = None
    if args.live:
        load_dotenv_defaults(args.env_file.expanduser())
        llm = OpenAICompatibleAgentClient.from_env()
        if llm is None:
            raise RuntimeError("live mode requires MAHJONG_LLM_MODEL and provider credentials")
    evaluator = RealGroupChatFlowEvaluator(llm_client=llm, quiet_seconds=args.quiet_seconds)
    report = evaluator.evaluate(read_jsonl(args.dataset), dataset_path=args.dataset)
    report_path = args.report_path.expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if args.strict and report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
