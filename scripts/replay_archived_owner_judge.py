#!/usr/bin/env python3
"""对一个已归档的老板场景补做独立语义诊断，保留原始执行和门禁结论。"""

from __future__ import annotations

import argparse
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
import threading
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.agent_state import restore_user_message
from mahjong_agent_runtime.env import load_dotenv_defaults
from mahjong_agent_runtime.llm import OpenAICompatibleAgentClient
from mahjong_agent_runtime.stores.sqlite.serialization import tool_result_from_payload
import run_inquiry_grounding_eval as judge
import run_real_owner_chat_live_eval as owner
from reproduce_empty_llm_response import write_private_json
from run_evals import validate_owner_recovery_evidence

LEGACY_RUBRICS = {"responsibility_scope_v9_20261124":
    "2b6e8b6c77579aec787c929f0357c6e205df701daed407e007e6ee2e38daabde"}
SEMANTIC_CHECK = "reply_should_match_expected_meaning"
RECOVERY_CHECK = "action_contract_errors_should_be_safely_recovered"


class ArchivedSQLite(SQLiteAgentStore):
    """复用正式读取投影，但关闭数据库初始化、写入与按当前时间推进生命周期。"""

    def __post_init__(self) -> None:
        """只打开已落盘的归档，SQLite 不创建日志、锁文件或辅助文件。"""
        self.path = Path(self.path).resolve()
        self._connection = sqlite3.connect(self.path.as_uri() + "?mode=ro&immutable=1", uri=True)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()

    def _expire_stale_games(self, **_kwargs) -> None:
        """归档中的生命周期状态已经冻结，诊断只投影原运行写入的状态。"""


def _canonical_bytes(value: Any) -> bytes:
    """稳定指纹覆盖语义对象或保留的原始响应字符串，忽略报告对象键的打印次序。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_archived_artifact(run: Path, path: str) -> tuple[Path, bytes]:
    """所选原始轨迹和数据库必须来自同一归档运行目录。"""
    artifact = Path(path).resolve(strict=True)
    if not artifact.is_relative_to(run) or not artifact.is_file():
        raise ValueError("archived artifact must belong to the source evaluation run")
    return artifact, artifact.read_bytes()


def _valid_archived_verdict(raw: str, canonical_ids: list[str], *, legacy: bool) -> bool:
    """识别原协议中的有效正负判决；旧文本标识只用于离线资格核验。"""
    try:
        if legacy:
            data = json.loads(raw, object_pairs_hook=judge._unique_json_object)
            if not isinstance(data, dict) or not isinstance(data.get("criteria"), list):
                return False
            indexes = {name: index for index, name in enumerate(canonical_ids, start=1)}
            for criterion in data["criteria"]:
                if not isinstance(criterion, dict) or not isinstance(criterion.get("id"), str):
                    return False
                criterion["id"] = indexes.get(criterion["id"])
            raw = json.dumps(data, ensure_ascii=False)
        judge._parse_semantic_assessment(raw, canonical_ids)
    except ValueError:
        return False
    return True


def _check_archived_judgment(assessment: dict, expectation: dict, trace_id: str) -> dict:
    """核验旧请求投影与独立 HTTP 回执，任何有效原判决都会阻断诊断重判。"""
    version = assessment.get("judge_contract_version")
    legacy = version in LEGACY_RUBRICS
    rubric = assessment.get("rubric")
    correct_rubric = (isinstance(rubric, str) and sha256(rubric.encode()).hexdigest() == LEGACY_RUBRICS[version]
                      if legacy else version == judge.SEMANTIC_JUDGE_CONTRACT_VERSION
                      and rubric == judge.REPLY_SEMANTIC_JUDGE_RUBRIC)
    ids = judge._semantic_criteria_ids(expectation)
    current_projection = judge._project_judge_expectation(expectation)
    projected = current_projection
    if legacy:
        projected = {**current_projection, "checks": [
            {**row, "id": canonical} for row, canonical in zip(current_projection["checks"], ids, strict=True)
        ]}
    attempts = assessment.get("attempts")
    if (not correct_rubric or assessment.get("expectation") != expectation
            or assessment.get("judge_expectation") != projected or assessment.get("passed") is not False
            or assessment.get("criteria") != [] or assessment.get("transport_failures") or assessment.get("eval_inconclusive")
            or not isinstance(attempts, list) or len(attempts) not in {1, 2}
            or type(assessment.get("contract_repair_count")) is not int
            or assessment["contract_repair_count"] != len(attempts) - 1):
        raise ValueError("archived semantic request or malformed-response eligibility does not match")
    # 两版只改变审查消息及规则标识。归档规则先在上面验定，HTTP 证据沿用同一校验器。
    receipt_view = {**assessment, "judge_contract_version": judge.SEMANTIC_JUDGE_CONTRACT_VERSION,
                    "judge_expectation": current_projection}
    if (not judge.semantic_judge_config_matches(receipt_view, require_http=True)
            or not judge.semantic_judge_attempt_chain_matches(receipt_view, trace_id=trace_id)):
        raise ValueError("archived judge responses lack the original independent HTTP identity")
    for attempt in attempts:
        if (_valid_archived_verdict(attempt["raw_response"], ids, legacy=legacy)
                or not isinstance(attempt.get("error"), str)
                or not attempt["error"].startswith(("ValueError: ", "JSONDecodeError: "))):
            raise ValueError("archived judge already had a valid verdict or a different failure")
    if (assessment.get("raw_response") != attempts[-1]["raw_response"]
            or assessment.get("error") != attempts[-1]["error"]):
        raise ValueError("archived judge failure differs from its terminal response")
    return {"rubric": rubric, "evidence": assessment["judge_evidence"], "expectation": projected,
            "responses": [attempt["raw_response"] for attempt in attempts]}


def load_archived_case(report_path: Path, scenario_id: str, iteration: int) -> tuple[dict, list, dict]:
    """仅从原运行证明输入、回复、工具、状态和既有硬断言，从不运行场景或业务工具。"""
    source = report_path.resolve(strict=True)
    raw_report = source.read_bytes()
    payload = json.loads(raw_report, object_pairs_hook=judge._unique_json_object)
    scope = payload.get("live", payload) if isinstance(payload, dict) else None
    rows = scope.get("reports") if isinstance(scope, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("source report lacks owner case rows")
    cases = [row for row in rows if row.get("scenario_id") == scenario_id and row.get("iteration") == iteration]
    if len(cases) != 1 or type(cases[0]["iteration"]) is not int:
        raise ValueError("source report must identify exactly one owner case iteration")
    case = cases[0]
    declared = next((scene for scene in owner.live_eval_scenarios() if scene.scenario_id == scenario_id), None)
    identity = f"trace_owner_real_live_eval_{scenario_id}_{iteration:02d}"
    if (declared is None or case.get("status") != "failed" or case.get("trace_id") != identity
            or case.get("execution_id", f"{scenario_id}_run_{iteration}") != f"{scenario_id}_run_{iteration}"):
        raise ValueError("source must be an identified failed owner scenario")
    expectation = owner.reply_semantic_expectation(declared)
    assessment = case["semantic_assessment"]
    archived_request = _check_archived_judgment(assessment, expectation, f"{identity}_semantic_judge")
    raw_path, raw_trace = _read_archived_artifact(source.parent, case["raw_trace_path"])
    db_path, raw_db = _read_archived_artifact(source.parent, case["db_path"])
    wal = db_path.with_name(db_path.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("archived database has uncheckpointed WAL")
    trace = json.loads(raw_trace, object_pairs_hook=judge._unique_json_object)
    if (not isinstance(trace, dict) or (trace.get("trace_id"), trace.get("scenario_id"), trace.get("iteration"))
            != (identity, scenario_id, iteration)):
        raise ValueError("archived trace identity does not match selected scenario")
    raw_events = trace.get("events")
    if (not isinstance(raw_events, list) or not raw_events or any(not isinstance(event, dict)
            or event.get("trace_id") != identity or not isinstance(event.get("step"), str)
            or not isinstance(event.get("content"), dict) for event in raw_events)):
        raise ValueError("archived trace events are malformed")
    events = [SimpleNamespace(**event) for event in raw_events]
    message = case["input"]
    inputs = [event.content["message"] for event in events if event.step == "user_input"]
    outputs = [event.content for event in events if event.step == "final_output"]
    if (inputs != [message] or outputs != [{"reply": case["final_reply"], "objective_status": case["runtime_status"]}]
            or any(message.get(key) != declared.message.to_dict().get(key)
                   for key in ("conversation_id", "message_id", "sender_id", "sender_name", "text"))
            or message["conversation_id"] != case["conversation_id"]):
        raise ValueError("archived input or reply differs from original trace and source scenario")
    scenario = replace(declared, message=restore_user_message(message))
    tools = [tool_result_from_payload(event.content) for event in events if event.step == "tool_result"]
    result = SimpleNamespace(trace_id=identity, final_reply=case["final_reply"], status=case["runtime_status"],
                             tool_results=tools)
    store = ArchivedSQLite(db_path)
    with closing(store._connection):
        evidence = owner.reply_semantic_evidence(result, scenario, events, store)
        frozen_observed_at = assessment["evidence"]["persisted_state"]["follow_up_state"]["observed_at"]
        if datetime.fromisoformat(frozen_observed_at).tzinfo is None:
            raise ValueError("archived judge observation lacks time zone")
        evidence["persisted_state"]["follow_up_state"]["observed_at"] = frozen_observed_at
        validation = owner.validate_result(result, scenario, events, store, semantic_assessment=assessment,
            money_fact_check=case["money_fact_check"], query_reply_state_assessment=case.get("query_reply_state_assessment"))
    if (evidence != assessment["evidence"] or case["tool_result_summaries"] != owner.summarize_tool_results(tools)
            or case["tool_names"] != validation["tool_names"]
            or case["product_contract_revision_ids"] != expectation["product_contract_revision_ids"]):
        raise ValueError("archived judge evidence differs from original SQLite, trace or source contract")
    probe = deepcopy(case)
    hard = [check for check in case["checks"] if check["name"] not in {SEMANTIC_CHECK, RECOVERY_CHECK}]
    recomputed = [check for check in validation["checks"] if check["name"] not in {SEMANTIC_CHECK, RECOVERY_CHECK}]
    board_checks = (case.get("board_recovery") or {}).get("checks", [])
    if (hard != [*recomputed, *board_checks] or not hard or any(check["passed"] is not True for check in hard)
            or len({check["name"] for check in case["checks"]}) != len(case["checks"])
            or len(case["checks"]) != len(hard) + 2):
        raise ValueError("archived case has missing or failed business checks")
    probe["checks"] = hard
    probe["recovery_metrics"] = owner.assess_action_contract_recovery(
        events, runtime_status=case["runtime_status"], final_reply=case["final_reply"],
        business_checks_passed=True, semantic_passed=True)
    validate_owner_recovery_evidence(probe, {"report_path": str(source)},
        require_query_reply_state_check=declared.requires_query_reply_state_check, expected_query_request=message)
    if declared.require_board_recovery_delivery:
        owner.validate_board_recovery_evidence(case, source)
    fingerprints = {"source_report_path": str(source), "source_report_sha256": sha256(raw_report).hexdigest(),
        "raw_trace_path": str(raw_path), "raw_trace_sha256": sha256(raw_trace).hexdigest(),
        "db_path": str(db_path), "db_sha256": sha256(raw_db).hexdigest(),
        "archived_judge_request_identity_sha256": sha256(_canonical_bytes(
            {key: value for key, value in archived_request.items() if key != "responses"})).hexdigest(),
        "archived_judge_response_sha256": [sha256(raw.encode()).hexdigest() for raw in archived_request["responses"]]}
    return case, events, fingerprints


def replay_case(client, case: dict, events: list, fingerprints: dict) -> dict:
    """只调用共享独立判定器，用已有 owner 判定规则输出新的局部诊断。"""
    original = case["semantic_assessment"]
    assessment = judge.assess_reply_semantics(client, evidence=original["evidence"], expectation=original["expectation"],
        trace_id=f"{case['trace_id']}_semantic_diagnostic_{uuid4().hex}",
        judge_timeout_seconds=original["judge_transport"]["total_timeout_seconds"])
    if assessment.get("error") is None and not judge.semantic_judge_config_matches(assessment, require_http=True):
        assessment.update(passed=False, error="current independent judge response lacks verified HTTP evidence")
    updated = owner.owner_judge_disposition(deepcopy(case), events, assessment)
    return {"kind": "archived_owner_judge_diagnostic", "main_agent_executed": False, "full_gate": False,
            "status": updated["status"], "scenario_id": case["scenario_id"], "iteration": case["iteration"],
            "original_case_status": case["status"], "original_judge_contract_version": original["judge_contract_version"],
            "source": fingerprints, "final_reply": case["final_reply"], "runtime_status": case["runtime_status"],
            "semantic_assessment": assessment, "checks": updated["checks"], "recovery_metrics": updated["recovery_metrics"]}


def main(argv: list[str] | None = None) -> int:
    """命令行默认只在指定新文件写报告；预检模式完全离线。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.iteration < 1:
        parser.error("--iteration must be positive")
    try:
        case, events, fingerprints = load_archived_case(args.source_report, args.scenario, args.iteration)
        if args.verify_only:
            print(json.dumps(fingerprints, ensure_ascii=False, indent=2))
            return 0
        if args.output.exists() or not args.output.parent.is_dir():
            raise ValueError("diagnostic output must be a new file in an existing directory")
        load_dotenv_defaults(ROOT / ".env")
        client = OpenAICompatibleAgentClient.from_env()
        if client is None:
            raise ValueError("independent semantic judge model configuration is unavailable")
        diagnostic = replay_case(client, case, events, fingerprints)
        write_private_json(args.output, diagnostic)
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"archived owner judge diagnostic: {exc}\n")
    print(f"Archived judge diagnostic {diagnostic['status']}: {args.output}")
    return 0 if diagnostic["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
