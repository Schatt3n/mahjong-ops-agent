#!/usr/bin/env python3
"""统一编排项目评测命令并输出失败步骤。"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mahjong_agent_runtime.env import load_dotenv_defaults  # noqa: E402


LIVE_SCOPES = (
    "real_owner_all_scenarios_with_reply_semantics",
    "real_group_all_scenarios",
    "real_concurrency_all_scenarios",
    "inquiry_grounding_six_cases",
    "privacy_isolation_all_cases",
)
OFFLINE_SEMANTIC_NOTICE = "未运行真实模型语义评测，不能称全量业务通过"
DEFAULT_PYTEST_GROUPS = (
    ("tests/test_agent_runtime.py",),
    ("tests/test_real_owner_chat_golden.py",),
    ("tests/test_real_group_chat_dataset.py",),
    ("tests/test_context_summary.py",),
    ("tests/test_context_summary_quality.py",),
    ("tests/test_agent_runtime_eval.py",),
    ("tests/test_agent_app.py",),
    ("tests/test_input_aggregation.py",),
    ("tests/test_progress_monitor.py",),
    ("tests/test_future_game_recruitment.py",),
    ("tests/test_action_contract_repairs.py",),
    ("tests/test_agent_runtime_package.py",),
    ("tests/test_operational_waiting.py", "tests/test_manual_simulation.py"),
    ("tests/test_cross_conversation_participation.py",),
    ("tests/test_operational_chat_scenarios.py",),
    ("tests/test_inquiry_evidence_scope.py",),
)


def run_command(args: list[str], *, log_path: Path | None = None) -> None:
    """顺序执行已有评测命令，保留失败退出码和独立日志，不把未执行阶段算作通过。"""
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = f"{src_path}:{existing_pythonpath}" if existing_pythonpath else src_path
    print(f"\n$ {' '.join(args)}", flush=True)
    if log_path is None:
        subprocess.run(args, cwd=ROOT, env=env, check=True)
    else:
        with log_path.open("w", encoding="utf-8") as output:
            subprocess.run(args, cwd=ROOT, env=env, check=True, stdout=output, stderr=subprocess.STDOUT)


def parse_args() -> argparse.Namespace:
    """解析args，并把非结构化输入转换为稳定合同。"""
    parser = argparse.ArgumentParser(description="Run current Agent Runtime evals.")
    parser.add_argument(
        "--full", action="store_true",
        help="run the full existing offline matrix, all pytest tests and every live owner/group/concurrency/inquiry/privacy suite",
    )
    parser.add_argument(
        "--live-real-owner",
        action="store_true",
        help="also run real owner chat live LLM eval; requires MAHJONG_LLM_MODEL and an API key",
    )
    parser.add_argument(
        "--live-concurrency",
        action="store_true",
        help="also run the real-DeepSeek concurrent eval; requires MAHJONG_LLM_MODEL and an API key",
    )
    parser.add_argument(
        "--live-real-group",
        action="store_true",
        help="also replay semantic real-group cases through the configured model and main Agent",
    )
    return parser.parse_args()


def create_run_directory() -> Path:
    """用 UTC 微秒时间戳和随机后缀隔离每次证据，不能覆盖以前的报告或运行库。"""
    parent = ROOT / "runtime_data" / "eval_runs"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path(tempfile.mkdtemp(prefix=f"{stamp}_", dir=parent))


def build_eval_plan(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    """复用现有运行器构建明确范围；完整门禁不添加场景筛选或跳过审查的参数。"""
    plan: list[dict[str, Any]] = []

    def add(scope: str, command: list[str], **artifacts: Any) -> None:
        """记录待执行范围和产物，状态在真正执行后才变为通过或失败。"""
        plan.append({"scope": scope, "command": command, "status": "pending", **artifacts})

    for scope, script in (
        ("runtime_boundary", "scripts/verify_agent_runtime_boundary.py"),
        ("customer_visible_contract", "scripts/verify_customer_visible_contract.py"),
        ("badcase_regression_coverage", "scripts/check_badcase_regression_coverage.py"),
        ("owner_dataset_validation", "scripts/validate_real_owner_chat_golden.py"),
        ("group_dataset_validation", "scripts/validate_real_group_chat_dataset.py"),
    ):
        add(scope, [sys.executable, script])
    group_datasets = sorted((ROOT / "eval" / "golden").glob("real_group_chat_*.jsonl")) if args.full else [None]
    if not group_datasets:
        raise ValueError("full gate requires the existing dated group golden datasets")
    for dataset in group_datasets:
        suffix = f"_{dataset.stem.removeprefix('real_group_chat_')}" if dataset is not None else ""
        group_report = run_dir / f"real_group_chat_flow_eval_deterministic{suffix}.json"
        add(f"group_flow_deterministic{suffix}", [
            sys.executable,
            "scripts/run_real_group_chat_flow_eval.py",
            *(["--dataset", str(dataset)] if dataset is not None else []),
            "--strict",
            "--report-path",
            str(group_report),
        ], report_path=str(group_report), dataset=str(dataset) if dataset is not None else "default_20260722")
    add("runtime_deterministic", [sys.executable, "scripts/run_agent_runtime_eval.py"])
    deterministic_report = run_dir / "concurrency_eval_deterministic_report.json"
    add("concurrency_deterministic", [
            sys.executable,
            "scripts/run_concurrency_eval.py",
            "--mode",
            "deterministic",
            "--operations",
            "40",
            "--workers",
            "8",
            "--work-dir",
            str(run_dir / "concurrency_deterministic"),
            "--strict",
            "--report-path",
            str(deterministic_report),
        ], report_path=str(deterministic_report))
    if args.full:
        add("all_pytest", [sys.executable, "-m", "pytest", "-q"])
        add("chinese_docstrings", [sys.executable, "scripts/check_chinese_docstrings.py"])
        add("git_diff_check", ["git", "diff", "--check"])
    else:
        for index, group in enumerate(DEFAULT_PYTEST_GROUPS, start=1):
            add(f"offline_pytest_group_{index:02d}", [sys.executable, "-m", "pytest", "-q", *group])
    if args.full or args.live_real_owner:
        report = run_dir / "real_owner_chat_live_eval_report.json"
        add(LIVE_SCOPES[0], [
                sys.executable,
                "scripts/run_real_owner_chat_live_eval.py",
                "--strict",
                "--db-path",
                str(run_dir / "owner" / "owner_live.sqlite3"),
                "--report-path",
                str(report),
            ], report_path=str(report))
    if args.full or args.live_real_group:
        for dataset in group_datasets:
            suffix = f"_{dataset.stem.removeprefix('real_group_chat_')}" if dataset is not None else ""
            report = run_dir / f"real_group_chat_flow_eval_live{suffix}.json"
            add(f"{LIVE_SCOPES[1]}{suffix}", [
                sys.executable,
                "scripts/run_real_group_chat_flow_eval.py",
                *(["--dataset", str(dataset)] if dataset is not None else []),
                *(["--env-file", str(ROOT / ".env")] if args.full else []),
                "--live",
                "--strict",
                "--report-path",
                str(report),
            ], report_path=str(report), scope_family=LIVE_SCOPES[1],
                dataset=str(dataset) if dataset is not None else "default_20260722")
    if args.full or args.live_concurrency:
        report = run_dir / "concurrency_eval_live_report.json"
        add(LIVE_SCOPES[2], [
                sys.executable,
                "scripts/run_concurrency_eval.py",
                "--mode",
                "live",
                "--live-workers",
                "4",
                "--live-repeats",
                "2",
                "--work-dir",
                str(run_dir / "concurrency_live"),
                "--strict",
                "--report-path",
                str(report),
            ], report_path=str(report))
    if args.full:
        inquiry_run_id = f"{run_dir.name}_inquiry"
        source_report = ROOT / "runtime_data" / "periodic_chat_simulation" / "runs" / inquiry_run_id / "report.json"
        add(LIVE_SCOPES[3], [sys.executable, "scripts/run_inquiry_grounding_eval.py", "--run-id", inquiry_run_id],
            source_report_path=str(source_report), report_path=str(run_dir / "inquiry_grounding_report.json"))
        report = run_dir / "privacy_isolation_live_eval_report.json"
        add(LIVE_SCOPES[4], [
            sys.executable, "scripts/run_privacy_isolation_live_eval.py", "--strict",
            "--db-path", str(run_dir / "privacy" / "privacy_live.sqlite3"),
            "--report-path", str(report),
        ], report_path=str(report))
    return plan


def save_gate_report(report_path: Path, report: dict[str, Any]) -> None:
    """保存本次执行范围与阶段状态；未选择或未运行的范围不能被计为通过。"""
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def archive_external_scope_report(phase: dict[str, Any], *, required: bool) -> None:
    """复制既有咨询运行器的报告到本次目录，同时保留原有模拟证据不移动、不覆盖。"""
    if not phase.get("source_report_path"):
        return
    source = Path(phase["source_report_path"])
    if source.exists():
        shutil.copy2(source, Path(phase["report_path"]))
    elif required:
        raise FileNotFoundError(f"scope report was not produced: {source}")


def load_owner_semantic_contract() -> dict[str, Any]:
    """复用真实运行器目录、黄金关联和人工语义期望，不信子报告自报的案例总数。"""
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from run_real_owner_chat_live_eval import golden_reply_coverage, live_eval_scenarios, reply_semantic_expectation
    from run_inquiry_grounding_eval import _semantic_criteria_ids

    scenarios = live_eval_scenarios()
    ids = [scenario.scenario_id for scenario in scenarios]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("owner source catalog must contain unique non-empty scenarios")
    expectations = {scenario.scenario_id: reply_semantic_expectation(scenario) for scenario in scenarios}
    coverage = golden_reply_coverage(set(ids), set(ids))
    golden_ids = coverage["executed_case_ids"]
    if (not golden_ids or len(golden_ids) != coverage["total_cases"]
            or len(set(golden_ids)) != len(golden_ids)
            or coverage["unmapped_case_ids"] or coverage["partially_executed_case_ids"]):
        raise ValueError("owner source catalog does not map every golden case completely")
    return {
        "scenario_ids": sorted(ids), "expectations": expectations,
        "criterion_ids": {scenario_id: _semantic_criteria_ids(expected) for scenario_id, expected in expectations.items()},
        "golden_case_ids": sorted(golden_ids), "timeline_assets": coverage["timeline_assets"],
    }


def validate_owner_scope_coverage(payload: dict[str, Any], phase: dict[str, Any]) -> None:
    """对照源码集合核验每个回放及逐项判定，重复或伪造条目不能充当漏跑的场景。"""
    if payload.get("execution_scope") != "all_live_scenarios" or payload.get("semantic_judge_ran") is not True:
        raise ValueError("owner scope must run every live scenario with independent reply semantics")
    contract = load_owner_semantic_contract()
    from run_inquiry_grounding_eval import _parse_semantic_assessment
    from run_real_owner_chat_live_eval import summarize_action_contract_recovery

    scenario_ids = contract["scenario_ids"]
    definitions, repeat = payload.get("scenario_definition_count"), payload.get("repeat")
    if (type(definitions) is not int or definitions != len(scenario_ids)
            or type(repeat) is not int or not 1 <= repeat <= 20
            or payload["scenario_count"] != definitions * repeat):
        raise ValueError("owner scenario counts do not match the source catalog and repetitions")
    expected_runs = {(scenario_id, iteration) for scenario_id in scenario_ids for iteration in range(1, repeat + 1)}
    observed_runs: set[tuple[str, int]] = set()
    for item in payload["reports"]:
        scenario_id, iteration = item.get("scenario_id"), item.get("iteration")
        if not isinstance(scenario_id, str) or type(iteration) is not int:
            raise ValueError("owner report requires its real scenario_id and integer iteration")
        identity = (scenario_id, iteration)
        if identity not in expected_runs or identity in observed_runs:
            raise ValueError("owner report contains an unknown or duplicate scenario iteration")
        observed_runs.add(identity)
        assessment = item.get("semantic_assessment")
        if (not isinstance(assessment, dict) or assessment.get("passed") is not True
                or "error" not in assessment or assessment["error"] not in (None, "")):
            raise ValueError("owner per-case semantic assessment is absent, failed or invalid")
        if assessment.get("expectation") != contract["expectations"][scenario_id]:
            raise ValueError("owner judge expectation differs from the source human expectation")
        # 复用判定器的严格 JSON/ID/布尔合同，不另写一套更宽松的汇总逻辑。
        expected_ids = contract["criterion_ids"][scenario_id]
        criteria = assessment.get("criteria")
        if not isinstance(criteria, list) or any(
            not isinstance(criterion, dict) or set(criterion) != {"id", "passed", "reason"}
            or type(criterion["passed"]) is not bool for criterion in criteria
        ):
            raise ValueError("owner outward criteria require boolean passed and exact criterion fields")
        # outward 是后端派生的通过方向；先验类型，再反向投影为原始违规检测合同。
        # 不能直接 not 一个字符串或数字，否则错误类型会被悄悄转换成合法布尔值。
        structured = _parse_semantic_assessment(json.dumps({
            "reason": assessment.get("reason"), "criteria": [
                {"id": criterion["id"], "violation_detected": not criterion["passed"], "reason": criterion["reason"]}
                for criterion in criteria
            ],
        }, ensure_ascii=False), expected_ids)
        raw = assessment.get("raw_response")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("owner semantic assessment requires its original judge JSON")
        original = _parse_semantic_assessment(raw, expected_ids)
        if structured["passed"] is not True or original != structured:
            raise ValueError("owner judge criteria failed or contradict their original JSON")
        validate_owner_recovery_evidence(item, phase)
    if observed_runs != expected_runs:
        raise ValueError("owner report did not execute every source scenario iteration")
    recovery_summary = summarize_action_contract_recovery(payload["reports"])
    if (recovery_summary.get("quality_status") != "passed"
            or json.dumps(payload.get("recovery_summary"), sort_keys=True) != json.dumps(recovery_summary, sort_keys=True)):
        raise ValueError("owner recovery summary contradicts its per-case evidence")
    phase["recovery_summary"] = recovery_summary

    coverage = payload.get("golden_reply_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("owner golden reply coverage is absent")
    golden_ids = contract["golden_case_ids"]
    if type(coverage.get("total_cases")) is not int or coverage["total_cases"] != len(golden_ids):
        raise ValueError("owner golden case count differs from the source dataset")
    for key in ("executed_case_ids", "passed_case_ids"):
        reported_ids = coverage.get(key)
        if (not isinstance(reported_ids, list) or len(reported_ids) != len(golden_ids)
                or any(not isinstance(case_id, str) for case_id in reported_ids)
                or len(set(reported_ids)) != len(golden_ids) or set(reported_ids) != set(golden_ids)):
            raise ValueError(f"owner golden {key} does not match the source case identities")
    if coverage.get("partially_executed_case_ids") != [] or coverage.get("unmapped_case_ids") != []:
        raise ValueError("owner golden reply cases are partial or unmapped")
    if coverage.get("timeline_assets") != contract["timeline_assets"]:
        raise ValueError("owner timeline limitations differ from the source coverage declaration")
    phase["case_coverage"].update(
        semantic_judged=len(observed_runs), scenario_definition_count=len(scenario_ids), repeat=repeat,
        expected_scenario_ids=scenario_ids, executed_scenario_ids=sorted({identity[0] for identity in observed_runs}),
        golden_reply_total=len(golden_ids), golden_reply_passed=len(golden_ids), golden_case_ids=golden_ids,
        timeline_assets=contract["timeline_assets"],
    )


def validate_owner_recovery_evidence(item: dict[str, Any], phase: dict[str, Any]) -> None:
    """从本次原始 trace 重算纠错计量，不能把失败改名为 recovered 后放行。

    主模型格式错误只有在后端确实拒绝、反馈被后续合法动作处理，且最终业务和
    语义均通过时才允许单列。原始文件必须属于本次证据目录，报告计数不能自证。
    """
    from run_real_owner_chat_live_eval import assess_action_contract_recovery

    trace_path = item.get("raw_trace_path")
    if not isinstance(trace_path, str) or not trace_path:
        raise ValueError("owner recovery evidence requires its original trace file")
    source = Path(trace_path).resolve()
    run_dir = Path(phase["report_path"]).resolve().parent
    if not source.is_relative_to(run_dir) or not source.is_file():
        raise ValueError("owner original trace must belong to this evaluation run")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if (not isinstance(raw, dict) or raw.get("trace_id") != item.get("trace_id")
            or raw.get("scenario_id") != item.get("scenario_id") or raw.get("iteration") != item.get("iteration")):
        raise ValueError("owner original trace identity differs from the case report")
    events = raw.get("events")
    if (not isinstance(events, list) or not events or any(
            not isinstance(event, dict) or event.get("trace_id") != raw["trace_id"]
            or not isinstance(event.get("step"), str) or not isinstance(event.get("content"), dict)
            for event in events)):
        raise ValueError("owner original trace events are missing or malformed")
    if item.get("trace_steps") != [event["step"] for event in events]:
        raise ValueError("owner report omitted or changed original trace steps")
    checks = item.get("checks")
    if (not isinstance(checks, list) or not checks
            or any(not isinstance(check, dict) or check.get("passed") is not True for check in checks)):
        raise ValueError("owner final business checks failed or are absent")
    recovered = assess_action_contract_recovery(
        [SimpleNamespace(**event) for event in events],
        runtime_status=item.get("runtime_status"), final_reply=item.get("final_reply"),
        business_checks_passed=True, semantic_passed=True,
    )
    if (json.dumps(recovered, sort_keys=True) != json.dumps(item.get("recovery_metrics"), sort_keys=True)
            or recovered.get("quality_status") != "passed"):
        raise ValueError("owner recovery metrics failed or contradict the original trace")


def _controlled_review_database_state(connection: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """只读实际运行库并投影复查身份与状态，不能用任务行数冒充有效唤醒安排。"""
    tables = {"game_count": "runtime_games", "invite_count": "runtime_invite_drafts",
              "outbound_count": "runtime_outbound_message_drafts", "scheduled_count": "runtime_scheduled_agent_tasks",
              "waiting_count": "waiting_demands"}
    state = {key: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for key, table in tables.items()}
    tasks: list[dict[str, Any]] = []
    for row in connection.execute("SELECT task_id, task_type, aggregate_id, conversation_id, status, due_at, payload "
                                  "FROM runtime_scheduled_agent_tasks ORDER BY task_id"):
        task = json.loads(row[-1])
        if any(task.get(key) != row[index] for index, key in enumerate(
                ("task_id", "task_type", "aggregate_id", "conversation_id", "status", "due_at"))):
            raise ValueError("controlled review task columns contradict their persisted payload")
        tasks.append(task)
    state["scheduled_tasks"] = [{**{key: task.get(key) for key in (
        "task_id", "task_type", "status", "conversation_id", "subject_id", "aggregate_type", "aggregate_id", "due_at")},
        "payload": {key: value for key, value in task.get("payload", {}).items() if key in {
            "goal", "waiting_for", "task_context_id", "source_message_id", "expires_at", "review_budget", "review_count", "closed_reason",
        }}} for task in tasks]
    return state, tasks


def _validate_controlled_follow_up_evidence(source: dict[str, Any], item: dict[str, Any], events: list[dict[str, Any]],
                                           request: dict[str, Any], connection: Any, tasks: list[dict[str, Any]]) -> None:
    """逐项核对已声明夹具的真实工具执行、牌局和复查任务，不解析客户自然语言。"""
    from mahjong_agent_runtime.domains.context_builders.tool_results import compact_tool_result_dict

    fixture = source["fixture"]
    names = ["create_game", "search_customers"]
    if fixture in {"open_game_follow_up", "open_game_cancelled_follow_up"}:
        names.append("schedule_follow_up")
    if fixture == "open_game_cancelled_follow_up":
        names.append("cancel_follow_up")
    results = item.get("prerequisite_results")
    originals = [event["content"] for event in events if event["step"] == "tool_result"
                 and event["content"].get("name") != "customer_visible_content_review"]
    calls = [event["content"].get("call", {}) for event in events if event["step"] == "tool_called"]
    completions = [event["content"] for event in events if event["step"] == "tool_gateway_completed"]
    if (not isinstance(results, list) or results != originals or [result.get("name") for result in results] != names
            or [call.get("name") for call in calls] != names or len(completions) != len(names)):
        raise ValueError("controlled follow-up prerequisite calls or original results are incomplete")
    prompt_index = next(index for index, event in enumerate(events) if event["step"] == "customer_visible_content_review_prompt")
    if any(index >= prompt_index for index, event in enumerate(events) if event["step"] in {"tool_called", "tool_gateway_completed"}
           or (event["step"] == "tool_result" and event["content"].get("name") != "customer_visible_content_review")):
        raise ValueError("controlled follow-up business preparation did not finish before the model judgement")
    for result, completion in zip(results, completions):
        if (result.get("called") is not True or result.get("allowed") is not True or result.get("error")
                or not result.get("idempotency_key") or result.get("deduplicated") is not False
                or completion.get("tool_name") != result["name"] or completion.get("called") is not True
                or completion.get("allowed") is not True or completion.get("error")
                or completion.get("idempotency_key") != result["idempotency_key"]
                or compact_tool_result_dict(result) not in request.get("turn_tool_evidence", [])):
            raise ValueError("controlled follow-up is missing a real successful prerequisite or its model evidence")
    if item.get("tool_result") != results[1]:
        raise ValueError("controlled follow-up candidate search differs from the original prerequisite")
    created, searched = results[0]["result"], results[1]["result"]
    game = created.get("game", {})
    message = request.get("current_message", {})
    row = connection.execute("SELECT conversation_id, status, payload FROM runtime_games WHERE game_id=?", (game.get("game_id"),)).fetchone()
    if (row is None or row[0] != message.get("conversation_id") or row[1] != "forming"
            or game.get("organizer_id") != message.get("sender_id") or game.get("recruitment_status") != "open"
            or game.get("remaining_seats") != 3 or game.get("requirement") != json.loads(row[2]).get("requirement")
            or any(game.get("requirement", {}).get(key) != value for key, value in source["requirement"].items())
            or searched.get("bound_game_id") != game.get("game_id") or searched.get("candidates") != []
            or calls[0].get("arguments", {}).get("requirement") != source["requirement"]
            or calls[1].get("arguments", {}).get("game_id") != game.get("game_id")
            or created.get("recruitment_policy", {}).get("scheduled_task") is not None):
        raise ValueError("controlled follow-up lacks its persisted open game and real exhausted candidate search")
    if (connection.execute("SELECT COUNT(*) FROM runtime_game_participants").fetchone()[0] != 1
            or connection.execute("SELECT COUNT(*) FROM runtime_game_participants WHERE game_id=? AND customer_id=? "
                                  "AND seat_count=1 AND status='joined'", (game["game_id"], message.get("sender_id"))).fetchone()[0] != 1):
        raise ValueError("controlled follow-up game requester occupancy is missing or changed")
    contexts = [event["content"] for event in events if event["step"] == "context_built"]
    context_id = contexts[-1].get("task_context_window", {}).get("task_context_id") if contexts else None
    active = [task for task in tasks if task.get("status") in {"pending", "processing"}]
    expected_task_count = 0 if fixture == "open_game_empty_candidates" else (2 if fixture == "open_game_cancelled_follow_up" else 1)
    if len(tasks) != expected_task_count or len(active) != (1 if fixture == "open_game_follow_up" else 0):
        raise ValueError("controlled follow-up task status does not match its declared fixture")
    for task in tasks:
        detail = task.get("payload", {})
        if (task.get("task_type") != "conversation_follow_up" or task.get("aggregate_type") != "conversation_goal"
                or task.get("conversation_id") != message.get("conversation_id") or task.get("subject_id") != message.get("sender_id")
                or not context_id or detail.get("task_context_id") != context_id or task.get("aggregate_id") != context_id):
            raise ValueError("controlled follow-up task belongs to a different customer or goal")
    if tasks:
        scheduled = results[2]["result"].get("follow_up", {})
        persisted = next((task for task in tasks if task["task_id"] == scheduled.get("task_id")), None)
        if (persisted is None or scheduled.get("status") != "pending"
                or any(persisted.get(key) != scheduled.get(key) for key in (
                    "task_id", "task_type", "aggregate_type", "aggregate_id", "conversation_id", "subject_id", "due_at", "payload"))
                or calls[2].get("arguments", {}).get("due_at") != scheduled.get("due_at")
                or any(calls[2].get("arguments", {}).get(key) != scheduled.get("payload", {}).get(key)
                       for key in ("goal", "waiting_for", "expires_at"))):
            raise ValueError("controlled follow-up persisted task differs from the actual scheduling result")
        if fixture == "open_game_cancelled_follow_up" and (
                any(task["status"] != "cancelled" for task in tasks) or results[3]["result"].get("closed") is not True
                or not any(task["task_id"] != scheduled["task_id"]
                           and task["payload"].get("closed_reason") == calls[3].get("arguments", {}).get("reason") for task in tasks)):
            raise ValueError("controlled follow-up cancellation did not persist")
    if active:
        # 按原审查时刻验证有效期，完整门禁稍后复核时任务可能已正常到期。
        # 当前TraceEvent按项目DEFAULT_TZ格式化为无偏移时间；不能从客户消息猜审计时区。
        from mahjong_agent_runtime.models import DEFAULT_TZ
        reviewed_at = dt.datetime.strptime(events[prompt_index]["time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=DEFAULT_TZ)
        for task in active:
            due_at = dt.datetime.fromisoformat(task["due_at"])
            expires_at = dt.datetime.fromisoformat(task["payload"]["expires_at"])
            if due_at.tzinfo is None or expires_at.tzinfo is None or not reviewed_at < due_at <= expires_at:
                raise ValueError("controlled follow-up was not valid when the model reviewed the commitment")
    summaries = [{**{key: task[key] for key in ("task_id", "task_type", "status", "due_at", "aggregate_type", "aggregate_id")},
                  **{key: task["payload"][key] for key in ("task_context_id", "goal", "waiting_for", "expires_at")}} for task in active]
    if request.get("active_follow_up_summaries") != summaries:
        raise ValueError("controlled follow-up model projection omits an active task or revives a cancelled task")


def validate_customer_review_scope(payload: Any, phase: dict[str, Any]) -> None:
    """核验受控真实审查的全部源案例、原始判决和只读数据库，禁止标签自证通过。"""
    import sqlite3
    from run_concurrency_eval import load_customer_visible_review_contract_cases, load_follow_up_commitment_review_cases
    from mahjong_agent_runtime.customer_visible_review import (
        item_reviews_approved, normalize_item_reviews, parse_reply_self_review,
    )
    from mahjong_agent_runtime.domains.context_builders.tool_results import compact_tool_result_dict

    datasets = ("eval/golden/customer_visible_review_contract.jsonl", "eval/golden/follow_up_commitment_review.jsonl")
    groups = (load_customer_visible_review_contract_cases(), load_follow_up_commitment_review_cases())
    sources = {case["case_id"]: case for group in groups for case in group}
    source_assets = {case["case_id"]: asset for asset, group in zip(datasets, groups) for case in group}
    if (len(sources) != sum(map(len, groups)) or not isinstance(payload, dict) or payload.get("status") != "passed"
            or payload.get("execution_scope") != "controlled_customer_visible_review_with_real_sqlite_tool_evidence"
            or payload.get("case_count") != len(sources) or payload.get("passed_count") != len(sources)
            or type(payload.get("failed_count")) is not int or payload["failed_count"] != 0
            or payload.get("source_ids") != list(sources) or payload.get("dataset_paths") != list(datasets)):
        raise ValueError("controlled review scope is missing, partial or failed")
    reports = payload.get("reports")
    if not isinstance(reports, list) or len(reports) != len(sources):
        raise ValueError("controlled review per-case evidence is incomplete")
    root = Path(phase["report_path"]).resolve().parent
    seen: set[str] = set()
    for item in reports:
        case_id = item.get("case_id")
        if case_id not in sources or case_id in seen or item.get("source_id") != case_id or item.get("status") != "passed":
            raise ValueError("controlled review contains duplicate, unknown or failed cases")
        seen.add(case_id)
        source = sources[case_id]
        if (item.get("text") != source["text"] or item.get("source") != source["source"]
                or type(item.get("expected_approved")) is not bool or item["expected_approved"] is not source["expected_approved"]
                or item.get("fixture") != source.get("fixture", "empty_inventory") or item.get("asset_path") != source_assets[case_id]):
            raise ValueError("controlled review changed the fixed source text or expected verdict")
        raw_path, db_path = Path(item.get("raw_trace_path", "")).resolve(), Path(item.get("db_path", "")).resolve()
        if any(not path.is_relative_to(root) or not path.is_file() for path in (raw_path, db_path)):
            raise ValueError("controlled review raw evidence must belong to this run")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if any(raw.get(key) != item.get(key) for key in ("trace_id", "case_id", "source_id")):
            raise ValueError("controlled review original trace identity differs")
        events = raw.get("events")
        if not isinstance(events, list) or not events or item.get("trace_steps") != [event.get("step") for event in events]:
            raise ValueError("controlled review original events are missing or changed")
        if any(event.get("trace_id") != item["trace_id"] for event in events):
            raise ValueError("controlled review mixed unrelated trace events")
        prompts = [event["content"]["messages"] for event in events if event["step"] == "customer_visible_content_review_prompt"]
        responses = [event["content"].get("content") for event in events if event["step"] == "customer_visible_content_review_response"]
        if len(prompts) != 1 or len(responses) != 1 or responses[0] != item.get("raw_response"):
            raise ValueError("controlled review requires exactly one original model judgement")
        request = json.loads(prompts[0][-1]["content"])
        serialized_request = json.dumps(prompts[0], ensure_ascii=False)
        if (case_id in serialized_request or "expected_approved" in serialized_request
                or source_assets[case_id] in serialized_request
                or ("fixture" in source and source["fixture"] in serialized_request)):
            raise ValueError("controlled review leaked its label into the model request")
        review_items = request.get("review_items")
        if (not isinstance(review_items, list) or len(review_items) != 1
                or review_items[0].get("text") != source["text"] or review_items[0].get("source_text") != source["text"]
                or review_items[0].get("source") != "reply_to_user"
                or review_items[0].get("recipient_id") != request.get("current_message", {}).get("sender_id")):
            raise ValueError("controlled review model did not judge the actual fixed text")
        parsed, errors = parse_reply_self_review(responses[0])
        normalized = normalize_item_reviews(parsed, review_items) if not errors else []
        final = bool(not errors and item_reviews_approved(normalized, review_items)
                     and not parsed.get("needs_human") and not parsed.get("violations"))
        review = item.get("review_result")
        if (not isinstance(review, dict) or review.get("name") != "customer_visible_content_review"
                or review.get("called") is not True or review.get("allowed") is not True or review.get("error")
                or type(item.get("raw_approved")) is not bool or item["raw_approved"] is not parsed.get("approved")
                or item["raw_approved"] is not source["expected_approved"]
                or type(item.get("final_approved")) is not bool or item["final_approved"] is not final
                or final is not source["expected_approved"]
                or review.get("result", {}).get("approved") is not final
                or review.get("result", {}).get("raw_approved") is not item["raw_approved"]
                or review.get("result", {}).get("item_reviews") != normalized):
            raise ValueError("controlled review verdict contradicts the original judgement")
        results = [event["content"] for event in events if event["step"] == "tool_result"]
        if [result for result in results if result.get("name") == "customer_visible_content_review"] != [review]:
            raise ValueError("controlled review is missing its single original review result")
        query = item.get("tool_result")
        if "fixture" in source:
            tables = {"game_count": "runtime_games", "invite_count": "runtime_invite_drafts",
                      "outbound_count": "runtime_outbound_message_drafts", "scheduled_count": "runtime_scheduled_agent_tasks"}
            with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as connection:
                state, tasks = _controlled_review_database_state(connection)
                _validate_controlled_follow_up_evidence(source, item, events, request, connection, tasks)
            before_after = item.get("business_state")
            if (not isinstance(before_after, dict) or before_after != {"before": state, "after": state}
                    or item.get("persisted_state") != {key: state[key] for key in tables}
                    or state["game_count"] != 1 or any(state[key] for key in ("invite_count", "outbound_count", "waiting_count"))):
                raise ValueError("controlled follow-up review changed the declared business state")
            expected_checks = {"actual_open_game_empty_candidate_search", "prerequisites_executed_in_declared_order",
                               "follow_up_state_matches_fixture", "review_completed_once", "review_verdict_matches_expected", "no_review_business_side_effects"}
            checks = item.get("checks")
            if (not isinstance(checks, list) or len(checks) != len(expected_checks)
                    or {check.get("name") for check in checks} != expected_checks
                    or any(check.get("passed") is not True for check in checks)):
                raise ValueError("controlled follow-up omitted a required case check")
            continue
        if (not isinstance(query, dict) or query not in results or review not in results
                or query.get("name") != "search_current_games" or query.get("called") is not True
                or query.get("allowed") is not True or query.get("error")
                or query.get("result", {}).get("matches") != [] or query.get("result", {}).get("alternatives") != []
                or any(query["result"].get("requirement", {}).get(key) != value for key, value in source["requirement"].items())
                or compact_tool_result_dict(query) not in request.get("turn_tool_evidence", [])):
            raise ValueError("controlled review lacks its successful real empty inventory read")
        if item.get("prerequisite_results") != [query]:
            raise ValueError("controlled review lost its original prerequisite read")
        calls = [event["content"].get("call", {}).get("name") for event in events if event["step"] == "tool_called"]
        if calls != ["search_current_games"]:
            raise ValueError("controlled review executed an unexpected business action")
        tables = {"game_count": "runtime_games", "invite_count": "runtime_invite_drafts",
                  "outbound_count": "runtime_outbound_message_drafts", "scheduled_count": "runtime_scheduled_agent_tasks"}
        with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as connection:
            complete_state, _ = _controlled_review_database_state(connection)
            state = {key: complete_state[key] for key in tables}
        if (any(state.values()) or complete_state["waiting_count"] or item.get("persisted_state") != state
                or item.get("business_state") != {"before": complete_state, "after": complete_state}):
            raise ValueError("controlled review mutated business data or changed state evidence")
        checks = item.get("checks")
        if (not isinstance(checks, list) or {check.get("name") for check in checks} != {
                "actual_empty_sqlite_read", "review_completed_once", "review_verdict_matches_expected", "no_business_side_effects"}
                or any(check.get("passed") is not True for check in checks)):
            raise ValueError("controlled review omitted a required case check")
    phase["case_coverage"]["customer_visible_review_contract"] = {"total": len(sources), "passed": len(seen), "failed": 0}


def validate_scope_report(phase: dict[str, Any]) -> None:
    """复核子报告，防止旧运行器以零退出码跳过模型或产生零案例而被门禁误判通过。"""
    if not phase.get("report_path"):
        return
    payload = json.loads(Path(phase["report_path"]).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scope report must be an object")
    if "status" in payload and payload["status"] not in {"passed", "completed"}:
        raise ValueError(f"scope report status is {payload['status']!r}, not passed")
    if "quality_status" in payload and payload["quality_status"] != "passed":
        raise ValueError(f"scope quality_status is {payload['quality_status']!r}, not passed")
    scope = phase["scope"]
    summary = payload.get("summary")
    if isinstance(summary, dict):
        counts = {key: summary.get(key) for key in ("total", "executed", "passed", "failed", "skipped")}
        if any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError("scope summary requires non-negative integer case counts")
        if counts["total"] <= 0 or counts["executed"] <= 0 or counts["failed"] or counts["passed"] != counts["executed"]:
            raise ValueError("scope summary contains failed or unexecuted cases")
        if counts["executed"] + counts["skipped"] != counts["total"]:
            raise ValueError("scope summary case counts contradict each other")
        if not scope.startswith("group_flow_deterministic") and counts["skipped"]:
            raise ValueError("live scope must not silently skip cases")
        cases = payload.get("cases")
        if not isinstance(cases, list) or len(cases) != counts["total"]:
            raise ValueError("group scope per-case evidence is incomplete")
        if any(not isinstance(case, dict) or case.get("status") not in {"passed", "skipped"} for case in cases):
            raise ValueError("group scope contains a failed or invalid case")
        if sum(case["status"] == "skipped" for case in cases) != counts["skipped"]:
            raise ValueError("group case statuses contradict their summary")
        phase["case_coverage"] = counts
    elif scope.startswith(("group_flow_deterministic", LIVE_SCOPES[1])):
        raise ValueError("group scope did not produce its case summary")
    if scope == LIVE_SCOPES[0] or scope == LIVE_SCOPES[4]:
        count_key = "scenario_count" if scope == LIVE_SCOPES[0] else "case_count"
        count = payload.get(count_key)
        if type(count) is not int or count <= 0 or payload.get("passed_count") != count or payload.get("failed_count") != 0:
            raise ValueError("live scope did not pass every selected case")
        reports = payload.get("reports")
        if not isinstance(reports, list) or len(reports) != count or any(item.get("status") != "passed" for item in reports):
            raise ValueError("live scope per-case results are missing, failed or skipped")
        phase["case_coverage"] = {"total": count, "passed": count, "failed": 0, "skipped": 0}
        if scope == LIVE_SCOPES[0]:
            validate_owner_scope_coverage(payload, phase)
    if scope == LIVE_SCOPES[2]:
        live = payload.get("live")
        if not isinstance(live, dict) or live.get("status") != "passed":
            raise ValueError("concurrency scope skipped or failed its live suite")
        count = live.get("scenario_count")
        if type(count) is not int or count <= 0 or live.get("passed_count") != count or live.get("failed_count") != 0:
            raise ValueError("concurrency live scenario coverage is incomplete")
        reports = live.get("reports")
        if (not isinstance(reports, list) or len(reports) != count
                or any(not isinstance(item, dict) or item.get("status") != "passed" for item in reports)):
            raise ValueError("concurrency case reports are missing or contain failures")
        phase["case_coverage"] = {"total": count, "passed": count, "failed": 0, "skipped": 0}
        contract = load_owner_semantic_contract()
        repeat = int(phase["command"][phase["command"].index("--live-repeats") + 1])
        # 并发场景来自同一个老板目录；复用逐案例核验，不能只信子报告自报的通过次数。
        validate_owner_scope_coverage({
            "execution_scope": "all_live_scenarios", "semantic_judge_ran": True,
            "scenario_count": count, "scenario_definition_count": len(contract["scenario_ids"]),
            "repeat": repeat, "reports": live.get("reports", []),
            "recovery_summary": live.get("recovery_summary"),
            "golden_reply_coverage": {
                "total_cases": len(contract["golden_case_ids"]),
                "executed_case_ids": contract["golden_case_ids"], "passed_case_ids": contract["golden_case_ids"],
                "unmapped_case_ids": [], "partially_executed_case_ids": [], "timeline_assets": contract["timeline_assets"],
            },
        }, phase)
        script_dir = str(Path(__file__).resolve().parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from run_concurrency_eval import load_active_revision_fixture
        from run_inquiry_grounding_eval import _parse_semantic_assessment, _semantic_criteria_ids

        fixture = load_active_revision_fixture()
        revision = live.get("active_input_revision")
        if (not isinstance(revision, dict) or revision.get("status") != "passed"
                or revision.get("case_id") != fixture["id"] or revision.get("fixture") != fixture):
            raise ValueError("concurrency scope skipped or changed the active input revision fixture")
        checks = revision.get("checks")
        expected_checks = fixture["expected"]["hard_checks"]
        if (not isinstance(checks, list) or len(checks) != len(expected_checks)
                or {item.get("name") for item in checks if isinstance(item, dict)} != set(expected_checks)
                or any(not isinstance(item, dict) or item.get("passed") is not True for item in checks)):
            raise ValueError("active input revision did not pass every declared hard check")
        assessment = revision.get("semantic_assessment")
        if (not isinstance(assessment, dict) or assessment.get("passed") is not True
                or assessment.get("error") is not None or assessment.get("expectation") != fixture["expected"]):
            raise ValueError("active input revision semantic judgement is absent or failed")
        parsed = _parse_semantic_assessment(assessment.get("raw_response", ""), _semantic_criteria_ids(fixture["expected"]))
        if (parsed["passed"] is not True or parsed["reason"] != assessment.get("reason")
                or json.dumps(parsed["criteria"], sort_keys=True) != json.dumps(assessment.get("criteria"), sort_keys=True)):
            raise ValueError("active input revision judgement contradicts its raw output")
        if (revision.get("old_result", {}).get("status") != "superseded"
                or revision.get("old_result", {}).get("final_reply")
                or revision.get("latest_result", {}).get("status") != "waiting_user"):
            raise ValueError("active input revision retained an old reply or lost the pending user decision")
        recovery = revision.get("recovery_evidence")
        if (not isinstance(recovery, dict) or recovery.get("scenario_id") != fixture["id"]
                or recovery.get("iteration") != 1 or recovery.get("checks") != checks
                or recovery.get("runtime_status") != revision["latest_result"].get("status")
                or recovery.get("final_reply") != revision["latest_result"].get("final_reply")
                or recovery.get("trace_id") != revision["latest_result"].get("trace_id")):
            raise ValueError("active input revision recovery evidence differs from its latest result")
        validate_owner_recovery_evidence(recovery, phase)
        phase["case_coverage"]["active_input_revision"] = {"case_id": fixture["id"], "passed": 1, "failed": 0}
        phase["active_input_recovery_metrics"] = recovery["recovery_metrics"]
        validate_customer_review_scope(live.get("customer_visible_review_contract"), phase)
    if scope == "concurrency_deterministic":
        cases = payload.get("deterministic")
        if not isinstance(cases, list) or not cases or any(item.get("status") != "passed" for item in cases):
            raise ValueError("deterministic concurrency cases are missing or failed")
        phase["case_coverage"] = {"total": len(cases), "passed": len(cases), "failed": 0, "skipped": 0}
    if scope == LIVE_SCOPES[3]:
        if payload.get("status") != "completed" or payload.get("quality_status") != "passed" or payload.get("total_messages") != 6:
            raise ValueError("inquiry scope must complete all six real-model cases")
        checks = payload.get("checks")
        if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
            raise ValueError("inquiry semantic checks are absent or failed")
        phase["case_coverage"] = {"total": 6, "passed": 6, "failed": 0, "skipped": 0}
    if "status" not in payload and not isinstance(summary, dict):
        raise ValueError("scope report has no verifiable completion status")


def main() -> int:
    """完整门禁收集独立范围的全部失败；局部模式和用户中断仍立即停止。

    每个范围的报告独立保存。完整运行中出现失败不会跳过后续集合，但任何失败
    都使最终退出码为非零，不能用其他集合的成功覆盖它。
    """
    args = parse_args()
    # 子运行器在解析参数时读取环境默认值，须先加载应用配置再启动各范围；外部环境优先。
    load_dotenv_defaults(ROOT / ".env")
    run_dir = create_run_directory()
    print(f"本次评测证据目录：{run_dir}", flush=True)
    if not args.full:
        if not (args.live_real_owner or args.live_real_group or args.live_concurrency):
            print(OFFLINE_SEMANTIC_NOTICE, flush=True)
        else:
            print("仅运行显式选择的部分真实模型评测；其他范围未执行，不能称全量业务通过。", flush=True)
    plan = build_eval_plan(args, run_dir)
    requested = {phase.get("scope_family", phase["scope"]) for phase in plan}
    report: dict[str, Any] = {
        "run_id": run_dir.name, "full": args.full, "status": "running",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scopes": plan, "passed_scopes": [], "failed_scopes": [],
        "not_requested_scopes": [scope for scope in LIVE_SCOPES if scope not in requested],
        "coverage_limits": {
            "owner_timelines": "既有专用pytest提供时间线回归，是否执行以本次scopes为准；不声称已逐条真实模型回放",
            "group_adversarial": "既有数据校验覆盖；现有群聊运行器只执行明确选中的golden，不声称歧义对抗集已真实回放",
        },
    }
    report_path = run_dir / "gate_report.json"
    save_gate_report(report_path, report)
    for index, phase in enumerate(plan, start=1):
        phase.update(status="running", log_path=str(run_dir / f"{index:02d}_{phase['scope']}.log"))
        save_gate_report(report_path, report)
        try:
            run_command(phase["command"], log_path=Path(phase["log_path"]))
            archive_external_scope_report(phase, required=True)
            validate_scope_report(phase)
        except (Exception, KeyboardInterrupt) as exc:
            phase.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            report["failed_scopes"].append(phase["scope"])
            try:
                archive_external_scope_report(phase, required=False)
            except OSError as archive_error:
                phase["archive_error"] = f"{type(archive_error).__name__}: {archive_error}"
            print(f"评测失败范围：{phase['scope']}；日志：{phase['log_path']}", flush=True)
            if not args.full or isinstance(exc, KeyboardInterrupt):
                for later in plan[index:]:
                    later["status"] = "not_run"
                report.update(status="failed", finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
                save_gate_report(report_path, report)
                print("后续未运行范围：" + ", ".join(item["scope"] for item in plan[index:]), flush=True)
                print(f"范围与结果清单：{report_path}", flush=True)
                return 1
            # 各范围使用独立夹具和证据目录；失败只阻断最终通过结论，不阻断收集其他失败。
            report["status"] = "running_with_failures"
            save_gate_report(report_path, report)
            print("完整门禁继续执行其余独立范围；本次已记录失败，最终仍会返回非零。", flush=True)
            continue
        phase["status"] = "passed"
        report["passed_scopes"].append(phase["scope"])
        save_gate_report(report_path, report)
    report.update(status="failed" if report["failed_scopes"] else "passed",
                  finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    save_gate_report(report_path, report)
    if report["failed_scopes"]:
        print("本次门禁未通过，失败范围：" + ", ".join(report["failed_scopes"]), flush=True)
    print("本次已执行并通过的范围：" + ", ".join(report["passed_scopes"]), flush=True)
    if report["not_requested_scopes"]:
        print("本次未执行范围：" + ", ".join(report["not_requested_scopes"]), flush=True)
    for name, limitation in report["coverage_limits"].items():
        print(f"覆盖说明 {name}：{limitation}", flush=True)
    print(f"范围与结果清单：{report_path}", flush=True)
    return 1 if report["failed_scopes"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
