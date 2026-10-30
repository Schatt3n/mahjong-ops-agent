"""完整门禁必须选择全部既有评测范围，离线成功和未执行范围不能冒充业务通过。"""

from __future__ import annotations

import datetime as dt
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_evals.py"


@pytest.mark.parametrize("file_timeout,external_timeout,expected", [(None, None, 45), ("90", None, 90), ("90", "105", 105)])
def test_main_loads_config_before_real_child_parser_defaults(gate, monkeypatch, capsys, file_timeout, external_timeout, expected):
    """临时配置经真实子进程传到三个解析器，解析后立即截停，不读取真实密钥或发模型请求。"""
    timeout_key = "MAHJONG_AGENT_LLM_TIMEOUT_SECONDS"
    canary_key = "MAHJONG_EVAL_CONFIG_CANARY"
    monkeypatch.delenv(timeout_key, raising=False)
    monkeypatch.delenv(canary_key, raising=False)
    if external_timeout is not None:
        monkeypatch.setenv(timeout_key, external_timeout)
    if file_timeout is not None:
        (gate.ROOT / ".env").write_text(f"{timeout_key}={file_timeout}\n{canary_key}=unit-config-value-never-logged\n")
    monkeypatch.setenv("PYTHONPATH", str(SCRIPT.parent.parent / "src"))
    monkeypatch.setattr(sys, "argv", ["run_evals.py"])
    probe = """
import argparse, importlib, json, sys
sys.path.insert(0, SCRIPT_DIRECTORY)
class Parsed(Exception):
    def __init__(self, parsed):
        self.parsed = parsed
original_parse = argparse.ArgumentParser.parse_args
def capture_parse(parser, *args, **kwargs):
    raise Parsed(original_parse(parser, *args, **kwargs))
argparse.ArgumentParser.parse_args = capture_parse
results = {}
for name in ('run_real_owner_chat_live_eval', 'run_concurrency_eval', 'run_privacy_isolation_live_eval'):
    module = importlib.import_module(name)
    try:
        module.main([])
    except Parsed as captured:
        results[name] = captured.parsed.timeout_seconds
    else:
        raise AssertionError('runner did not parse arguments')
print(json.dumps(results))
""".replace("SCRIPT_DIRECTORY", repr(str(SCRIPT.parent)))
    monkeypatch.setattr(gate, "build_eval_plan", lambda args, run_dir: [
        {"scope": "config_probe", "command": [sys.executable, "-c", probe], "status": "pending"},
    ])
    assert gate.main() == 0
    reports = list((gate.ROOT / "runtime_data/eval_runs").glob("*/gate_report.json"))
    assert len(reports) == 1
    phase = json.loads(reports[0].read_text())["scopes"][0]
    child_output = Path(phase["log_path"]).read_text()
    assert phase["status"] == "passed"
    assert json.loads(child_output) == {
        name: expected for name in ("run_real_owner_chat_live_eval", "run_concurrency_eval", "run_privacy_isolation_live_eval")
    }
    captured = capsys.readouterr()
    assert "unit-config-value-never-logged" not in captured.out + captured.err + child_output


def stub_owner_source_contract():
    expectations = {
        scenario_id: {
            "required_meaning": [{"id": f"{scenario_id}.required", "description": "满足本例必要应答含义。"}],
            "forbidden_meaning": [{"id": f"{scenario_id}.forbidden", "description": "不能虚构已执行动作。"}],
            "reply_reference": "测试参考句，不逐字匹配。",
        } for scenario_id in ("scene-a", "scene-b")
    }
    return {
        "scenario_ids": list(expectations), "expectations": expectations,
        "criterion_ids": {scenario_id: [entry["id"] for group in ("required_meaning", "forbidden_meaning") for entry in expected[group]]
                          for scenario_id, expected in expectations.items()},
        "golden_case_ids": ["golden-a", "golden-b"],
        "timeline_assets": {"live_status": "not_run", "deterministic_scope": "test fixture", "files": []},
    }


@pytest.fixture
def gate(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("eval_gate_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_owner_semantic_contract", stub_owner_source_contract)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    golden = tmp_path / "eval" / "golden"
    golden.mkdir(parents=True)
    for date in ("20260722", "20260723", "20260726"):
        (golden / f"real_group_chat_{date}.jsonl").write_text('{"test_stub":true}\n')
    return module


def successful_stub_report(command, gate, *, repeat=1):
    if "scripts/run_real_owner_chat_live_eval.py" in command:
        from run_real_owner_chat_live_eval import assess_action_contract_recovery, summarize_action_contract_recovery

        contract = gate.load_owner_semantic_contract()
        reports = []
        for iteration in range(1, repeat + 1):
            for scenario_id in contract["scenario_ids"]:
                criteria = [{"id": criterion_id, "passed": True, "reason": "固定测试判定，不调用模型。"}
                            for criterion_id in contract["criterion_ids"][scenario_id]]
                original = {"reason": "测试通过", "criteria": [
                    {"id": item["id"], "violation_detected": False, "reason": item["reason"]} for item in criteria
                ]}
                trace_id = f"trace_stub_{scenario_id}_{iteration}"
                events = [{"trace_id": trace_id, "step": "final_output", "content": {"reply": "固定测试回复"}, "level": "INFO"}]
                report_path = Path(command[command.index("--report-path") + 1]) if "--report-path" in command else gate.ROOT / "stub_report.json"
                trace_root = report_path.parent / "stub_traces"
                trace_root.mkdir(parents=True, exist_ok=True)
                trace_path = trace_root / f"{uuid.uuid4().hex}.json"
                trace_path.write_text(json.dumps({"trace_id": trace_id, "scenario_id": scenario_id,
                    "iteration": iteration, "events": events}))
                recovery = assess_action_contract_recovery([SimpleNamespace(**event) for event in events],
                    runtime_status="waiting_user", final_reply="固定测试回复", business_checks_passed=True, semantic_passed=True)
                reports.append({
                    "status": "passed", "scenario_id": scenario_id, "iteration": iteration,
                    "trace_id": trace_id, "runtime_status": "waiting_user", "final_reply": "固定测试回复",
                    "trace_steps": [event["step"] for event in events], "raw_trace_path": str(trace_path),
                    "checks": [{"name": "unit_stub_business_check", "passed": True}], "recovery_metrics": recovery,
                    "semantic_assessment": {
                        "reason": original["reason"], "criteria": criteria, "passed": True, "error": None,
                        "expectation": deepcopy(contract["expectations"][scenario_id]),
                        "raw_response": json.dumps(original, ensure_ascii=False),
                    },
                })
        return {
            "status": "passed", "scenario_count": len(reports), "passed_count": len(reports), "failed_count": 0,
            "scenario_definition_count": len(contract["scenario_ids"]), "repeat": repeat,
            "reports": reports,
            "recovery_summary": summarize_action_contract_recovery(reports),
            "execution_scope": "all_live_scenarios", "semantic_judge_ran": True,
            "golden_reply_coverage": {
                "total_cases": len(contract["golden_case_ids"]), "executed_case_ids": list(contract["golden_case_ids"]),
                "passed_case_ids": list(contract["golden_case_ids"]),
                "partially_executed_case_ids": [], "unmapped_case_ids": [],
                "timeline_assets": deepcopy(contract["timeline_assets"]),
            },
        }
    if "scripts/run_privacy_isolation_live_eval.py" in command:
        return {"status": "passed", "case_count": 10, "passed_count": 10, "failed_count": 0,
                "reports": [{"status": "passed"} for _ in range(10)]}
    if "scripts/run_real_group_chat_flow_eval.py" in command:
        return {"mode": "live" if "--live" in command else "deterministic",
                "summary": {"total": 1, "executed": 1, "passed": 1, "failed": 0, "skipped": 0},
                "cases": [{"case_id": "stub", "status": "passed"}]}
    if "scripts/run_concurrency_eval.py" in command:
        from run_concurrency_eval import (
            load_active_revision_fixture, load_customer_visible_review_contract_cases, load_follow_up_commitment_review_cases,
            run_live_customer_visible_review_contract,
        )
        from mahjong_agent_runtime import AgentLLMConfig
        fixture = load_active_revision_fixture()
        criteria = [{"id": item["id"], "passed": True, "reason": "测试合同，不发真实模型请求"}
                    for key in ("required_meaning", "forbidden_meaning") for item in fixture["expected"][key]]
        reason = "测试合同通过"
        assessment = {"passed": True, "error": None, "reason": reason, "criteria": criteria,
            "expectation": fixture["expected"], "raw_response": json.dumps({"reason": reason, "criteria": [
                {"id": item["id"], "violation_detected": False, "reason": item["reason"]} for item in criteria]})}
        from run_real_owner_chat_live_eval import summarize_action_contract_recovery

        owner_reports = successful_stub_report(["scripts/run_real_owner_chat_live_eval.py", "--report-path",
            command[command.index("--report-path") + 1]], gate, repeat=2)["reports"]
        recovery_evidence = deepcopy(owner_reports[0])
        recovery_evidence["scenario_id"] = fixture["id"]
        recovery_evidence["checks"] = [{"name": name, "passed": True} for name in fixture["expected"]["hard_checks"]]
        raw_path = Path(recovery_evidence["raw_trace_path"])
        raw = json.loads(raw_path.read_text())
        raw["scenario_id"] = fixture["id"]
        revision_trace = raw_path.with_name(f"{uuid.uuid4().hex}.json")
        revision_trace.write_text(json.dumps(raw))
        recovery_evidence["raw_trace_path"] = str(revision_trace)
        review_sources = [*load_customer_visible_review_contract_cases(), *load_follow_up_commitment_review_cases()]

        class FixedReviewClient:
            """仅模拟本地模型边界，验证gate证据连线，不能作为真实语义评测结果。"""
            config = AgentLLMConfig(api_key="unit-only-credential", model="unit-stub", base_url="https://example.invalid")
            index = 0

            def complete(self, messages, **kwargs):
                request = json.loads(messages[-1]["content"])
                item = request["review_items"][0]
                case = review_sources[self.index]
                self.index += 1
                assert item["text"] == case["text"]
                approved = case["expected_approved"]
                violations = [] if approved else ["unit_test_rejected"]
                return json.dumps({"approved": approved, "needs_human": False, "reasoning_summary": "确定性模型替身",
                    "violations": violations, "item_reviews": [{"item_id": item["item_id"], "approved": approved,
                        "suggested_safe_text": item["text"] if approved else "要我帮你问问吗？",
                        "reasoning_summary": "仅验证原始证据与判决校验", "violations": violations}]})

        review_scope = run_live_customer_visible_review_contract(FixedReviewClient(),
            SimpleNamespace(timeout_seconds=1, max_tokens_per_call=64000),
            Path(command[command.index("--report-path") + 1]).parent / f"review_contract_{uuid.uuid4().hex}", include_follow_up_cases=True)
        return {
            "status": "passed", "deterministic": [{"status": "passed"} for _ in range(8)],
            "live": {"status": "passed", "scenario_count": 4, "passed_count": 4, "failed_count": 0,
                "reports": owner_reports,
                "recovery_summary": summarize_action_contract_recovery(owner_reports),
                "customer_visible_review_contract": review_scope,
                "active_input_revision": {"case_id": fixture["id"], "status": "passed", "fixture": fixture,
                    "checks": recovery_evidence["checks"], "recovery_evidence": recovery_evidence,
                    "old_result": {"status": "superseded", "final_reply": ""},
                    "latest_result": {"status": "waiting_user", "final_reply": recovery_evidence["final_reply"],
                        "trace_id": recovery_evidence["trace_id"]}, "semantic_assessment": assessment}},
        }
    if "scripts/run_inquiry_grounding_eval.py" in command:
        return {"status": "completed", "quality_status": "passed", "total_messages": 6,
                "checks": {"six_cases_semantic_stub": True}}
    raise AssertionError(f"unexpected JSON report command: {command}")


@pytest.mark.parametrize("fault", ["missing", "changed_fixture", "wrong_id", "missing_check", "numeric_flag",
                                       "missing_judge", "raw_violation", "old_reply", "completed_latest", "missing_recovery"])
def test_active_revision_missing_or_failed_evidence_blocks_full_gate(gate, monkeypatch, fault):
    """老板旧集合全通过也不能掩盖正在处理时改目标的漏测、错误或判定矛盾。"""
    def alter(command, payload):
        if "scripts/run_concurrency_eval.py" not in command or "live" not in command:
            return payload
        live = payload["live"]
        revision = live["active_input_revision"]
        if fault == "missing":
            live.pop("active_input_revision")
        elif fault == "changed_fixture":
            revision["fixture"]["messages"][1]["text"] = "changed standard"
        elif fault == "wrong_id":
            revision["case_id"] = "not-in-golden"
        elif fault == "missing_check":
            revision["checks"].pop()
        elif fault == "numeric_flag":
            revision["checks"][0]["passed"] = 1
        elif fault == "missing_judge":
            revision.pop("semantic_assessment")
        elif fault == "raw_violation":
            raw = json.loads(revision["semantic_assessment"]["raw_response"])
            raw["criteria"][0]["violation_detected"] = True
            revision["semantic_assessment"]["raw_response"] = json.dumps(raw)
        elif fault == "old_reply":
            revision["old_result"]["final_reply"] = "outdated reply"
        elif fault == "completed_latest":
            revision["latest_result"]["status"] = "completed"
        elif fault == "missing_recovery":
            revision.pop("recovery_evidence")
        return payload

    install_recording_subprocess(monkeypatch, gate, alter_report=alter)
    monkeypatch.setattr(sys, "argv", ["run_evals.py", "--full"])
    assert gate.main() == 1
    _, report = only_gate_report(gate)
    assert report["failed_scopes"] == ["real_concurrency_all_scenarios"]
    assert "privacy_isolation_all_cases" in report["passed_scopes"]


@pytest.mark.parametrize("fault", ["missing", "missing_case", "unknown_source", "changed_expected", "raw_contradiction",
                                       "label_leak", "missing_query", "hidden_database_write", "numeric_flag"])
def test_controlled_review_requires_all_real_sources_and_consistent_raw_evidence(gate, monkeypatch, fault):
    """其他回放全绿也不能掩盖审查漏测、伪造标签或源数据库中的写入。"""
    def alter(command, payload):
        if "scripts/run_concurrency_eval.py" not in command or "live" not in command:
            return payload
        live = payload["live"]
        scope = live["customer_visible_review_contract"]
        if fault == "missing":
            live.pop("customer_visible_review_contract")
            return payload
        first = scope["reports"][0]
        raw_path = Path(first["raw_trace_path"])
        raw = json.loads(raw_path.read_text())
        if fault == "missing_case":
            scope["reports"].pop()
        elif fault == "unknown_source":
            first["source_id"] = "not-a-source"
        elif fault == "changed_expected":
            first["expected_approved"] = not first["expected_approved"]
        elif fault == "raw_contradiction":
            response = next(event for event in raw["events"] if event["step"] == "customer_visible_content_review_response")
            parsed = json.loads(response["content"]["content"])
            parsed["approved"] = not parsed["approved"]
            response["content"]["content"] = json.dumps(parsed)
            first["raw_response"] = response["content"]["content"]
        elif fault == "label_leak":
            prompt = next(event for event in raw["events"] if event["step"] == "customer_visible_content_review_prompt")
            request = json.loads(prompt["content"]["messages"][-1]["content"])
            request["expected_approved"] = first["expected_approved"]
            prompt["content"]["messages"][-1]["content"] = json.dumps(request)
        elif fault == "missing_query":
            raw["events"] = [event for event in raw["events"] if event["step"] != "tool_called"]
            first["trace_steps"] = [event["step"] for event in raw["events"]]
        elif fault == "hidden_database_write":
            import sqlite3
            with sqlite3.connect(first["db_path"]) as connection:
                connection.execute("INSERT INTO runtime_games VALUES (?, ?, ?, ?, ?)",
                    ("unexpected", "unit", "forming", "{}", "2026-10-29"))
        elif fault == "numeric_flag":
            first["final_approved"] = int(first["final_approved"])
        raw_path.write_text(json.dumps(raw))
        return payload

    install_recording_subprocess(monkeypatch, gate, alter_report=alter)
    monkeypatch.setattr(sys, "argv", ["run_evals.py", "--full"])
    assert gate.main() == 1
    _, report = only_gate_report(gate)
    assert report["failed_scopes"] == ["real_concurrency_all_scenarios"]
    assert "privacy_isolation_all_cases" in report["passed_scopes"]


@pytest.fixture
def controlled_review_evidence(gate, tmp_path):
    """用真实隔离SQLite和固定审查替身构造证据，测试不得调用外部模型。"""
    report_path = tmp_path / "controlled_review_gate.json"
    payload = successful_stub_report(["scripts/run_concurrency_eval.py", "--report-path", str(report_path)], gate)
    scope = payload["live"]["customer_visible_review_contract"]
    phase = {"report_path": str(report_path), "case_coverage": {}}
    gate.validate_customer_review_scope(scope, phase)
    return scope, phase


def test_controlled_review_full_gate_covers_every_source_and_real_task_status(gate, controlled_review_evidence):
    """完整门禁覆盖两个资产的全部ID，正反例与取消状态都来自真实前置工具。"""
    from run_concurrency_eval import load_customer_visible_review_contract_cases, load_follow_up_commitment_review_cases

    scope, phase = controlled_review_evidence
    sources = [*load_customer_visible_review_contract_cases(), *load_follow_up_commitment_review_cases()]
    assert scope["source_ids"] == [source["case_id"] for source in sources]
    assert [item["case_id"] for item in scope["reports"]] == scope["source_ids"]
    assert scope["dataset_paths"] == ["eval/golden/customer_visible_review_contract.jsonl", "eval/golden/follow_up_commitment_review.jsonl"]
    assert sum(source["expected_approved"] for source in sources) == 7
    assert phase["case_coverage"]["customer_visible_review_contract"] == {"total": len(sources), "passed": len(sources), "failed": 0}
    for item in scope["reports"]:
        state = item["business_state"]["after"]
        assert item["business_state"]["before"] == state
        if item["fixture"] == "open_game_cancelled_follow_up":
            assert state["scheduled_count"] == 2
            assert [task["status"] for task in state["scheduled_tasks"]] == ["cancelled", "cancelled"]
        elif item["fixture"] == "open_game_follow_up":
            assert state["scheduled_count"] == 1
            assert state["scheduled_tasks"][0]["status"] == "pending"


@pytest.mark.parametrize("fault", [
    "old_eight_only", "missing_new_case", "missing_dataset", "changed_fixture", "missing_prerequisite",
    "missing_gateway", "wrong_tool_order", "tool_after_review", "missing_review_result", "missing_active_projection",
    "cancelled_projection_revived", "deleted_task", "cancelled_task_reactivated", "task_wrong_customer", "task_wrong_goal",
    "same_count_game_mutation", "source_label_in_system", "fixture_label_in_request", "second_review_prompt",
    "second_review_response", "new_missing_check", "new_numeric_flag", "review_changed_state", "raw_aggregate_repaired", "task_expired_at_review",
])
def test_controlled_follow_up_gate_rejects_incomplete_or_forged_mechanism_proof(gate, controlled_review_evidence, fault):
    """源标签、假任务、取消复活或重复判定不能替代完整工具与持久状态证据。"""
    import sqlite3

    scope, phase = controlled_review_evidence
    target_fixture = "open_game_cancelled_follow_up" if fault in {"cancelled_projection_revived", "cancelled_task_reactivated"} else "open_game_follow_up"
    item = next(item for item in scope["reports"] if item["fixture"] == target_fixture)
    raw_path = Path(item["raw_trace_path"])
    raw = json.loads(raw_path.read_text())
    prompt = next(event for event in raw["events"] if event["step"] == "customer_visible_content_review_prompt")
    request = json.loads(prompt["content"]["messages"][-1]["content"])
    if fault == "old_eight_only":
        scope["reports"] = [item for item in scope["reports"] if item["fixture"] == "empty_inventory"]
        scope["case_count"] = scope["passed_count"] = len(scope["reports"])
        scope["source_ids"] = [item["case_id"] for item in scope["reports"]]
        scope["dataset_paths"].pop()
    elif fault == "missing_new_case":
        scope["reports"].remove(item)
    elif fault == "missing_dataset":
        scope["dataset_paths"].pop()
    elif fault == "changed_fixture":
        item["fixture"] = "open_game_empty_candidates"
    elif fault == "missing_prerequisite":
        item["prerequisite_results"].pop()
    elif fault == "missing_gateway":
        raw["events"] = [event for event in raw["events"] if event["step"] != "tool_gateway_completed"]
    elif fault == "wrong_tool_order":
        calls = [event for event in raw["events"] if event["step"] == "tool_called"]
        calls[0]["content"]["call"]["name"], calls[1]["content"]["call"]["name"] = calls[1]["content"]["call"]["name"], calls[0]["content"]["call"]["name"]
    elif fault == "tool_after_review":
        event = next(event for event in raw["events"] if event["step"] == "tool_called")
        raw["events"].remove(event)
        raw["events"].append(event)
    elif fault == "missing_review_result":
        raw["events"] = [event for event in raw["events"] if event["step"] != "tool_result"
                         or event["content"].get("name") != "customer_visible_content_review"]
    elif fault == "missing_active_projection":
        request["active_follow_up_summaries"] = []
    elif fault == "cancelled_projection_revived":
        request["active_follow_up_summaries"] = [{"task_id": item["business_state"]["after"]["scheduled_tasks"][0]["task_id"], "status": "pending"}]
    elif fault in {"deleted_task", "cancelled_task_reactivated", "task_wrong_customer", "task_wrong_goal"}:
        with sqlite3.connect(item["db_path"]) as connection:
            task_id, payload = connection.execute("SELECT task_id, payload FROM runtime_scheduled_agent_tasks ORDER BY task_id LIMIT 1").fetchone()
            task = json.loads(payload)
            if fault == "deleted_task":
                connection.execute("DELETE FROM runtime_scheduled_agent_tasks WHERE task_id=?", (task_id,))
            elif fault == "cancelled_task_reactivated":
                task["status"] = "pending"
                connection.execute("UPDATE runtime_scheduled_agent_tasks SET status='pending',payload=? WHERE task_id=?", (json.dumps(task), task_id))
            else:
                if fault == "task_wrong_customer":
                    task["subject_id"] = "another_customer"
                else:
                    task["payload"]["task_context_id"] = "another_goal"
                connection.execute("UPDATE runtime_scheduled_agent_tasks SET payload=? WHERE task_id=?", (json.dumps(task), task_id))
    elif fault == "same_count_game_mutation":
        with sqlite3.connect(item["db_path"]) as connection:
            game_id, payload = connection.execute("SELECT game_id,payload FROM runtime_games").fetchone()
            game = json.loads(payload)
            game["requirement"]["stake"] = "2"
            connection.execute("UPDATE runtime_games SET payload=? WHERE game_id=?", (json.dumps(game), game_id))
    elif fault == "source_label_in_system":
        prompt["content"]["messages"][0]["content"] += " expected_approved=true"
    elif fault == "fixture_label_in_request":
        request["fixture"] = item["fixture"]
    elif fault == "task_expired_at_review":
        task = item["business_state"]["after"]["scheduled_tasks"][0]
        expired = dt.datetime.fromisoformat(task["payload"]["expires_at"]) + dt.timedelta(seconds=1)
        prompt["time"] = expired.strftime("%Y-%m-%d %H:%M:%S")
    elif fault == "second_review_prompt":
        raw["events"].append(deepcopy(prompt))
    elif fault == "second_review_response":
        raw["events"].append(deepcopy(next(event for event in raw["events"] if event["step"] == "customer_visible_content_review_response")))
    elif fault == "new_missing_check":
        item["checks"].pop()
    elif fault == "new_numeric_flag":
        item["checks"][0]["passed"] = 1
    elif fault == "review_changed_state":
        item["business_state"]["before"]["scheduled_count"] = 0
    elif fault == "raw_aggregate_repaired":
        response = next(event for event in raw["events"] if event["step"] == "customer_visible_content_review_response")
        parsed = json.loads(response["content"]["content"])
        parsed["approved"] = False
        response["content"]["content"] = json.dumps(parsed)
        item["raw_response"] = response["content"]["content"]
        item["raw_approved"] = False
    prompt["content"]["messages"][-1]["content"] = json.dumps(request)
    item["trace_steps"] = [event["step"] for event in raw["events"]]
    raw_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        gate.validate_customer_review_scope(scope, phase)


def install_recording_subprocess(monkeypatch, gate, *, fail_script=None, interrupt_script=None,
                                 produce_inquiry_report=True, alter_report=None, recorded_states=None):
    calls = []

    def run(command, **kwargs):
        assert kwargs["check"] is True
        assert kwargs["cwd"] == gate.ROOT
        assert kwargs["env"]["PYTHONPATH"].split(":")[0] == str(gate.ROOT / "src")
        calls.append(list(command))
        if recorded_states is not None:
            recorded_states.append(only_gate_report(gate)[1])
        kwargs["stdout"].write("deterministic subprocess stub; no model request\n")
        if interrupt_script and interrupt_script in command:
            raise KeyboardInterrupt()
        if fail_script and fail_script in command:
            raise subprocess.CalledProcessError(2, command)
        if "--report-path" in command:
            report = Path(command[command.index("--report-path") + 1])
            payload = successful_stub_report(command, gate)
            if alter_report is not None:
                payload = alter_report(command, payload)
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(payload))
        if "scripts/run_inquiry_grounding_eval.py" in command and produce_inquiry_report:
            run_id = command[command.index("--run-id") + 1]
            report = gate.ROOT / "runtime_data" / "periodic_chat_simulation" / "runs" / run_id / "report.json"
            report.parent.mkdir(parents=True)
            payload = {"run_id": run_id, **successful_stub_report(command, gate)}
            if alter_report is not None:
                payload = alter_report(command, payload)
            report.write_text(json.dumps(payload))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(gate.subprocess, "run", run)
    return calls


def only_gate_report(gate):
    paths = list((gate.ROOT / "runtime_data" / "eval_runs").glob("*/gate_report.json"))
    assert len(paths) == 1
    return paths[0], json.loads(paths[0].read_text())


def command_for(calls, script, *, mode=None):
    found = [command for command in calls if script in command and (mode is None or command[command.index("--mode") + 1] == mode)]
    assert len(found) == 1
    return found[0]


def test_owner_source_contract_loads_real_catalog_and_all_golden_fragments():
    spec = importlib.util.spec_from_file_location("eval_gate_real_owner_contract_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = module.load_owner_semantic_contract()
    assert len(contract["scenario_ids"]) == 34
    assert len(set(contract["scenario_ids"])) == 34
    assert len(contract["golden_case_ids"]) == 35
    assert set(contract["expectations"]) == set(contract["scenario_ids"])
    assert all(expected["golden_sources"] and expected["required_meaning"] for expected in contract["expectations"].values())
    assert all(item["id"] != "current_request" for expected in contract["expectations"].values() for item in expected["required_meaning"])
    assert {"flexible_time_empty_pool_offers_help", "explicit_range_creates_pending_game"} <= set(contract["scenario_ids"])
    assert {"single_profile_game_complete_query", "custom_profile_game_complete_query",
            "inventory_all_games_ignores_personal_profile", "inventory_explicit_game_overrides_profile"} <= set(contract["scenario_ids"])
    assert "known_three_person_party_matches_capacity" in contract["scenario_ids"]
    assert all(ids and len(ids) == len(set(ids)) for ids in contract["criterion_ids"].values())
    assert contract["timeline_assets"]["live_status"] == "not_run"
    assert len(contract["timeline_assets"]["files"]) == 2
    assert sum(asset["eval_case_count"] for asset in contract["timeline_assets"]["files"]) == 17
    assert sum(asset["replay_scenario_count"] for asset in contract["timeline_assets"]["files"]) == 13


def test_full_gate_runs_all_existing_live_suites_and_all_pytest_with_unique_artifacts(gate, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_evals.py", "--full"])
    calls = install_recording_subprocess(monkeypatch, gate)
    assert gate.main() == 0
    output = capsys.readouterr().out
    path, report = only_gate_report(gate)
    run_dir = path.parent

    assert report["full"] is True and report["status"] == "passed"
    assert report["not_requested_scopes"] == []
    assert report["passed_scopes"] == [phase["scope"] for phase in report["scopes"]]
    assert len(calls) == len(report["scopes"])
    assert all(phase["status"] == "passed" for phase in report["scopes"])
    assert [sys.executable, "-m", "pytest", "-q"] in calls
    assert not any(command[1:4] == ["-m", "pytest", "-q"] and len(command) > 4 for command in calls)
    assert [sys.executable, "scripts/check_chinese_docstrings.py"] in calls
    assert ["git", "diff", "--check"] in calls

    owner = command_for(calls, "scripts/run_real_owner_chat_live_eval.py")
    groups = [command for command in calls if "scripts/run_real_group_chat_flow_eval.py" in command and "--live" in command]
    deterministic_groups = [command for command in calls if "scripts/run_real_group_chat_flow_eval.py" in command and "--live" not in command]
    concurrency = command_for(calls, "scripts/run_concurrency_eval.py", mode="live")
    inquiry = command_for(calls, "scripts/run_inquiry_grounding_eval.py")
    privacy = command_for(calls, "scripts/run_privacy_isolation_live_eval.py")
    assert "--strict" in owner and "--strict" in privacy and "--strict" in concurrency
    assert len(groups) == len(deterministic_groups) == 3
    expected_datasets = {str(gate.ROOT / "eval" / "golden" / f"real_group_chat_{date}.jsonl") for date in ("20260722", "20260723", "20260726")}
    for commands in (groups, deterministic_groups):
        assert {command[command.index("--dataset") + 1] for command in commands} == expected_datasets
        assert all("--strict" in command for command in commands)
    assert all(command[command.index("--env-file") + 1] == str(gate.ROOT / ".env") for command in groups)
    assert concurrency[concurrency.index("--live-workers") + 1] == "4"
    assert concurrency[concurrency.index("--live-repeats") + 1] == "2"
    for command in (owner, *groups, concurrency, inquiry, privacy):
        assert not {"--scenario", "--case", "--skip-review", "--skip-text-generation"} & set(command)
    for command in (owner, privacy):
        assert Path(command[command.index("--db-path") + 1]).is_relative_to(run_dir)
    assert owner[owner.index("--db-path") + 1] != privacy[privacy.index("--db-path") + 1]
    for phase in report["scopes"]:
        assert Path(phase["log_path"]).is_relative_to(run_dir)
        if "report_path" in phase:
            assert Path(phase["report_path"]).is_relative_to(run_dir)
    inquiry_scope = next(phase for phase in report["scopes"] if phase["scope"] == "inquiry_grounding_six_cases")
    assert Path(inquiry_scope["source_report_path"]).exists()
    assert Path(inquiry_scope["report_path"]).read_text() == Path(inquiry_scope["source_report_path"]).read_text()
    assert inquiry[inquiry.index("--run-id") + 1].startswith(report["run_id"])
    assert "本次已执行并通过的范围" in output
    assert gate.OFFLINE_SEMANTIC_NOTICE not in output
    owner_scope = next(phase for phase in report["scopes"] if phase["scope"] == gate.LIVE_SCOPES[0])
    expected = gate.load_owner_semantic_contract()
    assert owner_scope["case_coverage"]["expected_scenario_ids"] == expected["scenario_ids"]
    assert owner_scope["case_coverage"]["executed_scenario_ids"] == expected["scenario_ids"]
    assert owner_scope["case_coverage"]["golden_case_ids"] == expected["golden_case_ids"]
    assert owner_scope["case_coverage"]["timeline_assets"]["live_status"] == "not_run"


def test_default_gate_keeps_offline_matrix_and_explicitly_disclaims_live_business_coverage(gate, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_evals.py"])
    calls = install_recording_subprocess(monkeypatch, gate)
    assert gate.main() == 0
    output = capsys.readouterr().out
    _, report = only_gate_report(gate)

    assert gate.OFFLINE_SEMANTIC_NOTICE in output
    assert report["full"] is False and report["status"] == "passed"
    assert set(report["not_requested_scopes"]) == set(gate.LIVE_SCOPES)
    assert not any(scope in report["passed_scopes"] for scope in gate.LIVE_SCOPES)
    expected_subset_files = {
        "tests/test_agent_runtime.py", "tests/test_real_owner_chat_golden.py", "tests/test_real_group_chat_dataset.py",
        "tests/test_context_summary.py", "tests/test_context_summary_quality.py", "tests/test_agent_runtime_eval.py",
        "tests/test_agent_app.py", "tests/test_input_aggregation.py", "tests/test_progress_monitor.py",
        "tests/test_future_game_recruitment.py", "tests/test_action_contract_repairs.py", "tests/test_agent_runtime_package.py",
        "tests/test_operational_waiting.py", "tests/test_manual_simulation.py", "tests/test_cross_conversation_participation.py",
        "tests/test_operational_chat_scenarios.py", "tests/test_inquiry_evidence_scope.py",
    }
    actual_files = {name for command in calls if command[1:4] == ["-m", "pytest", "-q"] for name in command[4:]}
    assert actual_files == expected_subset_files
    for script in (
        "scripts/verify_agent_runtime_boundary.py", "scripts/verify_customer_visible_contract.py",
        "scripts/check_badcase_regression_coverage.py", "scripts/validate_real_owner_chat_golden.py",
        "scripts/validate_real_group_chat_dataset.py", "scripts/run_real_group_chat_flow_eval.py",
        "scripts/run_agent_runtime_eval.py",
    ):
        assert any(script in command for command in calls)
    command_for(calls, "scripts/run_concurrency_eval.py", mode="deterministic")
    assert not any("scripts/run_real_owner_chat_live_eval.py" in command for command in calls)
    assert not any("scripts/run_privacy_isolation_live_eval.py" in command for command in calls)
    assert not any("scripts/run_inquiry_grounding_eval.py" in command for command in calls)


def test_partial_live_request_does_not_mark_other_live_scopes_as_passed(gate, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_evals.py", "--live-real-owner"])
    calls = install_recording_subprocess(monkeypatch, gate)
    assert gate.main() == 0
    _, report = only_gate_report(gate)
    owner_scope = "real_owner_all_scenarios_with_reply_semantics"
    assert owner_scope in report["passed_scopes"]
    assert owner_scope not in report["not_requested_scopes"]
    assert set(report["not_requested_scopes"]) == set(gate.LIVE_SCOPES) - {owner_scope}
    command_for(calls, "scripts/run_real_owner_chat_live_eval.py")
    assert "不能称全量业务通过" in capsys.readouterr().out


def test_full_gate_collects_first_failure_and_still_runs_every_remaining_scope(gate, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_evals.py", "--full"])
    states = []
    calls = install_recording_subprocess(monkeypatch, gate, fail_script="scripts/verify_agent_runtime_boundary.py", recorded_states=states)
    assert gate.main() == 1
    _, report = only_gate_report(gate)
    assert report["status"] == "failed"
    phases = report["scopes"]
    assert phases[0]["status"] == "failed" and phases[0]["error"]
    assert Path(phases[0]["log_path"]).exists()
    assert all(phase["status"] == "passed" for phase in phases[1:])
    assert report["passed_scopes"] == [phase["scope"] for phase in phases[1:]]
    assert report["failed_scopes"] == ["runtime_boundary"]
    assert len(calls) == len(phases)
    assert report["not_requested_scopes"] == []
    assert any("scripts/run_privacy_isolation_live_eval.py" in command for command in calls)
    assert states[1]["status"] == "running_with_failures"
    assert states[1]["scopes"][0]["status"] == "failed"
    assert states[1]["scopes"][1]["status"] == "running"
    assert all(phase["status"] == "pending" for phase in states[1]["scopes"][2:])
    output = capsys.readouterr().out
    assert "本次门禁未通过" in output and "privacy_isolation_all_cases" in output
    assert "后续未运行范围" not in output


def test_missing_inquiry_report_is_failure_even_if_subprocess_returns_zero(gate, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_evals.py", "--full"])
    calls = install_recording_subprocess(monkeypatch, gate, produce_inquiry_report=False)
    assert gate.main() == 1
    _, report = only_gate_report(gate)
    inquiry = next(phase for phase in report["scopes"] if phase["scope"] == "inquiry_grounding_six_cases")
    privacy = next(phase for phase in report["scopes"] if phase["scope"] == "privacy_isolation_all_cases")
    assert inquiry["status"] == "failed" and "FileNotFoundError" in inquiry["error"]
    assert privacy["status"] == "passed"
    assert report["failed_scopes"] == [inquiry["scope"]]
    assert any("scripts/run_privacy_isolation_live_eval.py" in command for command in calls)


@pytest.mark.parametrize("target", ["owner_skipped", "privacy_skipped", "concurrency_skipped", "group_skipped", "inquiry_degraded", "owner_zero_cases"])
def test_zero_exit_code_cannot_hide_skipped_degraded_or_empty_live_reports(gate, monkeypatch, target):
    monkeypatch.setattr(sys, "argv", ["run_evals.py", "--full"])

    def alter(command, payload):
        if target.startswith("owner") and "scripts/run_real_owner_chat_live_eval.py" in command:
            if target == "owner_skipped":
                return {"status": "skipped", "reason": "no key"}
            return {**payload, "scenario_count": 0, "passed_count": 0, "reports": []}
        if target == "privacy_skipped" and "scripts/run_privacy_isolation_live_eval.py" in command:
            return {"status": "skipped", "reason": "no key"}
        if target == "concurrency_skipped" and "scripts/run_concurrency_eval.py" in command and "live" in command:
            return {**payload, "live": {"status": "skipped"}}
        if target == "group_skipped" and "scripts/run_real_group_chat_flow_eval.py" in command and "--live" in command:
            return {**payload, "summary": {"total": 2, "executed": 1, "passed": 1, "failed": 0, "skipped": 1},
                    "cases": [{"status": "passed"}, {"status": "skipped"}]}
        if target == "inquiry_degraded" and "scripts/run_inquiry_grounding_eval.py" in command:
            return {**payload, "quality_status": "degraded"}
        return payload

    calls = install_recording_subprocess(monkeypatch, gate, alter_report=alter)
    assert gate.main() == 1
    _, report = only_gate_report(gate)
    failed = [phase for phase in report["scopes"] if phase["status"] == "failed"]
    assert len(failed) == (3 if target == "group_skipped" else 1)
    assert all(phase["error"] for phase in failed)
    assert report["status"] == "failed" and failed[0]["scope"] not in report["passed_scopes"]
    assert len(calls) == len(report["passed_scopes"]) + len(failed) == len(report["scopes"])
    assert report["failed_scopes"] == [phase["scope"] for phase in failed]


def test_offline_group_skips_remain_visible_without_claiming_live_semantics(gate, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_evals.py"])

    def alter(command, payload):
        if "scripts/run_real_group_chat_flow_eval.py" in command:
            return {**payload, "summary": {"total": 2, "executed": 1, "passed": 1, "failed": 0, "skipped": 1},
                    "cases": [{"status": "passed"}, {"status": "skipped"}]}
        return payload

    install_recording_subprocess(monkeypatch, gate, alter_report=alter)
    assert gate.main() == 0
    _, report = only_gate_report(gate)
    group = next(phase for phase in report["scopes"] if phase["scope"] == "group_flow_deterministic")
    assert group["case_coverage"]["skipped"] == 1
    assert "real_group_all_scenarios" in report["not_requested_scopes"]
    assert gate.OFFLINE_SEMANTIC_NOTICE in capsys.readouterr().out


@pytest.mark.parametrize("failure", [
    "missing_judge", "judge_failed", "judge_error", "repeat_missing", "repeat_zero", "definition_missing",
    "definition_product_mismatch", "empty_golden", "golden_not_evaluated", "golden_not_passed",
    "duplicate_golden", "mismatched_golden_ids", "partial_golden", "unmapped_golden",
])
def test_owner_global_pass_cannot_hide_missing_semantic_judgments_or_partial_golden_coverage(gate, monkeypatch, failure):
    monkeypatch.setattr(sys, "argv", ["run_evals.py", "--full"])

    def alter(command, payload):
        if "scripts/run_real_owner_chat_live_eval.py" not in command:
            return payload
        if failure == "missing_judge":
            payload["reports"][0].pop("semantic_assessment")
        elif failure == "judge_failed":
            payload["reports"][0]["semantic_assessment"]["passed"] = False
        elif failure == "judge_error":
            payload["reports"][0]["semantic_assessment"]["error"] = "invalid judge response"
        elif failure == "repeat_missing":
            payload.pop("repeat")
        elif failure == "repeat_zero":
            payload["repeat"] = 0
        elif failure == "definition_missing":
            payload.pop("scenario_definition_count")
        elif failure == "definition_product_mismatch":
            payload["repeat"] = 2
        else:
            coverage = payload["golden_reply_coverage"]
            if failure == "empty_golden":
                coverage["total_cases"] = 0
            elif failure == "golden_not_evaluated":
                coverage["executed_case_ids"] = ["golden-a"]
            elif failure == "golden_not_passed":
                coverage["passed_case_ids"] = ["golden-a"]
            elif failure == "duplicate_golden":
                coverage["executed_case_ids"] = ["golden-a", "golden-a"]
            elif failure == "mismatched_golden_ids":
                coverage["passed_case_ids"] = ["golden-a", "golden-unseen"]
            elif failure == "partial_golden":
                coverage["partially_executed_case_ids"] = ["golden-a"]
            elif failure == "unmapped_golden":
                coverage["unmapped_case_ids"] = ["golden-a"]
        assert payload["status"] == "passed" and payload["semantic_judge_ran"] is True
        return payload

    calls = install_recording_subprocess(monkeypatch, gate, alter_report=alter)
    assert gate.main() == 1
    _, report = only_gate_report(gate)
    owner = next(phase for phase in report["scopes"] if phase["scope"] == "real_owner_all_scenarios_with_reply_semantics")
    assert owner["status"] == "failed" and owner["error"]
    assert owner["scope"] not in report["passed_scopes"]
    assert report["failed_scopes"] == [owner["scope"]]
    assert any("scripts/run_privacy_isolation_live_eval.py" in command for command in calls)


def test_owner_scope_accepts_exact_catalog_identity_for_every_repeat(gate, tmp_path):
    payload = successful_stub_report(["scripts/run_real_owner_chat_live_eval.py"], gate, repeat=2)
    path = tmp_path / "repeated-owner.json"
    path.write_text(json.dumps(payload))
    phase = {"scope": gate.LIVE_SCOPES[0], "report_path": str(path)}
    gate.validate_scope_report(phase)
    assert phase["case_coverage"]["total"] == 4
    assert phase["case_coverage"]["semantic_judged"] == 4
    assert phase["case_coverage"]["repeat"] == 2
    assert phase["case_coverage"]["executed_scenario_ids"] == ["scene-a", "scene-b"]


@pytest.mark.parametrize("fault", ["missing_metrics", "forged_count", "boolean_count", "missing_trace", "foreign_trace",
    "wrong_trace_identity", "omitted_error_step", "unrecovered_error", "business_check_failed", "forged_summary"])
def test_owner_recovery_must_match_original_trace_and_final_checks(gate, tmp_path, fault):
    """安全纠错的标签与计数不能替代原始拒绝证据，或掩盖最终业务失败。"""
    path = tmp_path / "recovery-owner.json"
    payload = successful_stub_report(["scripts/run_real_owner_chat_live_eval.py", "--report-path", str(path)], gate)
    first = payload["reports"][0]
    trace_path = Path(first["raw_trace_path"])
    raw = json.loads(trace_path.read_text())
    if fault == "missing_metrics":
        first.pop("recovery_metrics")
    elif fault == "forged_count":
        first["recovery_metrics"]["safe_recovered_count"] = 1
    elif fault == "boolean_count":
        first["recovery_metrics"]["safe_recovered_count"] = False
    elif fault == "missing_trace":
        trace_path.unlink()
    elif fault == "foreign_trace":
        first["raw_trace_path"] = str(tmp_path.parent / "not-this-run.json")
    elif fault == "wrong_trace_identity":
        raw["scenario_id"] = "another-case"
    elif fault in {"omitted_error_step", "unrecovered_error"}:
        raw["events"].insert(0, {"trace_id": raw["trace_id"], "step": "action_contract_error",
            "content": {"errors": ["needs_tool requires empty reply_to_user"]}, "level": "WARN"})
        if fault == "unrecovered_error":
            first["trace_steps"] = [event["step"] for event in raw["events"]]
    elif fault == "business_check_failed":
        first["checks"][0]["passed"] = False
    elif fault == "forged_summary":
        payload["recovery_summary"]["safe_recovered_count"] = 99
    if trace_path.exists():
        trace_path.write_text(json.dumps(raw))
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        gate.validate_scope_report({"scope": gate.LIVE_SCOPES[0], "report_path": str(path)})


@pytest.mark.parametrize("failure", [
    "forged_catalog_count", "missing_scene_id", "unknown_scene_id", "duplicate_scene_iteration",
    "wrong_iteration", "boolean_iteration", "incomplete_repeat", "duplicate_repeated_iteration", "forged_golden_count", "forged_golden_ids",
    "missing_criterion", "duplicate_criterion", "unknown_criterion", "false_criterion", "string_criterion_bool",
    "integer_criterion_bool", "missing_criterion_reason", "changed_human_expectation", "missing_raw_judge",
    "raw_judge_contradiction", "raw_string_violation", "raw_integer_violation", "raw_duplicate_json_key", "passed_only_judge", "timeline_claimed_live",
])
def test_owner_scope_cannot_forge_catalog_or_omit_per_criterion_evidence(gate, tmp_path, failure):
    payload = successful_stub_report(["scripts/run_real_owner_chat_live_eval.py"], gate)
    first = payload["reports"][0]
    assessment = first["semantic_assessment"]
    if failure == "forged_catalog_count":
        payload.update(scenario_count=1, passed_count=1, scenario_definition_count=1, reports=[first])
    elif failure == "missing_scene_id":
        first.pop("scenario_id")
    elif failure == "unknown_scene_id":
        first["scenario_id"] = "not_in_catalog"
    elif failure == "duplicate_scene_iteration":
        payload["reports"][1] = deepcopy(first)
    elif failure == "wrong_iteration":
        first["iteration"] = 2
    elif failure == "boolean_iteration":
        first["iteration"] = True
    elif failure == "incomplete_repeat":
        payload["repeat"] = 2
    elif failure == "duplicate_repeated_iteration":
        payload = successful_stub_report(["scripts/run_real_owner_chat_live_eval.py"], gate, repeat=2)
        payload["reports"][-1]["iteration"] = 1
    elif failure == "forged_golden_count":
        payload["golden_reply_coverage"].update(total_cases=1, executed_case_ids=["golden-a"], passed_case_ids=["golden-a"])
    elif failure == "forged_golden_ids":
        payload["golden_reply_coverage"].update(executed_case_ids=["golden-a", "invented"], passed_case_ids=["golden-a", "invented"])
    elif failure == "missing_criterion":
        assessment["criteria"].pop()
    elif failure == "duplicate_criterion":
        assessment["criteria"].append(deepcopy(assessment["criteria"][0]))
    elif failure == "unknown_criterion":
        assessment["criteria"][0]["id"] = "not_in_human_expectation"
    elif failure == "false_criterion":
        assessment["criteria"][0]["passed"] = False
    elif failure == "string_criterion_bool":
        assessment["criteria"][0]["passed"] = "true"
    elif failure == "integer_criterion_bool":
        assessment["criteria"][0]["passed"] = 1
    elif failure == "missing_criterion_reason":
        assessment["criteria"][0].pop("reason")
    elif failure == "changed_human_expectation":
        assessment["expectation"]["required_meaning"][0]["description"] = "无条件通过。"
    elif failure == "missing_raw_judge":
        assessment.pop("raw_response")
    elif failure == "raw_judge_contradiction":
        original = json.loads(assessment["raw_response"])
        original["criteria"][0]["violation_detected"] = True
        assessment["raw_response"] = json.dumps(original)
    elif failure in {"raw_string_violation", "raw_integer_violation"}:
        original = json.loads(assessment["raw_response"])
        original["criteria"][0]["violation_detected"] = "false" if failure == "raw_string_violation" else 0
        assessment["raw_response"] = json.dumps(original)
    elif failure == "raw_duplicate_json_key":
        assessment["raw_response"] = assessment["raw_response"].replace('"reason":', '"reason":"first", "reason":', 1)
    elif failure == "passed_only_judge":
        first["semantic_assessment"] = {"passed": True, "error": None}
    elif failure == "timeline_claimed_live":
        payload["golden_reply_coverage"]["timeline_assets"]["live_status"] = "passed"
    path = tmp_path / f"owner-{failure}.json"
    path.write_text(json.dumps(payload))
    phase = {"scope": gate.LIVE_SCOPES[0], "report_path": str(path)}
    with pytest.raises(ValueError):
        gate.validate_scope_report(phase)


@pytest.mark.parametrize("flags", [[], ["--live-real-owner"]])
def test_default_and_partial_modes_still_stop_after_the_first_failure(gate, monkeypatch, capsys, flags):
    monkeypatch.setattr(sys, "argv", ["run_evals.py", *flags])
    calls = install_recording_subprocess(monkeypatch, gate, fail_script="scripts/verify_agent_runtime_boundary.py")
    assert gate.main() == 1
    _, report = only_gate_report(gate)
    assert len(calls) == 1
    assert report["failed_scopes"] == ["runtime_boundary"] and report["passed_scopes"] == []
    assert all(phase["status"] == "not_run" for phase in report["scopes"][1:])
    assert "后续未运行范围" in capsys.readouterr().out


@pytest.mark.parametrize("flags", [[], ["--full"]])
def test_user_interrupt_always_stops_and_preserves_unexecuted_scopes(gate, monkeypatch, capsys, flags):
    monkeypatch.setattr(sys, "argv", ["run_evals.py", *flags])
    calls = install_recording_subprocess(monkeypatch, gate, interrupt_script="scripts/verify_agent_runtime_boundary.py")
    assert gate.main() == 1
    _, report = only_gate_report(gate)
    assert len(calls) == 1 and report["status"] == "failed"
    assert report["failed_scopes"] == ["runtime_boundary"]
    assert "KeyboardInterrupt" in report["scopes"][0]["error"]
    assert all(phase["status"] == "not_run" for phase in report["scopes"][1:])
    assert "后续未运行范围" in capsys.readouterr().out


def test_run_directories_are_unique_even_at_the_same_utc_microsecond(gate, monkeypatch):
    class FrozenDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz == dt.timezone.utc
            return cls(2026, 10, 29, 3, 4, 5, 123456, tzinfo=dt.timezone.utc)

    monkeypatch.setattr(gate.dt, "datetime", FrozenDateTime)
    first, second = gate.create_run_directory(), gate.create_run_directory()
    assert first != second and first.exists() and second.exists()
    for path in (first, second):
        assert path.parent == gate.ROOT / "runtime_data" / "eval_runs"
        assert re.fullmatch(r"20261029T030405123456Z_[a-z0-9_]+", path.name)
