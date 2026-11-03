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


def test_privacy_scope_rejects_source_less_pass_report(gate, tmp_path):
    """一条自报成功且无案例身份、无模型轨迹的报告不能充当完整隐私集合。"""
    report_path = tmp_path / "privacy_report.json"
    report_path.write_text(json.dumps({"status": "passed", "case_count": 1, "passed_count": 1,
                                      "failed_count": 0, "reports": [{"status": "passed"}]}))
    with pytest.raises(ValueError):
        gate.validate_scope_report({"scope": "privacy_isolation_all_cases", "report_path": str(report_path)})


@pytest.fixture
def privacy_scope_evidence(gate, tmp_path):
    """使用真实隐私运行链路和隔离SQLite，只以固定响应替代外部模型。"""
    report_path = tmp_path / "current-privacy-run" / "privacy_report.json"
    report_path.parent.mkdir()
    payload = successful_stub_report(["scripts/run_privacy_isolation_live_eval.py", "--report-path", str(report_path)], gate)
    phase = {"scope": "privacy_isolation_all_cases", "report_path": str(report_path)}
    gate.validate_privacy_scope_coverage(payload, phase)
    return payload, phase


def test_privacy_scope_recomputes_all_source_checks_from_actual_runtime(gate, privacy_scope_evidence):
    """十条真实源身份、主调用、审查和持久化输出全部核对后才形成全量覆盖。"""
    from run_privacy_isolation_live_eval import DEFAULT_CASES_PATH, load_privacy_probes

    payload, phase = privacy_scope_evidence
    ids = {source.case_id for source in load_privacy_probes(DEFAULT_CASES_PATH)}
    assert len(payload["reports"]) == len(ids) == 10
    assert set(phase["case_coverage"]["executed_case_ids"]) == ids
    assert phase["case_coverage"]["passed"] == 10
    assert all(item["checks"]["main_model_call_ran"] is True for item in payload["reports"])
    assert all(item["checks"]["customer_visible_review_ran"] is True for item in payload["reports"])


def test_privacy_scope_allows_new_current_conversation_memory(gate, privacy_scope_evidence):
    """当前会话后来产生的合法记忆不等于跨会话泄露，仍核对预检空记忆和私密canary。"""
    payload, phase = privacy_scope_evidence
    item = payload["reports"][0]
    path = Path(item["raw_trace_path"])
    raw = json.loads(path.read_text())
    memory = [{"conversation_id": "wechaty:contact:privacy_a", "customer_id": "privacy_a",
               "field": "smoke_preference", "value": "no_smoking", "visibility": "current_customer"}]
    for event in raw["events"]:
        if event["step"] == "context_built":
            event["content"]["task_memories"] = deepcopy(memory)
        elif event["step"] == "llm_prompt":
            request = json.loads(event["content"]["messages"][-1]["content"])
            request["task_memories"] = deepcopy(memory)
            event["content"]["messages"][-1]["content"] = json.dumps(request)
    assert raw["preflight_context"]["task_memories"] == []
    path.write_text(json.dumps(raw))
    gate.validate_privacy_scope_coverage(payload, phase)
    assert phase["case_coverage"]["passed"] == 10


@pytest.mark.parametrize("fault", [
    "partial", "duplicate", "unknown", "changed_input", "missing_check", "numeric_check", "failed_check",
    "missing_trace", "outside_trace", "stale_run", "wrong_trace_identity", "wrong_raw_input", "wrong_context", "private_actual_memory",
    "missing_main_prompt", "missing_main_response", "empty_main_response", "missing_action", "changed_action",
    "missing_review_prompt", "missing_review_response", "missing_review_result", "rejected_raw_review", "review_other_text",
    "missing_final", "changed_final", "hidden_outgoing_leak", "false_call_count", "false_total_calls",
])
def test_privacy_scope_rejects_partial_stale_or_forged_execution_evidence(gate, privacy_scope_evidence, tmp_path, fault):
    """不信任汇总通过：删轨迹、改案例、隐藏泄露或缺少真实调用均须阻断全量门禁。"""
    import sqlite3

    payload, phase = privacy_scope_evidence
    item = payload["reports"][0]
    path = Path(item["raw_trace_path"])
    raw = json.loads(path.read_text())
    if fault == "partial":
        payload["reports"].pop()
        payload["case_count"] = payload["passed_count"] = len(payload["reports"])
    elif fault == "duplicate":
        payload["reports"][-1] = deepcopy(item)
    elif fault == "unknown":
        item["case_id"] = "fake-privacy-source"
    elif fault == "changed_input":
        item["input"] = "改为更容易回答的请求"
    elif fault == "missing_check":
        item["checks"].pop("no_customer_visible_leak")
    elif fault == "numeric_check":
        item["checks"]["no_customer_visible_leak"] = 1
    elif fault == "failed_check":
        item["checks"]["no_customer_visible_leak"] = False
    elif fault == "missing_trace":
        item.pop("raw_trace_path")
    elif fault == "outside_trace":
        old = tmp_path / "previous-run-trace.json"
        old.write_text(path.read_text())
        item["raw_trace_path"] = str(old)
    elif fault == "stale_run":
        item["run_id"] = raw["run_id"] = "previous-run"
    elif fault == "wrong_trace_identity":
        raw["events"][0]["trace_id"] = "another-case-trace"
    elif fault == "wrong_raw_input":
        raw["message"]["text"] = "另一条输入"
    elif fault == "wrong_context":
        raw["preflight_context"]["current_message"]["text"] = "另一条输入"
    elif fault == "private_actual_memory":
        next(event for event in raw["events"] if event["step"] == "context_built")["content"]["task_memories"] = [
            {"evidence": f"B_PRIVATE_CANARY_{item['case_id']}_91D7"}]
    elif fault.startswith("missing_"):
        missing = {"missing_main_prompt": "llm_prompt", "missing_main_response": "llm_response",
            "missing_action": "action_proposed", "missing_review_prompt": "customer_visible_content_review_prompt",
            "missing_review_response": "customer_visible_content_review_response",
            "missing_review_result": "customer_visible_content_review_result", "missing_final": "final_output"}[fault]
        raw["events"] = [event for event in raw["events"] if event["step"] != missing]
        item["trace_steps"] = [event["step"] for event in raw["events"]]
    elif fault == "empty_main_response":
        next(event for event in raw["events"] if event["step"] == "llm_response")["content"]["content"] = ""
    elif fault == "changed_action":
        next(event for event in raw["events"] if event["step"] == "action_proposed")["content"]["objective_status"] = "invalid-status"
    elif fault == "rejected_raw_review":
        response = next(event for event in raw["events"] if event["step"] == "customer_visible_content_review_response")
        parsed = json.loads(response["content"]["content"])
        parsed["item_reviews"][0]["approved"] = False
        response["content"]["content"] = json.dumps(parsed)
    elif fault == "review_other_text":
        prompt = next(event for event in raw["events"] if event["step"] == "customer_visible_content_review_prompt")
        request = json.loads(prompt["content"]["messages"][-1]["content"])
        request["review_items"][0]["text"] = "审查的是另一句话"
        prompt["content"]["messages"][-1]["content"] = json.dumps(request)
    elif fault == "changed_final":
        next(event for event in raw["events"] if event["step"] == "final_output")["content"]["reply"] = "另一句对客回复"
    elif fault == "hidden_outgoing_leak":
        with sqlite3.connect(item["db_path"]) as connection:
            connection.execute("INSERT INTO runtime_outbound_message_drafts VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("hidden-leak", "wechaty:contact:privacy_a", "privacy_a", "wechat", "pending",
                 json.dumps({"message_text": "A上次催得太急"}), "2026-11-02"))
    elif fault == "false_call_count":
        item["model_call_count"] += 1
    elif fault == "false_total_calls":
        payload["total_model_calls"] += 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        gate.validate_privacy_scope_coverage(payload, phase)


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


def semantic_config_stub(trace_id, raw, *, passed=True, evidence=None):
    """离线只替换HTTP传输，真实客户端生成固定配置诊断；不宣称真实远端语义通过。"""
    from mahjong_agent_runtime import AgentLLMConfig
    from mahjong_agent_runtime.llm import completion_diagnostics
    from run_inquiry_grounding_eval import (
        SEMANTIC_JUDGE_CONTRACT_VERSION, SEMANTIC_JUDGE_FIXED_CONFIG, make_semantic_judge_config,
        resolve_semantic_judge_transport,
    )
    from tests.semantic_http_stub import semantic_http_completion

    transport = resolve_semantic_judge_transport(180, retry_attempts=1)
    config = make_semantic_judge_config(AgentLLMConfig(api_key="unit-only", model="unit-model", provider="deepseek",
        base_url="https://example.invalid", retry_attempts=1), judge_transport=transport)
    response = semantic_http_completion(raw, config=config, trace_id=trace_id)
    from semantic_judge_context import project, PROJECTION_VERSION
    evidence = deepcopy(evidence or {})
    return {"evidence": evidence, "judge_evidence": project(evidence), "judge_input_projection_version": PROJECTION_VERSION,
        "judge_contract_version": SEMANTIC_JUDGE_CONTRACT_VERSION,
        "judge_config": {"model": "unit-model", "provider": "deepseek", **deepcopy(SEMANTIC_JUDGE_FIXED_CONFIG),
            "attempt_timeout_seconds": config.attempt_timeout_seconds, "retry_attempts": config.retry_attempts},
        "judge_transport": transport,
        "contract_repair_count": 0,
        "attempts": [{"trace_id": trace_id, "raw_response": str(response), "error": None, "passed": passed,
            "timeout_seconds": transport["total_timeout_seconds"], "completion_diagnostics": completion_diagnostics(response)}]}


def successful_inquiry_stub_report():
    """用真实固定输入与生效期望构造完整报告形状，仅模拟独立判定输出，不调用模型。"""
    from run_inquiry_grounding_eval import (
        inquiry_business_side_effect_checks, inquiry_input_catalog, inquiry_reply_expectation, _parse_semantic_assessment,
    )

    inputs = inquiry_input_catalog()
    transcript = []
    for index, (case_id, source) in enumerate(inputs.items(), start=1):
        expectation = inquiry_reply_expectation(source["criteria_incomplete"], case_id=case_id)
        raw = json.dumps({"reason": "固定离线判定，不代表真实模型语义通过。", "criteria": [
            {"id": item["id"], "violation_detected": False, "reason": "本地测试判定器证据接线。"}
            for field in ("required_meaning", "forbidden_meaning") for item in expectation[field]
        ]}, ensure_ascii=False)
        criterion_ids = [item["id"] for field in ("required_meaning", "forbidden_meaning") for item in expectation[field]]
        preferred_game = source["preferred_game"]
        reply = "测试回复，不用于业务验收。"
        before = {"customer_related_games": [], "customer_invites": [], "customer_outbounds": [],
                  "domain_counts": {"games": len(source["fixture_game_types"]), "invites": 0, "outbounds": 0, "reservations": 0}}
        business_state = {"before": before, "after": deepcopy(before)}
        evidence = {"request": source["text"], "reply": reply,
                    "known_profile": {"preferred_games": [preferred_game] if preferred_game else [],
                                      "preferred_stakes": ["1"] if preferred_game else []},
                    "public_search_results": [], "tool_results": [],
                    "persisted_state": {"business_state_before": before, "business_state_after": business_state["after"]}}
        transcript.append({
            "sequence": index, "thread_id": case_id, "conversation_id": "inquiry:" + case_id,
            "user": {"text": source["text"], "generation": {"context": {"case": case_id}}},
            "agent": {"reply": reply, "status": "waiting_user"}, "tool_results": [],
            "business_state": business_state,
            "business_side_effect_checks": inquiry_business_side_effect_checks([], business_state),
            "inquiry_assessment": {**_parse_semantic_assessment(raw, criterion_ids), "error": None,
                                   "raw_response": raw, "expectation": expectation, "evidence": evidence,
                                   **semantic_config_stub("inquiry-stub:" + case_id, raw, evidence=evidence)},
            "product_contract_revision_ids": list(expectation["product_contract_revision_ids"]),
        })
    return {"status": "completed", "quality_status": "passed", "total_messages": len(inputs),
            "configuration": {"cases": list(inputs)}, "transcript": transcript,
            "checks": {**dict.fromkeys(inputs, True), "annotated_original_reply_rejected": True},
            "reviewer_probe": {"approved": False, "needs_human": False, "reasoning_summary": "固定测试拒绝。"}}


def active_revision_semantic_stub(evidence, expectation, *, failed_ids=(), trace_id="trace_active_revision_semantics"):
    """生成严格原始JSON与派生结论的受控替身，标签不作为生产模型语义证据。"""
    from run_inquiry_grounding_eval import REPLY_SEMANTIC_JUDGE_RUBRIC, _project_judge_expectation

    criteria = [{"id": item["id"], "passed": item["id"] not in failed_ids, "reason": "本地受控判定标签"}
        for group in ("required_meaning", "forbidden_meaning") for item in expectation[group]]
    reason = "本地接线替身，不调用真实模型"
    result = {"passed": all(item["passed"] for item in criteria), "error": None, "reason": reason,
        "criteria": criteria, "expectation": deepcopy(expectation), "evidence": deepcopy(evidence),
        "raw_response": json.dumps({"reason": reason, "criteria": [
            {"id": item["id"], "violation_detected": not item["passed"], "reason": item["reason"]}
            for item in criteria]}, ensure_ascii=False)}
    result.update(trace_id=trace_id, rubric=REPLY_SEMANTIC_JUDGE_RUBRIC,
        judge_expectation=_project_judge_expectation(expectation),
        **semantic_config_stub(trace_id, result["raw_response"], passed=result["passed"], evidence=evidence))
    return result


def active_reply_state_stub(evidence, act, *, trace_id):
    """固定盲分类结果供门禁连线测试，状态不进入分类输入或原始输出。"""
    from reply_state_eval import COMMUNICATION_ACT_RUBRIC, project_reply_state_input, reply_state_input_digest

    projected = project_reply_state_input(evidence)
    reason = "本地沟通行为标签，不代表真实模型分类通过"
    raw = json.dumps({"communication_act": act, "reason": reason}, ensure_ascii=False)
    return {"communication_act": act, "reason": reason, "error": None,
        "input": projected, "input_sha256": reply_state_input_digest(projected), "raw_response": raw,
        "rubric": COMMUNICATION_ACT_RUBRIC, "trace_id": trace_id, "contract_repair_count": 0,
        "judge_config": {"temperature": 0, "thinking_mode": "disabled", "max_tokens": 900, "reasoning_effort": None},
        "attempts": [{"trace_id": trace_id, "raw_response": raw, "error": None, "communication_act": act}]}


@pytest.mark.parametrize("fault", [None, "missing_version", "old_config", "wrong_effort", "wrong_budget", "wrong_format",
    "missing_http", "empty_http", "http_disabled", "http_budget", "http_provider", "http_model", "http_length",
    "missing_transport", "short_attempt", "wrong_actual_timeout", "wrong_retry_budget", "default_source_wrong_budget"])
def test_general_semantic_gate_requires_fixed_reasoned_version_and_original_http_request(gate, fault):
    """新配置自报正确仍须核对实际请求；金额和状态盲分类不使用此复杂语义合同。"""
    expectation = {"required_meaning": [{"id": "fact", "description": "只说明已有事实"}], "forbidden_meaning": []}
    assessment = active_revision_semantic_stub({"reply": "有现成局。"}, expectation)
    if fault == "missing_version":
        assessment.pop("judge_contract_version")
    elif fault in {"old_config", "wrong_effort", "wrong_budget", "wrong_format"}:
        key, value = {"old_config": ("thinking_mode", "disabled"), "wrong_effort": ("reasoning_effort", None),
            "wrong_budget": ("max_tokens", 4000), "wrong_format": ("response_format", "text")}[fault]
        assessment["judge_config"][key] = value
    elif fault in {"missing_http", "empty_http"}:
        if fault == "missing_http":
            assessment["attempts"][0].pop("completion_diagnostics")
        else:
            assessment["attempts"][0]["completion_diagnostics"] = {}
    elif fault == "missing_transport":
        assessment.pop("judge_transport")
    elif fault == "short_attempt":
        assessment["judge_transport"]["attempt_timeout_seconds"] = 60.0
    elif fault == "wrong_actual_timeout":
        assessment["attempts"][0]["timeout_seconds"] = 60.0
    elif fault == "wrong_retry_budget":
        assessment["judge_transport"]["retry_attempts"] = 0
    elif fault == "default_source_wrong_budget":
        assessment["judge_transport"].update(budget_source="default", total_timeout_seconds=90.0, attempt_timeout_seconds=90.0)
    elif fault is not None:
        diagnostic = assessment["attempts"][0]["completion_diagnostics"]
        if fault == "http_disabled":
            diagnostic["request"]["thinking_mode"] = "disabled"
        elif fault == "http_budget":
            diagnostic["request"]["max_tokens"] = 4000
        elif fault == "http_provider":
            diagnostic["provider"] = "other"
        elif fault == "http_model":
            diagnostic["requested_model"] = "another-model"
        elif fault == "http_length":
            diagnostic["attempts"][-1]["content"]["length"] += 1
    if fault is None:
        gate._validate_reported_semantic_assessment(assessment, expectation, scope="owner")
    else:
        with pytest.raises(ValueError, match="configuration or actual request"):
            gate._validate_reported_semantic_assessment(assessment, expectation, scope="owner")


def active_revision_calibration_stub(fixture, evidence):
    """四个固定对照保留盲分类及后端映射，标签从不进入分类模型输入。"""
    from run_concurrency_eval import load_query_reply_state_calibration_cases

    cases = []
    for source in load_query_reply_state_calibration_cases():
        controlled = deepcopy(evidence)
        controlled["reply"] = source["reply"]
        controlled["runtime_status"] = source["runtime_status"]
        controlled["execution_lifecycle"]["turn"]["status"] = source["runtime_status"]
        matching_status = source["runtime_status"] if source["expected_passed"] else (
            "waiting_user" if source["runtime_status"] == "completed" else "completed")
        act = "inform_only" if matching_status == "completed" else "requires_user_response"
        assessment = active_reply_state_stub(controlled, act, trace_id="trace_active_revision_state_calibration:" + source["id"])
        cases.append({"case_id": source["id"], "source": deepcopy(source), "evidence": controlled,
                      "assessment": assessment, "passed": True})
    return {"status": "passed", "test_kind": "controlled_reply_state_calibration",
            "source_evidence": deepcopy(evidence), "cases": cases}


def successful_stub_report(command, gate, *, repeat=1):
    if "scripts/run_reply_scope_calibration.py" in command:
        from tests.test_reply_scope_calibration import build_scope_stub

        root = Path(command[command.index("--report-path") + 1]).parent
        root.mkdir(parents=True, exist_ok=True)
        payload, _ = build_scope_stub(root / f"reply_scope_stub_{uuid.uuid4().hex}")
        return payload
    if "--calibration-only" in command:
        from tests.test_reply_semantic_calibration import build_calibration_stub

        root = Path(command[command.index("--report-path") + 1]).parent
        root.mkdir(parents=True, exist_ok=True)
        payload, _ = build_calibration_stub(root / f"semantic_calibration_stub_{uuid.uuid4().hex}")
        return payload
    if "scripts/run_real_owner_chat_live_eval.py" in command:
        from run_real_owner_chat_live_eval import assess_action_contract_recovery, summarize_action_contract_recovery
        from reply_money_eval import check_reply_money_facts

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
                query_request = contract.get("query_reply_state_requests", {}).get(scenario_id)
                reply = "固定查询结果，你打吗？" if query_request is not None else "固定测试回复"
                events = []
                state_assessment = None
                if query_request is not None:
                    events.append({"trace_id": trace_id, "step": "context_built", "content": {
                        "current_message": deepcopy(query_request), "recent_conversation": []}, "level": "INFO"})
                    state_assessment = active_reply_state_stub({"request": query_request, "reply": reply, "recent_conversation": []},
                        "requires_user_response", trace_id=f"{trace_id}_query_reply_state")
                events.append({"trace_id": trace_id, "step": "final_output", "content": {"reply": reply, "objective_status": "waiting_user"}, "level": "INFO"})
                if state_assessment is not None:
                    events.append({"trace_id": trace_id, "step": "query_reply_state_assessment", "content": state_assessment, "level": "INFO"})
                # 无金额替身仍走同一证据API，不能用手写通过标志绕过新增门禁。
                money_check = check_reply_money_facts(None, reply=reply, tool_results=[],
                    trace_id=f"{trace_id}_money_fact_check")
                events.append({"trace_id": trace_id, "step": "reply_money_fact_check", "content": money_check, "level": "INFO"})
                report_path = Path(command[command.index("--report-path") + 1]) if "--report-path" in command else gate.ROOT / "stub_report.json"
                trace_root = report_path.parent / "stub_traces"
                trace_root.mkdir(parents=True, exist_ok=True)
                trace_path = trace_root / f"{uuid.uuid4().hex}.json"
                trace_path.write_text(json.dumps({"trace_id": trace_id, "scenario_id": scenario_id,
                    "iteration": iteration, "events": events}))
                recovery = assess_action_contract_recovery([SimpleNamespace(**event) for event in events],
                    runtime_status="waiting_user", final_reply=reply, business_checks_passed=True, semantic_passed=True)
                reports.append({
                    "status": "passed", "scenario_id": scenario_id, "iteration": iteration,
                    "trace_id": trace_id, "runtime_status": "waiting_user", "final_reply": reply,
                    "trace_steps": [event["step"] for event in events], "raw_trace_path": str(trace_path),
                    "checks": [{"name": "unit_stub_business_check", "passed": True}], "recovery_metrics": recovery,
                    "money_fact_check": money_check,
                    "requires_query_reply_state_check": query_request is not None,
                    **({"query_reply_state_assessment": state_assessment, "input": deepcopy(query_request)} if query_request is not None else {}),
                    "semantic_assessment": {
                        "reason": original["reason"], "criteria": criteria, "passed": True, "error": None,
                        "expectation": deepcopy(contract["expectations"][scenario_id]),
                        "raw_response": json.dumps(original, ensure_ascii=False),
                        **semantic_config_stub(trace_id + "_semantic_judge", json.dumps(original, ensure_ascii=False), evidence={"reply":reply}),
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
        from mahjong_agent_runtime import AgentAction
        from run_privacy_isolation_live_eval import DEFAULT_CASES_PATH, load_privacy_probes, run_case

        class FixedPrivacyClient:
            """仅替换模型响应，保留真实隐私运行器、SQLite、规划和审查轨迹。"""

            def complete(self, messages, **kwargs):
                request = json.loads(messages[-1]["content"])
                if "review_items" in request:
                    return json.dumps({"approved": True, "needs_human": False, "reasoning_summary": "离线审查替身",
                        "violations": [], "item_reviews": [{"item_id": item["item_id"], "approved": True,
                            "suggested_safe_text": item["text"], "reasoning_summary": "仅验证证据接线", "violations": []}
                            for item in request["review_items"]]})
                if "source_text" in request:
                    return json.dumps({"text": request["source_text"]})
                return json.dumps(AgentAction(goal="保护客户隐私", objective_status="completed", reasoning_summary="不透露其他客户信息",
                                             reply_to_user="别人的私事我不方便说。", stop_reason={"can_stop": True, "why": "已答复",
                                                 "pending_work": [], "depends_on_tool_results": False}).to_dict(), ensure_ascii=False)

        report_path = Path(command[command.index("--report-path") + 1])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        db_path = report_path.parent / f"privacy_stub_{uuid.uuid4().hex}.sqlite3"
        args = SimpleNamespace(db_path=db_path, report_path=report_path, max_tokens_per_call=64000,
            max_calls_per_turn=8, max_steps=4, timeout_seconds=1)
        reports = [run_case(FixedPrivacyClient(), args, source) for source in load_privacy_probes(DEFAULT_CASES_PATH)]
        return {"status": "passed", "case_count": len(reports), "passed_count": len(reports), "failed_count": 0,
                "reports": reports, "total_model_calls": sum(item["model_call_count"] for item in reports)}
    if "scripts/run_real_group_chat_flow_eval.py" in command:
        return {"mode": "live" if "--live" in command else "deterministic",
                "summary": {"total": 1, "executed": 1, "passed": 1, "failed": 0, "skipped": 0},
                "cases": [{"case_id": "stub", "status": "passed"}]}
    if "scripts/run_concurrency_eval.py" in command:
        from run_concurrency_eval import (
            active_business_semantic_expectation, load_active_revision_fixture, load_customer_visible_review_contract_cases, load_follow_up_commitment_review_cases,
            run_live_customer_visible_review_contract,
        )
        from mahjong_agent_runtime import AgentLLMConfig
        fixture = load_active_revision_fixture()
        from run_real_owner_chat_live_eval import summarize_action_contract_recovery

        owner_reports = successful_stub_report(["scripts/run_real_owner_chat_live_eval.py", "--report-path",
            command[command.index("--report-path") + 1]], gate, repeat=2)["reports"]
        recovery_evidence = deepcopy(owner_reports[0])
        recovery_evidence["scenario_id"] = fixture["id"]
        recovery_evidence["checks"] = [{"name": name, "passed": True} for name in fixture["expected"]["hard_checks"]]
        raw_path = Path(recovery_evidence["raw_trace_path"])
        raw = json.loads(raw_path.read_text())
        raw["scenario_id"] = fixture["id"]
        # 输入修订只复用纠错审计，不继承另一老板场景的金额校验或覆盖声明。
        recovery_evidence.pop("money_fact_check")
        raw["events"] = [event for event in raw["events"] if event["step"] != "reply_money_fact_check"]
        recovery_evidence["trace_steps"] = [event["step"] for event in raw["events"]]
        revision_trace = raw_path.with_name(f"{uuid.uuid4().hex}.json")
        revision_trace.write_text(json.dumps(raw))
        recovery_evidence["raw_trace_path"] = str(revision_trace)
        evidence = {"request": {"conversation_id": fixture["conversation_id"], "sender_id": fixture["customer_id"],
                **fixture["messages"][-1]}, "reply": recovery_evidence["final_reply"], "runtime_status": "waiting_user",
            "execution_lifecycle": {"turn": {"status": "waiting_user", "scope": "synchronous_turn_only"}},
            "tool_results": [], "persisted_state": {}}
        assessment = active_revision_semantic_stub(evidence, active_business_semantic_expectation(fixture["expected"]))
        calibration = active_revision_calibration_stub(fixture, evidence)
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
                    "product_contract_revision_ids": list(fixture["product_contract_revision_ids"]),
                    "reply_state_calibration": calibration,
                    "reply_state_assessment": active_reply_state_stub(evidence, "requires_user_response", trace_id="trace_active_revision_communication"), "checks": recovery_evidence["checks"], "recovery_evidence": recovery_evidence,
                    "old_result": {"status": "superseded", "final_reply": ""},
                    "latest_result": {"status": "waiting_user", "final_reply": recovery_evidence["final_reply"],
                        "trace_id": recovery_evidence["trace_id"]}, "semantic_assessment": assessment}},
        }
    if "scripts/run_inquiry_grounding_eval.py" in command:
        return successful_inquiry_stub_report()
    raise AssertionError(f"unexpected JSON report command: {command}")


@pytest.mark.parametrize("fault", ["missing", "changed_fixture", "wrong_id", "missing_check", "numeric_flag",
                                       "missing_judge", "raw_violation", "old_reply", "invalid_latest_status", "missing_recovery", "missing_state_assessment", "missing_act", "mismatch_act", "missing_revision_ids"])
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
        elif fault == "invalid_latest_status":
            revision["latest_result"]["status"] = "needs_tool"
        elif fault == "missing_revision_ids":
            revision.pop("product_contract_revision_ids")
        elif fault == "missing_state_assessment":
            revision.pop("reply_state_assessment")
        elif fault == "missing_act":
            revision["reply_state_assessment"].pop("communication_act")
        elif fault == "mismatch_act":
            assessment = revision["reply_state_assessment"]
            assessment["communication_act"] = "inform_only"
            assessment["raw_response"] = json.dumps({"communication_act": "inform_only", "reason": assessment["reason"]})
            assessment["attempts"][0].update(raw_response=assessment["raw_response"], communication_act="inform_only")
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


def command_for(calls, script, *, mode=None, calibration=False):
    found = [command for command in calls if script in command and ("--calibration-only" in command) is calibration
             and (mode is None or command[command.index("--mode") + 1] == mode)]
    assert len(found) == 1
    return found[0]


def test_owner_source_contract_loads_real_catalog_and_all_golden_fragments():
    spec = importlib.util.spec_from_file_location("eval_gate_real_owner_contract_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = module.load_owner_semantic_contract()
    assert len(contract["scenario_ids"]) == 35
    assert len(set(contract["scenario_ids"])) == 35
    assert len(contract["golden_case_ids"]) == 37
    assert "synthetic_live_casual_operational_discussion_boundary" in contract["golden_case_ids"]
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
    calibration = command_for(calls, "scripts/run_real_owner_chat_live_eval.py", calibration=True)
    assert "--strict" in calibration and "--scenario" not in calibration
    assert next(phase for phase in report["scopes"] if phase["scope"] == "reply_semantic_calibration")["case_coverage"]["total"] == 8
    scope_calibration = command_for(calls, "scripts/run_reply_scope_calibration.py")
    assert "--strict" in scope_calibration and "--scenario" not in scope_calibration
    assert next(phase for phase in report["scopes"] if phase["scope"] == "reply_scope_calibration")["case_coverage"]["total"] == 4
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
        "tests/test_reply_semantic_calibration.py", "tests/test_reply_scope_calibration.py",
        "tests/test_reply_scope_calibration_revisions.py",
        "tests/test_semantic_judge_context.py", "tests/test_count_query_presentation_policy.py", "tests/test_eval_contract_consistency.py",
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


def test_inquiry_scope_accepts_actual_report_shape_and_effective_product_revisions(gate, tmp_path):
    """真实运行器形状的六条报告按同一生效合同通过，未知与已知场景均不得漏验。"""
    payload = successful_inquiry_stub_report()
    report_path = tmp_path / "inquiry-report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False))
    phase = {"scope": gate.LIVE_SCOPES[3], "report_path": str(report_path)}
    gate.validate_scope_report(phase)
    assert phase["case_coverage"]["semantic_judged"] == 6
    assert phase["case_coverage"]["executed_case_ids"] == sorted(payload["configuration"]["cases"])
    assert payload["transcript"][0]["product_contract_revision_ids"]
    assert payload["transcript"][-1]["product_contract_revision_ids"] == []


@pytest.mark.parametrize("fault", [
    "missing_transcript", "missing_case", "duplicate_case", "unknown_case", "changed_input", "wrong_input_mapping",
    "missing_source_cases", "changed_configuration", "missing_assessment", "changed_expectation", "missing_revision_ids",
    "changed_revision_ids", "missing_case_check", "missing_probe_check", "missing_probe", "probe_approved",
    "missing_raw", "raw_contradiction", "missing_criterion", "numeric_passed", "judge_error",
    "changed_judge_input", "changed_judge_reply", "changed_judge_profile", "failed_runtime",
    "missing_business_state", "missing_side_effect_checks", "changed_state_evidence", "changed_tool_evidence",
    "changed_search_evidence", "hidden_write_tool", "hidden_state_change",
])
def test_inquiry_global_pass_cannot_hide_source_contract_or_raw_judgment_drift(gate, fault):
    """汇总仍全绿时，错题、漏跑、过期合同及伪造逐项判定都必须阻断咨询范围。"""
    payload = successful_inquiry_stub_report()
    first = payload["transcript"][0]
    assessment = first["inquiry_assessment"]
    if fault == "missing_transcript":
        payload.pop("transcript")
    elif fault == "missing_case":
        payload["transcript"].pop()
    elif fault == "duplicate_case":
        payload["transcript"][-1] = deepcopy(first)
    elif fault == "unknown_case":
        first["thread_id"] = "case-not-in-source"
    elif fault == "changed_input":
        first["user"]["text"] = "替换成另一条输入"
    elif fault == "wrong_input_mapping":
        first["user"]["generation"]["context"]["case"] = "unknown_pool"
    elif fault == "missing_source_cases":
        payload.pop("configuration")
    elif fault == "changed_configuration":
        payload["configuration"]["cases"][0] = "wrong-case"
    elif fault == "missing_assessment":
        first.pop("inquiry_assessment")
    elif fault == "changed_expectation":
        assessment["expectation"]["required_meaning"][0]["description"] = "只要回复就通过。"
    elif fault == "missing_revision_ids":
        first.pop("product_contract_revision_ids")
    elif fault == "changed_revision_ids":
        first["product_contract_revision_ids"] = []
    elif fault == "missing_case_check":
        payload["checks"].pop(first["thread_id"])
    elif fault == "missing_probe_check":
        payload["checks"].pop("annotated_original_reply_rejected")
    elif fault == "missing_probe":
        payload.pop("reviewer_probe")
    elif fault == "probe_approved":
        payload["reviewer_probe"]["approved"] = True
    elif fault == "missing_raw":
        assessment.pop("raw_response")
    elif fault == "raw_contradiction":
        raw = json.loads(assessment["raw_response"])
        raw["criteria"][0]["violation_detected"] = True
        assessment["raw_response"] = json.dumps(raw)
    elif fault == "missing_criterion":
        assessment["criteria"].pop()
    elif fault == "numeric_passed":
        assessment["criteria"][0]["passed"] = 1
    elif fault == "judge_error":
        assessment["error"] = "模型判定失败"
    elif fault == "changed_judge_input":
        assessment["evidence"]["request"] = "判了另一条问题"
    elif fault == "changed_judge_reply":
        assessment["evidence"]["reply"] = "判了另一条回复"
    elif fault == "changed_judge_profile":
        assessment["evidence"]["known_profile"]["preferred_games"] = ["invented_game"]
    elif fault == "missing_business_state":
        first.pop("business_state")
    elif fault == "missing_side_effect_checks":
        first.pop("business_side_effect_checks")
    elif fault == "changed_state_evidence":
        assessment["evidence"]["persisted_state"] = {}
    elif fault == "changed_tool_evidence":
        assessment["evidence"]["tool_results"] = [{"name": "unexpected_tool"}]
    elif fault == "changed_search_evidence":
        assessment["evidence"]["public_search_results"] = [{"matches": []}]
    elif fault == "hidden_write_tool":
        from run_inquiry_grounding_eval import project_inquiry_tool_results
        first["tool_results"] = [{"name": "create_game", "called": True, "allowed": True, "error": None, "result": {}}]
        assessment["evidence"]["tool_results"] = project_inquiry_tool_results(first["tool_results"])
    elif fault == "hidden_state_change":
        first["business_state"]["after"]["customer_related_games"].append({"game_id": "unauthorized"})
    else:
        first["agent"]["status"] = "needs_help"
    with pytest.raises(ValueError):
        gate.validate_inquiry_scope_coverage(payload, {})


def test_inquiry_gate_accepts_real_read_results_with_the_runner_privacy_projection(gate):
    """真实本地查局结果按运行器相同公开投影送审，原始内部字段不要求进入判定器。"""
    from mahjong_agent_runtime import InMemoryAgentStore, ToolCall
    from mahjong_agent_runtime.domains.tools.search_tools import search_current_games
    from run_inquiry_grounding_eval import (
        inquiry_business_side_effect_checks, inquiry_business_state, inquiry_input_catalog, project_inquiry_tool_results,
    )

    payload = successful_inquiry_stub_report()
    entry = next(item for item in payload["transcript"] if item["thread_id"] == "unknown_pool")
    store = InMemoryAgentStore()
    for index, game_type in enumerate(inquiry_input_catalog()["unknown_pool"]["fixture_game_types"]):
        store.create_game(conversation_id=f"fixture:{index}", organizer_id=f"owner:{index}", organizer_name="发起人",
            requirement={"game_type": game_type, "stake": "1", "smoke_preference": "any", "start_time_kind": "asap_when_full"},
            known_players=[{"customer_id": f"owner:{index}", "display_name": "发起人", "seat_count": 3}], trace_id="unit-fixture")
    before = inquiry_business_state(store, customer_id="customer", conversation_id=entry["conversation_id"])
    read = search_current_games(store, ToolCall(name="search_current_games", arguments={"requirement": {}}),
        "unit-read", entry["conversation_id"], "customer", "需求测试客户")
    entry["tool_results"] = [read.to_dict()]
    entry["business_state"] = {"before": before,
        "after": inquiry_business_state(store, customer_id="customer", conversation_id=entry["conversation_id"])}
    entry["business_side_effect_checks"] = inquiry_business_side_effect_checks(entry["tool_results"], entry["business_state"])
    tools = project_inquiry_tool_results(entry["tool_results"])
    assert tools[0]["result"] != read.result
    entry["inquiry_assessment"]["evidence"].update(tool_results=tools, public_search_results=[tools[0]["result"]],
        persisted_state={"business_state_before": before, "business_state_after": entry["business_state"]["after"]})
    from semantic_judge_context import project
    entry["inquiry_assessment"]["judge_evidence"] = project(entry["inquiry_assessment"]["evidence"])
    phase = {}
    gate.validate_inquiry_scope_coverage(payload, phase)
    assert phase["case_coverage"]["passed"] == 6


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
        if target.startswith("owner") and "scripts/run_real_owner_chat_live_eval.py" in command and "--calibration-only" not in command:
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
        if "scripts/run_real_owner_chat_live_eval.py" not in command or "--calibration-only" in command:
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


@pytest.mark.parametrize("status,reply", [
    ("completed", "明晚七点有个国标2块三缺一的。"),
    ("waiting_user", "明晚七点有个国标2块三缺一的，你打吗？"),
])
def test_active_revision_gate_accepts_both_coherent_reply_states_with_complete_evidence(gate, tmp_path, status, reply):
    """仅两种状态的独立一致性结论可通过，原raw判定、隔离及四例校准仍完整验收。"""
    path = tmp_path / "coherent-concurrency.json"
    payload = successful_stub_report(["scripts/run_concurrency_eval.py", "--report-path", str(path)], gate)
    revision = payload["live"]["active_input_revision"]
    revision["latest_result"].update(status=status, final_reply=reply)
    recovery = revision["recovery_evidence"]
    recovery.update(runtime_status=status, final_reply=reply)
    raw_path = Path(recovery["raw_trace_path"])
    raw = json.loads(raw_path.read_text())
    next(event for event in raw["events"] if event["step"] == "final_output")["content"]["reply"] = reply
    raw_path.write_text(json.dumps(raw))
    evidence = revision["semantic_assessment"]["evidence"]
    evidence.update(runtime_status=status, reply=reply)
    evidence["execution_lifecycle"]["turn"]["status"] = status
    from run_concurrency_eval import active_business_semantic_expectation
    revision["semantic_assessment"] = active_revision_semantic_stub(evidence, active_business_semantic_expectation(revision["fixture"]["expected"]))
    revision["reply_state_assessment"] = active_reply_state_stub(evidence,
        "inform_only" if status == "completed" else "requires_user_response", trace_id="trace_active_revision_communication")
    revision["reply_state_calibration"] = active_revision_calibration_stub(revision["fixture"], evidence)
    path.write_text(json.dumps(payload))
    phase = {"scope": "real_concurrency_all_scenarios", "report_path": str(path),
             "command": ["scripts/run_concurrency_eval.py", "--live-repeats", "2"]}
    gate.validate_scope_report(phase)
    assert phase["case_coverage"]["active_input_revision"]["passed"] == 1


@pytest.mark.parametrize("fault", [
    "actual_reply", "actual_status", "actual_request", "actual_turn_status", "missing_calibration",
    "missing_case", "duplicate_case", "changed_source", "changed_baseline", "changed_controlled_facts",
    "borrowed_assessment", "missing_act", "negative_false_pass", "negative_other_act", "invalid_act_type",
    "changed_rubric", "changed_blind_input", "changed_temperature", "changed_thinking", "missing_attempts",
    "changed_attempt_raw", "changed_attempt_trace", "changed_repair_count", "valid_judgement_retried", "wrong_repair_identity",
])
def test_active_revision_gate_rejects_missing_or_unbound_reply_state_calibration(gate, tmp_path, fault):
    """受控校准必须基于本轮真实输入，仅改三处；错误标签、自报通过与借用判定均拒绝。"""
    path = tmp_path / "calibration-concurrency.json"
    payload = successful_stub_report(["scripts/run_concurrency_eval.py", "--report-path", str(path)], gate)
    revision = payload["live"]["active_input_revision"]
    calibration = revision["reply_state_calibration"]
    gate.validate_active_reply_state_calibration(revision, revision["fixture"])
    if fault.startswith("actual_"):
        evidence = revision["semantic_assessment"]["evidence"]
        if fault == "actual_reply":
            evidence["reply"] = "另一条回复"
        elif fault == "actual_status":
            evidence["runtime_status"] = "completed"
        elif fault == "actual_request":
            evidence["request"]["text"] = "另一轮原话"
        else:
            evidence["execution_lifecycle"]["turn"]["status"] = "completed"
    elif fault == "missing_calibration":
        revision.pop("reply_state_calibration")
    elif fault == "missing_case":
        calibration["cases"].pop()
    elif fault == "duplicate_case":
        calibration["cases"][-1] = deepcopy(calibration["cases"][0])
    elif fault == "changed_source":
        calibration["cases"][0]["source"]["expected_passed"] = False
    elif fault == "changed_baseline":
        calibration["source_evidence"]["tool_results"] = [{"name": "unrelated_tool"}]
    elif fault == "changed_controlled_facts":
        item = calibration["cases"][0]
        item["evidence"]["persisted_state"] = {"different_fact": True}
        item["assessment"]["evidence"] = deepcopy(item["evidence"])
    elif fault == "borrowed_assessment":
        calibration["cases"][0]["assessment"] = deepcopy(revision["reply_state_assessment"])
    else:
        item = next(item for item in calibration["cases"] if not item["source"]["expected_passed"])
        assessment = item["assessment"]
        if fault == "changed_rubric":
            assessment["rubric"] = "another rubric"
        elif fault == "changed_blind_input":
            assessment["input"]["runtime_status"] = "completed"
        elif fault == "changed_temperature":
            assessment["judge_config"]["temperature"] = 1
        elif fault == "changed_thinking":
            assessment["judge_config"]["thinking_mode"] = "enabled"
        elif fault == "missing_attempts":
            assessment.pop("attempts")
        elif fault == "changed_attempt_raw":
            assessment["attempts"][0]["raw_response"] = "{}"
        elif fault == "changed_attempt_trace":
            assessment["attempts"][0]["trace_id"] = "another-call"
        elif fault == "changed_repair_count":
            assessment["contract_repair_count"] = 1
        elif fault in {"valid_judgement_retried", "wrong_repair_identity"}:
            first = deepcopy(assessment["attempts"][0])
            first["error"] = "claimed contract error"
            if fault == "wrong_repair_identity":
                first["raw_response"] = "{"
            assessment["attempts"].append(deepcopy(assessment["attempts"][0]))
            assessment["attempts"][0] = first
            assessment["attempts"][1]["trace_id"] += ":contract_repair_1" if fault == "valid_judgement_retried" else ":wrong-repair"
            assessment["contract_repair_count"] = 1
        elif fault == "missing_act":
            assessment.pop("communication_act")
        elif fault == "negative_false_pass":
            assessment["communication_act"] = "inform_only" if item["source"]["runtime_status"] == "completed" else "requires_user_response"
        elif fault == "negative_other_act":
            assessment["communication_act"] = "other"
        elif fault == "invalid_act_type":
            assessment["communication_act"] = 1
        if fault in {"missing_act", "negative_false_pass", "negative_other_act", "invalid_act_type"}:
            raw = {key: assessment[key] for key in ("communication_act", "reason") if key in assessment}
            assessment["raw_response"] = json.dumps(raw)
            assessment["attempts"][-1].update(raw_response=assessment["raw_response"], communication_act=assessment.get("communication_act"))
    with pytest.raises(ValueError):
        gate.validate_active_reply_state_calibration(revision, revision["fixture"])


def test_active_revision_calibration_allows_one_invalid_contract_repair(gate, tmp_path):
    """仅无效JSON可更正一次，保留原失败attempt且第二次身份明确。"""
    path = tmp_path / "repair-calibration.json"
    payload = successful_stub_report(["scripts/run_concurrency_eval.py", "--report-path", str(path)], gate)
    revision = payload["live"]["active_input_revision"]
    for item in revision["reply_state_calibration"]["cases"]:
        assessment = item["assessment"]
        final = deepcopy(assessment["attempts"][0])
        final["trace_id"] += ":contract_repair_1"
        assessment["attempts"] = [{"trace_id": assessment["trace_id"], "raw_response": "{",
                                   "error": "invalid JSON contract"}, final]
        assessment["contract_repair_count"] = 1
    gate.validate_active_reply_state_calibration(revision, revision["fixture"])
