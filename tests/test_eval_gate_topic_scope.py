"""新群入口范围必须完整执行，校准负例与真实主模型回放不能互相冒充。"""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from scripts import run_evals as gate
from tests.test_eval_gate import semantic_config_stub

sys.path.insert(0, str(Path(gate.__file__).parent))


def assessment_stub(evidence, expectation, trace_id, *, passed=True):
    """只替换远端HTTP输出，保留真实客户端产生的来源合同供门禁校验。"""
    from run_inquiry_grounding_eval import _parse_semantic_assessment, _semantic_criteria_ids

    ids = _semantic_criteria_ids(expectation)
    raw = json.dumps({"reason": "本地测试判决", "criteria": [
        {"id": identifier, "verdict": "clear" if passed else "violated", "reason": "本地测试证据"}
        for identifier in ids]})
    return {**_parse_semantic_assessment(raw, ids), "error": None, "expectation": expectation,
            "raw_response": raw, **semantic_config_stub(trace_id, raw, passed=passed, evidence=evidence)}


@pytest.fixture
def group_evidence(tmp_path, monkeypatch):
    """真实群入口、工具和SQLite，只有模型传输使用离线HTTP桩。"""
    import run_group_topic_regression_eval as runner
    from mahjong_agent_runtime import AgentLLMConfig
    from tests.semantic_http_stub import semantic_http_completion
    import sim_adapter
    from mahjong_agent_runtime import JsonlTraceRecorder

    original_build = runner.build_runtime

    def build(store, mode):
        monkeypatch.setattr(sim_adapter, "InMemoryTraceRecorder",
                            lambda: JsonlTraceRecorder(store.path.parent / "agent_trace.log"))
        runtime, client = original_build(store, "mock")
        client.config = AgentLLMConfig(api_key="unit-only", model="unit-model", provider="deepseek",
            base_url="https://example.invalid", thinking_mode="enabled", reasoning_effort="high",
            max_tokens=16384, retry_attempts=1, attempt_timeout_seconds=60)
        complete = client.complete

        def http_complete(messages, **kwargs):
            raw = complete(messages, **kwargs)
            return semantic_http_completion(raw, config=client.config, trace_id=kwargs["trace_id"])

        client.complete = http_complete
        return runtime, client

    monkeypatch.setattr(runner, "build_runtime", build)
    monkeypatch.setattr(runner, "assess_reply_semantics", lambda _client, **kwargs: assessment_stub(**kwargs))
    case = next(case for case in runner.load_cases(runner.DEFAULT_DATASET)
                if case["case_id"] == "delegation_query_contrast")
    dataset = tmp_path / "source.jsonl"
    dataset.write_text(json.dumps(case) + "\n")
    work = tmp_path / "work"
    folder = work / "run_current"
    report = runner.run_case(case, folder / case["case_id"])
    assert report["status"] == "passed", report
    summary = {"status": "passed", "dataset": str(dataset), "selected_cases": [case["case_id"]],
        "cases": [{"case_id": case["case_id"], "status": "passed", "checks": report["checks"], "error": None}]}
    source = folder / "summary.json"
    source.write_text(json.dumps(summary))
    phase = {"scope": "group_offline_contract", "scope_family": gate.LIVE_SCOPES[7],
        "dataset": str(dataset), "dataset_sha256": gate.file_digest(dataset), "work_dir": str(work),
        "expected_case_ids": [case["case_id"]], "report_path": str(tmp_path / "scope_report.json")}
    gate.archive_external_scope_report(phase, required=True)
    return phase, case, folder / case["case_id"]


def test_group_gate_revalidates_current_http_trace_sqlite_and_independent_judgment(group_evidence):
    phase, case, _ = group_evidence
    gate.validate_scope_report(phase)
    assert phase["case_coverage"]["executed_case_ids"] == [case["case_id"]]
    assert phase["recovery_evidence"][0]["recovery"][0]["quality_status"] == "passed"


@pytest.mark.parametrize("fault", ["partial", "duplicate", "changed_source", "missing_trace", "failed_check",
    "numeric_check", "missing_http", "wrong_model", "changed_input", "frozen", "wrong_expectation",
    "wrong_raw_judgment", "missing_criterion", "changed_sqlite_projection", "wrong_judgment_reply",
    "raw_request", "delivery_claim"])
def test_group_gate_rejects_incomplete_or_forged_execution(group_evidence, fault):
    phase, case, folder = group_evidence
    summary_path = Path(phase["source_report_path"])
    summary = json.loads(summary_path.read_text())
    report_path = folder / "report.json"
    report = json.loads(report_path.read_text())
    if fault == "partial":
        summary["cases"] = []
    elif fault == "duplicate":
        summary["cases"].append(deepcopy(summary["cases"][0]))
    elif fault == "changed_source":
        Path(phase["dataset"]).write_text("{}\n")
    elif fault == "missing_trace":
        (folder / "agent_trace.log").unlink()
    elif fault in {"failed_check", "numeric_check"}:
        key = next(iter(report["checks"]))
        report["checks"][key] = False if fault == "failed_check" else 1
        summary["cases"][0]["checks"] = deepcopy(report["checks"])
    elif fault in {"missing_http", "wrong_model", "raw_request"}:
        from mahjong_agent_runtime.tracing import TraceEvent
        path = folder / "agent_trace.log"
        lines = path.read_text().splitlines()
        events = [TraceEvent.from_log_line(line) for line in lines]
        target = next(event for event in events if event is not None and event.step == "llm_response")
        if fault == "missing_http":
            target.content.pop("response_source")
        elif fault == "wrong_model":
            report["model_config"]["model"] = "unexecuted-model"
        else:
            next(event for event in events if event is not None and event.step == "user_input").content["message"]["text"] = "错接原始输入"
        if fault != "wrong_model":
            path.write_text("\n".join(event.to_log_line() if event is not None else line
                                     for event, line in zip(events, lines)) + "\n")
    elif fault == "changed_input":
        report["turns"][0]["input"]["text"] = "另一条更容易的输入"
    elif fault == "frozen":
        report["main_agent_executed"] = False
    elif fault == "wrong_expectation":
        report["semantic_assessment"]["expectation"] = {"required_meaning": []}
    elif fault == "wrong_raw_judgment":
        report["semantic_assessment"]["raw_response"] = "{}"
    elif fault == "missing_criterion":
        report["semantic_assessment"]["criteria"].pop()
    elif fault == "changed_sqlite_projection":
        report["semantic_assessment"]["evidence"]["persisted_state"]["games"] = [{"game_id": "invented"}]
    elif fault == "wrong_judgment_reply":
        report["semantic_assessment"]["evidence"]["reply"] = "判定的是另一句回复"
    elif fault == "delivery_claim":
        report["turns"][0]["outputs"] = []
    report_path.write_text(json.dumps(report))
    summary_path.write_text(json.dumps(summary))
    gate.archive_external_scope_report(phase, required=True)
    with pytest.raises((ValueError, FileNotFoundError)):
        gate.validate_scope_report(phase)


def test_group_archive_rejects_multiple_runs_instead_of_choosing_passed_report(tmp_path):
    for name in ("run_failed", "run_passed"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="exactly one"):
        gate.archive_external_scope_report({"scope_family": gate.LIVE_SCOPES[7], "work_dir": str(tmp_path)}, required=True)


@pytest.fixture
def recruitment_evidence(tmp_path):
    import run_recruitment_obligation_calibration as runner

    labels = {case["case_id"]: case["expected_passed"] for case in runner.load_cases()}

    def judge(_client, **kwargs):
        return assessment_stub(**kwargs, passed=labels[kwargs["trace_id"].removesuffix(":judge")])

    work = tmp_path / "calibration"
    runner.run_calibration(None, work, judge=judge)
    return {"scope": gate.LIVE_SCOPES[8], "dataset": str(runner.DATASET),
        "dataset_sha256": gate.file_digest(runner.DATASET), "work_dir": str(work),
        "report_path": str(work / "report.json")}


def test_recruitment_gate_accepts_fixed_positive_and_negative_labels(recruitment_evidence):
    gate.validate_scope_report(recruitment_evidence)
    assert recruitment_evidence["case_coverage"]["total"] == 7


@pytest.mark.parametrize("fault", ["partial", "duplicate", "source_hash", "changed_label", "wrong_negative",
    "missing_http", "side_effect", "missing_fixture", "wrong_reply", "missing_raw",
    "wrong_request", "missing_fixture_tool", "missing_activation"])
def test_recruitment_gate_rejects_missing_or_mismatched_calibration(recruitment_evidence, fault):
    phase = recruitment_evidence
    path = Path(phase["report_path"])
    payload = json.loads(path.read_text())
    item = next(report for report in payload["reports"] if report["source"]["expected_passed"] is False
                and report["source"]["fixture"] == "board_activation")
    folder = Path(phase["work_dir"]) / item["source"]["case_id"]
    if fault == "partial":
        payload["reports"].pop()
    elif fault == "duplicate":
        payload["reports"][-1] = deepcopy(payload["reports"][0])
    elif fault == "source_hash":
        payload["dataset_sha256"] = "stale"
    elif fault == "changed_label":
        item["source"]["expected_passed"] = True
    elif fault == "wrong_negative":
        item["semantic_assessment"]["passed"] = True
    elif fault == "missing_http":
        item["semantic_assessment"].pop("attempts")
    elif fault == "side_effect":
        item["checks"]["no_business_side_effects"] = False
    elif fault == "missing_fixture":
        (folder / "fixture.json").unlink()
    elif fault == "wrong_reply":
        item["semantic_assessment"]["evidence"]["reply"] = "凭空改写"
    elif fault == "missing_raw":
        item["semantic_assessment"].pop("raw_response")
    elif fault in {"wrong_request", "missing_fixture_tool", "missing_activation"}:
        fixture_path = folder / "fixture.json"
        fixture = json.loads(fixture_path.read_text())
        if fault == "wrong_request":
            fixture["evidence"]["request"]["text"] = "另一条更容易的请求"
            next(event for event in fixture["events"] if event["step"] == "context_built")["content"]["current_message"]["text"] = "另一条更容易的请求"
        elif fault == "missing_fixture_tool":
            fixture["events"] = [event for event in fixture["events"] if not
                (event["step"] == "tool_result" and event["content"]["name"] == "create_game")]
            fixture["evidence"]["tool_results"] = [tool for tool in fixture["evidence"]["tool_results"] if tool["name"] != "create_game"]
        else:
            fixture["evidence"]["persisted_state"]["recruitment_activations"] = []
        item["semantic_assessment"]["evidence"] = deepcopy(fixture["evidence"])
        fixture_path.write_text(json.dumps(fixture))
    (folder / "report.json").write_text(json.dumps(item))
    path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, FileNotFoundError)):
        gate.validate_scope_report(phase)
