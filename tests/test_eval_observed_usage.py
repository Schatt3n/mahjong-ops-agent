"""只依据范围报告的供应商诊断观察用量，不能用统计代替验收或账单。"""

import json
import sys

import pytest

from scripts import run_evals as gate


def test_partial_usage_deduplicates_copies_and_preserves_unknowns():
    def diagnostics(*attempts):
        return {"provider": "test", "attempts": list(attempts)}

    first = {"attempt_id": "a", "response_id": "r1", "usage": {"prompt_tokens": 0, "completion_tokens": 7}}
    retry = {"attempt_id": "b", "response_id": "r2", "usage": {"prompt_tokens": 11}}
    unknown = {"attempt_id": "c", "usage": {"completion_tokens": 20}}
    payload = [
        {"completion_diagnostics": diagnostics(first, retry)},
        {"completion_diagnostics": {"route": "fallback", "primary": diagnostics(first, unknown),
                                    "fallback": diagnostics({**first, "attempt_id": "d"})}},
        {"estimated": {"usage": {"prompt_tokens": 900}}},
    ]
    actual = gate.observed_scope_usage(payload)
    assert actual == {
        "coverage": "partial_scope_report_diagnostics", "observed_attempts": 4,
        "attempts_without_response_id": 1, "observed_provider_responses": 2,
        "observed_prompt_tokens": 11, "observed_completion_tokens": 7,
        "responses_without_verified_prompt_tokens": 0, "responses_without_verified_completion_tokens": 1,
    }
    first["usage"]["completion_tokens"] = 8
    payload.append({"completion_diagnostics": diagnostics(
        {**first, "usage": {"prompt_tokens": True, "completion_tokens": 7}})})
    assert gate.observed_scope_usage(payload)["observed_completion_tokens"] is None


@pytest.mark.parametrize("return_error", [False, True])
def test_main_records_usage_on_success_or_error(monkeypatch, tmp_path, return_error):
    path = tmp_path / "live.json"
    scope = gate.LIVE_SCOPES[0]
    plan = [{"scope": scope, "report_path": str(path), "command": ["unit-live"]}]
    monkeypatch.setattr(sys, "argv", ["run_evals.py", "--full"])
    monkeypatch.setattr(gate, "create_run_directory", lambda: tmp_path)
    monkeypatch.setattr(gate, "load_dotenv_defaults", lambda path: None)
    monkeypatch.setattr(gate, "build_eval_plan", lambda args, run_dir: plan)
    monkeypatch.setattr(gate, "validate_scope_report", lambda phase: None)

    def run(command, log_path):
        path.write_text(json.dumps({"completion_diagnostics": {"provider": "unit", "attempts": [
            {"attempt_id": "a", "response_id": "response", "usage": {"prompt_tokens": 10}}]}}))
        if return_error:
            raise RuntimeError("scenario failed")

    monkeypatch.setattr(gate, "run_command", run)
    assert gate.main() == int(return_error)
    phase = json.loads((tmp_path / "gate_report.json").read_text())["scopes"][0]
    assert phase["observed_usage"]["observed_prompt_tokens"] == 10
    assert phase["observed_usage"]["observed_completion_tokens"] is None
    assert phase["status"] == ("failed" if return_error else "passed")


def test_unreadable_usage_report_does_not_override_phase_failure(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{")
    phase = {"scope": gate.LIVE_SCOPES[0], "status": "failed", "report_path": str(path)}
    gate.record_observed_usage(phase)
    assert phase["observed_usage"] == {"coverage": "report_unavailable"}
    assert phase["status"] == "failed"


def test_group_topic_usage_reads_current_case_reports_after_early_failure(tmp_path):
    """群汇总不含回执，失败案例的独立判定用量仍从本轮明细读取。"""
    folder = tmp_path / "run_current"
    case = folder / "case_a"
    case.mkdir(parents=True)
    summary = folder / "summary.json"
    summary.write_text(json.dumps({"status": "failed"}))
    (case / "report.json").write_text(json.dumps({"status": "failed", "semantic_assessment": {
        "completion_diagnostics": {"provider": "unit", "attempts": [
            {"attempt_id": "attempt", "response_id": "response", "usage": {"completion_tokens": 11}}]}}}))
    phase = {"scope": "group_topic_case_a", "scope_family": gate.LIVE_SCOPES[7], "status": "failed",
             "report_path": str(summary), "source_report_path": str(summary)}
    gate.record_observed_usage(phase)
    assert phase["observed_usage"]["observed_completion_tokens"] == 11
    assert phase["observed_usage"]["observed_prompt_tokens"] is None
    assert phase["status"] == "failed"
