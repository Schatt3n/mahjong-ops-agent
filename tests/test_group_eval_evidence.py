"""群回放异常必须留下实际已产生证据并失败，独立用例和后续运行不覆盖历史原迹。"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime.llm import LLMCompletionText

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_group_eval_envelope import GroupQueryClient, _module


@pytest.fixture
def module():
    """使用现有群runner和生产Runtime，真实模型边界仅用明确的离线替身。"""
    return _module()


def original_cases(module):
    """取已保存的匿名原案例；测试不写入或重标任何原始资产。"""
    semantic = next(c for c in module.read_jsonl(module.ROOT / "eval/golden/real_group_chat_20260726.jsonl")
                    if c["id"] == "real_group_fragmented_smoking_query_20260726")
    deterministic = next(c for c in module.read_jsonl(module.DEFAULT_DATASET) if c["case_type"] == "quick_filter")
    return semantic, deterministic


def test_counting_client_persists_started_before_delegating_and_keeps_real_body(module, tmp_path):
    journal = tmp_path / "calls.jsonl"
    actual_messages = [{"role": "user", "content": "实际请求，含换行\n第二行"}]
    raw = LLMCompletionText("原始响应正文", {"schema_version": 1, "call_id": "unit-only", "attempts": []})

    class Delegate:
        def complete(self, messages, **kwargs):
            started = [json.loads(line) for line in journal.read_text().splitlines()]
            assert len(started) == 1 and started[0]["status"] == "started"
            assert started[0]["messages"] == messages == actual_messages
            return raw

    client = module.CountingClient(Delegate(), evidence_path=journal)
    assert client.complete(actual_messages, trace_id="actual-call", timeout_seconds=2) is raw
    saved = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [item["status"] for item in saved] == ["started", "completed"]
    assert saved[-1]["raw_response"] == raw
    assert saved[-1]["completion_diagnostics"] == raw._completion_diagnostics
    assert client.calls[0] == saved[-1]


def test_counting_exception_keeps_started_request_and_does_not_invent_a_response(module, tmp_path):
    journal = tmp_path / "calls.jsonl"
    credential = "UNIT_CONFIGURED_CREDENTIAL_8e3aa"
    original_error = ValueError("normalization failed: " + credential)

    class Delegate:
        config = SimpleNamespace(api_key=credential)

        def complete(self, messages, **kwargs):
            assert json.loads(journal.read_text().splitlines()[0])["status"] == "started"
            raise original_error

    client = module.CountingClient(Delegate(), evidence_path=journal)
    with pytest.raises(ValueError) as raised:
        client.complete([{"role": "user", "content": "actual input"}], trace_id="failed-call", timeout_seconds=2)
    assert raised.value is original_error
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["status"] for row in rows] == ["started", "error"]
    assert "raw_response" not in rows[-1]
    assert credential not in journal.read_text()
    assert "[redacted credential]" in rows[-1]["error"]


def test_all_actually_configured_credentials_are_redacted_without_dumping_client_internals(module, tmp_path):
    primary, fallback = "PRIMARY_CREDENTIAL_8ab5", "FALLBACK_CREDENTIAL_9eb0"
    reasoning = "PROVIDER_REASONING_BODY_MUST_NOT_BE_RECORDED"
    ordinary = "ORDINARY_NONCREDENTIAL_TOKEN"
    messages = [{"role": "user", "content": f"{primary} {fallback} {ordinary}"}]
    seen = []

    class Delegate:
        config = SimpleNamespace(api_key=primary)
        fallback_client = SimpleNamespace(config=SimpleNamespace(api_key=fallback))
        headers = {"authorization": primary}
        reasoning_content = reasoning

        def complete(self, actual, **kwargs):
            seen.append(deepcopy(actual))
            return LLMCompletionText(f"{primary} {fallback} {ordinary}", {
                "schema_version": 1, "reasoning_fields": {"reasoning_content": {"present": True, "length": len(reasoning)}}})

    client = module.CountingClient(Delegate(), evidence_path=tmp_path / "calls.jsonl")
    result = client.complete(messages, trace_id="known-config-only", timeout_seconds=2)
    assert seen == [messages] and primary in result and fallback in result
    saved = (tmp_path / "calls.jsonl").read_text()
    assert primary not in saved and fallback not in saved and reasoning not in saved
    assert ordinary in saved
    assert not {"config", "headers", "authorization"} & client.calls[0].keys()


def fail_after_real_main(monkeypatch, module, *, credential=""):
    """故障注入在完整生产主循环返回边界，已有工具/终态追踪不能让异常用例被误计通过。"""
    real_runtime = module.AgentRuntime

    class FaultRuntime(real_runtime):
        def handle_user_message(self, message, *, trace_id):
            super().handle_user_message(message, trace_id=trace_id)
            raise ValueError("unit injected after completed main: " + credential)

    monkeypatch.setattr(module, "AgentRuntime", FaultRuntime)


def test_main_exception_archives_every_existing_trace_and_continues_independent_cases(module, monkeypatch, tmp_path):
    semantic, noise = original_cases(module)
    before, after = deepcopy(noise), deepcopy(noise)
    before["id"], after["id"] = "noise-before-failure", "noise-after-failure"
    run = tmp_path / "run"
    run.mkdir()
    fail_after_real_main(monkeypatch, module)
    client = GroupQueryClient()
    report = module.RealGroupChatFlowEvaluator(llm_client=client, evidence_dir=run).evaluate([before, semantic, after])
    assert [case["status"] for case in report["cases"]] == ["passed", "failed", "passed"]
    assert report["summary"]["passed"] == 2 and report["summary"]["failed"] == 1
    failed = report["cases"][1]
    main = failed["actual"]["main_agent"]
    assert main["status"] == "failed" and "ValueError" in main["error"]
    assert main["input_message"]["text"] == "有烟的\n有缺人的没"
    assert main["tool_calls"] == ["search_current_games"]
    assert {"user_input", "llm_prompt", "llm_response", "tool_called", "tool_result", "final_output"} <= set(main["trace_steps"])
    assert main["group_messages"] == []
    assert len(client.main_payloads) == 2  # 未因异常追加主模型重试或兜底业务动作。
    directory = run / "case_0002"
    archived = json.loads((directory / "main_agent.json").read_text())
    recorder = module.JsonlTraceRecorder(directory / "main_trace.jsonl")
    assert archived["trace_events"] == main["trace_events"] == [e.to_dict() for e in recorder.get_trace(main["trace_id"])]
    assert json.loads((directory / "source_case.json").read_text()) == semantic
    assert json.loads((directory / "result.json").read_text())["status"] == "failed"
    assert json.loads((run / "progress.json").read_text())["cases"][-1]["case_id"] == after["id"]


def test_main_exception_stays_failed_when_all_other_case_assertions_succeed(module, monkeypatch, tmp_path):
    semantic, _ = original_cases(module)
    run = tmp_path / "run"
    run.mkdir()
    fail_after_real_main(monkeypatch, module)
    monkeypatch.setattr(module.RealGroupChatFlowEvaluator, "_check_case", staticmethod(lambda *a, **k: []))
    result = module.RealGroupChatFlowEvaluator(llm_client=GroupQueryClient(), evidence_dir=run).evaluate([semantic])["cases"][0]
    assert result["status"] == "failed"
    assert len(result["errors"]) == 1 and "main Agent raised an exception" in result["errors"][0]


def test_group_entry_exception_preserves_a_result_that_the_runtime_already_returned(module, monkeypatch, tmp_path):
    semantic, _ = original_cases(module)
    original = module.GroupMessageHandler._handle_agent_loop

    def fail_after_entry(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise ValueError("unit post-entry error")

    monkeypatch.setattr(module.GroupMessageHandler, "_handle_agent_loop", fail_after_entry)
    run = tmp_path / "run"
    run.mkdir()
    result = module.RealGroupChatFlowEvaluator(llm_client=GroupQueryClient(), evidence_dir=run).evaluate([semantic])["cases"][0]
    main = result["actual"]["main_agent"]
    assert result["status"] == main["status"] == "failed"
    assert main["captured_result_status"] == "completed"
    assert main["final_reply"] == "暂时没有有烟的缺人局。"
    assert main["tool_calls"] == ["search_current_games"] and len(main["group_messages"]) == 1


def test_main_exception_keeps_inline_trace_without_an_evidence_directory(module, monkeypatch):
    semantic, _ = original_cases(module)
    fail_after_real_main(monkeypatch, module)
    result = module.RealGroupChatFlowEvaluator(llm_client=GroupQueryClient()).evaluate_case(semantic)
    assert result["status"] == "failed"
    assert result["actual"]["main_agent"]["trace_events"]
    assert "final_output" in result["actual"]["main_agent"]["trace_steps"]


def test_runtime_technical_stop_does_not_pass_when_other_assertions_succeed(module, monkeypatch):
    semantic, _ = original_cases(module)
    original = module.RealGroupChatFlowEvaluator._run_main_agent

    def technical_stop(**kwargs):
        result = original(**kwargs)
        result["status"] = "needs_help"
        return result

    monkeypatch.setattr(module.RealGroupChatFlowEvaluator, "_run_main_agent", staticmethod(technical_stop))
    monkeypatch.setattr(module.RealGroupChatFlowEvaluator, "_check_case", staticmethod(lambda *a, **k: []))
    result = module.RealGroupChatFlowEvaluator(llm_client=GroupQueryClient()).evaluate_case(semantic)
    assert result["status"] == "failed" and "needs_help" in result["errors"][0]


def test_pipeline_exception_is_archived_and_does_not_stop_following_case(module, monkeypatch, tmp_path):
    semantic, noise = original_cases(module)
    run = tmp_path / "run"
    run.mkdir()
    original = module.RealGroupChatFlowEvaluator._pipeline

    def pipeline(self, store, llm, case):
        if case["id"] == semantic["id"]:
            raise ValueError("unit pipeline setup failed")
        return original(self, store, llm, case)

    monkeypatch.setattr(module.RealGroupChatFlowEvaluator, "_pipeline", pipeline)
    report = module.RealGroupChatFlowEvaluator(llm_client=GroupQueryClient(), evidence_dir=run).evaluate([semantic, noise])
    assert [case["status"] for case in report["cases"]] == ["failed", "passed"]
    saved = json.loads((run / "case_0001/result.json").read_text())
    assert saved["actual"]["partial_state"]["domain_counts"]["games"] == 0
    assert saved["model_calls"] == []


def test_trace_and_main_error_are_redacted_before_append_not_after_failure(module, monkeypatch, tmp_path):
    semantic, _ = original_cases(module)
    credential = "KNOWN_MAIN_TRACE_CREDENTIAL_8ce5"
    client = GroupQueryClient()
    client.config = SimpleNamespace(api_key=credential)
    real = module.AgentRuntime

    class FaultRuntime(real):
        def handle_user_message(self, message, *, trace_id):
            self.trace_recorder.record(trace_id, "unit_observation", {"text": credential})
            assert credential not in Path(self.trace_recorder.path).read_text()
            raise ValueError("unit failure " + credential)

    monkeypatch.setattr(module, "AgentRuntime", FaultRuntime)
    run = tmp_path / "run"
    run.mkdir()
    report = module.RealGroupChatFlowEvaluator(llm_client=client, evidence_dir=run).evaluate([semantic])
    assert report["cases"][0]["status"] == "failed"
    assert credential not in json.dumps(report)
    assert all(credential not in path.read_text() for path in run.rglob("*") if path.is_file())


def test_repeated_cli_runs_keep_unique_complete_evidence_without_overwriting_old_raw(module, monkeypatch, tmp_path, capsys):
    _, noise = original_cases(module)
    dataset = tmp_path / "source.jsonl"
    dataset.write_text(json.dumps(noise, ensure_ascii=False) + "\n")
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(live=False, strict=True, dataset=dataset,
        report_path=report_path, quiet_seconds=10, env_file=tmp_path / "absent.env"))
    assert module.main() == 0
    first = json.loads(report_path.read_text())
    directory = Path(first["evidence_dir"])
    before = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob("*") if p.is_file()}
    assert {"summary.json", "progress.json", "case_0001/source_case.json", "case_0001/result.json"} <= set(before)
    assert module.main() == 0
    second = json.loads(report_path.read_text())
    assert first["evidence_dir"] != second["evidence_dir"]
    assert before == {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob("*") if p.is_file()}
    assert json.loads((directory / "summary.json").read_text()) == first
    capsys.readouterr()
