from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from mahjong_agent_v2 import InMemoryAgentStoreV2, InMemoryEvalRecorderV2, ToolGatewayV2
from mahjong_agent_v2.tracing import InMemoryTraceRecorderV2, validate_agent_runtime_trace_completeness


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "scripts" / "run_agent_v2_app.py"
BOUNDARY_SCRIPT = ROOT / "scripts" / "verify_agent_runtime_v2_boundary.py"


def load_app_module_without_runtime():
    spec = importlib.util.spec_from_file_location("run_agent_v2_app_for_test", APP_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    module.build_runtime = lambda: None
    spec.loader.exec_module(module)
    return module


def load_boundary_module():
    spec = importlib.util.spec_from_file_location("verify_agent_runtime_v2_boundary_for_test", BOUNDARY_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_agent_v2_console_exposes_observable_panels() -> None:
    module = load_app_module_without_runtime()

    html = module.index_html()

    assert "本轮结果" in html
    assert "模型决策" in html
    assert "工具调用" in html
    assert "状态变化" in html
    assert "Trace 完整性" in html
    assert "Trace 事件" in html
    assert "Badcase" in html
    assert "message_id" in html
    assert "/api/v2/message" in html
    assert "/api/v2/state" in html
    assert "/api/v2/traces" in html
    assert "/api/v2/badcases" in html
    assert "traceCompleteness" in html
    assert "completeness" in html
    assert "归档当前回复为 badcase" in html
    assert "recordBadcase" in html


def test_agent_v2_trace_payload_includes_completeness_report() -> None:
    module = load_app_module_without_runtime()
    trace = InMemoryTraceRecorderV2()
    trace.record("trace_observable", "user_input", {"text": "通宵有人吗"})
    runtime = SimpleNamespace(trace_recorder=trace)

    payload = module.trace_payload(runtime, "trace_observable")

    assert payload["trace_id"] == "trace_observable"
    assert payload["trace_log_path"].endswith("logs/agent_runtime_v2_trace.jsonl")
    assert len(payload["events"]) == 1
    assert payload["completeness"]["complete"] is False
    assert "context_packed" in payload["completeness"]["missing_steps"]


def test_agent_v2_manual_badcase_is_recorded_through_tool_gateway_trace() -> None:
    module = load_app_module_without_runtime()
    store = InMemoryAgentStoreV2()
    trace = InMemoryTraceRecorderV2()
    eval_recorder = InMemoryEvalRecorderV2()
    gateway = ToolGatewayV2(store=store, eval_recorder=eval_recorder, trace_recorder=trace)
    runtime = SimpleNamespace(trace_recorder=trace, tool_gateway=gateway)
    trace.record(
        "source_trace",
        "user_input",
        {
            "message": {
                "conversation_id": "boss_trial_v2",
                "sender_id": "zhang",
                "sender_name": "张哥",
                "text": "通宵有人吗",
            }
        },
    )
    trace.record("source_trace", "final_output", {"reply": "通宵有的，你几个人？"})

    result = module.record_manual_badcase(
        runtime,
        {
            "trace_id": "source_trace",
            "reason": "回复声称有通宵局但当前局池没有证据",
            "expected": {"reply_style": "应该先查当前局或说明没有现成局"},
            "audit_trace_id": "manual_badcase_trace",
        },
    )

    assert result["audit_trace_id"] == "manual_badcase_trace"
    assert result["source_trace_id"] == "source_trace"
    assert result["tool_result"]["name"] == "record_badcase"
    assert result["tool_result"]["called"] is True
    assert eval_recorder.records[0]["source"] == "manual_operator"
    assert eval_recorder.records[0]["trace_id"] == "manual_badcase_trace"
    assert eval_recorder.records[0]["conversation_id"] == "boss_trial_v2"
    assert eval_recorder.records[0]["input"]["message"]["text"] == "通宵有人吗"
    assert eval_recorder.records[0]["actual"]["reply"] == "通宵有的，你几个人？"
    steps = [event.step for event in trace.get_trace("manual_badcase_trace")]
    assert steps == [
        "manual_badcase_input",
        "tool_called",
        "tool_gateway_received",
        "tool_idempotency_checked",
        "tool_definition_checked",
        "tool_schema_checked",
        "tool_permission_checked",
        "tool_gateway_completed",
        "tool_result",
        "manual_badcase_recorded",
    ]
    assert validate_agent_runtime_trace_completeness(trace.get_trace("manual_badcase_trace")).complete is True


def test_agent_v2_app_defaults_to_main_trial_port(monkeypatch) -> None:
    monkeypatch.delenv("MAHJONG_AGENT_V2_PORT", raising=False)

    module = load_app_module_without_runtime()

    assert module.PORT == 8790


def test_agent_v2_budget_is_configured_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_AGENT_V2_MAX_TOKENS_PER_CALL", "64000")
    monkeypatch.setenv("MAHJONG_AGENT_V2_MAX_CALLS_PER_TURN", "9")

    module = load_app_module_without_runtime()
    budget = module.budget_from_env()

    assert budget.max_tokens_per_call == 64000
    assert budget.max_calls_per_turn == 9


def test_agent_v2_budget_uses_legacy_env_alias_for_transition(monkeypatch) -> None:
    monkeypatch.delenv("MAHJONG_AGENT_V2_MAX_TOKENS_PER_CALL", raising=False)
    monkeypatch.setenv("MAHJONG_LLM_MAX_TOKENS_PER_CALL", "48000")

    module = load_app_module_without_runtime()

    assert module.budget_from_env().max_tokens_per_call == 48000


def test_agent_v2_reply_review_defaults_on_and_can_be_disabled(monkeypatch) -> None:
    monkeypatch.delenv("MAHJONG_AGENT_V2_REPLY_REVIEW_ENABLED", raising=False)
    module = load_app_module_without_runtime()

    assert module.env_bool("MAHJONG_AGENT_V2_REPLY_REVIEW_ENABLED", True) is True

    monkeypatch.setenv("MAHJONG_AGENT_V2_REPLY_REVIEW_ENABLED", "false")
    assert module.env_bool("MAHJONG_AGENT_V2_REPLY_REVIEW_ENABLED", True) is False


def test_agent_v2_entrypoints_do_not_import_legacy_main_chain() -> None:
    files = [
        ROOT / "scripts" / "run_agent_v2_app.py",
        ROOT / "scripts" / "run_agent_runtime_v2_eval.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in files)

    forbidden = [
        "from mahjong_agent import",
        "import mahjong_agent.",
        "semantic_resolver",
        "controlled_workflow",
        "reply_guard",
        "backend_fallback",
    ]
    for item in forbidden:
        assert item not in source


def test_agent_runtime_v2_boundary_script_rejects_legacy_imports(tmp_path) -> None:
    module = load_boundary_module()
    bad_file = tmp_path / "bad_v2_import.py"
    bad_file.write_text("from mahjong_agent.parser import parse_message\n", encoding="utf-8")

    violations = module.verify_files([bad_file])

    messages = "\n".join(violation.message for violation in violations)
    assert "legacy package" in messages
    assert "parser" in messages


def test_agent_runtime_v2_boundary_script_passes_current_main_chain() -> None:
    module = load_boundary_module()

    violations = module.verify_files()

    assert violations == []
