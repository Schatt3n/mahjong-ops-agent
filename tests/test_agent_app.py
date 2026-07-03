from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from mahjong_agent_runtime import (
    AgentRuntime,
    InMemoryTraceRecorder,
    SQLiteAgentStore,
    StaticAgentClient,
    ToolGateway,
    UserMessage,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "agent_runtime_app.py"
MAIN_SCRIPT = ROOT / "scripts" / "run_agent_app.py"


def load_app_module():
    spec = importlib.util.spec_from_file_location("agent_runtime_app_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manual_badcase_is_recorded_through_tool_gateway(tmp_path) -> None:
    app = load_app_module()
    store = SQLiteAgentStore(tmp_path / "agent_runtime_manual_badcase.sqlite3")
    trace = InMemoryTraceRecorder()
    runtime = AgentRuntime(
        llm_client=StaticAgentClient(
            [
                app.json.dumps(
                    {
                        "goal": "测试回复",
                        "objective_status": "completed",
                        "reasoning_summary": "模型直接回复。",
                        "reply_to_user": "好的，我先帮你留意下。",
                        "tool_calls": [],
                        "needs_human": False,
                        "stop_reason": {
                            "can_stop": True,
                            "why": "测试场景模拟模型提前停止，后续由人工 badcase 入口归档。",
                            "pending_work": [],
                            "depends_on_tool_results": False,
                        },
                        "badcase": None,
                    },
                    ensure_ascii=False,
                )
            ]
        ),
        store=store,
        tool_gateway=ToolGateway(store=store, trace_recorder=trace),
        trace_recorder=trace,
    )
    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_manual_badcase",
            sender_id="zhang",
            sender_name="张哥",
            text="组",
            message_id="msg_manual_badcase",
        ),
        trace_id="trace_manual_source",
    )

    response = app.record_manual_badcase(
        runtime,
        {
            "source_trace_id": result.trace_id,
            "audit_trace_id": "trace_manual_audit",
            "reason": "回复停在留意，没有继续规划",
            "expected": {"behavior": "应该继续规划或追问关键缺口"},
            "operator_id": "tester",
            "operator_name": "测试者",
        },
    )

    assert response["tool_result"]["called"] is True
    assert response["tool_result"]["allowed"] is True
    assert len(store.badcases) == 1
    badcase = store.badcases[0]
    assert badcase["reason"] == "回复停在留意，没有继续规划"
    assert badcase["input"]["message"]["text"] == "组"
    assert badcase["actual"]["reply"] == "好的，我先帮你留意下。"
    assert badcase["metadata"]["source_trace_id"] == "trace_manual_source"
    steps = [event.step for event in trace.get_trace("trace_manual_audit")]
    assert steps == [
        "manual_badcase_input",
        "tool_called",
        "tool_gateway_received",
        "tool_idempotency_checked",
        "tool_definition_checked",
        "tool_schema_checked",
        "tool_permission_checked",
        "tool_idempotency_claimed",
        "tool_gateway_completed",
        "tool_result",
        "manual_badcase_recorded",
    ]


def test_app_defaults_to_main_trial_port(monkeypatch) -> None:
    monkeypatch.delenv("MAHJONG_AGENT_PORT", raising=False)

    app = load_app_module()

    assert app.PORT == 8790


def test_main_agent_app_entrypoint_exists_without_versioned_operator_name() -> None:
    text = MAIN_SCRIPT.read_text(encoding="utf-8")

    assert "from agent_runtime_app import main" in text


def test_main_app_imports_stable_runtime_package() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "from mahjong_agent_runtime import" in text
    assert "from mahjong_agent_v3 import" not in text
    assert "from mahjong_agent_v3.tracing" not in text


def test_runtime_manifest_identifies_current_main_chain(tmp_path) -> None:
    app = load_app_module()
    store = SQLiteAgentStore(tmp_path / "agent_runtime_manifest.sqlite3")
    trace = InMemoryTraceRecorder()
    runtime = AgentRuntime(
        llm_client=StaticAgentClient([]),
        store=store,
        tool_gateway=ToolGateway(store=store, trace_recorder=trace),
        trace_recorder=trace,
    )

    manifest = app.runtime_manifest(runtime)

    assert manifest["runtime"] == "mahjong_agent_runtime"
    assert manifest["main_chain"] == "agent_runtime"
    assert manifest["implementation_package"] == "mahjong_agent_runtime"
    assert manifest["compatibility_packages"] == ["mahjong_agent_v3"]
    assert manifest["legacy_reference_only"] is True
    assert manifest["legacy_entrypoints"]["legacy_analyze_endpoint"] == "not_exposed"
    assert "/api/message" in manifest["endpoints"]["message"]
    assert "legacy_endpoint_aliases" not in manifest
    assert "search_current_games" in manifest["available_tools"]
    assert "update_context_checkpoint" in manifest["available_tools"]
    assert "/api/analyze" not in app.json.dumps(manifest, ensure_ascii=False)


def test_log_tail_exposes_trace_log_path(tmp_path, monkeypatch) -> None:
    app = load_app_module()
    trace_path = tmp_path / "agent_runtime_trace.log"
    trace_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    monkeypatch.setattr(app, "TRACE_PATH", trace_path)

    assert app.tail_trace_log(2) == ["line2", "line3"]
