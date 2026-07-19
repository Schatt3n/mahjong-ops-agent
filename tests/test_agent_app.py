from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from mahjong_agent_runtime import (
    AgentRuntime,
    CustomerProfile,
    InMemoryTraceRecorder,
    SQLiteAgentStore,
    StaticAgentClient,
    ToolResult,
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
        "tool_authorization_checked",
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


def test_operator_console_exposes_test_observability_page() -> None:
    app = load_app_module()

    page = app.test_observability_html()
    manifest_text = app.index_html()

    assert "测试与回放" in page
    assert "重跑确定性并发测试" in page
    assert "调用真实 DeepSeek 回放" in page
    assert "record_candidate_reply" in page
    assert "/api/test-observability/run" in page
    assert 'href="/tests"' in manifest_text


def test_observability_runner_rejects_non_allowlisted_commands() -> None:
    app = load_app_module()

    try:
        app.run_fixed_suite("rm -rf /")
    except ValueError as exc:
        assert "unsupported test suite" in str(exc)
    else:
        raise AssertionError("arbitrary commands must not reach the test runner")


def test_main_app_does_not_expose_legacy_trial_defaults() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "boss_trial" not in text
    assert "runtime_trial" in text


def test_api_message_builder_requires_explicit_sender_identity() -> None:
    app = load_app_module()

    message, missing_fields = app.build_api_user_message({"text": "现在有人吗"})

    assert message is None
    assert missing_fields == ["conversation_id", "sender_id", "sender_name"]


def test_api_message_builder_preserves_explicit_identity_and_quote() -> None:
    app = load_app_module()

    message, missing_fields = app.build_api_user_message(
        {
            "conversation_id": "runtime_trial",
            "sender_id": "wang02",
            "sender_name": "王哥",
            "text": "可以",
            "quoted_message": {
                "message_id": "wechat_msg_001",
                "text": "七点三缺一，打吗？",
                "business_ref_type": "outbound_message_draft",
                "business_ref_id": "draft_001",
            },
        }
    )

    assert missing_fields == []
    assert message is not None
    assert message.conversation_id == "runtime_trial"
    assert message.sender_id == "wang02"
    assert message.sender_name == "王哥"
    assert message.text == "可以"
    assert message.quoted_message is not None
    assert message.quoted_message.message_id == "wechat_msg_001"
    assert message.quoted_message.business_ref_type == "outbound_message_draft"
    assert message.quoted_message.business_ref_id == "draft_001"


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
    assert "compatibility_packages" not in manifest
    assert manifest["legacy_reference_only"] is True
    assert manifest["legacy_entrypoints"]["legacy_analyze_endpoint"] == "not_exposed"
    assert manifest["legacy_entrypoints"]["default_runtime_entrypoint"] == "scripts/run_agent_app.py"
    assert "/api/message" in manifest["endpoints"]["message"]
    assert "legacy_endpoint_aliases" not in manifest
    assert "search_current_games" in manifest["available_tools"]
    assert "update_context_checkpoint" in manifest["available_tools"]
    assert manifest["runtime_config"]["progress_monitor"] == {
        "repeated_observation_limit": 2,
        "consecutive_no_progress_limit": 2,
        "max_replan_attempts": 1,
        "max_cycle_period": 3,
    }
    assert "/api/reset-state" in manifest["endpoints"]["reset_state"]
    assert "/api/analyze" not in app.json.dumps(manifest, ensure_ascii=False)


def test_operator_reset_state_clears_runtime_state_but_preserves_assets(tmp_path) -> None:
    app = load_app_module()
    store = SQLiteAgentStore(tmp_path / "agent_runtime_reset.sqlite3")
    trace = InMemoryTraceRecorder()
    runtime = AgentRuntime(
        llm_client=StaticAgentClient(
            [
                app.json.dumps(
                    {
                        "goal": "测试回复",
                        "objective_status": "completed",
                        "reasoning_summary": "模型直接回复。",
                        "reply_to_user": "收到。",
                        "tool_calls": [],
                        "needs_human": False,
                        "stop_reason": {
                            "can_stop": True,
                            "why": "测试场景模拟模型回复。",
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
    store.upsert_customer(CustomerProfile(customer_id="zhang", display_name="张哥"))
    runtime.handle_user_message(
        UserMessage(
            conversation_id="reset_conversation",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_reset_001",
        ),
        trace_id="trace_before_reset",
    )
    game, _ = store.create_game(
        conversation_id="reset_conversation",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong"},
        known_players=[],
        trace_id="trace_before_reset",
    )
    store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[{"customer_id": "ran", "display_name": "冉姐", "message_text": "冉姐，打吗？"}],
        trace_id="trace_before_reset",
    )
    store.upsert_conversation_checkpoint(
        conversation_id="reset_conversation",
        summary="用户正在组局",
        facts={"stake": "1"},
        open_questions=[],
        trace_id="trace_before_reset",
    )
    store.remember_result("tool_key", ToolResult(name="search_current_games", called=True, allowed=True))
    store.record_badcase({"reason": "保留样本"}, trace_id="trace_before_reset", conversation_id="reset_conversation")

    response = app.reset_runtime_state(runtime, {"trace_id": "trace_reset_operator"})

    assert response["deleted"]["games"] == 1
    assert response["deleted"]["invite_drafts"] == 1
    assert response["deleted"]["conversation_turns"] > 0
    assert response["deleted"]["conversation_checkpoints"] == 1
    assert response["deleted"]["idempotency_ledger"] > 0
    assert response["deleted"]["message_results"] == 1
    assert response["deleted"]["customers"] == 0
    assert response["deleted"]["badcases"] == 0
    assert store.games == {}
    assert store.invite_drafts == {}
    assert store.conversation_checkpoints == {}
    assert store.idempotent_result("tool_key") is None
    assert len(store.customers) == 1
    assert len(store.badcases) == 1
    steps = [event.step for event in trace.get_trace("trace_reset_operator")]
    assert steps == ["operator_reset_state_requested", "operator_reset_state_completed"]


def test_index_html_has_reset_state_button() -> None:
    app = load_app_module()
    html = app.index_html()

    assert "清空状态和记忆" in html
    assert "/api/reset-state" in html
    assert "客户画像、badcase/eval 和日志" in html


def test_log_tail_exposes_trace_log_path(tmp_path, monkeypatch) -> None:
    app = load_app_module()
    trace_path = tmp_path / "agent_runtime_trace.log"
    trace_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    monkeypatch.setattr(app, "TRACE_PATH", trace_path)

    assert app.tail_trace_log(2) == ["line2", "line3"]


def test_human_approved_invite_is_sent_once_and_persisted(tmp_path, monkeypatch) -> None:
    app = load_app_module()
    store = SQLiteAgentStore(tmp_path / "invite_delivery.sqlite3")
    trace = InMemoryTraceRecorder()
    runtime = AgentRuntime(
        llm_client=StaticAgentClient([]),
        store=store,
        tool_gateway=ToolGateway(store=store, trace_recorder=trace),
        trace_recorder=trace,
    )
    game, _ = store.create_game(
        conversation_id="invite_delivery_conversation",
        organizer_id="customer_a",
        organizer_name="客户A",
        requirement={"game_type": "hangzhou_mahjong", "start_time_kind": "asap_when_full"},
        known_players=[{"customer_id": "customer_a", "display_name": "客户A"}],
        trace_id="trace_invite_delivery_setup",
    )
    drafts, _ = store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[
            {
                "customer_id": "customer_b",
                "display_name": "客户B",
                "message_text": "0.5无烟，打吗？",
                "metadata": {"content_review_approved": True, "content_review_trace_id": "trace_review"},
            }
        ],
        trace_id="trace_invite_delivery_setup",
    )
    calls: list[tuple[str, dict | None]] = []

    def fake_request(path: str, *, payload=None, timeout_seconds=3.0):
        calls.append((path, payload))
        return {"send_channel_enabled": True} if path == "/health" else {"ok": True}

    monkeypatch.setattr(app, "request_local_json", fake_request)
    first = app.handle_invite_draft_action(
        runtime,
        {"draft_id": drafts[0].draft_id, "action": "approve_send", "trace_id": "trace_human_approve"},
    )
    second = app.handle_invite_draft_action(
        runtime,
        {"draft_id": drafts[0].draft_id, "action": "approve_send", "trace_id": "trace_human_retry"},
    )

    assert first["sent"] is True
    assert second["deduplicated"] is True
    assert store.invite_drafts[drafts[0].draft_id].status.value == "sent"
    assert [path for path, _ in calls] == ["/health", "/send"]
