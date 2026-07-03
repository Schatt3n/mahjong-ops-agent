from __future__ import annotations

import json
import importlib.util
import sys
import threading
import time
from pathlib import Path
from typing import Any

from mahjong_agent_runtime import (
    AgentRuntime,
    CustomerProfile,
    InMemoryAgentStore,
    InMemoryTraceRecorder,
    JsonlTraceRecorder,
    SQLiteAgentStore,
    StaticAgentClient,
    ToolCall,
    ToolGateway,
    ToolResult,
    TokenBudget,
    UserMessage,
)
from mahjong_agent_runtime.tracing import trace_steps, validate_trace


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_SCRIPT = ROOT / "scripts" / "verify_agent_runtime_boundary.py"


def load_boundary_module():
    spec = importlib.util.spec_from_file_location("verify_agent_runtime_boundary_for_test", BOUNDARY_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_main_chain_does_not_import_legacy_parser_workflow_or_guard() -> None:
    forbidden = [
        "mahjong_agent_v2",
        "from mahjong_agent.",
        "import mahjong_agent.",
        "reply_guard",
        "workflow",
        "semantic_resolver",
        "trial_",
        "responder",
    ]
    for path in (ROOT / "src" / "mahjong_agent_runtime").glob("**/*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path} contains forbidden legacy token {token!r}"


def test_runtime_boundary_script_rejects_semantic_patch_code(tmp_path) -> None:
    module = load_boundary_module()
    bad_file = tmp_path / "bad_runtime_semantic_patch.py"
    bad_file.write_text(
        "def patch(text):\n"
        "    return re.sub('0，5', '0.5', text)\n",
        encoding="utf-8",
    )

    violations = module.verify_files([bad_file])

    messages = "\n".join(violation.message for violation in violations)
    assert "semantic boundary violation" in messages
    assert "正则替换修麻将语义" in messages
    assert "0.5 口误 badcase" in messages


def test_runtime_boundary_script_rejects_legacy_analyze_endpoint_in_entrypoint(tmp_path, monkeypatch) -> None:
    module = load_boundary_module()
    bad_entrypoint = tmp_path / "run_agent_runtime_app.py"
    bad_entrypoint.write_text(
        "def route(parsed):\n"
        "    if parsed.path == '/api/analyze':\n"
        "        return 'legacy analyze'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "RUNTIME_ENTRYPOINTS", (bad_entrypoint,))

    violations = module.verify_files([bad_entrypoint])

    messages = "\n".join(violation.message for violation in violations)
    assert "entrypoint boundary violation" in messages
    assert "旧试用台 analyze 接口" in messages


def test_runtime_boundary_script_passes_current_main_chain() -> None:
    module = load_boundary_module()

    assert module.verify_files() == []


def test_runtime_default_eval_runner_only_targets_current_main_chain() -> None:
    runner = (ROOT / "scripts" / "run_evals.py").read_text(encoding="utf-8")
    assert "verify_agent_runtime_boundary.py" in runner
    assert "run_agent_runtime_eval.py" in runner
    assert "tests/test_agent_runtime.py" in runner
    assert "tests/test_agent_runtime_v3.py" not in runner
    assert "tests/test_agent_v3_app.py" not in runner
    assert "run_agent_runtime_v3_eval.py" not in runner
    assert "verify_agent_runtime_v3_boundary.py" not in runner
    assert "verify_agent_runtime_v2_boundary.py" not in runner
    assert "run_agent_runtime_v2_eval.py" not in runner
    assert "run_controlled_workflow_eval.py" not in runner
    assert "run_scenario_eval.py" not in runner


def test_runtime_lets_model_drive_tool_sequence_until_final_reply() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = PlanningClient(store)
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_test",
            sender_id="zhang",
            sender_name="张哥",
            text="通宵1块有人吗？没有就帮我组一个",
            message_id="msg_runtime_drive_001",
        ),
        trace_id="trace_drive_001",
    )

    assert result.final_reply == "好的，我帮你问问，有消息跟你说。"
    assert [call.name for action in result.actions for call in action.tool_calls] == [
        "search_current_games",
        "create_game",
        "search_customers",
        "create_invite_drafts",
    ]
    assert len(store.games) == 1
    assert len(store.invite_drafts) == 2
    assert [draft.message_text for draft in store.invite_drafts.values()] == [
        "冉姐，1块通宵，打吗？",
        "何哥，1块通宵，打吗？",
    ]
    events = trace.get_trace("trace_drive_001")
    assert validate_trace(events)["complete"] is True
    steps = trace_steps(events)
    assert steps.count("llm_prompt") == 5
    assert steps.count("tool_called") == 4
    assert "state_transition" in steps
    prompts = [event.content for event in events if event.step == "llm_prompt"]
    last_payload = json.loads(prompts[-1]["messages"][1]["content"])
    assert last_payload["previous_tool_results"][0]["name"] == "create_invite_drafts"


def test_runtime_backend_does_not_interpret_short_confirmation_as_create_game() -> None:
    store = seeded_store()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="模型选择只回复，不调用工具。",
                reply_to_user="好的。",
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_no_backend_semantic",
            sender_id="zhang",
            sender_name="张哥",
            text="组",
            message_id="msg_runtime_no_backend_semantic",
        ),
        trace_id="trace_no_backend_semantic",
    )

    assert result.final_reply == "好的。"
    assert result.tool_results == []
    assert store.games == {}
    assert len(client.calls) == 1


def test_runtime_tool_schema_error_is_fed_back_to_model_not_repaired_by_backend() -> None:
    store = seeded_store()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                tool_calls=[
                    {
                        "name": "create_invite_drafts",
                        "arguments": {"invitations": []},
                        "reason": "故意缺 game_id，验证工具错误回喂模型。",
                    }
                ],
            ),
            action_json(
                objective_status="waiting_user",
                reasoning_summary="上一步工具返回 schema 错误，模型决定等待人工补充。",
                reply_to_user="我先确认一下。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_schema_error",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我问问",
            message_id="msg_runtime_schema_error",
        ),
        trace_id="trace_schema_error",
    )

    assert result.tool_results[0].error == "missing required argument: game_id"
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert second_prompt["previous_tool_results"][0]["error"] == "missing required argument: game_id"
    assert store.games == {}
    assert result.final_reply == "我先确认一下。"


def test_runtime_schema_rejects_empty_invite_draft_list_without_state_change() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="runtime_empty_invites",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="setup_empty_invites",
    )
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型错误地请求创建空邀约草稿。",
                tool_calls=[
                    {
                        "name": "create_invite_drafts",
                        "arguments": {"game_id": game.game_id, "invitations": []},
                        "reason": "验证空数组不会产生空副作用。",
                    }
                ],
            ),
            action_json(
                objective_status="waiting_user",
                reasoning_summary="工具 schema 拒绝空 invitations，模型需要重新规划。",
                reply_to_user="我先重新确认一下要问谁。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_empty_invites",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我问问",
            message_id="msg_runtime_empty_invites",
        ),
        trace_id="trace_empty_invites",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert result.tool_results[0].error == "invitations must contain at least 1 item(s)"
    assert store.games[game.game_id].status.value == "forming"
    assert store.invite_drafts == {}
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert second_prompt["previous_tool_results"][0]["error"] == "invitations must contain at least 1 item(s)"
    assert result.final_reply == "我先重新确认一下要问谁。"


def test_runtime_schema_rejects_empty_outbound_message_draft_list() -> None:
    store = seeded_store()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型错误地请求创建空外发草稿。",
                tool_calls=[
                    {
                        "name": "create_outbound_message_drafts",
                        "arguments": {"drafts": []},
                        "reason": "验证空数组不会让模型假装已经生成草稿。",
                    }
                ],
            ),
            action_json(
                objective_status="waiting_user",
                reasoning_summary="工具 schema 拒绝空 drafts，模型需要重新规划。",
                reply_to_user="我先重新生成一版草稿。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_empty_outbound",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我回一句",
            message_id="msg_runtime_empty_outbound",
        ),
        trace_id="trace_empty_outbound",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert result.tool_results[0].error == "drafts must contain at least 1 item(s)"
    assert store.outbound_message_drafts == {}
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert second_prompt["previous_tool_results"][0]["error"] == "drafts must contain at least 1 item(s)"
    assert result.final_reply == "我先重新生成一版草稿。"


def test_runtime_create_game_requires_explicit_organizer_identity() -> None:
    store = seeded_store()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型尝试建局，但没有显式提供组织者身份。",
                tool_calls=[
                    {
                        "name": "create_game",
                        "arguments": {
                            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1"},
                            "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
                        },
                        "reason": "验证后端不会用 sender_id 脑补 organizer。",
                    }
                ],
            ),
            action_json(
                objective_status="waiting_user",
                reasoning_summary="previous_tool_results 返回 organizer_id 缺失，模型需要修正工具参数或转人工。",
                reply_to_user="我先确认一下。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_create_game_identity",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_runtime_create_game_identity",
        ),
        trace_id="trace_create_game_identity",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert result.tool_results[0].error == "missing required argument: organizer_id"
    assert store.games == {}
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert second_prompt["previous_tool_results"][0]["error"] == "missing required argument: organizer_id"
    assert result.final_reply == "我先确认一下。"


def test_runtime_tool_schema_rejects_empty_critical_strings_without_backend_defaults() -> None:
    store = seeded_store()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型提供了空 organizer_id，后端不能替换成 sender_id。",
                tool_calls=[
                    {
                        "name": "create_game",
                        "arguments": {
                            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1"},
                            "organizer_id": "",
                            "organizer_name": "张哥",
                            "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
                        },
                        "reason": "验证空关键字段会被 schema 拒绝。",
                    }
                ],
            ),
            action_json(
                objective_status="waiting_user",
                reasoning_summary="previous_tool_results 返回 organizer_id 为空，模型需要重新给出合法参数。",
                reply_to_user="我先确认一下。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_empty_organizer_identity",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_runtime_empty_organizer_identity",
        ),
        trace_id="trace_empty_organizer_identity",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].error == "organizer_id must have length >= 1"
    assert store.games == {}
    assert result.final_reply == "我先确认一下。"


def test_runtime_action_contract_rejects_invalid_top_level_types_before_tools() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            json.dumps(
                {
                    "goal": "测试非法顶层类型",
                    "objective_status": "needs_tool",
                    "reasoning_summary": "needs_human 不是布尔值，badcase 不是对象。",
                    "reply_to_user": "",
                    "tool_calls": [
                        {
                            "name": "create_game",
                            "arguments": {
                                "requirement": {"game_type": "hangzhou_mahjong", "stake": "1"},
                                "organizer_id": "zhang",
                                "organizer_name": "张哥",
                                "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
                            },
                            "reason": "如果合同失败，这个工具不应执行。",
                        }
                    ],
                    "needs_human": "false",
                    "badcase": "badcase should be object",
                },
                ensure_ascii=False,
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_action_contract",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_runtime_action_contract",
        ),
        trace_id="trace_action_contract",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.tool_results == []
    assert store.games == {}
    contract_event = next(event for event in trace.get_trace("trace_action_contract") if event.step == "action_contract_error")
    assert "needs_human must be boolean" in contract_event.content["errors"]
    assert "badcase side-channel is not allowed; call record_badcase tool instead" in contract_event.content["errors"]


def test_runtime_action_contract_rejects_badcase_side_channel_before_audit_write() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="模型错误地把 badcase 放在旁路字段。",
                reply_to_user="我记下来了。",
                badcase={
                    "reason": "旁路 badcase 不应该被 runtime 自动落库",
                    "input": {"text": "组"},
                    "actual": {"reply": "留意"},
                    "expected": {"behavior": "显式调用 record_badcase 工具"},
                },
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_badcase_side_channel",
            sender_id="zhang",
            sender_name="张哥",
            text="组",
            message_id="msg_runtime_badcase_side_channel",
        ),
        trace_id="trace_badcase_side_channel",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.tool_results == []
    assert store.badcases == []
    contract_event = next(event for event in trace.get_trace("trace_badcase_side_channel") if event.step == "action_contract_error")
    assert "badcase side-channel is not allowed; call record_badcase tool instead" in contract_event.content["errors"]


def test_runtime_action_contract_requires_human_status_to_set_human_flag() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_human",
                reasoning_summary="模型说需要人工，但忘了设置 needs_human=true。",
                reply_to_user="我先确认一下。",
                needs_human=False,
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_human_contract",
            sender_id="zhang",
            sender_name="张哥",
            text="这个需要人工吧",
            message_id="msg_runtime_human_contract",
        ),
        trace_id="trace_human_contract",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    contract_event = next(event for event in trace.get_trace("trace_human_contract") if event.step == "action_contract_error")
    assert "needs_human objective_status requires needs_human=true" in contract_event.content["errors"]


def test_runtime_action_contract_rejects_terminal_status_without_customer_reply() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="模型声称完成，但没有给客户可见回复。",
                reply_to_user="   ",
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_empty_terminal_reply",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我看看",
            message_id="msg_runtime_empty_terminal_reply",
        ),
        trace_id="trace_empty_terminal_reply",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    contract_event = next(event for event in trace.get_trace("trace_empty_terminal_reply") if event.step == "action_contract_error")
    assert "completed requires non-empty reply_to_user" in contract_event.content["errors"]


def test_runtime_action_contract_requires_auditable_stop_reason() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            json.dumps(
                {
                    "goal": "测试提前停止",
                    "objective_status": "completed",
                    "reasoning_summary": "模型直接说完成，但没有解释为什么可以停。",
                    "reply_to_user": "好的，我先看看。",
                    "tool_calls": [],
                    "needs_human": False,
                    "badcase": None,
                },
                ensure_ascii=False,
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_stop_reason_contract",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_runtime_stop_reason_contract",
        ),
        trace_id="trace_stop_reason_contract",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.tool_results == []
    assert store.games == {}
    contract_event = next(event for event in trace.get_trace("trace_stop_reason_contract") if event.step == "action_contract_error")
    assert "missing required key: stop_reason" in contract_event.content["errors"]
    assert "stop_reason must be object" in contract_event.content["errors"]


def test_runtime_action_contract_rejects_unknown_tool_calls_and_human_flag_conflict() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="unknown",
                reasoning_summary="模型状态不自洽：unknown 还想执行工具，并且 needs_human 标志和状态冲突。",
                reply_to_user="我没看懂，先确认一下。",
                tool_calls=[
                    {
                        "name": "create_game",
                        "arguments": {"requirement": {"game_type": "hangzhou_mahjong"}},
                        "reason": "状态 unknown 时不应该执行这个副作用工具。",
                    }
                ],
                needs_human=True,
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_unknown_contract",
            sender_id="zhang",
            sender_name="张哥",
            text="随便看看",
            message_id="msg_runtime_unknown_contract",
        ),
        trace_id="trace_unknown_contract",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.tool_results == []
    assert store.games == {}
    contract_event = next(event for event in trace.get_trace("trace_unknown_contract") if event.step == "action_contract_error")
    assert "unknown must not include tool_calls" in contract_event.content["errors"]
    assert "needs_human=true requires objective_status=needs_human" in contract_event.content["errors"]


def test_runtime_action_contract_rejects_untraceable_tool_call_fields() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            json.dumps(
                {
                    "goal": "测试工具调用合同",
                    "objective_status": "needs_tool",
                    "reasoning_summary": "模型给出的工具调用缺少可审计字段。",
                    "reply_to_user": "我先看看。",
                    "tool_calls": [
                        {
                            "name": "search_current_games",
                            "arguments": {"requirement": {}},
                            "reason": "",
                        },
                        {
                            "name": "search_customers",
                            "reason": "缺少 arguments 时不能默认为空对象。",
                        },
                        {
                            "name": "create_game",
                            "arguments": {"requirement": {"game_type": "hangzhou_mahjong"}},
                            "reason": "验证 idempotency_key 类型。",
                            "idempotency_key": 123,
                        },
                    ],
                    "needs_human": False,
                    "badcase": None,
                },
                ensure_ascii=False,
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_untraceable_tool_call",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我看看",
            message_id="msg_runtime_untraceable_tool_call",
        ),
        trace_id="trace_untraceable_tool_call",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.tool_results == []
    assert store.games == {}
    contract_event = next(event for event in trace.get_trace("trace_untraceable_tool_call") if event.step == "action_contract_error")
    assert "needs_tool requires empty reply_to_user" in contract_event.content["errors"]
    assert "tool_calls[1].reason is required" in contract_event.content["errors"]
    assert "tool_calls[2].arguments is required" in contract_event.content["errors"]
    assert "tool_calls[3].idempotency_key must be string or null" in contract_event.content["errors"]


def test_runtime_invalid_candidate_status_is_rejected_by_tool_schema() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="runtime_candidate_schema",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="setup_candidate_schema",
    )
    drafts, _ = store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[{"customer_id": "ran", "display_name": "冉姐", "message_text": "冉姐，1块，打吗？"}],
        trace_id="setup_candidate_schema",
    )
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型给了非法候选人状态。",
                tool_calls=[
                    {
                        "name": "record_candidate_reply",
                        "arguments": {
                            "game_id": game.game_id,
                            "customer_id": "ran",
                            "display_name": "冉姐",
                            "status": "maybe",
                        },
                        "reason": "验证非法状态不会被后端脑补。",
                    }
                ],
            ),
            action_json(
                objective_status="waiting_user",
                reasoning_summary="工具 schema 拒绝了非法 status，等待模型下一步修正或追问。",
                reply_to_user="我确认一下她到底来不来。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_candidate_schema",
            sender_id="ran",
            sender_name="冉姐",
            text="看情况吧",
            message_id="msg_runtime_candidate_schema",
        ),
        trace_id="trace_candidate_schema",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert "status must be one of" in (result.tool_results[0].error or "")
    assert [item.customer_id for item in store.games[game.game_id].participants] == ["zhang"]
    assert store.invite_drafts[drafts[0].draft_id].status.value == "pending_approval"
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert "status must be one of" in second_prompt["previous_tool_results"][0]["error"]
    assert result.final_reply == "我确认一下她到底来不来。"


def test_runtime_candidate_join_is_traced_and_persisted_as_state_transition(tmp_path) -> None:
    db_path = tmp_path / "agent_runtime_candidate_join.sqlite3"
    store = seeded_store(SQLiteAgentStore(db_path))
    game, _ = store.create_game(
        conversation_id="runtime_candidate_join",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="setup_candidate_join",
    )
    store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[{"customer_id": "ran", "display_name": "冉姐", "message_text": "冉姐，1块，打吗？"}],
        trace_id="setup_candidate_join",
    )
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="候选人明确确认，模型记录候选人加入。",
                tool_calls=[
                    {
                        "name": "record_candidate_reply",
                        "arguments": {
                            "game_id": game.game_id,
                            "customer_id": "ran",
                            "display_name": "冉姐",
                            "status": "accepted",
                        },
                        "reason": "候选人确认参加，记录状态变化。",
                    }
                ],
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="候选人已加入局。",
                reply_to_user="好的，加你进来了。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_candidate_join",
            sender_id="ran",
            sender_name="冉姐",
            text="可以",
            message_id="msg_runtime_candidate_join",
        ),
        trace_id="trace_candidate_join",
    )

    assert any(item.customer_id == "ran" for item in store.games[game.game_id].participants)
    participant_transition = next(
        transition
        for transition in result.state_transitions
        if transition.entity_type == "game_participant" and transition.entity_id == f"{game.game_id}:ran"
    )
    assert participant_transition.from_status is None
    assert participant_transition.to_status == "confirmed"
    trace_transition = next(
        event
        for event in trace.get_trace("trace_candidate_join")
        if event.step == "state_transition" and event.content["entity_type"] == "game_participant"
    )
    assert trace_transition.content["entity_id"] == f"{game.game_id}:ran"
    reopened = SQLiteAgentStore(db_path)
    persisted = [
        transition
        for transition in reopened.transitions
        if transition.entity_type == "game_participant" and transition.entity_id == f"{game.game_id}:ran"
    ]
    assert len(persisted) == 1
    assert persisted[0].to_status == "confirmed"
    assert result.final_reply == "好的，加你进来了。"


def test_runtime_outbound_message_draft_is_tool_driven_and_persisted(tmp_path) -> None:
    db_path = tmp_path / "agent_runtime_outbound_draft.sqlite3"
    store = seeded_store(SQLiteAgentStore(db_path))
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型决定先生成待审批外发草稿，而不是声称已经发送。",
                tool_calls=[
                    {
                        "name": "create_outbound_message_drafts",
                        "arguments": {
                            "drafts": [
                                {
                                    "recipient_id": "zhang",
                                    "recipient_name": "张哥",
                                    "channel": "console",
                                    "message_text": "好的，我先帮你问问，有消息跟你说。",
                                    "purpose": "reply_to_organizer",
                                    "metadata": {"game_context": "runtime_outbound_draft"},
                                }
                            ]
                        },
                        "reason": "把客户可见回复作为待审批草稿落库，便于后续人工审批和多通道发送。",
                    }
                ],
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="待审批外发草稿已创建。",
                reply_to_user="已生成待审批回复草稿。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_outbound_draft",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_runtime_outbound_draft",
        ),
        trace_id="trace_outbound_draft",
    )

    assert result.final_reply == "已生成待审批回复草稿。"
    assert [tool.name for tool in result.tool_results] == ["create_outbound_message_drafts"]
    assert len(store.outbound_message_drafts) == 1
    draft = next(iter(store.outbound_message_drafts.values()))
    assert draft.recipient_id == "zhang"
    assert draft.channel == "console"
    assert draft.status.value == "pending_approval"
    assert draft.message_text == "好的，我先帮你问问，有消息跟你说。"
    transition = next(
        item
        for item in result.state_transitions
        if item.entity_type == "outbound_message_draft" and item.entity_id == draft.draft_id
    )
    assert transition.to_status == "pending_approval"
    trace_transition = next(
        event
        for event in trace.get_trace("trace_outbound_draft")
        if event.step == "state_transition" and event.content["entity_type"] == "outbound_message_draft"
    )
    assert trace_transition.content["entity_id"] == draft.draft_id
    reopened = SQLiteAgentStore(db_path)
    assert len(reopened.outbound_message_drafts) == 1
    persisted = next(iter(reopened.outbound_message_drafts.values()))
    assert persisted.message_text == draft.message_text
    assert persisted.metadata["game_context"] == "runtime_outbound_draft"


def test_runtime_illegal_game_status_transition_is_rejected_by_state_machine() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="runtime_state_machine",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="setup_state_machine",
    )
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型请求非法状态迁移。",
                tool_calls=[
                    {
                        "name": "update_game_status",
                        "arguments": {"game_id": game.game_id, "status": "finished", "reason": "model requested impossible close"},
                        "reason": "验证状态机拒绝非法迁移。",
                    }
                ],
            ),
            action_json(
                objective_status="needs_human",
                reasoning_summary="状态机拒绝非法迁移，交给人工确认。",
                reply_to_user="这个我先确认一下。",
                needs_human=True,
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_state_machine",
            sender_id="zhang",
            sender_name="张哥",
            text="结束掉吧",
            message_id="msg_runtime_state_machine",
        ),
        trace_id="trace_state_machine",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert "illegal game status transition: forming->finished" in (result.tool_results[0].error or "")
    assert store.games[game.game_id].status.value == "forming"
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert "illegal game status transition" in second_prompt["previous_tool_results"][0]["error"]
    assert result.final_reply == "这个我先确认一下。"


def test_runtime_tool_permission_denial_is_fed_back_to_model_without_side_effect() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="runtime_permission",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="setup_permission",
    )
    trace = InMemoryTraceRecorder()
    gateway = ToolGateway(
        store=store,
        trace_recorder=trace,
        allowed_execution_modes={"read_only", "state_write", "audit_write"},
    )
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型请求创建邀约草稿，但当前权限禁止 draft_write。",
                tool_calls=[
                    {
                        "name": "create_invite_drafts",
                        "arguments": {
                            "game_id": game.game_id,
                            "invitations": [{"customer_id": "ran", "display_name": "冉姐", "message_text": "冉姐，1块，打吗？"}],
                        },
                        "reason": "验证权限拦截。",
                    }
                ],
            ),
            action_json(
                objective_status="needs_human",
                reasoning_summary="工具权限拒绝创建草稿，交给人工或等待配置恢复。",
                reply_to_user="我先确认一下能不能发邀约。",
                needs_human=True,
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, tool_gateway=gateway, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_permission",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我问冉姐",
            message_id="msg_runtime_permission",
        ),
        trace_id="trace_permission",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert result.tool_results[0].error == "tool execution_mode not allowed: draft_write"
    assert store.invite_drafts == {}
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert second_prompt["previous_tool_results"][0]["error"] == "tool execution_mode not allowed: draft_write"
    permission_events = [event for event in trace.get_trace("trace_permission") if event.step == "tool_permission_checked"]
    assert permission_events[0].level == "WARN"
    assert permission_events[0].content["allowed"] is False
    assert validate_trace(trace.get_trace("trace_permission"))["complete"] is True
    assert result.final_reply == "我先确认一下能不能发邀约。"


def test_runtime_tool_handler_exception_is_traced_and_fed_back_without_side_effect() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    gateway = ToolGateway(store=store, trace_recorder=trace)

    def failing_handler(call_arg, trace_id: str, conversation_id: str, sender_id: str, sender_name: str):
        raise RuntimeError("database temporarily unavailable")

    gateway.tools["create_game"].handler = failing_handler
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型决定建局。",
                tool_calls=[
                    {
                        "name": "create_game",
                        "arguments": {
                            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1"},
                            "organizer_id": "zhang",
                            "organizer_name": "张哥",
                            "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
                        },
                        "reason": "验证工具内部异常会进入 trace 和下一轮上下文。",
                    }
                ],
            ),
            action_json(
                objective_status="needs_human",
                reasoning_summary="工具返回 RuntimeError，模型不能声称已建局。",
                reply_to_user="这个我先确认一下。",
                needs_human=True,
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, tool_gateway=gateway, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_tool_exception",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_runtime_tool_exception",
        ),
        trace_id="trace_tool_exception",
    )

    assert result.final_reply == "这个我先确认一下。"
    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert result.tool_results[0].error == "RuntimeError: database temporarily unavailable"
    assert store.games == {}
    exception_event = next(event for event in trace.get_trace("trace_tool_exception") if event.step == "tool_exception")
    assert exception_event.level == "ERROR"
    assert exception_event.content["error_type"] == "RuntimeError"
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert second_prompt["previous_tool_results"][0]["error"] == "RuntimeError: database temporarily unavailable"
    assert validate_trace(trace.get_trace("trace_tool_exception"))["complete"] is True


def test_runtime_budget_denial_happens_before_llm_call_and_has_complete_trace() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient([])
    runtime = AgentRuntime(
        llm_client=client,
        store=store,
        trace_recorder=trace,
        token_budget=TokenBudget(max_tokens_per_call=1, max_calls_per_turn=8),
    )

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_budget",
            sender_id="zhang",
            sender_name="张哥",
            text="通宵1块有人吗？没有就帮我组一个",
            message_id="msg_runtime_budget",
        ),
        trace_id="trace_budget",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.actions == []
    assert result.tool_results == []
    assert client.calls == []
    events = trace.get_trace("trace_budget")
    steps = trace_steps(events)
    assert "llm_prompt" in steps
    assert "budget_checked" in steps
    assert "llm_response" not in steps
    budget_event = next(event for event in events if event.step == "budget_checked")
    assert budget_event.content["allowed"] is False
    assert "single call token estimate exceeded" in budget_event.content["reason"]
    assert validate_trace(events)["complete"] is True


def test_runtime_llm_error_has_complete_trace_and_no_tool_side_effect() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    runtime = AgentRuntime(
        llm_client=FailingAgentClient(RuntimeError("llm timeout")),
        store=store,
        trace_recorder=trace,
    )

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_llm_error",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_runtime_llm_error",
        ),
        trace_id="trace_llm_error",
    )

    events = trace.get_trace("trace_llm_error")
    steps = trace_steps(events)
    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.actions == []
    assert result.tool_results == []
    assert store.games == {}
    assert "llm_prompt" in steps
    assert "llm_error" in steps
    assert "llm_response" not in steps
    assert "tool_called" not in steps
    assert validate_trace(events)["complete"] is True


def test_runtime_trace_completeness_requires_context_packing_audit() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="模型直接回复。",
                reply_to_user="好的。",
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_context_pack_trace",
            sender_id="zhang",
            sender_name="张哥",
            text="好的",
            message_id="msg_runtime_context_pack_trace",
        ),
        trace_id="trace_context_pack_required",
    )

    events = trace.get_trace("trace_context_pack_required")
    assert validate_trace(events)["complete"] is True
    without_context_packed = [event for event in events if event.step != "context_packed"]
    completeness = validate_trace(without_context_packed)
    assert completeness["complete"] is False
    assert "context_packed" in completeness["missing"]


def test_runtime_duplicate_message_id_returns_cached_result_without_reexecuting_side_effects() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = PlanningClient(store)
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)
    message = UserMessage(
        conversation_id="runtime_message_idempotency",
        sender_id="zhang",
        sender_name="张哥",
        text="通宵1块有人吗？没有就帮我组一个",
        message_id="msg_runtime_message_idempotency",
    )

    first = runtime.handle_user_message(message, trace_id="trace_message_idempotency_1")
    second = runtime.handle_user_message(message, trace_id="trace_message_idempotency_2")

    assert first.final_reply == "好的，我帮你问问，有消息跟你说。"
    assert second.final_reply == first.final_reply
    assert second.trace_id == first.trace_id
    assert len(client.calls) == 5
    assert len(store.games) == 1
    assert len(store.invite_drafts) == 2
    dedupe_events = trace.get_trace("trace_message_idempotency_2")
    dedupe_steps = trace_steps(dedupe_events)
    assert dedupe_steps == ["user_input", "message_deduplicated", "final_output"]
    assert validate_trace(dedupe_events)["complete"] is True
    dedupe_event = next(event for event in dedupe_events if event.step == "message_deduplicated")
    assert dedupe_event.content["original_trace_id"] == "trace_message_idempotency_1"


def test_runtime_trace_completeness_rejects_deduplicated_trace_with_execution_steps() -> None:
    trace = InMemoryTraceRecorder()
    trace_id = "trace_bad_deduplicated_execution"
    trace.record(trace_id, "user_input", {"message": {"message_id": "msg_duplicate"}})
    trace.record(trace_id, "message_deduplicated", {"message_id": "msg_duplicate", "original_trace_id": "trace_original"})
    trace.record(trace_id, "llm_prompt", {"messages": []})
    trace.record(trace_id, "final_output", {"reply": "cached"})

    completeness = validate_trace(trace.get_trace(trace_id))

    assert completeness["complete"] is False
    assert "deduplicated_trace_must_not_execute_llm_or_tools" in completeness["missing"]


def test_runtime_concurrent_duplicate_message_id_serializes_and_deduplicates_side_effects() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = PlanningClient(store)
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)
    message = UserMessage(
        conversation_id="runtime_concurrent_message",
        sender_id="zhang",
        sender_name="张哥",
        text="通宵1块有人吗？没有就帮我组一个",
        message_id="msg_runtime_concurrent_message",
    )
    start = threading.Barrier(3)
    results: dict[str, Any] = {}
    errors: list[BaseException] = []

    def worker(trace_id: str) -> None:
        try:
            start.wait()
            results[trace_id] = runtime.handle_user_message(message, trace_id=trace_id)
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=("trace_concurrent_message_1",)),
        threading.Thread(target=worker, args=("trace_concurrent_message_2",)),
    ]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 2
    assert len({result.trace_id for result in results.values()}) == 1
    assert len(client.calls) == 5
    assert len(store.games) == 1
    assert len(store.invite_drafts) == 2
    duplicate_traces = [
        trace_id
        for trace_id in results
        if "message_deduplicated" in trace_steps(trace.get_trace(trace_id))
    ]
    assert len(duplicate_traces) == 1
    duplicate_events = trace.get_trace(duplicate_traces[0])
    assert trace_steps(duplicate_events) == ["user_input", "message_deduplicated", "final_output"]
    assert validate_trace(duplicate_events)["complete"] is True


def test_runtime_token_budget_is_isolated_per_concurrent_message_turn() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = TwoStepBarrierClient(expected_first_calls=2)
    runtime = AgentRuntime(
        llm_client=client,
        store=store,
        trace_recorder=trace,
        token_budget=TokenBudget(max_tokens_per_call=24_000, max_calls_per_turn=2),
    )
    start = threading.Barrier(3)
    results: dict[str, Any] = {}
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            start.wait()
            results[f"trace_budget_isolation_{index}"] = runtime.handle_user_message(
                UserMessage(
                    conversation_id=f"runtime_budget_isolation_{index}",
                    sender_id=f"user_{index}",
                    sender_name=f"用户{index}",
                    text="先查一下，没有就回复我",
                    message_id=f"msg_runtime_budget_isolation_{index}",
                ),
                trace_id=f"trace_budget_isolation_{index}",
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in (1, 2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert sorted(results) == ["trace_budget_isolation_1", "trace_budget_isolation_2"]
    assert all(result.final_reply == "查过了，先这样回复。" for result in results.values())
    assert all(len(result.actions) == 2 for result in results.values())
    for trace_id in sorted(results):
        events = trace.get_trace(trace_id)
        assert validate_trace(events)["complete"] is True
        budget_events = [event for event in events if event.step == "budget_checked"]
        assert len(budget_events) == 2
        assert all(event.content["allowed"] is True for event in budget_events)
    assert runtime.token_budget.calls_this_turn == 0


def test_runtime_tool_gateway_serializes_concurrent_same_backend_idempotency_key() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    gateway = ToolGateway(store=store, trace_recorder=trace)
    call = ToolCall(
        name="create_game",
        arguments={
            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1", "user_visible_summary": "杭麻 1块"},
            "organizer_id": "zhang",
            "organizer_name": "张哥",
            "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
        },
        idempotency_key="model-key-should-not-win",
    )
    original_handler = gateway.tools["create_game"].handler
    execution_count = 0
    execution_count_lock = threading.Lock()

    def slow_handler(call_arg, trace_id: str, conversation_id: str, sender_id: str, sender_name: str):
        nonlocal execution_count
        with execution_count_lock:
            execution_count += 1
        time.sleep(0.05)
        return original_handler(call_arg, trace_id, conversation_id, sender_id, sender_name)

    gateway.tools["create_game"].handler = slow_handler
    start = threading.Barrier(3)
    results = []
    results_lock = threading.Lock()

    def worker(trace_id: str) -> None:
        start.wait()
        result = gateway.execute(
            call,
            trace_id=trace_id,
            conversation_id="runtime_concurrent_tool",
            sender_id="zhang",
            sender_name="张哥",
            step_index=101,
            source_message_id="msg_runtime_concurrent_tool",
        )
        with results_lock:
            results.append(result)

    threads = [
        threading.Thread(target=worker, args=("trace_concurrent_tool_1",)),
        threading.Thread(target=worker, args=("trace_concurrent_tool_2",)),
    ]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()

    assert execution_count == 1
    assert len(results) == 2
    assert len(store.games) == 1
    assert sorted(result.deduplicated for result in results) == [False, True]
    assert len({result.idempotency_key for result in results}) == 1
    assert all(
        result.idempotency_key.startswith("message:msg_runtime_concurrent_tool:tool:create_game:args:")
        for result in results
    )
    hit_values = []
    for trace_id in ("trace_concurrent_tool_1", "trace_concurrent_tool_2"):
        events = trace.get_trace(trace_id)
        hit_values.extend(event.content["hit"] for event in events if event.step == "tool_idempotency_checked")
    assert sorted(hit_values) == [False, True]


def test_runtime_tool_gateway_claims_idempotency_before_executing_side_effect(tmp_path) -> None:
    store = seeded_store(SQLiteAgentStore(tmp_path / "agent_runtime_tool_claim.sqlite3"))
    trace = InMemoryTraceRecorder()
    gateway = ToolGateway(store=store, trace_recorder=trace)
    call = ToolCall(
        name="create_game",
        arguments={
            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1"},
            "organizer_id": "zhang",
            "organizer_name": "张哥",
            "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
        },
        idempotency_key="model-key-ignored-by-backend-when-message-id-exists",
    )

    result = gateway.execute(
        call,
        trace_id="trace_tool_claim",
        conversation_id="runtime_tool_claim",
        sender_id="zhang",
        sender_name="张哥",
        step_index=101,
        source_message_id="msg_runtime_tool_claim",
    )

    assert result.called is True
    assert result.allowed is True
    assert len(store.games) == 1
    assert result.idempotency_key
    persisted = store.idempotent_result(result.idempotency_key)
    assert persisted is not None
    assert persisted.called is True
    assert persisted.result["game"]["organizer_id"] == "zhang"
    claim_event = next(event for event in trace.get_trace("trace_tool_claim") if event.step == "tool_idempotency_claimed")
    assert claim_event.content["claimed"] is True
    assert claim_event.content["idempotency_key"] == result.idempotency_key
    steps = trace_steps(trace.get_trace("trace_tool_claim"))
    assert steps == [
        "tool_gateway_received",
        "tool_idempotency_checked",
        "tool_definition_checked",
        "tool_schema_checked",
        "tool_permission_checked",
        "tool_idempotency_claimed",
        "tool_gateway_completed",
    ]


def test_runtime_sqlite_idempotency_claim_is_atomic_across_store_instances(tmp_path) -> None:
    db_path = tmp_path / "agent_runtime_claim_atomic.sqlite3"
    first_store = SQLiteAgentStore(db_path)
    second_store = SQLiteAgentStore(db_path)
    key = "message:shared:tool:create_game:args:same"
    first_claim = ToolResult(
        name="create_game",
        called=False,
        allowed=True,
        result={"idempotency_status": "claimed", "claimed_by_trace_id": "trace_first"},
        error="tool execution is already in progress for this idempotency key",
        idempotency_key=key,
    )
    second_claim = ToolResult(
        name="create_game",
        called=False,
        allowed=True,
        result={"idempotency_status": "claimed", "claimed_by_trace_id": "trace_second"},
        error="tool execution is already in progress for this idempotency key",
        idempotency_key=key,
    )

    acquired, existing = first_store.claim_idempotent_result(key, first_claim)
    duplicate_acquired, duplicate_existing = second_store.claim_idempotent_result(key, second_claim)

    assert acquired is True
    assert existing is None
    assert duplicate_acquired is False
    assert duplicate_existing is not None
    assert duplicate_existing.result["claimed_by_trace_id"] == "trace_first"

    final = ToolResult(
        name="create_game",
        called=True,
        allowed=True,
        result={"game": {"game_id": "game_claimed"}},
        idempotency_key=key,
    )
    first_store.remember_result(key, final)
    persisted = second_store.idempotent_result(key)
    assert persisted is not None
    assert persisted.called is True
    assert persisted.result["game"]["game_id"] == "game_claimed"


def test_runtime_jsonl_trace_is_structured_and_replayable(tmp_path) -> None:
    store = seeded_store()
    trace = JsonlTraceRecorder(tmp_path / "agent_runtime_trace.log")
    client = PlanningClient(store)
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_jsonl_trace",
            sender_id="zhang",
            sender_name="张哥",
            text="通宵1块有人吗？没有就帮我组一个",
            message_id="msg_runtime_jsonl_trace",
        ),
        trace_id="trace_jsonl_trace",
    )

    events = trace.get_trace("trace_jsonl_trace")
    steps = trace_steps(events)
    assert validate_trace(events)["complete"] is True
    assert "raw_log_line" not in steps
    assert "llm_prompt" in steps
    assert "llm_response" in steps
    assert "tool_called" in steps
    assert "tool_result" in steps
    assert "state_transition" in steps
    prompt_payload = json.loads(next(event for event in events if event.step == "llm_prompt").content["messages"][1]["content"])
    assert prompt_payload["runtime"] == "mahjong_agent_runtime"


def test_runtime_trace_completeness_requires_action_proposed_after_llm_success() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="模型直接回复。",
                reply_to_user="好的。",
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_trace_action",
            sender_id="zhang",
            sender_name="张哥",
            text="好的",
            message_id="msg_runtime_trace_action",
        ),
        trace_id="trace_action_required",
    )

    events = trace.get_trace("trace_action_required")
    assert validate_trace(events)["complete"] is True
    without_action = [event for event in events if event.step != "action_proposed"]
    completeness = validate_trace(without_action)
    assert completeness["complete"] is False
    assert "action_proposed" in completeness["missing"]


def test_runtime_trace_completeness_requires_state_transition_for_tool_side_effects() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型决定建局。",
                tool_calls=[
                    {
                        "name": "create_game",
                        "arguments": {
                            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1"},
                            "organizer_id": "zhang",
                            "organizer_name": "张哥",
                            "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
                        },
                        "reason": "创建待组局记录。",
                    }
                ],
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="建局完成。",
                reply_to_user="好的，我帮你问问。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_trace_transition",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_runtime_trace_transition",
        ),
        trace_id="trace_transition_required",
    )

    events = trace.get_trace("trace_transition_required")
    assert validate_trace(events)["complete"] is True
    without_transition = [event for event in events if event.step not in {"state_transition", "state_transition_replayed"}]
    completeness = validate_trace(without_transition)
    assert completeness["complete"] is False
    assert "state_transition" in completeness["missing"]


def test_runtime_sqlite_store_persists_runtime_state_and_idempotency(tmp_path) -> None:
    db_path = tmp_path / "agent_runtime.sqlite3"
    store = seeded_store(SQLiteAgentStore(db_path))
    trace = InMemoryTraceRecorder()
    client = PlanningClient(store)
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)
    message = UserMessage(
        conversation_id="runtime_sqlite",
        sender_id="zhang",
        sender_name="张哥",
        text="通宵1块有人吗？没有就帮我组一个",
        message_id="msg_runtime_sqlite_persist",
    )

    result = runtime.handle_user_message(message, trace_id="trace_sqlite_1")

    assert result.final_reply == "好的，我帮你问问，有消息跟你说。"
    assert len(store.games) == 1
    assert len(store.invite_drafts) == 2
    assert result.tool_results[0].idempotency_key

    reopened = SQLiteAgentStore(db_path)
    assert len(reopened.customers) == 3
    assert len(reopened.games) == 1
    assert len(reopened.invite_drafts) == 2
    assert len(reopened.transitions) >= 3
    assert len(reopened.recent_turns("runtime_sqlite")) >= 3
    assert reopened.idempotent_result(result.tool_results[0].idempotency_key) is not None

    cached_client = StaticAgentClient([])
    runtime_after_restart = AgentRuntime(
        llm_client=cached_client,
        store=reopened,
        trace_recorder=InMemoryTraceRecorder(),
    )
    cached = runtime_after_restart.handle_user_message(message, trace_id="trace_sqlite_2")

    assert cached.final_reply == result.final_reply
    assert cached_client.calls == []
    assert len(reopened.games) == 1
    assert reopened.idempotent_message_result("msg_runtime_sqlite_persist") is not None


def test_runtime_sqlite_store_persists_badcases_from_tool(tmp_path) -> None:
    store = seeded_store(SQLiteAgentStore(tmp_path / "agent_runtime_badcase.sqlite3"))
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型主动归档 badcase。",
                tool_calls=[
                    {
                        "name": "record_badcase",
                        "arguments": {
                            "reason": "测试回复不合适",
                            "input": {"text": "组"},
                            "actual": {"reply": "留意"},
                            "expected": {"reply": "应该继续规划"},
                        },
                        "reason": "记录评测样本。",
                    }
                ],
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="badcase 已记录。",
                reply_to_user="我记下来了。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_badcase",
            sender_id="zhang",
            sender_name="张哥",
            text="组",
            message_id="msg_runtime_badcase",
        ),
        trace_id="trace_badcase",
    )

    reopened = SQLiteAgentStore(tmp_path / "agent_runtime_badcase.sqlite3")
    assert len(reopened.badcases) == 1
    assert reopened.badcases[0]["reason"] == "测试回复不合适"


def test_runtime_record_badcase_requires_eval_contract_before_persisting() -> None:
    store = seeded_store()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型错误地提交空 badcase。",
                tool_calls=[
                    {
                        "name": "record_badcase",
                        "arguments": {"reason": "只有原因，没有输入实际期望"},
                        "reason": "验证 badcase 工具不会记录不可评测样本。",
                    }
                ],
            ),
            action_json(
                objective_status="needs_tool",
                reasoning_summary="previous_tool_results 返回缺 input，模型修正 badcase 参数。",
                tool_calls=[
                    {
                        "name": "record_badcase",
                        "arguments": {
                            "reason": "回复停在留意没有继续规划",
                            "input": {"text": "组"},
                            "actual": {"reply": "好的，我先留意下。"},
                            "expected": {"behavior": "继续结合上下文规划或追问关键缺口"},
                        },
                        "reason": "补齐可评测 badcase 字段后再次记录。",
                    }
                ],
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="badcase 已记录。",
                reply_to_user="我记下来了。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_badcase_contract",
            sender_id="zhang",
            sender_name="张哥",
            text="组",
            message_id="msg_runtime_badcase_contract",
        ),
        trace_id="trace_badcase_contract",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert result.tool_results[0].error == "missing required argument: input"
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert second_prompt["previous_tool_results"][0]["error"] == "missing required argument: input"
    assert len(store.badcases) == 1
    assert store.badcases[0]["expected"]["behavior"] == "继续结合上下文规划或追问关键缺口"


def test_runtime_context_checkpoint_is_tool_driven_and_survives_context_packing() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="用户补充了长期任务事实，模型显式写入 checkpoint。",
                tool_calls=[
                    {
                        "name": "update_context_checkpoint",
                        "arguments": {
                            "summary": "张哥正在让老板帮忙组一桌杭麻，倾向人齐开，烟况都可。",
                            "facts": {
                                "organizer_id": "zhang",
                                "game_type": "hangzhou_mahjong",
                                "start_time_kind": "asap_when_full",
                                "smoke_preference": "any",
                            },
                            "open_questions": ["还需要确认档位和当前人数"],
                        },
                        "reason": "这些事实需要跨多轮保留，避免上下文窗口裁剪后丢失。",
                    }
                ],
            ),
            action_json(
                objective_status="waiting_user",
                reasoning_summary="checkpoint 已更新，继续追问缺失信息。",
                reply_to_user="行，那你这边几个人，打多大？",
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="下一轮能看到 checkpoint，模型无需后端补语义。",
                reply_to_user="收到。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    first = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_checkpoint",
            sender_id="zhang",
            sender_name="张哥",
            text="人齐开吧，有烟无烟都行",
            message_id="msg_runtime_checkpoint_1",
        ),
        trace_id="trace_checkpoint_1",
    )

    checkpoint = store.get_conversation_checkpoint("runtime_checkpoint")
    assert checkpoint is not None
    assert checkpoint.summary == "张哥正在让老板帮忙组一桌杭麻，倾向人齐开，烟况都可。"
    assert checkpoint.facts["start_time_kind"] == "asap_when_full"
    assert first.tool_results[0].name == "update_context_checkpoint"
    assert any(
        event.step == "state_transition" and event.content["entity_type"] == "conversation_checkpoint"
        for event in trace.get_trace("trace_checkpoint_1")
    )

    runtime.context_builder.packing_policy.max_recent_conversation_tokens = 1
    second = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_checkpoint",
            sender_id="zhang",
            sender_name="张哥",
            text="一个人，1块的",
            message_id="msg_runtime_checkpoint_2",
        ),
        trace_id="trace_checkpoint_2",
    )

    second_prompt = json.loads(client.calls[2]["messages"][1]["content"])
    assert second.final_reply == "收到。"
    assert second_prompt["conversation_checkpoint"]["summary"] == checkpoint.summary
    assert second_prompt["conversation_checkpoint"]["facts"]["smoke_preference"] == "any"
    assert second_prompt["context_budget"]["conversation_checkpoint_present"] is True
    assert second_prompt["context_budget"]["omitted_turn_count"] >= 1
    assert len(second_prompt["recent_conversation"]) == 1


def test_runtime_sqlite_store_persists_context_checkpoint(tmp_path) -> None:
    db_path = tmp_path / "agent_runtime_checkpoint.sqlite3"
    store = SQLiteAgentStore(db_path)

    checkpoint, transition = store.upsert_conversation_checkpoint(
        conversation_id="runtime_checkpoint_sqlite",
        summary="张哥当前在组 1 块杭麻，人齐开。",
        facts={"organizer_id": "zhang", "stake": "1", "start_time_kind": "asap_when_full"},
        open_questions=["烟况是否都可"],
        trace_id="trace_checkpoint_sqlite",
    )

    assert checkpoint.source_trace_id == "trace_checkpoint_sqlite"
    assert transition.entity_type == "conversation_checkpoint"
    reopened = SQLiteAgentStore(db_path)
    persisted = reopened.get_conversation_checkpoint("runtime_checkpoint_sqlite")
    assert persisted is not None
    assert persisted.summary == "张哥当前在组 1 块杭麻，人齐开。"
    assert persisted.facts["stake"] == "1"
    assert persisted.open_questions == ["烟况是否都可"]
    assert any(item.entity_type == "conversation_checkpoint" for item in reopened.transitions)


def seeded_store(store=None):
    store = store or InMemoryAgentStore()
    store.upsert_customer(
        CustomerProfile(
            customer_id="zhang",
            display_name="张哥",
            gender="男",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["0.5", "1"],
            smoke_preference="any",
            response_score=0.9,
        )
    )
    store.upsert_customer(
        CustomerProfile(
            customer_id="ran",
            display_name="冉姐",
            gender="女",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["1"],
            smoke_preference="any",
            response_score=0.9,
        )
    )
    store.upsert_customer(
        CustomerProfile(
            customer_id="he",
            display_name="何哥",
            gender="男",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["1"],
            smoke_preference="any",
            response_score=0.8,
        )
    )
    return store


class PlanningClient:
    def __init__(self, store: InMemoryAgentStore) -> None:
        self.store = store
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        self.calls.append({"messages": messages, "trace_id": trace_id, "timeout_seconds": timeout_seconds})
        call_index = len(self.calls)
        if call_index == 1:
            return action_json(
                objective_status="needs_tool",
                reasoning_summary="需要先查询当前局池。",
                tool_calls=[
                    {
                        "name": "search_current_games",
                        "arguments": {"requirement": {"game_type": "hangzhou_mahjong", "stake": "1", "duration_kind": "overnight"}, "limit": 5},
                        "reason": "回答现有局前先查状态。",
                    }
                ],
            )
        if call_index == 2:
            return action_json(
                objective_status="needs_tool",
                reasoning_summary="没有现成局，用户也要求帮忙组。",
                tool_calls=[
                    {
                        "name": "create_game",
                        "arguments": {
                            "requirement": {
                                "game_type": "hangzhou_mahjong",
                                "stake": "1",
                                "duration_kind": "overnight",
                                "user_visible_summary": "杭麻 1块 通宵",
                            },
                            "organizer_id": "zhang",
                            "organizer_name": "张哥",
                            "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
                        },
                        "reason": "创建待组局记录。",
                    }
                ],
            )
        if call_index == 3:
            return action_json(
                objective_status="needs_tool",
                reasoning_summary="已有待组局，继续找候选人。",
                tool_calls=[
                    {
                        "name": "search_customers",
                        "arguments": {
                            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1", "duration_kind": "overnight"},
                            "exclude_customer_ids": ["zhang"],
                            "limit": 2,
                        },
                        "reason": "搜索匹配候选人。",
                    }
                ],
            )
        if call_index == 4:
            game_id = next(iter(self.store.games.values())).game_id
            return action_json(
                objective_status="needs_tool",
                reasoning_summary="候选人已返回，生成待审批邀约草稿。",
                tool_calls=[
                    {
                        "name": "create_invite_drafts",
                        "arguments": {
                            "game_id": game_id,
                            "invitations": [
                                {"customer_id": "ran", "display_name": "冉姐", "message_text": "冉姐，1块通宵，打吗？"},
                                {"customer_id": "he", "display_name": "何哥", "message_text": "何哥，1块通宵，打吗？"},
                            ],
                        },
                        "reason": "只创建待审批草稿，不直接发送。",
                    }
                ],
            )
        return action_json(
            objective_status="completed",
            reasoning_summary="待审批草稿已创建，向发起人自然确认。",
            reply_to_user="好的，我帮你问问，有消息跟你说。",
        )


class FailingAgentClient:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        self.calls.append({"messages": messages, "trace_id": trace_id, "timeout_seconds": timeout_seconds})
        raise self.exc


class TwoStepBarrierClient:
    def __init__(self, *, expected_first_calls: int) -> None:
        self.first_call_barrier = threading.Barrier(expected_first_calls)
        self.calls_by_trace: dict[str, int] = {}
        self.calls: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        with self.lock:
            call_number = self.calls_by_trace.get(trace_id, 0) + 1
            self.calls_by_trace[trace_id] = call_number
            self.calls.append(
                {
                    "messages": messages,
                    "trace_id": trace_id,
                    "timeout_seconds": timeout_seconds,
                    "call_number": call_number,
                }
            )
        if call_number == 1:
            self.first_call_barrier.wait(timeout=5)
            return action_json(
                objective_status="needs_tool",
                reasoning_summary="先查当前局池。",
                tool_calls=[
                    {
                        "name": "search_current_games",
                        "arguments": {"requirement": {"game_type": "hangzhou_mahjong"}, "limit": 1},
                        "reason": "验证并发预算隔离时先执行一个只读工具。",
                    }
                ],
            )
        return action_json(
            objective_status="completed",
            reasoning_summary="第二轮模型调用可以正常完成。",
            reply_to_user="查过了，先这样回复。",
        )


def action_json(
    *,
    objective_status: str,
    reasoning_summary: str = "test",
    reply_to_user: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    needs_human: bool = False,
    stop_reason: dict[str, Any] | None = None,
    badcase: dict[str, Any] | None = None,
) -> str:
    if stop_reason is None:
        if objective_status == "needs_tool":
            stop_reason = {
                "can_stop": False,
                "why": "还需要执行工具才能继续。",
                "pending_work": [call.get("name", "tool") for call in tool_calls or []],
                "depends_on_tool_results": False,
            }
        else:
            stop_reason = {
                "can_stop": True,
                "why": "本轮已经可以停止并回复用户。",
                "pending_work": [],
                "depends_on_tool_results": False,
            }
    return json.dumps(
        {
            "goal": "测试 Agent Runtime 主链路",
            "objective_status": objective_status,
            "reasoning_summary": reasoning_summary,
            "reply_to_user": reply_to_user,
            "tool_calls": tool_calls or [],
            "needs_human": needs_human,
            "stop_reason": stop_reason,
            "badcase": badcase,
        },
        ensure_ascii=False,
    )
