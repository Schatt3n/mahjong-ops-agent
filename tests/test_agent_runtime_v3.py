from __future__ import annotations

import json
import importlib.util
import sys
import threading
import time
from pathlib import Path
from typing import Any

from mahjong_agent_v3 import (
    AgentRuntimeV3,
    CustomerProfileV3,
    InMemoryAgentStoreV3,
    InMemoryTraceRecorderV3,
    JsonlTraceRecorderV3,
    SQLiteAgentStoreV3,
    StaticAgentClientV3,
    ToolCallV3,
    ToolGatewayV3,
    TokenBudgetV3,
    UserMessageV3,
)
from mahjong_agent_v3.tracing import trace_steps, validate_trace_v3


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_SCRIPT = ROOT / "scripts" / "verify_agent_runtime_v3_boundary.py"


def load_boundary_module():
    spec = importlib.util.spec_from_file_location("verify_agent_runtime_v3_boundary_for_test", BOUNDARY_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v3_main_chain_does_not_import_legacy_parser_workflow_or_guard() -> None:
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
    for path in (ROOT / "src" / "mahjong_agent_v3").glob("**/*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path} contains forbidden legacy token {token!r}"


def test_v3_boundary_script_rejects_semantic_patch_code(tmp_path) -> None:
    module = load_boundary_module()
    bad_file = tmp_path / "bad_v3_semantic_patch.py"
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


def test_v3_boundary_script_passes_current_main_chain() -> None:
    module = load_boundary_module()

    assert module.verify_files() == []


def test_v3_runtime_lets_model_drive_tool_sequence_until_final_reply() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorderV3()
    client = PlanningClient(store)
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_test",
            sender_id="zhang",
            sender_name="张哥",
            text="通宵1块有人吗？没有就帮我组一个",
            message_id="msg_v3_drive_001",
        ),
        trace_id="trace_v3_drive_001",
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
    events = trace.get_trace("trace_v3_drive_001")
    assert validate_trace_v3(events)["complete"] is True
    steps = trace_steps(events)
    assert steps.count("llm_prompt") == 5
    assert steps.count("tool_called") == 4
    assert "state_transition" in steps
    prompts = [event.content for event in events if event.step == "llm_prompt"]
    last_payload = json.loads(prompts[-1]["messages"][1]["content"])
    assert last_payload["previous_tool_results"][0]["name"] == "create_invite_drafts"


def test_v3_backend_does_not_interpret_short_confirmation_as_create_game() -> None:
    store = seeded_store()
    client = StaticAgentClientV3(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="模型选择只回复，不调用工具。",
                reply_to_user="好的。",
            )
        ]
    )
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorderV3())

    result = runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_no_backend_semantic",
            sender_id="zhang",
            sender_name="张哥",
            text="组",
            message_id="msg_v3_no_backend_semantic",
        ),
        trace_id="trace_v3_no_backend_semantic",
    )

    assert result.final_reply == "好的。"
    assert result.tool_results == []
    assert store.games == {}
    assert len(client.calls) == 1


def test_v3_tool_schema_error_is_fed_back_to_model_not_repaired_by_backend() -> None:
    store = seeded_store()
    client = StaticAgentClientV3(
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
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorderV3())

    result = runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_schema_error",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我问问",
            message_id="msg_v3_schema_error",
        ),
        trace_id="trace_v3_schema_error",
    )

    assert result.tool_results[0].error == "missing required argument: game_id"
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert second_prompt["previous_tool_results"][0]["error"] == "missing required argument: game_id"
    assert store.games == {}
    assert result.final_reply == "我先确认一下。"


def test_v3_action_contract_rejects_invalid_top_level_types_before_tools() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorderV3()
    client = StaticAgentClientV3(
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
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_action_contract",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_v3_action_contract",
        ),
        trace_id="trace_v3_action_contract",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.tool_results == []
    assert store.games == {}
    contract_event = next(event for event in trace.get_trace("trace_v3_action_contract") if event.step == "action_contract_error")
    assert "needs_human must be boolean" in contract_event.content["errors"]
    assert "badcase side-channel is not allowed; call record_badcase tool instead" in contract_event.content["errors"]


def test_v3_action_contract_rejects_badcase_side_channel_before_audit_write() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorderV3()
    client = StaticAgentClientV3(
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
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_badcase_side_channel",
            sender_id="zhang",
            sender_name="张哥",
            text="组",
            message_id="msg_v3_badcase_side_channel",
        ),
        trace_id="trace_v3_badcase_side_channel",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.tool_results == []
    assert store.badcases == []
    contract_event = next(event for event in trace.get_trace("trace_v3_badcase_side_channel") if event.step == "action_contract_error")
    assert "badcase side-channel is not allowed; call record_badcase tool instead" in contract_event.content["errors"]


def test_v3_action_contract_requires_human_status_to_set_human_flag() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorderV3()
    client = StaticAgentClientV3(
        [
            action_json(
                objective_status="needs_human",
                reasoning_summary="模型说需要人工，但忘了设置 needs_human=true。",
                reply_to_user="我先确认一下。",
                needs_human=False,
            )
        ]
    )
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_human_contract",
            sender_id="zhang",
            sender_name="张哥",
            text="这个需要人工吧",
            message_id="msg_v3_human_contract",
        ),
        trace_id="trace_v3_human_contract",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    contract_event = next(event for event in trace.get_trace("trace_v3_human_contract") if event.step == "action_contract_error")
    assert "needs_human objective_status requires needs_human=true" in contract_event.content["errors"]


def test_v3_action_contract_rejects_terminal_status_without_customer_reply() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorderV3()
    client = StaticAgentClientV3(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="模型声称完成，但没有给客户可见回复。",
                reply_to_user="   ",
            )
        ]
    )
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_empty_terminal_reply",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我看看",
            message_id="msg_v3_empty_terminal_reply",
        ),
        trace_id="trace_v3_empty_terminal_reply",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    contract_event = next(event for event in trace.get_trace("trace_v3_empty_terminal_reply") if event.step == "action_contract_error")
    assert "completed requires non-empty reply_to_user" in contract_event.content["errors"]


def test_v3_action_contract_rejects_unknown_tool_calls_and_human_flag_conflict() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorderV3()
    client = StaticAgentClientV3(
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
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_unknown_contract",
            sender_id="zhang",
            sender_name="张哥",
            text="随便看看",
            message_id="msg_v3_unknown_contract",
        ),
        trace_id="trace_v3_unknown_contract",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.tool_results == []
    assert store.games == {}
    contract_event = next(event for event in trace.get_trace("trace_v3_unknown_contract") if event.step == "action_contract_error")
    assert "unknown must not include tool_calls" in contract_event.content["errors"]
    assert "needs_human=true requires objective_status=needs_human" in contract_event.content["errors"]


def test_v3_invalid_candidate_status_is_rejected_by_tool_schema() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="v3_candidate_schema",
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
    client = StaticAgentClientV3(
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
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorderV3())

    result = runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_candidate_schema",
            sender_id="ran",
            sender_name="冉姐",
            text="看情况吧",
            message_id="msg_v3_candidate_schema",
        ),
        trace_id="trace_v3_candidate_schema",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert "status must be one of" in (result.tool_results[0].error or "")
    assert [item.customer_id for item in store.games[game.game_id].participants] == ["zhang"]
    assert store.invite_drafts[drafts[0].draft_id].status.value == "pending_approval"
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert "status must be one of" in second_prompt["previous_tool_results"][0]["error"]
    assert result.final_reply == "我确认一下她到底来不来。"


def test_v3_illegal_game_status_transition_is_rejected_by_state_machine() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="v3_state_machine",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="setup_state_machine",
    )
    client = StaticAgentClientV3(
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
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorderV3())

    result = runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_state_machine",
            sender_id="zhang",
            sender_name="张哥",
            text="结束掉吧",
            message_id="msg_v3_state_machine",
        ),
        trace_id="trace_v3_state_machine",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert "illegal game status transition: forming->finished" in (result.tool_results[0].error or "")
    assert store.games[game.game_id].status.value == "forming"
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert "illegal game status transition" in second_prompt["previous_tool_results"][0]["error"]
    assert result.final_reply == "这个我先确认一下。"


def test_v3_tool_permission_denial_is_fed_back_to_model_without_side_effect() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="v3_permission",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="setup_permission",
    )
    trace = InMemoryTraceRecorderV3()
    gateway = ToolGatewayV3(
        store=store,
        trace_recorder=trace,
        allowed_execution_modes={"read_only", "state_write", "audit_write"},
    )
    client = StaticAgentClientV3(
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
    runtime = AgentRuntimeV3(llm_client=client, store=store, tool_gateway=gateway, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_permission",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我问冉姐",
            message_id="msg_v3_permission",
        ),
        trace_id="trace_v3_permission",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert result.tool_results[0].error == "tool execution_mode not allowed: draft_write"
    assert store.invite_drafts == {}
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert second_prompt["previous_tool_results"][0]["error"] == "tool execution_mode not allowed: draft_write"
    permission_events = [event for event in trace.get_trace("trace_v3_permission") if event.step == "tool_permission_checked"]
    assert permission_events[0].level == "WARN"
    assert permission_events[0].content["allowed"] is False
    assert validate_trace_v3(trace.get_trace("trace_v3_permission"))["complete"] is True
    assert result.final_reply == "我先确认一下能不能发邀约。"


def test_v3_budget_denial_happens_before_llm_call_and_has_complete_trace() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorderV3()
    client = StaticAgentClientV3([])
    runtime = AgentRuntimeV3(
        llm_client=client,
        store=store,
        trace_recorder=trace,
        token_budget=TokenBudgetV3(max_tokens_per_call=1, max_calls_per_turn=8),
    )

    result = runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_budget",
            sender_id="zhang",
            sender_name="张哥",
            text="通宵1块有人吗？没有就帮我组一个",
            message_id="msg_v3_budget",
        ),
        trace_id="trace_v3_budget",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.actions == []
    assert result.tool_results == []
    assert client.calls == []
    events = trace.get_trace("trace_v3_budget")
    steps = trace_steps(events)
    assert "llm_prompt" in steps
    assert "budget_checked" in steps
    assert "llm_response" not in steps
    budget_event = next(event for event in events if event.step == "budget_checked")
    assert budget_event.content["allowed"] is False
    assert "single call token estimate exceeded" in budget_event.content["reason"]
    assert validate_trace_v3(events)["complete"] is True


def test_v3_duplicate_message_id_returns_cached_result_without_reexecuting_side_effects() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorderV3()
    client = PlanningClient(store)
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=trace)
    message = UserMessageV3(
        conversation_id="v3_message_idempotency",
        sender_id="zhang",
        sender_name="张哥",
        text="通宵1块有人吗？没有就帮我组一个",
        message_id="msg_v3_message_idempotency",
    )

    first = runtime.handle_user_message(message, trace_id="trace_v3_message_idempotency_1")
    second = runtime.handle_user_message(message, trace_id="trace_v3_message_idempotency_2")

    assert first.final_reply == "好的，我帮你问问，有消息跟你说。"
    assert second.final_reply == first.final_reply
    assert second.trace_id == first.trace_id
    assert len(client.calls) == 5
    assert len(store.games) == 1
    assert len(store.invite_drafts) == 2
    dedupe_steps = trace_steps(trace.get_trace("trace_v3_message_idempotency_2"))
    assert dedupe_steps == ["user_input", "message_deduplicated", "final_output"]
    dedupe_event = next(event for event in trace.get_trace("trace_v3_message_idempotency_2") if event.step == "message_deduplicated")
    assert dedupe_event.content["original_trace_id"] == "trace_v3_message_idempotency_1"


def test_v3_concurrent_duplicate_message_id_serializes_and_deduplicates_side_effects() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorderV3()
    client = PlanningClient(store)
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=trace)
    message = UserMessageV3(
        conversation_id="v3_concurrent_message",
        sender_id="zhang",
        sender_name="张哥",
        text="通宵1块有人吗？没有就帮我组一个",
        message_id="msg_v3_concurrent_message",
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
        threading.Thread(target=worker, args=("trace_v3_concurrent_message_1",)),
        threading.Thread(target=worker, args=("trace_v3_concurrent_message_2",)),
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
    assert trace_steps(trace.get_trace(duplicate_traces[0])) == ["user_input", "message_deduplicated", "final_output"]


def test_v3_tool_gateway_serializes_concurrent_same_backend_idempotency_key() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorderV3()
    gateway = ToolGatewayV3(store=store, trace_recorder=trace)
    call = ToolCallV3(
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
            conversation_id="v3_concurrent_tool",
            sender_id="zhang",
            sender_name="张哥",
            step_index=101,
            source_message_id="msg_v3_concurrent_tool",
        )
        with results_lock:
            results.append(result)

    threads = [
        threading.Thread(target=worker, args=("trace_v3_concurrent_tool_1",)),
        threading.Thread(target=worker, args=("trace_v3_concurrent_tool_2",)),
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
        result.idempotency_key.startswith("message:msg_v3_concurrent_tool:tool:create_game:args:")
        for result in results
    )
    hit_values = []
    for trace_id in ("trace_v3_concurrent_tool_1", "trace_v3_concurrent_tool_2"):
        events = trace.get_trace(trace_id)
        hit_values.extend(event.content["hit"] for event in events if event.step == "tool_idempotency_checked")
    assert sorted(hit_values) == [False, True]


def test_v3_jsonl_trace_is_structured_and_replayable(tmp_path) -> None:
    store = seeded_store()
    trace = JsonlTraceRecorderV3(tmp_path / "agent_v3_trace.log")
    client = PlanningClient(store)
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=trace)

    runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_jsonl_trace",
            sender_id="zhang",
            sender_name="张哥",
            text="通宵1块有人吗？没有就帮我组一个",
            message_id="msg_v3_jsonl_trace",
        ),
        trace_id="trace_v3_jsonl_trace",
    )

    events = trace.get_trace("trace_v3_jsonl_trace")
    steps = trace_steps(events)
    assert validate_trace_v3(events)["complete"] is True
    assert "raw_log_line" not in steps
    assert "llm_prompt" in steps
    assert "llm_response" in steps
    assert "tool_called" in steps
    assert "tool_result" in steps
    assert "state_transition" in steps
    prompt_payload = json.loads(next(event for event in events if event.step == "llm_prompt").content["messages"][1]["content"])
    assert prompt_payload["runtime"] == "mahjong_agent_v3"


def test_v3_sqlite_store_persists_runtime_state_and_idempotency(tmp_path) -> None:
    db_path = tmp_path / "agent_v3.sqlite3"
    store = seeded_store(SQLiteAgentStoreV3(db_path))
    trace = InMemoryTraceRecorderV3()
    client = PlanningClient(store)
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=trace)
    message = UserMessageV3(
        conversation_id="v3_sqlite",
        sender_id="zhang",
        sender_name="张哥",
        text="通宵1块有人吗？没有就帮我组一个",
        message_id="msg_v3_sqlite_persist",
    )

    result = runtime.handle_user_message(message, trace_id="trace_v3_sqlite_1")

    assert result.final_reply == "好的，我帮你问问，有消息跟你说。"
    assert len(store.games) == 1
    assert len(store.invite_drafts) == 2
    assert result.tool_results[0].idempotency_key

    reopened = SQLiteAgentStoreV3(db_path)
    assert len(reopened.customers) == 3
    assert len(reopened.games) == 1
    assert len(reopened.invite_drafts) == 2
    assert len(reopened.transitions) >= 3
    assert len(reopened.recent_turns("v3_sqlite")) >= 3
    assert reopened.idempotent_result(result.tool_results[0].idempotency_key) is not None

    cached_client = StaticAgentClientV3([])
    runtime_after_restart = AgentRuntimeV3(
        llm_client=cached_client,
        store=reopened,
        trace_recorder=InMemoryTraceRecorderV3(),
    )
    cached = runtime_after_restart.handle_user_message(message, trace_id="trace_v3_sqlite_2")

    assert cached.final_reply == result.final_reply
    assert cached_client.calls == []
    assert len(reopened.games) == 1
    assert reopened.idempotent_message_result("msg_v3_sqlite_persist") is not None


def test_v3_sqlite_store_persists_badcases_from_tool(tmp_path) -> None:
    store = seeded_store(SQLiteAgentStoreV3(tmp_path / "agent_v3_badcase.sqlite3"))
    client = StaticAgentClientV3(
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
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorderV3())

    runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_badcase",
            sender_id="zhang",
            sender_name="张哥",
            text="组",
            message_id="msg_v3_badcase",
        ),
        trace_id="trace_v3_badcase",
    )

    reopened = SQLiteAgentStoreV3(tmp_path / "agent_v3_badcase.sqlite3")
    assert len(reopened.badcases) == 1
    assert reopened.badcases[0]["reason"] == "测试回复不合适"


def seeded_store(store=None):
    store = store or InMemoryAgentStoreV3()
    store.upsert_customer(
        CustomerProfileV3(
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
        CustomerProfileV3(
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
        CustomerProfileV3(
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
    def __init__(self, store: InMemoryAgentStoreV3) -> None:
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


def action_json(
    *,
    objective_status: str,
    reasoning_summary: str = "test",
    reply_to_user: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    needs_human: bool = False,
    badcase: dict[str, Any] | None = None,
) -> str:
    return json.dumps(
        {
            "goal": "测试 V3 agent 主链路",
            "objective_status": objective_status,
            "reasoning_summary": reasoning_summary,
            "reply_to_user": reply_to_user,
            "tool_calls": tool_calls or [],
            "needs_human": needs_human,
            "badcase": badcase,
        },
        ensure_ascii=False,
    )
