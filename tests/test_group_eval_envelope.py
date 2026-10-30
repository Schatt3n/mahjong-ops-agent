"""群评测复用生产信封，保留公开范围和原始主循环证据，不扩大通道权限。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import AgentAction, CustomerProfile, InMemoryAgentStore, ToolCall, UserMessage
from mahjong_agent_runtime.group_chat import ChannelIdentity, ChatSession, GameConversationLink, GroupMessage


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval/golden/real_group_chat_20260726.jsonl"


def _module():
    """加载现有 runner，验证其真实主循环接线而不是另建评测执行器。"""
    path = ROOT / "scripts/run_real_group_chat_flow_eval.py"
    spec = importlib.util.spec_from_file_location("group_eval_envelope_for_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _original_case(module):
    return next(case for case in module.read_jsonl(DATASET)
                if case["id"] == "real_group_fragmented_smoking_query_20260726")


class GroupQueryClient:
    """确定性输出分类及真实只读查询动作，同时捕获实际发给主模型的上下文。"""

    def __init__(self, *, clarify_only=False):
        self.main_payloads = []
        self.clarify_only = clarify_only

    def complete(self, messages, *, trace_id, **kwargs):
        if not trace_id.startswith("trace_main_"):
            return json.dumps({
                "intent": "query", "extracted_features": {"smoking": "有烟"},
                "matched_board_no": None, "confidence": 0.9,
                "reasoning": "只查询公开烟况下是否有缺人局", "response": None,
                "channel_action": "group_reply",
            }, ensure_ascii=False)
        payload = json.loads(messages[-1]["content"])
        self.main_payloads.append(payload)
        queried = any(item["name"] == "search_current_games" for item in payload.get("turn_tool_evidence", []))
        if not queried and not self.clarify_only:
            action = AgentAction(
                goal="按用户已明确烟况浏览公开局池", objective_status="needs_tool", reasoning_summary="先读取真实局况",
                tool_calls=[ToolCall(name="search_current_games", arguments={"requirement": {"smoke_preference": "有烟"}},
                                     call_id="public-query", depends_on=[], reason="只读浏览当前公开范围")],
                stop_reason={"can_stop": False, "why": "需要查询事实", "pending_work": ["查询公开局况"], "depends_on_tool_results": True},
            )
        else:
            action = AgentAction(
                goal="回应本次公开查询", objective_status="waiting_user" if self.clarify_only else "completed",
                reasoning_summary="不扩展为建局或邀约",
                reply_to_user="想打什么玩法、多大档位？" if self.clarify_only else "暂时没有有烟的缺人局。",
                stop_reason={"can_stop": True, "why": "等待补充" if self.clarify_only else "已回答公开局况",
                             "pending_work": [], "depends_on_tool_results": queried},
            )
        return json.dumps(action.to_dict(), ensure_ascii=False)


@pytest.mark.parametrize("file_timeout,external_timeout,expected", [(None, None, 45), ("90", None, 90), ("90", "105", 105)])
def test_group_main_passes_configured_timeout_to_real_agent_client(monkeypatch, tmp_path, capsys, file_timeout, external_timeout, expected):
    """真实 main 读取临时配置并运行生产群入口；仅模型边界为桩，分类器仍保持独立30秒。"""
    module = _module()
    timeout_key = "MAHJONG_AGENT_LLM_TIMEOUT_SECONDS"
    monkeypatch.delenv(timeout_key, raising=False)
    if external_timeout is not None:
        monkeypatch.setenv(timeout_key, external_timeout)
    env_file = tmp_path / ".env"
    if file_timeout is not None:
        env_file.write_text(f"{timeout_key}={file_timeout}\n")
    dataset = tmp_path / "one-original-group-case.jsonl"
    dataset.write_text(json.dumps(_original_case(module), ensure_ascii=False) + "\n")
    report_path = tmp_path / "group-report.json"

    class TimeoutCapturingClient(GroupQueryClient):
        """捕获主循环实际 complete 参数，避免仅检查运行时构造参数造成假通过。"""
        def __init__(self):
            super().__init__()
            self.timeouts = []

        def complete(self, messages, *, trace_id, timeout_seconds):
            self.timeouts.append((trace_id, timeout_seconds))
            return super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)

    client = TimeoutCapturingClient()
    monkeypatch.setattr(module.OpenAICompatibleAgentClient, "from_env", lambda: client)
    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(
        live=True, strict=True, env_file=env_file, dataset=dataset, report_path=report_path, quiet_seconds=10,
    ))
    assert module.main() == 0
    report = json.loads(report_path.read_text())
    assert report["summary"]["passed"] == report["summary"]["total"] == 1
    main_timeouts = [timeout for trace_id, timeout in client.timeouts if trace_id.startswith("trace_main_")]
    classifier_timeouts = [timeout for trace_id, timeout in client.timeouts if not trace_id.startswith("trace_main_")]
    assert main_timeouts and all(timeout == expected for timeout in main_timeouts)
    assert classifier_timeouts and all(timeout == 30 for timeout in classifier_timeouts)
    capsys.readouterr()


def test_original_fragmented_smoking_query_uses_production_envelope_and_persists_full_trace(monkeypatch, tmp_path):
    module = _module()
    original_bytes = DATASET.read_bytes()
    case = _original_case(module)
    invoked = []
    original_entry = module.GroupMessageHandler._handle_agent_loop

    def capture_entry(self, message, routing, *, trace_id):
        invoked.append(message)
        return original_entry(self, message, routing, trace_id=trace_id)

    monkeypatch.setattr(module.GroupMessageHandler, "_handle_agent_loop", capture_entry)
    client = GroupQueryClient()
    report = module.RealGroupChatFlowEvaluator(llm_client=client).evaluate([case], dataset_path=DATASET)
    result = report["cases"][0]
    assert result["status"] == "passed" and result["errors"] == []
    assert len(invoked) == 1 and invoked[0].text == "有烟的\n有缺人的没"
    assert result["actual"]["session_count"] == result["actual"]["main_agent_invocation_count"] == 1
    main = result["actual"]["main_agent"]
    message = main["input_message"]
    assert message["text"] == "有烟的\n有缺人的没"
    assert message["message_id"] == f"real:{case['id']}:1"
    assert message["sender_id"] == "wechaty:customer_eta_1"
    assert message["metadata"] == {
        "source": "group", "room_id": "room_eta", "source_conversation_id": "wechaty:room:room_eta",
        "reply_constraints": {"max_length": 20, "no_private_info": True},
    }
    assert client.main_payloads[0]["reply_constraints"] == {"max_length": 20, "no_private_info": True}
    assert main["tool_calls"] == ["search_current_games"]
    events = main["trace_events"]
    assert [event["step"] for event in events] == main["trace_steps"]
    assert all(event["trace_id"] == main["trace_id"] for event in events)
    assert {"context_built", "llm_prompt", "llm_response", "action_proposed", "tool_called", "tool_result", "final_output"} <= set(main["trace_steps"])
    assert next(event for event in events if event["step"] == "user_input")["content"]["message"] == message
    assert main["group_messages"][0]["text"] == main["final_reply"]
    assert len(main["group_messages"]) == 1
    persisted_path = tmp_path / "group_report.json"
    persisted_path.write_text(json.dumps(report, ensure_ascii=False))
    persisted = json.loads(persisted_path.read_text())["cases"][0]["actual"]["main_agent"]
    assert persisted["trace_events"] == events
    assert case["expected"]["required_tool_calls"] == ["search_current_games"]
    assert DATASET.read_bytes() == original_bytes


def test_original_smoking_query_still_fails_when_main_only_clarifies_without_required_query():
    module = _module()
    original_bytes = DATASET.read_bytes()
    case = _original_case(module)
    report = module.RealGroupChatFlowEvaluator(llm_client=GroupQueryClient(clarify_only=True)).evaluate_case(case)
    assert report["status"] == "failed"
    assert report["errors"] == ["main Agent did not call required tool search_current_games"]
    assert report["actual"]["main_agent"]["trace_events"]
    assert report["expected"] == case["expected"]
    assert DATASET.read_bytes() == original_bytes


def test_group_eval_reuses_identity_isolation_and_public_board_without_private_context():
    module = _module()
    store = InMemoryAgentStore()
    private_canary = "OTHER_PRIVATE_RAW_CANARY"
    store.append_user_turn(UserMessage(
        conversation_id="private:other", sender_id="other-customer", sender_name="另一客户",
        text=private_canary, message_id="private-observation",
    ), "private-seed")
    store.upsert_customer(CustomerProfile(customer_id="owner", display_name="PRIVATE_DISPLAY_CANARY", public_name="公开昵称"))
    game, _ = store.create_game(
        conversation_id="private:owner", organizer_id="owner", organizer_name="PRIVATE_DISPLAY_CANARY",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "无烟",
                     "start_time_kind": "asap_when_full", "user_visible_summary": "三缺一"},
        known_players=[{"customer_id": "owner", "display_name": "PRIVATE_DISPLAY_CANARY", "seat_count": 3}],
        trace_id="board-seed",
    )
    store.link_game_conversation(GameConversationLink(
        link_id="board-link", game_id=game.game_id, conversation_id="private:owner", room_id="public-room",
        customer_id="owner", link_type="board_import",
    ))
    store.upsert_channel_identity(ChannelIdentity(
        channel="wechaty", external_user_id="external-a", customer_id="customer-a", public_name="群友甲",
        private_conversation_id="private:customer-a", can_private_message=True,
    ))
    original_game = store.require_game(game.game_id).to_dict()
    other_message = GroupMessage(room_id="public-room", conversation_id="wechaty:room:public-room",
        sender_external_id="external-b", sender_name="群友乙", text="OTHER_GROUP_MEMBER_RAW_CANARY", message_id="other-message")
    source_message = GroupMessage(room_id="public-room", conversation_id="wechaty:room:public-room",
        sender_external_id="external-a", sender_name="群友甲", text="有烟的\n有缺人的没", message_id="aggregated-a",
        metadata={"other_private_context": private_canary})
    session = ChatSession(id="session-a", room_id="public-room", messages=[other_message, source_message])
    pipeline = SimpleNamespace(session_router=SimpleNamespace(list_sessions=lambda room_id: [session]))
    outcome = SimpleNamespace(session_id="session-a", classification=SimpleNamespace(
        to_dict=lambda: {"reasoning": "CLASSIFIER_REASON_CANARY", "response": "CLASSIFIER_REPLY_CANARY"}))
    client = GroupQueryClient()
    result = module.RealGroupChatFlowEvaluator._run_main_agent(
        case={"id": "public-scope", "source": {"room_alias": "public-room"}}, store=store,
        pipeline=pipeline, outcome=outcome, llm=client,
    )
    message = result["input_message"]
    assert message["sender_id"] == "customer-a"
    assert message["conversation_id"] == "group:public-room:customer:customer-a"
    assert message["text"] == source_message.text
    initial_context = client.main_payloads[0]
    assert initial_context["reply_constraints"] == {"max_length": 20, "no_private_info": True}
    assert len(initial_context["group_room_board_games"]) == 1
    public_board = initial_context["group_room_board_games"][0]
    assert public_board["game_id"] == game.game_id
    assert not {"participants", "organizer_id", "organizer_name", "seat_claims", "public_participant_names"} & public_board.keys()
    assert initial_context["active_games"] == []
    assert initial_context["active_game_visible_summaries"] == []
    raw_evidence = json.dumps(result, ensure_ascii=False)
    for canary in (private_canary, "OTHER_GROUP_MEMBER_RAW_CANARY", "PRIVATE_DISPLAY_CANARY", "CLASSIFIER_REASON_CANARY", "CLASSIFIER_REPLY_CANARY"):
        assert canary not in raw_evidence
    assert result["tool_calls"] == ["search_current_games"]
    assert store.require_game(game.game_id).to_dict() == original_game
    assert not store.invite_drafts and not store.outbound_message_drafts
    assert result["group_messages"] == [{
        "room_id": "public-room", "text": result["final_reply"],
        "metadata": {"trace_id": result["trace_id"], "source_message_id": "aggregated-a"},
    }]
