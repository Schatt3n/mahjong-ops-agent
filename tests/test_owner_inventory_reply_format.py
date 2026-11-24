"""公开盘点的分行展示须绑定真实回放中的一条备发回复。"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys

import pytest

from mahjong_agent_runtime import AgentRuntime, InMemoryTraceRecorder, StaticAgentClient
from tests.test_real_owner_chat_golden import action_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_real_owner_chat_live_eval as owner


LIST_REPLY = "现在在组的有两桌：\n1. 杭麻1块，三缺一\n2. 国标2块，两缺二"
SEMANTIC_STUB = {"passed": True, "error": None}


@contextmanager
def runtime_reply(tmp_path, scenario_id, reply, *, requirement=None):
    """仅替换模型输出；查询、持久化和追踪经过正常运行时。"""
    scenario = next(scene for scene in owner.live_eval_scenarios() if scene.scenario_id == scenario_id)
    outputs = []
    if requirement is not None:
        outputs.append(action_json(objective_status="needs_tool", tool_calls=[{
            "name": "search_current_games", "arguments": {"requirement": requirement}, "reason": "查看公开桌况",
        }]))
    outputs.append(action_json(objective_status="completed", reply_to_user=reply))
    store = owner.SQLiteAgentStore(tmp_path / "real-inventory-format.sqlite3")
    try:
        scenario.setup(store)
        trace = InMemoryTraceRecorder()
        runtime = AgentRuntime(llm_client=StaticAgentClient(outputs), store=store, trace_recorder=trace)
        result = runtime.handle_user_message(scenario.message, trace_id="inventory-reply-format")
        assert result.final_reply == reply and result.status == "completed"
        yield scenario, result, trace.get_trace(result.trace_id), store
    finally:
        store._connection.close()


def single_message_check(result, scenario, events, store):
    validation = owner.validate_result(result, scenario, events, store, SEMANTIC_STUB)
    check = next(item for item in validation["checks"] if item["name"] == "final_reply_should_be_single_message")
    return validation, check


def test_multi_table_inventory_preserves_one_verified_multiline_prepared_reply(tmp_path):
    with runtime_reply(tmp_path, "inventory_all_games_ignores_personal_profile", LIST_REPLY, requirement={}) as replay:
        scenario, result, events, store = replay
        validation, check = single_message_check(result, scenario, events, store)
        assert check["passed"] is True and validation["passed"], [
            item["name"] for item in validation["checks"] if not item["passed"]]
        assistants = [turn for turn in store.recent_turns(scenario.message.conversation_id)
                      if turn.role.value == "assistant" and turn.trace_id == result.trace_id]
        assert len(assistants) == 1 and assistants[0].content == LIST_REPLY
        assert assistants[0].metadata["delivery_status"] == "pending_operator_send"
        assert not store.outbound_message_drafts


@pytest.mark.parametrize("fault", ["duplicate_final", "missing_final", "wrong_final", "foreign_final",
                                   "duplicate_turn", "missing_turn", "wrong_turn", "foreign_turn",
                                   "missing_query_completion", "repeated_table"])
def test_inventory_multiline_reply_requires_both_unique_exact_output_sources(tmp_path, fault):
    with runtime_reply(tmp_path, "inventory_all_games_ignores_personal_profile", LIST_REPLY, requirement={}) as replay:
        scenario, result, events, store = replay
        final = next(event for event in events if event.step == "final_output")
        conversation = scenario.message.conversation_id
        if fault == "duplicate_final":
            events.append(deepcopy(final))
        elif fault == "missing_final":
            events.remove(final)
        elif fault == "wrong_final":
            final.content["reply"] = "不同的正文"
        elif fault == "foreign_final":
            final.trace_id = "other-turn"
        elif fault == "duplicate_turn":
            store.append_assistant_turn(conversation, LIST_REPLY, result.trace_id)
        elif fault == "missing_query_completion":
            events.remove(next(event for event in events if event.step == "tool_gateway_completed"))
        elif fault == "repeated_table":
            query = next(tool for tool in result.tool_results if tool.name == "search_current_games")
            query.result["matches"] = [query.result["matches"][0]] * 2
            next(event for event in events if event.step == "tool_result"
                 and event.content["name"] == "search_current_games").content = query.to_dict()
        else:
            store._connection.execute(
                "DELETE FROM runtime_conversation_turns WHERE conversation_id=? AND trace_id=? AND role='assistant'",
                (conversation, result.trace_id))
            if fault == "wrong_turn":
                store.append_assistant_turn(conversation, "不同的正文", result.trace_id)
            elif fault == "foreign_turn":
                store.append_assistant_turn(conversation, LIST_REPLY, "other-turn")
        validation, check = single_message_check(result, scenario, events, store)
        assert check["passed"] is False and validation["passed"] is False


@pytest.mark.parametrize("scenario_id,reply,requirement", [
    ("inventory_explicit_game_overrides_profile", "现在一桌：\n杭麻1块，三缺一", {"game_type": "hangzhou_mahjong"}),
    ("casual_chat_should_not_pollute_business_state", "嗯。\n确实。", None),
])
def test_single_table_and_ordinary_replies_keep_single_line_style(tmp_path, scenario_id, reply, requirement):
    with runtime_reply(tmp_path, scenario_id, reply, requirement=requirement) as replay:
        scenario, result, events, store = replay
        validation, check = single_message_check(result, scenario, events, store)
        assert check["passed"] is False and validation["passed"] is False
