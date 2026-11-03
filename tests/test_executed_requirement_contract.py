"""真实读参数的连续性投影必须与执行守卫同源，不把模型解释锁成用户事实。"""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from mahjong_agent_runtime import (
    AgentAction, AgentContextBuilder, AgentRuntime, InMemoryAgentStore, InMemoryTraceRecorder,
    SQLiteAgentStore, ToolCall, ToolGateway, ToolResult, UserMessage,
)
from mahjong_agent_runtime.tool_consistency import (
    CONSISTENT_REQUIREMENT_FIELDS, executed_requirement_contract, latest_read_requirement,
    validate_tool_call_consistency,
)


def read_result(requirement, *, call_id="real-read", called=True, allowed=True, error=None):
    """不以模型计划伪造执行；测试来源是运行时接收的真实结果合同。"""
    return ToolResult(name="search_current_games", called=called, allowed=allowed, error=error,
                      call_id=call_id, result={"requirement": deepcopy(requirement)})


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    current = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "continuity.sqlite3")
    yield current
    if isinstance(current, SQLiteAgentStore):
        current._connection.close()


@pytest.mark.parametrize("field,before,after", [
    ("game_type", "hangzhou_mahjong", "national_standard_mahjong"),
    ("stake", "1", "2"),
    ("stake_range", {"min": 1, "max": 3}, {"min": 2, "max": 4}),
    ("cap_score", 16, 32),
    ("smoke_preference", "no_smoking", "smoking"),
    ("start_time_kind", "now", "asap_when_full"),
    ("start_time", "18:30", "19:30"),
    ("planned_start_at", "2099-01-01T18:30:00+08:00", "2099-01-02T18:30:00+08:00"),
    ("duration_kind", "fixed", "overnight"),
    ("duration_hours", 4, 2),
])
def test_projection_and_guard_protect_same_generic_fields_without_interpreting_text(field, before, after):
    read = read_result({field: before})
    snapshot = deepcopy(read.to_dict())
    contract = executed_requirement_contract([read])
    assert contract["preserve_requirement"] == {field: before}
    assert contract["source_tool"] == "search_current_games"
    assert contract["source_call_id"] == "real-read"
    proposed = deepcopy(contract["preserve_requirement"])
    assert validate_tool_call_consistency(ToolCall("create_game", {"requirement": proposed}), [read]) is None
    proposed[field] = after
    assert validate_tool_call_consistency(ToolCall("create_game", {"requirement": proposed}), [read])
    assert read.to_dict() == snapshot


def test_context_does_not_copy_people_table_snapshots_or_extra_query_fields(store):
    message = UserMessage(conversation_id="current", sender_id="customer", sender_name="客户", text="继续处理")
    requirement = {"stake": 1, "smoke_preference": "no_smoking", "known_player_count": 3,
        "needed_seats": 1, "seat_format": "371", "private_note": "PRIVATE_MARKER",
        "requesting_party": {"contact_id": "PRIVATE_ID", "contact_name": "PRIVATE_NAME", "seat_count": 3}}
    read = read_result(requirement)
    before = deepcopy(read.to_dict())
    context = AgentContextBuilder(store, ToolGateway(store)).build(message, trace_id="build",
        previous_tool_results=[read], turn_tool_evidence=[read])
    contract = context.payload["executed_requirement_contract"]
    assert contract["preserve_requirement"] == {"stake": "1", "smoke_preference": "no_smoking"}
    assert set(contract["preserve_requirement"]) <= set(CONSISTENT_REQUIREMENT_FIELDS)
    assert "PRIVATE_" not in json.dumps(contract)
    assert json.loads(context.messages[-1]["content"])["executed_requirement_contract"] == contract
    contract["preserve_requirement"]["stake"] = "999"
    assert read.to_dict() == before
    assert not store.games and not store.invite_drafts and not store.outbound_message_drafts


@pytest.mark.parametrize("call_id", [None, "", 17])
def test_absent_read_identity_is_unknown_not_a_model_or_failed_write_identity(call_id):
    assert executed_requirement_contract([read_result({"stake": "1"}, call_id=call_id)])["source_call_id"] is None
    feedback = ToolResult(name="create_game", called=False, allowed=False, call_id="rejected-write", error="mismatch",
        result={"reference_tool_name": "search_current_games", "reference_requirement": {"stake": "1"}})
    contract = executed_requirement_contract([feedback])
    assert contract["source_call_id"] is None
    assert contract["source_kind"] == "runtime_reference_feedback"
    assert contract["preserve_requirement"] == {"stake": "1"}
    assert validate_tool_call_consistency(ToolCall("create_game", {"requirement": {"stake": "2"}}), [feedback])


def test_successful_requery_replaces_reference_but_failed_or_denied_query_does_not():
    first = read_result({"stake": "1", "smoke_preference": "no_smoking"}, call_id="first")
    failed = read_result({"stake": "2"}, call_id="failed", error="failed read")
    denied = read_result({"stake": "2"}, call_id="denied", called=False, allowed=False)
    history = [first, failed, denied]
    contract = executed_requirement_contract(history)
    assert contract["source_call_id"] == "first"
    assert contract["preserve_requirement"] == first.result["requirement"]
    new = read_result({"stake": "2", "smoke_preference": "smoking"}, call_id="rechecked")
    history.append(new)
    contract = executed_requirement_contract(history)
    assert contract["source_call_id"] == "rechecked"
    assert contract["preserve_requirement"] == new.result["requirement"]
    assert latest_read_requirement(history, tool_name="search_current_games") == new.result["requirement"]
    call = ToolCall("create_game", {"requirement": deepcopy(contract["preserve_requirement"])})
    assert validate_tool_call_consistency(call, history) is None


def test_reference_feedback_and_projection_share_existing_guard_selection():
    first = read_result({"stake": "1"})
    feedback = ToolResult(name="create_game", called=False, allowed=False, error="rejected",
        result={"reference_tool_name": "search_current_games", "reference_requirement": {"stake": "2"}})
    history = [first, feedback]
    assert executed_requirement_contract(history)["preserve_requirement"] == latest_read_requirement(history, tool_name="search_current_games")
    assert validate_tool_call_consistency(ToolCall("create_game", {"requirement": {"stake": "1"}}), history)
    assert validate_tool_call_consistency(ToolCall("create_game", {"requirement": {"stake": "2"}}), history) is None


@pytest.mark.parametrize("requirement", [{}, {"known_player_count": 3}, {"stake": ""}])
def test_empty_or_non_guarded_reads_do_not_invent_a_write_binding(requirement):
    assert executed_requirement_contract([read_result(requirement)]) is None


def test_old_plan_checkpoint_and_message_metadata_do_not_bind_a_new_run(store):
    builder = AgentContextBuilder(store, ToolGateway(store))
    message = UserMessage(conversation_id="current", sender_id="customer", sender_name="客户", text="现在改成两块",
        metadata={"executed_requirement_contract": {"preserve_requirement": {"stake": "1"}}})
    action = AgentAction(goal="旧解释", objective_status="needs_tool", reasoning_summary="仅是提案",
        objective_state={"known_facts": {"stake": "1"}, "fact_assertions": [{"field": "stake", "value": "1", "role": "current"}]})
    store.upsert_conversation_checkpoint(conversation_id="current", summary="先前一块", facts={"stake": "1"}, open_questions=[], trace_id="old")
    context = builder.build(message, trace_id="new-run", previous_action=action)
    assert context.payload["previous_decision"] is not None
    assert context.payload["executed_requirement_contract"] is None
    failed = read_result({"stake": "1"}, called=False, allowed=False, error="not executed")
    context = builder.build(message, trace_id="failed-read", turn_tool_evidence=[failed])
    assert context.payload["executed_requirement_contract"] is None


def test_real_runtime_keeps_read_binding_across_unrelated_step_then_uses_successful_revision(store):
    contexts = []
    def action(calls=None, reply=""):
        calls = calls or []
        for index, call in enumerate(calls):
            call.reason = "按当前真实读参数继续本轮测试"
            call.call_id = call.call_id or f"test-call-{len(contexts)}-{index}"
            call.depends_on = []
        return AgentAction(goal="核对条件后继续", objective_status="needs_tool" if calls else "completed",
            reasoning_summary="只使用当前成功查询的条件", tool_calls=calls, reply_to_user=reply,
            stop_reason={"can_stop": not calls, "why": "继续核对" if calls else "结束",
                         "pending_work": ["真实工具"] if calls else [], "depends_on_tool_results": bool(calls)})
    class Client:
        def complete(self, messages, **kwargs):
            context = json.loads(messages[-1]["content"])
            contexts.append(context)
            step = len(contexts)
            if step == 1:
                proposal = action([ToolCall("search_current_games", {"requirement": {"stake": "1", "smoke_preference": "no_smoking"}}, call_id="initial-read")])
            elif step == 2:
                proposal = action([ToolCall("check_room_availability", {})])
            elif step == 3:
                proposal = action([ToolCall("search_current_games", {"requirement": {"stake": "2", "smoke_preference": "smoking"}}, call_id="rechecked-read")])
            elif step == 4:
                proposal = action([ToolCall("create_game", {"organizer_id": "customer", "organizer_name": "客户",
                    "requirement": context["executed_requirement_contract"]["preserve_requirement"]})])
            elif step == 5:
                proposal = action([ToolCall("search_customers", {"game_id": context["active_games"][0]["game_id"]})])
            else:
                proposal = action(reply="好。")
            return json.dumps(proposal.to_dict(), ensure_ascii=False)
    runtime = AgentRuntime(store=store, llm_client=Client(), trace_recorder=InMemoryTraceRecorder())
    result = runtime.handle_user_message(UserMessage(conversation_id="current", sender_id="customer", sender_name="客户",
        text="按实际核对结果继续处理", message_id="current-input"), trace_id="continuity")
    assert result.status == "completed" and len(contexts) == 6
    assert contexts[0]["executed_requirement_contract"] is None
    assert contexts[2]["previous_tool_results"][0]["name"] == "check_room_availability"
    assert contexts[2]["executed_requirement_contract"]["source_call_id"] == "initial-read"
    assert contexts[3]["executed_requirement_contract"]["source_call_id"] == "rechecked-read"
    assert contexts[3]["executed_requirement_contract"]["preserve_requirement"] == {"stake": "2", "smoke_preference": "smoking"}
    games = store.active_games("current")
    assert len(games) == 1 and games[0].requirement["stake"] == "2"
    assert games[0].requirement["smoke_preference"] == "smoking"
