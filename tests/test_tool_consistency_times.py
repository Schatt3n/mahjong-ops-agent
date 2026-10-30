"""时间一致性保留来源日期与时区，钟点不能凭壁钟或词法候选冒充绝对瞬时。"""

from __future__ import annotations

from copy import deepcopy

import pytest

from mahjong_agent_runtime import (
    AgentAction, InMemoryAgentStore, InMemoryTraceRecorder, SQLiteAgentStore,
    ToolCall, ToolGateway, ToolResult, UserMessage,
)
from mahjong_agent_runtime.services import ToolExecutionService
from mahjong_agent_runtime.tool_consistency import validate_explicit_task_fact_consistency, validate_tool_call_consistency


ANCHOR = "2026-10-30T18:30:00+08:00"
UTC_EQUIVALENT = "2026-10-30T10:30:00Z"


def explicit_error(reference, proposed):
    call = ToolCall(name="search_current_games", arguments={"requirement": proposed})
    original = deepcopy(call.arguments)
    error, _ = validate_explicit_task_fact_consistency(call, {"explicit_task_facts": {"facts": reference}})
    assert call.arguments == original
    return error


@pytest.mark.parametrize("reference,proposed", [
    ({"start_time": "18:30", "planned_start_at": ANCHOR}, {"start_time": ANCHOR}),
    ({"start_time": "18:30", "planned_start_at": ANCHOR}, {"start_time": UTC_EQUIVALENT}),
    ({"start_time": ANCHOR, "planned_start_at": ANCHOR}, {"start_time": "18:30", "planned_start_at": UTC_EQUIVALENT}),
    ({"start_time": ANCHOR}, {"start_time": UTC_EQUIVALENT}),
    ({"planned_start_at": ANCHOR}, {"planned_start_at": UTC_EQUIVALENT}),
    ({"start_time": "2026-10-30T00:30:00+08:00"}, {"start_time": "2026-10-29T16:30:00Z"}),
    ({"start_time": "00:30", "planned_start_at": "2026-10-30T00:30:00+08:00"}, {"start_time": "2026-10-29T16:30:00Z"}),
    ({"start_time": "18.30"}, {"start_time": "18:30"}),
])
def test_time_guard_accepts_only_proven_representation_equivalence(reference, proposed):
    assert explicit_error(reference, proposed) is None


@pytest.mark.parametrize("reference,proposed", [
    ({"start_time": "18:30"}, {"start_time": ANCHOR}),
    ({"start_time": "18:30", "planned_start_at": "2026-10-30T18:30:00"}, {"start_time": ANCHOR}),
    ({"start_time": "18:30", "planned_start_at": ANCHOR}, {"start_time": "2026-10-31T18:30:00+08:00"}),
    ({"start_time": "18:30", "planned_start_at": ANCHOR}, {"start_time": "2026-10-30T18:30:00+09:00"}),
    ({"start_time": "18:30", "planned_start_at": ANCHOR}, {"start_time": "2026-10-30T10:31:00Z"}),
    ({"start_time": "18:30", "planned_start_at": ANCHOR}, {"start_time": "18:30"}),
    ({"start_time": ANCHOR}, {"start_time": "18:30"}),
    ({"start_time": ANCHOR, "planned_start_at": ANCHOR}, {"start_time": "18:30"}),
    ({"start_time": "18:30", "planned_start_at": ANCHOR}, {"start_time": ANCHOR, "planned_start_at": "2026-10-31T18:30:00+08:00"}),
    ({"start_time": "18:30", "planned_start_at": ANCHOR}, {"start_time": ANCHOR, "planned_start_at": "2026-10-30T18:30:00"}),
    ({"start_time": "18:30", "planned_start_at": "2026-10-30T17:30:00+08:00"}, {"start_time": "2026-10-30T17:30:00+08:00"}),
    ({"start_time": ANCHOR}, {"start_time": "2026-10-31T18:30:00+08:00"}),
    ({"planned_start_at": ANCHOR}, {"planned_start_at": "2026-10-30T18:30:00+09:00"}),
    ({"planned_start_at": ANCHOR}, {"planned_start_at": "2026-10-30T18:30:00"}),
])
def test_time_guard_rejects_missing_anchor_day_offset_and_internal_conflicts(reference, proposed):
    assert explicit_error(reference, proposed)


@pytest.mark.parametrize("proposed,allowed", [
    ({"start_time": "18:30", "planned_start_at": UTC_EQUIVALENT}, True),
    ({"start_time": UTC_EQUIVALENT}, True),
    ({"start_time": "18:30"}, False),
    ({"start_time": "18:30", "planned_start_at": "2026-10-31T18:30:00+08:00"}, False),
    ({"start_time": "18:30", "planned_start_at": "2026-10-30T18:30:00+09:00"}, False),
    ({"start_time": "18:30", "planned_start_at": "2026-10-30T18:30:00"}, False),
])
def test_read_to_create_retains_absolute_date_and_compares_utc_instant(proposed, allowed):
    read = ToolResult(name="search_current_games", called=True, allowed=True,
                      result={"requirement": {"start_time": "18:30", "planned_start_at": ANCHOR}})
    call = ToolCall(name="create_game", arguments={"requirement": proposed})
    before = deepcopy(call.arguments)
    assert (validate_tool_call_consistency(call, [read]) is None) is allowed
    assert call.arguments == before


def test_failed_read_cannot_establish_an_absolute_time_anchor():
    failed_read = ToolResult(name="search_current_games", called=True, allowed=True, error="query failed",
                            result={"requirement": {"start_time": "18:30", "planned_start_at": ANCHOR}})
    call = ToolCall(name="create_game", arguments={"requirement": {"start_time": "19:30"}})
    assert validate_tool_call_consistency(call, [failed_read]) is None


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("anchor_source", ["current", "condition", "missing", "other_sender", "candidate_only"])
def test_service_only_uses_source_bound_current_anchor(backend, anchor_source, tmp_path):
    store = InMemoryAgentStore() if backend == "memory" else SQLiteAgentStore(tmp_path / "time-anchor.sqlite3")
    try:
        trace = InMemoryTraceRecorder()
        message = UserMessage(conversation_id="time-source", sender_id="customer", sender_name="客户",
                              text="2026年10月30日18:30，东八区", message_id="time-source-message")
        store.append_user_turn(message, "time-source")
        store.append_user_turn(UserMessage(conversation_id=message.conversation_id, sender_id="other", sender_name="其他客户",
                                         text="2026年10月30日18:30，东八区", message_id="other-message"), "other-trace")
        assertions = [{"field": "start_time", "value": "18:30", "role": "current", "source_message_id": message.message_id}]
        if anchor_source != "candidate_only":
            assertions.append({"field": "planned_start_at", "value": ANCHOR,
                "role": "condition" if anchor_source == "condition" else "current",
                "source_message_id": "absent" if anchor_source == "missing" else "other-message" if anchor_source == "other_sender" else message.message_id})
        call = ToolCall(name="search_current_games", arguments={"requirement": {"start_time": UTC_EQUIVALENT}})
        service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store=store), trace_recorder=trace)
        processed = service.execute_tool_calls(
            AgentAction(goal="按客户时间查局", objective_status="needs_tool", reasoning_summary="测试来源锚", tool_calls=[call],
                        objective_state={"fact_assertions": assertions}),
            message=message, trace_id="time-source-test", previous_step_tool_results=[], step_index=1,
            run_id="time-source-run", run_version=store.conversation_version(message.conversation_id),
            context_payload={"explicit_task_facts": {"candidates": {"planned_start_at": ANCHOR}}},
        )
        result = processed.tool_results[0]
        assert result.called is (anchor_source == "current")
        assert result.allowed is (anchor_source == "current")
        assert (result.error is None) is (anchor_source == "current")
        assert store.games == {} and not store.invite_drafts
    finally:
        if backend == "sqlite":
            store._connection.close()
