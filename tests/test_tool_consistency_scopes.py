"""跨工具条件按所属对象比较，不能把目标桌况复制成新局人数。"""

from __future__ import annotations

from copy import deepcopy

import pytest

from mahjong_agent_runtime import (
    AgentAction,
    InMemoryAgentStore,
    InMemoryTraceRecorder,
    SQLiteAgentStore,
    ToolCall,
    ToolGateway,
    UserMessage,
)
from mahjong_agent_runtime.services import ToolExecutionService
from mahjong_agent_runtime.tool_consistency import validate_explicit_task_fact_consistency


@pytest.fixture(params=["memory", "sqlite"])
def chain(request, tmp_path):
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "scopes.sqlite3")
    trace = InMemoryTraceRecorder()
    gateway = ToolGateway(store=store, trace_recorder=trace)
    service = ToolExecutionService(store=store, tool_gateway=gateway, trace_recorder=trace)
    message = UserMessage(
        conversation_id="scope-chat", sender_id="customer", sender_name="客户",
        text="我自己来，想找三缺一，没有就帮我组", message_id="scope-message",
    )
    store.append_user_turn(message, "scope-trace")
    yield store, service, message
    if isinstance(store, SQLiteAgentStore):
        store._connection.close()


def execute(chain, call, *, facts=None, previous=None, step=1):
    store, service, message = chain
    assertions = [
        {"field": field, "value": value, "role": "current", "source_message_id": message.message_id}
        for field, value in (facts or {}).items()
    ]
    processed = service.execute_tool_calls(
        AgentAction(
            goal="推进客户组局", objective_status="needs_tool", reasoning_summary="验证工具作用域",
            tool_calls=[call], objective_state={"fact_assertions": assertions},
        ),
        message=message, trace_id="scope-trace", previous_step_tool_results=previous or [],
        step_index=step, run_id="scope-run", run_version=store.conversation_version(message.conversation_id),
    )
    return processed.tool_results[0]


def create_call(requirement, *, seats=1):
    return ToolCall(
        name="create_game",
        arguments={
            "requirement": requirement, "organizer_id": "customer", "organizer_name": "客户",
            "requesting_party": {"contact_id": "customer", "contact_name": "客户", "seat_count": seats},
        },
    )


@pytest.mark.parametrize("stake", [{"stake": "1"}, {"stake_range": {"min": 1, "max": 256}}])
@pytest.mark.parametrize("party_size", [1, 2])
def test_target_three_present_query_can_create_a_different_new_game(chain, stake, party_size):
    shared = {"game_type": "hangzhou_mahjong", "start_time_kind": "asap_when_full", **stake}
    facts = {"known_player_count": party_size, "needed_seats": 4 - party_size, **shared}
    query = execute(chain, ToolCall(name="search_current_games", arguments={"requirement": {
        **shared, "known_player_count": 3, "needed_seats": 1,
        "requesting_party": {"contact_id": "customer", "seat_count": party_size},
    }}), facts=facts)
    assert query.allowed and query.called
    assert query.result["matches"] == []
    assert query.result["requirement"]["known_player_count"] == 3
    assert query.result["requirement"]["needed_seats"] == 1

    created = execute(chain, create_call({**shared, "known_player_count": party_size, "needed_seats": 4 - party_size},
                                        seats=party_size), facts=facts, previous=[query], step=2)
    assert created.allowed and created.called and created.error is None
    game_id = created.result["game"]["game_id"]
    store = chain[0]
    if isinstance(store, SQLiteAgentStore):
        # 必须验证落库后的事实，不能只核对工具返回的内存对象。
        recovered = SQLiteAgentStore(store.path)
        try:
            game = recovered.require_game(game_id)
        finally:
            recovered._connection.close()
    else:
        game = store.require_game(game_id)
    assert game.requirement["known_player_count"] == party_size
    assert game.requirement["needed_seats"] == 4 - party_size
    assert game.remaining_seats() == 4 - party_size
    assert [(party.contact_id, party.seat_count) for party in game.parties] == [("customer", party_size)]


@pytest.mark.parametrize("tool_name", ["search_current_games", "search_customers"])
def test_query_table_snapshot_does_not_redefine_current_customer_facts(tool_name):
    call = ToolCall(name=tool_name, arguments={"requirement": {
        "known_player_count": 3, "needed_seats": 1, "seat_format": "371",
    }})
    facts = {"known_player_count": 1, "needed_seats": 3, "seat_format": "173"}
    assert validate_explicit_task_fact_consistency(call, {"explicit_task_facts": {"facts": facts}})[0] is None


def test_absent_requesting_party_is_not_compared_as_an_explicit_single_person():
    call = ToolCall(name="search_current_games", arguments={"requirement": {
        "known_player_count": 3, "needed_seats": 1,
    }})
    error, reference = validate_explicit_task_fact_consistency(
        call, {"explicit_task_facts": {"facts": {"known_player_count": 2, "needed_seats": 2}}},
    )
    assert error is None
    assert reference["known_player_count"] == 2


@pytest.mark.parametrize("size_field", ["seat_count", "party_size", "requesting_party"])
@pytest.mark.parametrize("fact_field", ["known_player_count", "party_size", "seat_count", "requesting_party"])
def test_search_requesting_party_still_must_match_explicit_customer_count(chain, size_field, fact_field):
    requirement = {"game_type": "hangzhou_mahjong", "known_player_count": 3, "needed_seats": 1}
    requirement[size_field] = {"contact_id": "customer", "seat_count": 1} if size_field == "requesting_party" else 1
    fact = {"seat_count": 2} if fact_field == "requesting_party" else 2
    result = execute(chain, ToolCall(name="search_current_games", arguments={"requirement": requirement}),
                     facts={fact_field: fact})
    assert not result.allowed and not result.called
    assert "requesting_party.seat_count" in result.error
    assert chain[0].games == {}


@pytest.mark.parametrize("change", ["declared_count", "requesting_party", "known_players"])
def test_create_cannot_hide_a_customer_count_change_in_another_argument(chain, change):
    call = create_call({"known_player_count": 2, "needed_seats": 2}, seats=2)
    if change == "declared_count":
        call.arguments["requirement"] = {"known_player_count": 1, "needed_seats": 3}
    elif change == "requesting_party":
        call.arguments["requesting_party"]["seat_count"] = 1
    else:
        # 已有 known_players 的显式人数优先于 requesting_party；与真实写入路径保持一致。
        call.arguments["known_players"] = [{"customer_id": "customer", "display_name": "客户", "seat_count": 1}]
    result = execute(chain, call, facts={"known_player_count": 2, "needed_seats": 2})
    assert not result.allowed and not result.called
    assert "explicit current-task facts" in result.error
    assert chain[0].games == {}


def test_separately_named_companion_is_part_of_the_confirmed_initial_count(chain):
    call = create_call({"known_player_count": 2, "needed_seats": 2}, seats=1)
    call.arguments["known_players"] = [{"customer_id": "friend", "display_name": "同行朋友", "seat_count": 1}]
    result = execute(chain, call, facts={"known_player_count": 2, "needed_seats": 2})
    assert result.allowed and result.called
    assert result.result["game"]["requirement"]["known_player_count"] == 2
    assert result.result["game"]["remaining_seats"] == 2


def test_explicit_own_party_size_is_checked_separately_from_game_population(chain):
    call = create_call({"known_player_count": 3, "needed_seats": 1}, seats=1)
    call.arguments["known_players"] = [{"customer_id": "friend", "display_name": "同行朋友", "seat_count": 2}]
    result = execute(chain, call, facts={"known_player_count": 3, "party_size": 2})
    assert not result.allowed and not result.called
    assert "requesting_party.seat_count" in result.error
    assert chain[0].games == {}


def test_candidate_search_keeps_persisted_game_population_over_the_old_customer_snapshot(chain):
    store = chain[0]
    created = execute(chain, create_call({"game_type": "hangzhou_mahjong", "known_player_count": 1, "needed_seats": 3}),
                      facts={"known_player_count": 1, "needed_seats": 3})
    game_id = created.result["game"]["game_id"]
    store.record_candidate_reply(
        game_id=game_id, customer_id="friend", display_name="同行朋友", status="accepted", seat_count=1,
        trace_id="friend-joined",
    )
    searched = execute(chain, ToolCall(name="search_customers", arguments={
        "game_id": game_id, "requirement": {"known_player_count": 1, "needed_seats": 3},
    }), facts={"known_player_count": 1, "needed_seats": 3}, previous=[created], step=2)
    assert searched.allowed and searched.called
    assert searched.result["requirement"]["known_player_count"] == 2
    assert searched.result["requirement"]["needed_seats"] == 2
    assert store.require_game(game_id).requirement["known_player_count"] == 2


@pytest.mark.parametrize("field,value", [
    ("stake", "2"), ("start_time_kind", "scheduled"), ("smoke_preference", "smoke"),
    ("game_type", "red_center_mahjong"), ("cap_score", 128),
    ("duration_hours", 6), ("stake_range", {"min": 1, "max": 2}), ("start_time", "18:00"),
])
def test_same_customer_constraints_still_cannot_drift_between_query_and_creation(chain, field, value):
    requirement = {
        "game_type": "hangzhou_mahjong", "stake": "1", "cap_score": 256,
        "start_time_kind": "asap_when_full", "smoke_preference": "no_smoke", "duration_hours": 4,
        "known_player_count": 3, "needed_seats": 1,
    }
    if field == "stake_range":
        requirement.pop("stake")
        requirement["stake_range"] = {"min": 1, "max": 3}
    if field == "start_time":
        requirement.update(start_time_kind="scheduled", start_time="16:00")
    query = execute(chain, ToolCall(name="search_current_games", arguments={"requirement": requirement}))
    assert query.allowed and query.called
    new_requirement = {**deepcopy(requirement), "known_player_count": 1, "needed_seats": 3, field: value}
    result = execute(chain, create_call(new_requirement), previous=[query], step=2)
    assert not result.allowed and not result.called
    assert "previous search_current_games" in result.error
    assert field in result.error
    assert chain[0].games == {}


@pytest.mark.parametrize("invalid", ["party_schema", "capacity", "ownership"])
def test_scope_separation_does_not_weaken_gateway_or_store_guards(chain, invalid):
    call = create_call({"game_type": "hangzhou_mahjong"})
    if invalid == "party_schema":
        call.arguments["requesting_party"]["seat_count"] = 5
        error = "seat_count must be <= 4"
    elif invalid == "capacity":
        call.arguments["requesting_party"]["seat_count"] = 3
        call.arguments["known_players"] = [{"customer_id": "friend", "display_name": "同行朋友", "seat_count": 2}]
        error = "exceed table capacity"
    else:
        call.arguments["organizer_id"] = "other-customer"
        error = "authenticated sender_id"
    result = execute(chain, call)
    assert not result.allowed and not result.called
    assert error in result.error
    assert chain[0].games == {}
