from __future__ import annotations

import pytest

from mahjong_agent_runtime import CustomerProfile, InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway
from mahjong_agent_runtime.domains.game_domain import normalize_requirement
from mahjong_agent_runtime.domains.customer_domain import classify_current_game_match, score_stake_preference
from mahjong_agent_runtime.domains.stake_values import parse_stake_range, stake_requirements_overlap
from mahjong_agent_runtime.group_chat.board_engine import BoardEngine
from mahjong_agent_runtime.group_chat.projections import public_group_game_summary
from mahjong_agent_runtime.domains.context_builders.game_context import active_game_visible_summary
from mahjong_agent_runtime.tool_consistency import validate_tool_call_consistency, validate_explicit_task_fact_consistency
from mahjong_agent_runtime.models import ToolResult
from mahjong_agent_runtime.matching import waiting_demand_mismatch_reason


RANGE = {"min": 1, "max": 3}


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    instance = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "ranges.sqlite3")
    yield instance
    if isinstance(instance, SQLiteAgentStore):
        instance._connection.close()


def create_game(store, requirement, *, owner="owner"):
    return store.create_game(
        conversation_id=f"chat-{owner}", organizer_id=owner, organizer_name="发起人",
        requirement={"game_type": "red_center_mahjong", "start_time_kind": "asap_when_full", **requirement},
        known_players=[{"customer_id": owner, "display_name": "发起人"}], trace_id=f"setup-{owner}",
    )[0]


def test_accepted_range_is_not_a_fixed_stake_or_cap():
    requirement = normalize_requirement({"stake_range": RANGE})
    assert requirement["stake_range"] == RANGE
    assert requirement["stake_label"] == "1～3元"
    assert not {"stake", "base_stake", "cap_score"} & requirement.keys()
    assert normalize_requirement(requirement) == requirement


@pytest.mark.parametrize("value", ["2-16", "216", "2/16"])
def test_cap_protocol_keeps_base_and_cap(value):
    requirement = normalize_requirement({"stake": value})
    assert requirement["stake"] == "2"
    assert requirement["cap_score"] == 16
    assert "stake_range" not in requirement


@pytest.mark.parametrize("game_type", ["hangzhou_mahjong", "national_standard_mahjong", "club_custom_mahjong"])
def test_structured_base_cap_protocol_is_independent_of_game_type(game_type):
    requirement = normalize_requirement({"game_type": game_type, "stake": "1-256"})
    assert requirement["stake"] == "1"
    assert requirement["cap_score"] == 256
    assert "stake_range" not in requirement


def test_explicit_currency_range_and_separate_cap_remain_distinct():
    requirement = normalize_requirement({"stake_range": {"min": 1, "max": 3}, "cap_score": 256})
    assert requirement["stake_range"] == RANGE and requirement["cap_score"] == 256
    assert "stake" not in requirement
    assert requirement["stake_label"] == "1～3元，256封顶"
    assert normalize_requirement({"stake_range": {"min": 1, "max": 256}})["stake_range"] == {"min": 1, "max": 256}


@pytest.mark.parametrize("bounds", [
    {"min": 3, "max": 1}, {"min": 0, "max": 3}, {"min": True, "max": 3},
    {"min": 1, "max": float("nan")}, {"min": 1, "max": float("inf")},
    {"min": "1", "max": 3}, {"min": 1}, {"min": 1, "max": 3, "extra": 0}, None,
])
def test_invalid_explicit_range_fails_closed(bounds):
    with pytest.raises(ValueError):
        normalize_requirement({"stake_range": bounds})


def test_conflicting_fixed_stake_and_range_are_rejected():
    with pytest.raises(ValueError, match="fixed stake"):
        normalize_requirement({"stake": "5", "stake_range": RANGE})


@pytest.mark.parametrize("stake, matched", [("0.5", False), ("1", True), ("2", True), ("3", True), ("5", False)])
def test_current_games_match_the_whole_closed_range(store, stake, matched):
    game = create_game(store, {"stake": stake})
    results = store.search_current_games({"stake_range": RANGE})
    assert [item["game"]["game_id"] for item in results] == ([game.game_id] if matched else [])


def test_fixed_query_can_match_range_pending_game(store):
    game = create_game(store, {"stake_range": RANGE})
    results = store.search_current_games({"stake": "2"})
    assert [item["game"]["game_id"] for item in results] == [game.game_id]
    assert results[0]["match_kind"] == "exact"
    assert game.requirement["stake_range"] == RANGE
    assert "stake" not in game.requirement


def test_range_intersection_and_cap_are_independent():
    assert stake_requirements_overlap({"stake_range": RANGE}, {"stake_range": {"min": 2, "max": 5}})
    assert not stake_requirements_overlap({"stake_range": RANGE}, {"stake_range": {"min": 4, "max": 5}})
    assert not stake_requirements_overlap({"stake_range": RANGE, "cap_score": 16}, {"stake": "2-32"})
    assert score_stake_preference({"stake_range": RANGE}, ["2"])[0] == 25
    assert classify_current_game_match({"stake_range": RANGE}, {"stake": "2-16"}) == ("exact", [])


def test_candidate_range_is_a_constraint_not_a_soft_score(store):
    preferences = {"one": ["1"], "two": ["2"], "three": ["3"], "unresolved": ["2～5元都可以"],
                   "low": ["0.5"], "high": ["5"], "unknown": []}
    for customer_id, values in preferences.items():
        store.upsert_customer(CustomerProfile(customer_id=customer_id, display_name=customer_id,
            preferred_games=["red_center_mahjong"], preferred_stakes=values,
            smoke_preference="any", response_score=1.0))
    candidates = store.search_customers({"game_type": "red_center_mahjong", "stake_range": RANGE, "smoke_preference": "any"})
    assert {item["customer"]["customer_id"] for item in candidates} == {"one", "two", "three", "unresolved"}
    for item in candidates:
        if item["customer"]["customer_id"] == "unresolved":
            assert "stake_preference_requires_model_interpretation" in item["reasons"]
            assert "stake_matched" not in item["reasons"]
        else:
            assert "stake_matched" in item["reasons"]


def test_game_bound_candidate_search_preserves_range(store):
    game = create_game(store, {"stake_range": RANGE})
    for customer_id, value in [("inside", "2"), ("outside", "5")]:
        store.upsert_customer(CustomerProfile(customer_id=customer_id, display_name=customer_id,
            preferred_games=["red_center_mahjong"], preferred_stakes=[value], response_score=1.0))
    result = ToolGateway(store).execute(ToolCall(name="search_customers", arguments={"game_id": game.game_id}),
        trace_id="range-search", conversation_id="chat-owner", sender_id="owner", sender_name="发起人",
        step_index=1, source_message_id="range-search-message")
    assert result.allowed and result.called
    assert result.result["requirement"]["stake_range"] == RANGE
    assert [item["customer"]["customer_id"] for item in result.result["candidates"]] == ["inside"]


def test_pending_range_is_preserved_in_board_and_public_projection(store):
    game = create_game(store, {"stake_range": RANGE, "smoke_preference": "any"})
    assert "1～3元" in BoardEngine._render_game(1, game)
    assert "档位待定" not in BoardEngine._render_game(1, game)
    assert public_group_game_summary(game)["public_requirement"]["stake_range"] == RANGE
    assert active_game_visible_summary(game)["public_requirement"]["stake_range"] == RANGE


@pytest.mark.parametrize("text", ["1-3块都行", "杭麻打1-256", "1块至3块都可以", "1-256都行"])
def test_backend_does_not_infer_natural_language_range_meaning(text):
    assert parse_stake_range(text) is None


def test_create_cannot_drop_or_narrow_previous_search_range():
    previous = [ToolResult(name="search_current_games", called=True, allowed=True,
        result={"requirement": {"stake_range": RANGE}})]
    for requirement in ({"stake": "1"}, {"stake_range": {"min": 1, "max": 2}}, {}):
        call = ToolCall(name="create_game", arguments={"requirement": requirement})
        assert validate_tool_call_consistency(call, previous)
    assert validate_tool_call_consistency(ToolCall(name="create_game", arguments={"requirement": {"stake_range": RANGE}}), previous) is None


def test_explicit_range_facts_cannot_be_lost():
    call = ToolCall(name="search_current_games", arguments={"requirement": {"stake": "2"}})
    error, _ = validate_explicit_task_fact_consistency(call, {"explicit_task_facts": {"facts": {"stake_range": RANGE}}})
    assert error and "stake_range" in error


def test_partial_patch_preserves_range_but_cannot_narrow_it():
    context = {"explicit_task_facts": {"facts": {"stake_range": RANGE}}}
    call = ToolCall(name="update_game_requirement", arguments={"requirement_patch": {"start_time": "16:00"}})
    assert validate_explicit_task_fact_consistency(call, context)[0] is None
    call.arguments["requirement_patch"]["stake"] = "2"
    assert "stake_range" in validate_explicit_task_fact_consistency(call, context)[0]


def test_sqlite_range_survives_close_and_reopen(tmp_path):
    path = tmp_path / "recovery.sqlite3"
    store = SQLiteAgentStore(path)
    game = create_game(store, {"stake_range": RANGE})
    store._connection.close()
    recovered = SQLiteAgentStore(path)
    try:
        assert recovered.require_game(game.game_id).requirement["stake_range"] == RANGE
        assert recovered.search_current_games({"stake": "2"})[0]["game"]["game_id"] == game.game_id
    finally:
        recovered._connection.close()


def test_requirement_patch_switches_between_fixed_and_range_without_stale_fields(store):
    game = create_game(store, {"stake": "2", "user_visible_summary": "2块红中"})
    updated, _ = store.update_game_requirement(game_id=game.game_id, requirement_patch={"stake_range": RANGE},
        reason="客户接受区间", trace_id="to-range")
    assert updated.requirement["stake_range"] == RANGE
    assert "stake" not in updated.requirement and "base_stake" not in updated.requirement
    assert "user_visible_summary" not in updated.requirement
    fixed, _ = store.update_game_requirement(game_id=game.game_id, requirement_patch={"stake": "2"},
        reason="客户确认具体档位", trace_id="to-fixed")
    assert fixed.requirement["stake"] == "2" and "stake_range" not in fixed.requirement
    assert fixed.requirement["stake_label"] == "2"


def test_range_waiting_demand_is_persisted_and_matches_only_accepted_stakes(store):
    result = ToolGateway(store).execute(ToolCall(name="register_waiting_demand", arguments={
        "stake_range": RANGE, "smoke_preference": "不限", "time_preference": "不限"}),
        trace_id="waiting-range", conversation_id="waiting-customer-chat", sender_id="waiting-customer",
        sender_name="候补客户", step_index=1, source_message_id="waiting-range-message")
    assert result.called and result.allowed
    demand = next(iter(store.waiting_demands.values()))
    assert demand.demand["stake_range"] == RANGE and "stake" not in demand.demand
    for stake in ("1", "2", "3", "5"):
        game = create_game(store, {"stake": stake, "smoke_preference": "any"}, owner=f"owner-{stake}")
        assert waiting_demand_mismatch_reason(demand, game) == ("stake_mismatch" if stake == "5" else "")


def test_range_waiting_demand_keeps_cap_and_rejects_cap_conflict(store):
    result = ToolGateway(store).execute(ToolCall(name="register_waiting_demand", arguments={
        "stake_range": RANGE, "cap_score": 256, "smoke_preference": "不限", "time_preference": "不限"}),
        trace_id="waiting-cap", conversation_id="waiting-cap-chat", sender_id="waiting-customer",
        sender_name="候补客户", step_index=1, source_message_id="waiting-cap-message")
    assert result.called and result.allowed
    demand = next(iter(store.waiting_demands.values()))
    assert demand.demand["stake_range"] == RANGE and demand.demand["cap_score"] == 256
    for cap in (128, 256):
        game = create_game(store, {"stake": "2", "cap_score": cap, "smoke_preference": "any"}, owner=f"cap-{cap}")
        assert waiting_demand_mismatch_reason(demand, game) == ("stake_mismatch" if cap == 128 else "")
    assert "256封顶" in BoardEngine._render_game(1, create_game(store, {"stake_range": RANGE, "cap_score": 256}))
