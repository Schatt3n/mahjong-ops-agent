"""查询人数使用独立typed合同；Gateway拒绝错层字段，不能靠后台补数或吞字段执行。"""
from __future__ import annotations

from copy import deepcopy

import pytest

from mahjong_agent_runtime import CustomerProfile, InMemoryAgentStore, InMemoryTraceRecorder, SQLiteAgentStore, ToolCall, ToolGateway


@pytest.fixture(params=["memory", "sqlite"])
def query_gateway(request, tmp_path):
    """两事实源都从真实局池检索，拒绝后只允许增加审计/幂等记录，业务状态不能改变。"""
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "query-party.sqlite3")
    store.upsert_customer(CustomerProfile(customer_id="requester", display_name="请求方",
        preferred_games=["national_standard_mahjong"], preferred_stakes=["1"], smoke_preference="any"))
    ids = {}
    for remaining in [1, 3]:
        owner = f"owner-{remaining}"
        game, _ = store.create_game(conversation_id=f"pool-{remaining}", organizer_id=owner, organizer_name=owner,
            requirement={"game_type": "national_standard_mahjong", "stake": "1", "smoke_preference": "any",
                "start_time_kind": "asap_when_full"},
            known_players=[{"customer_id": owner, "display_name": owner, "seat_count": 4-remaining}],
            trace_id=f"setup-{remaining}")
        ids[remaining] = game.game_id
    trace = InMemoryTraceRecorder()
    yield store, ToolGateway(store=store, trace_recorder=trace), trace, ids
    if isinstance(store, SQLiteAgentStore):
        store._connection.close()


def requirement():
    return {"game_type": "national_standard_mahjong", "stake": "1", "smoke_preference": "any"}


def execute(gateway, arguments, *, source="query-input", trace_id="query-trace"):
    return gateway.execute(ToolCall(name="search_current_games", arguments=arguments, reason="验证查询人数合同"),
        trace_id=trace_id, conversation_id="request-chat", sender_id="requester", sender_name="请求方",
        step_index=1, source_message_id=source)


def business_snapshot(store):
    """错误诊断可持久化，但不能创建/修改业务实体、人数、版本或对话内容。"""
    return {"games": {key: value.to_dict() for key, value in store.games.items()},
        "invites": {key: value.to_dict() for key, value in store.invite_drafts.items()},
        "outbound": {key: value.to_dict() for key, value in store.outbound_message_drafts.items()},
        "scheduled": {key: value.to_dict() for key, value in store.scheduled_tasks.items()},
        "version": store.conversation_version("request-chat"),
        "turns": [turn.to_dict() for turn in store.recent_turns("request-chat")],
        "transitions": [transition.to_dict() for transition in store.transitions]}


def assert_schema_rejected(store, gateway, trace, arguments):
    before = business_snapshot(store)
    original = deepcopy(arguments)
    result = execute(gateway, arguments)
    assert result.called is False and result.allowed is False and result.error
    assert result.state_transitions == [] and business_snapshot(store) == before
    assert arguments == original
    assert any(event.step == "tool_schema_checked" and event.content["allowed"] is False
        for event in trace.get_trace("query-trace"))
    return result


def test_query_schema_declares_party_under_requirement_without_forcing_unknown_seats(query_gateway):
    _, gateway, _, _ = query_gateway
    schema = gateway.tools["search_current_games"].schema
    assert set(schema["properties"]) == {"requirement", "limit"}
    assert schema["additionalProperties"] is False and schema["required"] == ["requirement"]
    query = schema["properties"]["requirement"]
    assert "requesting_party" in query["properties"]
    assert "requesting_party" not in query.get("required", [])
    party = query["properties"]["requesting_party"]
    assert party["type"] == "object" and party["additionalProperties"] is False
    assert "seat_count" not in party.get("required", []) and "contact_id" not in party.get("required", [])
    seat = party["properties"]["seat_count"]
    assert seat["type"] == "integer" and seat["minimum"] == 1 and seat["maximum"] == 4


@pytest.mark.parametrize("contact", [False, True])
def test_query_correct_nested_three_person_party_reaches_real_capacity_filter(query_gateway, contact):
    store, gateway, _, ids = query_gateway
    party = {"seat_count": 3}
    if contact:
        party.update(contact_id="requester", contact_name="请求方")
    before = business_snapshot(store)
    result = execute(gateway, {"requirement": {**requirement(), "requesting_party": party}, "limit": 8})
    assert result.called and result.allowed and not result.error
    assert [item["game"]["game_id"] for item in result.result["matches"]] == [ids[3]]
    projection = result.result["matches"][0]["join_projection"]
    assert projection["requested_seats"] == 3 and projection["would_fill_game"] is True
    assert projection["would_overfill_game"] is False
    assert result.result["requirement"]["requesting_party"]["seat_count"] == 3
    assert business_snapshot(store) == before


@pytest.mark.parametrize("query", [{}, {"game_type": "national_standard_mahjong"},
    {"requesting_party": {}}, {"requesting_party": {"contact_id": "requester"}}])
def test_inventory_and_unknown_party_queries_may_omit_personal_seat_count(query_gateway, query):
    store, gateway, _, ids = query_gateway
    before = business_snapshot(store)
    result = execute(gateway, {"requirement": deepcopy(query)})
    assert result.called and result.allowed and not result.error
    assert {item["game"]["game_id"] for item in result.result["matches"]} == set(ids.values())
    assert "seat_count" not in result.result["requirement"].get("requesting_party", {})
    assert business_snapshot(store) == before


@pytest.mark.parametrize("seats", [1, 4])
def test_query_party_valid_integer_bounds_are_accepted_without_inventing_capacity(query_gateway, seats):
    store, gateway, _, ids = query_gateway
    before = business_snapshot(store)
    result = execute(gateway, {"requirement": {**requirement(), "requesting_party": {"seat_count": seats}}})
    assert result.called and result.allowed and not result.error
    expected = set(ids.values()) if seats == 1 else set()
    assert {item["game"]["game_id"] for item in result.result["matches"]} == expected
    assert business_snapshot(store) == before


def test_target_game_population_is_not_substituted_for_requester_seats(query_gateway):
    _, gateway, _, ids = query_gateway
    result = execute(gateway, {"requirement": {**requirement(), "known_player_count": 3, "needed_seats": 1}})
    assert result.called and result.allowed and not result.error
    one_seat = next(item for item in result.result["matches"] if item["game"]["game_id"] == ids[1])
    assert one_seat["join_projection"]["requested_seats"] == 1
    assert "requesting_party" not in result.result["requirement"]


@pytest.mark.parametrize("size", [0, 5, -1, 1.5, 3.0, "3", True, False, None])
def test_query_party_seat_count_requires_real_integer_one_to_four(query_gateway, size):
    store, gateway, trace, _ = query_gateway
    assert_schema_rejected(store, gateway, trace, {"requirement": {**requirement(), "requesting_party": {"seat_count": size}}})


@pytest.mark.parametrize("party", [None, "three", []])
def test_query_party_itself_must_be_typed_object(query_gateway, party):
    store, gateway, trace, _ = query_gateway
    assert_schema_rejected(store, gateway, trace, {"requirement": {**requirement(), "requesting_party": party}})


@pytest.mark.parametrize("extra", [{"requesting_party": {"seat_count": 3}}, {"seat_count": 3},
    {"party_size": 3}, {"game_id": "not-a-query-argument"}])
def test_misplaced_query_party_or_unknown_top_level_field_is_not_silently_swallowed(query_gateway, extra):
    store, gateway, trace, _ = query_gateway
    assert_schema_rejected(store, gateway, trace, {"requirement": requirement(), **extra})


@pytest.mark.parametrize("value", [True, False])
def test_query_limit_boolean_is_not_an_integer_limit(query_gateway, value):
    store, gateway, trace, _ = query_gateway
    assert_schema_rejected(store, gateway, trace, {"requirement": requirement(), "limit": value})


def test_successful_cached_read_does_not_authorize_new_misplaced_party_arguments(query_gateway):
    store, gateway, trace, _ = query_gateway
    valid = execute(gateway, {"requirement": requirement()})
    assert valid.called and valid.allowed and not valid.error
    assert_schema_rejected(store, gateway, trace, {"requirement": requirement(), "requesting_party": {"seat_count": 3}})


def test_rejected_outer_party_can_be_corrected_in_same_source_turn_without_backend_repair(query_gateway):
    store, gateway, trace, ids = query_gateway
    assert_schema_rejected(store, gateway, trace, {"requirement": requirement(), "requesting_party": {"seat_count": 3}})
    result = execute(gateway, {"requirement": {**requirement(), "requesting_party": {"seat_count": 3}}}, trace_id="corrected-query")
    assert result.called and result.allowed and not result.error
    assert [item["game"]["game_id"] for item in result.result["matches"]] == [ids[3]]


def test_schema_rejection_does_not_poison_valid_same_canonical_query_when_extra_field_is_removed(query_gateway):
    store, gateway, trace, ids = query_gateway
    assert_schema_rejected(store, gateway, trace, {"requirement": requirement(), "requesting_party": {"seat_count": 3}})
    # 错层参数从未执行，删除该参数后的合法未知人数查询也不能复用它的拒绝缓存。
    result = execute(gateway, {"requirement": requirement()}, trace_id="valid-unknown-query")
    assert result.called and result.allowed and not result.error
    assert result.deduplicated is False
    assert {item["game"]["game_id"] for item in result.result["matches"]} == set(ids.values())
    assert "requesting_party" not in result.result["requirement"]


@pytest.mark.parametrize("field", ["start_time", "start_time_kind", "planned_start_at"])
@pytest.mark.parametrize("value", [None, True, 1830, {"time": "18:30"}])
def test_query_time_fields_when_present_require_strings_before_handler(query_gateway, field, value):
    """可省略不代表允许任意类型；原始参数须在规范化或检索处理器之前被拒绝。"""
    store, gateway, trace, _ = query_gateway
    assert_schema_rejected(store, gateway, trace, {"requirement": {**requirement(), field: value}})


def test_query_time_schema_exposes_optional_fields_and_actual_result_keeps_calendar_anchor(query_gateway):
    store, gateway, _, _ = query_gateway
    schema = gateway.tools["search_current_games"].schema["properties"]["requirement"]
    fields = {"start_time", "start_time_kind", "planned_start_at"}
    assert fields <= schema["properties"].keys()
    assert fields.isdisjoint(schema.get("required", []))
    assert all(schema["properties"][key]["type"] == "string" for key in fields)
    time_requirement = {"start_time": "18:30", "start_time_kind": "scheduled",
        "planned_start_at": "2026-10-30T18:30:00+08:00"}
    before = business_snapshot(store)
    result = execute(gateway, {"requirement": {**requirement(), **time_requirement}})
    assert result.called and result.allowed and not result.error
    assert all(result.result["requirement"][key] == value for key, value in time_requirement.items())
    assert business_snapshot(store) == before
