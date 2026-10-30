"""查局通配值是明确的只读参数协议，不能变成自然语言解析或修改写入条件。"""

from copy import deepcopy

import pytest

from mahjong_agent_runtime import CustomerProfile, InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway
from mahjong_agent_runtime.domains.game_domain import normalize_current_game_query, normalize_requirement
from mahjong_agent_runtime.domains.tools.search_tools import canonical_search_current_games_arguments


@pytest.fixture(params=["memory", "sqlite", "sqlite_reopened"])
def query_store(request, tmp_path):
    path = tmp_path / "query-protocol.sqlite3"
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(path)
    store.upsert_customer(CustomerProfile(customer_id="requester", display_name="测试客户",
        preferred_games=["national_standard_mahjong"], preferred_stakes=["2"]))
    for game_type, stake, seats in [("hangzhou_mahjong", "1", 1), ("national_standard_mahjong", "2", 1),
                                    ("custom_any_game", "1", 3)]:
        store.create_game(conversation_id="fixture-"+game_type, organizer_id="owner-"+game_type,
            organizer_name="测试发起人", requirement={"game_type": game_type, "stake": stake,
                "start_time_kind": "asap_when_full", "smoke_preference": "any"},
            known_players=[{"customer_id": "owner-"+game_type, "display_name": "测试发起人", "seat_count": seats}],
            trace_id="setup-"+game_type)
    if request.param == "sqlite_reopened":
        store._connection.close()
        store = SQLiteAgentStore(path)
    yield store
    if isinstance(store, SQLiteAgentStore):
        store._connection.close()


def result_types(rows):
    return {item["game"]["requirement"]["game_type"] for item in rows}


def test_query_normalization_preserves_all_other_constraints_and_caller_payload():
    original = {"game_type": "any", "stake_range": {"min": 1, "max": 3}, "cap_score": 256,
        "smoke_preference": "no_smoking", "requesting_party": {"contact_id": "a", "seat_count": 3},
        "start_time": "19:00", "custom_constraint": {"value": "unchanged"}}
    before = deepcopy(original)
    assert normalize_current_game_query(original) == normalize_requirement({k: v for k, v in original.items() if k != "game_type"})
    assert original == before


@pytest.mark.parametrize("value", ["ANY", "Any", " any ", "任意", "不限", "custom_any_game", "unlisted_game"])
def test_query_only_reserves_exact_protocol_value_and_never_parses_synonyms(value):
    assert normalize_current_game_query({"game_type": value})["game_type"] == value


def test_any_and_omitted_game_type_have_same_store_results_without_personal_profile_filter(query_store):
    omitted = query_store.search_current_games({}, sender_id="requester", limit=20)
    wildcard = query_store.search_current_games({"game_type": "any"}, sender_id="requester", limit=20)
    assert wildcard == omitted
    assert result_types(wildcard) == {"hangzhou_mahjong", "national_standard_mahjong", "custom_any_game"}


@pytest.mark.parametrize("game_type", ["hangzhou_mahjong", "national_standard_mahjong", "custom_any_game"])
def test_explicit_open_game_types_remain_scoped(query_store, game_type):
    result = query_store.search_current_games({"game_type": game_type}, sender_id="requester")
    assert result_types(result) == {game_type}


def test_wildcard_does_not_remove_stake_or_party_capacity_filters(query_store):
    query = {"game_type": "any", "stake": "1", "requesting_party": {"contact_id": "requester", "seat_count": 3}}
    result = query_store.search_current_games(query, sender_id="requester")
    assert result_types(result) == {"hangzhou_mahjong"}
    assert result[0]["join_projection"]["requested_seats"] == 3
    assert result[0]["join_projection"]["would_fill_game"] is True


def test_gateway_returns_actual_unfiltered_scope_keeps_original_proposal_and_query_identity(query_store):
    gateway = ToolGateway(store=query_store)
    arguments = {"requirement": {"game_type": "any"}}
    call = ToolCall(name="search_current_games", arguments=deepcopy(arguments))
    result = gateway.execute(call, trace_id="query-any", conversation_id="request-chat",
        sender_id="requester", sender_name="测试客户", step_index=1)
    assert result.called and result.allowed and result.error is None
    assert call.arguments == arguments
    assert result.result["requirement"] == {}
    assert result_types(result.result["matches"]) == {"hangzhou_mahjong", "national_standard_mahjong", "custom_any_game"}
    contract = result.result["customer_reply_contract"]
    assert contract["evidence_scope"]["kind"] == "unfiltered_inventory"
    assert contract["evidence_scope"]["applied_filter_fields"] == []
    assert contract["matched_query_requirement"] == {}
    assert canonical_search_current_games_arguments(arguments) == canonical_search_current_games_arguments({"requirement": {}})
    assert len(query_store.games) == 3 and not query_store.invite_drafts


def test_query_wildcard_never_changes_game_write_normalization_or_persisted_payload(query_store):
    requirement = {"game_type": "any", "stake": "1", "start_time_kind": "asap_when_full"}
    assert normalize_requirement(requirement)["game_type"] == "any"
    game, _ = query_store.create_game(conversation_id="write-any", organizer_id="writer", organizer_name="写入客户",
        requirement=requirement, known_players=[{"customer_id": "writer", "display_name": "写入客户"}], trace_id="write-any")
    assert query_store.require_game(game.game_id).requirement["game_type"] == "any"


def test_model_tool_contract_declares_query_only_wildcard():
    spec = next(item for item in ToolGateway(store=InMemoryAgentStore()).tool_specs_for_prompt() if item["name"] == "search_current_games")
    assert "查询保留通配值 any" in spec["description"]
    assert "其余玩法标识按原值过滤" in spec["description"]
