from __future__ import annotations

import pytest

from mahjong_agent_runtime import (
    CustomerProfile,
    InMemoryAgentStore,
    SQLiteAgentStore,
    ToolCall,
    ToolGateway,
)
from mahjong_agent_runtime.domains.tools.search_tools import search_current_games, search_customers


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryAgentStore()
    return SQLiteAgentStore(tmp_path / "search-classification.sqlite3")


def _create_open_game(store, *, stake: str = "0.5") -> None:
    store.create_game(
        conversation_id="source-conversation",
        organizer_id="owner-customer",
        organizer_name="发起人",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": stake,
            "smoke_preference": "no_smoke",
            "start_time_kind": "asap_when_full",
            "needed_seats": 3,
            "user_visible_summary": f"{stake}无烟，人齐开",
        },
        known_players=[{"customer_id": "owner-customer", "display_name": "发起人"}],
        trace_id="setup-search-classification",
    )


def _execute_search(store, *, sender_id: str) -> dict:
    result = search_current_games(
        store,
        ToolCall(
            name="search_current_games",
            arguments={
                "requirement": {
                    "game_type": "hangzhou_mahjong",
                    "stake": "1",
                    "smoke_preference": "no_smoke",
                    "start_time_kind": "asap_when_full",
                }
            },
        ),
        "trace-search-classification",
        "request-conversation",
        sender_id,
        "测试用户",
    )
    assert result.called is True
    return result.result


def test_explicit_stake_mismatch_without_profile_evidence_is_not_actionable(store) -> None:
    store.upsert_customer(
        CustomerProfile(
            customer_id="requester",
            display_name="测试用户",
            preferred_games=["hangzhou_mahjong"],
        )
    )
    _create_open_game(store, stake="0.5")

    payload = _execute_search(store, sender_id="requester")

    assert payload["matches"] == []
    assert payload["alternatives"] == []
    assert payload["customer_reply_contract"]["search_result_semantics"]["status"] == "no_actionable_match"


def test_profile_supported_stake_mismatch_is_labeled_as_decision_required_alternative(store) -> None:
    store.upsert_customer(
        CustomerProfile(
            customer_id="requester",
            display_name="测试用户",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["0.5", "1"],
            smoke_preference="no_smoke",
        )
    )
    _create_open_game(store, stake="0.5")

    payload = _execute_search(store, sender_id="requester")

    assert payload["matches"] == []
    assert len(payload["alternatives"]) == 1
    alternative = payload["alternatives"][0]
    assert alternative["match_kind"] == "profile_supported_alternative"
    assert alternative["decision_required_fields"] == ["stake"]
    semantics = payload["customer_reply_contract"]["search_result_semantics"]
    assert semantics["status"] == "decision_required_alternatives"
    assert semantics["actionable_match_count"] == 0
    assert semantics["alternative_count"] == 1


def test_exact_game_is_returned_as_actionable_match(store) -> None:
    store.upsert_customer(CustomerProfile(customer_id="requester", display_name="测试用户"))
    _create_open_game(store, stake="1")

    payload = _execute_search(store, sender_id="requester")

    assert len(payload["matches"]) == 1
    assert payload["matches"][0]["match_kind"] == "exact"
    assert payload["matches"][0]["decision_required_fields"] == []
    assert payload["alternatives"] == []
    assert payload["customer_reply_contract"]["search_result_semantics"]["status"] == "actionable_matches"


def test_smoke_aliases_do_not_turn_exact_game_into_alternative(store) -> None:
    store.upsert_customer(CustomerProfile(customer_id="requester", display_name="测试用户"))
    _create_open_game(store, stake="0.5")

    result = search_current_games(
        store,
        ToolCall(
            name="search_current_games",
            arguments={
                "requirement": {
                    "game_type": "hangzhou_mahjong",
                    "stake": "0.5",
                    "smoke_preference": "no_smoking",
                    "start_time_kind": "asap_when_full",
                }
            },
        ),
        "trace-smoke-alias-classification",
        "request-conversation",
        "requester",
        "测试用户",
    )

    assert len(result.result["matches"]) == 1
    assert result.result["alternatives"] == []
    assert result.result["matches"][0]["decision_required_fields"] == []


def test_candidate_search_excludes_requester_and_existing_players(store) -> None:
    for customer_id in ("requester", "existing-player", "candidate"):
        store.upsert_customer(
            CustomerProfile(
                customer_id=customer_id,
                display_name=customer_id,
                preferred_games=["hangzhou_mahjong"],
                preferred_stakes=["1"],
                smoke_preference="no_smoke",
                response_score=0.8,
            )
        )

    result = search_customers(
        store,
        ToolCall(
            name="search_customers",
            arguments={
                "requirement": {
                    "game_type": "hangzhou_mahjong",
                    "stake": "1",
                    "smoke_preference": "no_smoke",
                    "existing_player_ids": ["existing-player"],
                    "requesting_party": {
                        "contact_id": "requester",
                        "known_member_ids": ["requester"],
                        "seat_count": 1,
                    },
                }
            },
        ),
        "trace-candidate-exclusion",
        "request-conversation",
        "requester",
        "测试用户",
    )

    candidate_ids = [item["customer"]["customer_id"] for item in result.result["candidates"]]
    assert candidate_ids == ["candidate"]
    assert result.result["exclude_customer_ids"] == ["existing-player", "requester"]


def _create_three_seat_party_game(store):
    game, _ = store.create_game(
        conversation_id="candidate-search-conversation",
        organizer_id="requester",
        organizer_name="发起人",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "1",
            "smoke_preference": "no_smoke",
            "start_time_kind": "asap_when_full",
            "known_player_count": 3,
            "needed_seats": 1,
            "seat_format": "371",
        },
        known_players=[
            {
                "customer_id": "requester",
                "display_name": "发起人",
                "seat_count": 3,
                "known_member_ids": ["requester"],
                "anonymous_seat_count": 2,
            }
        ],
        trace_id="setup-three-seat-party-game",
    )
    return game


@pytest.mark.parametrize("include_game_id", [True, False])
def test_candidate_search_binds_authoritative_seat_facts_from_active_game(
    store,
    include_game_id: bool,
) -> None:
    for customer_id in ("requester", "candidate"):
        store.upsert_customer(
            CustomerProfile(
                customer_id=customer_id,
                display_name=customer_id,
                preferred_games=["hangzhou_mahjong"],
                preferred_stakes=["1"],
                smoke_preference="no_smoke",
                response_score=0.8,
            )
        )
    game = _create_three_seat_party_game(store)
    arguments = {
        # Deliberately omit all seat counters. The aggregate, not the model,
        # owns current occupancy after create_game has succeeded.
        "requirement": {
            "game_type": "hangzhou_mahjong",
            "stake": "1",
            "smoke_preference": "no_smoke",
            "start_time_kind": "asap_when_full",
        }
    }
    if include_game_id:
        arguments["game_id"] = game.game_id

    result = search_customers(
        store,
        ToolCall(name="search_customers", arguments=arguments),
        "trace-authoritative-candidate-search",
        "candidate-search-conversation",
        "requester",
        "发起人",
    )

    assert result.called is True
    assert result.allowed is True
    assert result.result["bound_game_id"] == game.game_id
    assert result.result["requirement_source"] == "active_game_aggregate"
    assert result.result["requirement"]["known_player_count"] == 3
    assert result.result["requirement"]["needed_seats"] == 1
    assert result.result["requirement"]["remaining_seats"] == 1
    assert result.result["requirement"]["seat_format"] == "371"
    assert result.result["exclude_customer_ids"] == ["requester"]


def test_tool_gateway_accepts_game_bound_candidate_search_without_requirement(store) -> None:
    for customer_id in ("requester", "candidate"):
        store.upsert_customer(
            CustomerProfile(
                customer_id=customer_id,
                display_name=customer_id,
                preferred_games=["hangzhou_mahjong"],
                preferred_stakes=["1"],
                smoke_preference="no_smoke",
                response_score=0.8,
            )
        )
    game = _create_three_seat_party_game(store)

    result = ToolGateway(store).execute(
        ToolCall(
            name="search_customers",
            arguments={
                "game_id": game.game_id,
                "exclude_customer_ids": ["requester"],
                "limit": 8,
            },
            reason="按刚创建的局查询候选人。",
        ),
        trace_id="trace-gateway-game-bound-search",
        conversation_id="candidate-search-conversation",
        sender_id="requester",
        sender_name="发起人",
        step_index=1,
        source_message_id="msg-gateway-game-bound-search",
    )

    assert result.called is True
    assert result.allowed is True
    assert result.error is None
    assert result.result["bound_game_id"] == game.game_id
    assert result.result["requirement_source"] == "active_game_aggregate"
    assert result.result["requirement"]["known_player_count"] == 3
    assert result.result["requirement"]["needed_seats"] == 1
    assert result.result["requirement"]["seat_format"] == "371"


@pytest.mark.parametrize("arguments", [{}, {"requirement": {}}])
def test_tool_gateway_rejects_unbounded_candidate_search(store, arguments: dict) -> None:
    result = ToolGateway(store).execute(
        ToolCall(
            name="search_customers",
            arguments=arguments,
            reason="缺少查询边界。",
        ),
        trace_id=f"trace-gateway-unbounded-search-{len(arguments)}",
        conversation_id="candidate-search-conversation",
        sender_id="requester",
        sender_name="发起人",
        step_index=1,
        source_message_id=f"msg-gateway-unbounded-search-{len(arguments)}",
    )

    assert result.called is False
    assert result.allowed is False
    assert result.error
