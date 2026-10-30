"""只读检索的容量硬边界在评分前生效，Memory与SQLite保持同一可入座语义。"""
from __future__ import annotations

import pytest

from mahjong_agent_runtime import CustomerProfile, InMemoryAgentStore, SQLiteAgentStore, ToolCall
from mahjong_agent_runtime.domains import join_projection
from mahjong_agent_runtime.domains.customer_domain import score_requirement
from mahjong_agent_runtime.domains.tools.search_tools import search_current_games


@pytest.fixture(params=["memory", "sqlite", "sqlite_reopened"])
def capacity_store(request, tmp_path):
    """重开模式在种子写入后关闭连接，保证检索读取的确是SQLite已持久化的容量。"""
    db_path = tmp_path / "search-capacity.sqlite3"
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(db_path)
    store.upsert_customer(CustomerProfile(customer_id="requester", display_name="请求方",
        preferred_games=["national_standard_mahjong"], preferred_stakes=["1", "2"], smoke_preference="any"))
    active = [store]

    def reopen():
        if request.param == "sqlite_reopened":
            active[0]._connection.close()
            active[0] = SQLiteAgentStore(db_path)
        return active[0]

    yield store, reopen
    if isinstance(active[0], SQLiteAgentStore):
        active[0]._connection.close()


def create_game(store, label, remaining, *, stake="1", extra=None, organizer_id=None):
    organizer = organizer_id or f"owner-{label}"
    game, _ = store.create_game(conversation_id=f"conversation-{label}", organizer_id=organizer,
        organizer_name=organizer, requirement={"game_type": "national_standard_mahjong", "stake": stake,
            "smoke_preference": "any", "start_time_kind": "asap_when_full", **(extra or {})},
        known_players=[{"customer_id": organizer, "display_name": organizer, "seat_count": 4-remaining}],
        trace_id=f"setup-{label}")
    assert game.remaining_seats() == remaining
    return game


def request_for(seats):
    return {"game_type": "national_standard_mahjong", "stake": "1", "smoke_preference": "any",
        "requesting_party": {"contact_id": "requester", "seat_count": seats}}


@pytest.mark.parametrize("seats", [1, 2, 3])
def test_search_filters_explicit_party_capacity_for_one_two_and_three_remaining_seats(capacity_store, seats):
    store, reopen = capacity_store
    games = {remaining: create_game(store, str(remaining), remaining) for remaining in [1, 2, 3]}
    store = reopen()
    results = store.search_current_games(request_for(seats), sender_id="requester", limit=10)
    assert {item["game"]["game_id"] for item in results} == {
        game.game_id for remaining, game in games.items() if remaining >= seats}
    for item in results:
        projection = item["join_projection"]
        assert projection["requested_seats"] == seats
        assert projection["would_overfill_game"] is False
        assert projection["would_fill_game"] is (projection["remaining_seats_before_join"] == seats)


def test_capacity_filter_runs_before_ranking_and_top_limit(capacity_store):
    store, reopen = capacity_store
    overfull = create_game(store, "overfull-high-score", 1, extra={"duration_kind": "fixed", "duration_hours": 4})
    usable = create_game(store, "usable-low-score", 3)
    requirement = {**request_for(3), "duration_kind": "fixed"}
    assert score_requirement(requirement, overfull.requirement)[0] > score_requirement(requirement, usable.requirement)[0]
    store = reopen()
    results = store.search_current_games(requirement, sender_id="requester", limit=1)
    assert [item["game"]["game_id"] for item in results] == [usable.game_id]


def test_tool_matches_and_profile_alternatives_both_exclude_overcapacity(capacity_store):
    store, reopen = capacity_store
    exact = create_game(store, "exact-can-seat-three", 3)
    alternative = create_game(store, "alternative-can-seat-three", 3, stake="2")
    for remaining in [1, 2]:
        create_game(store, f"exact-too-small-{remaining}", remaining)
        create_game(store, f"alternative-too-small-{remaining}", remaining, stake="2")
    store = reopen()
    result = search_current_games(store, ToolCall(name="search_current_games",
        arguments={"requirement": request_for(3), "limit": 10}, reason="查询三人可入座的局"),
        trace_id="capacity-tool", conversation_id="request-chat", sender_id="requester", sender_name="请求方")
    assert [item["game"]["game_id"] for item in result.result["matches"]] == [exact.game_id]
    assert [item["game"]["game_id"] for item in result.result["alternatives"]] == [alternative.game_id]
    assert result.result["customer_reply_contract"]["search_result_semantics"]["actionable_match_count"] == 1
    assert all(not item["join_projection"]["would_overfill_game"] for item in result.result["matches"] + result.result["alternatives"])


@pytest.mark.parametrize("party", [None, {"contact_id": "requester"}])
def test_unknown_party_size_keeps_existing_default_and_never_uses_target_population(capacity_store, party):
    store, reopen = capacity_store
    game = create_game(store, "one-seat-default", 1)
    requirement = {"game_type": "national_standard_mahjong", "stake": "1", "known_player_count": 3, "needed_seats": 1}
    if party is not None:
        requirement["requesting_party"] = party
    store = reopen()
    result = store.search_current_games(requirement, sender_id="requester")
    assert [item["game"]["game_id"] for item in result] == [game.game_id]
    assert result[0]["join_projection"]["requested_seats"] == 1
    assert result[0]["join_projection"]["would_fill_game"] is True


def test_same_game_existing_sender_does_not_add_their_party_again_or_resize_it(capacity_store):
    store, reopen = capacity_store
    game = create_game(store, "already-joined", 2, organizer_id="requester")
    store = reopen()
    results = store.search_current_games(request_for(3), sender_id="requester")
    assert [item["game"]["game_id"] for item in results] == [game.game_id]
    projection = results[0]["join_projection"]
    assert projection["sender_already_joined"] is True and projection["requested_seats"] == 0
    assert projection["remaining_seats_before_join"] == projection["remaining_seats_after_join"] == 2
    assert projection["would_fill_game"] is False and projection["would_overfill_game"] is False
    persisted = store.require_game(game.game_id)
    assert persisted.participants[0].seat_count == 2


def test_valid_three_seat_match_really_accepts_three_and_persists_after_reopen(capacity_store):
    store, reopen = capacity_store
    game = create_game(store, "can-really-join", 3)
    store = reopen()
    result = store.search_current_games(request_for(3), sender_id="requester")
    assert result[0]["join_projection"]["would_fill_game"] is True
    joined, _ = store.join_game(game_id=game.game_id, customer_id="requester", display_name="请求方", seat_count=3,
        trace_id="actual-three-seat-join")
    assert joined.remaining_seats() == 0
    persisted = reopen().require_game(game.game_id)
    assert persisted.remaining_seats() == 0
    assert next(item for item in persisted.participants if item.customer_id == "requester").seat_count == 3


@pytest.mark.parametrize("remaining,requested,overfills,fills", [(1, 3, True, False), (2, 3, True, False), (3, 3, False, True)])
def test_join_projection_never_reports_overfill_as_exact_fill(capacity_store, remaining, requested, overfills, fills):
    store, reopen = capacity_store
    game = create_game(store, f"projection-{remaining}", remaining)
    projection = join_projection(reopen().require_game(game.game_id), sender_id="requester", requested_seats=requested)
    assert projection["would_overfill_game"] is overfills
    assert projection["would_fill_game"] is fills
