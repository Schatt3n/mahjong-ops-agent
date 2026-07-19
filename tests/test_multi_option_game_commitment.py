from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest

from mahjong_agent_runtime import CustomerProfile, InMemoryAgentStore, SQLiteAgentStore, ToolGateway
from mahjong_agent_runtime.models import GameStatus, ToolCall, now


def _store(kind: str, tmp_path, name: str = "multi_option"):
    if kind == "memory":
        return InMemoryAgentStore()
    return SQLiteAgentStore(tmp_path / f"{name}.sqlite3")


def _scheduled_requirement(start_at, duration_hours: float = 4) -> dict:
    return {
        "game_type": "hangzhou_mahjong",
        "stake": "0.5",
        "smoke_preference": "no_smoking",
        "start_time_kind": "scheduled",
        "start_at": start_at.isoformat(),
        "duration_hours": duration_hours,
        "known_player_count": 2,
        "needed_seats": 2,
    }


def _create_two_person_option(
    store,
    *,
    conversation_id: str,
    organizer_id: str,
    shared_customer_id: str,
    requirement: dict,
):
    return store.create_game(
        conversation_id=conversation_id,
        organizer_id=organizer_id,
        organizer_name=organizer_id,
        requirement=requirement,
        known_players=[
            {"customer_id": organizer_id, "display_name": organizer_id, "seat_count": 1},
            {"customer_id": shared_customer_id, "display_name": shared_customer_id, "seat_count": 1},
        ],
        trace_id=f"trace_create_{conversation_id}",
    )[0]


def _confirm(store, game_id: str, customer_id: str):
    return store.record_candidate_reply(
        game_id=game_id,
        customer_id=customer_id,
        display_name=customer_id,
        status="confirmed",
        seat_count=1,
        trace_id=f"trace_confirm_{game_id}_{customer_id}",
    )


def _active_customer_ids(game) -> set[str]:
    return {
        item.customer_id
        for item in game.participants
        if item.status in {"joined", "confirmed"}
    }


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_first_full_option_commits_shared_customer_and_releases_losing_option(kind: str, tmp_path) -> None:
    store = _store(kind, tmp_path)
    start = now() + timedelta(days=1)
    original = store.create_game(
        conversation_id="conversation_a",
        organizer_id="A",
        organizer_name="A",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "no_smoking",
            "start_time_kind": "asap_when_full",
            "known_player_count": 2,
            "needed_seats": 2,
        },
        known_players=[
            {"customer_id": "A", "display_name": "A"},
            {"customer_id": "B", "display_name": "B"},
        ],
        trace_id="trace_create_a",
    )[0]
    original, _ = store.update_game_requirement(
        game_id=original.game_id,
        requirement_patch={
            "start_time_kind": "scheduled",
            "planned_start_at": start.isoformat(),
            "duration_hours": 4,
        },
        reason="A确认最晚开始时间和最长时长",
        trace_id="trace_update_a",
    )
    alternative = _create_two_person_option(
        store,
        conversation_id="conversation_c",
        organizer_id="C",
        shared_customer_id="B",
        requirement=_scheduled_requirement(start + timedelta(minutes=30), duration_hours=5),
    )

    _confirm(store, alternative.game_id, "D")
    winner, transitions = _confirm(store, alternative.game_id, "E")

    games = store.games
    losing = games[original.game_id]
    winner = games[winner.game_id]
    assert winner.status == GameStatus.READY
    assert _active_customer_ids(winner) == {"B", "C", "D", "E"}
    assert losing.status in {GameStatus.FORMING, GameStatus.INVITING}
    assert _active_customer_ids(losing) == {"A"}
    assert losing.remaining_seats() == 3
    released_b = next(item for item in losing.participants if item.customer_id == "B")
    assert released_b.status == "superseded"
    assert any(
        item.entity_type == "game_participant"
        and item.entity_id == f"{original.game_id}:B"
        and item.to_status == "superseded"
        and item.reason == f"participant_committed_to_game:{alternative.game_id}"
        for item in transitions
    )
    with pytest.raises(ValueError, match="already committed"):
        _confirm(store, original.game_id, "B")


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_one_customer_can_be_provisional_in_many_options_then_kept_only_by_winner(kind: str, tmp_path) -> None:
    store = _store(kind, tmp_path, "many_options")
    start = now() + timedelta(days=1)
    options = [
        _create_two_person_option(
            store,
            conversation_id=f"conversation_{index}",
            organizer_id=f"organizer_{index}",
            shared_customer_id="B",
            requirement=_scheduled_requirement(start + timedelta(minutes=index * 5)),
        )
        for index in range(5)
    ]
    assert all("B" in _active_customer_ids(game) for game in options)

    winner = options[3]
    _confirm(store, winner.game_id, "D")
    _confirm(store, winner.game_id, "E")

    refreshed = store.games
    assert refreshed[winner.game_id].status == GameStatus.READY
    for option in options:
        current = refreshed[option.game_id]
        if option.game_id == winner.game_id:
            assert "B" in _active_customer_ids(current)
            continue
        assert "B" not in _active_customer_ids(current)
        assert next(item for item in current.participants if item.customer_id == "B").status == "superseded"
        assert current.remaining_seats() == 3


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_same_customer_can_commit_to_non_overlapping_ready_games(kind: str, tmp_path) -> None:
    store = _store(kind, tmp_path, "non_overlap")
    morning = now() + timedelta(days=1)
    evening = morning + timedelta(hours=8)
    first = _create_two_person_option(
        store,
        conversation_id="morning",
        organizer_id="A",
        shared_customer_id="B",
        requirement=_scheduled_requirement(morning, duration_hours=4),
    )
    second = _create_two_person_option(
        store,
        conversation_id="evening",
        organizer_id="C",
        shared_customer_id="B",
        requirement=_scheduled_requirement(evening, duration_hours=4),
    )
    for game, prefix in ((first, "morning"), (second, "evening")):
        _confirm(store, game.game_id, f"{prefix}_D")
        _confirm(store, game.game_id, f"{prefix}_E")

    refreshed = store.games
    assert refreshed[first.game_id].status == GameStatus.READY
    assert refreshed[second.game_id].status == GameStatus.READY
    assert "B" in _active_customer_ids(refreshed[first.game_id])
    assert "B" in _active_customer_ids(refreshed[second.game_id])


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_customer_search_keeps_provisional_candidate_but_excludes_time_committed_candidate(kind: str, tmp_path) -> None:
    store = _store(kind, tmp_path, "candidate_search")
    start = now() + timedelta(days=1)
    requirement = _scheduled_requirement(start)
    store.upsert_customer(
        CustomerProfile(
            customer_id="B",
            display_name="B",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["0.5"],
            smoke_preference="no_smoking",
            response_score=1.0,
        )
    )
    game = _create_two_person_option(
        store,
        conversation_id="candidate_source",
        organizer_id="A",
        shared_customer_id="B",
        requirement=requirement,
    )

    provisional = store.search_customers(requirement, exclude_customer_ids=["A"])
    b_candidate = next(item for item in provisional if item["customer"]["customer_id"] == "B")
    assert "provisional_in_1_overlapping_options" in b_candidate["reasons"]

    _confirm(store, game.game_id, "D")
    _confirm(store, game.game_id, "E")
    committed = store.search_customers(requirement, exclude_customer_ids=["A"])
    assert all(item["customer"]["customer_id"] != "B" for item in committed)

    non_overlapping = _scheduled_requirement(start + timedelta(hours=8))
    available_later = store.search_customers(non_overlapping, exclude_customer_ids=["A"])
    assert any(item["customer"]["customer_id"] == "B" for item in available_later)


def test_sqlite_concurrent_final_confirmations_choose_exactly_one_overlapping_winner(tmp_path) -> None:
    path = tmp_path / "concurrent_options.sqlite3"
    setup = SQLiteAgentStore(path)
    start = now() + timedelta(days=1)
    first = _create_two_person_option(
        setup,
        conversation_id="concurrent_a",
        organizer_id="A",
        shared_customer_id="B",
        requirement=_scheduled_requirement(start),
    )
    second = _create_two_person_option(
        setup,
        conversation_id="concurrent_c",
        organizer_id="C",
        shared_customer_id="B",
        requirement=_scheduled_requirement(start + timedelta(minutes=30)),
    )
    _confirm(setup, first.game_id, "D1")
    _confirm(setup, second.game_id, "D2")

    stores = [SQLiteAgentStore(path), SQLiteAgentStore(path)]
    barrier = threading.Barrier(2)

    def finish(index: int, game_id: str, customer_id: str) -> None:
        barrier.wait()
        _confirm(stores[index], game_id, customer_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(finish, 0, first.game_id, "E1"),
            executor.submit(finish, 1, second.game_id, "E2"),
        ]
        for future in futures:
            future.result(timeout=10)

    final_store = SQLiteAgentStore(path)
    games = [final_store.games[first.game_id], final_store.games[second.game_id]]
    winners = [game for game in games if game.status == GameStatus.READY]
    losers = [game for game in games if game.status != GameStatus.READY]
    assert len(winners) == 1
    assert len(losers) == 1
    assert "B" in _active_customer_ids(winners[0])
    assert "B" not in _active_customer_ids(losers[0])
    assert next(item for item in losers[0].participants if item.customer_id == "B").status == "superseded"
    assert losers[0].remaining_seats() == 1


def test_tool_result_reports_cross_game_commitment_back_to_agent() -> None:
    store = InMemoryAgentStore()
    start = now() + timedelta(days=1)
    losing = _create_two_person_option(
        store,
        conversation_id="conversation_a",
        organizer_id="A",
        shared_customer_id="B",
        requirement=_scheduled_requirement(start),
    )
    winner = _create_two_person_option(
        store,
        conversation_id="conversation_c",
        organizer_id="C",
        shared_customer_id="B",
        requirement=_scheduled_requirement(start + timedelta(minutes=30), duration_hours=5),
    )
    _confirm(store, winner.game_id, "D")
    store.create_invite_drafts(
        game_id=winner.game_id,
        invitations=[{"customer_id": "E", "display_name": "E", "message": "18:00打吗？"}],
        trace_id="trace_invite_e",
    )

    result = ToolGateway(store).execute(
        ToolCall(
            name="record_candidate_reply",
            arguments={
                "game_id": winner.game_id,
                "customer_id": "E",
                "display_name": "E",
                "status": "confirmed",
            },
            reason="E确认参加当前邀约的局",
        ),
        trace_id="trace_gateway_commit",
        conversation_id="conversation_e",
        sender_id="E",
        sender_name="E",
        step_index=1,
        source_message_id="message_e_confirmed",
    )

    assert result.called is True
    assert result.allowed is True
    commitment = result.result["cross_game_commitment"]
    assert commitment["winner_game_ids"] == [winner.game_id]
    assert commitment["affected_game_ids"] == sorted([losing.game_id, winner.game_id])
    assert commitment["released_participations"] == [
        {
            "customer_id": "B",
            "released_from_game_id": losing.game_id,
            "committed_to_game_id": winner.game_id,
        }
    ]
