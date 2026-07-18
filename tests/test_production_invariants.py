from __future__ import annotations

import pytest
from concurrent.futures import ThreadPoolExecutor

from mahjong_agent_runtime import (
    AgentContextBuilder,
    InMemoryAgentStore,
    SQLiteAgentStore,
    ToolGateway,
    ToolCall,
    ToolResult,
    UserMessage,
)
from mahjong_agent_runtime.coordination import FileCoordinationManager


def _create_game(store, *, conversation_id: str = "conversation_a", organizer_id: str = "customer_a", seats: int = 1):
    return store.create_game(
        conversation_id=conversation_id,
        organizer_id=organizer_id,
        organizer_name=organizer_id,
        requirement={"game_type": "hangzhou_mahjong", "stake": "0.5", "known_player_count": seats},
        known_players=[
            {
                "customer_id": organizer_id,
                "display_name": organizer_id,
                "seat_count": seats,
            }
        ],
        trace_id=f"trace_{conversation_id}",
    )


def test_context_does_not_preload_other_conversation_private_state() -> None:
    store = InMemoryAgentStore()
    game, _ = _create_game(store)
    store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[{"customer_id": "candidate_a", "display_name": "候选人A", "message_text": "七点打吗？"}],
        trace_id="trace_invite_a",
    )
    store.create_outbound_message_drafts(
        conversation_id="conversation_a",
        drafts=[
            {
                "recipient_id": "candidate_a",
                "recipient_name": "候选人A",
                "channel": "wechaty",
                "message_text": "七点打吗？",
                "purpose": "invite",
            }
        ],
        trace_id="trace_outbound_a",
    )

    built = AgentContextBuilder(store, ToolGateway(store)).build(
        UserMessage(
            conversation_id="conversation_b",
            sender_id="customer_b",
            sender_name="客户B",
            text="现在有人吗",
            message_id="message_b",
        ),
        trace_id="trace_context_b",
    )

    assert built.payload["active_games"] == []
    assert built.payload["active_parties"] == []
    assert built.payload["outbound_message_drafts"] == []
    assert store.search_current_games({"game_type": "hangzhou_mahjong", "stake": "0.5"})


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_confirmed_party_cannot_overfill_table(kind: str, tmp_path) -> None:
    store = InMemoryAgentStore() if kind == "memory" else SQLiteAgentStore(tmp_path / "capacity.sqlite3")
    game, _ = _create_game(store, seats=3)

    with pytest.raises(ValueError, match="seat capacity exceeded"):
        store.record_candidate_reply(
            game_id=game.game_id,
            customer_id="customer_b",
            display_name="客户B",
            status="confirmed",
            seat_count=2,
            trace_id="trace_overfill",
        )

    persisted = store.require_game(game.game_id)
    assert persisted.remaining_seats() == 1
    assert all(item.customer_id != "customer_b" for item in persisted.participants)


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_duplicate_active_game_for_same_requester_is_rejected(kind: str, tmp_path) -> None:
    store = InMemoryAgentStore() if kind == "memory" else SQLiteAgentStore(tmp_path / "duplicate.sqlite3")
    _create_game(store)

    with pytest.raises(ValueError, match="active game already exists"):
        _create_game(store)


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_expired_game_rejects_new_invitation_drafts(kind: str, tmp_path) -> None:
    import datetime as dt

    store = InMemoryAgentStore() if kind == "memory" else SQLiteAgentStore(tmp_path / "expired_invite.sqlite3")
    expired_start = dt.datetime.now().astimezone() - dt.timedelta(hours=8)
    game, _ = store.create_game(
        conversation_id="expired_conversation",
        organizer_id="customer_a",
        organizer_name="customer_a",
        requirement={
            "game_type": "hangzhou_mahjong",
            "start_time_kind": "scheduled",
            "start_at": expired_start.isoformat(),
            "duration_hours": 4,
        },
        known_players=[{"customer_id": "customer_a", "display_name": "customer_a"}],
        trace_id="trace_expired_game",
    )

    with pytest.raises(ValueError, match="does not accept invitations"):
        store.create_invite_drafts(
            game_id=game.game_id,
            invitations=[{"customer_id": "customer_b", "display_name": "customer_b", "message_text": "打吗？"}],
            trace_id="trace_expired_invite",
        )

    assert store.require_game(game.game_id).status.value == "cancelled"


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_same_customer_cannot_have_two_open_invitations_for_one_game(kind: str, tmp_path) -> None:
    store = InMemoryAgentStore() if kind == "memory" else SQLiteAgentStore(tmp_path / "duplicate_invite.sqlite3")
    game, _ = _create_game(store)
    invitation = {"customer_id": "customer_b", "display_name": "客户B", "message_text": "七点打吗？"}
    store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[invitation],
        trace_id="trace_first_invite",
    )

    with pytest.raises(ValueError, match="already has an open invitation"):
        store.create_invite_drafts(
            game_id=game.game_id,
            invitations=[invitation],
            trace_id="trace_duplicate_invite",
        )

    drafts = [item for item in store.invite_drafts.values() if item.game_id == game.game_id]
    assert len(drafts) == 1


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_closing_game_releases_room_inventory(kind: str, tmp_path) -> None:
    import datetime as dt

    store = InMemoryAgentStore() if kind == "memory" else SQLiteAgentStore(tmp_path / "release_room.sqlite3")
    store.configure_rooms(["room_a"])
    game, _ = _create_game(store)
    start_at = dt.datetime.now().astimezone() + dt.timedelta(hours=1)
    end_at = start_at + dt.timedelta(hours=4)
    reservation, _ = store.reserve_room(
        conversation_id=game.conversation_id,
        game_id=game.game_id,
        start_at=start_at,
        end_at=end_at,
        room_id="room_a",
        trace_id="trace_reserve_room",
    )

    store.update_game_status(
        game_id=game.game_id,
        status="cancelled",
        reason="requester_cancelled",
        trace_id="trace_cancel_game",
    )

    availability = store.search_room_availability(start_at=start_at, end_at=end_at)
    assert availability["available_room_ids"] == ["room_a"]
    assert store.room_reservations[reservation.reservation_id].status == "released"


def test_tool_gateway_binds_write_subject_to_authenticated_sender() -> None:
    store = InMemoryAgentStore()
    gateway = ToolGateway(store)

    result = gateway.execute(
        ToolCall(
            name="create_game",
            arguments={
                "organizer_id": "another_customer",
                "organizer_name": "其他人",
                "requirement": {"game_type": "hangzhou_mahjong"},
                "known_players": [{"customer_id": "another_customer", "display_name": "其他人"}],
            },
            reason="attempt cross-customer write",
        ),
        trace_id="trace_subject_auth",
        conversation_id="conversation_a",
        sender_id="customer_a",
        sender_name="客户A",
        step_index=1,
        source_message_id="message_a",
    )

    assert result.called is False
    assert result.allowed is False
    assert "subject mismatch" in str(result.error)
    assert store.games == {}


def test_sqlite_expired_input_processing_lease_is_recoverable(tmp_path) -> None:
    import datetime as dt

    path = tmp_path / "recover_input.sqlite3"
    store = SQLiteAgentStore(path)
    batch, _, _ = store.upsert_pending_input_fragment(
        UserMessage(
            conversation_id="recover_conversation",
            sender_id="recover_sender",
            sender_name="恢复用户",
            text="帮我组个局",
            message_id="recover_message",
        ),
        trace_id="trace_buffer",
        quiet_deadline=dt.datetime.now().astimezone() - dt.timedelta(seconds=1),
    )
    claimed, _ = store.claim_pending_input_batch(
        batch_id=batch.batch_id,
        expected_version=batch.version,
        trace_id="trace_first_worker",
    )
    assert claimed is not None
    stale_at = dt.datetime.now().astimezone() - dt.timedelta(minutes=5)
    claimed.updated_at = stale_at
    with store._lock, store._connection:
        store._save_pending_input_batch(claimed)

    restarted = SQLiteAgentStore(path)
    due = restarted.due_pending_input_batches(at=dt.datetime.now().astimezone())
    assert [item.batch_id for item in due] == [batch.batch_id]
    reclaimed, _ = restarted.claim_pending_input_batch(
        batch_id=batch.batch_id,
        expected_version=batch.version,
        trace_id="trace_recovery_worker",
    )
    assert reclaimed is not None


def test_sqlite_expired_tool_claim_can_be_reclaimed(tmp_path) -> None:
    path = tmp_path / "recover_tool.sqlite3"
    key = "tool:recover"
    store = SQLiteAgentStore(path)
    placeholder = ToolResult(
        name="create_game",
        called=False,
        allowed=True,
        result={"idempotency_status": "claimed"},
        error="in progress",
        idempotency_key=key,
    )
    acquired, _ = store.claim_idempotent_result(key, placeholder)
    assert acquired is True
    with store._lock, store._connection:
        store._connection.execute(
            "UPDATE runtime_idempotency_ledger SET created_at = ? WHERE idempotency_key = ?",
            ("2000-01-01T00:00:00+08:00", key),
        )

    restarted = SQLiteAgentStore(path)
    assert restarted.idempotent_result(key) is None
    reacquired, existing = restarted.claim_idempotent_result(key, placeholder)
    assert reacquired is True
    assert existing is None


def test_sqlite_conversation_version_increment_is_atomic_across_connections(tmp_path) -> None:
    path = tmp_path / "versions.sqlite3"
    stores = [SQLiteAgentStore(path), SQLiteAgentStore(path)]

    def advance(index: int) -> int:
        version, _ = stores[index % 2].advance_conversation_version(
            "same_conversation",
            trace_id=f"trace_version_{index}",
            reason="concurrent_test",
        )
        return version

    with ThreadPoolExecutor(max_workers=8) as pool:
        versions = list(pool.map(advance, range(40)))

    assert sorted(versions) == list(range(1, 41))
    assert SQLiteAgentStore(path).conversation_version("same_conversation") == 40


def test_file_coordination_serializes_same_scope(tmp_path) -> None:
    managers = [FileCoordinationManager(tmp_path / "locks"), FileCoordinationManager(tmp_path / "locks")]
    state = {"inside": 0, "max_inside": 0}

    def critical(index: int) -> None:
        with managers[index % 2].lock("conversation:same"):
            state["inside"] += 1
            state["max_inside"] = max(state["max_inside"], state["inside"])
            import time

            time.sleep(0.005)
            state["inside"] -= 1

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(critical, range(20)))

    assert state["max_inside"] == 1


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_room_inventory_prevents_overlapping_reservations(store_kind: str, tmp_path) -> None:
    store = InMemoryAgentStore() if store_kind == "memory" else SQLiteAgentStore(tmp_path / "rooms.sqlite3")
    store.configure_rooms(["room_1"])
    first, _ = store.reserve_room(
        conversation_id="conversation_a",
        game_id=None,
        start_at="2026-07-18T14:00:00+08:00",
        end_at="2026-07-18T18:00:00+08:00",
        room_id=None,
        trace_id="trace_room_a",
    )

    availability = store.search_room_availability(
        start_at="2026-07-18T15:00:00+08:00",
        end_at="2026-07-18T17:00:00+08:00",
    )

    assert first.room_id == "room_1"
    assert availability["available_count"] == 0
    with pytest.raises(ValueError, match="no room|unavailable"):
        store.reserve_room(
            conversation_id="conversation_b",
            game_id=None,
            start_at="2026-07-18T15:00:00+08:00",
            end_at="2026-07-18T17:00:00+08:00",
            room_id=None,
            trace_id="trace_room_b",
        )
