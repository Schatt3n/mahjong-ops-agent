"""补位、并行备选及更新合同的双Store可信执行回归。"""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore, ToolGateway
from mahjong_agent_runtime.models import ToolCall, now
from mahjong_agent_runtime.domains.context_builders.game_context import build_game_context


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    result = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "recruitment.sqlite3")
    yield result
    if request.param == "sqlite":
        result._connection.close()


def create(store, members="ABC", start=None, **extra):
    start = start or now() + timedelta(hours=1)
    return store.create_game(
        conversation_id="private_a", organizer_id="A", organizer_name="A",
        requirement={"game_type": "hangzhou_mahjong", "game_variant": "caiqiao", "stake": "0.5",
                     "smoke_preference": "no_smoking", "planned_start_at": start.isoformat(), "duration_hours": 4},
        known_players=[{"customer_id": actor, "display_name": actor} for actor in members],
        trace_id="create", **extra,
    )[0]


def execute(store, name, arguments, sender="A", conversation="private_a", message="m1"):
    return ToolGateway(store).execute(ToolCall(name, arguments), trace_id="trace_" + message,
        conversation_id=conversation, sender_id=sender, sender_name=sender, step_index=1, source_message_id=message)


def active_members(game):
    return {p.customer_id for p in game.participants if p.status in {"joined", "confirmed"}}


def test_reported_withdrawal_keeps_original_game_and_is_idempotent(store):
    game = create(store, "ABCD")
    store.configure_rooms(["room"])
    reservation, _ = store.reserve_room(conversation_id="private_a", game_id=game.game_id,
        start_at=game.planned_start_at, end_at=game.planned_end_at, trace_id="reserve", room_id="room")
    reserved_before = deepcopy(reservation.to_dict())
    store.record_candidate_reply(game_id=game.game_id, customer_id="A", display_name="A", status="confirmed", trace_id="confirmed")
    original = store.require_game(game.game_id)
    assert original.status == "ready"
    args = {"game_id": game.game_id, "customer_id": "D", "reason": "报名联系人报告同行临时有事"}
    result = execute(store, "record_registered_member_withdrawal", args)
    assert result.allowed and result.called and not result.error
    repeated = execute(store, "record_registered_member_withdrawal", args)
    assert repeated.deduplicated
    state = store.require_game(game.game_id)
    assert len(store.games) == 1 and state.status in {"forming", "inviting"}
    assert active_members(state) == set("ABC") and state.remaining_seats() == 1
    assert state.planned_start_at == original.planned_start_at
    assert state.created_at == original.created_at
    assert any(t.reason == "organizer_reported_withdrawal:A" for t in result.state_transitions)
    assert not any(t.to_status == "cancelled" for t in store.transitions)
    assert store.room_reservations[reservation.reservation_id].to_dict() == reserved_before


@pytest.mark.parametrize("violation", ["wrong_sender", "wrong_conversation", "self_confirmed", "outsider"])
def test_reporting_contact_cannot_override_independent_or_unrelated_customer(store, violation):
    game = create(store, "ABCD")
    target = "D"
    if violation == "self_confirmed":
        store.record_candidate_reply(game_id=game.game_id, customer_id="D", display_name="D", status="confirmed", trace_id="d_self")
    if violation == "outsider":
        target = "E"
    before = deepcopy(store.require_game(game.game_id).to_dict())
    result = execute(store, "record_registered_member_withdrawal", {"game_id": game.game_id, "customer_id": target, "reason": "测试拒绝"},
        sender="B" if violation == "wrong_sender" else "A", conversation="other" if violation == "wrong_conversation" else "private_a")
    assert not result.allowed and not result.called and result.error
    assert store.require_game(game.game_id).to_dict() == before


def test_cannot_impersonate_member_or_cancel_whole_game_from_other_identity(store):
    game = create(store)
    result = execute(store, "record_candidate_reply", {"game_id": game.game_id, "customer_id": "B", "display_name": "B", "status": "declined"})
    assert not result.allowed and "subject mismatch" in result.error
    result = execute(store, "update_game_status", {"game_id": game.game_id, "status": "cancelled", "reason": "越权"}, sender="B", message="m2")
    assert not result.allowed and store.require_game(game.game_id).status == "forming"


def test_reported_withdrawal_cannot_bypass_unresolved_quote_or_store_authority(store):
    game = create(store, "ABCD")
    before = deepcopy(store.require_game(game.game_id).to_dict())
    result = ToolGateway(store).execute(ToolCall("record_registered_member_withdrawal",
        {"game_id": game.game_id, "customer_id": "D", "reason": "转述"}),
        trace_id="quote", conversation_id="private_a", sender_id="A", sender_name="A", step_index=1,
        message_reference_contract={"quoted_message_present": True, "business_reference_resolved": False})
    assert not result.allowed and "quoted-message" in result.error
    with pytest.raises(ValueError, match="registration contact"):
        store.record_candidate_reply(game_id=game.game_id, customer_id="D", display_name="D", status="declined",
            trace_id="direct_invalid", reported_by=("B", "private_a"))
    assert store.require_game(game.game_id).to_dict() == before


def test_linked_options_keep_stable_people_and_commit_only_one_side(store):
    early = create(store)
    late = create(store, "ABE", start=early.planned_start_at + timedelta(minutes=30), alternative_to_game_id=early.game_id)
    assert late.conversation_id == early.conversation_id and late.option_root_game_id == early.game_id
    assert active_members(late) == set("ABE")
    store.join_game(game_id=late.game_id, customer_id="F", display_name="F", trace_id="f_join")
    assert active_members(store.require_game(late.game_id)) == set("ABEF")
    assert store.require_game(late.game_id).status == "ready"
    assert active_members(store.require_game(early.game_id)) == {"C"}
    with pytest.raises(ValueError, match="active game already exists|source must"):
        create(store, start=early.planned_start_at + timedelta(minutes=45), alternative_to_game_id=early.game_id)


def test_option_relation_is_not_a_general_duplicate_or_auth_bypass(store):
    early = create(store)
    with pytest.raises(ValueError, match="active game already exists"):
        create(store, start=early.planned_start_at + timedelta(minutes=30))
    with pytest.raises(ValueError, match="active game already exists"):
        create(store, start=early.planned_start_at, alternative_to_game_id=early.game_id)
    with pytest.raises(ValueError, match="source game not found"):
        create(store, alternative_to_game_id="missing")
    args = {"organizer_id": "B", "organizer_name": "B", "requirement": {"start_time_kind": "asap_when_full"}, "alternative_to_game_id": early.game_id}
    denied = execute(store, "create_game", args, sender="B")
    assert not denied.allowed and len(store.games) == 1


@pytest.mark.parametrize("field", ["planned_end_at", "known_player_count", "needed_seats", "seat_format", "unknown_business_field"])
def test_patch_schema_rejects_derived_and_unknown_fields_without_mutation(store, field):
    game = create(store)
    before = deepcopy(store.require_game(game.game_id).to_dict())
    result = execute(store, "update_game_requirement", {"game_id": game.game_id, "requirement_patch": {field: "bad"}, "reason": "测试拒绝"})
    assert not result.called and not result.allowed and "unexpected argument" in result.error
    assert store.require_game(game.game_id).to_dict() == before


def test_time_patch_derives_end_and_preserves_seats_and_context_authority(store):
    game = create(store)
    start = game.planned_start_at + timedelta(minutes=30)
    result = execute(store, "update_game_requirement", {"game_id": game.game_id, "requirement_patch": {"planned_start_at": start.isoformat()}, "reason": "已同意改时"})
    assert result.allowed and not result.error
    updated = store.require_game(game.game_id)
    assert updated.planned_start_at == start and updated.planned_end_at == start + timedelta(hours=4)
    assert active_members(updated) == set("ABC")
    context = build_game_context(store, conversation_id="private_a", sender_id="A")
    assert {p["registered_by_customer_id"] for p in context.model_contexts[0]["participants"]} == {"A"}


def test_sqlite_reopen_retains_option_relation_and_contact_permissions(tmp_path):
    path = tmp_path / "restart.sqlite3"
    store = SQLiteAgentStore(path)
    early = create(store)
    late = create(store, "ABE", start=early.planned_start_at + timedelta(minutes=30), alternative_to_game_id=early.game_id)
    store._connection.close()
    store = SQLiteAgentStore(path)
    try:
        assert store.require_game(late.game_id).option_root_game_id == early.game_id
        assert next(p for p in store.require_game(late.game_id).participants if p.customer_id == "E").registered_by_customer_id == "A"
    finally:
        store._connection.close()


def test_sqlite_parallel_duplicate_option_creation_is_atomic(tmp_path):
    path = tmp_path / "concurrent.sqlite3"
    first = SQLiteAgentStore(path)
    early = create(first)
    second = SQLiteAgentStore(path)
    def attempt(store):
        try:
            return create(store, "ABE", start=early.planned_start_at + timedelta(minutes=30), alternative_to_game_id=early.game_id).game_id
        except ValueError:
            return None
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, [first, second]))
        assert sum(result is not None for result in results) == 1 and len(first.games) == 2
    finally:
        first._connection.close()
        second._connection.close()
