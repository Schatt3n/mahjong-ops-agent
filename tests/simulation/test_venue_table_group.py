"""用真实盘面和持久回执复验牌局小群的准入、隔离和重复投递。"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import DEFAULT_TZ, GameStatus, using_clock

from .sim_adapter import build_runtime
from .venue_actors import ActorContext, generate_personas
from .venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel
from .venue_role_journal import VenueRoleJournal
from .venue_table_group import VenueTableGroup


_OPENED = datetime(2026, 11, 27, 18, 5, tzinfo=DEFAULT_TZ)
_ACTORS = generate_personas(42, count=6)
_HOST, _B, _C, _D, _COMPANION, _OUTSIDER = (actor.sender_id for actor in _ACTORS)
_ROSTER = {actor.sender_id for actor in _ACTORS}


def _scenario(tmp_path, *, anonymous=False, ready=True):
    """使用实际业务接口形成具名或带一位匿名同行的牌桌。"""
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    with using_clock(lambda: _OPENED - timedelta(minutes=5)):
        game, _ = store.create_game(
            conversation_id="organizer-inbox", organizer_id=_HOST, organizer_name=_ACTORS[0].name,
            known_players=[{"customer_id": _HOST, "seat_count": 2, "anonymous_seat_count": 1}] if anonymous else [],
            requirement={"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoking",
                         "start_time_kind": "scheduled", "planned_start_at": _OPENED.replace(hour=19, minute=0).isoformat()},
            trace_id="game-request",
        )
        others = (1, 2) if anonymous else (1, 2, 3)
        for index in others if ready else others[:-1]:
            actor = _ACTORS[index]
            game, _ = store.join_game(game_id=game.game_id, customer_id=actor.sender_id, display_name=actor.name,
                                       trace_id=f"confirmed-seat:{index}")
    assert (game.status == GameStatus.READY) is ready
    return store, game


def _table(store, tmp_path, game_id, **options):
    return VenueTableGroup(store, tmp_path / "table-chat.sqlite3", game_id,
                           **({"actor_ids": _ROSTER, "opened_at": _OPENED} | options))


def _message(table, sender, message_id, minutes, *, quote=None):
    return MessageEnvelope(message_id, sender, "我有点事，可能晚一会儿到。", _OPENED + timedelta(minutes=minutes),
                           "group", table.room_id, quote)


def test_ready_table_isolated_from_public_group_and_wakes_actual_recipients(tmp_path):
    """四位具名同桌和店家接收本群投递，主群成员和消息不会串入。"""
    store, game = _scenario(tmp_path)
    runtime, _ = build_runtime(store, "mock")
    public = SandboxChannel(runtime, room_id="venue-public",
        identities={actor.sender_id: actor.name for actor in _ACTORS}, archive_path=tmp_path / "main-chat.sqlite3")
    public.publish_owner(kind="group", recipient_id=public.room_id, text="群友晚上好。",
                         sent_at=_OPENED, message_id="public-opening")
    table = _table(store, tmp_path, game.game_id)
    assert table.member_ids() == {AGENT_ID, _HOST, _B, _C, _D}
    assert table.conversation_id not in {public.group_conversation_id, public.private_conversation_id(_HOST)}

    with pytest.raises(ValueError, match="sender is not admitted"):
        table.send(_message(table, _OUTSIDER, "outsider", 1))
    with pytest.raises(ValueError, match="quote is not visible"):
        table.send(_message(table, _HOST, "cross-quote", 1, quote="public-opening"))

    journal = VenueRoleJournal(tmp_path / VenueRoleJournal.filename)
    notice = journal.enqueue(actor_id=_HOST, kind="overtime_assigned", origin_id="shift-update",
        due_at=_OPENED + timedelta(minutes=15), details={"reason": "临时加班"})
    outbound = _message(table, _HOST, f"customer:{notice.event_id}", 15)
    delivered = table.send(outbound)
    assert delivered.recipient_ids == tuple(sorted({AGENT_ID, _B, _C, _D}))
    wakes = []
    for actor in delivered.recipient_ids:
        receipt = table.received(actor, outbound.message_id)
        assert receipt == delivered
        wake = journal.enqueue(actor_id=actor, kind="table_receipt", origin_id=outbound.message_id,
                               due_at=outbound.sent_at, caused_by=receipt.message.message_id)
        assert journal.enqueue(actor_id=actor, kind="table_receipt", origin_id=outbound.message_id,
                               due_at=outbound.sent_at, caused_by=receipt.message.message_id) == wake
        wakes.append(wake)
    assert {wake.actor_id for wake in wakes} == {AGENT_ID, _B, _C, _D}
    assert table.received(_HOST, outbound.message_id) is None
    assert table.received(_OUTSIDER, outbound.message_id) is None
    assert not table.events_for(_OUTSIDER)
    assert [row["message_id"] for row in public.events_for(_OUTSIDER)] == ["public-opening"]
    wake = next(wake for wake in wakes if wake.actor_id == _B)
    role_context = ActorContext(_ACTORS[1], wake.due_at.isoformat(), history=table.events_for(_B),
        events=[{"kind": "delivered", "caused_by": wake.caused_by}], group_targets=(table.room_id,))
    assert role_context.to_dict()["delivery"]["text"] == outbound.text
    assert set(store.games) == {game.game_id}
    assert store.require_game(game.game_id).to_dict() == game.to_dict()


def test_companion_joins_only_from_their_contact_invite_without_rewriting_old_receipts(tmp_path):
    """匿名席保持为业务席位，报名联系人和同行各完成一项真实准入动作。"""
    store, game = _scenario(tmp_path, anonymous=True)
    table = _table(store, tmp_path, game.game_id)
    before = _message(table, AGENT_ID, "original", 1)
    original = table.send(before)
    assert table.member_ids() == {AGENT_ID, _HOST, _B, _C}
    assert original.recipient_ids == tuple(sorted({_HOST, _B, _C}))
    invite_at = _OPENED + timedelta(minutes=2)
    accepted_at = invite_at + timedelta(seconds=30)
    with pytest.raises(ValueError, match="only the party contact"):
        table.invite_companion("wrong-contact", inviter_id=_B, recipient_id=_COMPANION, at=invite_at)
    with pytest.raises(ValueError, match="existing role"):
        table.invite_companion("invented", inviter_id=_HOST, recipient_id="fabricated-id", at=invite_at)
    invite = table.invite_companion("link", inviter_id=_HOST, recipient_id=_COMPANION, at=invite_at)
    assert invite.accepted is None
    assert table.invitation_for(_COMPANION, "link") == invite
    assert table.invitation_for(_HOST, "link") == invite
    assert table.invitation_for(AGENT_ID, "link") is None
    assert table.invitation_for(_OUTSIDER, "link") is None
    assert table.invite_companion("link", inviter_id=_HOST, recipient_id=_COMPANION, at=invite_at) == invite
    with pytest.raises(ValueError, match="conflicts"):
        table.invite_companion("link", inviter_id=_HOST, recipient_id=_OUTSIDER, at=invite_at)
    with pytest.raises(ValueError, match="anonymous seats"):
        table.invite_companion("another", inviter_id=_HOST, recipient_id=_OUTSIDER, at=invite_at)
    with pytest.raises(ValueError, match="from the invited role"):
        table.respond_companion("link", actor_id=_OUTSIDER, at=accepted_at, accept=True)
    with pytest.raises(ValueError, match="from the invited role"):
        table.respond_companion("link", actor_id=_COMPANION, at=_OPENED, accept=True)
    assert _COMPANION not in table.member_ids()
    assert not table.events_for(_COMPANION)
    with pytest.raises(ValueError, match="sender is not admitted"):
        table.send(_message(table, _COMPANION, "pending", 2))

    accepted = table.respond_companion("link", actor_id=_COMPANION, at=accepted_at, accept=True)
    assert accepted.accepted is True and accepted.response_at == accepted_at
    reopened = _table(SQLiteAgentStore(store.path), tmp_path, game.game_id)
    assert reopened.respond_companion("link", actor_id=_COMPANION, at=accepted_at, accept=True) == accepted
    assert reopened.member_ids() == {AGENT_ID, _HOST, _B, _C, _COMPANION}
    assert reopened.member_ids(at=invite_at) == {AGENT_ID, _HOST, _B, _C}
    with pytest.raises(ValueError, match="response conflicts"):
        reopened.respond_companion("link", actor_id=_COMPANION, at=accepted_at, accept=False)
    assert reopened.send(before) == original
    assert reopened.received(_COMPANION, before.message_id) is None
    assert not reopened.events_for(_COMPANION)
    with pytest.raises(ValueError, match="quote is not visible"):
        reopened.send(_message(reopened, _COMPANION, "hidden-quote", 3, quote=before.message_id))

    after = _message(reopened, _HOST, "after-join", 3, quote=before.message_id)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(reopened.send, (after, after))
    assert first == second and _COMPANION in first.recipient_ids
    assert reopened.delivered_receipts() == (original, first)
    seen, = reopened.events_for(_COMPANION)
    assert seen["message_id"] == after.message_id and seen["quoted_message_id"] is None
    assert reopened.received(_COMPANION, after.message_id).message.quoted_message_id is None
    assert reopened.received(_B, after.message_id).message.quoted_message_id == before.message_id
    assert not reopened.events_for(_OUTSIDER)
    with pytest.raises(ValueError, match="ID conflicts"):
        reopened.send(replace(after, text="改用重投覆盖正文。"))
    assert store.require_game(game.game_id).seat_summary()["claimed_seats"] == 4
    assert store.require_game(game.game_id).seat_summary()["anonymous_seat_count"] == 1


def test_companion_decline_and_stale_reentry_never_admit_or_write_seats(tmp_path):
    """拒绝可让出邀请名额，原局退出后旧邀请失去权限。"""
    store, game = _scenario(tmp_path, anonymous=True)
    table = _table(store, tmp_path, game.game_id)
    at = _OPENED + timedelta(minutes=1)
    table.invite_companion("decline", inviter_id=_HOST, recipient_id=_COMPANION, at=at)
    declined = table.respond_companion("decline", actor_id=_COMPANION, at=at, accept=False)
    assert declined.accepted is False and _COMPANION not in table.member_ids()
    table.invite_companion("stale", inviter_id=_HOST, recipient_id=_OUTSIDER, at=at)
    at += timedelta(minutes=1)
    with using_clock(lambda: at):
        store.update_game_status(game_id=game.game_id, status="cancelled", reason="本人不来了", trace_id="cancelled")
    with pytest.raises(ValueError, match="four current confirmed seats"):
        table.respond_companion("stale", actor_id=_OUTSIDER, at=at, accept=True)
    assert table.invitation_for(_OUTSIDER, "stale").accepted is None
    assert _OUTSIDER not in table.member_ids()
    assert not table.events_for(_OUTSIDER)


def test_group_cannot_be_created_from_unconfirmed_or_unregistered_seats(tmp_path):
    store, incomplete = _scenario(tmp_path, ready=False)
    with pytest.raises(ValueError, match="four current confirmed seats"):
        _table(store, tmp_path, incomplete.game_id)
    with using_clock(lambda: _OPENED):
        ready, _ = store.join_game(game_id=incomplete.game_id, customer_id=_D, display_name=_ACTORS[3].name,
                                   trace_id="last-seat")
    with pytest.raises(ValueError, match="unregistered named seats"):
        _table(store, tmp_path, ready.game_id, actor_ids=_ROSTER - {_D})
    table = _table(store, tmp_path, ready.game_id)
    with pytest.raises(ValueError, match="saved room"):
        _table(store, tmp_path, ready.game_id, actor_ids=_ROSTER - {_OUTSIDER})
    assert table.member_ids() == {AGENT_ID, _HOST, _B, _C, _D}
