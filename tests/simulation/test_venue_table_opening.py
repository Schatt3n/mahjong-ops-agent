"""用实际桌群收件和正式 SQLite 盘面核对人齐开后的时间协商。"""

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import DEFAULT_TZ, GameStatus, using_clock

from .venue_channel import AGENT_ID, MessageEnvelope
from .venue_table_group import VenueTableGroup
from .venue_table_opening import VenueTableOpening


_OPEN = datetime(2026, 11, 27, 18, 5, tzinfo=DEFAULT_TZ)
_MEMBERS = ("a", "b", "c", "d")
_EARLY = _OPEN.replace(hour=19, minute=0)
_LATE = _EARLY + timedelta(minutes=30)


def _table(tmp_path):
    """经正式建桌和报名形成一张尚未定时的人齐开局。"""
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    with using_clock(lambda: _OPEN - timedelta(minutes=3)):
        store.configure_rooms(["room"])
        game, _ = store.create_game(
            conversation_id="owner-dm-a", organizer_id="a", organizer_name="A", known_players=[],
            requirement={"game_type": "hangzhou_mahjong", "stake": 1, "smoke_preference": "no_smoking",
                         "start_time_kind": "asap_when_full", "duration_hours": 4},
            trace_id="create-asap",
        )
        for actor in _MEMBERS[1:]:
            store.join_game(game_id=game.game_id, customer_id=actor, display_name=actor, trace_id=f"join:{actor}")
    group = VenueTableGroup(store, tmp_path / "table.sqlite3", game.game_id,
                           actor_ids=(*_MEMBERS, "outsider"), opened_at=_OPEN)
    return store, game.game_id, group, VenueTableOpening(group)


def _propose(opening, group, message_id, minute, start):
    message = opening.proposal_message(message_id, at=_OPEN + timedelta(minutes=minute), start_at=start)
    receipt = group.send(message)
    opening.record_proposal(receipt, start_at=start)
    return receipt


def _vote(group, actor, message_id, minute, text, quote=None):
    return group.send(MessageEnvelope(message_id, actor, text, _OPEN + timedelta(minutes=minute),
                                      "group", group.room_id, quote))


def test_four_delivered_answers_set_the_same_opening_and_preserve_the_four_hour_room_loop(tmp_path):
    """三人表态或有人拒绝不会定时，另提议后收齐四方意见才改变同一牌局。"""
    store, game_id, group, opening = _table(tmp_path)
    assert group.member_ids() == {AGENT_ID, *_MEMBERS}
    assert store.require_game(game_id).status == GameStatus.READY
    assert store.require_game(game_id).planned_start_at is None
    initial = _propose(opening, group, "first", 1, _EARLY)
    assert initial.recipient_ids == _MEMBERS
    for index, actor in enumerate(("a", "b", "c"), 2):
        receipt = _vote(group, actor, f"early-{actor}", index, "七点我没问题。", "first")
        assert opening.record_vote(receipt, proposal_id="first", accepts=True) == "waiting"
    late = _vote(group, "d", "reject-early", 5, "我最早七点半，到晚一点大家等得及吗？", "first")
    assert opening.record_vote(late, proposal_id="first", accepts=False) == "declined"
    assert store.require_game(game_id).planned_start_at is None

    _propose(opening, group, "second", 6, _LATE)
    old_answer = _vote(group, "b", "stale-first", 7, "之前七点也能到。", "first")
    assert opening.record_vote(old_answer, proposal_id="first", accepts=True) == "stale_proposal"
    for index, actor in enumerate(_MEMBERS, 8):
        receipt = _vote(group, actor, f"late-{actor}", index, "七点半行。", "second")
        expected = "committed" if actor == "d" else "waiting"
        assert opening.record_vote(receipt, proposal_id="second", accepts=True) == expected

    reopened = SQLiteAgentStore(store.path)
    room = VenueTableGroup(reopened, group.archive_path, game_id,
                          actor_ids=(*_MEMBERS, "outsider"), opened_at=_OPEN)
    resumed = VenueTableOpening(room)
    assert resumed.current() == {"proposal_id": "second", "start_at": _LATE.isoformat(),
                                 "accepted_count": 4, "committed": True}
    last = room.received(AGENT_ID, "late-d")
    assert resumed.record_vote(last, proposal_id="second", accepts=True) == "committed"
    game = reopened.require_game(game_id)
    assert game.planned_start_at == _LATE and game.planned_end_at == _LATE + timedelta(hours=4)
    assert game.status == GameStatus.READY and game.seat_summary()["claimed_seats"] == 4
    assert not game.actual_started_at and not reopened.room_reservations
    assert sum(row.reason == "venue_table_opening_confirmed" for row in reopened.transitions) == 1
    assert not room.events_for("outsider")

    with pytest.raises(ValueError, match="outside its start window"):
        reopened.start_game_play(game_id, attendee_ids=_MEMBERS, room_id="room", at=_EARLY, trace_id="too-early")
    playing, _ = reopened.start_game_play(
        game_id, attendee_ids=_MEMBERS, room_id="room", at=_LATE, trace_id="actual-start")
    reservation = reopened.room_reservations[playing.play_reservation_id]
    assert playing.is_playing and reservation.status == "occupied"
    with pytest.raises(ValueError, match="duration elapsed"):
        reopened.finish_game_play(game_id, at=_LATE + timedelta(hours=3), trace_id="too-early-finish")
    end = _LATE + timedelta(hours=4)
    ended, _ = reopened.finish_game_play(game_id, at=end, trace_id="physical-finish")
    assert ended.actual_ended_at == end and reopened.room_reservations[reservation.reservation_id].status == "released"


def test_unseen_or_inaccurate_proposal_cannot_supply_opening_authority(tmp_path):
    """拒绝伪送达、片面重解释、迟到票和顺序被破坏的历史重放。"""
    store, game_id, group, opening = _table(tmp_path)
    expected = opening.proposal_message("not-delivered", at=_OPEN, start_at=_EARLY)
    delivered = _propose(opening, group, "real", 1, _EARLY)
    with pytest.raises(ValueError, match="exact delivered"):
        opening.record_proposal(replace(delivered, message=expected), start_at=_EARLY)
    with pytest.raises(ValueError, match="exact delivered"):
        opening.record_proposal(delivered, start_at=_LATE)
    ballot = _vote(group, "a", "visible-vote", 3, "七点没问题。", "real")
    with pytest.raises(ValueError, match="completed delivery"):
        opening.record_vote(replace(ballot, recipient_ids=()), proposal_id="real", accepts=True)
    with pytest.raises(ValueError, match="visible proposal"):
        opening.record_vote(ballot, proposal_id="not-delivered", accepts=True)
    assert opening.record_vote(ballot, proposal_id="real", accepts=True) == "waiting"
    with pytest.raises(ValueError, match="changed its meaning"):
        opening.record_vote(ballot, proposal_id="real", accepts=False)
    assert opening.record_vote(_vote(group, "b", "out-of-order", 2, "行。", "real"),
                               proposal_id="real", accepts=True) == "stale_vote"
    assert opening.record_vote(_vote(group, "c", "late-reply", 55, "行。", "real"),
                               proposal_id="real", accepts=True) == "proposal_elapsed"
    assert store.require_game(game_id).planned_start_at is None


def test_same_roster_after_withdrawal_requires_new_proposal(tmp_path):
    """有人退席后又回来，同样四个名字也不复用上一版同意。"""
    store, game_id, group, opening = _table(tmp_path)
    _propose(opening, group, "old", 1, _EARLY)
    assert opening.record_vote(_vote(group, "a", "first-a", 2, "七点行。", "old"),
                               proposal_id="old", accepts=True) == "waiting"
    with using_clock(lambda: _OPEN + timedelta(minutes=3)):
        store.record_candidate_reply(game_id=game_id, customer_id="d", display_name="D", status="declined",
                                     seat_count=1, trace_id="exit")
    with using_clock(lambda: _OPEN + timedelta(minutes=4)):
        store.join_game(game_id=game_id, customer_id="d", display_name="D", trace_id="return")
    assert opening.record_vote(_vote(group, "d", "old-d", 5, "行。", "old"),
                               proposal_id="old", accepts=True) == "seats_changed"
    assert store.require_game(game_id).planned_start_at is None
