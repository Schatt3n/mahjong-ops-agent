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


def _table(tmp_path, *, anonymous=False):
    """经正式建桌和报名形成一张尚未定时的人齐开局。"""
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    with using_clock(lambda: _OPEN - timedelta(minutes=3)):
        store.configure_rooms(["room"])
        game, _ = store.create_game(
            conversation_id="owner-dm-a", organizer_id="a", organizer_name="A",
            known_players=[{"customer_id": "a", "seat_count": 2, "anonymous_seat_count": 1}] if anonymous else [],
            requirement={"game_type": "hangzhou_mahjong", "stake": 1, "smoke_preference": "no_smoking",
                         "start_time_kind": "asap_when_full", "duration_hours": 4},
            trace_id="create-asap",
        )
        for actor in _MEMBERS[1:3] if anonymous else _MEMBERS[1:]:
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


def test_shop_handler_only_commits_via_its_actually_delivered_proposals(tmp_path):
    """店家处理器只看桌群实际消息，拒绝旧回应并在第四票后才通告。"""
    import json

    from .venue_table_shop import VenueTableShop

    store, game_id, group, opening = _table(tmp_path)
    trace = tmp_path / "table_shop.jsonl"
    outcomes = {
        "suggest": ("other", _EARLY.isoformat()),
        "cant-at-seven": ("decline", _LATE.isoformat()),
        "late-old-agreement": ("accept", None),
        **{f"yes-{actor}": ("accept", None) for actor in _MEMBERS},
    }

    class Reviewer:
        """受控结构输出验证送达和落库路径，真实发言使用独立运行。"""

        def complete(self, messages, *, trace_id, timeout_seconds):
            """只用这一条实际来话命中受控对照。"""
            payload = json.loads(messages[-1]["content"])
            source = payload["current"]["message_id"]
            assert payload["current"]["text"] == next(
                row["text"] for row in group.events_for(AGENT_ID) if row["message_id"] == source)
            vote, start = outcomes[source]
            return json.dumps({"vote": vote, "proposed_start": start})

    shop = VenueTableShop(Reviewer(), trace)
    inquiry = group.send(shop(group, None, _OPEN))
    assert "各自几点能到" in inquiry.message.text

    def react(receipt):
        return shop(group, receipt, receipt.message.sent_at.astimezone(DEFAULT_TZ))

    suggestion = _vote(group, "a", "suggest", 1, "七点好像合适。", inquiry.message.message_id)
    unattached = react(suggestion)
    assert opening.current() is None
    first = group.send(unattached)
    deny = _vote(group, "d", "cant-at-seven", 2, "七点不行，我最早七点半。", first.message.message_id)
    second = group.send(react(deny))
    stale = _vote(group, "a", "late-old-agreement", 3, "我接受原来七点。", first.message.message_id)
    assert react(stale) is None
    assert opening.current()["accepted_count"] == 0
    for index, actor in enumerate(_MEMBERS, 4):
        receipt = _vote(group, actor, f"yes-{actor}", index, "七点半可以。", second.message.message_id)
        response = react(receipt)
        if actor != "d":
            assert response is None and store.require_game(game_id).planned_start_at is None
    assert store.require_game(game_id).planned_start_at == _LATE
    assert response.text == "那就 19:30 开，大家按这个时间到哈。"
    confirmation = group.send(response)
    assert confirmation.recipient_ids == _MEMBERS
    assert react(receipt) is None
    assert len(trace.read_text().splitlines()) == 7


def test_unquoted_message_sent_before_proposal_in_same_clock_tick_cannot_vote(tmp_path):
    """相同虚拟时刻也按实际送达顺序区分先前话题与本次同意。"""
    store, game_id, group, opening = _table(tmp_path)
    early_message = _vote(group, "a", "previous-discussion", 1, "可以。")
    _propose(opening, group, "not-visible-before-message", 1, _EARLY)
    with pytest.raises(ValueError, match="visible proposal"):
        opening.record_vote(early_message, proposal_id="not-visible-before-message", accepts=True)
    assert store.require_game(game_id).planned_start_at is None


def test_companion_room_membership_does_not_stand_in_for_an_identified_business_seat(tmp_path):
    """同行可先入群，但绑到实际匿名席后才能加入四席时间确认。"""
    from .venue_table_shop import VenueTableShop

    store, game_id, group, opening = _table(tmp_path, anonymous=True)
    shop = VenueTableShop(None, tmp_path / "negotiation.jsonl")
    assert shop(group, None, _OPEN) is None
    group.invite_companion("invite", inviter_id="a", recipient_id="d", at=_OPEN + timedelta(minutes=1))
    group.respond_companion("invite", actor_id="d", at=_OPEN + timedelta(minutes=2), accept=True)
    assert group.member_ids() == {AGENT_ID, *_MEMBERS}
    assert shop(group, None, _OPEN + timedelta(minutes=2)) is None
    assert opening.current() is None and store.require_game(game_id).planned_start_at is None
