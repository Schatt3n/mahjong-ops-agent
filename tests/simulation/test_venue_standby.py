"""核对备选、条款变更和真实业务席位各自的证据。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import GameStatus, using_clock

from .sim_adapter import build_runtime
from .venue_channel import OWNER_ID, MessageEnvelope, SandboxChannel
from .venue_fixture import VenueFixtureAgent
from .venue_standby import VenueStandby


_ROOM = "standby-rehearsal"
_PEOPLE = {"a": "甲", "b": "乙", "c": "丙", "d": "丁", "e": "戊"}
_AT = datetime.fromisoformat("2026-11-25T18:30:00+08:00")
_START = _AT + timedelta(minutes=30)
_DEADLINE = _AT + timedelta(minutes=18)
_FOUR = {"a", "b", "c", "d"}


def _at(minutes):
    return _AT + timedelta(minutes=minutes)


def _reopen(path, game_id):
    store = SQLiteAgentStore(path / "agent.sqlite3")
    runtime, _ = build_runtime(store, "mock", offline_client=VenueFixtureAgent(seed=21), local_coordination=True)
    channel = SandboxChannel(runtime, room_id=_ROOM, identities=_PEOPLE, archive_path=path / "chat.sqlite3")
    if game_id is None:
        return store, channel
    return VenueStandby(channel, path, game_id=game_id, incumbent="d", candidate="e")


@pytest.fixture
def standby(tmp_path):
    store, channel = _reopen(tmp_path, None)
    with using_clock(lambda: _AT):
        game, _ = store.create_game(
            conversation_id=channel.private_conversation_id("a"), organizer_id="a", organizer_name=_PEOPLE["a"],
            known_players=[], trace_id="setup-game", requirement={
                "game_type": "hangzhou_mahjong", "stake": "1", "start_time_kind": "scheduled",
                "planned_start_at": _START.isoformat(), "duration_kind": "fixed", "duration_hours": 4,
            },
        )
        for actor in ("b", "c", "d"):
            store.join_game(game_id=game.game_id, customer_id=actor, display_name=_PEOPLE[actor], trace_id=f"setup-{actor}")
    return VenueStandby(channel, tmp_path, game_id=game.game_id, incumbent="d", candidate="e")


def _send(standby, message_id, actor, text, *, at, quote=None):
    message = MessageEnvelope(message_id, actor, text, at, "private", OWNER_ID, quote)
    with using_clock(lambda: at):
        return standby.channel.send(message)


def _seats(standby, ids):
    game, = standby.store.games.values()
    assert game.game_id == standby.game_id
    assert {seat.customer_id for seat in game.participants if seat.status in {"joined", "confirmed"}} == ids
    assert game.remaining_seats() == 4 - len(ids)
    if len(ids) == 4:
        assert game.status == GameStatus.READY
    return game


def _accept(standby, *, label="original", minutes=1):
    invite_id, accept_id = f"invite-{label}", f"accept-{label}"
    assert standby.invite(invite_id, at=_at(minutes))["status"] == "invited"
    _send(standby, accept_id, "e", "行，有空位叫我。", at=_at(minutes + 1), quote=invite_id)
    assert standby.respond(accept_id, "accept_standby", reference_id=invite_id)["status"] == "standby"
    return accept_id


def _authorize_deadline(standby, *, message_id="signed-deadline", minutes=3, due=_DEADLINE):
    _send(standby, message_id, "d", f"我到 {due:%H:%M} 没回的话，这桌我的位置放掉吧。", at=_at(minutes))
    assert standby.respond(message_id, "set_deadline", deadline_at=due)["status"] == "deadline_registered"
    return message_id


def _withdraw(standby, *, minutes=5):
    _send(standby, "withdraw", "d", "我今晚不来了，帮我把这桌的位置取消。", at=_at(minutes))
    assert standby.respond("withdraw", "withdraw")["status"] == "released"


def test_standby_receipt_and_silence_leave_the_four_original_seats(standby, tmp_path):
    original = _seats(standby, _FOUR).to_dict()
    current = _accept(standby)
    assert standby.offer_seat("no-vacancy", acceptance_id=current, at=_at(4))["status"] == "full"
    assert standby.expire("invented-deadline", at=_DEADLINE)["status"] == "no_documented_deadline"
    _send(standby, "early-claim", "e", "这就算我有位置了吗？", at=_at(20), quote="invite-original")
    assert standby.respond("early-claim", "confirm_seat", reference_id="invite-original")["status"] == "wrong_offer"
    _send(standby, "third-party", "b", "替丁退掉吧。", at=_at(21))
    with pytest.raises(ValueError, match="acting customer's"):
        standby.respond("third-party", "withdraw")
    reopened = _reopen(tmp_path, standby.game_id)
    assert reopened.invite("invite-original", at=_at(1))["status"] == "invited"
    assert reopened.respond(current, "accept_standby", reference_id="invite-original")["status"] == "standby"
    assert _seats(reopened, _FOUR).to_dict() == original
    for actor in _PEOPLE:
        notices = [event for event in reopened.channel.events_for(actor) if event["direction"] == "owner"]
        assert [event["message_id"] for event in notices] == (["invite-original"] if actor == "e" else [])


def test_withdrawal_requires_a_new_candidate_confirmation_and_reopens_sqlite(standby, tmp_path):
    current = _accept(standby)
    _withdraw(standby)
    _seats(standby, _FOUR - {"d"})
    offer = standby.offer_seat("single-seat", acceptance_id=current, at=_at(6))
    assert offer["status"] == "seat_offered" and "19:00" in offer["text"]
    _seats(standby, _FOUR - {"d"})
    _send(standby, "no-offer-ref", "e", "留给我吧。", at=_at(7))
    with pytest.raises(ValueError, match="delivered private offer"):
        standby.respond("no-offer-ref", "confirm_seat", reference_id="single-seat")
    _send(standby, "confirm-seat", "e", "给我留一个，我七点到。", at=_at(8), quote="single-seat")
    confirmed = standby.respond("confirm-seat", "confirm_seat", reference_id="single-seat")
    assert confirmed["status"] == "confirmed"
    assert [(row["entity_id"], row["trace_id"]) for row in confirmed["transitions"]
            if row["entity_type"] == "game_participant"] == [(f"{standby.game_id}:e", confirmed["trace_id"])]
    reopened = _reopen(tmp_path, standby.game_id)
    assert _send(reopened, "confirm-seat", "e", "给我留一个，我七点到。", at=_at(8), quote="single-seat")[
        "delivery_deduplicated"]
    assert reopened.respond("confirm-seat", "confirm_seat", reference_id="single-seat") == confirmed
    assert reopened.offer_seat("single-seat", acceptance_id=current, at=_at(6)) == offer
    _seats(reopened, (_FOUR - {"d"}) | {"e"})
    transitions = reopened.store.transitions
    assert len([entry for entry in transitions if entry.entity_id == f"{reopened.game_id}:e"]) == 1
    assert [row["message_id"] for row in reopened.channel.events() if row["direction"] == "owner"] == [
        "invite-original", "single-seat",
    ]


@pytest.mark.parametrize("late_before_timer", [True, False])
def test_deadline_and_late_original_confirmation_in_either_order(standby, tmp_path, late_before_timer):
    current = _accept(standby)
    deadline = _authorize_deadline(standby)
    assert standby.expire(deadline, at=_DEADLINE - timedelta(seconds=1))["status"] == "not_due"
    _seats(standby, _FOUR)
    _send(standby, "late-original", "d", "我还能来，原来那个位置给我留着吧。",
          at=_DEADLINE, quote=deadline)
    if late_before_timer:
        assert standby.respond("late-original", "confirm_original", reference_id=deadline)["status"] == "deadline_passed"
    assert standby.expire(deadline, at=_DEADLINE)["status"] == "released"
    _seats(standby, _FOUR - {"d"})
    assert standby.offer_seat("vacancy", acceptance_id=current, at=_at(19))["status"] == "seat_offered"
    _send(standby, "candidate-yes", "e", "行，给我留一个。", at=_at(20), quote="vacancy")
    assert standby.respond("candidate-yes", "confirm_seat", reference_id="vacancy")["status"] == "confirmed"
    reopened = _reopen(tmp_path, standby.game_id)
    assert reopened.respond("late-original", "confirm_original", reference_id=deadline)["status"] == "deadline_passed"
    assert reopened.expire(deadline, at=_at(21))["status"] == "released"
    _seats(reopened, (_FOUR - {"d"}) | {"e"})
    assert len([row for row in reopened.events() if row["action"] == "expire"]) == 1


def test_timely_original_confirmation_cancels_conditional_release(standby):
    _accept(standby)
    first = _authorize_deadline(standby, due=_at(16))
    deadline = _authorize_deadline(standby, message_id="later-deadline", minutes=4)
    assert standby.expire(first, at=_at(16))["status"] == "deadline_superseded"
    _send(standby, "timely", "d", "刚看到，我能来，位置保留。", at=_at(17), quote=deadline)
    assert standby.respond("timely", "confirm_original", reference_id=deadline)["status"] == "original_confirmed"
    assert standby.expire(deadline, at=_DEADLINE)["status"] == "original_confirmed"
    _seats(standby, _FOUR)


@pytest.mark.parametrize("revision", ["clock", "duration", "smoke", "clock_round_trip"])
def test_changed_terms_invalidate_an_already_delivered_vacancy_offer(standby, revision):
    old = _accept(standby)
    _withdraw(standby)
    assert standby.offer_seat("previous-vacancy", acceptance_id=old, at=_at(6))["status"] == "seat_offered"
    patch = {
        "clock": {"planned_start_at": (_START + timedelta(minutes=30)).isoformat()},
        "duration": {"duration_hours": 3},
        "smoke": {"smoke_preference": "no_smoking"},
        "clock_round_trip": {"planned_start_at": (_START + timedelta(minutes=30)).isoformat()},
    }[revision]
    with using_clock(lambda: _at(7)):
        standby.store.update_game_requirement(game_id=standby.game_id, requirement_patch=patch,
                                              reason="组织者更改开场或时长", trace_id="change")
        if revision == "clock_round_trip":
            standby.store.update_game_requirement(game_id=standby.game_id,
                requirement_patch={"planned_start_at": _START.isoformat()}, reason="组织者改回原开场", trace_id="change-back")
    assert standby.offer_seat("do-not-resend", acceptance_id=old, at=_at(8))["status"] == "terms_changed"
    _send(standby, "late-old-yes", "e", "行，给我留着。", at=_at(8), quote="previous-vacancy")
    assert standby.respond("late-old-yes", "confirm_seat", reference_id="previous-vacancy")["status"] == "terms_changed"
    _seats(standby, _FOUR - {"d"})
    new = _accept(standby, label="revision", minutes=9)
    offered = standby.offer_seat("new-vacancy", acceptance_id=new, at=_at(11))
    assert offered["status"] == "seat_offered" and ("3 小时" if revision == "duration" else "4 小时") in offered["text"]
    assert ("19:30" if revision == "clock" else "19:00") in offered["text"]
    assert ("无烟" if revision == "smoke" else "烟况待定") in offered["text"]
    _send(standby, "fresh-yes", "e", "新条件可以，留我一个。", at=_at(12), quote="new-vacancy")
    assert standby.respond("fresh-yes", "confirm_seat", reference_id="new-vacancy")["status"] == "confirmed"
    _seats(standby, (_FOUR - {"d"}) | {"e"})


def test_two_sqlite_writers_compete_for_the_vacant_seat(standby, tmp_path):
    current = _accept(standby)
    _withdraw(standby)
    assert standby.offer_seat("vacancy", acceptance_id=current, at=_at(6))["status"] == "seat_offered"
    _send(standby, "candidate-yes", "e", "给我留一个。", at=_at(7), quote="vacancy")
    _send(standby, "original-new-request", "d", "改主意了，如果还有位置我重新报一个。", at=_at(7))
    independent = _reopen(tmp_path, standby.game_id)

    def new_original_request():
        """新申请不再沿用旧截止确认。用正式 Store 事务与候补同时争取一个空席。"""
        with using_clock(lambda: _at(7)):
            try:
                independent.store.join_game(game_id=standby.game_id, customer_id="d", display_name=_PEOPLE["d"],
                                            trace_id="explicit-new-original-request")
            except ValueError as exc:
                assert "seat capacity exceeded" in str(exc)
                return "full"
        return "confirmed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        original = pool.submit(new_original_request)
        candidate = pool.submit(standby.respond, "candidate-yes", "confirm_seat", reference_id="vacancy")
    results = (original.result(), candidate.result()["status"])
    assert sorted(results) == ["confirmed", "full"]
    winner = "d" if results[0] == "confirmed" else "e"
    _seats(_reopen(tmp_path, standby.game_id), {"a", "b", "c", winner})


def test_promotion_transaction_rolls_back_with_the_receipt_and_retries(standby, monkeypatch):
    current = _accept(standby)
    _withdraw(standby)
    assert standby.offer_seat("vacancy", acceptance_id=current, at=_at(6))["status"] == "seat_offered"
    _send(standby, "candidate-yes", "e", "给我留一个。", at=_at(7), quote="vacancy")
    original = SQLiteAgentStore._ensure_game_board_arrangement_in_transaction

    def fail_after_seat_write(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected failure before evidence commit")

    with monkeypatch.context() as patch:
        patch.setattr(SQLiteAgentStore, "_ensure_game_board_arrangement_in_transaction", fail_after_seat_write)
        with pytest.raises(RuntimeError, match="evidence commit"):
            standby.respond("candidate-yes", "confirm_seat", reference_id="vacancy")
    _seats(standby, _FOUR - {"d"})
    assert not any(row["event_id"] == "candidate-yes" for row in standby.events())
    assert standby.respond("candidate-yes", "confirm_seat", reference_id="vacancy")["status"] == "confirmed"
    _seats(standby, (_FOUR - {"d"}) | {"e"})
