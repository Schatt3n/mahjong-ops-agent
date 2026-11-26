"""验证匿名席经真实桌群入群证据绑定后仍为同一张四人桌。"""

from datetime import datetime, timedelta

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.domains.game_commitments import active_game_participant_ids
from mahjong_agent_runtime.models import GameStatus, using_clock

from .sim_adapter import build_runtime
from .venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel
from .venue_table_group import VenueTableGroup
from .venue_table_opening import VenueTableOpening
from .venue_table_party import VenueTableParty


_AT = datetime.fromisoformat("2026-11-27T18:05:00+08:00")
_INVITED = _AT + timedelta(minutes=1)
_ACCEPTED = _AT + timedelta(minutes=2)
_BIND = _AT + timedelta(minutes=3)
_START = _AT.replace(hour=19, minute=30)
_MEMBERS = ("host", "b", "c", "companion")
_ROSTER = (*_MEMBERS, "outsider", "rival", "f", "g", "h")
_INVITE = "host-invite-for-companion"


def _channel(store, tmp_path):
    """通过独立渠道建立已注册的模拟身份。"""
    runtime, _ = build_runtime(store, "mock")
    return SandboxChannel(runtime, room_id="public-fixture", archive_path=tmp_path / "chat.sqlite3",
                          identities={name: name for name in _ROSTER})


def _open(tmp_path):
    """由正式业务接口形成一张含一匿名席的满桌。"""
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    store.configure_rooms(["东二"])
    channel = _channel(store, tmp_path)
    with using_clock(lambda: _AT - timedelta(minutes=1)):
        game, _ = store.create_game(
            conversation_id=channel.private_conversation_id("host"), organizer_id="host", organizer_name="host",
            known_players=[{"customer_id": "host", "seat_count": 2, "anonymous_seat_count": 1}], trace_id="fixture-host",
            requirement={"game_type": "hangzhou_mahjong", "start_time_kind": "asap_when_full", "duration_hours": 4},
        )
        for player in ("b", "c"):
            game, _ = store.join_game(game_id=game.game_id, customer_id=player, display_name=player, trace_id=f"fixture:{player}")
    assert game.status == GameStatus.READY and game.seat_summary()["anonymous_seat_count"] == 1
    return store, channel, channel.open_table_group(game.game_id, at=_AT)


def _invite(group, *, accept=True):
    """对应报名联系人发出邀请，同行本人从同一回执链响应。"""
    group.invite_companion(_INVITE, inviter_id="host", recipient_id="companion", at=_INVITED)
    return group.respond_companion(_INVITE, actor_id="companion", at=_ACCEPTED, accept=accept)


def test_accepted_companion_identifies_one_seat_and_can_vote_and_attend(tmp_path):
    """入群回执独立保存，绑定后本人可对正式提议投票并作为第四个签到身份。"""
    store, channel, group = _open(tmp_path)
    assert group.member_ids() == {AGENT_ID, "host", "b", "c"}
    initial = store.require_game(group.game_id)
    original_party = next(row for row in initial.parties if row.contact_id == "host")
    _invite(group)
    assert group.member_ids() == {AGENT_ID, *_MEMBERS}
    assert store.require_game(group.game_id).to_dict() == initial.to_dict()

    party = VenueTableParty(group)
    with pytest.raises(ValueError, match="accepted and admitted invitation"):
        party.bind_companion(_INVITE, actor_id="host", at=_BIND)
    bound = party.bind_companion(_INVITE, actor_id="companion", at=_BIND)
    assert bound.status == GameStatus.READY and bound.remaining_seats() == 0
    assert bound.seat_summary()["claimed_seats"] == initial.seat_summary()["claimed_seats"] == 4
    assert bound.seat_summary()["anonymous_seat_count"] == 0
    assert active_game_participant_ids(bound) == set(_MEMBERS)
    assert {row.customer_id: row.seat_count for row in bound.participants} == dict.fromkeys(_MEMBERS, 1)
    assert bound.requirement["known_player_count"] == 4 and bound.requirement["needed_seats"] == 0
    assert next(row for row in bound.participants if row.customer_id == "companion").registered_by_customer_id == "companion"
    with using_clock(lambda: _BIND), pytest.raises(ValueError, match="original registration contact"):
        store.record_candidate_reply(game_id=group.game_id, customer_id="companion", display_name="companion",
                                     status="declined", trace_id="proxy-withdrawal", reported_by=("host", bound.conversation_id))
    transition, = [row for row in store.transitions if row.reason == "venue_table_companion_bound"]
    assert transition.from_status == f"{original_party.party_id}:anonymous"
    assert transition.trace_id.endswith(f":{_INVITE}")

    reopened = SQLiteAgentStore(store.path)
    restored = _channel(reopened, tmp_path).table_group(VenueTableGroup.room_id_for_game(group.game_id))
    assert VenueTableParty(restored).bind_companion(_INVITE, actor_id="companion", at=_BIND).to_dict() == bound.to_dict()
    assert sum(row.reason == "venue_table_companion_bound" for row in reopened.transitions) == 1
    opening = VenueTableOpening(restored)
    offered = restored.send(opening.proposal_message("actual-opening", at=_BIND + timedelta(minutes=1), start_at=_START))
    assert set(offered.recipient_ids) == set(_MEMBERS)
    opening.record_proposal(offered, start_at=_START)
    for index, player in enumerate(_MEMBERS, 2):
        ballot = restored.send(MessageEnvelope(f"opening-yes:{player}", player, "19:30可以。",
            _BIND + timedelta(minutes=index), "group", restored.room_id, offered.message.message_id))
        result = opening.record_vote(ballot, proposal_id=offered.message.message_id, accepts=True)
        assert result == ("committed" if player == "companion" else "waiting")
    assert reopened.require_game(group.game_id).planned_start_at == _START
    started, _ = reopened.start_game_play(group.game_id, attendee_ids=_MEMBERS, room_id="东二", at=_START,
                                         trace_id="identified-attendance")
    assert set(started.actual_attendee_ids) == set(_MEMBERS)
    assert not channel.events_for("outsider")


def test_binding_rejects_unaccepted_invitation_and_wrong_identity(tmp_path):
    """只有同行本人接受有效邀请，报名联系人也不能代替本人作此绑定。"""
    store, _, group = _open(tmp_path)
    party = VenueTableParty(group)
    initial = store.require_game(group.game_id)
    group.invite_companion(_INVITE, inviter_id="host", recipient_id="companion", at=_INVITED)
    with pytest.raises(ValueError, match="accepted and admitted invitation"):
        party.bind_companion(_INVITE, actor_id="companion", at=_BIND)
    group.respond_companion(_INVITE, actor_id="companion", at=_ACCEPTED, accept=False)
    for actor_id in ("host", "companion", "outsider"):
        with pytest.raises(ValueError, match="accepted and admitted invitation"):
            party.bind_companion(_INVITE, actor_id=actor_id, at=_BIND)
    assert store.require_game(group.game_id).to_dict() == initial.to_dict()


@pytest.mark.parametrize("companion_as_known_member", [False, True])
def test_binding_rejects_overlapping_known_commitment(tmp_path, companion_as_known_member):
    """不论另一局由本人报名还是原已有具名同行，时间冲突均保留原桌席位。"""
    store, channel, group = _open(tmp_path)
    host = "rival" if companion_as_known_member else "companion"
    party_payload = [{"customer_id": host, "seat_count": 2, "known_member_ids": [host, "companion"]}]
    with using_clock(lambda: _INVITED):
        other, _ = store.create_game(conversation_id=channel.private_conversation_id(host), organizer_id=host,
            organizer_name=host, known_players=party_payload if companion_as_known_member else [], trace_id="other-host",
            requirement={"game_type": "hongzhong_mahjong", "start_time_kind": "scheduled", "planned_start_at": _START.isoformat()})
        for name in ("f", "g") if companion_as_known_member else ("f", "g", "h"):
            other, _ = store.join_game(game_id=other.game_id, customer_id=name, display_name=name, trace_id=f"other:{name}")
    assert other.status == GameStatus.READY
    _invite(group)
    initial = store.require_game(group.game_id)
    with pytest.raises(ValueError, match="overlapping ready game"):
        VenueTableParty(group).bind_companion(_INVITE, actor_id="companion", at=_BIND)
    assert store.require_game(group.game_id).to_dict() == initial.to_dict()
    assert "companion" in group.member_ids()
    assert not [row for row in store.transitions if row.reason == "venue_table_companion_bound"]


def test_accepted_invitation_cannot_claim_a_withdrawn_and_recreated_contact_seat(tmp_path):
    """满员状态恢复后旧邀请也不能指向已经撤回并重新登记的席。"""
    store, _, group = _open(tmp_path)
    _invite(group)
    with using_clock(lambda: _BIND):
        store.record_candidate_reply(game_id=group.game_id, customer_id="host", display_name="host",
                                     status="declined", seat_count=2, trace_id="host-left")
    with using_clock(lambda: _BIND + timedelta(minutes=1)):
        ready, _ = store.join_game(game_id=group.game_id, customer_id="host", display_name="host", seat_count=2,
                                   trace_id="host-rejoined")
    assert ready.status == GameStatus.READY
    with pytest.raises(ValueError, match="predates a change"):
        VenueTableParty(group).bind_companion(_INVITE, actor_id="companion", at=_BIND + timedelta(minutes=2))
    assert store.require_game(group.game_id).seat_summary()["anonymous_seat_count"] == 1


def test_binding_retries_original_acceptance_after_transaction_failure(tmp_path, monkeypatch):
    """牌局写入后的失败会一起回滚业务证据，重新执行仍只保留一条确认。"""
    store, _, group = _open(tmp_path)
    _invite(group)
    party = VenueTableParty(group)
    initial = store.require_game(group.game_id)
    append = SQLiteAgentStore._append_transition

    def crash(self, transition):
        if transition.reason == "venue_table_companion_bound":
            raise RuntimeError("simulated write failure")
        return append(self, transition)

    with monkeypatch.context() as patch:
        patch.setattr(SQLiteAgentStore, "_append_transition", crash)
        with pytest.raises(RuntimeError, match="simulated write failure"):
            party.bind_companion(_INVITE, actor_id="companion", at=_BIND)
    assert store.require_game(group.game_id).to_dict() == initial.to_dict()
    assert group.invitation_for("companion", _INVITE).accepted is True
    assert party.bind_companion(_INVITE, actor_id="companion", at=_BIND).seat_summary()["claimed_seats"] == 4
    assert sum(row.reason == "venue_table_companion_bound" for row in store.transitions) == 1
