"""覆盖真实收件、独立同行选择、恢复及四位具名成员的桌群协商。"""

from dataclasses import replace
from datetime import datetime, timedelta
import json

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.domains.game_commitments import active_game_participant_ids
from mahjong_agent_runtime.models import GameStatus, using_clock

from .sim_adapter import build_runtime
from .venue_actors import ActorContext, CompanionInviteContext, ModelActor, Relationship, _parse_action, generate_personas
from .venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel
from .venue_rolling_companions import TABLE_COMPANION_WAKE
from .venue_rolling_roles import VenueRollingRoles
from .venue_table_group import VenueTableGroup
from .venue_table_opening import VenueTableOpening
from .venue_table_party import VenueTableParty
from .venue_table_shop import VenueTableShop


_AT = datetime.fromisoformat("2026-11-27T18:05:00+08:00")
_START = _AT.replace(hour=19, minute=30)
_PEOPLE = generate_personas(42, count=8)
_HOST, _B, _C, _COMPANION, _BYSTANDER, _INACTIVE = _PEOPLE[:6]
_PLAYER_IDS = {person.sender_id for person in (_HOST, _B, _C, _COMPANION)}
_SUGGESTION = "不如七点半开？"
_YES = "我觉得这个点行。"


class _ActorClient:
    """只为本回放提供人工选定的动作，通过正常模型角色合同解析。"""

    def __init__(self, response_action="accept_companion", *, offer_action="invite_companion"):
        self.response_action = response_action
        self.offer_action = offer_action
        self.invite_contexts = []

    def complete(self, messages, *, trace_id, timeout_seconds):
        body = json.loads(messages[-1]["content"])
        if "draft" in body:
            assert body["draft"] in {_SUGGESTION, _YES}
            # 这些测试短句只提出开场方案或表态，未单独报本人到店/离店时间。
            return json.dumps({"claims": []})
        delivery, persona = body["delivery"] or {}, body["persona"]["persona_id"]
        if context := body.get("companion_invite"):
            self.invite_contexts.append(body)
            assert "邀请卡" in messages[0]["content"]
            if context["stage"] == "offer":
                assert persona == _HOST.persona_id and delivery["sender_id"] == AGENT_ID
                assert context == {"stage": "offer", "contact_name": _COMPANION.name}
                return json.dumps({"action": self.offer_action})
            assert persona == _COMPANION.persona_id and not body["history"] and not delivery
            assert not body["private_targets"] and not body["group_targets"]
            assert context == {"stage": "received", "contact_name": _HOST.name}
            return json.dumps({"action": self.response_action})
        message_id = delivery.get("message_id", "")
        if message_id.startswith("table-opening-inquiry:") and persona == _HOST.persona_id:
            text = _SUGGESTION
        elif message_id.startswith("table-opening-offer|"):
            text = _YES
        else:
            return '{"action":"wait"}'
        return json.dumps({"action": "send", "kind": "group", "recipient_id": delivery["recipient_id"],
                           "text": text, "quote_message_id": message_id}, ensure_ascii=False)


class _ShopClient:
    """受控标注供正式桌群店家处理器解释受领消息，不代替落库。"""

    def complete(self, messages, *, trace_id, timeout_seconds):
        body = json.loads(messages[-1]["content"])
        current = body["current"]
        if current["text"] == _SUGGESTION:
            assert current["sender_id"] == _HOST.sender_id and body["active_proposal"] is None
            return json.dumps({"vote": "other", "proposed_start": _START.isoformat()})
        assert current["text"] == _YES and body["active_proposal"]["start_at"] == _START.isoformat()
        return '{"vote":"accept","proposed_start":null}'


def _rolling(tmp_path, client, *, partners=(_COMPANION,), shop=None):
    people = (replace(_HOST, relationships=tuple(Relationship(person.persona_id, person.name, "regular_partner", person.sender_id)
                                                 for person in partners)), *_PEOPLE[1:])
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    runtime, _ = build_runtime(store, "mock")
    channel = SandboxChannel(runtime, room_id="public-notice", archive_path=tmp_path / "chat.sqlite3",
                             identities={person.sender_id: person.name for person in people})
    shop = VenueTableShop(_ShopClient(), tmp_path / "shop.jsonl") if shop is None else shop
    return VenueRollingRoles(people=people, actor=ModelActor(client), judge=None, channel=channel,
        run_dir=tmp_path, start_at=_AT, seed=42, active_users=5, workers=4, dialogue_turns=2, judge_users=1,
        table_groups=True, table_shop=shop)


def _ready_game(channel, *, host=_HOST, confirmed=(_B, _C), anonymous=1, planned_start=None):
    store = channel.runtime.store
    requirement = {"game_type": "hangzhou_mahjong", "stake": 1,
                   "start_time_kind": "asap_when_full", "duration_hours": 4}
    if planned_start is not None:
        requirement.update(start_time_kind="scheduled", planned_start_at=planned_start.isoformat())
    with using_clock(lambda: _AT - timedelta(minutes=1)):
        game, _ = store.create_game(conversation_id=channel.private_conversation_id(host.sender_id),
            organizer_id=host.sender_id, organizer_name=host.name, requirement=requirement,
            known_players=[{"customer_id": host.sender_id, "seat_count": anonymous + 1, "anonymous_seat_count": anonymous}],
            trace_id=f"offline-companion-table:{host.sender_id}")
        for person in confirmed:
            game, _ = store.join_game(game_id=game.game_id, customer_id=person.sender_id, display_name=person.name,
                                     trace_id=f"offline-companion-claim:{host.sender_id}:{person.sender_id}")
    assert game.status == GameStatus.READY and game.seat_summary()["anonymous_seat_count"] == anonymous
    return game


@pytest.mark.parametrize("interruption", ["before_companion_wake", "before_original_seat_binding", "after_original_seat_binding"])
def test_roles_choose_and_accept_private_invite_then_all_four_negotiate_after_restart(tmp_path, monkeypatch, interruption):
    class Interrupted(BaseException):
        """模拟真实进程中断，已写决定和交付不能被吞掉或重算。"""

    client = _ActorClient()
    scenario = _rolling(tmp_path, client)
    original = _ready_game(scenario.channel)
    room = VenueTableGroup.room_id_for_game(original.game_id)
    with monkeypatch.context() as patch:
        if interruption == "before_companion_wake":
            enqueue = scenario.journal.enqueue

            def stop_after_invite(**kwargs):
                if kwargs["kind"] == TABLE_COMPANION_WAKE:
                    raise Interrupted
                return enqueue(**kwargs)

            patch.setattr(scenario.journal, "enqueue", stop_after_invite)
        elif interruption == "before_original_seat_binding":
            def stop_after_acceptance(*args, **kwargs):
                raise Interrupted

            patch.setattr(VenueTableParty, "bind_companion", stop_after_acceptance)
        else:
            finish = scenario.journal.finish

            def stop_after_binding(event, result):
                if event.kind == TABLE_COMPANION_WAKE:
                    raise Interrupted
                return finish(event, result)

            patch.setattr(scenario.journal, "finish", stop_after_binding)
        with pytest.raises(Interrupted):
            scenario.run()

    inviter, = [event for event in scenario.journal.events()
                if (event.decision or {}).get("decision", {}).get("action") == {"action": "invite_companion"}]
    invite_id = f"table-invite:{inviter.event_id}"
    group = scenario.channel.table_group(room)
    reminder, = group.delivered_receipts()
    actual_invite = group.invitation_for(_COMPANION.sender_id, invite_id)
    assert actual_invite.inviter_id == _HOST.sender_id and inviter.caused_by == reminder.message.message_id
    assert actual_invite.accepted is (None if interruption == "before_companion_wake" else True)
    saved_game = scenario.channel.runtime.store.require_game(original.game_id)
    if interruption == "after_original_seat_binding":
        assert active_game_participant_ids(saved_game) == _PLAYER_IDS and saved_game.planned_start_at is None
    else:
        assert saved_game.to_dict() == original.to_dict()
    assert set(reminder.recipient_ids) == _PLAYER_IDS - {_COMPANION.sender_id}
    assert all(group.invitation_for(viewer, invite_id) is None for viewer in (AGENT_ID, _B.sender_id, _BYSTANDER.sender_id))
    assert not scenario.channel.events_for(_COMPANION.sender_id) and not scenario.channel.events_for(_BYSTANDER.sender_id)

    resumed = _rolling(tmp_path, client)
    report = resumed.run()
    # 本地客户端的原始动作走模型合同，但没有伪造 HTTP 来源以通过真模型验收。
    assert {key for key, value in report["checks"].items() if not value} == {"customer_decisions_from_model"}
    assert report["rolling_table_groups"]["companion_outcomes"] == {"table_companion_invited": 1, "table_companion_bound": 1}
    assert [row["companion_invite"]["stage"] for row in client.invite_contexts] == ["offer", "received"]
    group = resumed.channel.table_group(room)
    accepted = group.invitation_for(_COMPANION.sender_id, invite_id)
    assert accepted.invited_at == actual_invite.invited_at and accepted.accepted is True
    assert not actual_invite.response_at or accepted.response_at == actual_invite.response_at
    recipient, = [event for event in resumed.journal.events() if event.kind == TABLE_COMPANION_WAKE]
    assert recipient.actor_id == _COMPANION.sender_id and recipient.caused_by == invite_id
    assert recipient.decision["visible_cause_verified"] and recipient.result["companion_invitation"]["accepted"]
    assert inviter.event_id in {event.event_id for event in resumed.journal.events() if event.status == "finished"}
    assert set(group.member_ids()) == {AGENT_ID, *_PLAYER_IDS}

    records = group.delivered_receipts()
    inquiry, = [row for row in records if row.message.message_id.startswith("table-opening-inquiry:")]
    proposal, = [row for row in records if row.message.message_id.startswith("table-opening-offer|")]
    assert inquiry.message.sent_at == accepted.response_at and set(inquiry.recipient_ids) == _PLAYER_IDS
    assert set(proposal.recipient_ids) == _PLAYER_IDS
    assert {row.message.sender_id for row in records if row.message.quoted_message_id == proposal.message.message_id} == _PLAYER_IDS
    assert VenueTableOpening(group).current()["committed"]
    assert reminder.message.message_id not in {row["message_id"] for row in resumed.channel.events_for(_COMPANION.sender_id)}
    assert not resumed.channel.events_for(_BYSTANDER.sender_id)
    assert invite_id not in {row["message_id"] for row in resumed.channel.events()}
    store = SQLiteAgentStore(resumed.channel.runtime.store.path)
    game = store.require_game(original.game_id)
    assert active_game_participant_ids(game) == _PLAYER_IDS
    assert game.seat_summary()["claimed_seats"] == 4 and game.seat_summary()["anonymous_seat_count"] == 0
    assert (game.planned_start_at, game.planned_end_at) == (_START, _START + timedelta(hours=4))
    assert sum(row.reason == "venue_table_companion_bound" for row in store.transitions) == 1
    assert sum(row.reason == "venue_table_opening_confirmed" for row in store.transitions) == 1
    last = _rolling(tmp_path, client)
    assert last.run()["rolling_table_groups"] == report["rolling_table_groups"]
    assert last.channel.table_group(room).delivered_receipts() == records and len(client.invite_contexts) == 2


@pytest.mark.parametrize("reply", ["decline_companion", "wait"])
def test_invitee_can_decline_or_not_respond_without_seat_or_group_access(tmp_path, reply):
    client = _ActorClient(reply)
    scenario = _rolling(tmp_path, client)
    game = _ready_game(scenario.channel)
    report = scenario.run()
    group = scenario.channel.table_group(VenueTableGroup.room_id_for_game(game.game_id))
    event, = [row for row in scenario.journal.events() if row.kind == TABLE_COMPANION_WAKE]
    assert group.invitation_for(_COMPANION.sender_id, event.caused_by).accepted is (False if reply == "decline_companion" else None)
    assert _COMPANION.sender_id not in group.member_ids() and not scenario.channel.events_for(_COMPANION.sender_id)
    assert len(group.delivered_receipts()) == 1 and scenario.channel.runtime.store.require_game(game.game_id).to_dict() == game.to_dict()
    assert report["rolling_table_groups"]["companion_outcomes"].get("table_companion_bound", 0) == 0


@pytest.mark.parametrize("partners", [(), (_COMPANION, _BYSTANDER), (_COMPANION, _INACTIVE), (_INACTIVE,), (_B,)])
def test_reminder_does_not_guess_a_recipient_without_one_eligible_known_partner(tmp_path, partners):
    client = _ActorClient()
    scenario = _rolling(tmp_path, client, partners=partners)
    game = _ready_game(scenario.channel)
    report = scenario.run()
    group = scenario.channel.table_group(VenueTableGroup.room_id_for_game(game.game_id))
    assert len(group.delivered_receipts()) == 1 and not client.invite_contexts
    assert not [row for row in scenario.journal.events() if row.kind == TABLE_COMPANION_WAKE]
    assert "companion_outcomes" not in report["rolling_table_groups"]


def test_multiple_anonymous_seats_do_not_assign_single_known_person_to_a_guess(tmp_path):
    client = _ActorClient()
    scenario = _rolling(tmp_path, client)
    _ready_game(scenario.channel, confirmed=(_B,), anonymous=2)
    scenario.run()
    assert not client.invite_contexts and not [row for row in scenario.journal.events() if row.kind == TABLE_COMPANION_WAKE]


def test_contact_can_wait_and_no_invitation_is_sent(tmp_path):
    client = _ActorClient(offer_action="wait")
    scenario = _rolling(tmp_path, client)
    _ready_game(scenario.channel)
    report = scenario.run()
    assert [row["companion_invite"]["stage"] for row in client.invite_contexts] == ["offer"]
    assert "companion_outcomes" not in report["rolling_table_groups"]


def test_ordinary_group_message_cannot_authorize_companion_invitation(tmp_path):
    def ordinary_shop(group, reply, at):
        return MessageEnvelope("different-receipt", AGENT_ID, "带朋友来的，麻烦把还没进群的同行拉进来哈。", at, "group", group.room_id)

    client = _ActorClient()
    scenario = _rolling(tmp_path, client, shop=ordinary_shop)
    _ready_game(scenario.channel)
    scenario.run()
    assert not client.invite_contexts
    assert not [row for row in scenario.journal.events() if row.kind == TABLE_COMPANION_WAKE]


@pytest.mark.parametrize("companion,action", [
    (None, {"action": "invite_companion"}),
    (CompanionInviteContext("offer", _COMPANION.name), {"action": "accept_companion"}),
    (CompanionInviteContext("offer", _COMPANION.name), {"action": "invite_companion", "recipient_id": _BYSTANDER.sender_id}),
])
def test_model_cannot_use_another_roles_action_or_insert_an_arbitrary_recipient(companion, action):
    projected = ActorContext(_HOST, _AT.isoformat(), companion_invite=companion).to_dict()
    with pytest.raises(ValueError, match="its own invitation context"):
        _parse_action(json.dumps(action), projected)


def test_overlapping_confirmed_table_blocks_admission_before_binding(tmp_path):
    scenario = _rolling(tmp_path, _ActorClient())
    game = _ready_game(scenario.channel)
    _ready_game(scenario.channel, host=_COMPANION, confirmed=_PEOPLE[5:], anonymous=0, planned_start=_START)
    report = scenario.run()
    group = scenario.channel.table_group(VenueTableGroup.room_id_for_game(game.game_id))
    event, = [row for row in scenario.journal.events() if row.kind == TABLE_COMPANION_WAKE]
    assert "overlapping ready game" in event.result["companion_error"]
    assert group.invitation_for(_COMPANION.sender_id, event.caused_by).accepted is None
    assert _COMPANION.sender_id not in group.member_ids() and scenario.channel.runtime.store.require_game(game.game_id).to_dict() == game.to_dict()
    assert report["rolling_table_groups"]["companion_outcomes"] == {"table_companion_invited": 1, "table_companion_unavailable": 1}
