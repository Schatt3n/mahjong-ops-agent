"""用真实建群、调度收件和 SQLite 盘面验证人齐开后的两轮协商。"""

from datetime import datetime, timedelta
import json

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import GameStatus, using_clock

from .sim_adapter import build_runtime
from .venue_actors import ActorAction, MockActor, generate_personas
from .venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel
from .venue_rolling_roles import VenueRollingRoles
from .venue_table_group import VenueTableGroup
from .venue_table_opening import VenueTableOpening
from .venue_table_shop import VenueTableShop


_AT = datetime.fromisoformat("2026-11-27T18:05:00+08:00")
_FIRST, _SECOND = (_AT.replace(hour=19, minute=minute) for minute in (0, 30))
_PEOPLE = generate_personas(42, count=5)
_PLAYERS = frozenset(person.sender_id for person in _PEOPLE[:4])
_HOST, _DECLINER, _BYSTANDER = (_PEOPLE[index].sender_id for index in (0, 3, 4))
_EARLY_YES, _EARLY_NO, _LATE_YES = "七点能到。", "七点赶不到，七点半可以。", "七点半能到。"


class _OpeningActor(MockActor):
    """固定人工标注的时间答复，只对自己实际收到的当前店家提议动作。"""

    def _choose(self, context, projected, event, rng):
        delivery = projected["delivery"] or {}
        if delivery.get("sender_id") != AGENT_ID or delivery.get("recipient_id") not in context.group_targets:
            return ActorAction("wait")
        proposal_id = delivery["message_id"]
        if proposal_id.endswith(":first"):
            text = _EARLY_NO if context.persona.sender_id == _DECLINER else _EARLY_YES
        elif proposal_id.endswith(":revised"):
            text = _LATE_YES
        else:
            return ActorAction("wait")
        return ActorAction("send", text, "group", delivery["recipient_id"], proposal_id)


class _OpeningShop:
    """在本固定对照内应用人工标注的票义，不从自由来话推断同意。"""

    def __init__(self):
        self.votes = []

    def __call__(self, group, reply, at):
        opening = VenueTableOpening(group)
        first, revised = (f"test-opening:{group.game_id}:{version}" for version in ("first", "revised"))
        if reply is None:
            return opening.proposal_message(first, at=at, start_at=_FIRST)

        proposal_id = reply.message.quoted_message_id
        answers = {first: {_EARLY_YES: True, _EARLY_NO: False}, revised: {_LATE_YES: True}}
        assert reply.message.text in answers[proposal_id]
        accepts = answers[proposal_id][reply.message.text]
        start_at = _FIRST if proposal_id == first else _SECOND
        offered, = (row for row in group.delivered_receipts() if row.message.message_id == proposal_id)
        opening.record_proposal(offered, start_at=start_at)
        result = opening.record_vote(reply, proposal_id=proposal_id, accepts=accepts)
        assert (group.store.require_game(group.game_id).planned_start_at is not None) == (result == "committed")
        self.votes.append((proposal_id, reply.message.sender_id, accepts, result))
        if result == "declined":
            return opening.proposal_message(revised, at=at, start_at=_SECOND)
        if result == "committed":
            return MessageEnvelope(f"test-opening:{group.game_id}:confirmed", AGENT_ID, "好，约好七点半开。", at,
                                   "group", group.room_id)
        return None


def _rolling_opening(tmp_path, *, people, actor, shop, workers, dialogue_turns):
    """打开同一个营业沙盒和桌群调度，可在重开后继续验证原回执。"""
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    runtime, _ = build_runtime(store, "mock")
    channel = SandboxChannel(runtime, room_id="rolling-opening-public", archive_path=tmp_path / "chat.sqlite3",
                             identities={person.sender_id: person.name for person in people})
    return VenueRollingRoles(people=people, actor=actor, judge=None, channel=channel,
                             run_dir=tmp_path, start_at=_AT, seed=42, active_users=len(people), workers=workers,
                             dialogue_turns=dialogue_turns, judge_users=1, table_groups=True, table_shop=shop)


def _ready_asap_game(channel, players):
    """构造由四条报名状态支持的原桌，开场时间留给桌群协商。"""
    store = channel.runtime.store
    host, *guests = players
    with using_clock(lambda: _AT - timedelta(minutes=1)):
        game, _ = store.create_game(conversation_id=channel.private_conversation_id(host.sender_id),
                                   organizer_id=host.sender_id, organizer_name=host.name,
                                   known_players=[], trace_id=f"asap-request:{host.sender_id}",
                                   requirement={"game_type": "hangzhou_mahjong", "stake": 1,
                                                "start_time_kind": "asap_when_full", "duration_hours": 4})
        for person in guests:
            store.join_game(game_id=game.game_id, customer_id=person.sender_id, display_name=person.name,
                            trace_id=f"join:{person.sender_id}")
    ready = store.require_game(game.game_id)
    assert ready.status == GameStatus.READY and ready.planned_start_at is None
    assert channel.table_group(VenueTableGroup.room_id_for_game(game.game_id)) is None
    return ready


def _rolling_fixture(tmp_path, *, actor, shop, dialogue_turns=4):
    scenario = _rolling_opening(tmp_path, people=_PEOPLE, actor=actor, shop=shop, workers=5,
                                dialogue_turns=dialogue_turns)
    return scenario, _ready_asap_game(scenario.channel, _PEOPLE[:4])


def test_ready_asap_table_rolls_through_a_revised_opening_without_losing_votes(tmp_path):
    """小群真实回执触发各人回复，拒绝后留空，收齐新版四票只写一次。"""
    shop = _OpeningShop()
    scenario, ready = _rolling_fixture(tmp_path, actor=_OpeningActor(seed=42), shop=shop)
    channel = scenario.channel
    room_id = VenueTableGroup.room_id_for_game(ready.game_id)
    report = scenario.run()
    assert all(report["checks"].values())
    assert report["rolling_table_groups"]["opened"] == 1
    assert report["rolling_table_groups"]["shop_receipts"] == 8
    group = channel.table_group(room_id)
    first, revised = (f"test-opening:{group.game_id}:{version}" for version in ("first", "revised"))
    for proposal_id in (first, revised):
        recipients = {event.actor_id for event in scenario.journal.events()
                      if event.kind == "table_receipt" and event.caused_by == proposal_id}
        assert recipients == _PLAYERS
        assert {row[1] for row in shop.votes if row[0] == proposal_id} == _PLAYERS
    assert (first, _DECLINER, False, "declined") in shop.votes
    assert sorted(row[3] for row in shop.votes if row[0] == revised) == ["committed", "waiting", "waiting", "waiting"]
    assert not channel.events_for(_BYSTANDER)
    confirmation, = (row for row in channel.events() if row["message_id"].endswith(":confirmed"))
    assert confirmation["recipient_id"] == room_id and confirmation["source_message_id"]

    persisted = SQLiteAgentStore(channel.runtime.store.path)
    updated = persisted.require_game(ready.game_id)
    assert updated.status == GameStatus.READY and updated.seat_claims() == ready.seat_claims()
    assert (updated.planned_start_at, updated.planned_end_at) == (_SECOND, _SECOND + timedelta(hours=4))
    assert sum(row.reason == "venue_table_opening_confirmed" for row in persisted.transitions) == 1
    assert scenario.run()["rolling_table_groups"] == report["rolling_table_groups"]
    assert len(shop.votes) == 8


_AVAILABILITY = "晚上能来，具体几点可以商量。"


class _InquiryActor(MockActor):
    """提议之前先回答店家询问时间，之后只对实际收件表态。"""

    def __init__(self, seed, *, decliners=frozenset({_DECLINER})):
        super().__init__(seed)
        self.decliners = decliners

    def _choose(self, context, projected, event, rng):
        delivery = projected["delivery"] or {}
        if delivery.get("sender_id") != AGENT_ID or delivery.get("recipient_id") not in context.group_targets:
            return ActorAction("wait")
        identifier = delivery["message_id"]
        if identifier.startswith("table-opening-inquiry:"):
            text = _AVAILABILITY
        elif identifier.startswith(f"table-opening-offer|{_FIRST.isoformat()}|"):
            text = _EARLY_NO if context.persona.sender_id in self.decliners else _EARLY_YES
        elif identifier.startswith(f"table-opening-offer|{_SECOND.isoformat()}|"):
            text = _LATE_YES
        else:
            return ActorAction("wait")
        return ActorAction("send", text, "group", delivery["recipient_id"], identifier)


class _AttestedShopClient:
    """只给真实桌群店家处理器提供本对照预先核准的判词。"""

    def __init__(self, *, hosts=frozenset({_HOST})):
        self.hosts = hosts

    def complete(self, messages, *, trace_id, timeout_seconds):
        """依据已送达来话和提议返回标注，无外部模型调用。"""
        payload = json.loads(messages[-1]["content"])
        current = payload["current"]
        vote, start = "other", None
        if current["text"] == _AVAILABILITY and current["sender_id"] in self.hosts and payload["active_proposal"] is None:
            start = _FIRST.isoformat()
        elif current["text"] == _EARLY_NO:
            assert payload["active_proposal"]["start_at"] == _FIRST.isoformat()
            vote, start = "decline", _SECOND.isoformat()
        elif current["text"] in {_EARLY_YES, _LATE_YES}:
            expected = _FIRST if current["text"] == _EARLY_YES else _SECOND
            assert payload["active_proposal"]["start_at"] == expected.isoformat()
            vote = "accept"
        return json.dumps({"vote": vote, "proposed_start": start})


@pytest.mark.parametrize("dialogue_turns", [2, 4])
def test_table_rounds_deliver_a_revised_proposal_after_shop_inquiry(tmp_path, dialogue_turns):
    """普通聊天限两轮时，桌群仍能询问、提议、处理拒绝并收到改约后的四份确认。"""
    shop = VenueTableShop(_AttestedShopClient(), tmp_path / "table_shop.jsonl")
    scenario, ready = _rolling_fixture(tmp_path, actor=_InquiryActor(seed=42), shop=shop, dialogue_turns=dialogue_turns)
    report = scenario.run()
    channel = scenario.channel
    group = channel.table_group(VenueTableGroup.room_id_for_game(ready.game_id))
    shop_ids = [receipt.message.message_id for receipt in group.delivered_receipts()
                if receipt.message.sender_id == AGENT_ID]
    assert shop_ids[0] == f"table-opening-inquiry:{ready.game_id}"
    first, revised = [identifier for identifier in shop_ids if identifier.startswith("table-opening-offer|")]
    shop_receipts = [event for event in scenario.journal.events() if event.kind == "table_receipt"]
    for identifier, depth in ((revised, 3), (first, 2), (shop_ids[0], 1)):
        wakes = [event for event in shop_receipts if event.caused_by == identifier]
        assert {event.actor_id for event in wakes} == _PLAYERS
        assert {event.details["depth"] for event in wakes} == {depth}
    assert report["rolling_table_groups"]["shop_receipts"] == 12
    assert all(report["checks"].values())
    game = channel.runtime.store.require_game(ready.game_id)
    assert game.planned_start_at == _SECOND and game.seat_claims() == ready.seat_claims()
    assert VenueTableOpening(group).current()["accepted_count"] == 4
    assert shop_ids[-1] == f"table-opening-confirm|{revised}"
    assert not channel.events_for(_BYSTANDER)


def test_fifty_actors_negotiate_twelve_isolated_table_groups(tmp_path):
    """50 人容量场的 12 桌并行定时，另两人旁观。重开后核对同局收件和写入唯一。"""
    people = generate_personas(42, count=50)
    rosters = [people[offset:offset + 4] for offset in range(0, 48, 4)]
    roles = _InquiryActor(seed=42, decliners={roster[-1].sender_id for roster in rosters})
    shop = VenueTableShop(_AttestedShopClient(hosts={roster[0].sender_id for roster in rosters}),
                         tmp_path / "table_shop.jsonl")
    options = {"people": people, "actor": roles, "shop": shop, "workers": 8, "dialogue_turns": 4}
    scenario = _rolling_opening(tmp_path, **options)
    games = [_ready_asap_game(scenario.channel, roster) for roster in rosters]
    report = scenario.run()
    assert all(report["checks"].values())
    assert report["rolling_table_groups"] == {
        "opened": 12, "durable_group_messages": 192, "role_receipts": 288, "shop_receipts": 144,
    }
    receipts_before = scenario.channel.events()

    restored = _rolling_opening(tmp_path, **options)
    assert restored.run()["rolling_table_groups"] == report["rolling_table_groups"]
    assert restored.channel.events() == receipts_before
    store = restored.channel.runtime.store
    for ready, roster in zip(games, rosters, strict=True):
        group = restored.channel.table_group(VenueTableGroup.room_id_for_game(ready.game_id))
        members = {person.sender_id for person in roster}
        assert group.member_ids() == {AGENT_ID, *members}
        expected = {row["message_id"] for row in receipts_before if row.get("table_game_id") == ready.game_id}
        assert len(expected) == 16
        for member in members:
            assert {row["message_id"] for row in restored.channel.events_for(member)} == expected
        opening = VenueTableOpening(group).current()
        assert opening["accepted_count"] == 4 and opening["committed"]
        assert store.require_game(ready.game_id).planned_start_at == _SECOND
    assert all(not restored.channel.events_for(person.sender_id) for person in people[-2:])
    assert sum(row.reason == "venue_table_opening_confirmed" for row in store.transitions) == 12


def test_confirmed_replacement_resumes_the_same_table_with_four_new_votes(tmp_path):
    """旧版已有三票才换人，重开账本后新阵容重谈且四人各自重新确认。"""
    fresh_availability = "我七点半以后有空，大家呢？"
    replacements = (_PLAYERS - {_DECLINER}) | {_BYSTANDER}
    class _ReplacementActor(_InquiryActor):
        def _choose(self, context, projected, event, rng):
            delivery = projected["delivery"] or {}
            identifier = delivery.get("message_id", "")
            if identifier == "previous-roster-inflight":
                return ActorAction("send", "我按原话回一下。", "group", delivery["recipient_id"], identifier)
            action = super()._choose(context, projected, event, rng)
            if action.action != "send":
                return action
            if identifier.startswith(f"table-opening-offer|{_FIRST.isoformat()}|") and context.persona.sender_id == _DECLINER:
                return ActorAction("wait")
            if identifier.startswith("table-opening-inquiry:") and "|table-roster:" in identifier:
                if context.persona.sender_id != _BYSTANDER:
                    return ActorAction("wait")
                return ActorAction("send", fresh_availability, "group", delivery["recipient_id"], identifier)
            return action

    class _ReplacementClient(_AttestedShopClient):
        def complete(self, messages, *, trace_id, timeout_seconds):
            payload = json.loads(messages[-1]["content"])
            if payload["current"]["text"] == fresh_availability:
                assert payload["current"]["sender_id"] == _BYSTANDER and payload["active_proposal"] is None
                assert len(payload["history"]) == 2
                return json.dumps({"vote": "other", "proposed_start": _SECOND.isoformat()})
            return super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)

    shop = VenueTableShop(_ReplacementClient(), tmp_path / "table_shop.jsonl")
    interrupted = False
    replaced = False

    def replace_after_three_votes(group, reply, at):
        nonlocal interrupted, replaced
        if reply is None and group.roster_revision(at=at) and not interrupted:
            interrupted = True
            raise RuntimeError("interrupt after durable roster reconciliation")
        message = shop(group, reply, at)
        active = VenueTableOpening(group).current()
        if not replaced and active and not active["committed"] and active["accepted_count"] == 3:
            replaced = True
            old_message = group.send(MessageEnvelope("previous-roster-inflight", AGENT_ID, "等会儿我再确认一下。",
                                                      at, "group", group.room_id)).message
            scenario.journal.enqueue(actor_id=_HOST, kind="table_receipt", origin_id=old_message.message_id,
                due_at=at + timedelta(seconds=5), caused_by=old_message.message_id,
                details={"table_room_id": group.room_id, "table_shop_delivery": True, "table_roster_revision": None,
                         "depth": 3})
            group.store.record_candidate_reply(game_id=group.game_id, customer_id=_DECLINER, display_name=_DECLINER,
                                               status="declined", trace_id="fourth-seat-withdrawn")
            group.store.join_game(game_id=group.game_id, customer_id=_BYSTANDER, display_name=_BYSTANDER,
                                  trace_id="named-replacement-confirmed")
            assert VenueTableOpening(group).current() is None
        return message

    scenario, initial = _rolling_fixture(tmp_path, actor=_ReplacementActor(seed=42), shop=replace_after_three_votes,
                                        dialogue_turns=2)
    with pytest.raises(RuntimeError, match="interrupt after durable roster reconciliation"):
        scenario.run()
    assert replaced and interrupted
    archive = SQLiteAgentStore(scenario.channel.runtime.store.path)
    runtime, _ = build_runtime(archive, "mock")
    scenario.channel = SandboxChannel(runtime, room_id="rolling-opening-public", archive_path=tmp_path / "chat.sqlite3",
                                      identities={person.sender_id: person.name for person in _PEOPLE})
    room_id = VenueTableGroup.room_id_for_game(initial.game_id)
    group = scenario.channel.table_group(room_id)
    assert group.member_ids() == replacements | {AGENT_ID}
    old = group.delivered_receipts()
    assert all(row.roster_revision is None for row in old)
    assert all(group.received(_BYSTANDER, row.message.message_id) is None for row in old)
    assert any(group.received(_DECLINER, row.message.message_id) for row in old)
    assert archive.require_game(initial.game_id).planned_start_at is None

    report = scenario.run()
    assert all(report["checks"].values())
    (superseded,) = [row.result for row in scenario.journal.events() if row.origin_id == "previous-roster-inflight"]
    assert superseded["runtime_status"] == "table_roster_superseded"
    assert "message_id" not in superseded and superseded["superseded_message_id"]
    assert report["rolling_roles"]["superseded_table_sends"] == 1
    assert not any(row["text"] == "我按原话回一下。" for row in scenario.channel.events())
    result = report["rolling_table_groups"]
    assert result["opened"] == 1
    roster, = result["roster_reconciliations"]
    assert roster["admitted"] == [_BYSTANDER] and roster["removed"] == [_DECLINER]
    assert set(roster["member_ids"]) == replacements | {AGENT_ID}
    assert group.member_ids(at=_AT) == _PLAYERS | {AGENT_ID}
    assert group.delivered_receipts()[:len(old)] == old
    fresh = group.delivered_receipts()[len(old):]
    assert all(row.roster_revision == roster["revision_id"] and _DECLINER not in row.recipient_ids for row in fresh)
    offer, = [row for row in fresh if row.message.message_id.startswith("table-opening-offer|")]
    assert set(offer.recipient_ids) == replacements
    votes = [row for row in fresh if row.message.quoted_message_id == offer.message.message_id]
    assert len(votes) == len(replacements) == 4 and {row.message.sender_id for row in votes} == replacements
    assert all(row.message.text == _LATE_YES for row in votes)
    assert VenueTableOpening(group).current()["accepted_count"] == 4
    committed = archive.require_game(initial.game_id)
    assert committed.planned_start_at == _SECOND and committed.seat_summary()["claimed_seats"] == 4
    assert sum(row.reason == "venue_table_opening_confirmed" for row in archive.transitions) == 1
    assert scenario.run()["rolling_table_groups"] == result
