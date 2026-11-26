"""用真实建群、调度收件和 SQLite 盘面验证人齐开后的两轮协商。"""

from datetime import datetime, timedelta

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import GameStatus, using_clock

from .sim_adapter import build_runtime
from .venue_actors import ActorAction, MockActor, generate_personas
from .venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel
from .venue_rolling_roles import VenueRollingRoles
from .venue_table_group import VenueTableGroup
from .venue_table_opening import VenueTableOpening


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


def test_ready_asap_table_rolls_through_a_revised_opening_without_losing_votes(tmp_path):
    """小群真实回执触发各人回复，拒绝后留空，收齐新版四票只写一次。"""
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    runtime, _ = build_runtime(store, "mock")
    channel = SandboxChannel(runtime, room_id="rolling-opening-public", archive_path=tmp_path / "chat.sqlite3",
                             identities={person.sender_id: person.name for person in _PEOPLE})
    with using_clock(lambda: _AT - timedelta(minutes=1)):
        game, _ = store.create_game(conversation_id=channel.private_conversation_id(_HOST), organizer_id=_HOST,
                                   organizer_name=_PEOPLE[0].name, known_players=[], trace_id="asap-request",
                                   requirement={"game_type": "hangzhou_mahjong", "stake": 1,
                                                "start_time_kind": "asap_when_full", "duration_hours": 4})
        for person in _PEOPLE[1:4]:
            store.join_game(game_id=game.game_id, customer_id=person.sender_id, display_name=person.name,
                            trace_id=f"join:{person.sender_id}")
    ready = store.require_game(game.game_id)
    assert ready.status == GameStatus.READY and ready.planned_start_at is None
    room_id = VenueTableGroup.room_id_for_game(game.game_id)
    assert channel.table_group(room_id) is None
    shop = _OpeningShop()
    scenario = VenueRollingRoles(people=_PEOPLE, actor=_OpeningActor(seed=42), judge=None, channel=channel,
                                 run_dir=tmp_path, start_at=_AT, seed=42, active_users=5, workers=5, dialogue_turns=4,
                                 judge_users=1, table_groups=True, table_shop=shop)
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

    persisted = SQLiteAgentStore(store.path)
    updated = persisted.require_game(game.game_id)
    assert updated.status == GameStatus.READY and updated.seat_claims() == ready.seat_claims()
    assert (updated.planned_start_at, updated.planned_end_at) == (_SECOND, _SECOND + timedelta(hours=4))
    assert sum(row.reason == "venue_table_opening_confirmed" for row in persisted.transitions) == 1
    assert scenario.run()["rolling_table_groups"] == report["rolling_table_groups"]
    assert len(shop.votes) == 8
