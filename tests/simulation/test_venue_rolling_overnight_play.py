"""用四份桌群确认和四小时物理占房验证跨午夜续跑。"""

from datetime import datetime, timedelta
import json

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import GameStatus, using_clock

from .sim_adapter import build_runtime
from .venue_actors import ActorAction, MockActor, generate_personas
from .venue_channel import SandboxChannel
from .venue_rolling_roles import VenueRollingRoles
from .venue_scenario import ACTUAL_START_DELAY, PLAY_DURATION, VENUE
from .venue_table_group import VenueTableGroup
from .venue_table_opening import TABLE_OPENING_CONFIRM_PREFIX, TABLE_OPENING_OFFER_PREFIX, VenueTableOpening
from .venue_table_shop import VenueTableShop


_RUN_START = datetime.fromisoformat("2026-11-27T18:00:00+08:00")
_PEOPLE = generate_personas(42, count=5)
_HOST = _PEOPLE[0].sender_id
_INTENT, _YES = "今晚这段我方便，大家可以定时间。", "这版时间可以。"


class _NightRoles(MockActor):
    """用明确标注的离线选择触发真实收件、定时和签到。"""

    def _choose(self, context, projected, event, rng):
        delivery = projected["delivery"] or {}
        identifier = delivery.get("message_id", "")
        if event["kind"] in {"arrival", "departure"}:
            assert identifier.startswith(TABLE_OPENING_CONFIRM_PREFIX)
            return ActorAction("arrive" if event["kind"] == "arrival" else "leave")
        if identifier.startswith("table-opening-inquiry:") and context.persona.sender_id == _HOST:
            body = _INTENT
        elif identifier.startswith(TABLE_OPENING_OFFER_PREFIX):
            body = _YES
        else:
            return ActorAction("wait")
        return ActorAction("send", body, "group", delivery["recipient_id"], identifier)


class _AttestedOpening:
    """只解释当轮实际发言，不替换送达或写入。"""

    def __init__(self, start):
        self.start = start

    def complete(self, messages, **options):
        body = json.loads(messages[-1]["content"])
        current = body["current"]
        if current["text"] == _INTENT:
            return json.dumps({"vote": "other", "proposed_start": self.start.isoformat()})
        assert current["text"] == _YES and body["active_proposal"]["start_at"] == self.start.isoformat()
        return '{"vote":"accept","proposed_start":null}'


def _run(tmp_path, *, planned):
    """把下一日的已满桌交给原始两日滚动入口，完整使用现有店家处理器。"""
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    store.configure_rooms([room["id"] for room in VENUE["rooms"]])
    runtime, _ = build_runtime(store, "mock")
    channel = SandboxChannel(runtime, room_id="overnight-public", archive_path=tmp_path / "chat.sqlite3",
                             identities={person.sender_id: person.name for person in _PEOPLE})
    filled_at = planned - timedelta(minutes=75)
    with using_clock(lambda: filled_at):
        game, _ = store.create_game(conversation_id=channel.private_conversation_id(_HOST), organizer_id=_HOST,
            organizer_name=_PEOPLE[0].name, known_players=[], trace_id="overnight-seat-1",
            requirement={"game_type": "hangzhou_mahjong", "stake": 1, "smoke_preference": "no_smoking",
                         "start_time_kind": "asap_when_full", "duration_hours": 4})
        for person in _PEOPLE[1:4]:
            game, _ = store.join_game(game_id=game.game_id, customer_id=person.sender_id, display_name=person.name,
                                     trace_id=f"overnight-seat:{person.sender_id}")
    runner = VenueRollingRoles(people=_PEOPLE, actor=_NightRoles(seed=42), judge=None, channel=channel, run_dir=tmp_path,
        start_at=_RUN_START, seed=42, active_users=5, workers=4, dialogue_turns=2, judge_users=1,
        table_groups=True, table_shop=VenueTableShop(_AttestedOpening(planned), tmp_path / "shop.jsonl"), physical_play=True)
    return runner, game.game_id


@pytest.mark.parametrize("closes", ["06:00", "04:00", "12:00"])
def test_same_table_can_arrive_before_midnight_and_play_four_hours_in_configured_venue(tmp_path, monkeypatch, closes):
    monkeypatch.setitem(VENUE, "closes", closes)
    planned = (_RUN_START + timedelta(days=1)).replace(hour=23, minute=57)
    runner, game_id = _run(tmp_path, planned=planned)
    result = runner.run()
    assert all(result["checks"].values())
    group = runner.channel.table_group(VenueTableGroup.room_id_for_game(game_id))
    assert VenueTableOpening(group).current()["accepted_count"] == 4
    physical = result["rolling_play"]
    assert physical["status"] == "passed" and physical["arrival_plans"] == 1
    assert physical["verified_arrival_decisions"] == physical["verified_departure_decisions"] == 4
    assert physical["occupied_one_minute_before_end"] == physical["four_hour_finishes"] == physical["same_room_rebookable"] == 1
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    game = store.require_game(game_id)
    assert game.status == GameStatus.FINISHED and game.planned_start_at == planned
    assert game.actual_started_at == planned + ACTUAL_START_DELAY
    assert game.actual_ended_at == game.actual_started_at + PLAY_DURATION
    third_day = (_RUN_START + timedelta(days=2)).date()
    assert game.actual_started_at.date() == third_day
    assert {event.due_at.astimezone(_RUN_START.tzinfo).date() for event in runner.journal.events() if event.kind == "timer"} == {
        _RUN_START.date(), (_RUN_START + timedelta(days=1)).date(),
    }
    assert store.room_reservations[game.play_reservation_id].status == "released"
    starts, finishes = [row for row in physical["evidence"] if row["kind"] in {"start", "finish"}]
    assert {row["at"][:10] for row in starts["decision"]["arrivals"].values()} == {planned.date().isoformat()}
    availability = finishes["result"]["availability"]
    next_opening = game.actual_ended_at.replace(hour=12) if closes == "04:00" else game.actual_ended_at
    assert datetime.fromisoformat(availability["start_at"]) == next_opening
    assert not group.events_for(_PEOPLE[-1].sender_id)
    assert runner.run()["rolling_play"] == physical


def test_default_closing_still_blocks_actual_play_even_if_the_planned_slot_fit(tmp_path):
    planned = (_RUN_START + timedelta(days=1)).replace(hour=19, minute=48)
    runner, game_id = _run(tmp_path, planned=planned)
    report = runner.run()
    group = runner.channel.table_group(VenueTableGroup.room_id_for_game(game_id))
    assert VenueTableOpening(group).current()["committed"]
    physical = report["rolling_play"]
    assert physical["arrival_plans"] == physical["actually_started"] == 0
    assert not runner.channel.runtime.store.room_reservations
