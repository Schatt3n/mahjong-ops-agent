"""从已送达的桌群协商接续同行签到、四小时占用和散场。"""

from dataclasses import replace
from datetime import timedelta

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import GameStatus, using_clock
from scripts.run_venue_table_opening import _CompanionProbe, _prepare_companion

from .sim_adapter import build_runtime
from .test_venue_rolling_late_opening import _FrozenReviewer, _ReplayActor, _REVISED
from .venue_actors import ActorAction
from .venue_channel import SandboxChannel
from .venue_rolling_play_link import PLAY_ROLE_KINDS
from .venue_scenario import ACTUAL_START_DELAY, PLAY_DURATION, VENUE
from .venue_table_group import VenueTableGroup
from .venue_table_opening import TABLE_OPENING_CONFIRM_PREFIX, VenueTableOpening
from .venue_table_shop import VenueTableShop


class _PhysicalRoles(_ReplayActor):
    """各位角色只按自己的实际确认收件作签到和四小时离店决定。"""

    def __init__(self, seed, *, unproven=False, wait=False):
        super().__init__(seed)
        self.unproven, self.wait = unproven, wait

    def decide(self, context, *, trace_id):
        decision = super().decide(context, trace_id=trace_id)
        if self.unproven and decision.action == ActorAction("accept_companion"):
            return replace(decision, source="no_delivery")
        return decision

    def _choose(self, context, projected, event, rng):
        if event["kind"] in PLAY_ROLE_KINDS:
            receipt = projected["delivery"]
            assert receipt["kind"] == "group" and receipt["message_id"].startswith(TABLE_OPENING_CONFIRM_PREFIX)
            assert "game_id" not in event["details"] and "table_room_id" not in event["details"]
            if self.wait and context.persona.sender_id == "venue-sender-0001":
                return ActorAction("wait")
            return ActorAction("arrive" if event["kind"] == "arrival" else "leave")
        return super()._choose(context, projected, event, rng)


def _fixture(tmp_path, case):
    at, seed, people, channel, game_id, _, checks = _prepare_companion(tmp_path, unavailable=False)
    assert all(checks.values())
    channel.runtime.store.configure_rooms([room["id"] for room in VENUE["rooms"]])
    shop = VenueTableShop(_FrozenReviewer(), tmp_path / "shop.jsonl")

    def dispatch_shop(group, reply, sent_at):
        message = shop(group, reply, sent_at)
        if message and message.message_id.startswith(TABLE_OPENING_CONFIRM_PREFIX):
            if case == "missing_confirmation":
                return None
            if case == "unconfirmed_text":
                return replace(message, text="开场还没定，我再问一下。")
        return message

    def scenario(channel):
        return _CompanionProbe(people=people, actor=_PhysicalRoles(seed, unproven=case == "unproven_acceptance", wait=case == "wait"),
            judge=None, channel=channel, run_dir=tmp_path, start_at=at, seed=seed, active_users=len(people), workers=3,
            dialogue_turns=2, judge_users=1, table_groups=True, table_shop=dispatch_shop, physical_play=True)

    return game_id, scenario(channel), scenario, people


def test_companion_votes_physical_actions_and_room_release_all_use_the_same_game(tmp_path):
    game_id, runner, scenario, people = _fixture(tmp_path, "valid")
    report = runner.run()["rolling_play"]
    assert report["status"] == "passed"
    assert report["participation_routes"] == {"table_companion": 1, "table_opening": 3}
    assert report["verified_arrival_decisions"] == report["verified_departure_decisions"] == 4
    assert report["occupied_one_minute_before_end"] == report["four_hour_finishes"] == report["same_room_rebookable"] == 1
    plan, = report["participation_evidence"]
    assert plan["game_id"] == game_id and set(plan["participants"]) == {person.sender_id for person in people[:-1]}
    assert plan["participants"]["venue-sender-0001"]["accepted_invitation_id"].startswith("table-invite:")
    assert len({proof["table_receipt_id"] for proof in plan["participants"].values()}) == 1
    store = SQLiteAgentStore(runner.channel.runtime.store.path)
    game = store.require_game(game_id)
    assert game.requirement.get("duration_kind") is None and game.requirement["duration_hours"] == 4
    assert game.status == GameStatus.FINISHED and game.planned_start_at == _REVISED
    assert game.actual_started_at == _REVISED + ACTUAL_START_DELAY and game.actual_ended_at - game.actual_started_at == PLAY_DURATION
    group = runner.channel.table_group(VenueTableGroup.room_id_for_game(game_id))
    assert not group.events_for(people[-1].sender_id)

    runtime, _ = build_runtime(store, "mock")
    channel = SandboxChannel(runtime, room_id=runner.channel.room_id, archive_path=runner.channel.archive_path,
                             identities={person.sender_id: person.name for person in people})
    assert scenario(channel).run()["rolling_play"] == report


@pytest.mark.parametrize("case", ["missing_confirmation", "unconfirmed_text", "unproven_acceptance", "wait"])
def test_table_cannot_start_from_only_votes_or_an_unproven_arrival(tmp_path, case):
    game_id, runner, _, _ = _fixture(tmp_path, case)
    report = runner.run()["rolling_play"]
    group = runner.channel.table_group(VenueTableGroup.room_id_for_game(game_id))
    assert VenueTableOpening(group).current()["committed"]
    if case == "wait":
        assert report["arrival_plans"] == 1 and report["verified_arrival_decisions"] == 3
    else:
        assert report["arrival_plans"] == report["verified_arrival_decisions"] == 0
    game = runner.channel.runtime.store.require_game(game_id)
    assert not game.actual_started_at and report["actually_started"] == 0


def test_same_four_names_cannot_reuse_a_confirmed_opening_after_a_seat_changes(tmp_path):
    game_id, runner, _, people = _fixture(tmp_path, "valid")
    link, runner.play = runner.play, None
    runner.run()
    now = runner.journal.clock() + timedelta(minutes=1)
    store = runner.channel.runtime.store
    member = people[2]
    with using_clock(lambda: now):
        store.record_candidate_reply(game_id=game_id, customer_id=member.sender_id, display_name=member.name,
                                     status="declined", trace_id="left-after-a-confirmed-opening")
        store.join_game(game_id=game_id, customer_id=member.sender_id, display_name=member.name,
                        trace_id="same-member-new-confirmation")
    group = runner.channel.table_group(VenueTableGroup.room_id_for_game(game_id))
    assert VenueTableOpening(group).current()["committed"]
    assert VenueTableOpening(group).confirmed_attendance(at=now) is None
    with runner.journal.run_owner():
        link.sync(now)
    assert link.report()["arrival_plans"] == 0
