"""验证小批场景只种植初始库存，并以真实通道、工具和持久业务状态验收。"""

from datetime import datetime
import json

import pytest

from mahjong_agent_runtime.models import DEFAULT_TZ
from mahjong_agent_runtime.stores.sqlite import SQLiteAgentStore
from scripts.summarize_venue_run import summary

from .sim_adapter import build_runtime
from .venue_actors import MockActor, generate_personas
from .venue_channel import SandboxChannel
from .venue_cohort import VenueCohort
from .venue_fixture import VenueFixtureAgent


@pytest.mark.parametrize("case, customers, expected", [
    ("formation", 1, {"formed": 1}),
    ("no-match", 2, {"no_match": 2}),
    ("mixed", 3, {"formed": 1, "no_match": 2}),
])
def test_independent_cohort_verifies_archive_causes_and_persisted_outcomes(tmp_path, case, customers, expected):
    """离线合同桩重放真实业务接线，入座前必须先从本人会话获取确认。"""
    people = generate_personas(42)
    store_path = tmp_path / "business.sqlite3"
    store = SQLiteAgentStore(store_path)
    fixture = VenueFixtureAgent(seed=42)
    runtime, _ = build_runtime(store, "mock", offline_client=fixture, local_coordination=True)
    channel = SandboxChannel(runtime, room_id="test-group", identities={person.sender_id: person.name for person in people},
                             archive_path=tmp_path / "chat.sqlite3")
    cohort = VenueCohort(
        people=people, case=case, actor=MockActor(42), judge=None, channel=channel, store=store, run_dir=tmp_path,
        start_at=datetime(2026, 11, 25, 19, tzinfo=DEFAULT_TZ), seed=42, active_users=customers, workers=customers,
        dialogue_turns=2, background_events=0, fixture_agent=fixture,
    )
    report = cohort.run()
    assert all(report["checks"].values()), report
    assert report["acceptance"]["outcome_counts"] == expected
    assert report["acceptance"]["multi_round_private_customers"] == customers
    brief = summary(report)
    assert brief["cohort"]["expected_counts"] == expected
    assert "venue-sender-" not in json.dumps(brief)
    reopened = SQLiteAgentStore(store_path)
    if "formed" in expected:
        initial = report["cohort"]["initial_game"]
        customer_id = cohort.sessions[0].person.sender_id
        assert initial["remaining_seats"] == 1 and customer_id not in {row["customer_id"] for row in initial["participants"]}
        final = reopened.require_game(initial["game_id"])
        assert final.status.value == "ready" and final.remaining_seats() == 0
        assert {person.status for person in final.participants} == {"confirmed"}
        assert set(reopened.games) == {initial["game_id"]}
    else:
        assert not reopened.games
    timeline = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert all(
        "case_intent" in row["decision"]["context"]["events"][-1]["details"]
        and "game_id" not in row["decision"]["context"]["events"][-1]["details"]
        for row in timeline
    )
    assert len([row for row in timeline if row["caused_by"]]) == customers
    if case == "mixed":
        claimant, *other_customers = cohort.sessions
        assert len({session.person.speaking_style for session in cohort.sessions}) == customers
        assert report["peak_inflight_deliveries"] >= 2
        assert report["cohort"]["expected_outcomes"] == {
            claimant.person.sender_id: "formed", **{session.person.sender_id: "no_match" for session in other_customers},
        }
        for row in timeline:
            phase, options = fixture.turns[row["message_id"]]
            if row["actor_id"] == claimant.person.sender_id:
                assert phase in {"cohort_offer", "cohort_join"}
            else:
                assert phase == "cohort_no_match"
                assert options["game_type"] in {"sichuan_mahjong", "national_standard_mahjong"}
                assert f"今天只考虑{row['decision']['context']['persona']['preferred_game']}" in (
                    row["decision"]["context"]["events"][-1]["details"]["case_intent"]
                )
                assert initial["game_id"] not in json.dumps(row["decision"]["context"])


def test_mixed_dialogues_without_fixture_never_register_offline_phases(tmp_path, monkeypatch):
    """传入无合同桩的角色时，编排器不得读取任何离线模型动作。"""
    people = generate_personas(42)
    store = SQLiteAgentStore(tmp_path / "business.sqlite3")
    runtime, _ = build_runtime(store, "mock", local_coordination=True)
    channel = SandboxChannel(runtime, room_id="test-group", identities={person.sender_id: person.name for person in people},
                             archive_path=tmp_path / "chat.sqlite3")

    def offline_phase_forbidden(*args):
        raise AssertionError("引导真实角色时不能读取离线合同")

    monkeypatch.setattr(VenueCohort, "_fixture_turn", offline_phase_forbidden)
    cohort = VenueCohort(
        people=people, case="mixed", actor=MockActor(42), judge=None, channel=channel, store=store, run_dir=tmp_path,
        start_at=datetime(2026, 11, 25, 19, tzinfo=DEFAULT_TZ), seed=42, active_users=3, workers=3,
        dialogue_turns=2, background_events=0, fixture_agent=None,
    )
    assert cohort.run()["chat_events"] >= 3
