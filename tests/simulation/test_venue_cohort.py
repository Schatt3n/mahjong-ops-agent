"""验证小批场景只种植初始库存，并以真实通道、工具和持久业务状态验收。"""

from datetime import datetime
import json

import pytest

from mahjong_agent_runtime.models import DEFAULT_TZ
from mahjong_agent_runtime.stores.sqlite import SQLiteAgentStore

from .sim_adapter import build_runtime
from .venue_actors import MockActor, generate_personas
from .venue_channel import SandboxChannel
from .venue_cohort import VenueCohort
from .venue_fixture import VenueFixtureAgent


@pytest.mark.parametrize("case, customers, expected", [("formation", 1, "formed"), ("no-match", 2, "no_match")])
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
    assert report["acceptance"]["outcome_counts"] == {expected: customers}
    assert report["acceptance"]["multi_round_private_customers"] == customers
    reopened = SQLiteAgentStore(store_path)
    if case == "formation":
        initial = report["cohort"]["initial_game"]
        customer_id = cohort.sessions[0].person.sender_id
        assert initial["remaining_seats"] == 1 and customer_id not in {row["customer_id"] for row in initial["participants"]}
        final = reopened.require_game(initial["game_id"])
        assert final.status.value == "ready" and final.remaining_seats() == 0
        assert {person.status for person in final.participants} == {"confirmed"}
    else:
        assert not reopened.games
    timeline = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert all(
        "case_intent" in row["decision"]["context"]["events"][-1]["details"]
        and "game_id" not in row["decision"]["context"]["events"][-1]["details"]
        for row in timeline
    )
    assert len([row for row in timeline if row["caused_by"]]) == customers
