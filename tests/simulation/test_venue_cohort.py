"""验证小批场景只种植初始库存，并以真实通道、工具和持久业务状态验收。"""

from dataclasses import replace
from datetime import datetime
import json

import pytest

from mahjong_agent_runtime.models import DEFAULT_TZ
from mahjong_agent_runtime.stores.sqlite import SQLiteAgentStore
from scripts.summarize_venue_run import summary

from .sim_adapter import build_runtime
from .venue_actors import ActorAction, MockActor, ModelActor, generate_personas
from .venue_channel import AGENT_ID, SandboxChannel
from .venue_cohort import VenueCohort, _opening_query_covers_window, _scoped_waitlist_effect
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
        assert {session.person.preferred_game for session in other_customers} == {"四川麻将", "国标麻将"}
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
                details = row["decision"]["context"]["events"][-1]["details"]
                assert (details["earliest_start_time"], details["latest_start_time"]) == ("19:00", "21:00")
                assert "start_time" not in details
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


@pytest.mark.parametrize("change", [
    None, ("authorization", "sender_id", "other"), ("authorization", "conversation_id", "other-private"),
    ("authorization", "source_message_id", "last-inbound"), ("authorization", "status", "invalid"),
    ("permission", "scope", "organize"), ("permission", "evidence", "只想查已有的牌局"),
    ("prior_reply", "recipient_id", "other"), ("reply", "source_message_id", "unrelated"),
    ("tool", "called", False), ("created", "sender_id", "other"), ("stored", "demand", {}),
    ("recorded", "sender_id", "other"), ("recorded", "status", "cancelled"),
])
def test_mixed_waitlist_transition_requires_personal_scope_and_persisted_effect(change):
    """只有本人的已交付续轮、授权和实际有效候补能够改变预期的无匹配终态。"""
    person, conversation = "guest", "guest-private"
    message = dict(kind="private", direction="inbound", sender_id=person, recipient_id=AGENT_ID,
                   conversation_id=conversation, text="有合适的请留意后通知我")
    prior_reply = dict(kind="private", direction="outbound", sender_id=AGENT_ID, recipient_id=person,
                       conversation_id=conversation, source_message_id="last-inbound")
    reply = {**prior_reply, "source_message_id": "current"}
    permission = {"scope": "waitlist", "evidence": "请留意后通知我"}
    authorization = dict(event="authorization", status="authorized", sender_id=person, conversation_id=conversation,
                         source_message_id="current", permissions=[permission])
    created = dict(demand_id="demand", sender_id=person, conversation_id=conversation, status="active")
    recorded = dict(created)
    stored = {"demand": recorded}
    tool = dict(name="register_waiting_demand", called=True, allowed=True, result={"waiting_demand": created})
    turn = dict(actor_id=person, message_id="current", caused_by="previous", reply_message_id="receipt",
                tool_results=[tool])
    if change:
        part, key, value = change
        pieces = dict(authorization=authorization, permission=permission, prior_reply=prior_reply, reply=reply,
                      tool=tool, created=created, stored=stored, recorded=recorded)
        pieces[part][key] = value

    result = _scoped_waitlist_effect(
        person, [turn], {"current": message, "previous": prior_reply, "receipt": reply}, [authorization], stored,
    )

    assert result == ("demand" if change is None else None)


def test_model_customer_can_stop_after_one_grounded_range_query(tmp_path):
    """顾客查完整范围后可以沉默；仍要求本次实际查局，且未写入任何新局。"""
    class OneQuestionActor(ModelActor):
        def decide(self, context, *, trace_id):
            decision = MockActor(42).decide(context, trace_id=trace_id)
            action = ActorAction("wait") if context.events[-1].get("caused_by") else ActorAction(
                "send", "老板，今晚七点到九点之间有现成的麻将局吗？", "private", AGENT_ID,
            )
            return replace(decision, action=action, source="model", response_source={"http_response_complete": True})

    people = generate_personas(42)
    store = SQLiteAgentStore(tmp_path / "business.sqlite3")
    fixture = VenueFixtureAgent(seed=42)
    runtime, _ = build_runtime(store, "mock", offline_client=fixture, local_coordination=True)
    channel = SandboxChannel(runtime, room_id="test-group", identities={person.sender_id: person.name for person in people},
                             archive_path=tmp_path / "chat.sqlite3")
    cohort = VenueCohort(
        people=people, case="no-match", actor=OneQuestionActor(None), judge=None, channel=channel, store=store,
        run_dir=tmp_path, start_at=datetime(2026, 11, 25, 19, tzinfo=DEFAULT_TZ), seed=42,
        active_users=1, workers=1, dialogue_turns=3, background_events=0,
    )
    customer_id = cohort.sessions[0].person.sender_id
    fixture.turns[f"customer:talk:{customer_id}:1"] = ("dialogue_followup", {})

    report = cohort.run()

    assert all(report["checks"].values()), report
    assert report["acceptance"]["multi_round_private_customers"] == 0
    assert report["acceptance"]["outcome_counts"] == {"no_match": 1}
    assert len(channel.events()) == 2 and not store.games
    assert [row["decision"]["action"]["action"] for row in cohort.sessions[0].turns] == ["send", "wait"]


@pytest.mark.parametrize("queries, expected", [
    ([{"start_time": "19:00"}], False),
    ([{"planned_start_at": "2026-11-25T19:00:00+08:00"}], False),
    ([{"start_time": "19:00"}, {"start_time": "20:00"}, {"start_time": "21:00"}], False),
    ([{"game_type": "sichuan_mahjong"}], True),
    ([{"start_time_range": {"start_at": "2026-11-25T19:00:00+08:00",
                            "end_at": "2026-11-25T21:00:00+08:00"}}], True),
    ([{"start_time_range": {"start_at": "2026-11-25T19:00:00+08:00",
                            "end_at": "2026-11-25T20:00:00+08:00"}}], False),
])
def test_mocked_empty_point_queries_do_not_prove_range_coverage(queries, expected):
    """复用真实查询合同字段核对范围；空结果和成功的工具调用都不能替代范围证据。"""
    tools = [
        {"name": "search_current_games", "called": True, "allowed": True,
         "result": {"requirement": query, "matches": [], "alternatives": []}}
        for query in queries
    ]
    start = datetime(2026, 11, 25, 19, tzinfo=DEFAULT_TZ)
    assert _opening_query_covers_window([{"tool_results": tools}], start) is expected
    assert not _opening_query_covers_window([{}, {"tool_results": tools}], start)
