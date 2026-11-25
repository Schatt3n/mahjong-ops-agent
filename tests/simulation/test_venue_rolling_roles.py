"""用本机角色替身验证两日滚动唤醒、消息隔离与断点恢复。"""

from collections import Counter
from datetime import datetime, timedelta
import json
from threading import Barrier
from time import sleep

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import Game, GameParticipant, GameStatus, RoomReservation, WaitingDemand

from . import venue_rolling_roles
from .sim_adapter import build_runtime
from .venue_actors import ActorAction, MockActor, generate_personas
from .venue_channel import AGENT_ID, SandboxChannel
from .venue_rolling_roles import VenueRollingRoles


class _MixedLocalActor(MockActor):
    """定时唤醒在群里或私聊开口，收到后来答复后主动保持沉默。"""

    def __init__(self, group_speaker=None):
        """按固定的小样本演员分配群聊与私聊。"""
        self.group_speaker = group_speaker
        self.calls = Counter()

    def decide(self, context, *, trace_id):
        """收集决定次数，给并发线程足够时间在同一波运行。"""
        self.calls[trace_id] += 1
        sleep(0.02)  # 允许其他线程处在同一决策波，验证真实重叠而非并发配置值。
        return super().decide(context, trace_id=trace_id)

    seed = 42

    def _choose(self, context, projected, event, rng):
        """在定时回合发送，其余事件只确认演员看到来话。"""
        if event["kind"] != "activity":
            return ActorAction("wait")
        if context.persona.sender_id == self.group_speaker:
            return ActorAction("send", "大家晚上好", "group", context.group_targets[0])
        return ActorAction("send", "老板还有位吗", "private", AGENT_ID)


def _scenario(tmp_path, *, people, actor, workers=3, scenario_class=VenueRollingRoles):
    """重建相同的正式业务入口与独立频道，从磁盘打开已有运行证据。"""
    runtime, client = build_runtime(SQLiteAgentStore(tmp_path / "agent.sqlite3"), "mock")
    client.semantic_intent = "chitchat"
    channel = SandboxChannel(runtime, room_id="rolling-room", archive_path=tmp_path / "chat.sqlite3",
                             identities={person.sender_id: person.name for person in people})
    scenario = scenario_class(people=people, actor=actor, judge=None, channel=channel, run_dir=tmp_path,
        start_at=datetime.fromisoformat("2026-11-24T20:00:00+08:00"), seed=42,
        active_users=len(people), workers=workers, dialogue_turns=2, judge_users=1)
    return scenario, client


def test_daily_timers_group_attention_and_private_inboxes_stay_separate(tmp_path, monkeypatch):
    """群内分配注意力不挤占本人私聊，同波决定看不到这波稍后的投递。"""
    people = generate_personas(42)[:3]
    actor = _MixedLocalActor(group_speaker=people[0].sender_id)
    scenario, _ = _scenario(tmp_path, people=people, actor=actor)
    send = scenario.channel.send
    first_wave = Barrier(2)

    def overlap_first_private_messages(message):
        """两个私聊必须同时进入正式入口，才允许继续首次送达。"""
        if message.kind == "private" and message.sent_at == scenario.start_at:
            first_wave.wait(timeout=2)
        return send(message)

    monkeypatch.setattr(scenario.channel, "send", overlap_first_private_messages)
    report = scenario.run()
    rolling = report["rolling_roles"]
    assert all(report["checks"].values()), report["checks"]
    assert report["peak_inflight_decisions"] == 3
    assert report["peak_inflight_deliveries"] >= 2
    assert rolling["actors_with_two_calendar_days"] == rolling["actors_with_own_prior_day_messages_in_context"] == 3
    assert rolling["actors_with_prior_day_private_receipts_in_context"] == 2
    assert rolling["sent_channels"] == {"private": 4, "group": 2}
    assert rolling["wake_kinds"] == {"timer": 6, "private_receipt": 4, "group_attention": 4, "rain": 1, "traffic": 1}
    assert rolling["actions"]["wait"] == 10
    assert rolling["cross_channel_replies"] == {"group_to_private": 0, "private_to_group": 0}
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    first_day_timers = [row for row in events if row["wake_kind"] == "timer" and row["virtual_at"].startswith("2026-11-24")]
    assert len(first_day_timers) == 3 and all(not row["decision"]["context"]["history"] for row in first_day_timers)
    for row in (item for item in events if item["wake_kind"] == "private_receipt"):
        assert row["decision"]["context"]["delivery"]["recipient_id"] == row["actor_id"]
    reopened, _ = _scenario(tmp_path, people=people, actor=actor)
    assert reopened.run()["rolling_roles"] == rolling
    assert set(actor.calls.values()) == {1}


def test_reopen_after_dispatch_reuses_saved_actor_decision_and_completed_channel_receipt(tmp_path):
    """进程丢失办结写入时，原决定和正式收件只执行一次。"""
    class _InterruptingScenario(VenueRollingRoles):
        """保留真正的渠道发送，只故障注入事件办结步骤。"""

        def _dispatch(self, event):
            """首个真实回执成功后退出，给恢复过程留下原决定。"""
            result, receipts = super()._dispatch(event)
            if result.get("message_id"):
                raise RuntimeError("断在真回执和事件办结之间")
            return result, receipts

    people = generate_personas(42)[:1]
    old_actor = _MixedLocalActor()
    initial, _ = _scenario(tmp_path, people=people, actor=old_actor, workers=1, scenario_class=_InterruptingScenario)
    with pytest.raises(RuntimeError, match="事件办结之间"):
        initial.run()
    _, outbound = initial.channel.events()
    new_actor = _MixedLocalActor()
    reopened, _ = _scenario(tmp_path, people=people, actor=new_actor, workers=1)
    report = reopened.run()
    assert all(report["checks"].values()), report["checks"]
    assert report["rolling_roles"]["reused_channel_receipts"] == 1
    assert not set(old_actor.calls) & set(new_actor.calls)
    assert [row for row in reopened.channel.events() if row["message_id"] == outbound["message_id"]] == [outbound]


def test_report_counts_store_seats_and_flags_overfilled_tool_snapshots(tmp_path, monkeypatch):
    """席位按报名人数计，最终存储和成功工具返回的超额证据分别保留。"""
    people = generate_personas(42)[:1]
    scenario, _ = _scenario(tmp_path, people=people, actor=_MixedLocalActor())
    at = scenario.start_at
    participant = GameParticipant("contact", "访客", status="confirmed", seat_count=2)
    current = Game("game_1", "chat", "contact", "访客", {}, status=GameStatus.INVITING, participants=[participant])
    held = RoomReservation("room_1", "label", "chat", None, at, at + timedelta(hours=1))
    waiting = WaitingDemand("demand_1", "chat", "contact", "访客", {}, at)
    # 状态快照替身不需要通过业务接口制造非法库存。工具证据与最终存储可能来自不同回合。
    monkeypatch.setattr(SQLiteAgentStore, "games", property(lambda self: {current.game_id: current}))
    monkeypatch.setattr(SQLiteAgentStore, "room_reservations", property(lambda self: {held.reservation_id: held}))
    monkeypatch.setattr(SQLiteAgentStore, "waiting_demands", property(lambda self: {waiting.demand_id: waiting}))
    overfill = {"game": {"seat_summary": {"seats_total": 4, "claimed_seats": 5}}}
    successful = {"name": "search_current_games", "called": True, "allowed": True, "result": {"matches": [overfill]}}
    row = {"actor_id": people[0].sender_id, "virtual_at": at.isoformat(), "wake_kind": "timer",
        "decision": {"action": {"action": "wait"}, "context": {}}, "visible_cause_verified": True,
        "tool_results": [successful, {**successful, "deduplicated": True}, {**successful, "error": "tool runtime failure"},
                         {**successful, "allowed": False}]}
    report = scenario._report((), [row], [])
    business = report["rolling_roles"]["business_state"]
    assert business["games_by_status"] == {"inviting": 1}
    assert business["seats"] == {"tables": 1, "capacity": 4, "claimed": 2, "vacant": 2, "full_tables": 0,
                                 "overfilled_tables": 0, "overflow_seats": 0}
    assert business["waiting_demands_by_status"] == {"active": 1}
    assert business["reservations_by_status"] == {"held": 1}
    assert business["successful_tool_executions"] == 1
    assert business["successful_tools_by_name"] == {"search_current_games": 1}
    assert business["tool_game_snapshots"] == {"checked": 1, "overfilled": 1}
    assert not report["checks"]["no_overfilled_table_in_store_or_tool_evidence"]
    assert not {"game_1", "contact", "room_1", "demand_1"} & set(json.dumps(business).split('"'))

    participant.seat_count = 5
    report = scenario._report((), [{**row, "tool_results": []}], [])
    seats = report["rolling_roles"]["business_state"]["seats"]
    assert seats["overfilled_tables"] == seats["overflow_seats"] == 1
    assert not report["checks"]["no_overfilled_table_in_store_or_tool_evidence"]


def test_semantic_sampling_covers_rare_actual_strata_and_reports_unsampled_roles(tmp_path, monkeypatch):
    """部分覆盖可重复，预算覆盖全部时逐人判定，摘要只包含类别计数。"""
    people = generate_personas(42)[:4]
    scenario, _ = _scenario(tmp_path, people=people, actor=_MixedLocalActor())
    scenario.judge = object()
    scenario.judge_users = 3
    calls = []

    def local_review(judge, *, customer_id, **kwargs):
        """不发真实模型请求，保留已抽和有效判决的统计差别。"""
        calls.append(customer_id)
        if customer_id == people[3].sender_id:
            return {"status": "invalid", "verdict": "inconclusive"}
        return {"status": "valid", "verdict": "pass"}

    monkeypatch.setattr(venue_rolling_roles, "review_dialogue", local_review)
    rows = {person.sender_id: [
        {"wake_kind": kind, "channel": channel, "message_id": f"message-{index}",
         "decision": {"action": {"action": action}, "context": {"events": [{"details": {"case_goal": goal}}]}}}
    ] for index, (person, kind, channel, action, goal) in enumerate(zip(people,
        ("timer", "timer", "group_attention", "traffic"), ("private", "private", "group", None),
        ("send", "send", "send", "wait"), ("lookup", "lookup", "organize_new", "wait_or_leave"), strict=True))}
    delivered = {"message-0", "message-1", "message-2"}
    _, partial = scenario._review(rows, [], delivered)
    assert partial["eligible_actors"] == 4 and partial["sampled_actors"] == 3 and partial["missing_sample_actors"] == 1
    assert partial["valid_judgment_actors"] == 2 and partial["actors_without_valid_judgment"] == 2
    assert partial["outcomes"] == {"pass": 2, "unavailable": 1}
    assert partial["strata"]["channel"]["group"] == {"observed_actors": 1, "sampled_actors": 1, "valid_judgment_actors": 1}
    assert partial["strata"]["channel"]["private"]["sampled_actors"] == 1
    assert partial["strata"]["event"]["traffic"] == {"observed_actors": 1, "sampled_actors": 1, "valid_judgment_actors": 0}
    assert partial["strata"]["action"]["wait"]["observed_actors"] == 1
    assert partial["strata"]["role_intent"]["organize_new"]["sampled_actors"] == 1
    assert all(person.sender_id not in json.dumps(partial) for person in people)
    selection = tuple(calls)
    calls.clear()
    assert scenario._review(dict(reversed(rows.items())), [], delivered)[1] == partial
    assert tuple(calls) == selection
    calls.clear()
    scenario.judge_users = 8
    _, complete = scenario._review(rows, [], delivered)
    assert set(calls) == set(rows)
    assert complete["sampled_actors"] == 4 and complete["missing_sample_actors"] == 0


def test_cross_channel_replies_require_an_archived_send_and_visible_cause(tmp_path):
    """只统计真实因果唤醒后的跨频道发送，定时开聊和未送达动作不得冒充联动。"""
    people = generate_personas(42)[:1]
    scenario, _ = _scenario(tmp_path, people=people, actor=_MixedLocalActor())
    actor_id, at = people[0].sender_id, scenario.start_at.isoformat()
    records = [
        {"message_id": "public-cause", "kind": "group", "sender_id": "other", "recipient_id": scenario.channel.room_id},
        {"message_id": "personal-cause", "kind": "private", "sender_id": AGENT_ID, "recipient_id": actor_id},
        {"message_id": "foreign-cause", "kind": "private", "sender_id": AGENT_ID, "recipient_id": "other"},
    ]
    rows = []
    for index, (wake, channel, cause) in enumerate((
        ("group_attention", "private", "public-cause"), ("private_receipt", "group", "personal-cause"),
        ("timer", "private", None), ("group_attention", "private", "public-cause"),
        ("private_receipt", "group", "foreign-cause"), ("group_attention", "private", "personal-cause"),
    )):
        recipient = AGENT_ID if channel == "private" else scenario.channel.room_id
        message_id = f"msg-{index}"
        rows.append({"actor_id": actor_id, "virtual_at": at, "wake_kind": wake, "caused_by": cause,
            "decision": {"action": {"action": "send", "text": "想问一下", "recipient_id": recipient}, "context": {}},
            "visible_cause_verified": True, "channel": channel, "message_id": message_id, "channel_trace_id": "trace"})
        if index != 3:
            records.append({"message_id": message_id, "text": "想问一下", "kind": channel, "sender_id": actor_id,
                            "recipient_id": recipient})
    report = scenario._report((), rows, records)
    assert report["rolling_roles"]["cross_channel_replies"] == {"group_to_private": 1, "private_to_group": 1}
