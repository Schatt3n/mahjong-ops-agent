"""用本机角色替身验证两日滚动唤醒、消息隔离与断点恢复。"""

from collections import Counter
from datetime import datetime
import json
from threading import Barrier
from time import sleep

import pytest

from mahjong_agent_runtime import SQLiteAgentStore

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
