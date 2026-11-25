"""以正式沙盒回执验证牌友能继续私聊，续跑不会泄漏或重复投递。"""

from collections import Counter
from dataclasses import replace
from datetime import datetime

import pytest

from mahjong_agent_runtime import SQLiteAgentStore

from .sim_adapter import build_runtime
from .venue_actors import ActorAction, MockActor, Relationship, generate_personas
from .venue_channel import AGENT_ID, OWNER_ID, MessageEnvelope, SandboxChannel
from .venue_rolling_roles import VenueRollingRoles


_START = datetime.fromisoformat("2026-11-24T20:00:00+08:00")
_PEER_FIRST = "你今晚还想打无烟桌吗？"
_PEER_REPLY = "我八点以后才到，你先问问。"
_OWNER_QUERY = "老板今晚有没有余位？"


class _PeerConversationActor(MockActor):
    """让受控角色从自己的目标清单和收到的来话决定动作。"""

    def __init__(self, people):
        """前三位各负责发问、续话和另行找老板，第四位仅观察。"""
        super().__init__(seed=42)
        self.initiator, self.partner, self.owner_caller, self.observer = (person.sender_id for person in people)
        self.calls = Counter()

    def decide(self, context, *, trace_id):
        """记录实际决定次数，以便判断进程重开是否复用了原决定。"""
        self.calls[trace_id] += 1
        return super().decide(context, trace_id=trace_id)

    def _choose(self, context, projected, event, rng):
        """用同一角色合同提交牌友消息，收到回执后才续话。"""
        sender = context.persona.sender_id
        delivery = projected["delivery"] or {}
        if (delivery.get("text") == _PEER_FIRST and sender == self.partner
                and event.get("details", {}).get("peer_id") == self.initiator and self.initiator in context.private_targets):
            return ActorAction("send", _PEER_REPLY, "private", self.initiator)
        if context.now.startswith(_START.date().isoformat()) and not delivery:
            if (sender == self.initiator and self.partner in context.private_targets
                    and not any(row.get("text") == _PEER_FIRST for row in projected["history"])):
                return ActorAction("send", _PEER_FIRST, "private", self.partner)
            if sender == self.owner_caller and event["kind"] == "activity":
                return ActorAction("send", _OWNER_QUERY, "private", AGENT_ID)
        return ActorAction("wait")


def _people():
    """模拟搭子关系，避桌对象与旁观者不具备私发的关系入口。"""
    initiator, partner, owner_caller, observer = generate_personas(42, count=4)
    return (
        replace(initiator, relationships=(
            Relationship(partner.persona_id, partner.name, "regular_partner", partner.sender_id),
            Relationship(observer.persona_id, observer.name, "avoid_same_table", observer.sender_id),
        )),
        replace(partner, relationships=()),
        replace(owner_caller, relationships=()),
        replace(observer, relationships=()),
    )


def _open(tmp_path, people, actor, *, peer_dialogues, active_users=None):
    """所有轮次使用真实的群身份、独立私聊、SQLite 和持久事件入口。"""
    runtime, _ = build_runtime(SQLiteAgentStore(tmp_path / "agent.sqlite3"), "mock")
    channel = SandboxChannel(
        runtime, room_id="rolling-peer-room", archive_path=tmp_path / "chat.sqlite3",
        identities={person.sender_id: person.name for person in people},
    )
    return VenueRollingRoles(
        people=people, actor=actor, judge=None, channel=channel, run_dir=tmp_path,
        start_at=_START, seed=42, active_users=len(people) if active_users is None else active_users, workers=2, dialogue_turns=3,
        judge_users=1, peer_dialogues=peer_dialogues,
    )


def test_peer_receipt_resumes_same_private_conversation_without_waking_bystanders(tmp_path, monkeypatch):
    """在渠道提交与唤醒之间中断，验证登记后的来话只能唤醒真实收件人。"""
    class Interruption(BaseException):
        """模拟渠道提交后进程硬中断。"""

    people = _people()
    initiator, partner, owner_caller, observer = (person.sender_id for person in people)
    first_actor = _PeerConversationActor(people)
    initial = _open(tmp_path, people, first_actor, peer_dialogues=True)
    original_dispatch = initial._dispatch
    interrupted = {}

    def stop_after_peer_dispatch(event):
        """档案与渠道回执已落盘，事件分发仍未执行。"""
        result, receipts = original_dispatch(event)
        if (event.decision["decision"].get("action") or {}).get("text") == _PEER_FIRST:
            assert result["runtime_status"] == "delivered"
            interrupted.update(event_id=event.event_id, message_id=result["message_id"])
            raise Interruption
        return result, receipts

    with monkeypatch.context() as patch:
        patch.setattr(initial, "_dispatch", stop_after_peer_dispatch)
        with pytest.raises(Interruption):
            initial.run()
    prior, = [row for row in initial.channel.events() if row["direction"] == "peer"]
    assert prior["message_id"] == interrupted["message_id"]
    assert not initial.channel.runtime.trace_recorder.get_trace(prior["trace_id"])
    assert not any(event.caused_by == prior["message_id"] for event in initial.journal.events())

    resumed_actor = _PeerConversationActor(people)
    reopened = _open(tmp_path, people, resumed_actor, peer_dialogues=True)
    report = reopened.run()
    assert all(report["checks"].values()), report["checks"]
    assert not report["failed_events"]
    assert report["rolling_peers"] == {
        "status": "observed", "delivered_messages": 2, "receipt_wakes": 2, "causal_replies": 1,
        "private_conversations": 1,
    }
    assert report["rolling_roles"]["sent_channels"] == {"private": 1}
    assert report["rolling_roles"]["wake_kinds"]["private_receipt"] == 1
    assert report["rolling_roles"]["actors_with_prior_day_private_receipts_in_context"] == 1
    assert not set(first_actor.calls) & set(resumed_actor.calls)
    assert reopened.journal.get(interrupted["event_id"]).result["delivery_deduplicated"]

    first, reply = [row for row in reopened.channel.events() if row["direction"] == "peer"]
    assert first == prior
    assert (first["sender_id"], first["recipient_id"], first["text"]) == (initiator, partner, _PEER_FIRST)
    assert (reply["sender_id"], reply["recipient_id"], reply["text"]) == (partner, initiator, _PEER_REPLY)
    assert first["conversation_id"] == reply["conversation_id"]
    assert first["conversation_id"] not in {reopened.channel.private_conversation_id(person.sender_id) for person in people}
    assert reopened.channel.events_for(initiator) == reopened.channel.events_for(partner) == [first, reply]
    assert not reopened.channel.events_for(observer)
    for other in (AGENT_ID, OWNER_ID, owner_caller):
        own = reopened.channel.events_for(other)
        assert own and any(row["text"] == _OWNER_QUERY for row in own)
        assert not any(row["direction"] == "peer" for row in own)

    events = reopened.journal.events()
    for message in (first, reply):
        wake, = [event for event in events if event.caused_by == message["message_id"]]
        assert wake.actor_id == message["recipient_id"] and wake.kind == "peer_receipt" and wake.status == "finished"
        assert wake.decision["visible_cause_verified"]
        context = wake.decision["decision"]["context"]
        assert context["events"][-1]["kind"] == "delivered"
        assert context["events"][-1]["details"]["peer_id"] == message["sender_id"]
        assert message["sender_id"] in context["private_targets"]
        delivery = context["delivery"]
        assert (delivery["message_id"], delivery["sender_id"], delivery["recipient_id"], delivery["text"]) == (
            message["message_id"], message["sender_id"], message["recipient_id"], message["text"],
        )
    events_by_actor = [event for event in events if event.actor_id in {observer, owner_caller}]
    assert not any(row["text"] in {_PEER_FIRST, _PEER_REPLY} for event in events_by_actor
                   for row in event.decision["decision"]["context"]["history"])
    opening = reopened.journal.get(interrupted["event_id"]).decision["decision"]["context"]
    assert partner in opening["private_targets"] and observer not in opening["private_targets"]

    store = reopened.channel.runtime.store
    assert not store.recent_turns(first["conversation_id"])
    assert not any(store.recent_turns(reopened.channel.private_conversation_id(role)) for role in (initiator, partner, observer))
    assert store.recent_turns(reopened.channel.private_conversation_id(owner_caller))
    assert not reopened.channel.runtime.trace_recorder.get_trace(first["trace_id"])
    assert not reopened.channel.runtime.trace_recorder.get_trace(reply["trace_id"])

    archive = reopened.channel.events()
    final_actor = _PeerConversationActor(people)
    final = _open(tmp_path, people, final_actor, peer_dialogues=True)
    final.run()
    assert final.channel.events() == archive and not final_actor.calls
    assert final.journal.events() == events


def test_peer_option_off_keeps_existing_owner_chat_and_no_peer_routes(tmp_path):
    """未启用牌友入口时，名册关系不会扩大角色可用收件人。"""
    people = _people()
    actor = _PeerConversationActor(people)
    scenario = _open(tmp_path, people, actor, peer_dialogues=False)
    report = scenario.run()
    assert all(report["checks"].values()) and not report["failed_events"]
    assert "rolling_peers" not in report
    assert not any(event.kind == "peer_receipt" for event in scenario.journal.events())
    for event in scenario.journal.events():
        assert event.decision["decision"]["context"]["private_targets"] == [AGENT_ID]
    assert all(row["direction"] in {"inbound", "outbound"} for row in scenario.channel.events())
    assert any(row["text"] == _OWNER_QUERY for row in scenario.channel.events())


def test_unfinished_peer_receipt_stays_out_of_both_inboxes(tmp_path, monkeypatch):
    """消息已入档但回执未办结时，双方均不能用该内容继续聊天。"""
    people = _people()
    sender, recipient, *_ = (person.sender_id for person in people)
    scenario = _open(tmp_path, people, _PeerConversationActor(people), peer_dialogues=True)
    channel = scenario.channel

    def interrupted_finish(message_id, response):
        """模拟消息提交后、最终回执写入前故障。"""
        raise OSError("controlled receipt write interruption")

    monkeypatch.setattr(channel, "_finish", interrupted_finish)
    with pytest.raises(OSError, match="controlled receipt write interruption"):
        channel.send(MessageEnvelope("interrupted-peer", sender, _PEER_FIRST, _START, "private", recipient))
    archived, = channel.events()
    assert (archived["direction"], archived["recipient_id"]) == ("peer", recipient)
    assert channel.completed_peer_message(archived["message_id"]) is None
    assert channel.events_for(sender) == channel.events_for(recipient) == []


def test_peer_targets_require_an_active_regular_partner(tmp_path):
    """仍在群名册的熟人需要本次有角色实例，才会进入发送目标。"""
    people = _people()
    scenario = _open(tmp_path, people, _PeerConversationActor(people), peer_dialogues=True, active_users=1)
    report = scenario.run()
    assert all(report["checks"].values()) and not report["failed_events"]
    assert report["rolling_peers"]["status"] == "not_observed"
    assert not scenario.channel.events()
    for event in scenario.journal.events():
        assert event.decision["decision"]["context"]["private_targets"] == [AGENT_ID]
