"""用本机角色替身验证两日滚动唤醒、消息隔离与断点恢复。"""

from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta
import json
from sqlite3 import OperationalError
from threading import Barrier
from time import sleep

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import Game, GameParticipant, GameStatus, RoomReservation, WaitingDemand, using_clock

from . import venue_rolling_roles
from .sim_adapter import build_runtime
from .venue_actors import ActorAction, MockActor, ModelActor, Relationship, generate_personas
from .venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel
from .venue_judging import AGENT_CHECK
from .venue_role_journal import VenueRoleJournal
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


def _scenario(tmp_path, *, people, actor, workers=3, scenario_class=VenueRollingRoles, publish_boards=False,
              active_users=None, peer_dialogues=False, start_at=None):
    """重建相同的正式业务入口与独立频道，从磁盘打开已有运行证据。"""
    runtime, client = build_runtime(SQLiteAgentStore(tmp_path / "agent.sqlite3"), "mock")
    client.semantic_intent = "chitchat"
    channel = SandboxChannel(runtime, room_id="rolling-room", archive_path=tmp_path / "chat.sqlite3",
                             identities={person.sender_id: person.name for person in people})
    scenario = scenario_class(people=people, actor=actor, judge=None, channel=channel, run_dir=tmp_path,
        start_at=start_at or datetime.fromisoformat("2026-11-24T20:00:00+08:00"), seed=42,
        active_users=active_users or len(people), workers=workers, dialogue_turns=2, judge_users=1,
        publish_boards=publish_boards, peer_dialogues=peer_dialogues)
    return scenario, client


@pytest.mark.parametrize(("depth", "at", "status", "wakes", "reply", "expected"), [
    (1, "2026-11-25T11:07:00+08:00", "waiting_user", [], "shop-last", "dialogue_turn_limit"),
    (0, "2026-11-25T23:59:00+08:00", "waiting_user", [], "shop-last", "observation_day_limit"),
    (0, "2026-11-25T11:07:00+08:00", "waiting_user", [], "shop-last", None),
    (1, "2026-11-25T11:07:00+08:00", "waiting_user", ["wake"], "shop-last", None),
    (1, "2026-11-25T19:37:00+08:00", "completed", [], "shop-last", "dialogue_turn_limit"),
    (0, "2026-11-25T23:59:00+08:00", "completed", [], "shop-last", "observation_day_limit"),
    (0, "2026-11-25T11:07:00+08:00", "completed", [], "shop-last", None),
    (1, "2026-11-25T11:07:00+08:00", "completed", ["wake"], "shop-last", None),
    (1, "2026-11-25T11:07:00+08:00", "completed", None, "shop-last", None),
    (1, "2026-11-25T11:07:00+08:00", "completed", [], None, None),
    (1, "2026-11-25T11:07:00+08:00", "needs_help", [], "shop-last", None),
    (1, "2026-11-25T11:07:00+08:00", "failed", [], "shop-last", None),
])
def test_archived_private_followup_observation_requires_actual_scheduler_limit(tmp_path, depth, at, status, wakes, reply, expected):
    """从真实成功回复的调度结果恢复上限依据，失败、有唤醒或无回执不记截断。"""
    people = generate_personas(42)[:1]
    scenario, _ = _scenario(tmp_path, people=people, actor=_MixedLocalActor())
    with scenario.journal.run_owner():
        scenario.journal.enqueue(actor_id=people[0].sender_id, kind="private_receipt", origin_id="shop-first",
                                 due_at=datetime.fromisoformat(at), details={"depth": depth})
        event, = scenario.journal.claim_wave(1)
        event = scenario.journal.save_decision(event, {"decision": {"status": "valid"}})
        scenario.journal.finish(event, {"runtime_status": status, "reply_message_id": reply,
                                       "channel": "private", "scheduled_wakes": wakes})
    reopened, _ = _scenario(tmp_path, people=people, actor=_MixedLocalActor())
    event, = reopened.journal.events()
    archived = reopened._archived_role_row(event)
    assert archived.get("private_followup_boundary") == expected
    assert event.result.get("private_followup_boundary") is None


def test_v6_timer_quotes_delivered_board_even_after_group_history_crowding(tmp_path, monkeypatch):
    """复现真实原稿和16条群历史压力，只为一个真实看板原件留位置。"""
    people = generate_personas(42)[:3]
    viewer, peer, organizer = people
    draft = "老板好 红中那桌下午两点后还有位吗 底2到3能打 烟况定了没"
    board_at = datetime.fromisoformat("2026-11-26T09:07:30+08:00")
    at = datetime.fromisoformat("2026-11-26T11:05:00+08:00")
    start = datetime.fromisoformat("2026-11-25T19:00:00+08:00")

    class FrozenActorClient:
        """按 V6 实际角色原稿发出选择，问题里的桌时间不算顾客到场保证。"""

        def complete(self, messages, **_):
            payload = json.loads(messages[-1]["content"])
            if "draft" in payload:
                assert payload["draft"] == draft
                return '{"claims":[]}'
            return json.dumps({"action": "send", "text": draft, "kind": "group", "recipient_id": room,
                               "quote_message_id": board["message_id"]}, ensure_ascii=False)

    scenario, _ = _scenario(tmp_path, people=people, actor=ModelActor(FrozenActorClient()), publish_boards=True,
                            start_at=start)
    channel, store = scenario.channel, scenario.channel.runtime.store
    room = channel.room_id
    with using_clock(lambda: board_at - timedelta(seconds=30)):
        store.create_game(conversation_id=channel.private_conversation_id(organizer.sender_id),
            organizer_id=organizer.sender_id, organizer_name=organizer.name, known_players=[], trace_id="v6-board-table",
            requirement={"game_type": "red_center_mahjong", "stake_range": {"min": 2, "max": 3},
                         "start_time_kind": "asap_when_full"})
    board, = channel.publish_board(trace_id="v6-delivery", sent_at=board_at)["archive_outputs"]
    assert "红中 2～3元 烟况待定 173" in board["text"]
    for index in range(8):
        receipt_at = board_at + timedelta(seconds=index + 1)
        with using_clock(lambda: receipt_at):
            channel.send(MessageEnvelope(f"v6-reference-{index}", peer.sender_id, "红中那桌烟况定了没？",
                                         receipt_at, "group", room, board["message_id"]))
        channel.publish_owner(kind="group", recipient_id=room, text="晚点再聊。", sent_at=receipt_at,
                              message_id=f"v6-followup-{index}")
    visible = [row for row in channel.events_for(viewer.sender_id) if row["kind"] == "group"]
    tail = visible[-16:]
    assert all(row["message_id"] != board["message_id"] for row in tail)
    assert any(row["quoted_message_id"] == board["message_id"] for row in tail)

    with scenario.journal.run_owner():
        timer = scenario.journal.enqueue(actor_id=viewer.sender_id, kind="timer", origin_id="v6-clock", due_at=at,
            details={"can_arrive": True, "earliest_arrival": "2026-11-26T11:14:00+08:00",
                     "free_from": "2026-11-26T11:00:00+08:00", "free_until": "2026-11-26T22:00:00+08:00"})
        claimed, = scenario.journal.claim_wave(1)
        assert timer.event_id == claimed.event_id
        with monkeypatch.context() as patch:
            patch.setattr(scenario, "group_members", scenario.group_members - {viewer.sender_id})
            assert board["message_id"] not in {row["message_id"] for row in scenario._history(claimed)}
        trigger = replace(claimed, kind="group_attention", caused_by=tail[0]["message_id"])
        with_trigger = {row["message_id"] for row in scenario._history(trigger)}
        assert {trigger.caused_by, board["message_id"]} <= with_trigger
        planned = scenario._plan(claimed)
        decision = planned.decision["decision"]
        assert decision["status"] == "valid" and decision["action"]["text"] == draft
        assert decision["action"]["quote_message_id"] == board["message_id"]
        claim_review, = decision["availability_reviews"]
        assert claim_review["claims"] == [] and claim_review["conflict"] is None
        context = decision["context"]
        assert context["delivery"] is None
        assert len([row for row in context["history"] if row["kind"] == "group"]) == 16
        result, _ = scenario._dispatch(planned)
        assert result.get("error_type") is None and result.get("channel_trace_id")
        scenario.journal.finish(planned, result)

    actual = next(row for row in channel.events() if row["message_id"] == result["message_id"])
    assert actual["text"] == draft and actual["quoted_message_id"] == board["message_id"]
    assert actual["recipient_id"] == room

    # 账本找不到原件、房间不一致、或送达晚于引用时均不给原件保留资格。
    assert scenario.board.latest_quoted_parent(tail, {}) is None
    assert scenario.board.latest_quoted_parent(tail, {
        board["message_id"]: {**board, "recipient_id": "foreign-group"},
    }) is None
    assert scenario.board.latest_quoted_parent(tail, {
        board["message_id"]: {**board, "sent_at": at.isoformat()},
    }) is None
    assert scenario.board.latest_quoted_parent(tail, {
        board["message_id"]: {**board, "text": "不吻合的看板原文"},
    }) is None


def test_board_delivery_wakes_roles_after_resume_without_republishing(tmp_path, monkeypatch):
    """发布已跨两库落盘但未办结时，重开只补原公开回执的角色唤醒。"""
    class Interruption(BaseException):
        """模拟中断，绕过正常业务异常处理。"""

    people = generate_personas(42)[:5]
    actor = _MixedLocalActor()
    scenario, _ = _scenario(tmp_path, people=people, actor=actor, publish_boards=True, active_users=4)
    organizer = people[0]
    start = scenario.start_at
    with using_clock(lambda: start):
        game, _ = scenario.channel.runtime.store.create_game(
            conversation_id=scenario.channel.private_conversation_id(organizer.sender_id),
            organizer_id=organizer.sender_id, organizer_name=organizer.name, trace_id="controlled-board-fixture",
            requirement={"game_type": "hangzhou_mahjong", "start_time": "21:00", "duration_kind": "fixed",
                         "duration_hours": 2, "planned_start_at": (start + timedelta(hours=1)).isoformat()},
            known_players=[{"customer_id": organizer.sender_id, "display_name": organizer.name}],
        )
    assert scenario.board.next_due() is not None
    real_finish = VenueRoleJournal.finish

    def stop_after_board(journal, event, result):
        """保证已归档看板，刻意不结算当前系统事件。"""
        if event.kind == "board_publication" and result["delivered_board_ids"]:
            raise Interruption
        return real_finish(journal, event, result)

    with monkeypatch.context() as patch:
        patch.setattr(VenueRoleJournal, "finish", stop_after_board)
        with pytest.raises(Interruption):
            scenario.run()

    board = scenario.channel.runtime.store.get_latest_board_snapshot(scenario.channel.room_id)
    assert board.items[0].game_id == game.game_id
    assert scenario.board.next_due() == game.planned_start_at + timedelta(microseconds=1)
    reopened, _ = _scenario(tmp_path, people=people, actor=actor, publish_boards=True, active_users=4)
    report = reopened.run()
    assert all(report["checks"].values()) and report["rolling_board"]["published_boards"] == 2
    assert report["rolling_board"]["publication_ticks"] == 2
    assert report["rolling_board"]["status"] == "not_observed"
    assert sum(row["message_id"] == board.external_message_id for row in reopened.channel.events()) == 1
    wakes = [event for event in reopened.journal.events() if event.caused_by == board.external_message_id]
    assert len(wakes) == 4
    assert {event.actor_id for event in wakes} == {person.sender_id for person in people[:4]}
    assert all(event.kind == "group_attention" and event.status == "finished" for event in wakes)
    assert all(event.decision["decision"]["context"]["delivery"]["message_id"] == board.external_message_id
               for event in wakes)
    assert all(actor.calls[event.decision["decision"]["trace_id"]] == 1 for event in wakes)
    assert board.external_message_id in {row["message_id"] for row in reopened.channel.events_for(people[4].sender_id)}
    assert report["rolling_roles"]["scheduled_events"] == len(reopened.journal.events()) - 2


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


@pytest.mark.parametrize(("disposition", "handling_status", "judge_verdict", "error_type"), [
    ("pass", "pending", "pass", None),
    ("uncited", "review_required", "inconclusive", None),
    ("fail", "failed", "fail", None),
    ("fallback", "failed", "pass", "runtime_human_review_fallback"),
    ("missing", "inconclusive", "pass", "runtime_human_review_not_delivered"),
    ("timeout", None, "pass", "timeout"),
    ("group_timeout", None, "pass", "timeout"),
])
def test_rolling_human_review_is_evaluated_using_actual_receipt(tmp_path, monkeypatch, disposition, handling_status,
                                                             judge_verdict, error_type):
    """有实质回执的人工确认进入店家评审，有真实超时则保留技术归因。"""
    people = generate_personas(42)[:1]
    group_speaker = people[0].sender_id if disposition == "group_timeout" else None
    scenario, _ = _scenario(tmp_path, people=people, actor=_MixedLocalActor(group_speaker), workers=1)
    scenario.judge = object()
    channel = scenario.channel
    original = channel._handle_private
    text = "这桌能不能晚到，得先和值班店员核实。"
    if disposition in {"fallback", "timeout", "group_timeout"}:
        text = "稍等一下"
    elif disposition == "missing":
        text = ""

    def handle(message, conversation_id, trace_id, quoted):
        if message.sent_at != scenario.start_at:
            return original(message, conversation_id, trace_id, quoted)
        if text:
            channel._record_output({"channel": "private", "recipient_id": message.sender_id,
                                    "conversation_id": conversation_id, "text": text}, message, trace_id)
        if disposition in {"timeout", "group_timeout"}:
            channel.runtime.trace_recorder.record(trace_id, "llm_error", {
                "completion_diagnostics": {"attempts": [{"error_type": "TimeoutError"}]},
            })
        return {"status": "needs_human", "trace_id": trace_id}, message

    original_group = channel._handle_group

    def handle_group(message, trace_id):
        if message.sent_at != scenario.start_at:
            return original_group(message, trace_id)
        result, source = handle(message, channel.private_conversation_id(message.sender_id), trace_id, None)
        return {**result, "group_processing": {"runtime_results": [{}]}}, source

    def review(judge, *, turns, records, **kwargs):
        event, = (row for row in turns if row.get("runtime_status") == "needs_human")
        result = {"status": "valid", "verdict": "fail" if disposition == "fail" else "pass", "findings": [], "evidence": []}
        if disposition in {"pass", "fail"}:
            output, = (row for row in records if row.get("source_message_id") == event["message_id"])
            result["findings"] = [{"check": AGENT_CHECK, "verdict": disposition, "evidence_ids": [output["message_id"]]}]
            result["evidence"] = [{"kind": "agent_reply", "evidence_id": output["message_id"]}]
        return result

    monkeypatch.setattr(channel, "_handle_private", handle)
    monkeypatch.setattr(channel, "_handle_group", handle_group)
    monkeypatch.setattr(venue_rolling_roles, "review_dialogue", review)
    report = scenario.run()
    targeted, = [event for event in scenario.journal.events() if (event.result or {}).get("runtime_status") == "needs_human"]
    assert targeted.result.get("error_type") == error_type
    assert report["failed_events"] == ([targeted.event_id] if error_type else [])
    assert bool(report["model_transport_errors"]) is (disposition in {"timeout", "group_timeout"})
    assert report["independent_judge"]["verdict"] == judge_verdict
    if handling_status is None:
        assert "human_review_handling" not in report
        return
    handling, = report["human_review_handling"]
    assert handling["status"] == handling_status and handling["message_id"] == targeted.result["message_id"]
    assert handling.get("delivered_reply_id") == targeted.result.get("reply_message_id")
    assert report["checks"]["human_review_requests_have_substantive_delivery"] is (disposition in {"pass", "uncited", "fail"})
    if disposition in {"pass", "uncited", "fail"}:
        strata = report["rolling_roles"]["semantic_sampling"]["strata"]["shop_handling"]
        assert strata["human_review_requested"]["sampled_actors"] == 1


@pytest.mark.parametrize("fault", ["after_inbound", "shop_store_error", "before_inbound", "wrong_inbound"])
def test_rolling_distinguishes_group_input_delivery_from_shop_processing_failure(tmp_path, monkeypatch, fault):
    """已入档的群来话仍唤醒同群角色，未入档或内容冲突不能补造成功收件。"""
    people = generate_personas(42)[:2]
    scenario, _ = _scenario(tmp_path, people=people, actor=_MixedLocalActor(group_speaker=people[0].sender_id))
    channel = scenario.channel
    send, handle = channel.send, channel._handle_group

    def group_handler(message, trace_id):
        if message.sent_at == scenario.start_at:
            if fault == "shop_store_error":
                raise OperationalError("模拟店家存储失败")
            raise RuntimeError("模拟店家群处理失败")
        return handle(message, trace_id)

    def channel_send(message):
        if message.kind == "group" and message.sent_at == scenario.start_at:
            if fault == "before_inbound":
                raise RuntimeError("模拟渠道未收到来话")
            if fault == "wrong_inbound":
                message = replace(message, text="同编号但不是角色原话")
        return send(message)

    monkeypatch.setattr(channel, "_handle_group", group_handler)
    monkeypatch.setattr(channel, "send", channel_send)
    report = scenario.run()
    committed = fault in {"after_inbound", "shop_store_error"}
    assert report["checks"]["role_sends_match_delivered_channel_rows"] is committed
    assert report["rolling_roles"]["sent_channels"]["group"] == (2 if committed else 1)
    failed, = [event for event in scenario.journal.events() if event.event_id in report["failed_events"]]
    result = failed.result
    assert result["error_type"] == ("OperationalError" if fault == "shop_store_error" else "RuntimeError")
    assert "模拟" in result["error"]
    archived = [row for row in channel.events() if row["message_id"] == result["message_id"]]
    wakes = [event for event in scenario.journal.events() if event.caused_by == result["message_id"]]
    assert not any(row["source_message_id"] == result["message_id"] for row in channel.events())
    if committed:
        receipt, = archived
        assert result["channel_trace_id"] == result["trace_id"] == receipt["trace_id"]
        assert result["runtime_status"] == "failed" and result["failure_stage"] == "agent"
        assert result["outputs"] == [] and result["reply_message_id"] is None
        assert receipt["text"] == failed.decision["decision"]["action"]["text"]
        wake, = wakes
        assert wake.kind == "group_attention" and wake.actor_id == people[1].sender_id
        assert wake.decision["decision"]["context"]["delivery"]["message_id"] == receipt["message_id"]
    else:
        assert len(archived) == (fault == "wrong_inbound")
        assert result["failure_stage"] == "delivery" and "channel_trace_id" not in result and not wakes


def test_rolling_does_not_wake_peer_from_unfinished_channel_receipt(tmp_path, monkeypatch):
    """牌友私信需要正式渠道完成收件，不能套用店家处理失败的入站记账。"""
    sender, partner = generate_personas(42)[:2]
    sender = replace(sender, relationships=(Relationship(partner.persona_id, partner.name, "regular_partner",
                                                         partner.sender_id),))

    class PeerActor(MockActor):
        def _choose(self, context, projected, event, rng):
            if context.persona == sender and context.now.startswith("2026-11-24") and event["kind"] == "activity":
                return ActorAction("send", "晚上打不打？", "private", partner.sender_id)
            return ActorAction("wait")

    scenario, _ = _scenario(tmp_path, people=[sender, partner], actor=PeerActor(42), peer_dialogues=True)
    finish = scenario.channel._finish

    def finish_peer(message_id, response):
        if response["status"] == "delivered":
            raise RuntimeError("模拟未完成牌友渠道回执")
        return finish(message_id, response)

    monkeypatch.setattr(scenario.channel, "_finish", finish_peer)
    report = scenario.run()
    receipt, = [row for row in scenario.channel.events() if row["direction"] == "peer"]
    assert scenario.channel.completed_peer_message(receipt["message_id"]) is None
    assert not report["checks"]["role_sends_match_delivered_channel_rows"]
    assert report["rolling_peers"]["delivered_messages"] == report["rolling_peers"]["receipt_wakes"] == 0


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
                            "recipient_id": recipient, "trace_id": "trace"})
    report = scenario._report((), rows, records)
    assert report["rolling_roles"]["cross_channel_replies"] == {"group_to_private": 1, "private_to_group": 1}
