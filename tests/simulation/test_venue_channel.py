"""验证模拟档案与正式入口的交互、持久回执和会话隐私。"""

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import Mock

import pytest

from mahjong_agent_runtime import InMemoryAgentStore
from mahjong_agent_runtime.domains.full_table_notice import FULL_TABLE_NOTICE_TASK_TYPE
from mahjong_agent_runtime.group_chat import BOARD_TASK_TYPE, BoardEngine, GroupMessage
from mahjong_agent_runtime.group_chat.full_table_notice import _declares_guaranteed_start
from mahjong_agent_runtime.input_aggregation import aggregate_pending_input_batch
from mahjong_agent_runtime.models import CustomerProfile, ToolResult, now, using_clock
from mahjong_agent_runtime.stores.sqlite import SQLiteAgentStore
from mahjong_agent_runtime.tracing import JsonlTraceRecorder

from .group_transport import SimulationMessenger
from .sim_adapter import build_runtime
from .venue_channel import AGENT_ID, OWNER_ID, MessageEnvelope, SandboxChannel
from .venue_fixture import VenueFixtureAgent, _reply
from .venue_timeout_recovery import verified_runtime_timeout_recoveries
from .venue_write_guard import VenueWriteGuard

MEMBERS = {"a": "阿岚", "b": "阿白", "c": "阿诚"}
ROOM = "sandbox-room"
ARCHIVED_START_GUARANTEE = "人齐了，今晚19:00准时开打。"


@pytest.fixture
def sandbox(tmp_path):
    store = SQLiteAgentStore(tmp_path / "business.sqlite3")
    runtime, client = build_runtime(store, "mock")
    channel = SandboxChannel(runtime, room_id=ROOM, identities=MEMBERS, archive_path=tmp_path / "messages.sqlite3")
    return channel, runtime, client


def test_all_500_owner_friends_have_empty_nickname_only_private_routes(tmp_path):
    """从冷启动核对五百个好友已能发起私聊，档案保留稳定账号及公开昵称。"""
    members = {f"customer-{i:03}": f"昵称{i % 7}" for i in range(500)}
    store_path = tmp_path / "business.sqlite3"
    archive_path = tmp_path / "messages.sqlite3"
    store = SQLiteAgentStore(store_path)
    runtime, _ = build_runtime(store, "mock")
    channel = SandboxChannel(runtime, room_id=ROOM, identities=members, archive_path=archive_path)
    reopened = SQLiteAgentStore(store_path)
    policy = reopened.get_group_room_policy(ROOM)

    assert channel.owner_friends == members
    assert reopened.customers == {account: CustomerProfile(account, nickname, nickname) for account, nickname in members.items()}
    assert channel.events_for(OWNER_ID) == []
    stamp = now()
    conversations = set()
    for account, nickname in members.items():
        identity = reopened.get_channel_identity(policy.channel, account)
        route = channel._conversation(MessageEnvelope(f"first-{account}", account, "老板好", stamp, "private", OWNER_ID))
        assert identity.is_friend and identity.can_private_message
        assert identity.external_user_id == identity.customer_id == account and identity.public_name == nickname
        assert route == channel.private_conversation_id(account) == identity.private_conversation_id
        assert reopened.recent_turns(route, limit=1) == []
        conversations.add(route)
    assert len(conversations) == 500


def test_same_nickname_friends_have_separate_owner_dialogues(tmp_path):
    """同名好友通过账号路由，各自的入站和店家回复不会合并。"""
    store = SQLiteAgentStore(tmp_path / "business.sqlite3")
    runtime, _ = build_runtime(store, "mock")
    channel = SandboxChannel(runtime, room_id=ROOM, identities={"a": "同名", "b": "同名"},
                             archive_path=tmp_path / "messages.sqlite3")
    stamp = now()
    first = MessageEnvelope("friend-a", "a", "老板今天七点有位吗", stamp, "private", OWNER_ID)
    second = MessageEnvelope("friend-b", "b", "老板明晚八点有位吗", stamp, "private", OWNER_ID)
    own_reply, = channel.send(first)["archive_outputs"]
    other_reply, = channel.send(second)["archive_outputs"]

    own_conversation = channel.private_conversation_id("a")
    other_conversation = channel.private_conversation_id("b")
    assert own_conversation != other_conversation
    assert own_reply["conversation_id"] == own_conversation and own_reply["recipient_id"] == "a"
    assert other_reply["conversation_id"] == other_conversation and other_reply["recipient_id"] == "b"
    assert [row["message_id"] for row in channel.events_for("a")] == [first.message_id, own_reply["message_id"]]
    assert [row["message_id"] for row in channel.events_for("b")] == [second.message_id, other_reply["message_id"]]
    assert channel.events_for(OWNER_ID) == channel.events_for(AGENT_ID)
    assert channel.send(replace(first, recipient_id=AGENT_ID))["delivery_deduplicated"] is True


def test_production_private_route_preserves_verified_quote_for_input_gate(sandbox, monkeypatch):
    """同一收件人的真实引用进入正式输入分流，其他私聊看到的引用不被转给模型。"""
    from mahjong_agent_runtime.models import AgentRuntimeResult
    from scripts import agent_runtime_app as app

    original, runtime, _ = sandbox
    channel = SandboxChannel(runtime, room_id=ROOM, identities=MEMBERS, archive_path=original.archive_path,
                             production_private_routing=True)
    stamp = now()
    channel.publish_owner(kind="private", recipient_id="a", text="五点半那桌，有人六点五十五才到，能等吗？",
                          sent_at=stamp, message_id="ask-a")
    inputs = []
    model_messages = []

    def gate(message, *, trace_id, runtime, quiet_period_elapsed):
        inputs.append(message)
        assert quiet_period_elapsed and message.conversation_id == channel.private_conversation_id("a")
        assert message.text == "行" and message.quoted_message.message_id == "ask-a"
        assert message.quoted_message.conversation_id == message.conversation_id
        return {"action": "process_business", "category": "followup_answer"}

    def main(runtime, message, *, trace_id):
        context = runtime.context_builder.build(message, trace_id=trace_id)
        model_messages.append(json.loads(context.messages[1]["content"])["current_message"])
        return AgentRuntimeResult(trace_id=trace_id, conversation_id=message.conversation_id,
                                  status="completed", final_reply="")

    monkeypatch.setattr(app, "run_wechaty_input_gate", gate)
    monkeypatch.setattr(type(runtime), "handle_user_message", main)
    with pytest.raises(ValueError, match="reference is outside"):
        channel.send(MessageEnvelope("wrong-answer", "b", "行", stamp, "private", AGENT_ID, "ask-a"))
    assert inputs == []
    response = channel.send(MessageEnvelope("right-answer", "a", " 行 ", stamp, "private", AGENT_ID, "ask-a"))
    assert len(inputs) == 1 and response["status"] == "completed" and response["archive_outputs"] == []
    assert response["channel_routing"]["input_gate"]["category"] == "followup_answer"
    model_message, = model_messages
    assert model_message["quoted_message"]["message_id"] == "ask-a"
    metadata = model_message["metadata"]
    assert metadata["channel"] == metadata["platform_name"] == "wechaty"
    assert metadata["source_message_id"] == model_message["message_id"] == "right-answer"
    assert metadata["modalities"] == ["text"] and metadata["message_type"] == 7
    assert "im:" not in json.dumps(metadata)
    assert {"party_size", "temperament", "ground_truth", "role_traits", "world", "simulator"}.isdisjoint(metadata)


@pytest.mark.parametrize("outcome", ["recovered", "exhausted", "ignore"])
def test_input_gate_timeout_uses_actual_sandbox_wait_and_original_private_message(tmp_path, monkeypatch, outcome):
    """输入分流超时只重走分流，等候和最终答复都有同一来话下的真实模拟回执。"""
    from scripts import agent_runtime_app as app

    at = datetime.fromisoformat("2026-11-26T11:39:30+08:00")
    store = SQLiteAgentStore(tmp_path / "business.sqlite3")
    fixture = VenueFixtureAgent(seed=42)
    traces = tmp_path / "agent_trace.log"
    runtime, _ = build_runtime(store, "mock", offline_client=fixture, trace_recorder=JsonlTraceRecorder(traces))
    channel = SandboxChannel(runtime, room_id=ROOM, identities=MEMBERS, archive_path=tmp_path / "messages.sqlite3",
                             production_private_routing=True)
    request = MessageEnvelope("input-gate-original", "a", "晚上七点那桌可以给我留个位置吗，我一个人。",
                              at, "private", AGENT_ID)
    fixture.turns[request.message_id] = ("opportunity", {})
    with using_clock(lambda: at):
        game, _ = store.create_game(conversation_id=channel.private_conversation_id("b"), organizer_id="b",
            organizer_name=MEMBERS["b"], known_players=[], trace_id="original-table",
            requirement={"game_type": "hangzhou_mahjong", "start_time_kind": "scheduled", "start_time": "19:00",
                         "duration_kind": "fixed", "duration_hours": 4})

    payloads = []

    class InputGateClient:
        def complete(self, messages, *, trace_id, timeout_seconds):
            payloads.append(json.loads(messages[-1]["content"]))
            if len(payloads) == 2:
                assert [row["text"] for row in channel.events_for("a")] == [request.text, "稍等一下"]
                batch = store.pending_input_batch(channel.private_conversation_id("a"), "a")
                assert batch.decision["gate_recovery"]["open"] is True
                assert store.require_game(game.game_id).remaining_seats() == 3
            if outcome == "exhausted" or (outcome == "recovered" and len(payloads) == 1):
                raise TimeoutError("read timeout from the verified ingress model")
            ignore = outcome == "ignore"
            return json.dumps({"action": "ignore" if ignore else "process_business", "should_route": not ignore,
                               "category": "uncertain" if ignore else "operational", "confidence": 1,
                               "reasoning_summary": "输入分流已明确判断。", "evidence": [],
                               "input_completeness": "complete", "expects_more_fragments": False,
                               "missing_information": []})

    monkeypatch.setenv("MAHJONG_WECHATY_INPUT_GATE_ENABLED", "true")
    monkeypatch.setenv("MAHJONG_WECHATY_INPUT_GATE_FAIL_OPEN", "false")
    monkeypatch.setattr(app, "build_wechaty_input_gate_client", InputGateClient)
    monkeypatch.setattr(app, "request_local_json", lambda *_args, **_kwargs: pytest.fail("仿真不可调用真实微信"))
    response = channel.send(request)
    outputs = response["archive_outputs"]

    assert len(payloads) == (1 if outcome == "ignore" else 2)
    for payload in payloads:
        assert payload["current_message"]["message_id"] == request.message_id
        assert payload["current_message"]["text"] == request.text
    assert all(row["source_message_id"] == request.message_id and row["trace_id"] == response["trace_id"]
               and row["conversation_id"] == channel.private_conversation_id("a") and row["recipient_id"] == "a"
               for row in outputs)
    if outcome == "recovered":
        assert [row["text"] for row in outputs] == ["稍等一下", "好，给你留上一个位置了。"]
        assert outputs[0]["sent_at"] < outputs[1]["sent_at"]
        assert response["status"] == "completed"
        assert store.require_game(game.game_id).remaining_seats() == 2
        assert sum(row["name"] == "join_game" and row["called"] for row in response["tool_results"]) == 1
        assert len([turn for turn in store.recent_turns(channel.private_conversation_id("a"), limit=30)
                    if turn.role.value == "user" and turn.metadata.get("source_message_id") == request.message_id]) == 1
    else:
        assert [row["text"] for row in outputs] == ([] if outcome == "ignore" else ["稍等一下"])
        assert store.require_game(game.game_id).remaining_seats() == 3
        assert not response.get("tool_results")
    assert channel.events_for("c") == []
    assert channel.send(request) == {**response, "delivery_deduplicated": True}
    assert len(channel.events_for("a")) == 1 + len(outputs)
    recoveries = verified_runtime_timeout_recoveries(traces, channel.events())
    if outcome == "recovered":
        evidence, = recoveries
        assert evidence["model_stage"] == "input_gate" and evidence["message_id"] == request.message_id
        assert evidence["reply_message_id"] == outputs[-1]["message_id"]
        # 待办即使记录成功，也不能用仅有等待回执的账本冒充最终恢复。
        assert verified_runtime_timeout_recoveries(traces, channel.events()[:-1]) == []
    else:
        assert recoveries == []


@pytest.mark.parametrize("ready_on_fourth", [True, False])
def test_private_bubbles_preserve_independent_receipts_and_actual_aggregate(tmp_path, monkeypatch, ready_on_fourth):
    """四条相隔五秒的来话实际走入口积累，在该批可处理时才交给主模型。"""
    from scripts import agent_runtime_app as app

    at = datetime.fromisoformat("2026-11-26T19:00:00+08:00")
    bodies = ["老板，今晚还有局吗？", "我七点半以后才到", "想打红中", "一块底，无烟的就行"]
    fragments = [MessageEnvelope(f"bubble-{index}", "a", text, at + timedelta(seconds=5 * index),
                                 "private", AGENT_ID) for index, text in enumerate(bodies)]
    main_inputs = []
    gate_inputs = []

    class MainClient:
        def complete(self, messages, **kwargs):
            main_inputs.append(json.loads(messages[-1]["content"])["current_message"])
            return _reply("好，想找七点半以后的一块底无烟红中。", status="completed")

    def input_gate(message, *, quiet_period_elapsed, **kwargs):
        gate_inputs.append((message.to_dict(), quiet_period_elapsed))
        ready = quiet_period_elapsed or ready_on_fourth and len(message.metadata["input_window"]["fragments"]) == 4
        return {"action": "process_business" if ready else "wait_for_more_input"}

    monkeypatch.setenv("MAHJONG_INPUT_QUIET_PERIOD_SECONDS", "30")
    monkeypatch.setenv("MAHJONG_INPUT_QUIET_BOUNDARY_GRACE_SECONDS", "5")
    monkeypatch.setattr(app, "run_wechaty_input_gate", input_gate)
    monkeypatch.setattr(app, "request_local_json", lambda *_args, **_kwargs: pytest.fail("仿真不可调用真实微信"))
    store = SQLiteAgentStore(tmp_path / "business.sqlite3")
    runtime, _ = build_runtime(store, "mock", offline_client=MainClient())
    channel = SandboxChannel(runtime, room_id=ROOM, identities=MEMBERS, archive_path=tmp_path / "messages.sqlite3",
                             production_private_routing=True, production_private_aggregation=True)

    for index, fragment in enumerate(fragments):
        assert channel.dispatch_due_private_inputs(at=fragment.sent_at) == []
        response = channel.send(fragment)
        pending = store.pending_input_batch(channel.private_conversation_id("a"), "a")
        assert pending.quiet_deadline == fragment.sent_at + timedelta(seconds=35)
        assert len(pending.fragments) == index + 1
        if index < 3 or not ready_on_fourth:
            assert response["input_status"] == "pending" and response["archive_outputs"] == []
            assert main_inputs == [] and store.games == {}

    due_at = pending.quiet_deadline
    if ready_on_fourth:
        assert channel.dispatch_due_private_inputs(at=due_at) == []
        assert response["input_status"] == "completed"
        delivered_at = fragments[-1].sent_at
    else:
        assert channel.dispatch_due_private_inputs(at=due_at - timedelta(milliseconds=1)) == []
        response, = channel.dispatch_due_private_inputs(at=due_at.astimezone(timezone.utc))
        delivered_at = due_at

    window = main_inputs[0]["metadata"]["input_window"]
    assert len(main_inputs) == 1 and main_inputs[0]["text"] == "\n".join(bodies)
    assert window["fragment_count"] == 4 and window["quiet_period_elapsed"] is not ready_on_fourth
    assert [item["message_id"] for item in window["fragments"]] == [item.message_id for item in fragments]
    assert [datetime.fromisoformat(item["sent_at"]) for item in window["fragments"]] == [item.sent_at for item in fragments]
    assert [len(message["metadata"]["input_window"]["fragments"]) for message, _ in gate_inputs] == (
        [1, 2, 3, 4] if ready_on_fourth else [1, 2, 3, 4, 4]
    )
    assert gate_inputs[-1][1] is not ready_on_fourth
    inbound = [row for row in channel.events() if row["direction"] == "inbound"]
    assert [row["text"] for row in inbound] == bodies
    assert [datetime.fromisoformat(row["sent_at"]) for row in inbound] == [item.sent_at for item in fragments]
    reply, = response["archive_outputs"]
    assert reply["recipient_id"] == "a" and reply["source_message_id"] == fragments[-1].message_id
    assert datetime.fromisoformat(reply["sent_at"]) == delivered_at
    assert store.games == {} and channel.events_for("b") == []
    assert channel.dispatch_due_private_inputs(at=due_at) == []
    assert channel.send(fragments[-1]) == {**response, "delivery_deduplicated": True}
    forged = aggregate_pending_input_batch(pending, quiet_period_elapsed=not ready_on_fourth,
                                          trigger="quiet_period_elapsed" if not ready_on_fourth else f"im:{ROOM}_message_arrived")
    forged.metadata["input_window"]["fragments"][0]["text"] = "伪造的聊天片段"
    forged_window = runtime.context_builder.build(forged, trace_id="forged-window").payload["current_message"]["metadata"]
    assert "input_window" not in forged_window


def test_group_routing_archives_model_request_and_response_for_original_message(sandbox):
    """群入口保留模型原请求与响应，后续能核对合同重试实际看到了什么。"""
    channel, runtime, _ = sandbox
    result = channel.send(MessageEnvelope("social-group-message", "a", "今天真热", now(), "group", ROOM))
    trace = runtime.trace_recorder.get_trace(result["trace_id"])
    request, = [event.content for event in trace if event.step == "group_semantic_routing_prompt"]
    response, = [event.content for event in trace if event.step == "group_semantic_routing_response"]
    assert json.loads(request["messages"][1]["content"])["message"]["message_id"] == "social-group-message"
    assert json.loads(response["content"])["source_message_id"] == "social-group-message"


@pytest.mark.parametrize("gate_wait_confirmed", [True, False])
def test_private_input_wait_is_not_repeated_when_the_main_planner_also_times_out(tmp_path, monkeypatch, gate_wait_confirmed):
    """原来话跨两级超时仍只有一条真实等待回执，新的顾客来话独立处理。"""
    from scripts import agent_runtime_app as app

    class TimedOutPlanner:
        def complete(self, messages, *, trace_id, timeout_seconds):
            raise TimeoutError("planner read timeout")

    store = SQLiteAgentStore(tmp_path / "business.sqlite3")
    traces = JsonlTraceRecorder(tmp_path / "agent_trace.log")
    runtime, _ = build_runtime(store, "mock", offline_client=TimedOutPlanner(), trace_recorder=traces)
    channel = SandboxChannel(runtime, room_id=ROOM, identities=MEMBERS, archive_path=tmp_path / "messages.sqlite3",
                             production_private_routing=True)
    stamp = datetime.fromisoformat("2026-11-25T19:00:00+08:00")
    source = MessageEnvelope("same-source-two-stage", "a", "老板好 我这边还在等位 能不能帮我组一桌 杭州麻将 1到3块底 我最早19点11到 能打到22点",
                             stamp, "private", AGENT_ID)

    def input_gate(message, *, input_batch=None, **_kwargs):
        if message.message_id == source.message_id and input_batch is None:
            return {"action": "ignore", "technical_failure": "read_timeout"}
        return {"action": "process_business", "category": "operational"}

    original_send = channel._record_output

    def record_output(output, source, trace_id):
        if output.get("metadata", {}).get("message_type") == "input_gate_wait" and not gate_wait_confirmed:
            return ""
        return original_send(output, source, trace_id)

    monkeypatch.setattr(app, "run_wechaty_input_gate", input_gate)
    monkeypatch.setattr(channel, "_record_output", record_output)
    result = channel.send(source)

    assert result["status"] == "needs_human" and result["final_reply"] == ("" if gate_wait_confirmed else "稍等一下")
    wait, = result["archive_outputs"]
    assert wait["text"] == "稍等一下" and wait["source_message_id"] == source.message_id and wait["recipient_id"] == "a"
    assert wait["message_id"].startswith("outbound:input-gate-wait:") is gate_wait_confirmed
    handoff, = app.owner_agent_handoffs(runtime)
    assert handoff["request"] == source.text and handoff["trace_id"] == result["trace_id"]
    assert not handoff["retrying"] and handoff["wait_delivered"] is gate_wait_confirmed
    start = next(event.content for event in traces.get_trace(result["trace_id"]) if event.step == "agent_run_started")
    if gate_wait_confirmed:
        assert start["prior_input_wait_delivery"]["source_message_id"] == source.message_id
    else:
        assert "prior_input_wait_delivery" not in start
    assert channel.send(source)["archive_outputs"] == [wait]

    later = replace(source, message_id="new-source-own-wait", text="老板，有进展吗？", sent_at=stamp + timedelta(minutes=2))
    followup = channel.send(later)
    next_wait, = followup["archive_outputs"]
    assert followup["status"] == "needs_human" and next_wait["text"] == "稍等一下"
    assert next_wait["source_message_id"] == later.message_id and next_wait["trace_id"] != wait["trace_id"]
    assert not store.games


@pytest.mark.parametrize("exhausted", [False, True])
def test_group_classifier_timeout_retries_once_with_original_group_receipt(monkeypatch, tmp_path, exhausted):
    """语义超时只重走群入站；留待办、去重及最终群回复都对应真实原消息。"""
    from scripts import agent_runtime_app as app

    traces = tmp_path / "topic-timeout.log"
    store = SQLiteAgentStore(tmp_path / "business.sqlite3")
    runtime, client = build_runtime(store, "mock", trace_recorder=JsonlTraceRecorder(traces))
    channel = SandboxChannel(runtime, room_id=ROOM, identities=MEMBERS, archive_path=tmp_path / "messages.sqlite3")
    ingress = MessageEnvelope("group-timeout", "a", "老板，今晚还有现成的杭麻吗？", now(), "group", ROOM)
    requests = []
    complete = client.complete

    def model(messages, *, trace_id, timeout_seconds):
        payload = client._context_payload(messages)
        if "topic_action" in payload.get("output_contract", {}).get("required", ()):
            requests.append(payload["message"])
            if len(requests) == 2:
                handoff, = app.owner_agent_handoffs(runtime)
                assert handoff["trace_id"] == trace_id and handoff["retrying"]
                assert handoff["request"] == ingress.text and not handoff["wait_delivered"]
                assert [row["text"] for row in channel.events_for("a")] == [ingress.text]
            if len(requests) == 1 or exhausted:
                raise TimeoutError("test group classifier read timeout")
        return complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(client, "complete", model)
    response = channel.send(ingress)
    assert len(requests) == 2 and requests[0] == requests[1]
    assert requests[0]["message_id"] == ingress.message_id
    assert response["group_processing"]["topic_recovery"]["action"] == "group_topic_retry_queued"
    assert all(row["text"] != "稍等一下" for row in response["archive_outputs"])
    if exhausted:
        assert response["status"] == "needs_human" and response["archive_outputs"] == []
        handoff, = app.owner_agent_handoffs(runtime)
        assert not handoff["retrying"] and not handoff["wait_delivered"]
        assert runtime.store.get_group_topic_message("im:sandbox-room", ROOM, "a", ingress.message_id) is None
    else:
        output, = response["archive_outputs"]
        assert output["source_message_id"] == ingress.message_id and output["trace_id"] == response["trace_id"]
        assert output["kind"] == "group" and output["recipient_id"] == ROOM
        assert datetime.fromisoformat(output["sent_at"]) > ingress.sent_at
        assert len(response["group_processing"]["runtime_results"]) == 1
        assert any(item["name"] == "search_current_games" and item["called"] for item in response["tool_results"])
        assert app.owner_agent_handoffs(runtime) == []
    assert not runtime.store.games
    assert channel.send(ingress)["delivery_deduplicated"] is True and len(requests) == 2
    proof = verified_runtime_timeout_recoveries(traces, channel.events())
    if exhausted:
        assert proof == []
    else:
        delivered, = proof
        assert delivered["model_stage"] == "group_topic" and delivered["message_id"] == ingress.message_id
        assert delivered["reply_message_id"] == output["message_id"]
        source, archived = channel.events()
        assert verified_runtime_timeout_recoveries(traces, [source]) == []
        assert verified_runtime_timeout_recoveries(traces, [source, {**archived, "recipient_id": "b"}]) == []


def test_owner_friendship_is_persisted_independently_from_group_membership(tmp_path):
    """群成员仍可正常发言，老板私聊由独立的好友名册约束。"""
    store = SQLiteAgentStore(tmp_path / "business.sqlite3")
    runtime, client = build_runtime(store, "mock")
    archive_path = tmp_path / "messages.sqlite3"
    channel = SandboxChannel(runtime, room_id=ROOM, identities=MEMBERS, owner_friend_ids={"a"}, archive_path=archive_path)
    client.semantic_intent = "chitchat"
    stamp = now()
    group = MessageEnvelope("group-b", "b", "大家好", stamp, "group", ROOM)
    assert channel.send(group)["group_processing"]["routing"]["action"] == "ignore"
    with pytest.raises(ValueError, match="outside"):
        channel.send(replace(group, message_id="dm-b", kind="private", recipient_id=OWNER_ID))
    with pytest.raises(ValueError, match="not a friend"):
        channel.publish_owner(kind="private", recipient_id="b", text="你好", sent_at=stamp, message_id="owner-b")

    reopened = SandboxChannel(runtime, room_id=ROOM, identities=MEMBERS, archive_path=archive_path)
    assert reopened.owner_friends == {"a": MEMBERS["a"]}
    policy = store.get_group_room_policy(ROOM)
    nonfriend = store.get_channel_identity(policy.channel, "b")
    assert not nonfriend.is_friend and not nonfriend.can_private_message
    assert [event["message_id"] for event in reopened.events_for("b")] == [group.message_id]
    with pytest.raises(ValueError, match="friendships differ"):
        SandboxChannel(runtime, room_id=ROOM, identities=MEMBERS, owner_friend_ids={"a", "b"}, archive_path=archive_path)


def test_group_chatter_and_peer_dm_stay_in_distinct_views(sandbox, monkeypatch):
    channel, runtime, client = sandbox
    client.semantic_intent = "chitchat"
    # 如果沙盒误继承本机缓存环境，正式路由会尝试连接这个无效地址。
    monkeypatch.setenv("MAHJONG_TOPIC_REDIS_URL", "invalid://sandbox-must-not-attempt-this")
    anchor = now()
    group = MessageEnvelope("g1", "a", "这周上海怎么又下雨", anchor, "group", ROOM)
    assert channel.send(group)["group_processing"]["routing"]["action"] == "ignore"
    private = MessageEnvelope("p1", "b", "我想找你聊点私事", anchor, "private", "c")
    assert channel.send(private)["status"] == "delivered"
    assert client.call_count == client.semantic_call_count == 1
    assert [row["message_id"] for row in channel.events_for("a")] == ["g1"]
    assert [row["message_id"] for row in channel.events_for("b")] == ["g1", "p1"]
    assert [row["message_id"] for row in channel.events_for("c")] == ["g1", "p1"]
    assert [row["message_id"] for row in channel.events_for(AGENT_ID)] == ["g1"]
    assert [row["direction"] for row in channel.events()] == ["inbound", "peer"]
    assert not runtime.store.customers["b"].preferred_games
    assert not runtime.store.recent_turns(channel.private_conversation_id("b"), limit=5)
    with pytest.raises(ValueError, match="visible conversation"):
        channel.send(replace(group, message_id="g2", quoted_message_id="p1"))
    with pytest.raises(ValueError, match="visible conversation"):
        channel.send(replace(private, message_id="p2", sender_id="a", quoted_message_id="p1"))
    with pytest.raises(ValueError, match="unknown simulation sender"):
        channel.send(replace(group, sender_id=AGENT_ID))
    assert len(channel.events()) == 2


def test_real_private_reply_survives_reopen_and_uses_stable_receipt(sandbox):
    channel, runtime, client = sandbox
    anchor = now()
    request = MessageEnvelope("one", "a", "老板还有位吗", anchor, "private", AGENT_ID)
    response = channel.send(request)
    delivered, = response["archive_outputs"]
    assert delivered["sender_id"] == AGENT_ID
    assert delivered["recipient_id"] == "a"
    assert delivered["source_message_id"] == request.message_id
    assert delivered["trace_id"] == response["trace_id"]
    assert response["final_reply"] == delivered["text"]
    assert channel.events_for("b") == []
    assert [row["message_id"] for row in channel.events_for("a")] == ["one", delivered["message_id"]]

    reopened = SandboxChannel(runtime, room_id=ROOM, identities=MEMBERS, archive_path=channel.archive_path)
    assert reopened.events() == channel.events()
    assert reopened.send(request) == {**response, "delivery_deduplicated": True}
    assert client.call_count == 2
    with pytest.raises(ValueError, match="message id conflict"):
        reopened.send(replace(request, sender_id="b"))
    with pytest.raises(ValueError, match="membership differs"):
        SandboxChannel(runtime, room_id=ROOM, identities={"a": "阿岚"}, archive_path=channel.archive_path)
    other = SandboxChannel(runtime, room_id="other-room", identities=MEMBERS, archive_path=channel.archive_path)
    alternate = other.send(replace(request, text="第二个沙盒里的问题"))
    assert alternate["trace_id"] != response["trace_id"]
    assert channel.events() == reopened.events()
    assert len(other.events()) == 2

    with pytest.raises(ValueError, match="visible conversation"):
        other.send(replace(request, message_id="bad-quote", quoted_message_id=delivered["message_id"]))
    followup = replace(request, message_id="two", text="刚才说的呢", sent_at=anchor + timedelta(seconds=1),
                       quoted_message_id=delivered["message_id"])
    reopened.send(followup)
    received = next(turn for turn in runtime.store.recent_turns(reopened.private_conversation_id("a"), limit=10)
                    if turn.role.value == "user" and turn.metadata.get("source_message_id") == "two")
    assert received.metadata["input_receipt_quoted_message"]["message_id"] == delivered["message_id"]


@pytest.mark.parametrize("show_scope", [False, True])
def test_write_authorization_gets_only_the_actual_shop_dialogue_once(sandbox, monkeypatch, show_scope):
    """授权检查随实际私聊入口运行，只收到同一顾客与店家的已见消息。"""
    channel, runtime, _ = sandbox
    calls = []
    active_scope = []
    runtime_calls = []

    class Guard:
        def authorize(self, message, conversation_id, history):
            calls.append((message.message_id, conversation_id, [row["message_id"] for row in history]))
            return message.message_id

        @contextmanager
        def planner_scope(self, message_id):
            active_scope.append(message_id)
            try:
                yield
            finally:
                active_scope.pop()

    handle = type(runtime).handle_user_message

    def observe_scope(instance, message, **kwargs):
        runtime_calls.append((message.message_id, list(active_scope)))
        return handle(instance, message, **kwargs)

    monkeypatch.setattr(type(runtime), "handle_user_message", observe_scope)
    guarded = SandboxChannel(runtime, room_id=ROOM, identities=MEMBERS, archive_path=channel.archive_path,
                             write_guard=Guard(), show_verified_write_scope=show_scope)
    stamp = now()
    guarded.send(MessageEnvelope("peer", "b", "我先找你商量", stamp, "private", "a"))
    guarded.send(MessageEnvelope("other", "b", "几点开始", stamp, "private", AGENT_ID))
    first = MessageEnvelope("first", "a", "老板晚点有位置吗", stamp, "private", AGENT_ID)
    delivery, = guarded.send(first)["archive_outputs"]
    next_message = replace(first, message_id="second", text="还有别的时间吗", sent_at=stamp + timedelta(seconds=1))
    guarded.send(next_message)
    guarded.send(next_message)
    conversation = guarded.private_conversation_id("a")
    assert calls == [
        ("other", guarded.private_conversation_id("b"), ["other"]),
        ("first", conversation, ["first"]),
        ("second", conversation, ["first", delivery["message_id"], "second"]),
    ]
    assert runtime_calls == [(key, [key] if show_scope else []) for key in ("other", "first", "second")]
    assert not active_scope


def test_actual_group_reply_is_shared_once_and_can_be_quoted(sandbox):
    channel, _, client = sandbox
    stamp = now()
    request = MessageEnvelope("question", "a", "今晚还有位吗", stamp, "group", ROOM)
    response = channel.send(request)
    assert response["group_processing"]["routing"]["action"] == "agent_loop"
    reply, = response["archive_outputs"]
    assert reply["kind"] == "group" and reply["recipient_id"] == ROOM
    assert channel.events() == channel.events_for("a") == channel.events_for("b") == channel.events_for("c")
    assert [event["message_id"] for event in channel.events()] == ["question", reply["message_id"]]
    client.semantic_intent = "chitchat"
    channel.send(replace(request, message_id="quote", sender_id="b", text="我刚看到这条",
                         sent_at=stamp + timedelta(seconds=1), quoted_message_id=reply["message_id"]))
    assert channel.events()[-1]["quoted_message_id"] == reply["message_id"]


def _board_game(channel, store, stamp, *, seat_format="371"):
    channel_id = store.get_group_room_policy(ROOM).channel
    messenger = SimulationMessenger(store, ROOM, channel=channel_id)
    board = BoardEngine(store=store, messenger=messenger, clock=lambda: stamp)
    game = board.import_game_from_post(GroupMessage(
        room_id=ROOM, conversation_id=channel.group_conversation_id, channel=channel_id,
        sender_external_id="a", sender_name=MEMBERS["a"], sent_at=stamp,
        text=f"{stamp + timedelta(hours=1):%Y-%m-%d %H:%M} 0.5 无烟 {seat_format}", message_id="original-post",
    ), trace_id="post")
    task, = [task for task in store.scheduled_tasks.values() if task.task_type == BOARD_TASK_TYPE]
    assert not messenger.outputs
    return game, task


@pytest.mark.parametrize(("text", "is_guarantee"), [
    (ARCHIVED_START_GUARANTEE, True),
    ("这桌人到齐了！晚上七点准点开局。", True),
    ("人齐了，今晚１９：００一定准时开打！", True),
    ("人齐了，原定今晚19:00开打。", False),
    ("人齐了，今晚19:00开打。", False),
    ("人齐了，按原来约的时间开。", False),
    ("人齐了，争取今晚19:00准时开打。", False),
    ("如果大家按时到，今晚19:00准时开打。", False),
    ("今晚19:00能准时开打吗？", False),
])
def test_full_table_notice_only_blocks_unambiguous_start_guarantees(text, is_guarantee):
    """确定性防线只处理直接担保，保留普通计划表达和由语义审查处理的边界。"""
    assert _declares_guaranteed_start(text) is is_guarantee


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("changes", [
    "allow", "denied_room", "revoked_contact", "retry", "reopened_before_retry", "ack_lost_then_input", "missing_review",
    "start_guarantee",
])
def test_private_last_seat_schedules_and_delivers_only_to_authorized_existing_players(tmp_path, monkeypatch, backend, changes):
    """私聊第四席与满桌任务同事务提交，真实回执只写入已授权参加者各自的私聊。"""

    notices = []
    reviewed_notices = []

    class NoticeFixture(VenueFixtureAgent):
        def complete(self, messages, *, trace_id, timeout_seconds):
            context = self._context_payload(messages)
            trigger = context.get("system_trigger") or {}
            if trigger.get("trigger_type") == "group_game_full":
                notices.append(trigger)
                if changes == "start_guarantee" and trigger["sender_id"] == "b":
                    if not trigger.get("delivery_review_feedback"):
                        return _reply(ARCHIVED_START_GUARANTEE)
                    return _reply("人齐了，原定今晚19:00开打。")
                return _reply("人齐了，按原来约的时间开。")
            return super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)

    store_path = tmp_path / "business.sqlite3"
    store = SQLiteAgentStore(store_path) if backend == "sqlite" else InMemoryAgentStore()
    client = NoticeFixture(seed=42)
    runtime, _ = build_runtime(store, "mock", offline_client=client)
    if changes != "missing_review":
        review = Mock()

        def approve(messages, **_):
            payload = json.loads(messages[-1]["content"])
            trigger = payload["current_message"]["metadata"].get("system_trigger") or {}
            if trigger.get("trigger_type") == "group_game_full":
                reviewed_notices.append(trigger)
            items = payload["review_items"]
            return json.dumps({"approved": True, "needs_human": False, "reasoning_summary": "与业务事实相符",
                "violations": [], "item_reviews": [
                    {"item_id": item["item_id"], "approved": True, "suggested_safe_text": item["text"],
                     "reasoning_summary": "内容可发", "follow_up_commitments": [], "violations": []}
                    for item in items
                ]}, ensure_ascii=False)

        review.complete.side_effect = approve
        runtime.action_processor.reply_self_review_enabled = True
        runtime.action_processor.reply_self_review_client = review
    people = {**MEMBERS, "d": "末席", "z": "旁观者"}
    channel = SandboxChannel(runtime, room_id=ROOM, identities=people,
                             archive_path=tmp_path / "messages.sqlite3")
    stamp = datetime.fromisoformat("2026-11-25T18:00:00+08:00") if changes == "start_guarantee" else now()
    start = stamp + timedelta(hours=1)
    bridge_attempts = []
    original_send = SimulationMessenger.send_private_message
    if changes in {"retry", "reopened_before_retry", "ack_lost_then_input"}:
        def retry_once(messenger, external_user_id, text, *, metadata=None):
            if external_user_id == "b" and (metadata or {}).get("message_type") == FULL_TABLE_NOTICE_TASK_TYPE:
                bridge_attempts.append((metadata["draft_id"], text))
                if len(bridge_attempts) == 1:
                    if changes == "ack_lost_then_input":
                        original_send(messenger, external_user_id, text, metadata=metadata)
                    return "suppressed:unconfirmed"
            return original_send(messenger, external_user_id, text, metadata=metadata)
        monkeypatch.setattr(SimulationMessenger, "send_private_message", retry_once)
    with using_clock(lambda: stamp):
        game, _ = store.create_game(
            conversation_id=channel.private_conversation_id("a"), organizer_id="a", organizer_name=people["a"],
            requirement={"game_type": "hangzhou_mahjong", "stake": "1", "planned_start_at": start.isoformat(),
                         "start_time_kind": "scheduled", "duration_hours": 4, "duration_kind": "fixed"},
            known_players=[{"customer_id": "a", "display_name": people["a"]}], trace_id="create",
        )
        for participant in ("b", "c"):
            store.join_game(game_id=game.game_id, customer_id=participant, display_name=people[participant],
                            trace_id="earlier-seat")
        if changes == "denied_room":
            store.upsert_group_room_policy(replace(store.get_group_room_policy(ROOM), outbound_enabled=False))
        if changes == "revoked_contact":
            store.upsert_customer(replace(store.customers["b"], no_contact=True))
        client.turns["last-private"] = ("opportunity", {})
        inbound = MessageEnvelope("last-private", "d", "给我留一个", stamp, "private", AGENT_ID)
        sent = channel.send(inbound)
        assert channel.send(inbound)["delivery_deduplicated"]
        assert store.require_game(game.game_id).remaining_seats() == 0
        if changes in {"retry", "reopened_before_retry", "ack_lost_then_input", "start_guarantee"}:
            retry_task, = [task for task in store.scheduled_tasks.values() if task.task_type == FULL_TABLE_NOTICE_TASK_TYPE
                           and task.subject_id == "b"]
            assert retry_task.status.value == "pending" and retry_task.attempts == 1
        if changes == "start_guarantee":
            assert retry_task.last_error == "RuntimeError: full_table_notice.unsupported_start_guarantee"
            assert store.resolve_message_reference(conversation_id=retry_task.conversation_id,
                                                   message_id=retry_task.idempotency_key) is None
        if changes == "reopened_before_retry":
            store.record_candidate_reply(game_id=game.game_id, customer_id="b", display_name=people["b"],
                                         status="declined", trace_id="seat-reopened")
    # 后续来话只推进虚拟时钟，通知按实际到期重新领取。
    retry_at = stamp + timedelta(seconds=61)
    with using_clock(lambda: retry_at):
        sender_id = "b" if changes == "ack_lost_then_input" else "z"
        channel.send(replace(inbound, message_id="later", sender_id=sender_id, text="今天先不来", sent_at=retry_at))
    reopened = SQLiteAgentStore(store_path) if backend == "sqlite" else store
    tasks = [task for task in reopened.scheduled_tasks.values() if task.task_type == FULL_TABLE_NOTICE_TASK_TYPE]
    expected = {"allow": {"a", "b", "c"}, "denied_room": set(), "revoked_contact": {"a", "c"},
                "retry": {"a", "b", "c"}, "reopened_before_retry": {"a", "c"},
                "ack_lost_then_input": {"a", "b", "c"}, "missing_review": set(),
                "start_guarantee": {"a", "b", "c"}}[changes]
    pending = {"b"} if changes == "ack_lost_then_input" else {"a", "b", "c"} if changes == "missing_review" else set()
    assert {task.subject_id for task in tasks if task.status.value == "pending"} == pending
    outputs = [row for row in channel.events() if row["message_id"].startswith(f"outbound:{FULL_TABLE_NOTICE_TASK_TYPE}:")]
    assert len(outputs) == len(expected)
    assert {row["recipient_id"] for row in outputs} == expected
    assert all(row["kind"] == "private" and row["conversation_id"] == channel.private_conversation_id(row["recipient_id"])
               for row in outputs)
    first_wave = expected - {"b"} if changes in {"retry", "start_guarantee"} else expected
    assert {row["recipient_id"] for row in sent["archive_outputs"]} == first_wave | {"d"}
    if changes == "start_guarantee":
        rejected, rewritten = [notice for notice in notices if notice["sender_id"] == "b"]
        assert rejected["game"] == rewritten["game"]
        assert rejected["game"]["planned_start_at"] == "2026-11-25T19:00:00+08:00"
        assert rejected["game"]["actual_started_at"] is None
        assert rejected["game"]["on_time_start_guaranteed"] is False
        assert "delivery_review_feedback" not in rejected
        assert rewritten["delivery_review_feedback"]["rule_id"] == "full_table_notice.unsupported_start_guarantee"
        own_reviews = [notice for notice in reviewed_notices if notice["sender_id"] == "b"]
        assert [notice["game"] for notice in own_reviews] == [rejected["game"], rewritten["game"]]
        assert all("审查必须拒绝把计划写成准时保证" in notice["reply_guidelines"] for notice in own_reviews)
        assert all(row["text"] != ARCHIVED_START_GUARANTEE for row in channel.events())
        receipt, = [row for row in outputs if row["recipient_id"] == "b"]
        assert receipt["text"] == "人齐了，原定今晚19:00开打。"
        assert store.resolve_message_reference(conversation_id=retry_task.conversation_id,
                                               message_id=retry_task.idempotency_key).text == receipt["text"]
    if changes == "retry":
        assert len(bridge_attempts) == 2 and bridge_attempts[0] == bridge_attempts[1]
    if changes == "reopened_before_retry":
        assert len(bridge_attempts) == 1 and reopened.require_game(game.game_id).remaining_seats() == 1
    if changes == "ack_lost_then_input":
        assert len(bridge_attempts) == 1
        delivered_key, _ = bridge_attempts[0]
        assert reopened.resolve_message_reference(
            conversation_id=channel.private_conversation_id("b"), message_id=delivered_key,
        ) is None


@pytest.mark.parametrize("authorized", [False, True])
def test_quoted_group_claim_preflights_only_original_public_source_before_private_join(tmp_path, authorized):
    """群自然报名经独立授权及原看板校验；编号认领和普通群咨询不经过该预核。"""

    class BoardClaimFixture(VenueFixtureAgent):
        def _semantic_topic_result(self, payload):
            decision = super()._semantic_topic_result(payload)
            if payload["message"]["message_id"] == "natural-claim":
                board, = payload["public_board"]
                decision["board_reference"] = {
                    "snapshot_id": board["snapshot_id"], "display_no": board["display_no"],
                }
            return decision

    authorizer = Mock()
    body = "1号那桌杭麻还缺人吗 能搭一桌吗 我报一个"
    authorizer.complete.return_value = json.dumps({"permissions": [{
        "scope": scope, "evidence": body, "offer_message_id": None, "offer_evidence": None,
    } for scope in ("participate", "reserve_room")] if authorized else []}, ensure_ascii=False)
    guard = VenueWriteGuard(authorizer, tmp_path / "authorization.jsonl")
    store = SQLiteAgentStore(tmp_path / "business.sqlite3")
    client = BoardClaimFixture(seed=42)
    runtime, _ = build_runtime(store, "mock", offline_client=client, tool_gateway=guard.guarded_gateway(store))
    channel = SandboxChannel(runtime, room_id=ROOM, identities=MEMBERS,
                             archive_path=tmp_path / "messages.sqlite3", write_guard=guard, show_verified_write_scope=True)
    stamp = now()
    game, task = _board_game(channel, store, stamp, seat_format="272")
    published = channel.publish_board(trace_id="publish", sent_at=task.due_at)
    board, = published["archive_outputs"]
    later = task.due_at + timedelta(seconds=1)
    for sender in ("b", "c"):
        channel.send(MessageEnvelope(f"{sender}-group", sender, "闲聊", later, "group", ROOM))
        channel.deliver_private_history(MessageEnvelope(f"{sender}-private", OWNER_ID, "私聊", later, "private", sender))

    client.semantic_intent = "query"
    query = MessageEnvelope("query", "c", "1号那桌还缺人吗", later, "group", ROOM, board["message_id"])
    assert channel.send(query)["group_processing"]["routing"]["action"] == "agent_loop"
    native = MessageEnvelope("native", "b", "1来", later, "group", ROOM, board["message_id"])
    assert channel.send(native)["group_processing"]["routing"]["detail"]["status"] == "claimed"
    assert store.require_game(game.game_id).remaining_seats() == 1
    assert store.get_game_claim_by_source(channel.group_conversation_id, "native") is not None
    assert channel.guarded_agent_input_count() == 0
    assert not authorizer.complete.called

    client.semantic_intent = "claim"
    client.turns["natural-claim"] = ("opportunity", {})
    claim = replace(query, message_id="natural-claim", text=body)
    response = channel.send(claim)
    assert response["group_processing"]["routing"]["action"] == "private_switch"
    assert response["group_processing"]["claim_write_preflight"] is True
    assert authorizer.complete.call_count == 1
    payload = json.loads(authorizer.complete.call_args.args[0][-1]["content"])
    assert payload["current"]["quoted_message_id"] == board["message_id"]
    source_ids = {row["message_id"] for row in payload["history"]}
    assert {board["message_id"], "c-group"} <= source_ids
    assert not source_ids.intersection({"b-group", "b-private", "c-private", "native", "natural-claim"})

    joined, = [tool for tool in response["tool_results"] if tool["name"] == "join_game"]
    assert joined["called"] is authorized and joined["allowed"] is authorized
    assert any(p.customer_id == "c" for p in store.require_game(game.game_id).participants) is authorized
    assert guard.summary()["authorized_scopes"] == ({"participate": 1} if authorized else {})
    assert guard.healthy(expected_inputs=channel.guarded_agent_input_count())
    assert channel.guarded_agent_input_count() == 1
    reopened = SandboxChannel(runtime, room_id=ROOM, identities=MEMBERS, archive_path=channel.archive_path)
    assert reopened.guarded_agent_input_count() == 1
    assert reopened.send(claim)["delivery_deduplicated"]
    assert authorizer.complete.call_count == 1


def test_claim_archives_real_receipts_to_third_members_without_broadcast(sandbox, monkeypatch):
    from scripts import agent_runtime_app

    def fail_network(*args, **kwargs):
        pytest.fail("sandbox publication touched the WeChat bridge")

    monkeypatch.setattr(agent_runtime_app, "request_local_json", fail_network)
    channel, runtime, _ = sandbox
    store = runtime.store
    stamp = now()
    game, task = _board_game(channel, store, stamp)
    before = channel.publish_board(trace_id="before-due", sent_at=stamp)
    assert before["status"] == "no_due_task" and before["archive_outputs"] == []
    published = channel.publish_board(trace_id="publish", sent_at=task.due_at)
    assert published["status"] == "completed"
    delivery, = published["archive_outputs"]
    snapshot, = published["snapshots"]
    assert store.scheduled_tasks[task.task_id].status.value == "completed"
    assert store.get_board_snapshot_by_message_id(ROOM, delivery["message_id"]).to_dict() == snapshot
    assert delivery["trace_id"] == published["tasks"][0]["trace_id"]
    assert delivery["sender_id"] == AGENT_ID and delivery["direction"] == "outbound"
    assert delivery["text"] == snapshot["rendered_text"]
    assert "无烟" in delivery["text"] and MEMBERS["a"] not in delivery["text"]
    anchor = store.resolve_message_reference(conversation_id=channel.group_conversation_id,
                                             message_id=delivery["message_id"])
    assert anchor is not None and anchor.channel == store.get_group_room_policy(ROOM).channel
    assert anchor.business_ref_id == snapshot["snapshot_id"]
    assert channel.publish_board(trace_id="duplicate", sent_at=task.due_at)["archive_outputs"] == []

    message = MessageEnvelope("claim", "b", "1来", task.due_at + timedelta(seconds=1), "group", ROOM,
                              delivery["message_id"])
    response = channel.send(message)
    assert response["group_processing"]["routing"]["detail"]["status"] == "claimed"
    assert not store.require_game(game.game_id).remaining_seats()
    outputs = response["archive_outputs"]
    assert {output["recipient_id"] for output in outputs} == {"a", "b"}
    assert all(output["kind"] == "private" and output["source_message_id"] == "claim" for output in outputs)
    assert all(output["trace_id"] == response["trace_id"] for output in outputs)
    assert all(output["message_id"] != message.message_id for output in outputs)
    assert [event["message_id"] for event in channel.events_for("c")] == [delivery["message_id"], "claim"]
    assert [event["recipient_id"] for event in channel.events_for("b") if event["direction"] == "outbound"
            and event["kind"] == "private"] == [
        "b" for _ in range(sum(output["recipient_id"] == "b" for output in outputs))
    ]


def test_board_defers_unapproved_room_and_never_delivers_rejected_review(sandbox, monkeypatch):
    from mahjong_agent_runtime.customer_visible_review import CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME

    channel, runtime, _ = sandbox
    store = runtime.store
    _, task = _board_game(channel, store, now())
    policy = store.get_group_room_policy(ROOM)
    store.upsert_group_room_policy(replace(policy, board_enabled=False))
    unapproved = channel.publish_board(trace_id="no-board-permission", sent_at=task.due_at)
    deferred = store.scheduled_tasks[task.task_id]
    assert deferred.status.value == "pending" and deferred.due_at > task.due_at and deferred.attempts == 0
    assert not unapproved["archive_outputs"] and not unapproved["snapshots"]

    calls = []

    def deny_review(_service, **kwargs):
        calls.append(kwargs)
        return ToolResult(CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME, called=True, allowed=True,
                          result={"approved": False})

    runtime.reply_self_review_enabled = True
    monkeypatch.setattr(type(runtime.action_processor), "review_customer_visible_content", deny_review)
    store.upsert_group_room_policy(policy)
    rejected = channel.publish_board(trace_id="review-rejects", sent_at=deferred.due_at)
    assert not rejected["archive_outputs"] and not rejected["snapshots"]
    assert store.scheduled_tasks[task.task_id].status.value == "pending"
    assert store.scheduled_tasks[task.task_id].attempts == 1
    assert calls[0]["review_scope"] == "group_board"
    public = calls[0]["context_payload"]["active_game_visible_summaries"]
    assert public and all("game_id" not in item and MEMBERS["a"] not in str(item) for item in public)
    assert channel.events() == []


def test_board_retry_recovers_snapshot_without_resending_durable_delivery(sandbox, monkeypatch):
    channel, runtime, _ = sandbox
    store = runtime.store
    _, task = _board_game(channel, store, now())
    save = type(store).save_board_snapshot

    def fail_snapshot_write(_store, _snapshot):
        raise RuntimeError("injected snapshot failure after archive")

    monkeypatch.setattr(type(store), "save_board_snapshot", fail_snapshot_write)
    interrupted = channel.publish_board(trace_id="interrupted", sent_at=task.due_at)
    delivery, = interrupted["archive_outputs"]
    assert interrupted["status"] == "pending" and not interrupted["snapshots"]
    assert channel.events() == [delivery]
    assert store.get_board_snapshot_by_message_id(ROOM, delivery["message_id"]) is None

    monkeypatch.setattr(type(store), "save_board_snapshot", save)
    retried = channel.publish_board(trace_id="retry", sent_at=store.scheduled_tasks[task.task_id].due_at)
    assert retried["status"] == "completed" and not retried["archive_outputs"]
    assert retried["snapshots"][0]["external_message_id"] == delivery["message_id"]
    assert channel.events() == [delivery]


@pytest.mark.parametrize("decision", ["accepted", "declined"])
def test_late_arrival_notices_receive_scoped_private_receipts(sandbox, monkeypatch, decision):
    """原桌联系人和询局人各收一条受审私聊，交付失败后同文案重试且不能提前占席。"""
    from mahjong_agent_runtime.customer_visible_review import CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME
    from mahjong_agent_runtime.group_chat.models import GameConversationLink
    from mahjong_agent_runtime.late_arrival import late_arrival_context, late_arrival_delivery_receipt

    channel, runtime, client = sandbox
    store = runtime.store
    host, candidate = channel.private_conversation_id("a"), channel.private_conversation_id("b")
    at = now().replace(hour=18, minute=35, second=0, microsecond=0)
    arrival = at + timedelta(minutes=20)
    review_recipients = []
    texts = iter(["我先帮你问问五点半那桌。",
                  "五点半那桌，有人得六点五十五才到，能等吗？",
                  "五点半那桌能等到六点五十五，你还来吗？" if decision == "accepted" else
                  "五点半那桌等不了六点五十五，这次先不算。"])
    monkeypatch.setattr(type(client), "complete", lambda *_args, **_kwargs: json.dumps({"text": next(texts)}))

    def approve_review(_service, **kwargs):
        facts, = kwargs["context_payload"]["active_game_visible_summaries"]
        assert facts["candidate_arrival_time"] == "18:55" and facts["seat_reserved"] is False
        assert not {"candidate_id", "candidate_conversation_id", "required_contact_ids"} & facts.keys()
        item, = kwargs["review_items"]
        if item["text"] == "我先帮你问问五点半那桌。":
            return ToolResult(CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME, called=True, allowed=True,
                              result={"approved": False, "violations": ["unverified_external_action"]})
        review_recipients.append(item["recipient_id"])
        return ToolResult(CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME, called=True, allowed=True,
                          result={"approved": True})

    runtime.reply_self_review_enabled = True
    monkeypatch.setattr(type(runtime.action_processor), "review_customer_visible_content", approve_review)
    with using_clock(lambda: at):
        game, _ = store.create_game(conversation_id=host, organizer_id="a", organizer_name=MEMBERS["a"],
            requirement={"game_type": "hangzhou_mahjong", "start_time_kind": "scheduled", "start_time": "17:30",
                         "duration_kind": "fixed", "duration_hours": 4}, known_players=[], trace_id="creator")
        store.link_game_conversation(GameConversationLink(
            "late-table", game.game_id, host, ROOM, "a", "private_board_created",
        ))
        source = MessageEnvelope("late-ask", "b", "我六点五十五到，能帮我问问五点半那桌吗？", at, "private", AGENT_ID)
        request = channel.deliver_private_history(source)
        store.append_tool_turn(candidate, json.dumps([{
            "name": "search_current_games", "called": True, "allowed": True, "result": {
                "matches": [{"match_kind": "exact", "game": game.to_dict()}],
            },
        }]), request["trace_id"])
        questions, _ = store.ensure_late_arrival_questions(game_id=game.game_id, arrival_at=arrival,
            sender_id="b", conversation_id=candidate, trace_id=request["trace_id"], at=at)
    question, = questions
    assert late_arrival_delivery_receipt(store, question, at=at) is None
    assert not late_arrival_context(store, conversation_id=host, sender_id="a", at=at)

    save_reference = type(store).register_message_reference

    def interrupted_receipt(store, reference):
        raise RuntimeError("injected receipt persistence failure after sandbox delivery")

    monkeypatch.setattr(type(store), "register_message_reference", interrupted_receipt)
    first, = channel.deliver_late_arrival_notice(question.task_id, source_message_id="clock-question", sent_at=at)
    assert first["recipient_id"] == "a" and first["conversation_id"] == host
    assert late_arrival_delivery_receipt(store, question, at=at) is None
    retry_at, retry_id = channel.next_late_arrival_delivery()
    assert retry_id == question.task_id and retry_at > at

    monkeypatch.setattr(type(store), "register_message_reference", save_reference)
    assert channel.deliver_late_arrival_notice(question.task_id, source_message_id="clock-question-retry",
                                              sent_at=retry_at) == []
    receipt = late_arrival_delivery_receipt(store, question, at=retry_at)
    assert receipt.metadata["delivery_id"] == first["message_id"]
    pending, = late_arrival_context(store, conversation_id=host, sender_id="a",
                                    at=retry_at)["pending_late_arrival_decisions"]
    assert pending["decision_task_id"] == question.task_id
    assert [row["message_id"] for row in channel.events_for("a")] == [first["message_id"]]

    answer_at = retry_at + timedelta(minutes=1)
    with using_clock(lambda: answer_at):
        answer = MessageEnvelope("late-answer", "a", "行" if decision == "accepted" else "等不了",
                                 answer_at, "private", AGENT_ID)
        response = channel.deliver_private_history(answer)
        _, _, status = store.resolve_late_arrival_question(question.task_id, decision=decision, sender_id="a",
            conversation_id=host, evidence_message_id=answer.message_id, trace_id=response["trace_id"], at=answer_at)
    assert status == decision
    due, update = channel.next_late_arrival_delivery()
    notice, = channel.deliver_late_arrival_notice(update, source_message_id="clock-update", sent_at=due)
    assert notice["recipient_id"] == "b" and notice["conversation_id"] == candidate
    assert review_recipients == ["a", "b"] and channel.events_for("c") == []
    assert store.require_game(game.game_id).remaining_seats() == 3
    assert store.require_game(game.game_id).planned_start_at == game.planned_start_at


def test_failed_handler_keeps_input_and_does_not_reexecute(sandbox, monkeypatch):
    channel, _, client = sandbox
    anchor = now()
    request = MessageEnvelope("failed", "a", "还有位吗", anchor, "group", ROOM)

    def fail(_messages, *, trace_id, timeout_seconds):
        del trace_id, timeout_seconds
        assert [event["message_id"] for event in channel.events()] == ["failed"]
        raise RuntimeError("injected failure")

    monkeypatch.setattr(client, "complete", fail)
    with pytest.raises(Exception, match="injected failure"):
        channel.send(request)
    with pytest.raises(ValueError, match="receipt is failed"):
        channel.send(request)
    assert len(channel.events()) == 1
