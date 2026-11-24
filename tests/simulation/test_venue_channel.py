"""验证模拟档案与正式入口的交互、持久回执和会话隐私。"""

from dataclasses import replace
from datetime import timedelta

import pytest

from mahjong_agent_runtime.group_chat import BOARD_TASK_TYPE, BoardEngine, GroupMessage
from mahjong_agent_runtime.models import ToolResult, now
from mahjong_agent_runtime.stores.sqlite import SQLiteAgentStore

from .group_transport import SimulationMessenger
from .sim_adapter import build_runtime
from .venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel

MEMBERS = {"a": "阿岚", "b": "阿白", "c": "阿诚"}
ROOM = "sandbox-room"


@pytest.fixture
def sandbox(tmp_path):
    store = SQLiteAgentStore(tmp_path / "business.sqlite3")
    runtime, client = build_runtime(store, "mock")
    channel = SandboxChannel(runtime, room_id=ROOM, identities=MEMBERS, archive_path=tmp_path / "messages.sqlite3")
    return channel, runtime, client


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


def test_write_authorization_gets_only_the_actual_shop_dialogue_once(sandbox):
    """授权检查随实际私聊入口运行，只收到同一顾客与店家的已见消息。"""
    channel, runtime, _ = sandbox
    calls = []

    class Guard:
        def authorize(self, message, conversation_id, history):
            calls.append((message.message_id, conversation_id, [row["message_id"] for row in history]))

    guarded = SandboxChannel(runtime, room_id=ROOM, identities=MEMBERS, archive_path=channel.archive_path,
                             write_guard=Guard())
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


def _board_game(channel, store, stamp):
    channel_id = store.get_group_room_policy(ROOM).channel
    messenger = SimulationMessenger(store, ROOM, channel=channel_id)
    board = BoardEngine(store=store, messenger=messenger, clock=lambda: stamp)
    game = board.import_game_from_post(GroupMessage(
        room_id=ROOM, conversation_id=channel.group_conversation_id, channel=channel_id,
        sender_external_id="a", sender_name=MEMBERS["a"], sent_at=stamp,
        text=f"{stamp + timedelta(hours=1):%Y-%m-%d %H:%M} 0.5 无烟 371", message_id="original-post",
    ), trace_id="post")
    task, = [task for task in store.scheduled_tasks.values() if task.task_type == BOARD_TASK_TYPE]
    assert not messenger.outputs
    return game, task


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
