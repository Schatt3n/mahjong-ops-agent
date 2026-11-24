"""让并行用户走同一个沙盒，核对真实运行时的会话隔离、投递和席位竞争。"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
from threading import Barrier, Event

import pytest

from mahjong_agent_runtime.group_chat import BOARD_TASK_TYPE, BoardEngine, GroupMessage
from mahjong_agent_runtime.models import DEFAULT_TZ, using_clock
from mahjong_agent_runtime.stores.sqlite import SQLiteAgentStore

from .group_transport import SimulationMessenger
from .sim_adapter import StaticAgentLLMClient, build_runtime
from .venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel

ROOM = "parallel-room"
STAMP = datetime(2031, 5, 5, 19, tzinfo=DEFAULT_TZ)
MEMBERS = {f"p{number:02}": f"牌友{number:02}" for number in range(16)}


def _channel(tmp_path, client=None):
    store = SQLiteAgentStore(tmp_path / "business.sqlite3")
    runtime, client = build_runtime(store, "mock", offline_client=client, local_coordination=True)
    channel = SandboxChannel(runtime, room_id=ROOM, identities=MEMBERS, archive_path=tmp_path / "chat.sqlite3")
    return channel, runtime, client


def _send(channel, message):
    """线程池不继承 ContextVar，使用本轮固定的模拟时钟执行业务写入。"""
    with using_clock(lambda: message.sent_at):
        return channel.send(message)


def _same_time(channel, messages):
    """屏障证明同批请求同时就绪，断言使用持久收据而不是完成顺序。"""
    workers = len(messages)
    ready = Barrier(workers, timeout=10)

    def dispatch(message):
        ready.wait()
        return _send(channel, message)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(dispatch, messages, timeout=30))


def test_parallel_group_queries_and_private_inboxes_are_complete(tmp_path):
    channel, runtime, client = _channel(tmp_path)
    messages = [
        MessageEnvelope(f"group-{member}", member, "老板，这会儿还有麻将局吗", STAMP, "group", ROOM)
        for member in MEMBERS
    ] + [
        MessageEnvelope(f"dm-{member}", member, f"只有本人可见的留言【{member}】，这会儿有位吗", STAMP, "private", AGENT_ID)
        for member in MEMBERS
    ]
    responses = _same_time(channel, messages)
    incoming = channel.events()
    assert client.semantic_call_count == len(MEMBERS)
    assert len(incoming) == len(messages) * 2
    assert len({event["message_id"] for event in incoming}) == len(incoming)
    assert {event["message_id"] for event in incoming if event["direction"] == "inbound"} == {
        message.message_id for message in messages
    }
    assert Counter(event["source_message_id"] for event in incoming if event["direction"] == "outbound") == {
        message.message_id: 1 for message in messages
    }
    for message, response in zip(messages, responses, strict=True):
        assert response["status"] == "completed"
        delivered, = response["archive_outputs"]
        assert delivered["recipient_id"] == (ROOM if message.kind == "group" else message.sender_id)
        assert delivered["trace_id"] == response["trace_id"]
        trace = runtime.trace_recorder.get_trace(response["trace_id"])
        assert any(event.step == "final_output" for event in trace)
        assert {event.occurred_at for event in trace} == {STAMP.strftime("%Y-%m-%d %H:%M:%S")}
        if message.kind == "private":
            personal = [row for row in channel.events_for(message.sender_id) if row["kind"] == "private"]
            assert [row["message_id"] for row in personal] == [message.message_id, delivered["message_id"]]
        else:
            assert all(delivered in channel.events_for(member) for member in MEMBERS)
            assert all("只有本人可见的留言" not in str(event.content) for event in trace if event.step == "context_built")


class _BlockingClient(StaticAgentLLMClient):
    """按消息 ID 阻塞一个模型请求，通过事件说明实际重叠和串行边界。"""

    def __init__(self):
        super().__init__()
        self.first_entered = Event()
        self.release_first = Event()
        self.next_entered = Event()
        self.other_entered = Event()

    def complete(self, messages, *, trace_id, timeout_seconds):
        payload = self._context_payload(messages)
        message_id = (payload.get("current_message") or {}).get("message_id")
        if message_id == "first" and not self.first_entered.is_set():
            self.first_entered.set()
            if not self.release_first.wait(10):
                raise RuntimeError("test did not release the first request")
        elif message_id == "next":
            self.next_entered.set()
        elif message_id == "other":
            self.other_entered.set()
        return super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)


def test_same_dm_serializes_model_while_another_conversation_overlaps(tmp_path, monkeypatch):
    client = _BlockingClient()
    channel, runtime, _ = _channel(tmp_path, client)
    received_next = Event()
    receive = type(runtime).receive_user_message

    def observe_receipt(self, message, **kwargs):
        receipt = receive(self, message, **kwargs)
        if self is runtime and message.message_id == "next":
            received_next.set()
        return receipt

    monkeypatch.setattr(type(runtime), "receive_user_message", observe_receipt)
    first = MessageEnvelope("first", "p00", "老板，今天有什么牌局", STAMP, "private", AGENT_ID)
    second = replace(first, message_id="next", text="我刚才问的是无烟局", sent_at=STAMP + timedelta(seconds=1))
    other = replace(first, message_id="other", sender_id="p01", text="老板今天我想找一局")
    with ThreadPoolExecutor(max_workers=3) as pool:
        in_flight = pool.submit(_send, channel, first)
        try:
            assert client.first_entered.wait(8)
            later = pool.submit(_send, channel, second)
            assert received_next.wait(8)
            independent = pool.submit(_send, channel, other)
            assert client.other_entered.wait(8)
            assert independent.result(timeout=8)["archive_outputs"]
            assert not client.next_entered.is_set()
        finally:
            client.release_first.set()
        first_result = in_flight.result(timeout=8)
        second_result = later.result(timeout=8)
    assert first_result["status"] == "superseded" and not first_result["archive_outputs"]
    assert client.next_entered.is_set()
    reply, = second_result["archive_outputs"]
    assert [row["message_id"] for row in channel.events_for("p00")] == ["first", "next", reply["message_id"]]


def test_concurrent_duplicate_waits_for_actual_receipt_and_rejects_collision(tmp_path, monkeypatch):
    client = _BlockingClient()
    channel, _, _ = _channel(tmp_path, client)
    waiting = Event()
    enter = SandboxChannel._receipt_lock
    lock_calls = Counter()

    def count_lock_attempts(self, message_id):
        # 回执锁调用发生在真正阻塞之前，可以确定重投已经进入通道。
        lock_calls[message_id] += 1
        if lock_calls[message_id] >= 2:
            waiting.set()
        return enter(self, message_id)

    monkeypatch.setattr(SandboxChannel, "_receipt_lock", count_lock_attempts)
    request = MessageEnvelope("first", "p00", "老板还有麻将局吗", STAMP, "private", AGENT_ID)
    with ThreadPoolExecutor(max_workers=3) as pool:
        original = pool.submit(_send, channel, request)
        try:
            assert client.first_entered.wait(8)
            duplicate = pool.submit(_send, channel, request)
            assert waiting.wait(8)
            collision = pool.submit(_send, channel, replace(request, sender_id="p01"))
        finally:
            client.release_first.set()
        response = original.result(timeout=8)
        assert duplicate.result(timeout=8) == {**response, "delivery_deduplicated": True}
        with pytest.raises(ValueError, match="message id conflict"):
            collision.result(timeout=8)
    assert client.call_count == 2
    assert [row["direction"] for row in channel.events()] == ["inbound", "outbound"]
    assert not channel.events_for("p01")


def test_parallel_quoted_claims_cannot_overfill_last_seat(tmp_path):
    channel, runtime, _ = _channel(tmp_path)
    store = runtime.store
    channel_id = store.get_group_room_policy(ROOM).channel
    engine = BoardEngine(store=store, messenger=SimulationMessenger(store, ROOM, channel=channel_id))
    with using_clock(lambda: STAMP):
        game = engine.import_game_from_post(GroupMessage(
            room_id=ROOM, conversation_id=channel.group_conversation_id, channel=channel_id,
            sender_external_id="p00", sender_name=MEMBERS["p00"], sent_at=STAMP,
            text="2031-05-05 20:00 杭麻 0.5 无烟 371", message_id="open-game",
        ), trace_id="board-seed")
    task, = [task for task in store.scheduled_tasks.values() if task.task_type == BOARD_TASK_TYPE]
    board, = channel.publish_board(trace_id="board-publish", sent_at=task.due_at)["archive_outputs"]
    claim_at = task.due_at + timedelta(seconds=1)
    requests = [
        MessageEnvelope(f"claim-{member}", member, "1来", claim_at, "group", ROOM, board["message_id"])
        for member in MEMBERS if member != "p00"
    ]
    responses = _same_time(channel, requests)
    winners = [
        message.sender_id for message, response in zip(requests, responses, strict=True)
        if response["group_processing"]["routing"]["detail"]["status"] == "claimed"
    ]
    winner, = winners
    persisted = store.require_game(game.game_id)
    assert persisted.remaining_seats() == 0
    assert {member.customer_id for member in persisted.participants} == {"p00", winner}
    assert sum(store.get_game_claim_by_source(channel.group_conversation_id, message.message_id) is not None
               for message in requests) == 1
    visible = channel.events()
    assert len([row for row in visible if row["direction"] == "inbound"]) == len(requests)
    assert len({row["message_id"] for row in visible}) == len(visible)
    for member in MEMBERS:
        assert all(row["recipient_id"] == member for row in channel.events_for(member) if row["kind"] == "private")
