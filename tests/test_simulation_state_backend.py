from __future__ import annotations

import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


SIMULATION_DIR = Path(__file__).resolve().parent / "simulation"
if str(SIMULATION_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATION_DIR))

from sim_state import (  # noqa: E402
    InboxMessage,
    InMemorySimulationStateBackend,
    RedisSimulationStateBackend,
    ReplyGate,
)


def _message(recipient_id: str, *, thread_id: str = "thread-a") -> InboxMessage:
    return InboxMessage(
        recipient_id=recipient_id,
        sender="mahjong_agent",
        text="有个三缺一，打吗？",
        trace_id="trace-1",
        source_message_id="message-1",
        channel="group",
        received_at=time.time(),
        conversation_id="group-1",
        thread_id=thread_id,
    )


def _gate(thread_id: str, user_id: str, *, acquired_at: float | None = None) -> ReplyGate:
    return ReplyGate(
        conversation_id="group-1",
        thread_id=thread_id,
        expected_user_id=user_id,
        source_message_id=f"message-{thread_id}",
        acquired_at=time.time() if acquired_at is None else acquired_at,
    )


def _assert_backend_contract(backend) -> None:
    backend.append_inboxes(
        [
            _message("user-a"),
            _message("user-b", thread_id="thread-b"),
        ]
    )
    assert backend.inbox_for("user-a")[0].thread_id == "thread-a"
    assert backend.inbox_sizes() == {"user-a": 1, "user-b": 1}

    first = _gate("thread-a", "user-a", acquired_at=time.time() - 20)
    second = _gate("thread-b", "user-b")
    backend.set_reply_gate(first)
    backend.set_reply_gate(second)
    assert backend.reply_gate("group-1", "thread-a") == first
    assert backend.reply_gate("group-1", "thread-b") == second
    assert backend.expired_reply_gates(10) == [first]

    assert backend.release_reply_gate(
        "group-1",
        "thread-a",
        expected_user_id="wrong-user",
    ) is False
    assert backend.release_reply_gate(
        "group-1",
        "thread-a",
        expected_user_id="user-a",
    ) is True
    assert backend.reply_gate("group-1", "thread-a") is None
    assert backend.reply_gate("group-1", "thread-b") == second

    event_id = backend.publish_event(
        "agent_message",
        {"conversation_id": "group-1", "thread_id": "thread-b", "text": "打吗？"},
    )
    assert event_id
    assert backend.recent_events(1)[0]["thread_id"] == "thread-b"
    assert backend.recent_events(0) == []


def test_in_memory_simulation_state_backend_keeps_topics_isolated() -> None:
    backend = InMemorySimulationStateBackend(["user-a", "user-b"])
    _assert_backend_contract(backend)


@pytest.fixture
def local_redis_url(tmp_path: Path):
    redis = pytest.importorskip("redis")
    executable = shutil.which("redis-server")
    if executable is None:
        pytest.skip("redis-server is not installed")
    port = 16_000 + (uuid.uuid4().int % 2_000)
    process = subprocess.Popen(
        [
            executable,
            "--bind",
            "127.0.0.1",
            "--port",
            str(port),
            "--save",
            "",
            "--appendonly",
            "no",
            "--dir",
            str(tmp_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"redis://127.0.0.1:{port}/15"
    client = redis.Redis.from_url(url, decode_responses=True)
    try:
        for _ in range(50):
            try:
                if client.ping():
                    break
            except redis.RedisError:
                time.sleep(0.02)
        else:
            pytest.fail("local redis-server did not start")
        yield url
    finally:
        try:
            client.shutdown(nosave=True)
        except redis.RedisError:
            pass
        process.wait(timeout=5)


@pytest.mark.integration
def test_local_redis_simulation_state_backend_is_atomic_and_observable(
    local_redis_url: str,
) -> None:
    backend = RedisSimulationStateBackend(
        local_redis_url,
        ["user-a", "user-b"],
        namespace=f"test-{uuid.uuid4().hex}",
    )
    _assert_backend_contract(backend)

    gates = [_gate(f"parallel-{index}", f"user-{index % 2}") for index in range(20)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(backend.set_reply_gate, gates))

    assert all(
        backend.reply_gate("group-1", gate.thread_id) == gate
        for gate in gates
    )
    assert backend.client.ttl(backend._inbox_key("user-a")) > 0
    assert backend.client.ttl(backend._gates_key) > 0
    assert backend.client.ttl(backend._events_key) > 0
