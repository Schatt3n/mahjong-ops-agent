from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from threading import Event

import pytest

from mahjong_agent_runtime import (
    AgentRuntime,
    InMemoryAgentStore,
    SQLiteAgentStore,
    StaticAgentClient,
    ToolGateway,
)
from mahjong_agent_runtime.channel_ingress import ChannelIngressWorker
from mahjong_agent_runtime.coordination import InProcessCoordinationManager


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "scripts" / "agent_runtime_app.py"


def load_app_module():
    spec = importlib.util.spec_from_file_location("agent_runtime_channel_observation_test", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_deferred_reply_leaves_route_slot_free_for_other_conversation(tmp_path, backend) -> None:
    store = InMemoryAgentStore() if backend == "memory" else SQLiteAgentStore(tmp_path / "ingress.sqlite3")
    sending_a = Event()
    release_a = Event()
    routed_b = Event()
    sent_b = Event()
    routed_next_a = Event()

    def enqueue(source: str, conversation: str, route_mode: str = "ingress_queue") -> None:
        store.enqueue_channel_ingress({
            "channel": "agent-wechat",
            "source_message_id": source,
            "conversation_id": conversation,
            "route_mode": route_mode,
            "payload": {},
        })

    def handle(record: dict) -> None:
        source = record["source_message_id"]
        if source == "a-reply":
            sending_a.set()
            assert release_a.wait(5)
            store.upsert_channel_observation({**record, "route_mode": "agent"})
            return
        if source == "b" and record["route_mode"] == "ingress_queue":
            store.upsert_channel_observation({**record, "route_mode": "ingress_reply_pending"})
            routed_b.set()
            return
        store.upsert_channel_observation({**record, "route_mode": "agent"})
        (sent_b if source == "b" else routed_next_a).set()

    enqueue("a-reply", "a", "ingress_reply_pending")
    worker = ChannelIngressWorker(
        store=store, coordination=InProcessCoordinationManager(), handler=handle, max_workers=1,
    )
    worker.start()
    try:
        assert sending_a.wait(2)
        enqueue("a-next", "a")
        enqueue("b", "b")
        worker.wake()
        assert routed_b.wait(2)
        assert not routed_next_a.is_set()
        assert not sent_b.is_set()
        release_a.set()
        assert sent_b.wait(2)
        assert routed_next_a.wait(2)
    finally:
        release_a.set()
        worker.stop()


def test_sqlite_channel_observation_upsert_is_idempotent_and_queryable(tmp_path) -> None:
    store = SQLiteAgentStore(tmp_path / "channel_observations.sqlite3")
    received = {
        "channel": "wechaty",
        "source_message_id": "msg_001",
        "trace_id": "trace_msg_001",
        "conversation_id": "wechaty:room:@@room_001",
        "room_id": "@@room_001",
        "room_topic": "大牌计划·24H自助棋牌",
        "sender_id": "@member_001",
        "sender_name": "群友",
        "text": "今晚七点 0.5 无烟 371",
        "message_type": "7",
        "is_room": True,
        "self_message": False,
        "route_status": "received",
        "route_mode": "pending",
        "route_reason": "",
        "semantic_action": "",
        "semantic_category": "",
        "semantic_confidence": 0.0,
        "business_message_detected": False,
        "received_at": "2026-07-22 10:00:00",
        "payload": {"raw_payload": {"text": "今晚七点 0.5 无烟 371"}},
    }

    store.upsert_channel_observation(received)
    store.upsert_channel_observation(
        {
            **received,
            "route_status": "analyzed",
            "route_mode": "observe_only",
            "route_reason": "observe_only_room_analyzed",
            "semantic_action": "process_business",
            "semantic_category": "operational",
            "semantic_confidence": 0.97,
            "business_message_detected": True,
            "payload": {
                "raw_payload": {"text": "今晚七点 0.5 无烟 371"},
                "route_result": {"observe_only": True},
            },
        }
    )

    records = store.list_channel_observations(
        channel="wechaty",
        room_topic_keyword="大牌计划",
        limit=10,
    )

    assert len(records) == 1
    assert records[0]["source_message_id"] == "msg_001"
    assert records[0]["route_status"] == "analyzed"
    assert records[0]["route_mode"] == "observe_only"
    assert records[0]["semantic_action"] == "process_business"
    assert records[0]["business_message_detected"] is True
    assert records[0]["payload"]["route_result"]["observe_only"] is True


def test_ingest_wechaty_raw_message_persists_observe_only_decision(monkeypatch, tmp_path) -> None:
    app = load_app_module()
    config_path = tmp_path / "observe_only_rooms.json"
    config_path.write_text(
        json.dumps(
            {
                "room_ids": ["@@room_observe"],
                "topic_keywords": ["大牌计划"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app, "WECHATY_RAW_LOG_PATH", tmp_path / "wechaty_raw.jsonl")
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "self_only")
    monkeypatch.setenv("MAHJONG_WECHATY_OBSERVE_ONLY_ROOMS_PATH", str(config_path))
    monkeypatch.delenv("MAHJONG_WECHATY_INPUT_GATE_LLM_MODEL", raising=False)
    store = InMemoryAgentStore()
    runtime = AgentRuntime(
        llm_client=StaticAgentClient(
            outputs=[
                json.dumps(
                    {
                        "action": "process_business",
                        "should_route": True,
                        "category": "operational",
                        "confidence": 0.96,
                        "reasoning_summary": "明确报局消息。",
                        "evidence": ["0.5", "无烟", "371"],
                    },
                    ensure_ascii=False,
                )
            ]
        ),
        store=store,
        tool_gateway=ToolGateway(store=store),
    )
    monkeypatch.setattr(app, "get_runtime", lambda: runtime)
    payload = {
        "conversation_id": "wechaty:room:@@room_observe",
        "sender_id": "@room_member",
        "sender_name": "群友",
        "message_id": "msg_observe_persist_001",
        "text": "今晚七点 0.5 无烟 371",
        "message_type": 7,
        "self_message": False,
        "is_room": True,
        "room": {"id": "@@room_observe", "topic": "大牌计划·24H自助棋牌"},
        "payload": {"roomId": "@@room_observe", "talkerId": "@room_member"},
    }

    result = app.ingest_wechaty_raw_message(payload)
    records = store.list_channel_observations(channel="wechaty", limit=10)

    assert result["route_result"]["observe_only"] is True
    assert len(records) == 1
    assert records[0]["route_status"] == "analyzed"
    assert records[0]["route_mode"] == "observe_only"
    assert records[0]["semantic_action"] == "process_business"
    assert records[0]["business_message_detected"] is True
    assert records[0]["payload"]["raw_payload"]["text"] == payload["text"]


def test_channel_observation_archive_survives_runtime_state_clear(tmp_path) -> None:
    store = SQLiteAgentStore(tmp_path / "channel_observation_retention.sqlite3")
    store.upsert_channel_observation(
        {
            "channel": "wechaty",
            "source_message_id": "msg_retained",
            "trace_id": "trace_retained",
            "conversation_id": "wechaty:room:@@room_retained",
            "room_id": "@@room_retained",
            "room_topic": "超大牌24H棋牌室",
            "sender_id": "@member",
            "sender_name": "群友",
            "text": "1块还有吗",
            "message_type": "7",
            "is_room": True,
            "self_message": False,
            "route_status": "analyzed",
            "route_mode": "observe_only",
            "route_reason": "observe_only_room_analyzed",
            "semantic_action": "process_business",
            "semantic_category": "operational",
            "semantic_confidence": 0.94,
            "business_message_detected": True,
            "received_at": "2026-07-22 10:10:00",
            "payload": {},
        }
    )

    store.clear_runtime_state()

    records = store.list_channel_observations(channel="wechaty", limit=10)
    assert [item["source_message_id"] for item in records] == ["msg_retained"]


def test_ingest_keeps_observation_when_semantic_routing_fails(monkeypatch, tmp_path) -> None:
    app = load_app_module()
    config_path = tmp_path / "observe_only_rooms.json"
    config_path.write_text(
        json.dumps({"room_ids": ["@@room_failure"], "topic_keywords": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(app, "WECHATY_RAW_LOG_PATH", tmp_path / "wechaty_failure_raw.jsonl")
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "self_only")
    monkeypatch.setenv("MAHJONG_WECHATY_OBSERVE_ONLY_ROOMS_PATH", str(config_path))
    monkeypatch.delenv("MAHJONG_WECHATY_INPUT_GATE_LLM_MODEL", raising=False)
    store = InMemoryAgentStore()
    runtime = AgentRuntime(
        llm_client=StaticAgentClient(outputs=[]),
        store=store,
        tool_gateway=ToolGateway(store=store),
    )
    monkeypatch.setattr(app, "get_runtime", lambda: runtime)

    def fail_route(*_args, **_kwargs):
        raise RuntimeError("semantic route unavailable")

    monkeypatch.setattr(app, "route_wechaty_raw_to_agent", fail_route)
    payload = {
        "conversation_id": "wechaty:room:@@room_failure",
        "sender_id": "@room_member",
        "sender_name": "群友",
        "message_id": "msg_observe_failure_001",
        "text": "今晚一块还有吗",
        "message_type": 7,
        "self_message": False,
        "is_room": True,
        "room": {"id": "@@room_failure", "topic": "超大牌24H棋牌室"},
        "payload": {"roomId": "@@room_failure", "talkerId": "@room_member"},
    }

    with pytest.raises(RuntimeError):
        app.ingest_wechaty_raw_message(payload)

    records = store.list_channel_observations(channel="wechaty", limit=10)
    assert len(records) == 1
    assert records[0]["route_status"] == "route_failed"
    assert records[0]["route_mode"] == "failed"
    assert records[0]["payload"]["raw_payload"]["text"] == payload["text"]
    assert records[0]["payload"]["route_error"]["type"] == "RuntimeError"
