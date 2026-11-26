from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from mahjong_agent_runtime import AgentRuntime, InMemoryAgentStore, ScheduledAgentTask, StaticAgentClient
from mahjong_agent_runtime.group_chat import BOARD_TASK_TYPE, GroupHandleResult, GroupRoomPolicy


ROOT = Path(__file__).resolve().parents[2]
APP_PATH = ROOT / "scripts" / "agent_runtime_app.py"
SPEC = importlib.util.spec_from_file_location("agent_runtime_app_for_group_test", APP_PATH)
assert SPEC is not None and SPEC.loader is not None
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)

TZ = ZoneInfo("Asia/Shanghai")


class FakeManagedHandler:
    def __init__(self) -> None:
        self.messages = []

    def handle(self, message, *, trace_id: str):
        self.messages.append((message, trace_id))
        return GroupHandleResult(action="board_import", game_id="game-managed")


class FakeBoardEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def publish(self, room_id: str, *, trace_id: str, publication_id: str | None = None):
        self.calls.append((room_id, trace_id))
        return None


class FakeHandlerWithBoard:
    def __init__(self) -> None:
        self.board_engine = FakeBoardEngine()


def _managed_config(tmp_path: Path, *, room_id: str = "@@managed") -> Path:
    path = tmp_path / "managed_rooms.json"
    path.write_text(
        json.dumps(
            {
                "room_ids": [room_id],
                "topic_keywords": [],
                "outbound_enabled": False,
                "board_enabled": True,
                "merge_window_seconds": 30,
                "self_recruitment_default_game_types": {
                    room_id: "hangzhou_mahjong", "@@unmanaged": "national_standard_mahjong", "@@invalid": "unknown",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_managed_room_bypasses_self_only_and_uses_group_handler(monkeypatch, tmp_path) -> None:
    config_path = _managed_config(tmp_path)
    observe_path = tmp_path / "observe.json"
    observe_path.write_text(json.dumps({"room_ids": [], "topic_keywords": []}), encoding="utf-8")
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "self_only")
    monkeypatch.setenv("MAHJONG_WECHATY_MANAGED_ROOMS_PATH", str(config_path))
    monkeypatch.setenv("MAHJONG_WECHATY_OBSERVE_ONLY_ROOMS_PATH", str(observe_path))
    runtime = AgentRuntime(llm_client=StaticAgentClient(outputs=[]), store=InMemoryAgentStore())
    app.synchronize_managed_room_policies(runtime.store)
    assert runtime.store.get_group_room_policy("@@managed").self_recruitment_default_game_type == "hangzhou_mahjong"
    assert runtime.store.get_group_room_policy("@@unmanaged") is None
    assert "@@invalid" not in app.configured_wechaty_managed_rooms()["self_recruitment_default_game_types"]
    handler = FakeManagedHandler()
    monkeypatch.setattr(app, "get_runtime", lambda: runtime)
    monkeypatch.setattr(app, "get_group_message_handler", lambda _runtime, **kwargs: handler)

    result = app.route_wechaty_raw_to_agent(
        {
            "conversation_id": "wechaty:room:@@managed",
            "sender_id": "@member",
            "sender_name": "老板备注不应使用",
            "message_id": "msg-managed",
            "text": "14:00 0.5 无烟 371",
            "self_message": False,
            "is_room": True,
            "room": {"id": "@@managed", "topic": "托管棋牌群"},
            "talker": {
                "id": "@member",
                "name": "公开昵称",
                "alias": "老板私密备注",
                "payload": {"friend": True},
            },
            "payload": {"roomId": "@@managed", "talkerId": "@member"},
        },
        trace_id="trace-managed",
    )

    assert result["managed_group"] is True
    assert result["group_result"]["action"] == "board_import"
    assert len(handler.messages) == 1
    message, trace_id = handler.messages[0]
    assert trace_id == "trace-managed"
    assert message.sender_name == "公开昵称"
    assert runtime.store.get_group_room_policy("@@managed").self_recruitment_default_game_type == "hangzhou_mahjong"
    identity = runtime.store.get_channel_identity("wechaty", "@member")
    assert identity is not None
    assert identity.public_name == "公开昵称"
    assert identity.can_private_message is True


@pytest.mark.parametrize("current_scope, transport", [
    ("automatic", "wechaty"), ("owner_claimed", "wechaty"), ("revoked", "wechaty"), ("new_message", "wechaty"),
    ("automatic", "bridge"), ("revision_changed", "bridge"), ("automatic", "agent-wechat"),
])
def test_managed_group_timeout_reuses_original_raw_in_scheduled_recovery(monkeypatch, tmp_path, current_scope, transport):
    """原信封只能在仍授权且店家未接手时重分类，正常群策略继续决定是否外发。"""
    from mahjong_agent_runtime import SQLiteAgentStore
    from mahjong_agent_runtime.tracing import InMemoryTraceRecorder

    native = transport == "agent-wechat"
    bridge = transport == "bridge"
    channel = "agent-wechat" if native else "wechaty"
    room_id = "123@chatroom" if native else "@@managed"
    config_path = _managed_config(tmp_path, room_id=room_id)
    monkeypatch.setenv("MAHJONG_WECHATY_MANAGED_ROOMS_PATH", str(config_path))
    if native:
        monkeypatch.setattr(app, "agent_wechat_group_permission_rule", lambda *_args: {})
    if bridge:
        config_path.write_text(json.dumps({"room_ids": [], "topic_keywords": []}), encoding="utf-8")
        monkeypatch.setenv("MAHJONG_AGENT_WECHAT_ACCOUNT", "shop-owner")
        setup = {"account": "shop-owner", "revision": "r1", "rooms": {room_id: "managed"}, "contacts": []}
        monkeypatch.setattr(app, "agent_wechat_setup", lambda: setup)
    monkeypatch.setattr(app, "TRACE_PATH", tmp_path / "trace.log")
    monkeypatch.setattr(app, "channel_raw_log_path", lambda _channel: tmp_path / "raw.jsonl")
    monkeypatch.setattr(app, "request_local_json", lambda *_args, **_kwargs: pytest.fail("群里禁止外发"))
    messages = []

    class TopicClient:
        def complete(self, inputs, *, trace_id, timeout_seconds):
            src = json.loads(inputs[1]["content"])["message"]
            messages.append(src)
            if len(messages) == 1:
                raise TimeoutError("classification input delivery read timeout")
            if current_scope == "owner_claimed":
                handoff, = app.owner_agent_handoffs(runtime)
                assert handoff["retrying"] and handoff["request"] == raw["text"]
                result, status = app.claim_owner_agent_handoff(runtime, handoff)
                assert result == {"ok": True} and status == 200
            return json.dumps({"intent": "member_discussion", "topic_action": "none", "topic_id": None,
                "goal_relation": "none", "task_context_id": None, "public_summary": {}, "constraints": {},
                "source_message_id": src["message_id"], "confidence": 1, "reasoning": "固定测试分类"})

    store = SQLiteAgentStore(tmp_path / "business.sqlite3")
    runtime = AgentRuntime(llm_client=TopicClient(), store=store, trace_recorder=InMemoryTraceRecorder())
    monkeypatch.setattr(app, "get_runtime", lambda: runtime)
    raw = {"message_id": "managed-timeout", "conversation_id": f"{app.native_conversation_namespace(channel)}:room:{room_id}",
           "sender_id": "@member", "sender_name": "群成员", "text": "今晚有没有人打呀？",
           "is_room": True, "room": {"id": room_id, "topic": "托管群"},
           "talker": {"name": "群成员", "friend": True}}
    if bridge:
        raw.update(platform_name="agent-wechat", agent_wechat_account="shop-owner", agent_wechat_revision="r1",
                   agent_wechat_primary_account="shop-owner")
    if native:
        raw["payload"] = {"account_id": "shop-owner", "threadId": room_id, "direction": "incoming",
                          "createdAt": "2026-11-26T11:39:30+08:00"}
    stored_room_id = app.bridge_room_key("shop-owner", room_id) if bridge else room_id
    if native:
        assert app.queue_agent_wechat_raw(runtime, raw, deferred_reply=True)["accepted"]
        app.process_queued_channel_raw(runtime, store.get_channel_observation(channel, raw["message_id"]))
    else:
        first = app.ingest_channel_raw_message(channel, raw)
        assert first["route_result"]["group_result"]["action"] == "group_topic_retry_queued"
    initial_observation = store.get_channel_observation(channel, raw["message_id"])
    assert initial_observation["route_mode"] == "group_topic_retry"
    if native:
        assert app.queue_agent_wechat_raw(runtime, raw, deferred_reply=True)["deduplicated"] is True
    else:
        assert app.ingest_channel_raw_message(channel, raw)["deduplicated"] is True
    assert store.get_channel_observation(channel, raw["message_id"]) == initial_observation
    batch, = store.pending_input_gate_handoffs()
    assert batch.decision["gate_recovery"]["wait_delivered"] is False
    assert not store.get_group_topic_message(channel, stored_room_id, "@member", raw["message_id"])
    current_identity = store.get_channel_identity(channel, "@member")
    assert current_identity.can_private_message
    store.upsert_channel_identity(replace(current_identity, is_friend=False, can_private_message=False))

    if current_scope == "revoked":
        config_path.write_text(json.dumps({"room_ids": [], "topic_keywords": []}), encoding="utf-8")
    if current_scope == "new_message":
        app.ingest_channel_raw_message(channel, {**raw, "message_id": "managed-new", "text": "不问啦"})
    if current_scope == "revision_changed":
        setup["revision"] = "r2"
    app.handle_due_pending_input_batch(batch, "due-callback")
    app.handle_due_pending_input_batch(batch, "duplicate-due-callback")

    record = store.get_group_topic_message(channel, stored_room_id, "@member", raw["message_id"])
    if current_scope == "automatic":
        assert len(messages) == 2 and messages[0] == messages[1]
        assert record["classification"]["intent"] == "member_discussion"
        assert store.get_channel_observation(channel, raw["message_id"])["route_mode"] == "managed_group"
    else:
        assert record is None
        assert len(messages) == (1 if current_scope in {"revoked", "revision_changed"} else 2)
    assert not store.games
    if current_scope != "new_message":
        assert not store.get_channel_identity(channel, "@member").can_private_message
    if current_scope in {"revoked", "revision_changed"}:
        item, = app.owner_agent_handoffs(runtime)
        assert item["room"] == "托管群" and not item["retrying"]
        assert store.get_channel_observation(channel, raw["message_id"])["route_mode"] == "group_topic_needs_human"
    else:
        assert app.owner_agent_handoffs(runtime) == []
    assert store.get_channel_observation(channel, raw["message_id"])["received_at"] == initial_observation["received_at"]


def test_recovered_native_managed_group_needs_original_authorized_delivery_callback(monkeypatch, tmp_path):
    """原生群恢复后继续使用原消息授权；生成文本只有原通道确认后才有送达号。"""
    from mahjong_agent_runtime import AgentRuntimeResult, SQLiteAgentStore
    from mahjong_agent_runtime.tracing import InMemoryTraceRecorder

    room_id = "123@chatroom"
    channel = "agent-wechat"
    monkeypatch.setenv("MAHJONG_WECHATY_MANAGED_ROOMS_PATH", str(_managed_config(tmp_path, room_id=room_id)))
    monkeypatch.setattr(app, "TRACE_PATH", tmp_path / "trace.log")
    monkeypatch.setattr(app, "channel_raw_log_path", lambda _channel: tmp_path / "raw.jsonl")
    allowed_at = "2026-11-26T10:00:00+08:00"
    monkeypatch.setattr(app, "agent_wechat_group_permission_rule", lambda *_args: {"mode": "allowlist",
        "thread_id": room_id, "managed": True, "managed_at": allowed_at,
        "auto_reply": True, "auto_reply_at": allowed_at})
    routed = []

    class TopicClient:
        def complete(self, inputs, *, trace_id, timeout_seconds):
            source = json.loads(inputs[1]["content"])["message"]
            routed.append((trace_id, source))
            if len(routed) == 1:
                raise TimeoutError("native group initial classifier timeout")
            return json.dumps({"intent": "query", "topic_action": "none", "topic_id": None,
                "goal_relation": "none", "task_context_id": None, "public_summary": {}, "constraints": {},
                "source_message_id": source["message_id"], "confidence": 1, "reasoning": "固定测试分类"})

    store = SQLiteAgentStore(tmp_path / "business.sqlite3")
    runtime = AgentRuntime(llm_client=TopicClient(), store=store, trace_recorder=InMemoryTraceRecorder())
    monkeypatch.setattr(app, "get_runtime", lambda: runtime)
    monkeypatch.setattr(AgentRuntime, "handle_user_message", lambda self, message, *, trace_id:
        AgentRuntimeResult(trace_id=trace_id, conversation_id=message.conversation_id, final_reply="", status="completed"))
    monkeypatch.setattr(app, "reviewed_agent_wechat_group_reply", lambda *_args, **_kwargs: "你今晚几点方便？")
    raw = {"is_room": True, "room": {"id": room_id}, "conversation_id": f"wechaty:room:{room_id}",
           "message_id": "native-group-original", "source_message_id": "native-group-original", "sender_id": "wxid_a",
           "message_type": 7, "text": "老板，晚上还有桌吗？", "payload": {"threadId": room_id,
               "account_id": "shop-account", "direction": "incoming", "native_type": 1,
               "createdAt": "2026-11-26T11:39:30+08:00"}}
    source_id = raw["message_id"]
    calls = []

    def deliver(path, *, payload, timeout_seconds):
        """核对适配器只收到原群、原消息和已审查的正文。"""
        calls.append(path)
        assert path == "/deliver-ingress-reply" and payload["payload"] == raw
        assert payload["result"]["route_result"] == {"scoped_group_reply": {
            "room_id": room_id, "source_message_id": source_id, "text": "你今晚几点方便？"}}
        return {"ok": True, "delivery_state": "accepted", "message_id": "native-group-reply"}

    monkeypatch.setattr(app, "request_local_json", deliver)
    app.queue_agent_wechat_raw(runtime, raw, deferred_reply=True)
    app.process_queued_channel_raw(runtime, store.get_channel_observation(channel, source_id))
    batch, = store.pending_input_gate_handoffs()
    app.handle_due_pending_input_batch(batch, "due-callback")
    ready = store.get_channel_observation(channel, source_id)
    assert ready["route_mode"] == "ingress_reply_pending" and calls == []
    assert "message_id" not in ready["payload"]["deferred_reply"]
    assert ready["payload"]["route_result"]["group_result"]["reply"] == ""
    app.process_queued_channel_raw(runtime, ready)
    completed = store.get_channel_observation(channel, source_id)
    assert completed["payload"]["deferred_reply"] == {"delivery_state": "accepted", "attempts": 1,
        "message_id": "native-group-reply", "text": "你今晚几点方便？"}
    assert calls == ["/deliver-ingress-reply"] and len(routed) == 2 and routed[0] == routed[1]


def test_managed_room_has_priority_over_observe_only(monkeypatch, tmp_path) -> None:
    managed_path = _managed_config(tmp_path)
    observe_path = tmp_path / "observe.json"
    observe_path.write_text(
        json.dumps({"room_ids": ["@@managed"], "topic_keywords": []}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "self_only")
    monkeypatch.setenv("MAHJONG_WECHATY_MANAGED_ROOMS_PATH", str(managed_path))
    monkeypatch.setenv("MAHJONG_WECHATY_OBSERVE_ONLY_ROOMS_PATH", str(observe_path))
    runtime = AgentRuntime(llm_client=StaticAgentClient(outputs=[]), store=InMemoryAgentStore())
    handler = FakeManagedHandler()
    monkeypatch.setattr(app, "get_runtime", lambda: runtime)
    monkeypatch.setattr(app, "get_group_message_handler", lambda _runtime: handler)

    result = app.route_wechaty_raw_to_agent(
        {
            "conversation_id": "wechaty:room:@@managed",
            "sender_id": "@member",
            "message_id": "msg-overlap",
            "text": "14:00 0.5 无烟 371",
            "self_message": False,
            "is_room": True,
            "room": {"id": "@@managed", "topic": "托管棋牌群"},
            "talker": {"name": "公开昵称", "payload": {"friend": False}},
            "payload": {"roomId": "@@managed", "talkerId": "@member"},
        },
        trace_id="trace-overlap",
    )

    assert result["managed_group"] is True
    assert "observe_only" not in result


def test_due_group_board_task_calls_board_engine(monkeypatch) -> None:
    runtime = AgentRuntime(llm_client=StaticAgentClient(outputs=[]), store=InMemoryAgentStore())
    runtime.store.upsert_group_room_policy(GroupRoomPolicy(room_id="@@managed", managed=True, board_enabled=True, outbound_enabled=True))
    handler = FakeHandlerWithBoard()
    monkeypatch.setattr(app, "get_runtime", lambda: runtime)
    monkeypatch.setattr(app, "get_group_message_handler", lambda _runtime, **kwargs: handler)
    task = ScheduledAgentTask(
        task_id="group_board_publish:@@managed",
        task_type=BOARD_TASK_TYPE,
        aggregate_type="group_room",
        aggregate_id="@@managed",
        conversation_id="wechaty:room:@@managed",
        subject_id="system",
        subject_name="system",
        due_at=datetime(2026, 7, 22, 12, 0, tzinfo=TZ),
        idempotency_key="group_board_publish:@@managed",
        payload={"publication_id": "publication-test"},
    )

    app.handle_due_scheduled_agent_task(task, "trace-board-task")

    assert handler.board_engine.calls == [("@@managed", "trace-board-task")]
    events = runtime.trace_recorder.get_trace("trace-board-task")
    assert any(item.step == "group_board_publish_completed" for item in events)


def test_wechaty_group_messenger_sends_room_target_with_reference_metadata(monkeypatch) -> None:
    runtime = AgentRuntime(llm_client=StaticAgentClient(outputs=[]), store=InMemoryAgentStore())
    runtime.store.upsert_group_room_policy(
        GroupRoomPolicy(room_id="@@managed", managed=True, outbound_enabled=True)
    )
    calls: list[tuple[str, dict | None]] = []

    def fake_request(path: str, *, payload=None, timeout_seconds=3.0):
        calls.append((path, payload))
        if path == "/health":
            return {"send_channel_enabled": True, "auto_send_reply": True}
        return {"ok": True, "message_id": "platform-board-message"}

    monkeypatch.setattr(app, "request_local_json", fake_request)
    # 此例只校验传输字段，内容审查的真实入口与拒绝边界另有专项覆盖。
    monkeypatch.setattr(app.WechatyGroupMessenger, "_review_group_board", lambda *args, **kwargs: None)
    messenger = app.WechatyGroupMessenger(runtime)

    message_id = messenger.send_group_message(
        "@@managed",
        "当前缺人局：\n1、14:00 0.5 无烟 371",
        metadata={
            "trace_id": "trace-room-send",
            "source_message_id": "group_board:snapshot-1",
            "business_ref_type": "group_board_snapshot",
            "business_ref_id": "snapshot-1",
        },
    )

    assert message_id == "platform-board-message"
    assert [path for path, _ in calls] == ["/health", "/send"]
    assert calls[-1][1]["target_type"] == "room"
    assert calls[-1][1]["to"] == "@@managed"
    assert calls[-1][1]["source_message_id"] == "group_board:snapshot-1"


def test_managed_room_outbound_switch_also_blocks_private_handoff(monkeypatch) -> None:
    runtime = AgentRuntime(llm_client=StaticAgentClient(outputs=[]), store=InMemoryAgentStore())
    runtime.store.upsert_group_room_policy(
        GroupRoomPolicy(room_id="@@managed", managed=True, outbound_enabled=False)
    )
    calls = []
    monkeypatch.setattr(app, "request_local_json", lambda *args, **kwargs: calls.append((args, kwargs)))
    messenger = app.WechatyGroupMessenger(runtime)

    message_id = messenger.send_private_message(
        "@friend",
        "几点能到？",
        metadata={"trace_id": "trace-private-blocked", "origin_room_id": "@@managed"},
    )

    assert message_id.startswith("suppressed:")
    assert calls == []
