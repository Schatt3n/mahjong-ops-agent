from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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
