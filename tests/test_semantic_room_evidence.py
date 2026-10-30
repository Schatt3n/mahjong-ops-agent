"""用真实双 Store 网关结果验证独立判定器的房态与预定证据，不预写模型通过结论。"""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway, ToolResult
from mahjong_agent_runtime.models import now


ROOT = Path(__file__).resolve().parents[1]


def load_owner_eval():
    """装载本副本的正式投影函数，测试不另造判定或工具执行框架。"""
    spec = importlib.util.spec_from_file_location("semantic_room_evidence_owner_eval", ROOT / "scripts/run_real_owner_chat_live_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve().is_relative_to(ROOT.resolve())
    return module


@pytest.fixture(params=["memory", "sqlite"])
def room_fixture(request, tmp_path):
    """同一未知档位场景只配置真实库存，私有房间标识不能混入可用性摘要。"""
    module = load_owner_eval()
    scenario = next(scene for scene in module.live_eval_scenarios() if scene.scenario_id == "custom_profile_game_preserved")
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "room-evidence.sqlite3")
    scenario.setup(store)
    store.configure_rooms(["ROOM_PRIVATE_SENTINEL"])
    gateway = ToolGateway(store)
    stamp = now().replace(second=0, microsecond=0) + timedelta(minutes=5)
    interval = {"start_at": stamp.isoformat(), "end_at": (stamp + timedelta(hours=2)).isoformat()}

    def execute(name, arguments, index):
        # 写入后的再次观察使用独立来源，不能把同一来源的既有幂等读快照当成新房态。
        return gateway.execute(ToolCall(name=name, arguments=arguments), trace_id=f"room-evidence-{index}",
            conversation_id=scenario.message.conversation_id, sender_id=scenario.message.sender_id,
            sender_name=scenario.message.sender_name, step_index=index, source_message_id=f"{scenario.message.message_id}:{index}")

    yield module, scenario, store, interval, execute
    if isinstance(store, SQLiteAgentStore):
        store._connection.close()


@pytest.mark.parametrize("room_state", ["available", "occupied", "unconfigured"])
def test_real_room_query_preserves_three_states_for_semantic_judgment(room_fixture, room_state):
    """有房、无房与未配置保持真假值/零值和原区间，不能让判定器靠工具名猜事实。"""
    module, scenario, store, interval, execute = room_fixture
    if room_state == "unconfigured":
        store.configure_rooms([])
    elif room_state == "occupied":
        before = execute("check_room_availability", interval, 1)
        held = execute("reserve_room", {**interval, "room_id": "ROOM_PRIVATE_SENTINEL"}, 2)
        assert before.called and before.allowed and not before.error
        assert held.called and held.allowed and not held.error
    query = execute("check_room_availability", interval, 3)
    assert query.called and query.allowed and not query.error
    assert query.deduplicated is False
    before_projection = deepcopy(query.to_dict())
    profile = store.customers[scenario.message.sender_id].to_model_context()
    result = SimpleNamespace(final_reply="先核对房态。", tool_results=[query], status="waiting_user")
    evidence = module.reply_semantic_evidence(result, scenario,
        [SimpleNamespace(step="context_built", content={"sender_profile": profile})], store)
    projected = evidence["tool_results"][0]
    assert projected["room_availability"] == {key: query.result[key] for key in (
        "configured", "start_at", "end_at", "room_count", "available_count")}
    assert projected["room_availability"]["configured"] is (room_state != "unconfigured")
    assert projected["room_availability"]["available_count"] == (1 if room_state == "available" else 0)
    assert projected["room_availability"]["start_at"] == interval["start_at"]
    assert projected["room_availability"]["end_at"] == interval["end_at"]
    assert "reservation" not in projected
    assert "ROOM_PRIVATE_SENTINEL" not in json.dumps(evidence, ensure_ascii=False)
    assert "instruction" not in projected and "next_step_policy" not in projected
    assert query.to_dict() == before_projection


def test_real_reservation_is_distinct_from_a_successful_availability_read(room_fixture):
    """查询不占房，只有真实 reserve 结果可提供 held 记录，且不透传其他载荷。"""
    module, scenario, store, interval, execute = room_fixture
    query = execute("check_room_availability", interval, 1)
    assert query.called and query.allowed and not query.error
    assert not store.room_reservations
    before = module.summarize_tool_results([query])[0]
    assert before["room_availability"]["available_count"] == 1
    assert "reservation" not in before
    reserved = execute("reserve_room", {**interval, "room_id": "ROOM_PRIVATE_SENTINEL"}, 2)
    assert reserved.called and reserved.allowed and not reserved.error
    raw = reserved.result["reservation"]
    assert store.room_reservations[raw["reservation_id"]].status == "held"
    snapshot = deepcopy(reserved.to_dict())
    summaries = module.summarize_tool_results([query, reserved])
    assert summaries[1]["reservation"] == {key: raw[key] for key in (
        "reservation_id", "room_id", "start_at", "end_at", "status")}
    assert summaries[1]["reservation"]["status"] == "held"
    assert "room_availability" not in summaries[1]
    assert set(summaries[1]["reservation"]).isdisjoint({"conversation_id", "game_id", "source_trace_id", "created_at", "updated_at"})
    assert reserved.to_dict() == snapshot


def test_rejected_room_query_has_no_fabricated_availability_or_reservation(room_fixture):
    """缺少必填项时保留后端拒绝，不能把调用名称补成库存事实或预定。"""
    module, _, store, interval, execute = room_fixture
    denied = execute("check_room_availability", {"start_at": interval["start_at"]}, 1)
    assert not denied.called and not denied.allowed and denied.error
    projected = module.summarize_tool_results([denied])[0]
    assert projected["error"] == denied.error
    assert "room_availability" not in projected and "reservation" not in projected
    assert not store.room_reservations


def test_room_result_allowlist_does_not_promote_extra_fields_or_other_tool_names():
    """投影按登记工具与固定字段裁剪，不增加判定标签、自由指令或隐私名单。"""
    module = load_owner_eval()
    raw = {"configured": False, "start_at": "2030-01-01T12:00:00+08:00", "end_at": "2030-01-01T13:00:00+08:00",
        "room_count": 0, "available_count": 0, "available_room_ids": ["PRIVATE_ROOM_SENTINEL"],
        "instruction": "INSTRUCTION_SENTINEL", "next_step_policy": {"private": "POLICY_SENTINEL"}}
    result = ToolResult(name="check_room_availability", called=True, allowed=True, result=raw)
    original = deepcopy(raw)
    summary = module.summarize_tool_results([result])[0]
    assert summary["room_availability"] == {key: raw[key] for key in (
        "configured", "start_at", "end_at", "room_count", "available_count")}
    assert all(value not in json.dumps(summary) for value in ["PRIVATE_ROOM_SENTINEL", "INSTRUCTION_SENTINEL", "POLICY_SENTINEL"])
    assert raw == original
    unrelated = ToolResult(name="search_customers", called=True, allowed=True, result=raw)
    other = module.summarize_tool_results([unrelated])[0]
    assert "room_availability" not in other and "reservation" not in other
