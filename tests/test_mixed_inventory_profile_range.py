"""用真实双 Store 查询验证新增个人范围夹具区分力，不把离线机制测试当模型语义成绩。"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ID = "profile_range_mixed_inventory_personal_query"
RECORD_ID = "owner_chat_mixed_inventory_profile_range_regression_20261102_001"


def load_owner():
    """只装载本副本的正式运行器，任何测试均不构造真实模型客户端。"""
    spec = importlib.util.spec_from_file_location("mixed_inventory_owner_eval", ROOT / "scripts/run_real_owner_chat_live_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["memory", "sqlite"])
def mixed_case(request, tmp_path):
    """三种库存全部由正式 Store 创建，容量相同且请求客户尚未加入。"""
    module = load_owner()
    scene = next(item for item in module.live_eval_scenarios() if item.scenario_id == SCENARIO_ID)
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "mixed.sqlite3")
    scene.setup(store)
    yield module, scene, store
    if isinstance(store, SQLiteAgentStore):
        store._connection.close()


def personal_requirement():
    """字段来自本例明示个人目标及可靠画像；未选唯一固定金额。"""
    return {"game_type": "red_center_mahjong", "stake_range": {"min": 1, "max": 3}, "requesting_party": {"seat_count": 1}}


def execute_query(scene, store, requirement, *, message_id="mixed-offline-query"):
    """使用真实网关执行查询，错误工具参数不能靠手填结果满足硬检查。"""
    return ToolGateway(store).execute(ToolCall("search_current_games", {"requirement": requirement, "limit": 10}),
        trace_id=message_id, conversation_id=scene.message.conversation_id, sender_id=scene.message.sender_id,
        sender_name=scene.message.sender_name, step_index=1, source_message_id=message_id)


def snapshot(store):
    """比较查询前后的领域事实，幂等审计记录不冒充占座或新招募副作用。"""
    return deepcopy({
        "games": [item.to_dict() for item in store.games.values()],
        "reservations": [item.to_dict() for item in store.room_reservations.values()],
        "invites": [item.to_dict() for item in store.invite_drafts.values()],
        "outbound": [item.to_dict() for item in store.outbound_message_drafts.values()],
        "scheduled": [item.to_dict() for item in store.scheduled_tasks.values()],
        "transitions": [item.to_dict() for item in store.transitions],
    })


def test_mixed_fixture_distinguishes_real_personal_filter_from_unfiltered_inventory(mixed_case):
    """先证实全店有三种库存，再验证个人参数只留下正确玩法和区间内那一桌。"""
    _, scene, store = mixed_case
    before = snapshot(store)
    broad = execute_query(scene, store, {}, message_id="mixed-broad-probe")
    assert broad.called and broad.allowed and not broad.error
    assert {(item["game"]["requirement"]["game_type"], item["game"]["requirement"]["stake"]) for item in broad.result["matches"]} == {
        ("red_center_mahjong", "2"), ("red_center_mahjong", "5"), ("national_standard_mahjong", "2")}
    result = execute_query(scene, store, personal_requirement())
    assert result.called and result.allowed and not result.error
    assert result.result["alternatives"] == []
    assert len(result.result["matches"]) == 1
    item = result.result["matches"][0]
    assert item["game"]["requirement"]["game_type"] == "red_center_mahjong"
    assert item["game"]["requirement"]["stake"] == "2"
    assert item["game"]["seat_summary"]["claimed_seats"] == 3
    assert item["game"]["seat_summary"]["remaining_seats"] == 1
    assert item["join_projection"]["sender_already_joined"] is False
    assert item["join_projection"]["requested_seats"] == 1
    assert result.result["customer_reply_contract"]["evidence_scope"]["unfiltered_available_pool_empty"] is False
    assert snapshot(store) == before


@pytest.mark.parametrize("status,reply", [("completed", "有个红中2块三缺一。"), ("waiting_user", "红中2块三缺一，你打吗？")])
def test_new_runner_hard_checks_allow_informing_or_asking_without_fixed_sentence(mixed_case, status, reply):
    """这里只验证硬检查接受两种状态；布尔语义桩不是独立模型判定或真实模型通过证据。"""
    module, scene, store = mixed_case
    result = execute_query(scene, store, personal_requirement())
    outcome = SimpleNamespace(final_reply=reply, status=status, tool_results=[result])
    validation = module.validate_result(outcome, scene, [], store, {"passed": True, "error": None})
    assert validation["passed"], validation["checks"]
    assert scene.required_reply_any == [] and scene.required_reply_contains == []
    assert scene.expected_status_any == ["completed", "waiting_user"]
    assert scene.max_reply_chars is None
    expectation = module.reply_semantic_expectation(scene)
    assert any(item["id"].endswith("reply_and_state_align") for item in expectation["required_meaning"])


@pytest.mark.parametrize("fault", ["unfiltered", "missing_game", "missing_range", "missing_party", "fixed_lower", "fixed_middle", "wrong_game", "two_people"])
def test_new_hard_contract_rejects_omitted_or_replaced_personal_filters(mixed_case, fault):
    """即使句子看似正确，省参数、选单点或错玩法也必须被真实结果硬断言识别。"""
    module, scene, store = mixed_case
    requirement = personal_requirement()
    if fault == "unfiltered":
        requirement = {}
    elif fault == "missing_game":
        requirement.pop("game_type")
    elif fault == "missing_range":
        requirement.pop("stake_range")
    elif fault == "missing_party":
        requirement.pop("requesting_party")
    elif fault in {"fixed_lower", "fixed_middle"}:
        requirement.pop("stake_range")
        requirement["stake"] = "1" if fault == "fixed_lower" else "2"
    elif fault == "wrong_game":
        requirement["game_type"] = "national_standard_mahjong"
    else:
        requirement["requesting_party"]["seat_count"] = 2
    before = snapshot(store)
    result = execute_query(scene, store, requirement)
    assert result.called and result.allowed and not result.error
    outcome = SimpleNamespace(final_reply="有个红中2块三缺一。", status="completed", tool_results=[result])
    validation = module.validate_result(outcome, scene, [], store, {"passed": True, "error": None})
    assert not validation["passed"]
    assert any(not check["passed"] and "result.requirement" in check["name"] for check in validation["checks"])
    assert snapshot(store) == before


def test_new_record_is_separate_controlled_source_and_old_single_game_fixture_remains():
    """原单局案例和原合成记录继续保留；新source不能冒充真人或已完成live。"""
    module = load_owner()
    records = [json.loads(line) for line in (ROOT / "eval/golden/real_owner_chat_golden.jsonl").read_text().splitlines() if line.strip()]
    record = next(item for item in records if item["id"] == RECORD_ID)
    assert record["source"]["kind"] == "synthetic_regression_fixture"
    assert record["source"]["not_original_human_transcript"] is True
    assert record["source"]["review_status"] == "controlled_fixture_pending_live_model_run"
    assert len(record["eval_cases"]) == 1 and record["eval_cases"][0]["live_scenario_id"] == SCENARIO_ID
    original = next(item for item in records if item["id"] == "owner_chat_live_semantic_regression_20261029_001")
    assert len(original["eval_cases"]) == 19
    store = InMemoryAgentStore()
    module.setup_range_profile_with_game(store)
    assert len(store.games) == 1
    game = next(iter(store.games.values()))
    assert game.requirement["game_type"] == "red_center_mahjong" and game.requirement["stake"] == "2"
