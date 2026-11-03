"""验收当前任务约束的真实可读回与隔离行为，不用scope标签代替执行证据。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
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
SCOPES = ("current_task", "session", "today", "long_term")


def load_module(name, relative):
    """只加载本隔离副本的实际运行器/校验器，不调用模型或改原始评测记录。"""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["memory", "sqlite"])
def case(request, tmp_path):
    """保留原四小时案例的Game与客户画像，任务上下文由各例显式建立。"""
    module = load_module("task_memory_equivalence_owner", "scripts/run_real_owner_chat_live_eval.py")
    scene = next(s for s in module.live_eval_scenarios() if s.scenario_id == "duration_limit_update_should_persist")
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "memory-scope.sqlite3")
    scene.setup(store)
    yield module, scene, store
    if isinstance(store, SQLiteAgentStore):
        store._connection.close()


def activate(store, scene, *, force_new=False, archive_previous=False, customer_id=None, conversation_id=None, started_at=None):
    """复用正式任务事务接口；不同目标和客户不是测试里伪造的metadata标签。"""
    stamp = started_at or now()
    return store.activate_task_context(conversation_id=conversation_id or scene.message.conversation_id,
        customer_id=customer_id or scene.message.sender_id, trace_id="scope-task", activity_at=stamp,
        started_at=stamp, reason="scope-equivalence-test", force_new=force_new, archive_previous=archive_previous)[0]


def record(store, scene, *, scope="today", field="max_duration_hours", value=4, metadata=None):
    """真实网关写任务记忆，不能通过直接注入全局画像绕过待审核边界。"""
    return ToolGateway(store).execute(ToolCall("record_user_memory", {"task_memories": [{
        "customer_id": scene.message.sender_id, "memory_type": "constraint", "field": field, "value": value,
        "evidence": scene.message.text, "confidence": 0.95, "risk_level": "low", "scope": scope,
        "metadata": {"start_time": "19:00", **(metadata or {})},
    }]}), trace_id="scope-memory", conversation_id=scene.message.conversation_id,
        sender_id=scene.message.sender_id, sender_name=scene.message.sender_name, step_index=1,
        source_message_id="scope-memory-message")


def validation(module, scene, store, result):
    """语义布尔仅用于单测硬验收，不是新模型输出或真实独立语义通过证据。"""
    return module.validate_result(SimpleNamespace(final_reply="好，记着了。", status="completed", tool_results=[result]),
        scene, [], store, {"passed": True, "error": None})


def bound_check(result):
    """新硬门槛必须单独存在，避免仅删scope断言而漏掉实际任务隔离。"""
    return next(item for item in result["checks"] if item["name"] == "task_memory_current_context_readback_1")


@pytest.mark.parametrize("scope", SCOPES)
def test_legal_scope_requires_real_current_task_readback_without_global_profile_changes(case, scope):
    module, scene, store = case
    task = activate(store, scene)
    profiles = deepcopy({key: asdict(value) for key, value in store.customers.items()})
    games = deepcopy([item.to_dict() for item in store.games.values()])
    result = record(store, scene, scope=scope)
    assert result.called and result.allowed and not result.error
    memory = result.result["task_memories"][0]
    assert memory["scope"] == scope  # 不coerce合法标签，也不新增跨任务或日历能力。
    assert memory["metadata"]["task_context_id"] == task.task_context_id
    assert memory in store.task_memory_context(scene.message.conversation_id, scene.message.sender_id)
    assert memory["field"] == "max_duration_hours" and memory["value"] == 4
    assessed = validation(module, scene, store, result)
    assert assessed["passed"], assessed["checks"]
    assert bound_check(assessed)["passed"]
    assert {key: asdict(value) for key, value in store.customers.items()} == profiles
    assert [item.to_dict() for item in store.games.values()] == games
    assert not store.pending_memory_candidates


@pytest.mark.parametrize("scope", SCOPES)
def test_scope_label_never_grants_other_customer_conversation_or_next_goal_visibility(case, scope):
    _, scene, store = case
    first = activate(store, scene)
    assert record(store, scene, scope=scope).allowed
    assert len(store.task_memory_context(scene.message.conversation_id, scene.message.sender_id)) == 1
    activate(store, scene, customer_id="other-customer")
    assert store.task_memory_context(scene.message.conversation_id, "other-customer") == []
    assert store.task_memory_context("another-conversation", scene.message.sender_id) == []
    latest = activate(store, scene, force_new=True, archive_previous=True)
    assert latest.task_context_id != first.task_context_id
    assert store.task_memory_context(scene.message.conversation_id, scene.message.sender_id) == []
    assert all(item.status == "archived" for item in store.task_memories.values())


@pytest.mark.parametrize("fault", ["wrong_field", "wrong_value", "wrong_task", "invalid_scope"])
def test_missing_or_wrong_persisted_fact_is_not_exempted_by_scope_equivalence(case, fault):
    module, scene, store = case
    activate(store, scene)
    options = {"wrong_field": {"field": "duration_hours"}, "wrong_value": {"value": 5},
               "wrong_task": {"metadata": {"task_context_id": "not-current-task"}},
               "invalid_scope": {"scope": "arbitrary-global-label"}}[fault]
    result = record(store, scene, **options)
    assessed = validation(module, scene, store, result)
    assert not assessed["passed"] and not bound_check(assessed)["passed"]
    if fault == "invalid_scope":
        assert not result.called and not result.allowed and not store.task_memories


def test_readable_legacy_unbound_fact_cannot_substitute_for_exact_current_task_binding(case):
    module, scene, store = case
    result = record(store, scene)  # 无当前任务时合法保存，但没有精确任务绑定。
    memory = result.result["task_memories"][0]
    assert "task_context_id" not in memory["metadata"]
    activate(store, scene, started_at=now() - timedelta(hours=1))
    assert memory in store.task_memory_context(scene.message.conversation_id, scene.message.sender_id)
    assessed = validation(module, scene, store, result)
    assert not assessed["passed"] and not bound_check(assessed)["passed"]


def test_successful_tool_claim_without_authoritative_readback_never_passes(case):
    module, scene, store = case
    task = activate(store, scene)
    fake = ToolResult("record_user_memory", called=True, allowed=True, result={"task_memories": [{
        "field": "max_duration_hours", "value": 4, "scope": "today", "status": "active",
        "customer_id": scene.message.sender_id, "conversation_id": scene.message.conversation_id,
        "metadata": {"task_context_id": task.task_context_id},
    }]})
    assessed = validation(module, scene, store, fake)
    assert not assessed["passed"] and not bound_check(assessed)["passed"]
    assert not store.task_memories


def test_unbound_duplicate_cannot_hide_an_existing_fully_bound_current_fact(case):
    """完整记录独立满足全部条件即可；不依赖同值记忆的排序，也不从两条记录拼证。"""
    module, scene, store = case
    activate(store, scene)
    valid = record(store, scene)
    duplicate = record(store, scene, metadata={"task_context_id": ""})
    assert valid.called and duplicate.called
    assert len(store.task_memory_context(scene.message.conversation_id, scene.message.sender_id)) == 2
    assert bound_check(validation(module, scene, store, duplicate))["passed"]


def test_value_from_unbound_row_cannot_borrow_binding_from_a_different_value(case):
    """四小时与真实任务绑定必须属于同一条持久记录，不能把两个不合格事实拼起来。"""
    module, scene, store = case
    activate(store, scene)
    assert record(store, scene, value=5).called
    unbound = record(store, scene, metadata={"task_context_id": ""})
    assert len(store.task_memory_context(scene.message.conversation_id, scene.message.sender_id)) == 2
    assessed = validation(module, scene, store, unbound)
    assert not assessed["passed"] and not bound_check(assessed)["passed"]


@pytest.mark.parametrize("scope", SCOPES)
def test_sqlite_reopen_retains_exact_binding_then_new_goal_hides_it(tmp_path, scope):
    module = load_module("task_memory_reopen_owner", "scripts/run_real_owner_chat_live_eval.py")
    scene = next(s for s in module.live_eval_scenarios() if s.scenario_id == "duration_limit_update_should_persist")
    path = tmp_path / "reopen.sqlite3"
    first = SQLiteAgentStore(path)
    scene.setup(first)
    task = activate(first, scene)
    result = record(first, scene, scope=scope)
    first._connection.close()
    reopened = SQLiteAgentStore(path)
    try:
        assert reopened.current_task_context(scene.message.conversation_id, scene.message.sender_id).task_context_id == task.task_context_id
        assert bound_check(validation(module, scene, reopened, result))["passed"]
        activate(reopened, scene, force_new=True, archive_previous=True)
        assert reopened.task_memory_context(scene.message.conversation_id, scene.message.sender_id) == []
        assert not bound_check(validation(module, scene, reopened, result))["passed"]
    finally:
        reopened._connection.close()


def test_current_context_guard_without_declared_fact_fails_closed(case):
    module, scene, store = case
    activate(store, scene)
    result = record(store, scene)
    empty = replace(scene, expected_task_memory_contains=[])
    assessed = validation(module, empty, store, result)
    assert not assessed["passed"]
    assert any(item["name"] == "task_memory_current_context_readback_required" and not item["passed"] for item in assessed["checks"])


def test_review_metadata_is_honest_and_preserves_original_source_and_exact_chain():
    validator = load_module("task_memory_equivalence_validator", "scripts/validate_real_owner_chat_golden.py")
    records = validator.read_jsonl(validator.DEFAULT_DATASET_PATH)
    record = next(r for r in records if r["id"] == "owner_chat_task_memory_scope_equivalence_review_20261103_001")
    assert record["source"]["kind"] == "code_contract_review"
    assert record["source"]["user_specified_scope"] is False
    assert record["source"]["review_scope"] == "behavior_equivalence"
    revision = record["hard_assertion_revisions"][0]
    assert revision["previous_snapshot"]["expected_task_memory_contains"][0]["scope"] == "current_task"
    assert "scope" not in revision["current_snapshot"]["expected_task_memory_contains"][0]
    assert revision["current_snapshot"]["require_task_memory_current_context"] is True
    assert validator.validate_dataset(records) == []
    forged = deepcopy(record)
    forged["source"]["user_specified_scope"] = True
    assert any("user_specified_scope=false" in error for error in validator._validate_hard_assertion_review_record(forged))
