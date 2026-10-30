"""独立语义判定只接收同一客户当前目标的复查末态，历史工具成功不能替代现况。"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import CustomerProfile, InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway, UserMessage
from mahjong_agent_runtime.models import now
from mahjong_agent_runtime.task_context import TaskContextManager


@pytest.fixture
def owner():
    """加载本副本的实际证据投影，不运行真实模型或改写黄金资产。"""
    path = Path(__file__).resolve().parents[1] / "scripts/run_real_owner_chat_live_eval.py"
    spec = importlib.util.spec_from_file_location("follow_up_semantic_evidence_owner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    """所有合同同时验证内存事实源及真实 SQLite 持久化。"""
    current = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "semantic-follow-up.sqlite3")
    yield current
    if isinstance(current, SQLiteAgentStore):
        current._connection.close()


def current_message(store):
    """真实创建客户、目标和原话，供 Gateway 定位安排责任的归属。"""
    message = UserMessage(conversation_id="current-chat", sender_id="current-customer", sender_name="客户",
                          text="按现在条件帮我看看，有进展再说", message_id="current-input")
    store.upsert_customer(CustomerProfile(customer_id=message.sender_id, display_name=message.sender_name))
    TaskContextManager(store).prepare(message, trace_id="prepare")
    store.append_user_turn(message, "input")
    return message


def execute(store, message, name, arguments, step):
    """所有初次安排和取消都通过真实 Gateway，并核对没有被拒绝或伪执行。"""
    result = ToolGateway(store).execute(ToolCall(name=name, arguments=arguments), trace_id=f"tool-{step}",
        conversation_id=message.conversation_id, sender_id=message.sender_id, sender_name=message.sender_name,
        step_index=step, source_message_id=message.message_id)
    assert result.called and result.allowed and not result.error
    return result


def schedule(store, message):
    """使用有界期限和次数，不把后续复查偷换成已经联系客户。"""
    stamp = now()
    return execute(store, message, "schedule_follow_up", {
        "goal": "复查本次原条件的候选情况", "waiting_for": "players", "review_budget": 1,
        "due_at": (stamp + timedelta(minutes=5)).isoformat(),
        "expires_at": (stamp + timedelta(minutes=30)).isoformat(),
    }, 1)


@contextmanager
def reopened_store(store):
    """SQLite 投影均从新连接重读，避免把执行对象缓存误认为已持久化状态。"""
    if isinstance(store, SQLiteAgentStore):
        reopened = SQLiteAgentStore(store.path)
        try:
            yield reopened
        finally:
            reopened._connection.close()
    else:
        yield store


def project(owner, store, message, tools):
    """沿真实 reply_semantic_evidence 路径，核对只读投影不回写任务状态。"""
    before = {key: task.to_dict() for key, task in store.scheduled_tasks.items()}
    with reopened_store(store) as reader:
        result = SimpleNamespace(tool_results=tools, final_reply="有进展跟你说。", status="completed")
        evidence = owner.reply_semantic_evidence(result, SimpleNamespace(message=message), [], reader)
    assert {key: task.to_dict() for key, task in store.scheduled_tasks.items()} == before
    return evidence


def test_real_scheduled_follow_up_retains_scope_budget_and_observed_time(owner, store):
    message = current_message(store)
    scheduled = schedule(store, message)
    history = deepcopy(scheduled.to_dict())
    before = now()
    evidence = project(owner, store, message, [scheduled])
    after = now()
    state = evidence["persisted_state"]["follow_up_state"]
    context_id = store.current_task_context(message.conversation_id, message.sender_id).task_context_id
    assert before <= datetime.fromisoformat(state["observed_at"]) <= after
    assert state["conversation_id"] == message.conversation_id
    assert state["customer_id"] == message.sender_id
    assert state["current_task_context_id"] == context_id
    raw = scheduled.result["follow_up"]
    assert state["tasks"] == [{
        "task_id": raw["task_id"], "task_type": "conversation_follow_up", "status": "pending",
        "conversation_id": message.conversation_id, "subject_id": message.sender_id,
        "aggregate_type": "conversation_goal", "aggregate_id": context_id,
        "task_context_id": context_id, "goal": raw["payload"]["goal"], "waiting_for": "players",
        "due_at": raw["due_at"], "expires_at": raw["payload"]["expires_at"], "review_budget": 1, "review_count": 0,
    }]
    assert scheduled.to_dict() == history
    assert not {"approved", "valid", "expected_approved", "semantic_passed"} & state["tasks"][0].keys()
    assert not store.outbound_message_drafts


@pytest.mark.parametrize("change", ["cancelled", "expired", "processing", "completed"])
def test_current_persisted_state_wins_over_successful_schedule_snapshot(owner, store, monkeypatch, change):
    """取消、到期和耗用次数保留真实末态，不能被历史 pending 工具快照洗白。"""
    message = current_message(store)
    scheduled = schedule(store, message)
    history = deepcopy(scheduled.to_dict())
    raw = scheduled.result["follow_up"]
    evidence_tools = [scheduled]
    observed = now()
    if change == "cancelled":
        evidence_tools.append(execute(store, message, "cancel_follow_up", {"reason": "客户不需要再跟进"}, 2))
    elif change == "expired":
        observed = datetime.fromisoformat(raw["payload"]["expires_at"]) + timedelta(seconds=1)
    else:
        observed = datetime.fromisoformat(raw["due_at"])
        claimed = store.claim_scheduled_task(raw["task_id"], at=observed, lease_seconds=30)
        assert claimed is not None
        if change == "completed":
            store.complete_scheduled_task(raw["task_id"], trace_id="completed", at=observed)
    monkeypatch.setattr(owner, "now", lambda: observed)
    evidence = project(owner, store, message, evidence_tools)
    state = evidence["persisted_state"]["follow_up_state"]
    task = next(task for task in state["tasks"] if task["task_id"] == raw["task_id"])
    assert scheduled.to_dict() == history and scheduled.result["follow_up"]["status"] == "pending"
    assert task["status"] == ("pending" if change == "expired" else change)
    assert state["observed_at"] == observed.isoformat()
    assert task["review_budget"] == 1
    assert task["review_count"] == (1 if change in {"processing", "completed"} else 0)
    if change == "expired":
        assert datetime.fromisoformat(task["expires_at"]) < datetime.fromisoformat(state["observed_at"])
    if change == "cancelled":
        assert all(task["status"] == "cancelled" for task in state["tasks"])
        assert [item["name"] for item in evidence["tool_results"]] == ["schedule_follow_up", "cancel_follow_up"]


@pytest.mark.parametrize("scope_change", [
    "old_goal", "other_customer", "other_conversation", "other_aggregate_type",
    "other_aggregate_id", "other_payload_context", "other_task_type", "missing_context",
])
def test_other_scope_never_becomes_current_responsibility_evidence(owner, store, scope_change):
    """逐项核对客户、会话、当前目标和聚合身份，不因工具成功或名称相似借用安排。"""
    message = current_message(store)
    scheduled = schedule(store, message)
    original = deepcopy(scheduled.to_dict())
    if scope_change == "old_goal":
        store.activate_task_context(conversation_id=message.conversation_id, customer_id=message.sender_id,
            trace_id="new-goal", activity_at=now(), started_at=now(), reason="new_goal",
            force_new=True, archive_previous=False)
    elif scope_change in {"other_customer", "other_conversation", "missing_context"}:
        message = replace(message,
            sender_id="other-customer" if scope_change == "other_customer" else message.sender_id,
            conversation_id="other-conversation" if scope_change in {"other_conversation", "missing_context"} else message.conversation_id)
        if scope_change != "missing_context":
            TaskContextManager(store).prepare(message, trace_id="other-scope")
    else:
        task = store.scheduled_tasks[scheduled.result["follow_up"]["task_id"]]
        changes = {
            "other_aggregate_type": {"aggregate_type": "game"},
            "other_aggregate_id": {"aggregate_id": "other-target"},
            "other_payload_context": {"payload": {**task.payload, "task_context_id": "other-target"}},
            "other_task_type": {"task_type": "game_recruitment"},
        }[scope_change]
        store.save_follow_up_task(replace(task, **changes), trace_id="different-scope-record")
    evidence = project(owner, store, message, [scheduled])
    state = evidence["persisted_state"]["follow_up_state"]
    assert state["tasks"] == []
    assert (state["current_task_context_id"] is None) == (scope_change == "missing_context")
    assert scheduled.to_dict() == original
    assert evidence["tool_results"][0]["name"] == "schedule_follow_up"
    assert evidence["tool_results"][0]["called"] is True
