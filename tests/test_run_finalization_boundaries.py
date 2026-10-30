"""核对所有运行入口、摘要快照和聚合片段共用同一版本边界。"""

from __future__ import annotations

import json
import threading
from datetime import timedelta

import pytest

from mahjong_agent_runtime import (
    AgentRuntime, AgentRuntimeResult, InMemoryAgentStore, SQLiteAgentStore,
    StaticAgentClient, SystemTriggerMessage, UserMessage, aggregate_pending_input_batch,
)
from mahjong_agent_runtime.models import now
from mahjong_agent_runtime.runtime import message_idempotency_key
from mahjong_agent_runtime.services.context_service import ContextLifecycleManager
from mahjong_agent_runtime.summary import ContextSummaryManager


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    current = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "boundaries.sqlite3")
    yield current
    if isinstance(current, SQLiteAgentStore):
        current._connection.close()


def user(identifier, text):
    return UserMessage(conversation_id="boundary", sender_id="customer", sender_name="客户", message_id=identifier, text=text)


def action(calls=None):
    calls = calls or []
    return json.dumps({
        "goal": "处理当前目标", "objective_status": "needs_tool" if calls else "completed",
        "reasoning_summary": "确定性执行边界测试", "objective_state": {"known_facts": {}, "missing_facts": [], "blockers": []},
        "objective_plan": [{"step_id": "s", "title": "处理", "status": "in_progress" if calls else "done"}],
        "plan_revision_reason": "按真实状态继续", "reply_to_user": "" if calls else "好。", "tool_calls": calls, "needs_human": False,
        "stop_reason": {"can_stop": not calls, "why": "继续" if calls else "结束", "pending_work": ["处理"] if calls else [], "depends_on_tool_results": False},
    }, ensure_ascii=False)


@pytest.mark.parametrize("entry", ["system_event", "system_trigger", "recovery"])
def test_new_input_during_after_turn_summary_suppresses_every_entry_cache(store, monkeypatch, entry):
    """各入口的摘要返回后必须重核版本，不能先把旧 completed 结果写进幂等缓存。"""
    reached, release = threading.Event(), threading.Event()
    client = StaticAgentClient([action()])
    runtime = AgentRuntime(llm_client=client, store=store)
    initial = user("initial", "原目标")
    runtime.receive_user_message(initial, trace_id="initial")

    def summarize(self, **kwargs):
        assert kwargs["expected_run_version"] == 1
        reached.set()
        assert release.wait(5)
        return None

    monkeypatch.setattr(ContextLifecycleManager, "maybe_summarize_after_turn", summarize)
    result, errors = {}, []
    if entry == "system_trigger":
        trigger = SystemTriggerMessage(trigger_id="trigger", trigger_type="test_trigger", conversation_id="boundary",
            sender_id="customer", sender_name="客户", payload={})
        source = trigger.to_user_message()
    else:
        source = user("internal" if entry == "system_event" else "recover", "内部继续")

    def work():
        try:
            if entry == "system_event":
                result["value"] = runtime.handle_system_event(source, trace_id="old")
            elif entry == "system_trigger":
                result["value"] = runtime.handle_system_trigger(trigger, trace_id="old")
            else:
                state = runtime.run_state_manager.start(source, trace_id="old", run_id="recover-run", run_version=1)
                outcome = AgentRuntimeResult(trace_id="old", conversation_id="boundary", final_reply="旧目标回复", status="completed")
                runtime._finalize_recovered_result(run_state=state, message=source, message_key=message_idempotency_key(source),
                    result=outcome, already_durable=False)
                result["value"] = outcome
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=work)
    thread.start()
    try:
        assert reached.wait(5)
        runtime.receive_user_message(user("latest", "新目标"), trace_id="latest")
        assert store.conversation_version("boundary") == 2
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and errors == []
    assert result["value"].status == "superseded" and result["value"].final_reply == ""
    cached = store.idempotent_message_result(message_idempotency_key(source))
    assert cached.status == "superseded" and cached.final_reply == ""
    run_ids = [event.content["run_id"] for event in runtime.trace_recorder.get_trace("old") if event.step == "agent_run_started"]
    assert run_ids and all(store.agent_run(identifier).runtime_status == "superseded" for identifier in run_ids)


def test_summary_uses_originating_run_version_before_snapshot_of_an_old_task(store):
    """旧 taskA 的运行不能在 v2 到达后捕获 v2 并把旧摘要写成当前事实。"""
    runtime = AgentRuntime(llm_client=StaticAgentClient([]), store=store)
    runtime.receive_user_message(user("old", "旧任务"), trace_id="old")
    old_task = store.current_task_context("boundary", "customer")
    stamp = now()
    new_task, _ = store.activate_task_context(conversation_id="boundary", customer_id="customer", trace_id="task-b",
        activity_at=stamp, started_at=stamp, reason="explicit_test_task_boundary", force_new=True, archive_previous=False)
    runtime.receive_user_message(user("new", "新任务"), trace_id="new")
    store.upsert_conversation_checkpoint(conversation_id="boundary", task_context_id=new_task.task_context_id,
        summary="新任务事实", facts={"current": "new"}, open_questions=[], trace_id="new")
    no_calls = StaticAgentClient([])
    summary = ContextSummaryManager(store=store, llm_client=no_calls)
    outcome = summary.summarize_for_context_budget(conversation_id="boundary", trace_id="old-summary",
        task_context_id=old_task.task_context_id, expected_run_version=1,
        estimated_context_tokens=100, max_context_tokens=100, trigger_threshold_tokens=80)
    assert outcome.transition is None and no_calls.calls == []
    assert store.current_task_context("boundary", "customer").task_context_id == new_task.task_context_id
    assert store.get_conversation_checkpoint("boundary").facts["current"] == "new"


def test_recovery_recheck_does_not_mutate_an_already_durable_cached_result(store):
    """恢复通知边界改向时保留原缓存证据，不能通过别名把已保存回复改成空。"""
    reached, release = threading.Event(), threading.Event()
    runtime = AgentRuntime(llm_client=StaticAgentClient([]), store=store)
    original = user("old", "原任务")
    runtime.receive_user_message(original, trace_id="old")
    state = runtime.run_state_manager.start(original, trace_id="old", run_id="recover-run", run_version=1)
    state.final_reply = "原始已保存回复"
    state.runtime_status = "completed"
    assert store.save_agent_run(state, expected_lease_owner=runtime.run_state_manager.worker_id)
    runtime.run_state_manager.mark_recoverable(state.run_id, error=RuntimeError("test restart"))
    original_result = AgentRuntimeResult(trace_id="old", conversation_id="boundary", final_reply="原始已保存回复", status="completed")
    store.remember_message_result(message_idempotency_key(original), original_result)
    runtime.hook_manager.register("after_agent_loop", lambda event: (reached.set(), release.wait(5)))
    results = []
    thread = threading.Thread(target=lambda: results.extend(runtime.resume_recoverable_runs()))
    thread.start()
    try:
        assert reached.wait(5)
        runtime.receive_user_message(user("new", "改变目标"), trace_id="new")
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert len(results) == 1 and results[0].status == "superseded" and results[0].final_reply == ""
    cached = store.idempotent_message_result(message_idempotency_key(original))
    assert cached.final_reply == "原始已保存回复" and cached.status == "completed"


def test_raw_fragment_invalidates_direct_run_before_aggregate_snapshot(store):
    """raw batch 接收也推进版本，跨入口改变目标不等待语义聚合或旧回复。"""
    reached, release = threading.Event(), threading.Event()
    contexts, results, errors = [], {}, []
    class Client:
        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]["content"])
            contexts.append(payload)
            if payload["current_message"]["message_id"] == "old":
                reached.set()
                assert release.wait(5)
                return action([{"name": "create_game", "arguments": {
                    "organizer_id": "customer", "organizer_name": "客户", "requirement": {"game_type": "red_center_mahjong", "stake": "1"},
                    "known_players": [],
                }, "reason": "旧目标动作"}])
            return action()
    runtime = AgentRuntime(llm_client=Client(), store=store)
    def old_work():
        try:
            results["old"] = runtime.handle_user_message(user("old", "原目标"), trace_id="old")
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=old_work)
    thread.start()
    fragment = user("raw-new", "改目标")
    deadline = now() + timedelta(seconds=30)
    try:
        assert reached.wait(5)
        with runtime.input_receipt_lock("boundary"):
            batch, _, added = store.upsert_pending_input_fragment(fragment, trace_id="raw-new", quiet_deadline=deadline)
            duplicate, _, repeated = store.upsert_pending_input_fragment(fragment, trace_id="raw-retry", quiet_deadline=deadline)
        assert added and not repeated and batch.version == duplicate.version == 1
        assert store.conversation_version("boundary") == 2
        assert [turn.content for turn in store.recent_turns("boundary") if turn.role.value == "user"] == ["原目标"]
    finally:
        release.set()
        thread.join(5)
    assert errors == [] and not thread.is_alive()
    assert results["old"].status == "superseded" and not results["old"].final_reply and not store.games
    aggregate = aggregate_pending_input_batch(batch, quiet_period_elapsed=True, trigger="test")
    latest = runtime.handle_user_message(aggregate, trace_id="latest")
    assert latest.final_reply == "好。" and store.conversation_version("boundary") == 3
    assert [turn["content"] for turn in contexts[-1]["recent_conversation"] if turn["role"] == "user"] == ["原目标", "改目标"]


def test_unfinished_goal_query_covers_receipt_gap_and_respects_customer_identity(store):
    """尚未 start run 的原始输入也必须触发重评，已终结输入和其他客户不能冒充。"""
    runtime = AgentRuntime(llm_client=StaticAgentClient([action()]), store=store)
    source = user("queued", "新目标")
    assert not store.has_unfinished_user_goal("boundary", "customer")
    runtime.receive_user_message(source, trace_id="queued")
    assert store.has_unfinished_user_goal("boundary", "customer")
    assert not store.has_unfinished_user_goal("boundary", "other-customer")
    assert not store.has_unfinished_user_goal("other-conversation", "customer")
    runtime.handle_user_message(source, trace_id="queued")
    assert not store.has_unfinished_user_goal("boundary", "customer")
    trigger = user("system-goal", "系统续办")
    run = runtime.run_state_manager.start(trigger, trace_id="system-goal", run_id="unfinished", run_version=1)
    assert store.has_unfinished_user_goal("boundary", "customer")
    runtime.run_state_manager.complete(run.run_id, final_reply="", runtime_status="completed")
    assert not store.has_unfinished_user_goal("boundary", "customer")


def test_old_uncached_receipt_and_late_old_run_do_not_extend_a_finished_new_goal(store):
    """v2 已完成后，v1 无缓存或迟到 running/recoverable 记录不能令所有后续闲聊重评。"""
    runtime = AgentRuntime(llm_client=StaticAgentClient([action()]), store=store)
    old = user("old-pending", "旧目标")
    latest = user("new-finished", "替换后的新目标")
    runtime.receive_user_message(old, trace_id="old-pending")
    assert store.has_unfinished_user_goal("boundary", "customer")
    runtime.receive_user_message(latest, trace_id="new-finished")
    runtime.handle_user_message(latest, trace_id="new-finished")
    assert store.idempotent_message_result(message_idempotency_key(old)) is None
    assert not store.has_unfinished_user_goal("boundary", "customer")
    late = runtime.run_state_manager.start(old, trace_id="late-old", run_id="late-old", run_version=1)
    assert not store.has_unfinished_user_goal("boundary", "customer")
    runtime.run_state_manager.mark_recoverable(late.run_id, error=RuntimeError("old worker interrupted"))
    assert not store.has_unfinished_user_goal("boundary", "customer")
