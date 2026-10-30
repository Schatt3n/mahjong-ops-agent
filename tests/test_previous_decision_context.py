"""同一运行保留合法模型计划，但不把提案升级为事实、权限或另一会话的上下文。"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from threading import Event

import pytest

from mahjong_agent_runtime import (
    AgentAction, AgentContextBuilder, AgentRuntime, InMemoryAgentStore, InMemoryTraceRecorder,
    SQLiteAgentStore, ToolCall, ToolGateway, UserMessage,
)
from mahjong_agent_runtime.budget import TokenBudget
from mahjong_agent_runtime.models import StateTransition, now
from mahjong_agent_runtime.services.context_service import ContextLifecycleManager
from mahjong_agent_runtime.services.run_state_service import AgentRunStateManager
from mahjong_agent_runtime.summary import ContextSummaryResult


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    value = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "previous-plan.sqlite3")
    yield value
    if request.param == "sqlite":
        value._connection.close()


def _query_action(marker="valid-plan", *, tool="search_current_games"):
    arguments = {"requirement": {"game_type": "hangzhou_mahjong", "stake": "1"}} if tool == "search_current_games" else {
        "start_at": "2099-01-01T18:00:00+08:00", "end_at": "2099-01-01T22:00:00+08:00",
    }
    return AgentAction(
        goal=marker, objective_status="needs_tool", reasoning_summary="RAW_REASON_NOT_IN_PROJECTION",
        objective_state={"known_facts": {"stake": "1"}, "missing_facts": ["己方人数"],
                         "fact_assertions": [{"field": "known_player_count", "value": 3, "role": "condition", "source_message_id": "old"}]},
        objective_plan=[{"step_id": "read", "title": "先查询后核对人数", "status": "in_progress", "depends_on": []}],
        plan_revision_reason="未知人数仍保持未知，不以目标桌况推断己方人数",
        tool_calls=[ToolCall(name=tool, arguments=arguments, reason="读取真实状态", call_id=tool, depends_on=[])],
        stop_reason={"can_stop": False, "why": "需要真实查询结果", "pending_work": ["读取状态"], "depends_on_tool_results": True},
    )


def _waiting_action():
    return AgentAction(goal="等待仍未知人数", objective_status="waiting_user", reasoning_summary="必要条件未建立",
        reply_to_user="你这边几个人？",
        stop_reason={"can_stop": True, "why": "等待必要信息", "pending_work": [], "depends_on_tool_results": True})


def _projection(action):
    return {"authority": "model_proposal_only", "goal": action.goal,
            "objective_state": action.objective_state, "objective_plan": action.objective_plan,
            "plan_revision_reason": action.plan_revision_reason}


@pytest.mark.parametrize("metadata", [{}, {"source": "group", "room_id": "room", "reply_constraints": {"max_length": 20}}])
def test_previous_decision_is_copied_user_data_not_promoted_fact_or_system_instruction(store, metadata):
    builder = AgentContextBuilder(store, ToolGateway(store))
    message = UserMessage(conversation_id="current", sender_id="customer", sender_name="客户", text="先看看", metadata=metadata)
    assert builder.build(message, trace_id="standalone").payload["previous_decision"] is None
    action = _query_action()
    original = deepcopy(action.to_dict())
    built = builder.build(message, trace_id="projection", previous_action=action)
    assert built.payload["previous_decision"] == _projection(action)
    assert json.loads(built.messages[-1]["content"])["previous_decision"] == _projection(action)
    assert "valid-plan" not in "\n".join(item["content"] for item in built.messages if item["role"] == "system")
    assert built.payload["explicit_task_facts"]["facts"] == {}
    assert "known_player_count" not in built.payload["explicit_task_facts"]["candidates"]
    assert set(built.payload["previous_decision"]) == {"authority", "goal", "objective_state", "objective_plan", "plan_revision_reason"}
    assert "RAW_REASON_NOT_IN_PROJECTION" not in json.dumps(built.payload["previous_decision"])
    built.payload["previous_decision"]["objective_state"]["missing_facts"].clear()
    built.payload["previous_decision"]["objective_plan"][0]["status"] = "done"
    assert action.to_dict() == original


def test_budget_summary_rebuild_keeps_previous_decision_as_the_same_low_trust_projection(store):
    class SummaryStub:
        def __init__(self, source):
            self.store = source

        def summarize_for_context_budget(self, **kwargs):
            return ContextSummaryResult(True, "测试重建传递", transition=StateTransition("checkpoint", "current", "", "updated", "summary", "budget"))

    lifecycle = ContextLifecycleManager(AgentContextBuilder(store, ToolGateway(store)), InMemoryTraceRecorder(),
        context_summary_manager=SummaryStub(store))
    message = UserMessage(conversation_id="current", sender_id="customer", sender_name="客户", text="继续看看")
    action = _query_action()
    args = dict(trace_id="budget", pending_tool_results=[], turn_tool_evidence=[], run_id="current-run", run_version=0,
                step_index=2, previous_action=action)
    built = lifecycle.build_and_trace_context(message, **args)
    rebuilt, transition = lifecycle.summarize_and_rebuild_context_if_needed(message, built=built,
        budget=TokenBudget(max_tokens_per_call=1), **args)
    assert transition is not None and rebuilt is not built
    assert rebuilt.payload["previous_decision"] == built.payload["previous_decision"] == _projection(action)
    assert json.loads(rebuilt.messages[-1]["content"])["previous_decision"] == _projection(action)


@pytest.mark.parametrize("metadata", [{}, {"source": "group", "room_id": "room", "reply_constraints": {"max_length": 20}}])
def test_main_loop_retains_condition_role_and_missing_facts_after_read(store, metadata):
    query = _query_action()
    seen = []

    class Client:
        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]["content"])
            seen.append(payload)
            return json.dumps((query if len(seen) == 1 else _waiting_action()).to_dict(), ensure_ascii=False)

    runtime = AgentRuntime(store=store, llm_client=Client(), trace_recorder=InMemoryTraceRecorder())
    result = runtime.handle_user_message(UserMessage(conversation_id="current", sender_id="customer", sender_name="客户",
        text="先看看有没有合适的局", message_id="old", metadata=metadata), trace_id="continue-plan")
    assert result.status == "waiting_user" and len(seen) == 2
    assert seen[0]["previous_decision"] is None
    assert seen[1]["previous_decision"] == _projection(query)
    assert seen[1]["turn_tool_evidence"][0]["name"] == "search_current_games"
    assert seen[1]["explicit_task_facts"]["facts"] == {}
    assert not store.games


def test_recovered_run_uses_its_checkpoint_action_without_repeating_the_read(store, monkeypatch):
    query = _query_action("recovered-plan")
    first_runtime = AgentRuntime(store=store, llm_client=SimpleClient([query]), trace_recorder=InMemoryTraceRecorder())
    original_checkpoint = AgentRunStateManager.checkpoint

    def checkpoint_then_crash(self, state, **kwargs):
        saved = original_checkpoint(self, state, **kwargs)
        if self is first_runtime.run_state_manager and kwargs["next_step_index"] == 2:
            raise RuntimeError("crash after committed action")
        return saved

    monkeypatch.setattr(AgentRunStateManager, "checkpoint", checkpoint_then_crash)
    with pytest.raises(RuntimeError, match="crash after committed action"):
        first_runtime.handle_user_message(UserMessage(conversation_id="current", sender_id="customer", sender_name="客户",
            text="先看看", message_id="recover-input"), trace_id="recover-plan")
    recoverable = store.recoverable_agent_runs(at=now(), limit=5)[0]
    assert len(recoverable.actions) == 1
    client = SimpleClient([_waiting_action()])
    restored = AgentRuntime(store=store, llm_client=client, trace_recorder=InMemoryTraceRecorder())
    results = restored.resume_recoverable_runs()
    assert len(results) == 1 and results[0].status == "waiting_user"
    assert client.payloads[0]["previous_decision"] == _projection(query)
    assert client.payloads[0]["turn_tool_evidence"][0]["name"] == "search_current_games"
    assert len([item for item in results[0].tool_results if item.name == "search_current_games"]) == 1


class SimpleClient:
    def __init__(self, actions):
        self.actions = list(actions)
        self.payloads = []

    def complete(self, messages, **kwargs):
        self.payloads.append(json.loads(messages[-1]["content"]))
        item = self.actions.pop(0)
        return item if isinstance(item, str) else json.dumps(item.to_dict(), ensure_ascii=False)


def test_contract_rejection_does_not_replace_previous_legal_plan_or_lose_recovery_evidence(store):
    query = _query_action("last-legal-plan")
    illegal = _query_action("ILLEGAL_PLAN_SENTINEL").to_dict()
    illegal["objective_plan"][0]["status"] = "not-a-plan-status"
    raw = json.dumps(illegal, ensure_ascii=False)
    room = _query_action("corrected-legal-plan", tool="check_room_availability")
    client = SimpleClient([query, raw, room, _waiting_action()])
    trace = InMemoryTraceRecorder()
    result = AgentRuntime(store=store, llm_client=client, trace_recorder=trace).handle_user_message(
        UserMessage(conversation_id="current", sender_id="customer", sender_name="客户", text="先看看", message_id="rejected-input"),
        trace_id="rejected-plan")
    assert client.payloads[2]["previous_decision"] == _projection(query)
    assert client.payloads[3]["previous_decision"] == _projection(room)
    assert [item.goal for item in result.actions] == [query.goal, room.goal, "等待仍未知人数"]
    assert len([event for event in trace.events if event.step == "action_contract_error"]) == 1
    feedback = [event for event in trace.events if event.step == "contract_error_feedback"]
    assert len(feedback) == 1 and feedback[0].content["result"]["raw_response"] == raw
    run_id = next(event.content["run_id"] for event in trace.events if event.step == "agent_run_started")
    run = store.agent_run(run_id)
    assert all(item["goal"] != "ILLEGAL_PLAN_SENTINEL" for item in run.actions)
    assert any(item["name"] == "agent_action_contract" and item["result"]["raw_response"] == raw for item in run.turn_tool_evidence)
    path = Path(__file__).resolve().parents[1] / "scripts/run_real_owner_chat_live_eval.py"
    spec = importlib.util.spec_from_file_location("previous_plan_recovery_assessor", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    metrics = module.assess_action_contract_recovery(trace.events, runtime_status=result.status,
        final_reply=result.final_reply, business_checks_passed=True, semantic_passed=True)
    assert metrics["action_contract_error_count"] == metrics["safe_recovered_count"] == 1, metrics
    assert metrics["unrecovered_count"] == 0 and metrics["quality_status"] == "passed"


def test_new_input_replans_without_reusing_the_old_runs_decision(store):
    blocked = Event()
    release = Event()
    seen = []
    query = _query_action("OLD_RUN_PRIVATE_PLAN")

    class Client:
        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]["content"])
            seen.append(payload)
            if payload["current_message"]["message_id"] == "new":
                assert payload["previous_decision"] is None
                return json.dumps(_waiting_action().to_dict(), ensure_ascii=False)
            if payload["previous_decision"] is None:
                return json.dumps(query.to_dict(), ensure_ascii=False)
            blocked.set()
            if not release.wait(5):
                raise RuntimeError("model barrier not released")
            return json.dumps(_waiting_action().to_dict(), ensure_ascii=False)

    runtime = AgentRuntime(store=store, llm_client=Client(), trace_recorder=InMemoryTraceRecorder())
    old = UserMessage(conversation_id="current", sender_id="customer", sender_name="客户", text="先看看", message_id="old")
    new = UserMessage(conversation_id="current", sender_id="customer", sender_name="客户", text="先不看了", message_id="new")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(runtime.handle_user_message, old, trace_id="old-plan")
        try:
            assert blocked.wait(5)
            runtime.receive_user_message(new, trace_id="new-plan")
            second = pool.submit(runtime.handle_user_message, new, trace_id="new-plan")
        finally:
            release.set()
        assert first.result(timeout=5).status == "superseded"
        assert second.result(timeout=5).status == "waiting_user"
    fresh = next(payload for payload in seen if payload["current_message"]["message_id"] == "new")
    assert fresh["previous_decision"] is None


@pytest.mark.parametrize("target_conversation,target_sender,metadata", [
    ("group:room:customer:customer", "customer", {"source": "group", "room_id": "room", "reply_constraints": {"max_length": 20}}),
    ("another-private-conversation", "another-customer", {}),
])
def test_private_plan_does_not_enter_another_customer_or_group_run(store, target_conversation, target_sender, metadata):
    query = _query_action("PRIVATE_PLAN_NOT_SHARED")
    client = SimpleClient([query, _waiting_action(), _waiting_action()])
    runtime = AgentRuntime(store=store, llm_client=client, trace_recorder=InMemoryTraceRecorder())
    runtime.handle_user_message(UserMessage(conversation_id="private", sender_id="customer", sender_name="客户",
        text="先看看", message_id="private-plan"), trace_id="private-plan")
    result = runtime.handle_user_message(UserMessage(conversation_id=target_conversation, sender_id=target_sender,
        sender_name="客户", text="先看看", message_id="another-run", metadata=metadata), trace_id="another-run")
    assert result.status == "waiting_user"
    assert client.payloads[1]["previous_decision"] == _projection(query)
    assert client.payloads[2]["previous_decision"] is None
    assert "PRIVATE_PLAN_NOT_SHARED" not in json.dumps(client.payloads[2], ensure_ascii=False)
