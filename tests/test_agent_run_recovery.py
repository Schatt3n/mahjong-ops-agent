from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest

from mahjong_agent_runtime import (
    AgentRunStatus,
    AgentRuntime,
    InMemoryAgentStore,
    InMemoryTraceRecorder,
    SQLiteAgentStore,
    StaticAgentClient,
    UserMessage,
)
from mahjong_agent_runtime.budget import TokenBudget
from mahjong_agent_runtime.input_aggregation import aggregate_pending_input_batch
from mahjong_agent_runtime.llm import LLMResponseFormatError
from mahjong_agent_runtime.models import now
from mahjong_agent_runtime.services import AgentRunLeaseLostError, AgentRunStateManager


def _action_json(
    *,
    objective_status: str,
    reply: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
) -> str:
    calls = tool_calls or []
    needs_tool = objective_status == "needs_tool"
    return json.dumps(
        {
            "goal": "完成恢复测试中的用户目标",
            "objective_status": objective_status,
            "reasoning_summary": "根据当前事实继续执行，不重放已经完成的步骤。",
            "objective_state": {
                "current_phase": "recovery_test",
                "known_facts": {},
                "missing_facts": [],
                "blockers": [],
            },
            "objective_plan": [
                {
                    "step_id": "step_1",
                    "title": "恢复并完成",
                    "status": "in_progress" if needs_tool else "done",
                    "tool": calls[0]["name"] if calls else None,
                    "depends_on": [],
                    "decision_rule": "只执行尚未完成的动作。",
                }
            ],
            "plan_revision_reason": "测试恢复语义。",
            "reply_to_user": "" if needs_tool else reply,
            "tool_calls": calls,
            "needs_human": False,
            "stop_reason": {
                "can_stop": not needs_tool,
                "why": "尚需工具结果。" if needs_tool else "目标已完成。",
                "pending_work": [item["name"] for item in calls],
                "depends_on_tool_results": False,
            },
            "badcase": None,
        },
        ensure_ascii=False,
    )


def _search_action() -> str:
    return _action_json(
        objective_status="needs_tool",
        tool_calls=[
            {
                "name": "search_current_games",
                "arguments": {
                    "requirement": {
                        "game_type": "hangzhou_mahjong",
                        "stake": "0.5",
                    },
                    "limit": 5,
                },
                "reason": "先查询当前局池。",
            }
        ],
    )


def _create_game_action() -> str:
    return _action_json(
        objective_status="needs_tool",
        tool_calls=[
            {
                "name": "create_game",
                "arguments": {
                    "requirement": {
                        "game_type": "hangzhou_mahjong",
                        "stake": "0.5",
                        "smoke_preference": "no_smoking",
                        "start_time_kind": "asap_when_full",
                    },
                    "organizer_id": "zhang",
                    "organizer_name": "张哥",
                    "known_players": [
                        {
                            "customer_id": "zhang",
                            "display_name": "张哥",
                            "seat_count": 1,
                        }
                    ],
                },
                "reason": "创建一个待组局。",
            }
        ],
    )


class _InterruptiblePlanner(StaticAgentClient):
    """保留真实主循环请求，按顺序注入技术故障和后续模型动作。"""

    def complete(self, messages, *, trace_id, timeout_seconds):
        response = super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)
        if isinstance(response, Exception):
            raise response
        return response


def _output_limit_error(reason: str = "max_output_tokens") -> LLMResponseFormatError:
    """构造收到完整 HTTP 包但没有可解析主模型动作的 Responses 响应。"""
    return LLMResponseFormatError("Responses envelope incomplete", {"attempts": [{
        "http_status": 200, "body_complete": True, "response_json_parsed": True,
        "response_status": "incomplete", "incomplete_reason": reason,
    }]})


@pytest.mark.parametrize(("outputs", "budget", "recovery_reason", "expected"), [
    ((TimeoutError("read timeout"),), 2, "transport_timeout", "completed"),
    ((TimeoutError("read timeout"), TimeoutError("retry timeout")), 3, "transport_timeout", "needs_human"),
    ((TimeoutError("read timeout"),), 1, "", "needs_human"),
    ((_output_limit_error("content_filter"),), 3, "", "needs_human"),
])
def test_planner_failure_is_recovered_once_only_when_a_safe_budgeted_retry_exists(outputs, budget, recovery_reason, expected):
    """首步也允许恢复，超时耗尽、预算不足和其他非完整原因不会冒充业务成功。"""
    client = _InterruptiblePlanner(list(outputs) + [_action_json(objective_status="completed", reply="你好，晚上好。")])
    runtime = AgentRuntime(llm_client=client, token_budget=TokenBudget(max_calls_per_turn=budget))
    message = UserMessage(conversation_id="planner-fault", sender_id="zhang", sender_name="模拟顾客",
                          text="老板晚上好", message_id="same-planner-input")
    result = runtime.handle_user_message(message)
    state, = runtime.store.agent_runs.values()
    assert result.status == expected
    assert state.planner_recovery_reason == recovery_reason
    assert state.planner_recovery_step == (1 if recovery_reason else 0)
    assert len(client.calls) == (2 if recovery_reason else 1) == state.budget_state["agent_calls"]
    assert not result.tool_results
    assert runtime.handle_user_message(message).to_dict() == result.to_dict()
    assert len(client.calls) == (2 if recovery_reason else 1)
    if expected == "completed":
        assert result.final_reply == "你好，晚上好。"
    else:
        assert result.final_reply != "你好，晚上好。"


def test_sqlite_planner_recovery_checkpoint_preserves_tool_result_and_spent_budget(tmp_path, monkeypatch):
    """在响应截断后的检查点强制中断，重启只能复用已查结果恢复同一步。"""
    database = tmp_path / "planner-recovery.db"
    store = SQLiteAgentStore(database)
    first = AgentRuntime(llm_client=_InterruptiblePlanner([_search_action(), _output_limit_error()]), store=store)
    original_checkpoint = AgentRunStateManager.checkpoint

    def interrupt_recovery(self, state, **kwargs):
        saved = original_checkpoint(self, state, **kwargs)
        if self is first.run_state_manager and state.planner_recovery_step == kwargs["next_step_index"]:
            raise RuntimeError("interrupted after saving same-step recovery")
        return saved

    message = UserMessage(conversation_id="interrupted-planner", sender_id="zhang", sender_name="模拟顾客",
                          text="今晚半块底杭麻有现成的吗？", message_id="planner-same-step")
    with monkeypatch.context() as patch:
        patch.setattr(AgentRunStateManager, "checkpoint", interrupt_recovery)
        with pytest.raises(RuntimeError, match="after saving same-step recovery"):
            first.handle_user_message(message, trace_id="planner-transport-recovery")

    before, = store.recoverable_agent_runs(at=now(), limit=2)
    assert before.planner_recovery_step == before.next_step_index == 2
    assert before.planner_recovery_reason == "max_output_tokens"
    assert before.budget_state["agent_calls"] == 2
    assert [item["name"] for item in before.tool_results] == ["search_current_games"]
    assert before.pending_tool_results == before.tool_results
    store._connection.close()

    reopened = SQLiteAgentStore(database)
    client = _InterruptiblePlanner([_action_json(objective_status="completed", reply="这会儿没查到半块底杭麻。")])
    resumed = AgentRuntime(llm_client=client, store=reopened)
    try:
        result, = resumed.resume_recoverable_runs()
        assert result.status == "completed" and result.final_reply == "这会儿没查到半块底杭麻。"
        assert len(client.calls) == 1 and [item.name for item in result.tool_results] == ["search_current_games"]
        messages = client.calls[0]["messages"]
        assert "主模型输出达到上限" in messages[-1]["content"]
        payload = json.loads(messages[-2]["content"])
        assert [item["name"] for item in payload["turn_tool_evidence"]] == ["search_current_games"]
        after = reopened.agent_run(before.run_id)
        assert after.status == AgentRunStatus.COMPLETED and after.budget_state["agent_calls"] == 3
        assert resumed.handle_user_message(message).to_dict() == result.to_dict()
        assert len(client.calls) == 1
    finally:
        reopened._connection.close()


def test_planner_response_limit_after_create_does_not_reexecute_the_write():
    """建桌已经提交时，只恢复截断的模型步骤并核对实际幂等业务结果。"""
    store = InMemoryAgentStore()
    final = _action_json(objective_status="completed", reply="这桌已经给你登记，目前还缺三个人。")

    class Planner(_InterruptiblePlanner):
        def complete(self, messages, *, trace_id, timeout_seconds):
            if len(self.calls) == 2:
                game_id, = store.games
                self.outputs.insert(0, _action_json(objective_status="needs_tool", tool_calls=[{
                    "name": "search_customers", "arguments": {"game_id": game_id}, "reason": "查找该局的可邀约顾客。",
                }]))
            return super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)

    client = Planner([_create_game_action(), _output_limit_error(), final])
    runtime = AgentRuntime(llm_client=client, store=store)
    message = UserMessage(conversation_id="post-create-output-cap", sender_id="zhang", sender_name="张哥",
                          text="帮我组个0.5无烟人齐开的", message_id="post-create-output-cap-message")

    result = runtime.handle_user_message(message)

    assert result.status == "completed" and result.final_reply == json.loads(final)["reply_to_user"]
    assert len(store.games) == 1 and len(client.calls) == 4
    state, = store.agent_runs.values()
    assert state.planner_recovery_reason == "max_output_tokens" and state.planner_recovery_step == 2
    assert [(item.name, item.deduplicated) for item in result.tool_results] == [
        ("create_game", False), ("search_customers", False),
    ]
    assert len([item for item in store.transitions if item.entity_type == "game" and item.from_status is None]) == 1
    assert runtime.handle_user_message(message).to_dict() == result.to_dict() and len(client.calls) == 4


def _inject_second_checkpoint_crash(
    monkeypatch,
    runtime: AgentRuntime,
    *,
    persist_before_crash: bool,
    error_message: str,
) -> None:
    """Crash at the deterministic boundary after the first Agent step."""

    original_checkpoint = AgentRunStateManager.checkpoint
    checkpoint_calls = 0

    def crash_at_second_checkpoint(self, *args, **kwargs):
        nonlocal checkpoint_calls
        if self is runtime.run_state_manager:
            checkpoint_calls += 1
            if checkpoint_calls == 2:
                if persist_before_crash:
                    original_checkpoint(self, *args, **kwargs)
                raise RuntimeError(error_message)
        return original_checkpoint(self, *args, **kwargs)

    monkeypatch.setattr(
        AgentRunStateManager,
        "checkpoint",
        crash_at_second_checkpoint,
    )


def test_silent_terminal_checkpoint_recovers_without_any_new_model_call(monkeypatch):
    """空客户回执也可能是已办结，重开时保留静默状态。"""
    store = InMemoryAgentStore()
    action = json.loads(_action_json(objective_status="completed"))
    action.update(response_mode="silent", silence_reason="这条简单致意无需打扰")
    first = AgentRuntime(llm_client=StaticAgentClient([json.dumps(action)]), store=store)
    _inject_second_checkpoint_crash(monkeypatch, first, persist_before_crash=True, error_message="silent terminal checkpoint")
    message = UserMessage(conversation_id="silent-planner", sender_id="zhang", sender_name="模拟顾客",
                          text="好嘞", message_id="silent-planner-message")

    with pytest.raises(RuntimeError, match="silent terminal checkpoint"):
        first.handle_user_message(message)
    before, = store.recoverable_agent_runs(at=now(), limit=2)
    assert before.runtime_status == "completed" and before.final_reply == ""

    client = StaticAgentClient([])
    replay = AgentRuntime(llm_client=client, store=store)
    result, = replay.resume_recoverable_runs()

    assert result.status == "completed" and result.final_reply == "" and not client.calls
    assert result.actions[-1].response_mode == "silent"


def test_sqlite_restart_resumes_from_checkpoint_with_complete_tool_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    database = tmp_path / "agent-recovery.db"
    first_store = SQLiteAgentStore(database)
    first_client = StaticAgentClient([_search_action()])
    first_runtime = AgentRuntime(
        llm_client=first_client,
        store=first_store,
        trace_recorder=InMemoryTraceRecorder(),
    )
    _inject_second_checkpoint_crash(
        monkeypatch,
        first_runtime,
        persist_before_crash=True,
        error_message="simulated process crash after post-tool checkpoint",
    )
    message = UserMessage(
        conversation_id="recovery_sqlite",
        sender_id="zhang",
        sender_name="张哥",
        text="0.5 有人吗",
        message_id="msg_recovery_sqlite",
    )

    with pytest.raises(RuntimeError, match="after post-tool checkpoint"):
        first_runtime.handle_user_message(message, trace_id="trace_recovery_sqlite")

    failed_run = first_store.recoverable_agent_runs(at=now(), limit=10)[0]
    assert failed_run.status == AgentRunStatus.RECOVERABLE
    assert failed_run.next_step_index == 2
    assert failed_run.tool_results[0]["name"] == "search_current_games"

    second_store = SQLiteAgentStore(database)
    second_client = StaticAgentClient(
        [_action_json(objective_status="completed", reply="现在没有现成的，要组一个吗？")]
    )
    recovered_deliveries: list[tuple[str, str]] = []
    second_runtime = AgentRuntime(
        llm_client=second_client,
        store=second_store,
        trace_recorder=InMemoryTraceRecorder(),
        recovered_result_handler=lambda restored_message, result, _: recovered_deliveries.append(
            (restored_message.message_id, result.final_reply)
        ),
    )

    recovered = second_runtime.resume_recoverable_runs()

    assert [item.final_reply for item in recovered] == ["现在没有现成的，要组一个吗？"]
    assert recovered[0].tool_results[0].name == "search_current_games"
    assert recovered_deliveries == [
        ("msg_recovery_sqlite", "现在没有现成的，要组一个吗？")
    ]
    resumed_payload = json.loads(second_client.calls[0]["messages"][-1]["content"])
    assert resumed_payload["previous_tool_results"][0]["name"] == "search_current_games"
    assert resumed_payload["turn_tool_evidence"][0]["name"] == "search_current_games"
    persisted = second_store.agent_run(failed_run.run_id)
    assert persisted is not None
    assert persisted.status == AgentRunStatus.COMPLETED


def test_sqlite_resume_repairs_only_executed_dependency_from_restored_current_turn(tmp_path, monkeypatch) -> None:
    database = tmp_path / "historical-dependency.db"
    first_store = SQLiteAgentStore(database)
    search = json.loads(_search_action())
    search["tool_calls"][0].update(call_id="read_pool", depends_on=[])
    first_runtime = AgentRuntime(
        llm_client=StaticAgentClient([json.dumps(search)]), store=first_store,
        trace_recorder=InMemoryTraceRecorder(),
    )
    _inject_second_checkpoint_crash(
        monkeypatch, first_runtime, persist_before_crash=True, error_message="crash after persisted read",
    )
    message = UserMessage(conversation_id="historical-dependency", sender_id="zhang", sender_name="张哥",
                          text="0.5有人吗", message_id="historical-dependency-message")
    with pytest.raises(RuntimeError, match="after persisted read"):
        first_runtime.handle_user_message(message, trace_id="historical-dependency-trace")
    failed_run = first_store.recoverable_agent_runs(at=now(), limit=10)[0]
    assert failed_run.turn_tool_evidence[0]["call_id"] == "read_pool"
    first_store._connection.close()

    second_store = SQLiteAgentStore(database)
    trace = InMemoryTraceRecorder()
    second_client = StaticAgentClient([
        _action_json(objective_status="needs_tool", tool_calls=[{
            "name": "search_customers", "call_id": "find_players", "depends_on": ["read_pool"],
            "arguments": {"requirement": {"game_type": "hangzhou_mahjong", "stake": "0.5"}, "limit": 5},
            "reason": "继续读取候选结果",
        }]),
        _action_json(objective_status="completed", reply="这会儿还没组上。"),
    ])
    second_runtime = AgentRuntime(llm_client=second_client, store=second_store, trace_recorder=trace)
    try:
        recovered = second_runtime.resume_recoverable_runs()
        assert len(recovered) == 1 and recovered[0].status == "completed"
        assert [result.name for result in recovered[0].tool_results] == ["search_current_games", "search_customers"]
        assert [result.call_id for result in recovered[0].tool_results] == ["read_pool", "find_players"]
        assert len(second_client.calls) == 2
        events = trace.get_trace(recovered[0].trace_id)
        assert "action_contract_error" not in [event.step for event in events]
        repair = next(event for event in events if event.step == "action_contract_repaired")
        assert repair.content["repairs"][0]["satisfied_historical_call_ids"] == ["read_pool"]
        assert second_store.agent_run(failed_run.run_id).status == AgentRunStatus.COMPLETED
    finally:
        second_store._connection.close()


def test_crash_after_side_effect_before_checkpoint_reuses_tool_idempotency(monkeypatch) -> None:
    store = InMemoryAgentStore()
    first_runtime = AgentRuntime(
        llm_client=StaticAgentClient([_create_game_action()]),
        store=store,
        trace_recorder=InMemoryTraceRecorder(),
    )
    message = UserMessage(
        conversation_id="recovery_tool_idempotency",
        sender_id="zhang",
        sender_name="张哥",
        text="帮我组个0.5无烟人齐开的",
        message_id="msg_recovery_tool_idempotency",
    )
    with monkeypatch.context() as scoped:
        _inject_second_checkpoint_crash(
            scoped,
            first_runtime,
            persist_before_crash=False,
            error_message="simulated crash before post-tool checkpoint",
        )
        with pytest.raises(RuntimeError, match="post-tool checkpoint"):
            first_runtime.handle_user_message(
                message,
                trace_id="trace_recovery_tool_idempotency",
            )

    assert len(store.games) == 1
    recovery_client = StaticAgentClient(
        [
            _create_game_action(),
            _action_json(objective_status="completed", reply="好的，我帮你问问。"),
        ]
    )
    second_runtime = AgentRuntime(
        llm_client=recovery_client,
        store=store,
        trace_recorder=InMemoryTraceRecorder(),
    )

    recovered = second_runtime.resume_recoverable_runs()

    assert len(store.games) == 1
    assert recovered[0].tool_results[0].name == "create_game"
    assert recovered[0].tool_results[0].deduplicated is True


def test_newer_user_message_supersedes_recoverable_run(monkeypatch) -> None:
    store = InMemoryAgentStore()
    first_runtime = AgentRuntime(
        llm_client=StaticAgentClient([_search_action()]),
        store=store,
        trace_recorder=InMemoryTraceRecorder(),
    )
    _inject_second_checkpoint_crash(
        monkeypatch,
        first_runtime,
        persist_before_crash=True,
        error_message="simulated process crash after post-tool checkpoint",
    )
    first_message = UserMessage(
        conversation_id="recovery_superseded",
        sender_id="zhang",
        sender_name="张哥",
        text="0.5有人吗",
        message_id="msg_recovery_superseded_1",
    )
    with pytest.raises(RuntimeError):
        first_runtime.handle_user_message(
            first_message,
            trace_id="trace_recovery_superseded_1",
        )
    stale_run = first_runtime.run_state_manager.recoverable()[0]

    second_runtime = AgentRuntime(
        llm_client=StaticAgentClient(
            [_action_json(objective_status="completed", reply="好，按你刚补充的来。")]
        ),
        store=store,
        trace_recorder=InMemoryTraceRecorder(),
    )
    second_runtime.handle_user_message(
        UserMessage(
            conversation_id="recovery_superseded",
            sender_id="zhang",
            sender_name="张哥",
            text="改成1块有烟",
            message_id="msg_recovery_superseded_2",
        ),
        trace_id="trace_recovery_superseded_2",
    )

    persisted = store.agent_run(stale_run.run_id)
    assert persisted is not None
    assert persisted.status == AgentRunStatus.SUPERSEDED
    assert second_runtime.resume_recoverable_runs() == []


def test_recovered_delivery_failure_retries_cached_terminal_result(monkeypatch) -> None:
    store = InMemoryAgentStore()
    first_runtime = AgentRuntime(
        llm_client=StaticAgentClient(
            [_action_json(objective_status="completed", reply="好，我帮你看看。")]
        ),
        store=store,
        trace_recorder=InMemoryTraceRecorder(),
    )
    original_complete = AgentRunStateManager.complete

    def crash_before_completion(self, *args, **kwargs):
        if self is first_runtime.run_state_manager:
            raise RuntimeError("simulated crash before run completion")
        return original_complete(self, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(AgentRunStateManager, "complete", crash_before_completion)
        with pytest.raises(RuntimeError, match="before run completion"):
            first_runtime.handle_user_message(
                UserMessage(
                    conversation_id="recovery_delivery",
                    sender_id="zhang",
                    sender_name="张哥",
                    text="0.5无烟有人吗",
                    message_id="msg_recovery_delivery",
                ),
                trace_id="trace_recovery_delivery",
            )

    running = next(iter(store.agent_runs.values()))
    recovery_at = (running.lease_until or now()) + timedelta(seconds=1)
    delivery_attempts: list[str] = []

    def flaky_delivery(_, result, __) -> None:
        delivery_attempts.append(result.final_reply)
        if len(delivery_attempts) == 1:
            raise RuntimeError("bridge temporarily unavailable")

    no_call_client = StaticAgentClient([])
    second_runtime = AgentRuntime(
        llm_client=no_call_client,
        store=store,
        trace_recorder=InMemoryTraceRecorder(),
        recovered_result_handler=flaky_delivery,
    )

    assert second_runtime.resume_recoverable_runs(at=recovery_at) == []
    after_failure = store.agent_run(running.run_id)
    assert after_failure is not None
    assert after_failure.status == AgentRunStatus.RECOVERABLE

    recovered = second_runtime.resume_recoverable_runs(at=recovery_at)

    assert [item.final_reply for item in recovered] == ["好，我帮你看看。"]
    assert delivery_attempts == ["好，我帮你看看。", "好，我帮你看看。"]
    assert no_call_client.calls == []
    completed = store.agent_run(running.run_id)
    assert completed is not None
    assert completed.status == AgentRunStatus.COMPLETED


def test_stale_worker_cannot_complete_run_owned_by_recovery_worker() -> None:
    store = InMemoryAgentStore()
    first = AgentRunStateManager(
        store=store,
        trace_recorder=InMemoryTraceRecorder(),
        worker_id="worker_first",
    )
    second = AgentRunStateManager(
        store=store,
        trace_recorder=InMemoryTraceRecorder(),
        worker_id="worker_second",
    )
    state = first.start(
        UserMessage(
            conversation_id="lease_cas",
            sender_id="zhang",
            sender_name="张哥",
            text="测试",
            message_id="msg_lease_cas",
        ),
        trace_id="trace_lease_cas",
        run_id="run_lease_cas",
        run_version=0,
    )
    first.mark_recoverable(state.run_id, error=RuntimeError("worker stopped"))
    claimed = second.claim(state.run_id)
    assert claimed is not None

    with pytest.raises(AgentRunLeaseLostError):
        first.complete(
            state.run_id,
            final_reply="旧结果",
            runtime_status="completed",
        )

    current = store.agent_run(state.run_id)
    assert current is not None
    assert current.status == AgentRunStatus.RUNNING
    assert current.lease_owner == "worker_second"


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("scenario", ["recovered", "exhausted", "acknowledged", "staff_claimed"])
def test_reply_review_timeout_opens_handoff_before_retrying_the_same_draft(tmp_path, backend, scenario):
    """实走查局和回执审查，核对超时、写入租约、店家待办及原稿复审。"""
    store = InMemoryAgentStore() if backend == "memory" else SQLiteAgentStore(tmp_path / "review_attention.sqlite")
    reply = "我查过了，现在还没凑上。"
    approved = json.dumps({
        "approved": True, "needs_human": False, "reasoning_summary": "按实际查局结果答复。", "violations": [],
        "item_reviews": [{"item_id": "reply_to_user", "approved": True, "follow_up_commitments": [],
                          "reasoning_summary": "只告知本次查询。", "violations": [],
                          "suggested_safe_text": reply}],
    }, ensure_ascii=False)
    review = _InterruptiblePlanner([TimeoutError("first read"), approved if scenario == "recovered"
                                    else TimeoutError("second read")])
    planner = StaticAgentClient([_search_action(), _action_json(objective_status="completed", reply=reply)])
    trace = InMemoryTraceRecorder()
    runtime = AgentRuntime(llm_client=planner, store=store, trace_recorder=trace,
                           reply_self_review_enabled=True, reply_self_review_client=review)
    message = UserMessage(conversation_id="review-attention", sender_id="zhang", sender_name="模拟顾客",
                          text="现在杭麻0.5有人吗", message_id="review-attention-msg")
    seen = []
    def on_attention(event):
        queued, = store.pending_agent_run_handoffs()
        assert queued.run_id == event.payload["run_id"]
        assert queued.handoff_reason == "reply_review_timeout"
        assert len(review.calls) == 1
        assert event.payload["retrying"] is True
        seen.append(queued.run_id)
        if scenario == "acknowledged":
            queued.handoff_wait_delivered = True
            assert store.save_agent_run(queued, expected_lease_owner=queued.lease_owner)
        if scenario == "staff_claimed":
            assert store.claim_agent_run_handoff(queued.run_id, expected_conversation_version=queued.run_version)
    runtime.hook_manager.register("agent_runtime_attention", on_attention)

    result = runtime.handle_user_message(message, trace_id="review-attention-test")

    assert len(seen) == 1
    assert len(planner.calls) == 2
    assert sum(item.name == "search_current_games" and item.called for item in result.tool_results) == 1
    events = [item.step for item in trace.get_trace("review-attention-test")]
    if scenario == "staff_claimed":
        assert len(review.calls) == 1
        assert result.status == "superseded" and result.final_reply == ""
        assert store.pending_agent_run_handoffs() == []
        return
    assert review.calls[0]["messages"] == review.calls[1]["messages"]
    if scenario == "recovered":
        assert result.status == "completed" and result.final_reply == reply
        assert store.pending_agent_run_handoffs() == []
        assert events.index("agent_runtime_attention") < events.index("customer_visible_content_review_retry_result")
        assert "agent_runtime_attention_resolved" in events
    else:
        assert result.status == "needs_human"
        assert result.final_reply == ("" if scenario == "acknowledged" else "稍等一下")
        assert len(store.pending_agent_run_handoffs()) == 1
        assert "agent_runtime_attention_resolved" not in events
        pending = [turn for turn in store.recent_turns(message.conversation_id) if turn.role.value == "assistant"]
        assert [turn.content for turn in pending] == ([] if scenario == "acknowledged" else ["稍等一下"])


@pytest.mark.parametrize("scope", ["matches", "wrong_source", "wrong_batch_version", "wrong_trace"])
def test_aggregated_wait_receipt_only_follows_its_exact_original_run(scope):
    """聚合入口只把已确证的等待回执交给本次来话，不从元数据或旧批次借用。"""
    store = InMemoryAgentStore()
    runtime = AgentRuntime(llm_client=StaticAgentClient(outputs=[]), store=store)
    conversation = "wechaty:contact:guest"
    trace_id = "gate-aggregate-proof"
    for source_id, text in (("first-fragment", "老板"), ("last-fragment", "现在有杭麻吗？")):
        batch, _, _ = store.upsert_pending_input_fragment(
            UserMessage(conversation, "guest", "客人", text, message_id=source_id),
            trace_id=trace_id, quiet_deadline=now(),
        )
    store.queue_pending_input_batch(batch_id=batch.batch_id, expected_version=batch.version, due_at=batch.quiet_deadline,
        decision={"gate_recovery": {"open": True, "wait_delivered": True, "route": "aggregated", "attempt": 1,
            "origin_trace_id": trace_id, "conversation_version": store.conversation_version(conversation)}})
    claimed, _ = store.claim_pending_input_batch(batch_id=batch.batch_id, expected_version=batch.version, trace_id=trace_id)
    message = aggregate_pending_input_batch(claimed, quiet_period_elapsed=True, trigger="input_gate_recovery")
    if scope == "wrong_source":
        message.metadata["input_window"]["source_message_ids"][-1] = "other-fragment"
    if scope == "wrong_batch_version":
        message.metadata["input_window"]["batch_version"] -= 1
    if scope == "wrong_trace":
        trace_id = "other-trace"

    receipt = runtime.receive_user_message(message, trace_id=trace_id)
    state = runtime.run_state_manager.start(message, trace_id=trace_id, run_id="aggregate-run",
                                          run_version=receipt["conversation_version"])

    assert state.handoff_wait_delivered is (scope == "matches")
