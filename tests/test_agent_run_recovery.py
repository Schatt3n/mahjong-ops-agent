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
    resumed_payload = json.loads(second_client.calls[0]["messages"][1]["content"])
    assert resumed_payload["previous_tool_results"][0]["name"] == "search_current_games"
    assert resumed_payload["turn_tool_evidence"][0]["name"] == "search_current_games"
    persisted = second_store.agent_run(failed_run.run_id)
    assert persisted is not None
    assert persisted.status == AgentRunStatus.COMPLETED


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
