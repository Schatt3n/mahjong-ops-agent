from __future__ import annotations

import json
from typing import Any

from mahjong_agent_runtime import (
    AgentAction,
    AgentRuntime,
    CustomerProfile,
    InMemoryAgentStore,
    InMemoryTraceRecorder,
    ToolCall,
    ToolGateway,
    ToolResult,
    UserMessage,
)
from mahjong_agent_runtime.domains.context_builders.tool_results import compact_tool_payload
from mahjong_agent_runtime.services.objective_continuation import blocking_continuation


def test_create_game_returns_open_recruitment_continuation() -> None:
    store = _seeded_store()
    gateway = ToolGateway(store)

    result = gateway.execute(
        ToolCall(
            name="create_game",
            arguments={
                "organizer_id": "zhang",
                "organizer_name": "张哥",
                "requirement": _requirement(),
                "known_players": [
                    {
                        "customer_id": "zhang",
                        "display_name": "张哥",
                        "seat_count": 3,
                    }
                ],
            },
            reason="创建明确的三缺一组局。",
        ),
        trace_id="trace_create_continuation",
        conversation_id="continuation_contract",
        sender_id="zhang",
        sender_name="张哥",
        step_index=100,
        source_message_id="msg_create_continuation",
    )

    continuation = result.result["continuation"]
    assert result.allowed is True
    assert continuation["can_stop"] is False
    assert continuation["pending_capabilities"] == ["discover_candidates"]
    assert continuation["suggested_tools"] == ["search_customers"]
    assert continuation["authoritative_facts"]["seat_summary"]["claimed_seats"] == 3
    assert continuation["authoritative_facts"]["seat_summary"]["remaining_seats"] == 1
    assert continuation["authoritative_facts"]["exclude_customer_ids"] == ["zhang"]


def test_unresolved_continuation_rejects_terminal_action_but_not_next_tool() -> None:
    continuation_result = ToolResult(
        name="create_game",
        called=True,
        allowed=True,
        result={
            "continuation": {
                "can_stop": False,
                "allowed_terminal_statuses": [],
                "pending_capabilities": ["discover_candidates"],
            }
        },
    )
    premature = AgentAction(
        goal="组局",
        objective_status="completed",
        reasoning_summary="建局后直接结束。",
        reply_to_user="好。",
    )
    advancing = AgentAction(
        goal="组局",
        objective_status="needs_tool",
        reasoning_summary="继续找候选人。",
        tool_calls=[ToolCall(name="search_customers", arguments={"requirement": _requirement()})],
    )

    assert blocking_continuation(premature, [continuation_result]) is not None
    assert blocking_continuation(advancing, [continuation_result]) is None


def test_waiting_user_is_allowed_only_when_continuation_explicitly_allows_it() -> None:
    waiting = AgentAction(
        goal="组局",
        objective_status="waiting_user",
        reasoning_summary="需要征求是否继续等待。",
        reply_to_user="暂时没人，要继续帮你留意吗？",
    )
    denied = ToolResult(
        name="search_customers",
        called=True,
        allowed=True,
        result={"continuation": {"can_stop": False, "allowed_terminal_statuses": []}},
    )
    allowed = ToolResult(
        name="search_customers",
        called=True,
        allowed=True,
        result={
            "continuation": {
                "can_stop": False,
                "allowed_terminal_statuses": ["waiting_user"],
            }
        },
    )

    assert blocking_continuation(waiting, [denied]) is not None
    assert blocking_continuation(waiting, [allowed]) is None


def test_tool_result_compaction_preserves_continuation_contract() -> None:
    continuation = {
        "can_stop": False,
        "authoritative_facts": {"game_id": "game_123"},
        "pending_capabilities": ["discover_candidates"],
    }

    compacted = compact_tool_payload({"game": {}, "continuation": continuation})

    assert compacted["continuation"] == continuation


def test_empty_candidate_search_closes_only_the_synchronous_outreach_phase() -> None:
    store = _seeded_store()
    gateway = ToolGateway(store)
    created = gateway.execute(
        ToolCall(
            name="create_game",
            arguments={
                "organizer_id": "zhang",
                "organizer_name": "张哥",
                "requirement": {**_requirement(), "game_type": "sichuan_mahjong"},
                "known_players": [
                    {"customer_id": "zhang", "display_name": "张哥", "seat_count": 3}
                ],
            },
        ),
        trace_id="trace_empty_candidates_create",
        conversation_id="empty_candidates",
        sender_id="zhang",
        sender_name="张哥",
        step_index=100,
        source_message_id="msg_empty_candidates",
    )
    assert created.result["continuation"]["can_stop"] is False

    searched = gateway.execute(
        ToolCall(
            name="search_customers",
            arguments={
                "requirement": {**_requirement(), "game_type": "sichuan_mahjong"},
                "exclude_customer_ids": ["zhang", "ran"],
                "limit": 8,
            },
        ),
        trace_id="trace_empty_candidates_search",
        conversation_id="empty_candidates",
        sender_id="zhang",
        sender_name="张哥",
        step_index=200,
        source_message_id="msg_empty_candidates",
    )

    assert searched.result["candidates"] == []
    assert searched.result["continuation"]["can_stop"] is True
    assert searched.result["continuation"]["pending_capabilities"] == []
    assert store.require_game(created.result["game"]["game_id"]).remaining_seats() == 1


def test_runtime_rejects_premature_completion_and_resumes_recruitment() -> None:
    store = _seeded_store()
    trace = InMemoryTraceRecorder()
    client = _PrematureCompletionClient(store)
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_continuation",
            sender_id="zhang",
            sender_name="张哥",
            text="杭麻一块无烟371，人齐开，帮我组",
            message_id="msg_runtime_continuation",
        ),
        trace_id="trace_runtime_continuation",
    )

    calls = [call.name for action in result.actions for call in action.tool_calls]
    assert result.final_reply == "好，我帮你问问。"
    assert calls == ["create_game", "search_customers", "create_invite_drafts"]
    assert len(store.games) == 1
    game = next(iter(store.games.values()))
    assert game.seat_summary()["claimed_seats"] == 3
    assert game.seat_summary()["remaining_seats"] == 1
    assert len(store.invite_drafts) == 1
    assert "objective_continuation_rejected" in [event.step for event in trace.events]

    feedback_call = client.calls[2]
    serialized_context = "\n".join(item["content"] for item in feedback_call["messages"])
    assert "objective_continuation_contract" in serialized_context
    assert "discover_candidates" in serialized_context


def test_loop_preserves_prior_read_evidence_when_memory_write_interposes() -> None:
    store = _seeded_store()
    client = _ReadThenRememberClient()
    runtime = AgentRuntime(llm_client=client, store=store)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="tool_evidence_ledger",
            sender_id="zhang",
            sender_name="张哥",
            text="杭麻一块无烟371，人齐开，帮我组",
            message_id="msg_tool_evidence_ledger",
        ),
        trace_id="trace_tool_evidence_ledger",
    )

    assert result.final_reply == "好。"
    assert client.third_step_tool_names == ["search_current_games", "record_user_memory"]


class _ReadThenRememberClient:
    def __init__(self) -> None:
        self.call_count = 0
        self.third_step_tool_names: list[str] = []

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        self.call_count += 1
        if self.call_count == 1:
            return _action_json(
                objective_status="needs_tool",
                tool_calls=[
                    {
                        "name": "search_current_games",
                        "arguments": {"requirement": _requirement()},
                        "reason": "先查询现有局。",
                    }
                ],
            )
        if self.call_count == 2:
            return _action_json(
                objective_status="needs_tool",
                tool_calls=[
                    {
                        "name": "record_user_memory",
                        "arguments": {
                            "task_memories": [
                                {
                                    "customer_id": "zhang",
                                    "memory_type": "task_constraint",
                                    "field": "smoke_preference",
                                    "value": "no_smoke",
                                    "evidence": "用户明确说无烟",
                                    "confidence": 1.0,
                                }
                            ]
                        },
                        "reason": "记录本轮无烟约束。",
                    }
                ],
            )

        payload = json.loads(messages[1]["content"])
        self.third_step_tool_names = [
            str(item.get("name") or "") for item in payload["turn_tool_evidence"]
        ]
        assert [
            str(item.get("name") or "") for item in payload["previous_tool_results"]
        ] == ["record_user_memory"]
        return _action_json(
            objective_status="completed",
            reply_to_user="好。",
            reasoning_summary="已读到本轮此前的查询结果和记忆写入结果，不重复查询。",
        )


class _PrematureCompletionClient:
    def __init__(self, store: InMemoryAgentStore) -> None:
        self.store = store
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        self.calls.append({"messages": messages, "trace_id": trace_id, "timeout_seconds": timeout_seconds})
        call_number = len(self.calls)
        if call_number == 1:
            return _action_json(
                objective_status="needs_tool",
                tool_calls=[
                    {
                        "name": "create_game",
                        "arguments": {
                            "organizer_id": "zhang",
                            "organizer_name": "张哥",
                            "requirement": _requirement(),
                            "known_players": [
                                {
                                    "customer_id": "zhang",
                                    "display_name": "张哥",
                                    "seat_count": 3,
                                }
                            ],
                        },
                        "reason": "创建三缺一组局。",
                    }
                ],
            )
        if call_number == 2:
            return _action_json(
                objective_status="completed",
                reply_to_user="建好了。",
                reasoning_summary="错误地认为建局等于完成组局。",
            )
        if call_number == 3:
            return _action_json(
                objective_status="needs_tool",
                tool_calls=[
                    {
                        "name": "search_customers",
                        "arguments": {
                            "requirement": _requirement(),
                            "exclude_customer_ids": ["zhang"],
                            "limit": 1,
                        },
                        "reason": "根据续办契约继续搜索候选人。",
                    }
                ],
            )
        if call_number == 4:
            game_id = next(iter(self.store.games.values())).game_id
            return _action_json(
                objective_status="needs_tool",
                tool_calls=[
                    {
                        "name": "create_invite_drafts",
                        "arguments": {
                            "game_id": game_id,
                            "invitations": [
                                {
                                    "customer_id": "ran",
                                    "display_name": "冉姐",
                                    "message_text": "一块无烟，人齐开，打吗？",
                                }
                            ],
                        },
                        "reason": "为匹配候选人创建待审批邀约草稿。",
                    }
                ],
            )
        return _action_json(
            objective_status="completed",
            reply_to_user="好，我帮你问问。",
            reasoning_summary="邀约草稿已经持久化。",
        )


def _seeded_store() -> InMemoryAgentStore:
    store = InMemoryAgentStore()
    store.upsert_customer(
        CustomerProfile(
            customer_id="zhang",
            display_name="张哥",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["1"],
            smoke_preference="no_smoke",
        )
    )
    store.upsert_customer(
        CustomerProfile(
            customer_id="ran",
            display_name="冉姐",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["1"],
            smoke_preference="no_smoke",
            response_score=0.9,
        )
    )
    return store


def _requirement() -> dict[str, Any]:
    return {
        "game_type": "hangzhou_mahjong",
        "stake": "1",
        "smoke_preference": "no_smoke",
        "start_time_kind": "asap_when_full",
        "known_player_count": 3,
        "needed_seats": 1,
        "requesting_party": {
            "contact_id": "zhang",
            "contact_name": "张哥",
            "seat_count": 3,
            "known_member_ids": ["zhang"],
            "anonymous_seat_count": 2,
        },
    }


def _action_json(
    *,
    objective_status: str,
    tool_calls: list[dict[str, Any]] | None = None,
    reply_to_user: str = "",
    reasoning_summary: str = "测试续办契约。",
) -> str:
    calls = list(tool_calls or [])
    needs_tool = objective_status == "needs_tool"
    return json.dumps(
        {
            "goal": "完成三缺一组局",
            "objective_status": objective_status,
            "reasoning_summary": reasoning_summary,
            "objective_state": {
                "current_phase": "recruitment",
                "known_facts": {"needed_seats": 1},
                "missing_facts": [],
                "blockers": [],
            },
            "objective_plan": [
                {
                    "step_id": "recruit",
                    "title": "完成候选人邀约",
                    "status": "in_progress" if needs_tool else "done",
                    "tool": calls[0]["name"] if calls else None,
                    "depends_on": [],
                    "decision_rule": "未形成待审批邀约前不能结束。",
                }
            ],
            "plan_revision_reason": "根据工具结果推进。",
            "reply_to_user": reply_to_user,
            "tool_calls": calls,
            "needs_human": False,
            "stop_reason": {
                "can_stop": not needs_tool,
                "why": "需要继续执行。" if needs_tool else "模型认为可结束。",
                "pending_work": [call["name"] for call in calls],
                "depends_on_tool_results": True,
            },
            "badcase": None,
        },
        ensure_ascii=False,
    )
