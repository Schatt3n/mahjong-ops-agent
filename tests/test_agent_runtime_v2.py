from __future__ import annotations

import json
import threading
import time

from mahjong_agent_v2 import (
    AgentRuntimeResultV2,
    AgentRuntimeV2,
    ContextBuilderV2,
    ContextPackingPolicyV2,
    CustomerProfileV2,
    InMemoryAgentStoreV2,
    InMemoryEvalRecorderV2,
    JsonlEvalRecorderV2,
    SQLiteAgentStoreV2,
    ToolGatewayV2,
    UserMessageV2,
)
from mahjong_agent_v2.context import DEFAULT_V2_PROMPT_PATH
from mahjong_agent_v2.llm import StaticAgentClientV2
from mahjong_agent_v2.models import ConversationRoleV2, ConversationTurnV2, ToolResultV2
from mahjong_agent_v2.tracing import TraceEventV2, InMemoryTraceRecorderV2, validate_agent_runtime_trace_completeness


def test_v2_runtime_lets_model_choose_tool_order_and_reply_after_results() -> None:
    store = seeded_store()
    client = StaticAgentClientV2(
        outputs=[
            json.dumps(
                {
                    "goal": "查询是否有现成通宵局",
                    "reasoning_summary": "用户问有没有人，先查当前局。",
                    "reply_to_user": "",
                    "tool_calls": [
                        {
                            "name": "search_current_games",
                            "arguments": {"requirement": {"duration_kind": "overnight"}, "limit": 5},
                        }
                    ],
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "goal": "回复当前没有匹配局",
                    "reasoning_summary": "工具返回没有匹配局。",
                    "reply_to_user": "现在没有通宵局，要组一个吗？",
                    "tool_calls": [],
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
        ],
        calls=[],
    )
    trace = InMemoryTraceRecorderV2()
    runtime = AgentRuntimeV2(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(message("通宵有人吗"), trace_id="trace_v2_search")

    assert result.final_reply == "现在没有通宵局，要组一个吗？"
    assert [tool.name for tool in result.tool_results] == ["search_current_games"]
    assert len(client.calls) == 2
    assert '"previous_tool_results"' in client.calls[1]["messages"][1]["content"]
    steps = [event.step for event in trace.get_trace("trace_v2_search")]
    assert "context_packed" in steps
    assert "llm_prompt" in steps
    assert "llm_response" in steps
    assert "tool_called" in steps
    assert "tool_gateway_received" in steps
    assert "tool_idempotency_checked" in steps
    assert "tool_definition_checked" in steps
    assert "tool_schema_checked" in steps
    assert "tool_permission_checked" in steps
    assert "tool_gateway_completed" in steps
    assert "tool_result" in steps
    assert "final_output" in steps
    assert steps.index("tool_called") < steps.index("tool_gateway_received")
    assert steps.index("tool_gateway_received") < steps.index("tool_idempotency_checked")
    assert steps.index("tool_idempotency_checked") < steps.index("tool_definition_checked")
    assert steps.index("tool_definition_checked") < steps.index("tool_schema_checked")
    assert steps.index("tool_schema_checked") < steps.index("tool_permission_checked")
    assert steps.index("tool_permission_checked") < steps.index("tool_gateway_completed")
    assert steps.index("tool_gateway_completed") < steps.index("tool_result")
    report = validate_agent_runtime_trace_completeness(trace.get_trace("trace_v2_search"))
    assert report.complete is True


def test_v2_context_builder_packs_recent_conversation_with_budget_audit() -> None:
    store = seeded_store()
    gateway = ToolGatewayV2(store=store)
    for index in range(10):
        store.append_turn(
            "budget_v2",
            ConversationTurnV2(
                role=ConversationRoleV2.USER,
                content=f"第{index}轮 " + ("很长的上下文 " * 30),
                trace_id=f"trace_old_{index}",
            ),
        )
    builder = ContextBuilderV2(
        store=store,
        tool_gateway=gateway,
        packing_policy=ContextPackingPolicyV2(
            max_turns_considered=10,
            max_recent_conversation_tokens=120,
        ),
    )

    built = builder.build(
        UserMessageV2(
            conversation_id="budget_v2",
            sender_id="zhang",
            sender_name="张哥",
            text="继续",
        ),
        trace_id="trace_v2_context_budget",
    )

    budget = built.payload["context_budget"]
    assert budget["turns_considered"] == 10
    assert budget["included_turn_count"] < 10
    assert budget["omitted_for_budget"] > 0
    assert budget["estimated_recent_conversation_tokens"] <= 120
    assert built.payload["recent_conversation"][-1]["content"].startswith("第9轮")
    assert "context_budget" in built.messages[1]["content"]


def test_v2_runtime_interrupts_and_audits_llm_failure_without_tools() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorderV2()
    runtime = AgentRuntimeV2(
        llm_client=FailingAgentClientV2(RuntimeError("deepseek timeout")),
        store=store,
        trace_recorder=trace,
        llm_timeout_seconds=1.5,
    )
    incoming = message("通宵有人吗")
    incoming.message_id = "llm-failure-message"

    result = runtime.handle_user_message(incoming, trace_id="trace_v2_llm_failure")

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.decisions == []
    assert result.tool_results == []
    assert result.state_transitions == []
    assert not store.games
    assert store.idempotent_message_result("llm-failure-message") is result
    events = trace.get_trace("trace_v2_llm_failure")
    assert any(event.step == "llm_error" and event.level == "ERROR" for event in events)
    assert any(event.step == "final_output" and event.content["reason"] == "llm_error" for event in events)
    assert store.recent_turns(incoming.conversation_id, limit=1)[0].content == "这个我先转人工确认一下。"


def test_v2_runtime_rejects_malformed_llm_decision_contract_without_tools() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorderV2()
    client = StaticAgentClientV2(outputs=["这不是 JSON"], calls=[])
    runtime = AgentRuntimeV2(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(message("通宵有人吗"), trace_id="trace_v2_bad_json_contract")

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.tool_results == []
    assert result.state_transitions == []
    assert result.decisions[0].goal == "decision_contract_invalid"
    events = trace.get_trace("trace_v2_bad_json_contract")
    assert any(event.step == "decision_contract_error" and event.level == "WARN" for event in events)
    assert any(
        "response is not valid JSON" in "\n".join(event.content["errors"])
        for event in events
        if event.step == "decision_contract_error"
    )


def test_v2_runtime_rejects_missing_decision_fields_without_using_reply_text() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorderV2()
    client = StaticAgentClientV2(
        outputs=[
            json.dumps(
                {
                    "goal": "缺少合同字段",
                    "reasoning_summary": "故意缺少 tool_calls 和 needs_human。",
                    "reply_to_user": "这个回复不能被直接采用。",
                },
                ensure_ascii=False,
            )
        ],
        calls=[],
    )
    runtime = AgentRuntimeV2(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(message("组"), trace_id="trace_v2_missing_contract_fields")

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.tool_results == []
    errors = [
        error
        for event in trace.get_trace("trace_v2_missing_contract_fields")
        if event.step == "decision_contract_error"
        for error in event.content["errors"]
    ]
    assert "tool_calls is required" in errors
    assert "needs_human is required" in errors


def test_v2_runtime_rejects_invalid_tool_call_shape_before_tool_gateway() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorderV2()
    client = StaticAgentClientV2(
        outputs=[
            json.dumps(
                {
                    "goal": "坏工具调用",
                    "reasoning_summary": "tool arguments 类型错误，不能进入工具网关。",
                    "reply_to_user": "",
                    "tool_calls": [
                        {
                            "name": "create_game",
                            "arguments": [],
                            "reason": "故意传错类型",
                        }
                    ],
                    "needs_human": False,
                },
                ensure_ascii=False,
            )
        ],
        calls=[],
    )
    runtime = AgentRuntimeV2(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(message("组"), trace_id="trace_v2_bad_tool_call_contract")

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.tool_results == []
    assert not store.games
    steps = [event.step for event in trace.get_trace("trace_v2_bad_tool_call_contract")]
    assert "decision_contract_error" in steps
    assert "tool_called" not in steps


def test_v2_trace_completeness_reports_missing_tool_result_pair() -> None:
    events = [
        TraceEventV2("trace_v2_incomplete", "user_input", {}),
        TraceEventV2("trace_v2_incomplete", "context_packed", {}),
        TraceEventV2("trace_v2_incomplete", "context_built", {}),
        TraceEventV2("trace_v2_incomplete", "llm_prompt", {}),
        TraceEventV2("trace_v2_incomplete", "budget_checked", {}),
        TraceEventV2("trace_v2_incomplete", "llm_response", {}),
        TraceEventV2("trace_v2_incomplete", "action_proposed", {}),
        TraceEventV2("trace_v2_incomplete", "tool_called", {}),
        TraceEventV2("trace_v2_incomplete", "final_output", {}),
    ]

    report = validate_agent_runtime_trace_completeness(events)

    assert report.complete is False
    assert "tool_gateway_received" in report.missing_steps
    assert "tool_idempotency_checked" in report.missing_steps
    assert "tool_gateway_completed" in report.missing_steps
    assert "tool_called count 1 != tool_result count 0" in report.pairing_errors


def test_v2_trace_completeness_reports_bad_event_order() -> None:
    events = [
        TraceEventV2("trace_v2_bad_order", "user_input", {}),
        TraceEventV2("trace_v2_bad_order", "llm_prompt", {}),
        TraceEventV2("trace_v2_bad_order", "context_packed", {}),
        TraceEventV2("trace_v2_bad_order", "context_built", {}),
        TraceEventV2("trace_v2_bad_order", "budget_checked", {}),
        TraceEventV2("trace_v2_bad_order", "final_output", {}),
    ]

    report = validate_agent_runtime_trace_completeness(events)

    assert report.complete is False
    assert "context_built must occur before llm_prompt" in report.ordering_errors


def test_v2_trace_completeness_pairs_each_llm_call_independently() -> None:
    events = [
        TraceEventV2("trace_v2_model_pair", "user_input", {}),
        TraceEventV2("trace_v2_model_pair", "context_packed", {}),
        TraceEventV2("trace_v2_model_pair", "context_built", {}),
        TraceEventV2("trace_v2_model_pair", "llm_prompt", {}),
        TraceEventV2("trace_v2_model_pair", "budget_checked", {"allowed": True}),
        TraceEventV2("trace_v2_model_pair", "llm_response", {}),
        TraceEventV2("trace_v2_model_pair", "action_proposed", {}),
        TraceEventV2("trace_v2_model_pair", "tool_called", {}),
        TraceEventV2("trace_v2_model_pair", "tool_gateway_received", {}),
        TraceEventV2("trace_v2_model_pair", "tool_idempotency_checked", {}),
        TraceEventV2("trace_v2_model_pair", "tool_gateway_completed", {}),
        TraceEventV2("trace_v2_model_pair", "tool_result", {}),
        TraceEventV2("trace_v2_model_pair", "context_packed", {}),
        TraceEventV2("trace_v2_model_pair", "context_built", {}),
        TraceEventV2("trace_v2_model_pair", "llm_prompt", {}),
        TraceEventV2("trace_v2_model_pair", "budget_checked", {"allowed": True}),
        TraceEventV2("trace_v2_model_pair", "final_output", {}),
    ]

    report = validate_agent_runtime_trace_completeness(events)

    assert report.complete is False
    assert "llm call 2 has 0 llm_response and 0 llm_error events" in report.pairing_errors


def test_v2_trace_completeness_pairs_each_tool_gateway_call_independently() -> None:
    events = [
        TraceEventV2("trace_v2_tool_pair", "user_input", {}),
        TraceEventV2("trace_v2_tool_pair", "context_packed", {}),
        TraceEventV2("trace_v2_tool_pair", "context_built", {}),
        TraceEventV2("trace_v2_tool_pair", "llm_prompt", {}),
        TraceEventV2("trace_v2_tool_pair", "budget_checked", {"allowed": True}),
        TraceEventV2("trace_v2_tool_pair", "llm_response", {}),
        TraceEventV2("trace_v2_tool_pair", "action_proposed", {}),
        TraceEventV2("trace_v2_tool_pair", "tool_called", {}),
        TraceEventV2("trace_v2_tool_pair", "tool_gateway_received", {}),
        TraceEventV2("trace_v2_tool_pair", "tool_idempotency_checked", {}),
        TraceEventV2("trace_v2_tool_pair", "tool_gateway_completed", {}),
        TraceEventV2("trace_v2_tool_pair", "tool_result", {}),
        TraceEventV2("trace_v2_tool_pair", "tool_called", {}),
        TraceEventV2("trace_v2_tool_pair", "tool_gateway_received", {}),
        TraceEventV2("trace_v2_tool_pair", "tool_idempotency_checked", {}),
        TraceEventV2("trace_v2_tool_pair", "tool_result", {}),
        TraceEventV2("trace_v2_tool_pair", "final_output", {}),
    ]

    report = validate_agent_runtime_trace_completeness(events)

    assert report.complete is False
    assert "tool_called count 2 != tool_gateway_completed count 1" in report.pairing_errors


def test_v2_trace_completeness_requires_reply_review_model_events() -> None:
    events = [
        TraceEventV2("trace_v2_review_incomplete", "user_input", {}),
        TraceEventV2("trace_v2_review_incomplete", "context_packed", {}),
        TraceEventV2("trace_v2_review_incomplete", "context_built", {}),
        TraceEventV2("trace_v2_review_incomplete", "llm_prompt", {}),
        TraceEventV2("trace_v2_review_incomplete", "budget_checked", {}),
        TraceEventV2("trace_v2_review_incomplete", "llm_response", {}),
        TraceEventV2("trace_v2_review_incomplete", "action_proposed", {}),
        TraceEventV2("trace_v2_review_incomplete", "reply_review_prompt", {}),
        TraceEventV2("trace_v2_review_incomplete", "reply_review_budget_checked", {}),
        TraceEventV2("trace_v2_review_incomplete", "final_output", {}),
    ]

    report = validate_agent_runtime_trace_completeness(events)

    assert report.complete is False
    assert "reply_review_response" in report.missing_steps
    assert "reply_review_proposed" in report.missing_steps


def test_v2_trace_completeness_pairs_each_reply_review_call_independently() -> None:
    events = [
        TraceEventV2("trace_v2_review_pair", "user_input", {}),
        TraceEventV2("trace_v2_review_pair", "context_packed", {}),
        TraceEventV2("trace_v2_review_pair", "context_built", {}),
        TraceEventV2("trace_v2_review_pair", "llm_prompt", {}),
        TraceEventV2("trace_v2_review_pair", "budget_checked", {"allowed": True}),
        TraceEventV2("trace_v2_review_pair", "llm_response", {}),
        TraceEventV2("trace_v2_review_pair", "action_proposed", {}),
        TraceEventV2("trace_v2_review_pair", "reply_review_prompt", {}),
        TraceEventV2("trace_v2_review_pair", "reply_review_budget_checked", {"allowed": True}),
        TraceEventV2("trace_v2_review_pair", "reply_review_response", {}),
        TraceEventV2("trace_v2_review_pair", "reply_review_proposed", {}),
        TraceEventV2("trace_v2_review_pair", "reply_review_prompt", {}),
        TraceEventV2("trace_v2_review_pair", "reply_review_budget_checked", {"allowed": True}),
        TraceEventV2("trace_v2_review_pair", "final_output", {}),
    ]

    report = validate_agent_runtime_trace_completeness(events)

    assert report.complete is False
    assert "reply review call 2 has 0 response and 0 error events" in report.pairing_errors


def test_v2_trace_completeness_reports_bad_reply_review_order() -> None:
    events = [
        TraceEventV2("trace_v2_review_order", "user_input", {}),
        TraceEventV2("trace_v2_review_order", "context_packed", {}),
        TraceEventV2("trace_v2_review_order", "context_built", {}),
        TraceEventV2("trace_v2_review_order", "llm_prompt", {}),
        TraceEventV2("trace_v2_review_order", "budget_checked", {}),
        TraceEventV2("trace_v2_review_order", "llm_response", {}),
        TraceEventV2("trace_v2_review_order", "action_proposed", {}),
        TraceEventV2("trace_v2_review_order", "reply_review_prompt", {}),
        TraceEventV2("trace_v2_review_order", "reply_review_response", {}),
        TraceEventV2("trace_v2_review_order", "reply_review_budget_checked", {}),
        TraceEventV2("trace_v2_review_order", "reply_review_proposed", {}),
        TraceEventV2("trace_v2_review_order", "final_output", {}),
    ]

    report = validate_agent_runtime_trace_completeness(events)

    assert report.complete is False
    assert "reply_review_budget_checked must occur before reply_review_response" in report.ordering_errors


class FailingAgentClientV2:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = []

    def complete(self, messages, *, trace_id, timeout_seconds):
        self.calls.append({"messages": messages, "trace_id": trace_id, "timeout_seconds": timeout_seconds})
        raise self.exc


def test_v2_runtime_creates_game_searches_customers_and_creates_llm_written_drafts() -> None:
    store = seeded_store()
    client = DynamicDraftClient()
    client.outputs.append(
        json.dumps(
            {
                "goal": "帮张哥组通宵1块局",
                "reasoning_summary": "用户已经确认要组局，先建局再找候选人。",
                "reply_to_user": "",
                "tool_calls": [
                    {
                        "name": "create_game",
                        "arguments": {
                            "requirement": {
                                "game_type": "hangzhou_mahjong",
                                "stake": "1",
                                "duration_kind": "overnight",
                                "start_time_kind": "asap_when_full",
                                "start_time_text": "人齐开",
                                "smoke_preference": "any",
                                "smoke_label": "烟都可",
                            },
                            "known_players": [
                                {"customer_id": "zhang", "display_name": "张哥", "source": "organizer"}
                            ],
                        },
                    },
                    {
                        "name": "search_customers",
                        "arguments": {
                            "requirement": {
                                "game_type": "hangzhou_mahjong",
                                "stake": "1",
                                "duration_kind": "overnight",
                                "start_time_kind": "asap_when_full",
                                "smoke_preference": "any",
                            },
                            "exclude_customer_ids": ["zhang"],
                            "limit": 2,
                        },
                    },
                ],
                "needs_human": False,
            },
            ensure_ascii=False,
        )
    )
    client.outputs.append(
        json.dumps(
            {
                "goal": "回复发起人",
                "reasoning_summary": "已经创建待审批邀约草稿。",
                "reply_to_user": "好的，我先帮你问问。",
                "tool_calls": [],
                "needs_human": False,
            },
            ensure_ascii=False,
        )
    )
    trace = InMemoryTraceRecorderV2()
    runtime = AgentRuntimeV2(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(message("一个人，1块的，通宵，人齐开，烟都可"), trace_id="trace_v2_form")

    assert result.final_reply == "好的，我先帮你问问。"
    assert [tool.name for tool in result.tool_results] == [
        "create_game",
        "search_customers",
        "create_invite_drafts",
    ]
    assert len(store.games) == 1
    assert len(store.invite_drafts) == 1
    draft = next(iter(store.invite_drafts.values()))
    assert draft.message_text == "冉姐，人齐开，1块通宵，打吗？"
    assert result.state_transitions
    assert any(event.step == "state_transition" for event in trace.get_trace("trace_v2_form"))


def test_v2_decision_review_can_keep_agent_from_stopping_before_required_tools() -> None:
    store = seeded_store()
    client = DecisionReviewRecoveryClient()
    trace = InMemoryTraceRecorderV2()
    runtime = AgentRuntimeV2(
        llm_client=client,
        store=store,
        trace_recorder=trace,
        decision_review_enabled=True,
    )

    result = runtime.handle_user_message(message("一个人，1块的，通宵，人齐开，烟都可"), trace_id="trace_v2_decision_review")

    assert result.final_reply == "好的，我帮你问问，有消息跟你说。"
    assert [tool.name for tool in result.tool_results] == [
        "record_badcase",
        "create_game",
        "search_customers",
        "create_invite_drafts",
    ]
    assert len(store.games) == 1
    assert len(store.invite_drafts) == 1
    steps = [event.step for event in trace.get_trace("trace_v2_decision_review")]
    assert "decision_review_prompt" in steps
    assert "decision_review_response" in steps
    assert "decision_review_proposed" in steps
    assert "decision_revised" in steps
    assert steps.index("decision_revised") < steps.index("context_packed", steps.index("decision_revised"))
    report = validate_agent_runtime_trace_completeness(trace.get_trace("trace_v2_decision_review"))
    assert report.complete is True


class DecisionReviewRecoveryClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, messages, *, trace_id, timeout_seconds):
        self.calls.append({"messages": messages, "trace_id": trace_id, "timeout_seconds": timeout_seconds})
        system_prompt = messages[0]["content"]
        payload = json.loads(messages[1]["content"])
        if "动作审查模型" in system_prompt:
            latest = payload["latest_decision"]
            if latest.get("reply_to_user") == "好的，我先帮你留意下。":
                return json.dumps(
                    {
                        "approved": False,
                        "reasoning_summary": "用户目标是组局，原动作直接停住，没有调用建局和候选人搜索工具。",
                        "revised_decision": {
                            "goal": "帮张哥组通宵1块局",
                            "reasoning_summary": "信息足够，应先建局并搜索候选人。",
                            "reply_to_user": "",
                            "tool_calls": [
                                {
                                    "name": "create_game",
                                    "arguments": {
                                        "requirement": {
                                            "game_type": "hangzhou_mahjong",
                                            "game_type_label": "杭麻",
                                            "stake": "1",
                                            "start_time_kind": "asap_when_full",
                                            "start_time_text": "人齐开",
                                            "duration_text": "通宵",
                                            "smoke_preference": "any",
                                            "smoke_label": "烟都可",
                                            "current_players": 1,
                                            "missing_players": 3,
                                            "seats_total": 4,
                                            "user_visible_summary": "杭麻 1档 人齐开 烟都可 通宵 缺3",
                                        },
                                        "known_players": [
                                            {"customer_id": "zhang", "display_name": "张哥", "source": "organizer"}
                                        ],
                                    },
                                    "reason": "用户明确要组局，先创建待组局状态",
                                },
                                {
                                    "name": "search_customers",
                                    "arguments": {
                                        "requirement": {
                                            "game_type": "hangzhou_mahjong",
                                            "stake": "1",
                                            "start_time_kind": "asap_when_full",
                                            "duration_text": "通宵",
                                            "smoke_preference": "any",
                                            "missing_players": 3,
                                            "user_visible_summary": "杭麻 1档 人齐开 烟都可 通宵 缺3",
                                        },
                                        "exclude_customer_ids": ["zhang"],
                                        "limit": 2,
                                    },
                                    "reason": "为新局搜索候选人",
                                },
                            ],
                            "needs_human": False,
                            "badcase": None,
                        },
                        "badcase": {
                            "reason": "模型准备回复留意但没有继续调用组局工具",
                            "input": {"text": "一个人，1块的，通宵，人齐开，烟都可"},
                            "actual": {"reply": "好的，我先帮你留意下。"},
                            "expected": {"tool_plan": "create_game -> search_customers"},
                            "tags": ["agent_runtime_v2", "decision_review"],
                            "source": "decision_review",
                        },
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "approved": True,
                    "reasoning_summary": "动作已足够，可以回复用户。",
                    "revised_decision": None,
                    "badcase": None,
                },
                ensure_ascii=False,
            )
        previous = payload.get("previous_tool_results") or []
        if any(item.get("name") == "search_customers" for item in previous):
            game = next(item for item in previous if item.get("name") == "create_game")["result"]["game"]
            candidate = next(item for item in previous if item.get("name") == "search_customers")["result"]["candidates"][0]
            customer = candidate["customer"]
            return json.dumps(
                {
                    "goal": "生成候选人待审批邀约",
                    "reasoning_summary": "候选人已经返回，继续生成待审批草稿。",
                    "reply_to_user": "",
                    "tool_calls": [
                        {
                            "name": "create_invite_drafts",
                            "arguments": {
                                "game_id": game["game_id"],
                                "invitations": [
                                    {
                                        "customer_id": customer["customer_id"],
                                        "display_name": customer["display_name"],
                                        "message_text": f"{customer['display_name']}，人齐开，1块通宵，打吗？",
                                    }
                                ],
                            },
                            "reason": "只创建待审批草稿，不直接发送",
                        }
                    ],
                    "needs_human": False,
                },
                ensure_ascii=False,
            )
        if any(item.get("name") == "create_invite_drafts" for item in previous):
            return json.dumps(
                {
                    "goal": "回复发起人",
                    "reasoning_summary": "已经创建待审批邀约草稿。",
                    "reply_to_user": "好的，我帮你问问，有消息跟你说。",
                    "tool_calls": [],
                    "needs_human": False,
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "goal": "回复发起人",
                "reasoning_summary": "这里模拟主模型过早停止。",
                "reply_to_user": "好的，我先帮你留意下。",
                "tool_calls": [],
                "needs_human": False,
            },
            ensure_ascii=False,
        )


class DynamicDraftClient:
    def __init__(self) -> None:
        self.outputs: list[str] = []
        self.calls: list[dict] = []

    def complete(self, messages, *, trace_id, timeout_seconds):
        self.calls.append({"messages": messages, "trace_id": trace_id, "timeout_seconds": timeout_seconds})
        if len(self.calls) == 2:
            payload = json.loads(messages[1]["content"])
            game_id = payload["previous_tool_results"][0]["result"]["game"]["game_id"]
            return json.dumps(
                {
                    "goal": "创建候选人邀约草稿",
                    "reasoning_summary": "根据候选人结果生成待审批邀约。",
                    "reply_to_user": "",
                    "tool_calls": [
                        {
                            "name": "create_invite_drafts",
                            "arguments": {
                                "game_id": game_id,
                                "invitations": [
                                    {
                                        "customer_id": "ran",
                                        "display_name": "冉姐",
                                        "message_text": "冉姐，人齐开，1块通宵，打吗？",
                                    }
                                ],
                            },
                        }
                    ],
                    "needs_human": False,
                },
                ensure_ascii=False,
            )
        if not self.outputs:
            raise AssertionError("no fake output")
        return self.outputs.pop(0)


def test_v2_gateway_rejects_invalid_tool_arguments_and_runtime_returns_result_to_model() -> None:
    store = seeded_store()
    client = StaticAgentClientV2(
        outputs=[
            json.dumps(
                {
                    "goal": "尝试建局",
                    "reasoning_summary": "模型误传了非法参数。",
                    "reply_to_user": "",
                    "tool_calls": [{"name": "create_game", "arguments": {"known_players": []}}],
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "goal": "修正并追问",
                    "reasoning_summary": "工具返回 schema 错误，不能建局。",
                    "reply_to_user": "我先确认一下，你想打多大的？",
                    "tool_calls": [],
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
        ],
        calls=[],
    )
    trace = InMemoryTraceRecorderV2()
    runtime = AgentRuntimeV2(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(message("组"), trace_id="trace_v2_invalid_args")

    assert result.final_reply == "我先确认一下，你想打多大的？"
    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert "requirement is required" in str(result.tool_results[0].error)
    second_prompt = client.calls[1]["messages"][1]["content"]
    assert "previous_tool_results" in second_prompt
    assert "requirement is required" in second_prompt
    events = trace.get_trace("trace_v2_invalid_args")
    schema_events = [event for event in events if event.step == "tool_schema_checked"]
    assert schema_events[0].level == "WARN"
    assert schema_events[0].content["allowed"] is False
    assert "requirement is required" in schema_events[0].content["error"]
    gateway_completed = [event for event in events if event.step == "tool_gateway_completed"]
    assert gateway_completed[0].content["outcome"] == "blocked"
    assert validate_agent_runtime_trace_completeness(events).complete is True


def test_v2_gateway_rejects_human_labels_in_internal_requirement_enums() -> None:
    store = seeded_store()
    client = StaticAgentClientV2(
        outputs=[
            json.dumps(
                {
                    "goal": "错误地把客户可见中文放进内部枚举",
                    "reasoning_summary": "模型应该输出 canonical 字段，这里故意传错。",
                    "reply_to_user": "",
                    "tool_calls": [
                        {
                            "name": "search_current_games",
                            "arguments": {
                                "requirement": {
                                    "start_time_kind": "人齐开",
                                    "duration_kind": "通宵",
                                    "smoke_preference": "烟都可",
                                },
                                "limit": 5,
                            },
                        }
                    ],
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "goal": "改用结构化字段查询",
                    "reasoning_summary": "工具返回 enum schema 错误，改成 canonical 内部枚举后重试。",
                    "reply_to_user": "",
                    "tool_calls": [
                        {
                            "name": "search_current_games",
                            "arguments": {
                                "requirement": {
                                    "start_time_kind": "asap_when_full",
                                    "duration_kind": "overnight",
                                    "smoke_preference": "any",
                                    "user_visible_summary": "人齐开 通宵 烟都可",
                                },
                                "limit": 5,
                            },
                        }
                    ],
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "goal": "回复查询结果",
                    "reasoning_summary": "结构化查询完成，没有匹配局。",
                    "reply_to_user": "现在没有现成的，要不要我帮你组一个？",
                    "tool_calls": [],
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
        ],
        calls=[],
    )
    runtime = AgentRuntimeV2(llm_client=client, store=store)

    result = runtime.handle_user_message(message("人齐开通宵烟都可有人吗"), trace_id="trace_v2_enum_contract")

    assert result.final_reply == "现在没有现成的，要不要我帮你组一个？"
    assert [tool.name for tool in result.tool_results] == ["search_current_games", "search_current_games"]
    assert result.tool_results[0].called is False
    assert "start_time_kind" in str(result.tool_results[0].error)
    assert result.tool_results[1].called is True
    assert "must be one of" in client.calls[1]["messages"][1]["content"]


def test_v2_tool_gateway_audits_idempotency_hit_inside_trace() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorderV2()
    client = StaticAgentClientV2(
        outputs=[
            json.dumps(
                {
                    "goal": "重复查询当前局测试幂等",
                    "reasoning_summary": "模型重复发起同一个只读工具调用，后端应该幂等去重。",
                    "reply_to_user": "",
                    "tool_calls": [
                        {
                            "name": "search_current_games",
                            "arguments": {"requirement": {"duration_text": "通宵"}, "limit": 5},
                            "idempotency_key": "same-current-game-search",
                        },
                        {
                            "name": "search_current_games",
                            "arguments": {"requirement": {"duration_text": "通宵"}, "limit": 5},
                            "idempotency_key": "same-current-game-search",
                        },
                    ],
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "goal": "回复查询结果",
                    "reasoning_summary": "第二次工具调用命中幂等结果，没有重复执行。",
                    "reply_to_user": "现在没有通宵局，要组一个吗？",
                    "tool_calls": [],
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
        ],
        calls=[],
    )
    runtime = AgentRuntimeV2(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(message("通宵有人吗"), trace_id="trace_v2_tool_idempotency")

    assert [tool.name for tool in result.tool_results] == ["search_current_games", "search_current_games"]
    assert result.tool_results[0].deduplicated is False
    assert result.tool_results[1].deduplicated is True
    events = trace.get_trace("trace_v2_tool_idempotency")
    idempotency_events = [event for event in events if event.step == "tool_idempotency_checked"]
    assert [event.content["hit"] for event in idempotency_events] == [False, True]
    completed = [event for event in events if event.step == "tool_gateway_completed"]
    assert [event.content["outcome"] for event in completed] == ["executed", "deduplicated"]
    assert validate_agent_runtime_trace_completeness(events).complete is True


def test_v2_current_game_search_uses_model_structured_requirement_fields() -> None:
    store = seeded_store()
    store.create_game(
        conversation_id="boss_v2",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "start_time_kind": "asap_when_full",
            "smoke_preference": "non_smoking",
            "duration_kind": "overnight",
            "duration_text": "通宵",
            "user_visible_summary": "杭麻 0.5档 人齐开 无烟 通宵 缺3",
        },
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="trace_v2_alias_search",
    )

    matches = store.search_current_games(
        {
            "game_type": "hangzhou_mahjong",
            "stake_options": ["0.5", "1"],
            "start_time_kind": "asap_when_full",
            "smoke_preference": "non_smoking",
            "duration_kind": "overnight",
        }
    )

    assert len(matches) == 1
    assert matches[0]["game"]["requirement"]["stake"] == "0.5"
    assert "smoke_preference_matched" in matches[0]["reasons"]
    assert "start_time_kind_matched" in matches[0]["reasons"]
    assert "duration_kind_matched" in matches[0]["reasons"]

    chinese_label_query = store.search_current_games(
        {
            "game_type": "hangzhou_mahjong",
            "stake_options": ["0.5", "1"],
            "start_time_kind": "人齐开",
            "smoke_preference": "无烟",
            "duration_text": "通宵",
        }
    )
    assert chinese_label_query == []


def test_v2_customer_search_uses_model_structured_smoke_preference() -> None:
    store = seeded_store()

    candidates = store.search_customers(
        {
            "game_type": "hangzhou_mahjong",
            "stake": "1",
            "smoke_preference": "any",
        },
        exclude_customer_ids=["zhang"],
    )

    assert candidates
    assert candidates[0]["customer"]["customer_id"] == "ran"
    assert "smoke_compatible" in candidates[0]["reasons"]

    chinese_label_candidates = store.search_customers(
        {
            "game_type": "hangzhou_mahjong",
            "stake": "1",
            "smoke_preference": "烟都可",
        },
        exclude_customer_ids=["zhang"],
    )
    assert chinese_label_candidates
    assert "smoke_compatible" not in chinese_label_candidates[0]["reasons"]


def test_v2_gateway_rejects_internal_codes_in_customer_visible_invites_and_model_retries() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="boss_v2",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={
            "game_type": "hangzhou_mahjong",
            "game_type_label": "杭麻",
            "stake": "1",
            "start_time_kind": "asap_when_full",
            "user_visible_summary": "杭麻 1档 人齐开 烟都可 通宵 缺3",
        },
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="trace_v2_public_text",
    )
    client = StaticAgentClientV2(
        outputs=[
            json.dumps(
                {
                    "goal": "创建候选人邀约草稿",
                    "reasoning_summary": "第一次误把内部字段写进客户可见文案。",
                    "reply_to_user": "",
                    "tool_calls": [
                        {
                            "name": "create_invite_drafts",
                            "arguments": {
                                "game_id": game.game_id,
                                "invitations": [
                                    {
                                        "customer_id": "ran",
                                        "display_name": "冉姐",
                                        "message_text": "冉姐，asap_when_full，1块通宵，打吗？",
                                    }
                                ],
                            },
                        }
                    ],
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "goal": "修正邀约草稿",
                    "reasoning_summary": "工具返回客户可见文案不能包含内部 snake_case，改用自然中文。",
                    "reply_to_user": "",
                    "tool_calls": [
                        {
                            "name": "create_invite_drafts",
                            "arguments": {
                                "game_id": game.game_id,
                                "invitations": [
                                    {
                                        "customer_id": "ran",
                                        "display_name": "冉姐",
                                        "message_text": "冉姐，人齐开，1块通宵，打吗？",
                                    }
                                ],
                            },
                        }
                    ],
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "goal": "回复发起人",
                    "reasoning_summary": "待审批邀约草稿已经创建。",
                    "reply_to_user": "好的，我先帮你问问。",
                    "tool_calls": [],
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
        ],
        calls=[],
    )
    runtime = AgentRuntimeV2(llm_client=client, store=store)

    result = runtime.handle_user_message(message("帮我问问"), trace_id="trace_v2_public_text")

    assert result.final_reply == "好的，我先帮你问问。"
    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert "cannot contain internal snake_case codes" in str(result.tool_results[0].error)
    assert result.tool_results[1].called is True
    assert next(iter(store.invite_drafts.values())).message_text == "冉姐，人齐开，1块通宵，打吗？"
    assert len(client.calls) == 3
    assert "cannot contain internal snake_case codes" in client.calls[1]["messages"][1]["content"]


def test_v2_reply_review_revises_final_reply_and_records_badcase() -> None:
    store = seeded_store()
    eval_recorder = InMemoryEvalRecorderV2()
    gateway = ToolGatewayV2(store=store, eval_recorder=eval_recorder)
    client = StaticAgentClientV2(
        outputs=[
            json.dumps(
                {
                    "goal": "回复发起人",
                    "reasoning_summary": "工具已经生成待审批草稿。",
                    "reply_to_user": "张哥，局已建好，已找到冉姐和何哥，草稿等你审批。",
                    "tool_calls": [],
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "approved": False,
                    "reasoning_summary": "客户可见回复暴露候选人和草稿审批状态，应该改成自然确认。",
                    "revised_reply": "好的，我帮你问问，有消息跟你说。",
                    "badcase": {
                        "reason": "客户可见回复暴露候选人和草稿审批状态",
                        "input": {"text": "一个人，1块的，通宵，人齐开"},
                        "actual": {"reply": "张哥，局已建好，已找到冉姐和何哥，草稿等你审批。"},
                        "expected": {"reply": "好的，我帮你问问，有消息跟你说。"},
                        "tags": ["agent_runtime_v2", "reply_visibility"],
                        "source": "reply_review",
                    },
                },
                ensure_ascii=False,
            ),
        ],
        calls=[],
    )
    trace = InMemoryTraceRecorderV2()
    runtime = AgentRuntimeV2(
        llm_client=client,
        store=store,
        tool_gateway=gateway,
        trace_recorder=trace,
        reply_review_enabled=True,
    )

    result = runtime.handle_user_message(message("一个人，1块的，通宵，人齐开"), trace_id="trace_v2_reply_review")

    assert result.final_reply == "好的，我帮你问问，有消息跟你说。"
    assert [item.name for item in result.tool_results] == ["record_badcase"]
    assert eval_recorder.records[0]["reason"] == "客户可见回复暴露候选人和草稿审批状态"
    steps = [event.step for event in trace.get_trace("trace_v2_reply_review")]
    assert "reply_review_prompt" in steps
    assert "reply_review_response" in steps
    assert "reply_review_proposed" in steps
    assert "reply_revised" in steps
    assert steps[-1] == "final_output"
    report = validate_agent_runtime_trace_completeness(trace.get_trace("trace_v2_reply_review"))
    assert report.complete is True


def test_v2_gateway_enforces_tool_execution_mode_permissions() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="boss_v2",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"user_visible_summary": "杭麻 1档 人齐开"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="trace_v2_permission",
    )
    gateway = ToolGatewayV2(
        store=store,
        allowed_execution_modes={"read_only", "state_write", "audit_write"},
    )
    client = StaticAgentClientV2(
        outputs=[
            json.dumps(
                {
                    "goal": "尝试创建邀约草稿",
                    "reasoning_summary": "模型提出高风险待审批草稿动作。",
                    "reply_to_user": "",
                    "tool_calls": [
                        {
                            "name": "create_invite_drafts",
                            "arguments": {
                                "game_id": game.game_id,
                                "invitations": [
                                    {"customer_id": "ran", "message_text": "冉姐，人齐开，1块，打吗？"}
                                ],
                            },
                        }
                    ],
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "goal": "转人工确认权限",
                    "reasoning_summary": "工具返回 create_pending 不允许，本轮不能创建邀约草稿。",
                    "reply_to_user": "这个邀约动作我先转人工确认一下。",
                    "tool_calls": [],
                    "needs_human": True,
                },
                ensure_ascii=False,
            ),
        ],
        calls=[],
    )
    trace = InMemoryTraceRecorderV2()
    runtime = AgentRuntimeV2(llm_client=client, store=store, tool_gateway=gateway, trace_recorder=trace)

    result = runtime.handle_user_message(message("帮我问一下冉姐"), trace_id="trace_v2_permission")

    assert result.final_reply == "这个邀约动作我先转人工确认一下。"
    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert result.tool_results[0].error == "tool execution_mode not allowed: create_pending"
    assert not store.invite_drafts
    assert "tool execution_mode not allowed" in client.calls[1]["messages"][1]["content"]
    permission_events = [
        event for event in trace.get_trace("trace_v2_permission") if event.step == "tool_permission_checked"
    ]
    assert permission_events[0].level == "WARN"
    assert permission_events[0].content["allowed"] is False
    assert permission_events[0].content["execution_mode"] == "create_pending"
    assert validate_agent_runtime_trace_completeness(trace.get_trace("trace_v2_permission")).complete is True


def test_v2_state_policy_rejects_candidate_reply_without_existing_invite_draft() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="boss_v2",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"user_visible_summary": "杭麻 1档 人齐开"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="trace_v2_state_no_draft",
    )
    store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[{"customer_id": "he", "message_text": "何哥，人齐开，1块，打吗？"}],
        trace_id="trace_v2_state_no_draft",
    )
    client = StaticAgentClientV2(
        outputs=[
            json.dumps(
                {
                    "goal": "记录候选人确认",
                    "reasoning_summary": "模型试图记录一个没有邀约草稿的候选人。",
                    "reply_to_user": "",
                    "tool_calls": [
                        {
                            "name": "record_candidate_reply",
                            "arguments": {
                                "game_id": game.game_id,
                                "customer_id": "ran",
                                "status": "confirmed",
                            },
                        }
                    ],
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "goal": "提示人工确认",
                    "reasoning_summary": "工具返回没有邀约草稿，不能凭空确认候选人。",
                    "reply_to_user": "这个确认没有对应邀约记录，我先转人工核一下。",
                    "tool_calls": [],
                    "needs_human": True,
                },
                ensure_ascii=False,
            ),
        ],
        calls=[],
    )
    runtime = AgentRuntimeV2(llm_client=client, store=store)

    result = runtime.handle_user_message(message("冉姐说来"), trace_id="trace_v2_state_no_draft")

    assert result.final_reply == "这个确认没有对应邀约记录，我先转人工核一下。"
    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert "candidate reply requires an existing invite draft" in str(result.tool_results[0].error)
    assert all(participant.customer_id != "ran" for participant in store.games[game.game_id].participants)


def test_v2_state_policy_rejects_illegal_invite_status_transition() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="boss_v2",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"user_visible_summary": "杭麻 1档 人齐开"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="trace_v2_state_transition",
    )
    store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[{"customer_id": "ran", "message_text": "冉姐，人齐开，1块，打吗？"}],
        trace_id="trace_v2_state_transition",
    )
    store.record_candidate_reply(
        game_id=game.game_id,
        customer_id="ran",
        status="confirmed",
        trace_id="trace_v2_state_transition",
    )
    client = StaticAgentClientV2(
        outputs=[
            json.dumps(
                {
                    "goal": "非法改写候选人状态",
                    "reasoning_summary": "模型试图把已确认邀约改成拒绝。",
                    "reply_to_user": "",
                    "tool_calls": [
                        {
                            "name": "record_candidate_reply",
                            "arguments": {
                                "game_id": game.game_id,
                                "customer_id": "ran",
                                "status": "declined",
                            },
                        }
                    ],
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "goal": "状态机拒绝后转人工",
                    "reasoning_summary": "工具返回已确认不能直接改成拒绝。",
                    "reply_to_user": "这个状态变更不合法，我先转人工确认。",
                    "tool_calls": [],
                    "needs_human": True,
                },
                ensure_ascii=False,
            ),
        ],
        calls=[],
    )
    runtime = AgentRuntimeV2(llm_client=client, store=store)

    result = runtime.handle_user_message(message("冉姐又说不来了"), trace_id="trace_v2_state_transition_retry")

    assert result.final_reply == "这个状态变更不合法，我先转人工确认。"
    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert "illegal invite status transition: confirmed -> declined" in str(result.tool_results[0].error)
    assert any(participant.customer_id == "ran" for participant in store.games[game.game_id].participants)


def test_v2_runtime_records_badcase_when_model_reports_it() -> None:
    store = seeded_store()
    eval_recorder = InMemoryEvalRecorderV2()
    gateway = ToolGatewayV2(store=store, eval_recorder=eval_recorder)
    client = StaticAgentClientV2(
        outputs=[
            json.dumps(
                {
                    "goal": "归档错误回复",
                    "reasoning_summary": "模型判断上一轮回复不合适，需要进入 badcase。",
                    "reply_to_user": "我先记一下这个问题。",
                    "tool_calls": [],
                    "needs_human": False,
                    "badcase": {
                        "reason": "候选人邀约暴露了内部状态",
                        "input": {"text": "人齐开"},
                        "actual": {"reply": "asap_when_full"},
                        "expected": {"reply": "不要暴露内部枚举"},
                        "tags": ["visibility", "tool_contract"],
                    },
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "goal": "回复用户",
                    "reasoning_summary": "badcase 已归档。",
                    "reply_to_user": "这个问题我已经记录到 badcase 里了。",
                    "tool_calls": [],
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
        ],
        calls=[],
    )
    trace = InMemoryTraceRecorderV2()
    runtime = AgentRuntimeV2(
        llm_client=client,
        store=store,
        tool_gateway=gateway,
        trace_recorder=trace,
    )

    result = runtime.handle_user_message(message("这个回复不对"), trace_id="trace_v2_badcase")

    assert result.final_reply == "这个问题我已经记录到 badcase 里了。"
    assert [tool.name for tool in result.tool_results] == ["record_badcase"]
    assert result.tool_results[0].called is True
    assert result.tool_results[0].allowed is True
    assert len(eval_recorder.records) == 1
    assert eval_recorder.records[0]["schema_version"] == "agent_runtime_v2.badcase.v1"
    assert eval_recorder.records[0]["reason"] == "候选人邀约暴露了内部状态"
    assert eval_recorder.records[0]["trace_id"] == "trace_v2_badcase"
    events = trace.get_trace("trace_v2_badcase")
    assert any(event.step == "tool_result" for event in events)
    proposed = [event for event in events if event.step == "action_proposed"][0]
    assert proposed.content["tool_calls"][0]["name"] == "record_badcase"


def test_v2_runtime_does_not_duplicate_explicit_record_badcase_tool_call() -> None:
    store = seeded_store()
    eval_recorder = InMemoryEvalRecorderV2()
    gateway = ToolGatewayV2(store=store, eval_recorder=eval_recorder)
    badcase_payload = {
        "reason": "回复不自然",
        "input": {"text": "组"},
        "expected": {"reply": "自然追问"},
    }
    client = StaticAgentClientV2(
        outputs=[
            json.dumps(
                {
                    "goal": "显式归档 badcase",
                    "reasoning_summary": "模型已经显式调用 record_badcase，不应再由 Runtime 补第二次。",
                    "reply_to_user": "",
                    "tool_calls": [
                        {
                            "name": "record_badcase",
                            "arguments": badcase_payload,
                            "reason": "显式归档",
                        }
                    ],
                    "needs_human": False,
                    "badcase": badcase_payload,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "goal": "回复用户",
                    "reasoning_summary": "badcase 已归档一次。",
                    "reply_to_user": "这个问题我已经记录下来了。",
                    "tool_calls": [],
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
        ],
        calls=[],
    )
    trace = InMemoryTraceRecorderV2()
    runtime = AgentRuntimeV2(
        llm_client=client,
        store=store,
        tool_gateway=gateway,
        trace_recorder=trace,
    )

    result = runtime.handle_user_message(message("这个也不对"), trace_id="trace_v2_badcase_no_duplicate")

    assert result.final_reply == "这个问题我已经记录下来了。"
    assert [tool.name for tool in result.tool_results] == ["record_badcase"]
    assert len(eval_recorder.records) == 1
    proposed = [
        event
        for event in trace.get_trace("trace_v2_badcase_no_duplicate")
        if event.step == "action_proposed"
    ][0]
    assert [call["name"] for call in proposed.content["tool_calls"]] == ["record_badcase"]


def test_v2_runtime_deduplicates_same_message_id_without_second_llm_call() -> None:
    store = seeded_store()
    client = StaticAgentClientV2(
        outputs=[
            json.dumps(
                {
                    "goal": "回复一次",
                    "reasoning_summary": "首次处理消息。",
                    "reply_to_user": "收到，我先看一下。",
                    "tool_calls": [],
                    "needs_human": False,
                },
                ensure_ascii=False,
            )
        ],
        calls=[],
    )
    runtime = AgentRuntimeV2(llm_client=client, store=store)
    incoming = message("老板")
    incoming.message_id = "same-message-id"

    first = runtime.handle_user_message(incoming, trace_id="trace_first")
    second = runtime.handle_user_message(incoming, trace_id="trace_second")

    assert first.final_reply == "收到，我先看一下。"
    assert second.final_reply == first.final_reply
    assert second.trace_id == first.trace_id
    assert len(client.calls) == 1


def test_v2_runtime_serializes_same_conversation_llm_calls() -> None:
    store = seeded_store()
    client = ConcurrencyProbeClient()
    runtime = AgentRuntimeV2(llm_client=client, store=store)
    first_message = message("第一条")
    first_message.message_id = "concurrent-1"
    second_message = message("第二条")
    second_message.message_id = "concurrent-2"
    results = []

    threads = [
        threading.Thread(target=lambda msg=first_message: results.append(runtime.handle_user_message(msg))),
        threading.Thread(target=lambda msg=second_message: results.append(runtime.handle_user_message(msg))),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 2
    assert client.max_active == 1
    assert len(client.calls) == 2


class ConcurrencyProbeClient:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = []

    def complete(self, messages, *, trace_id, timeout_seconds):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append({"messages": messages, "trace_id": trace_id})
        time.sleep(0.05)
        with self._lock:
            self.active -= 1
        return json.dumps(
            {
                "goal": "并发测试",
                "reasoning_summary": "直接回复。",
                "reply_to_user": "收到。",
                "tool_calls": [],
                "needs_human": False,
            },
            ensure_ascii=False,
        )


def test_v2_jsonl_eval_recorder_persists_badcase(tmp_path) -> None:
    path = tmp_path / "agent_runtime_v2_badcases.jsonl"
    recorder = JsonlEvalRecorderV2(path)

    record = recorder.record_badcase(
        {"reason": "回复太僵硬", "input": {"text": "组"}, "expected": {"reply": "自然追问"}},
        trace_id="trace_v2_eval_file",
        conversation_id="boss_v2",
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    persisted = json.loads(lines[0])
    assert persisted == record
    assert persisted["reason"] == "回复太僵硬"
    assert persisted["conversation_id"] == "boss_v2"


def test_v2_sqlite_store_persists_state_turns_and_idempotency(tmp_path) -> None:
    db_path = tmp_path / "agent_runtime_v2.sqlite3"
    store = SQLiteAgentStoreV2(db_path)
    store.upsert_customer(
        CustomerProfileV2(
            customer_id="zhang",
            display_name="张哥",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["1"],
        )
    )
    store.upsert_customer(
        CustomerProfileV2(
            customer_id="ran",
            display_name="冉姐",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["1"],
        )
    )
    game, transition = store.create_game(
        conversation_id="sqlite_v2",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="trace_v2_sqlite",
    )
    drafts, invite_transitions = store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[{"customer_id": "ran", "display_name": "冉姐", "message_text": "冉姐，1块，打吗？"}],
        trace_id="trace_v2_sqlite",
    )
    store.append_turn(
        "sqlite_v2",
        ConversationTurnV2(
            role=ConversationRoleV2.ASSISTANT,
            content="好的，我先帮你问问。",
            trace_id="trace_v2_sqlite",
        ),
    )
    store.remember_result(
        "idem:v2:test",
        ToolResultV2(
            name="create_game",
            called=True,
            allowed=True,
            result={"game_id": game.game_id},
            state_transitions=[transition, *invite_transitions],
        ),
    )
    store.remember_message_result(
        "message:v2:test",
        AgentRuntimeResultV2(
            trace_id="trace_v2_sqlite",
            final_reply="好的，我先帮你问问。",
            decisions=[],
            tool_results=[],
            state_transitions=[transition, *invite_transitions],
            conversation_id="sqlite_v2",
        ),
    )
    store.close()

    restored = SQLiteAgentStoreV2(db_path)

    assert sorted(restored.customers) == ["ran", "zhang"]
    assert game.game_id in restored.games
    assert drafts[0].draft_id in restored.invite_drafts
    assert restored.games[game.game_id].status.value == "inviting"
    assert restored.recent_turns("sqlite_v2", limit=1)[0].content == "好的，我先帮你问问。"
    restored_result = restored.idempotent_result("idem:v2:test")
    assert restored_result is not None
    assert restored_result.name == "create_game"
    assert restored_result.result["game_id"] == game.game_id
    restored_message = restored.idempotent_message_result("message:v2:test")
    assert restored_message is not None
    assert restored_message.final_reply == "好的，我先帮你问问。"
    assert restored_message.conversation_id == "sqlite_v2"
    assert len(restored.transitions) == 2
    assert restored.idempotent_message_result("missing") is None
    restored.close()


def test_v2_runtime_source_does_not_import_legacy_parser_workflow_or_guard() -> None:
    import inspect
    import mahjong_agent_v2.context as context
    import mahjong_agent_v2.runtime as runtime
    import mahjong_agent_v2.sqlite_store as sqlite_store
    import mahjong_agent_v2.state_policy as state_policy
    import mahjong_agent_v2.store as store
    import mahjong_agent_v2.tools as tools

    source = "\n".join(
        [
            inspect.getsource(context),
            inspect.getsource(runtime),
            inspect.getsource(sqlite_store),
            inspect.getsource(state_policy),
            inspect.getsource(store),
            inspect.getsource(tools),
        ]
    )
    assert "mahjong_agent.parser" not in source
    assert "semantic_resolver" not in source
    assert "controlled_workflow" not in source
    assert "reply_guard" not in source


def test_v2_system_prompt_separates_customer_reply_from_operator_notes() -> None:
    prompt = DEFAULT_V2_PROMPT_PATH.read_text(encoding="utf-8")
    review_prompt = DEFAULT_V2_PROMPT_PATH.with_name("agent_v2_reply_review.md").read_text(encoding="utf-8")
    decision_review_prompt = DEFAULT_V2_PROMPT_PATH.with_name("agent_v2_decision_review.md").read_text(encoding="utf-8")

    assert "`reply_to_user` 是发给当前消息发送者的客户可见回复" in prompt
    assert "后台事实、工具执行结果、候选人名单、草稿审批状态只能写在 `reasoning_summary`" in prompt
    assert "`create_invite_drafts` 只创建待审批草稿，不代表已经发出邀约" in prompt
    assert "内部结构化字段必须使用 schema 中的 canonical 值" in prompt
    assert "中文自然表达只写进 `start_time_text`、`duration_text`、`smoke_label`" in prompt
    assert "不要暴露候选人姓名、候选人数、建局、创建记录、草稿、审批、后台看板" in prompt
    assert "不要说“已建局/局已建好/已创建/已组好”" in prompt
    assert "不要要求用户去审批" in prompt
    assert "只要当前目标是“帮用户找人/组局”" in prompt
    assert "用户回复“可以/组/帮我组/好”" in prompt
    assert "“人齐开/找到人再商量/尽快开”是有效的 start_time_kind=asap_when_full" in prompt
    assert "最终回复审查模型" in review_prompt
    assert "approved" in review_prompt
    assert "badcase" in review_prompt
    assert "不负责硬编码改写某一句话" in review_prompt
    assert "不要一次性追问多个槽位" in review_prompt
    assert "不要无必要地反复问“杭麻还是川麻”" in review_prompt
    assert "动作审查模型" in decision_review_prompt
    assert "审查的是“该不该继续行动、该调用哪些工具”" in decision_review_prompt
    assert "不要让后端硬编码麻将语义" in decision_review_prompt
    assert "revised_decision" in decision_review_prompt


def seeded_store() -> InMemoryAgentStoreV2:
    store = InMemoryAgentStoreV2()
    store.upsert_customer(
        CustomerProfileV2(
            customer_id="zhang",
            display_name="张哥",
            gender="男",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["0.5", "1"],
            smoke_preference="any",
            response_score=0.9,
        )
    )
    store.upsert_customer(
        CustomerProfileV2(
            customer_id="ran",
            display_name="冉姐",
            gender="女",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["1"],
            smoke_preference="any",
            response_score=0.9,
        )
    )
    return store


def message(text: str) -> UserMessageV2:
    return UserMessageV2(
        conversation_id="test_v2",
        sender_id="zhang",
        sender_name="张哥",
        text=text,
    )
