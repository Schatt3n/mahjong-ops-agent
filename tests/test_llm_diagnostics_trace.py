"""调用诊断只写观测轨迹，空正文仍走原合同拒绝，不进入客户回复或下一轮上下文。"""

from __future__ import annotations

import json

import pytest

from mahjong_agent_runtime import (
    AgentAction, AgentRuntime, ContextSummaryManager, ContextSummaryPolicy,
    InMemoryAgentStore, InMemoryTraceRecorder, StaticAgentClient, TokenBudget, ToolCall, UserMessage,
)
from mahjong_agent_runtime.llm import LLMCompletionText, LLMResponseFormatError
from mahjong_agent_runtime.visibility import CustomerVisibleProcessor


def _diagnostics(stage):
    """用不同调用身份检验主模型、生成和审查不会互相覆盖元数据。"""
    return {"schema_version": 1, "call_id": f"diagnostic-only-{stage}", "route": "primary",
            "attempts": [{"finish_reason": "length", "usage": {"completion_tokens": 6400},
                          "content": {"empty": True, "length": 0}}]}


def _action():
    """缺少玩法时直接澄清，无需执行任何业务写工具。"""
    return AgentAction(goal="询问未知玩法", objective_status="waiting_user", reasoning_summary="需求仍未知",
        reply_to_user="你想打什么玩法？", stop_reason={"can_stop": True, "why": "等待客户选择玩法",
        "pending_work": [], "depends_on_tool_results": False}).to_dict()


class _Client:
    """仅替代远端文本，不替代真实 Runtime、审查解析及持久化边界。"""

    def __init__(self, stage, *, empty_first=False, raises=False):
        self.stage = stage
        self.empty_first = empty_first
        self.raises = raises
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append(messages)
        if self.raises:
            raise LLMResponseFormatError("LLM response has no text content", _diagnostics(self.stage))
        if self.empty_first and len(self.calls) == 1:
            return LLMCompletionText("", _diagnostics(self.stage))
        source = json.loads(messages[-1]["content"])
        if self.stage == "main":
            payload = _action()
        elif self.stage == "generation":
            payload = {"reasoning_summary": "保留澄清含义", "item_rewrites": [
                {"item_id": item["item_id"], "final_text": item["text"], "semantic_preserved": True}
                for item in source["items"]]}
        else:
            payload = {"approved": True, "needs_human": False, "reasoning_summary": "仅澄清当前客户需求",
                "violations": [], "item_reviews": [{"item_id": item["item_id"], "approved": True,
                    "suggested_safe_text": item["text"], "reasoning_summary": "无隐私或动作声明", "violations": []}
                    for item in source["review_items"]]}
        return LLMCompletionText(json.dumps(payload, ensure_ascii=False), _diagnostics(self.stage))


def _run(*, empty_stage=None, summarize=False):
    """通过实际执行链路核对诊断与阶段绑定；夹具不允许写局或外发。"""
    store, trace = InMemoryAgentStore(), InMemoryTraceRecorder()
    clients = {stage: _Client(stage, empty_first=stage == empty_stage)
               for stage in ("main", "generation", "review")}
    summary = ContextSummaryManager(store=store, trace_recorder=trace,
        llm_client=StaticAgentClient([LLMCompletionText(json.dumps({
            "summary": "客户想打麻将，正在等待其选择玩法。", "facts": {}, "open_questions": ["玩法"], "confidence": 0.9,
        }, ensure_ascii=False), _diagnostics("summary"))]),
        policy=ContextSummaryPolicy(min_turns_before_summary=2, min_turns_since_last_summary=1,
                                    max_recent_tokens_before_summary=1)) if summarize else None
    runtime = AgentRuntime(store=store, llm_client=clients["main"], trace_recorder=trace,
        customer_visible_text_generation_enabled=True, customer_visible_text_generation_client=clients["generation"],
        reply_self_review_enabled=True, reply_self_review_client=clients["review"], context_summary_manager=summary)
    message = UserMessage(conversation_id="diagnostic-chat", sender_id="customer", sender_name="客户",
                          text="我想打麻将", message_id="diagnostic-input")
    result = runtime.handle_user_message(message, trace_id="diagnostic-runtime")
    assert result.status == "waiting_user"
    assert result.final_reply == "你想打什么玩法？"
    assert not store.games and not store.invite_drafts and not store.outbound_message_drafts
    assert clients["generation"].calls == []
    for client in clients.values():
        assert "diagnostic-only-" not in json.dumps(client.calls, ensure_ascii=False)
    assert "diagnostic-only-" not in json.dumps([turn.to_dict() for turn in store.recent_turns(message.conversation_id)], ensure_ascii=False)
    return trace.get_trace(result.trace_id)


def test_response_diagnostics_follow_each_model_stage_without_context_leak():
    events = _run()
    assert not any("text_generation" in event.step for event in events)
    for step, stage in (("llm_response", "main"), ("customer_visible_content_review_response", "review")):
        event = next(event for event in events if event.step == step)
        assert event.content["completion_diagnostics"]["call_id"] == f"diagnostic-only-{stage}"


@pytest.mark.parametrize("stage,contract_error,response_step", [
    ("main", "action_contract_error", "llm_response"),
    ("review", "customer_visible_content_review_contract_error", "customer_visible_content_review_response"),
])
def test_empty_completion_keeps_original_rejection_and_diagnostic_evidence(stage, contract_error, response_step):
    events = _run(empty_stage=stage)
    assert any(event.step == contract_error for event in events)
    response = next(event for event in events if event.step == response_step)
    assert response.content["content"] == ""
    assert response.content["completion_diagnostics"] == _diagnostics(stage)


def _tool_generation(*, error=False):
    """邀请草稿仍走原生成入口，诊断隔离和失败回退继续有生产组件证据。"""
    trace = InMemoryTraceRecorder()
    client = _Client("generation", raises=error)
    processor = CustomerVisibleProcessor(llm_client=client, trace_recorder=trace, timeout_seconds=1,
        text_generation_enabled=True, text_generation_client=client)
    action = AgentAction(goal="创建邀约草稿", objective_status="needs_tool", reasoning_summary="",
        tool_calls=[ToolCall(name="create_invite_drafts", arguments={"game_id": "fixture", "invitations": [
            {"customer_id": "candidate", "display_name": "候选人", "message_text": "七点三缺一，打吗？"}]}, reason="邀约草稿")])
    result = processor.run_text_generation(message=UserMessage(conversation_id="draft-diagnostics",
        sender_id="customer", sender_name="客户", text="帮我问问"), trace_id="tool-diagnostics", action=action,
        items=[{"item_id": "tool_calls[1].arguments.invitations[1].message_text", "source": "tool_calls",
                "text": "七点三缺一，打吗？"}], context_payload={"private_should_not_enter": "private"},
        turn_budget=TokenBudget(max_calls_per_turn=1), generation_scope="tool_calls")
    assert len(client.calls) == 1
    assert "diagnostic-only-" not in json.dumps(client.calls, ensure_ascii=False)
    assert "private_should_not_enter" not in json.dumps(client.calls, ensure_ascii=False)
    assert action.tool_calls[0].arguments["invitations"][0]["message_text"] == "七点三缺一，打吗？"
    return result, trace.get_trace("tool-diagnostics")


def test_tool_generation_response_keeps_diagnostics_out_of_rewrite_result():
    result, events = _tool_generation()
    assert result is not None
    response = next(event for event in events if event.step == "customer_visible_text_generation_response")
    assert response.content["completion_diagnostics"] == _diagnostics("generation")
    assert "diagnostic-only-" not in json.dumps(result.to_dict(), ensure_ascii=False)


def test_tool_generation_protocol_error_keeps_diagnostics_and_original_fallback_behavior():
    result, events = _tool_generation(error=True)
    assert result is None
    error = next(event for event in events if event.step == "customer_visible_text_generation_error")
    assert error.content["completion_diagnostics"] == _diagnostics("generation")


def test_summary_response_keeps_diagnostics_out_of_checkpoint():
    events = _run(summarize=True)
    response = next(event for event in events if event.step == "context_summary_response")
    assert response.content["completion_diagnostics"] == _diagnostics("summary")
    checkpoint = next(event for event in events if event.step == "context_summary_saved")
    assert "diagnostic-only-" not in json.dumps(checkpoint.content, ensure_ascii=False)
