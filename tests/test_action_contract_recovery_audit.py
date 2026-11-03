"""动作拒绝恢复须逐调用绑定原响应、反馈和真实模型上下文，空文本也不能借最终成功过关。"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from mahjong_agent_runtime import AgentLLMConfig, AgentRuntime, InMemoryAgentStore, InMemoryTraceRecorder, OpenAICompatibleAgentClient, TokenBudget, UserMessage
from mahjong_agent_runtime.tracing import TraceEvent

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_real_owner_chat_live_eval as owner


def action_text(*, query=False):
    """合法动作只读查局或答复当前查询，永不创建业务副作用。"""
    return json.dumps({"goal": "读取并回答当前登记局", "objective_status": "needs_tool" if query else "completed",
        "reasoning_summary": "离线机制夹具，不是真实模型语义结论", "reply_to_user": "" if query else "现在没有登记中的局。",
        "tool_calls": [{"name": "search_current_games", "arguments": {"requirement": {}, "limit": 5}, "reason": "读取当前登记局"}] if query else [],
        "needs_human": False, "stop_reason": {"can_stop": not query, "why": "读取真实结果" if query else "本轮查询已完成",
            "pending_work": ["读取局池"] if query else [], "depends_on_tool_results": query}}, ensure_ascii=False)


class Response:
    """仅替换HTTP传输；诊断、响应来源和运行时拒绝链路均由生产代码生成。"""

    def __init__(self, content, index):
        self.status = 200
        self.headers = {"x-request-id": f"unit-request-{index}"}
        self.body = json.dumps({"id": f"unit-response-{index}", "model": "unit-model", "choices": [{
            "finish_reason": "stop", "message": {"content": content, "reasoning_content": "PRIVATE_REASONING"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.body


def run_recovery_case(*, rejected=("",), fallback=False, retry_first=False):
    """真实生产client接假HTTP，运行完整拒绝/回灌/只读工具/合法终态，不访问网络。"""
    outputs = iter([*rejected, action_text(query=True), action_text()])
    calls = []

    def transport(request, **kwargs):
        calls.append(json.loads(request.data))
        if retry_first and len(calls) == 1:
            raise ConnectionError("unit first attempt interrupted")
        return Response(next(outputs), len(calls))

    config = AgentLLMConfig(api_key="unit-only-key", model="unit-model", base_url="https://example.invalid",
                           retry_attempts=2 if retry_first else 1, retry_base_delay_seconds=0)
    client = OpenAICompatibleAgentClient(config=config, urlopen=transport)
    if fallback:
        def failed_primary(*args, **kwargs):
            raise ConnectionError("unit primary unavailable")
        primary_config = AgentLLMConfig(api_key="unit-primary-key", model="unit-primary", base_url="https://example.invalid", retry_attempts=1)
        client = OpenAICompatibleAgentClient(config=primary_config, urlopen=failed_primary, fallback_client=client)
    store, trace = InMemoryAgentStore(), InMemoryTraceRecorder()
    result = AgentRuntime(llm_client=client, store=store, trace_recorder=trace,
        token_budget=TokenBudget(max_tokens_per_call=64000, max_calls_per_turn=8), max_steps=8).handle_user_message(
            UserMessage(conversation_id="recovery-audit", sender_id="customer", sender_name="客户", text="只看看当前登记局", message_id="source-input"),
            trace_id="recovery-audit-trace")
    assert result.status == "completed" and result.final_reply == "现在没有登记中的局。"
    assert not store.games and not store.invite_drafts and not store.outbound_message_drafts
    return result, trace.get_trace(result.trace_id), calls


def audit(result, events, *, business=True, semantics=True):
    """独立语义与业务布尔只用于测试审计门槛，不作为真实模型通过率。"""
    return owner.assess_action_contract_recovery(events, runtime_status=result.status, final_reply=result.final_reply,
                                                business_checks_passed=business, semantic_passed=semantics)


def test_complete_empty_response_recovers_only_after_source_bound_feedback_and_real_tools():
    result, events, _ = run_recovery_case()
    metrics = audit(result, events)
    assert metrics["quality_status"] == "passed", metrics
    assert metrics["action_contract_error_count"] == metrics["safe_recovered_count"] == 1
    assert metrics["unrecovered_count"] == metrics["unsafe_rejected_step_count"] == 0
    response = next(event for event in events if event.step == "llm_response")
    assert response.content["content"] == ""
    assert response.content["completion_diagnostics"]["attempts"][-1]["content"]["length"] == 0


def test_serialized_trace_alone_contains_all_required_empty_recovery_evidence():
    """JSON往返去掉Python对象附带诊断，证明通过依赖落盘字段而非隐藏内存属性。"""
    result, original, _ = run_recovery_case()
    events = [TraceEvent(trace_id=event.trace_id, step=event.step, content=json.loads(json.dumps(event.content))) for event in original]
    assert audit(result, events)["safe_recovered_count"] == 1


def error_bundle(events):
    """选首个拒绝及其真实下一轮输入，故障只改临时副本，不触碰历史资产。"""
    return {name: next(event for event in events if event.step == name and (
        name not in {"context_built", "llm_prompt"} or (event.content.get("step_index") == 2 if name == "llm_prompt"
        else bool(event.content.get("previous_tool_results")))))
        for name in ("llm_response", "action_contract_error", "contract_error_feedback", "context_built", "llm_prompt")}


def rewrite_sources(value, rewrite):
    """按协议遍历证据来源；prompt JSON字符串须独立重编码以模拟真实传输缺证。"""
    if isinstance(value, list):
        for item in value:
            rewrite_sources(item, rewrite)
    if not isinstance(value, dict):
        return
    if "response_source" in value:
        rewrite(value)
    for key, item in list(value.items()):
        if key == "messages" and isinstance(item, list):
            for message in item:
                if message.get("role") == "user":
                    payload = json.loads(message["content"])
                    rewrite_sources(payload, rewrite)
                    message["content"] = json.dumps(payload, ensure_ascii=False)
        else:
            rewrite_sources(item, rewrite)


def assert_blocked(result, events):
    """任何证据缺失均保留错误次数和失败，终态成功不能抵扣。"""
    metrics = audit(result, events)
    assert metrics["quality_status"] == "failed", metrics
    assert metrics["unrecovered_count"] >= 1 and metrics["action_contract_error_count"] >= 1
    return metrics


@pytest.mark.parametrize("fallback,retry_first", [(True, False), (False, True), (True, True)])
def test_actual_return_leaf_and_last_retry_support_empty_recovery(fallback, retry_first):
    result, events, _ = run_recovery_case(fallback=fallback, retry_first=retry_first)
    assert audit(result, events)["safe_recovered_count"] == 1
    response = error_bundle(events)["llm_response"].content
    diagnostics = response["completion_diagnostics"]
    leaf = diagnostics["fallback"] if fallback else diagnostics
    assert response["response_source"]["call_id"] == leaf["call_id"]
    assert response["response_source"]["terminal_attempt_id"] == leaf["attempts"][-1]["attempt_id"]
    assert len(leaf["attempts"]) == (2 if retry_first else 1)


@pytest.mark.parametrize("stage", ["llm_response", "action_contract_error", "contract_error_feedback", "context_built", "llm_prompt"])
def test_each_link_must_keep_the_same_response_source(stage):
    result, events, _ = run_recovery_case()
    rewrite_sources(error_bundle(events)[stage].content, lambda value: value.pop("response_source"))
    assert_blocked(result, events)


@pytest.mark.parametrize("fault", [
    "content_missing", "content_null", "source_null", "source_hash", "source_step", "source_trace", "source_extra",
    "source_call", "source_attempt", "diagnostics_missing", "diagnostics_call", "attempt_missing", "attempt_repeated",
    "body_incomplete", "json_incomplete", "http_non200", "content_absent", "content_null_metadata", "content_nonstring",
    "content_length", "content_empty", "content_whitespace", "terminal_error", "no_choices", "error_step", "error_missing",
    "error_other_source", "feedback_other_source", "event_other_trace", "prompt_other_step", "prompt_other_trace", "prompt_missing",
])
def test_empty_response_identity_requires_complete_original_evidence(fault):
    result, events, _ = run_recovery_case()
    bundle = error_bundle(events)
    response = bundle["llm_response"].content
    source = response["response_source"]
    diagnostics = response["completion_diagnostics"]
    terminal = diagnostics["attempts"][-1]
    if fault == "content_missing":
        response.pop("content")
    elif fault == "content_null":
        response["content"] = None
    elif fault == "source_null":
        response["response_source"] = None
    elif fault.startswith("source_"):
        key, value = {"source_hash": ("raw_response_sha256", "0" * 64), "source_step": ("step_index", 9),
            "source_trace": ("trace_id", "other-trace"), "source_extra": ("untrusted", True),
            "source_call": ("call_id", "llm_call_" + "a" * 32), "source_attempt": ("terminal_attempt_id", "llm_attempt_" + "a" * 32)}[fault]
        source[key] = value
    elif fault == "diagnostics_missing":
        response.pop("completion_diagnostics")
    elif fault == "diagnostics_call":
        diagnostics["call_id"] = "not-an-actual-client-call"
    elif fault == "attempt_missing":
        diagnostics["attempts"] = []
    elif fault == "attempt_repeated":
        repeated = deepcopy(terminal)
        repeated["attempt_index"] = 2
        diagnostics["attempts"].append(repeated)
    elif fault in {"body_incomplete", "json_incomplete", "http_non200", "terminal_error", "no_choices"}:
        key, value = {"body_incomplete": ("body_complete", False), "json_incomplete": ("response_json_parsed", False),
            "http_non200": ("http_status", 201), "terminal_error": ("error_type", "Interrupted"), "no_choices": ("choice_count", 0)}[fault]
        terminal[key] = value
    elif fault.startswith("content_"):
        key, value = {"content_absent": ("present", False), "content_null_metadata": ("kind", "null"),
            "content_nonstring": ("kind", "non_string"), "content_length": ("length", 1),
            "content_empty": ("empty", False), "content_whitespace": ("whitespace_only", True)}[fault]
        terminal["content"][key] = value
    elif fault == "error_step":
        bundle["action_contract_error"].content["step_index"] = 2
    elif fault == "error_missing":
        bundle["action_contract_error"].content["errors"] = []
    elif fault in {"error_other_source", "feedback_other_source"}:
        other = next(event.content["response_source"] for event in events if event.step == "llm_response" and event.content["step_index"] == 2)
        target = bundle["action_contract_error"].content if fault == "error_other_source" else bundle["contract_error_feedback"].content["result"]
        target["response_source"] = deepcopy(other)
    elif fault == "event_other_trace":
        bundle["action_contract_error"].trace_id = "other-trace"
    elif fault == "prompt_other_step":
        bundle["llm_prompt"].content["step_index"] = 3
    elif fault == "prompt_other_trace":
        bundle["llm_prompt"].trace_id = "other-trace"
    elif fault == "prompt_missing":
        events.remove(bundle["llm_prompt"])
    assert_blocked(result, events)


@pytest.mark.parametrize("identity", ["call_id", "attempt_id"])
def test_call_and_attempt_identity_cannot_be_reused_in_another_trace_response(identity):
    result, events, _ = run_recovery_case()
    first, second = [event.content for event in events if event.step == "llm_response"][:2]
    if identity == "call_id":
        second["completion_diagnostics"]["call_id"] = first["completion_diagnostics"]["call_id"]
        second["response_source"]["call_id"] = first["response_source"]["call_id"]
    else:
        second["completion_diagnostics"]["attempts"][-1]["attempt_id"] = first["completion_diagnostics"]["attempts"][-1]["attempt_id"]
        second["response_source"]["terminal_attempt_id"] = first["response_source"]["terminal_attempt_id"]
    assert "model_call_or_attempt_identity_reused" in assert_blocked(result, events)["events"][0]["blocking_reasons"]


@pytest.mark.parametrize("fallback", [False, True])
def test_an_earlier_failed_attempt_cannot_be_used_as_the_returned_empty_response(fallback):
    result, events, _ = run_recovery_case(fallback=fallback, retry_first=True)
    response = error_bundle(events)["llm_response"].content
    leaf = response["completion_diagnostics"]["fallback"] if fallback else response["completion_diagnostics"]
    early_id = leaf["attempts"][0]["attempt_id"]
    rewrite_sources([event.content for event in events], lambda value: value["response_source"].update(
        terminal_attempt_id=early_id) if isinstance(value.get("response_source"), dict) and value["response_source"]["step_index"] == 1 else None)
    assert_blocked(result, events)


def test_fallback_must_not_borrow_failed_primary_identity():
    result, events, _ = run_recovery_case(fallback=True)
    response = error_bundle(events)["llm_response"].content
    primary = response["completion_diagnostics"]["primary"]
    rewrite_sources([event.content for event in events], lambda value: value["response_source"].update(
        call_id=primary["call_id"], terminal_attempt_id=primary["attempts"][-1]["attempt_id"])
        if isinstance(value.get("response_source"), dict) and value["response_source"]["step_index"] == 1 else None)
    assert_blocked(result, events)


def test_two_identical_empty_errors_each_require_their_own_feedback():
    result, events, _ = run_recovery_case(rejected=("", ""))
    assert audit(result, events)["safe_recovered_count"] == 2
    sources = [event.content["response_source"] for event in events if event.step == "action_contract_error"]
    assert sources[0] != sources[1] and sources[0]["raw_response_sha256"] == sources[1]["raw_response_sha256"]
    third_prompt = next(event for event in events if event.step == "llm_prompt" and event.content["step_index"] == 3)
    rewrite_sources(third_prompt.content, lambda value: value.update(response_source=deepcopy(sources[0]))
        if value.get("response_source") == sources[1] else None)
    metrics = assert_blocked(result, events)
    assert metrics["action_contract_error_count"] == 2 and metrics["safe_recovered_count"] == 1


def test_one_rejected_response_cannot_supply_two_recovery_events():
    result, events, _ = run_recovery_case()
    error = error_bundle(events)["action_contract_error"]
    events.insert(events.index(error) + 1, deepcopy(error))
    assert assert_blocked(result, events)["safe_recovered_count"] == 0


def test_last_compacted_prompt_must_still_carry_the_feedback():
    result, events, _ = run_recovery_case()
    prompt = error_bundle(events)["llm_prompt"]
    compacted = deepcopy(prompt)
    compacted.step = "llm_prompt_after_context_compaction"
    rewrite_sources(compacted.content, lambda value: value.pop("response_source"))
    events.insert(events.index(prompt) + 1, compacted)
    assert_blocked(result, events)


@pytest.mark.parametrize("fault", ["original_prompt_missing", "later_response_other_trace", "later_response_duplicate_step"])
def test_full_trace_cannot_borrow_a_request_or_response_from_another_call(fault):
    result, events, _ = run_recovery_case()
    if fault == "original_prompt_missing":
        events = [event for event in events if not (event.step == "llm_prompt" and event.content["step_index"] == 1)]
    else:
        terminal = [event for event in events if event.step == "llm_response"][-1]
        if fault == "later_response_other_trace":
            terminal.trace_id = "foreign-trace"
            terminal.content["response_source"]["trace_id"] = "foreign-trace"
        else:
            terminal.content["step_index"] = 2
            terminal.content["response_source"]["step_index"] = 2
    assert_blocked(result, events)


@pytest.mark.parametrize("business,semantics", [(False, True), (True, False)])
def test_empty_recovery_does_not_exempt_business_or_independent_semantic_failure(business, semantics):
    result, events, _ = run_recovery_case()
    metrics = audit(result, events, business=business, semantics=semantics)
    assert metrics["quality_status"] == "failed" and metrics["unrecovered_count"] == 1


@pytest.mark.parametrize("step,content", [
    ("tool_called", {"call": {"name": "create_game"}}),
    ("tool_result", {"name": "create_game", "called": True, "allowed": True, "result": {}}),
    ("state_transition", {"entity_type": "game", "entity_id": "unexpected"}),
    ("final_output", {"reply": "已经替你办好", "objective_status": "completed"}),
])
def test_http_empty_identity_does_not_exempt_side_effects_in_the_rejected_step(step, content):
    result, events, _ = run_recovery_case()
    error = error_bundle(events)["action_contract_error"]
    events.insert(events.index(error) + 1, TraceEvent(trace_id=result.trace_id, step=step, content=content))
    assert assert_blocked(result, events)["unsafe_rejected_step_count"] == 1


def test_legacy_nonempty_trace_keeps_text_recovery_but_never_upgrades_to_http_empty_proof():
    result, events, _ = run_recovery_case(rejected=("{invalid-json",))
    rewrite_sources([event.content for event in events], lambda value: value.pop("response_source"))
    assert audit(result, events)["safe_recovered_count"] == 1
    response = error_bundle(events)["llm_response"].content
    response["content"] = ""
    response["completion_diagnostics"]["attempts"][-1]["content"] = {"present": True, "kind": "string", "length": 0,
                                                                   "empty": True, "whitespace_only": False}
    assert_blocked(result, events)
