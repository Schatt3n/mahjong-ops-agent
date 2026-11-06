"""来源审计仅服务主循环；话术及客户审查不得接收工具反馈的内部来源元数据。"""

from copy import deepcopy
import json

import pytest

from mahjong_agent_runtime import AgentAction, AgentRuntime, InMemoryAgentStore, InMemoryTraceRecorder, StaticAgentClient, UserMessage
from mahjong_agent_runtime.budget import TokenBudget
from mahjong_agent_runtime.copywriting import build_customer_visible_text_generation_payload
from mahjong_agent_runtime.customer_visible_contract import customer_visible_text_contract_violations
from mahjong_agent_runtime.customer_visible_review import build_reply_self_review_payload, expand_review_transport_references


def message_and_action():
    """使用不含来源标识的普通客户消息，隔离工具证据投影造成的额外暴露。"""
    return (
        UserMessage(conversation_id="source-private-chat", sender_id="customer", sender_name="客户",
                    text="今晚国标一块，有没有局？", message_id="customer-input"),
        AgentAction(goal="回答已查到的局况", objective_status="waiting_user", reasoning_summary="已查真实局况",
                    reply_to_user="七点三缺一，可以不？"),
    )


def internal_feedback():
    """人工认证哨兵不是实际凭证，用来验证整块元数据没有进入任何可见模型请求。"""
    return {
        "name": "agent_action_contract", "called": False, "allowed": False,
        "call_id": "business-tool-call-1", "error": "AgentAction contract invalid",
        "result": {
            "errors": ["invalid JSON object"],
            "instruction": "Regenerate one legal action; no business tool executed.",
            "requirement": {"game_type": "national_standard_mahjong", "stake": "1", "smoke_preference": "any"},
            "remaining_seats": 1,
            "note": "字段 response_source 与 completion_diagnostics 不属于客户要听的牌局信息。",
            "response_source": {
                "response_id": "OPAQUE_RESPONSE_ID_SHOULD_STAY_INTERNAL",
                "call_id": "llm_call_aabbccddeeff00112233445566778899",
                "attempt_id": "llm_attempt_99887766554433221100ffeeddccbbaa",
                "untrusted_extra": {"provider": "PRIVATE_PROVIDER_ONLY", "url": "PRIVATE_ENDPOINT_ONLY"},
            },
            "completion_diagnostics": {
                "request_ids": {"x-request-id": "PRIVATE_REQUEST_ID_ONLY"},
                "reasoning_content": "SYNTHETIC_REASONING_MUST_NOT_ESCAPE",
                "api_key": "SYNTHETIC_CREDENTIAL_MUST_NOT_ESCAPE",
            },
        },
    }


def make_review(context):
    """经公开审查builder验证边界，不依赖某个新内部strip helper的实现方式。"""
    message, action = message_and_action()
    return build_reply_self_review_payload(
        message=message, action=action, context_payload=context, review_scope="reply_to_user",
        review_items=[{"item_id": "reply_to_user", "recipient_id": message.sender_id, "text": action.reply_to_user}],
    )


@pytest.mark.parametrize("path", ["turn_tool_evidence", "previous_tool_results", "recent_conversation"])
@pytest.mark.parametrize("location", ["result", "top_level"])
def test_visible_review_excludes_response_source_and_diagnostics_from_every_tool_evidence_path(path, location):
    """历史工具content是JSON字符串，也要按工具协议移除内部字段，不能漏掉第三条路径。"""
    feedback = internal_feedback()
    if location == "top_level":
        for key in ("response_source", "completion_diagnostics"):
            feedback[key] = feedback["result"].pop(key)
    if path == "recent_conversation":
        context = {path: [{"role": "tool", "message_id": "tool-turn", "content": json.dumps([feedback], ensure_ascii=False)}]}
    else:
        context = {path: [feedback]}
    before = deepcopy(context)
    payload = make_review(context)
    expanded = expand_review_transport_references(payload)
    if path == "recent_conversation":
        visible = json.loads(expanded["recent_conversation_tail"][0]["content"])[0]
    else:
        visible = expanded[path][0]
    expected = deepcopy(feedback)
    for key in ("response_source", "completion_diagnostics"):
        expected.pop(key, None)
        expected["result"].pop(key, None)
    assert visible == expected
    serialized = json.dumps(payload, ensure_ascii=False)
    for sentinel in ("OPAQUE_RESPONSE_ID_SHOULD_STAY_INTERNAL", "llm_call_aabb", "llm_attempt_9988",
                     "PRIVATE_PROVIDER_ONLY", "PRIVATE_ENDPOINT_ONLY", "PRIVATE_REQUEST_ID_ONLY",
                     "SYNTHETIC_REASONING_MUST_NOT_ESCAPE", "SYNTHETIC_CREDENTIAL_MUST_NOT_ESCAPE"):
        assert sentinel not in serialized
    assert context == before


def test_visible_review_does_not_mutate_shared_original_evidence_or_business_fields():
    """同一对象在两条反馈路径复用时仍只读，原始trace与后续主模型证据必须完整保留。"""
    feedback = internal_feedback()
    context = {"turn_tool_evidence": [feedback], "previous_tool_results": [feedback]}
    before = deepcopy(context)
    payload = make_review(context)
    assert context == before
    expanded = expand_review_transport_references(payload)
    for field in ("turn_tool_evidence", "previous_tool_results"):
        projected = expanded[field][0]
        assert projected["name"] == feedback["name"]
        assert projected["call_id"] == "business-tool-call-1"
        assert projected["called"] is False and projected["allowed"] is False
        assert projected["error"] == feedback["error"]
        for key in ("errors", "instruction", "requirement", "remaining_seats", "note"):
            assert projected["result"][key] == feedback["result"][key]
    payload["turn_tool_evidence"][0]["result"]["requirement"]["stake"] = "9"
    assert context == before


def test_same_named_fields_inside_user_or_assistant_text_are_not_silently_rewritten():
    """只剥离后端工具元数据，不按字符串关键词改写用户原话、引用或客户可见历史。"""
    literal = '{"response_source":"customer wording", "completion_diagnostics":"customer question"}'
    context = {"recent_conversation": [
        {"role": "user", "content": literal, "message_id": "user-original"},
        {"role": "assistant", "content": literal, "message_id": "assistant-original"},
        {"role": "tool", "content": json.dumps([internal_feedback()], ensure_ascii=False), "message_id": "tool-original"},
    ]}
    before = deepcopy(context)
    payload = make_review(context)
    assert payload["recent_conversation_tail"][0] == context["recent_conversation"][0]
    assert payload["recent_conversation_tail"][1] == context["recent_conversation"][1]
    assert context == before
    assert "response_source" not in json.loads(payload["recent_conversation_tail"][2]["content"])[0]["result"]


def test_text_generation_keeps_its_existing_minimal_scope_when_context_gains_response_sources():
    """话术生成原本不读取工具上下文，新增审计功能不能扩大它的数据权限。"""
    message, action = message_and_action()
    feedback = internal_feedback()
    context = {"turn_tool_evidence": [feedback], "previous_tool_results": [feedback],
        "recent_conversation": [{"role": "tool", "content": json.dumps([feedback])}]}
    items = [{"item_id": "reply_to_user", "source": "reply_to_user", "text": action.reply_to_user}]
    before = deepcopy(context)
    actual = build_customer_visible_text_generation_payload(message=message, action=action, items=items,
        context_payload=context, generation_scope="reply_to_user")
    baseline = build_customer_visible_text_generation_payload(message=message, action=action, items=items,
        context_payload={}, generation_scope="reply_to_user")
    assert actual == baseline
    assert context == before


def fullwidth(value):
    """只变换稳定ASCII协议字符，验证全角混淆不会绕过末端枚举安全规则。"""
    return "".join(chr(ord(char) + 0xFEE0) if 33 <= ord(char) <= 126 else char for char in value)


@pytest.mark.parametrize("prefix", ["llm_call_", "llm_attempt_"])
@pytest.mark.parametrize("transform", [lambda value: value, fullwidth, lambda value: " ".join(value)])
def test_real_response_identifiers_and_normalization_bypasses_are_blocked(prefix, transform):
    """真实32hex身份、全角与空格分隔同样是内部信息，不能只依赖模型自觉不抄录。"""
    identity = prefix + "0123456789abcdef" * 2
    violations = customer_visible_text_contract_violations("记录编号：" + transform(identity))
    assert "internal_response_source.identifier" in violations


@pytest.mark.parametrize("field", ["response_source", "terminal_attempt_id", "raw_response_sha256", "http_response_complete"])
@pytest.mark.parametrize("transform", [lambda value: value, fullwidth, lambda value: " ".join(value)])
def test_internal_response_source_fields_are_blocked_with_stable_rule_id(field, transform):
    """专用审计字段用统一规则编号拒绝，处理边界不依赖客户业务词汇。"""
    assert "internal_response_source.field" in customer_visible_text_contract_violations(transform(field))


@pytest.mark.parametrize("text", [
    "七点国标一块，三缺一，可以不？", "好，这桌我记下了。", "人齐开，烟都行。",
    "llm_call_abc", "llm_attempt_0123456789abcdef0123456789abcde",
    "call_0123456789abcdef0123456789abcdef", "两个人，最多等十分钟。",
])
def test_normal_business_text_and_short_id_lookalikes_are_not_misclassified(text):
    """正常业务和不足32hex的近似字符串不属于新身份风险，避免任意扩大过滤。"""
    assert not any(rule.startswith("internal_response_source") for rule in customer_visible_text_contract_violations(text))


def test_backend_rejects_identifier_even_when_review_model_approves_it():
    """调用真实审查解析与后端聚合路径，模型批准不能替代确定性末端校验。"""
    observed = []

    class ApprovingReviewClient:
        """受控模型替身只用于证明后端拒绝边界，不计入真实语义成绩。"""

        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]["content"])
            observed.append(payload)
            return json.dumps({
                "approved": True, "needs_human": False, "violations": [], "reasoning_summary": "批准原文",
                "item_reviews": [{"item_id": item["item_id"], "approved": True, "violations": [],
                    "reasoning_summary": "批准原文", "suggested_safe_text": item["text"]} for item in payload["review_items"]],
            }, ensure_ascii=False)

    message, action = message_and_action()
    action.reply_to_user = "llm_call_0123456789abcdef0123456789abcdef"
    store, trace = InMemoryAgentStore(), InMemoryTraceRecorder()
    runtime = AgentRuntime(store=store, llm_client=StaticAgentClient([]), trace_recorder=trace,
        reply_self_review_enabled=True, reply_self_review_client=ApprovingReviewClient())
    result = runtime.action_processor.review_customer_visible_content(
        message=message, trace_id="identifier-terminal-filter", action=action, context_payload={}, review_scope="reply_to_user",
        review_items=[{"item_id": "reply_to_user", "recipient_id": message.sender_id, "text": action.reply_to_user}],
        turn_budget=TokenBudget(max_tokens_per_call=64000, max_calls_per_turn=1),
    )
    assert len(observed) == 1 and result.called and result.allowed
    assert result.result["approved"] is False
    assert result.result["item_reviews"][0]["approved"] is False
    assert "customer_visible_contract:internal_response_source.identifier" in result.result["item_reviews"][0]["violations"]
    review = next(event.content for event in trace.events if event.step == "customer_visible_content_review_result")
    assert review["raw_approved"] is True and review["approved"] is False
    assert not store.games and not store.invite_drafts and not store.outbound_message_drafts
