"""同客户动作引用必须有真实来源、精确主体和完整文本绑定。"""

from copy import deepcopy
import json

import pytest

from mahjong_agent_runtime import (
    AgentAction, AgentRuntime, CustomerProfile, InMemoryAgentStore, InMemoryTraceRecorder,
    StaticAgentClient, UserMessage,
)
from mahjong_agent_runtime.action_attribution import build_action_attribution_context
from mahjong_agent_runtime.budget import TokenBudget
from mahjong_agent_runtime.customer_visible_review import (
    build_reply_self_review_payload, item_reviews_approved, normalize_item_reviews, parse_reply_self_review,
)
from mahjong_agent_runtime.visibility import CustomerVisibleProcessor


SOURCE = "我已经发了预算邮件，等回复就好。"
REPLY = "你已经发了预算邮件，等回复就好。"


def attribution(text=REPLY, source=SOURCE):
    return {"reviewed_text": text, "claim_span": [0, text.index("，")], "actor": "current_customer",
            "actor_span": [0, 1], "source_ref": "current_message",
            "source_quote": source[:source.index("，")], "source_claim_span": [0, source.index("，")],
            "source_actor_span": [0, 1]}


def setup_case(text=REPLY, source=SOURCE, context=None, recipient="customer", metadata=None, scope="reply_to_user"):
    message = UserMessage(conversation_id="own", sender_id="customer", sender_name="私有备注", text=source,
                          metadata=metadata or {})
    action = AgentAction(goal="回应客户当前工作话题", objective_status="completed", reasoning_summary="依据本人原话",
                         reply_to_user=text)
    items = [{"item_id": "reply_to_user", "recipient_id": recipient, "text": text}]
    payload = build_reply_self_review_payload(message=message, action=action, review_items=items,
        context_payload=context or {}, review_scope=scope)
    review = {"approved": True, "needs_human": False, "reasoning_summary": "原话归属",
              "violations": [], "item_reviews": [{"item_id": "reply_to_user", "approved": True,
                  "suggested_safe_text": text, "external_action_attributions": [attribution(text, source)]}]}
    return message, action, items, payload, review


def approved(items, payload, review):
    context = payload["external_action_attribution_context"]
    normalized = normalize_item_reviews(review, items, attribution_context=context)
    return item_reviews_approved(normalized, items, attribution_context=context), normalized


def test_actual_customer_work_sentence_passes_unchanged_without_delivery_or_second_call():
    message, action, items, payload, review = setup_case()

    class FakeClient:
        def __init__(self):
            self.requests = []

        def complete(self, messages, **kwargs):
            self.requests.append(messages)
            return json.dumps(review, ensure_ascii=False)

    class Recorder:
        def record(self, *args, **kwargs):
            pass

    client = FakeClient()
    processor = CustomerVisibleProcessor(llm_client=client, trace_recorder=Recorder(), timeout_seconds=90,
                                         review_enabled=True)
    result = processor.run_content_review(message=message, action=action, trace_id="attribution",
        review_items=items, context_payload={}, turn_budget=TokenBudget(), review_scope="reply_to_user")
    assert result.result["approved"] is True
    assert len(client.requests) == 1
    assert result.result["item_reviews"][0]["suggested_safe_text"] == REPLY
    assert approved(items, payload, review)[0]


@pytest.mark.parametrize("fault", [
    "agent_actor", "third_party", "unknown_source", "other_author", "changed_source", "changed_text",
    "changed_predicate", "partial_claim", "bool_span", "out_of_bounds", "actor_tail", "source_actor_tail",
    "forged_quote", "forged_verified_flag", "missing_proof", "wrong_recipient", "missing_context",
])
def test_mistaken_model_approval_cannot_forge_attribution(fault):
    _, _, items, payload, review = setup_case()
    proof = review["item_reviews"][0]["external_action_attributions"][0]
    context = payload["external_action_attribution_context"]
    if fault == "agent_actor":
        items[0]["text"] = SOURCE
        review["item_reviews"][0]["suggested_safe_text"] = SOURCE
        proof["reviewed_text"] = SOURCE
    elif fault == "third_party":
        proof["actor"] = "third_party"
    elif fault == "unknown_source":
        proof["source_ref"] = "recent_user_999"
    elif fault == "other_author":
        context["sources"][0]["author_id"] = "other"
    elif fault == "changed_source":
        context["sources"][0]["text"] = SOURCE.replace("已经", "还没")
    elif fault == "changed_text":
        proof["reviewed_text"] += "。"
    elif fault == "changed_predicate":
        items[0]["text"] = REPLY.replace("预算", "招募")
        review["item_reviews"][0]["suggested_safe_text"] = items[0]["text"]
        proof["reviewed_text"] = items[0]["text"]
    elif fault == "partial_claim":
        proof["claim_span"] = [0, 5]
    elif fault == "bool_span":
        proof["actor_span"] = [False, True]
    elif fault == "out_of_bounds":
        proof["claim_span"] = [0, 1000]
    elif fault == "actor_tail":
        proof["actor_span"] = [1, 2]
    elif fault == "source_actor_tail":
        proof["source_actor_span"] = [1, 2]
    elif fault == "forged_quote":
        proof["source_quote"] = "我已经发了招募消息"
    elif fault == "forged_verified_flag":
        proof["verified"] = True
    elif fault == "missing_proof":
        review["item_reviews"][0].pop("external_action_attributions")
    elif fault == "wrong_recipient":
        items[0]["recipient_id"] = "other"
    elif fault == "missing_context":
        payload["external_action_attribution_context"] = {}
    passed, normalized = approved(items, payload, review)
    assert not passed
    assert normalized[0]["violations"]


def test_second_agent_action_clause_cannot_hide_inside_customer_attribution():
    text = "你已经发了预算邮件，我已经发消息问过了。"
    _, _, items, payload, review = setup_case(text=text)
    assert not approved(items, payload, review)[0]
    # 扩大成整句也不能覆盖两个不同主语的分句。
    review["item_reviews"][0]["external_action_attributions"][0]["claim_span"] = [0, len(text)]
    assert not approved(items, payload, review)[0]


@pytest.mark.parametrize("source", ["我说我已经发了预算邮件，等回复就好。", "我说：我已经发了预算邮件，等回复就好。"])
def test_nested_speaker_is_not_a_proven_simple_actor_replacement(source):
    _, _, items, payload, review = setup_case(text="你" + source[1:], source=source)
    assert not approved(items, payload, review)[0]


def test_history_source_pool_excludes_other_customers_assistants_tools_and_other_conversations():
    turns = [
        {"role": "user", "sender_id": "other", "content": "他人私聊"},
        {"role": "assistant", "sender_id": "customer", "content": SOURCE},
        {"role": "tool", "sender_id": "customer", "content": SOURCE},
        {"role": "user", "sender_id": "customer", "conversation_id": "elsewhere", "content": SOURCE},
        {"role": "user", "sender_id": "customer", "content": SOURCE},
    ]
    _, _, items, payload, review = setup_case(context={"recent_conversation": turns})
    context = payload["external_action_attribution_context"]
    assert [source["source_ref"] for source in context["sources"]] == ["current_message", "recent_user_4"]
    proof = review["item_reviews"][0]["external_action_attributions"][0]
    proof["source_ref"] = "recent_user_4"
    assert approved(items, payload, review)[0]
    proof["source_ref"] = "recent_user_0"
    assert not approved(items, payload, review)[0]


@pytest.mark.parametrize("kwargs", [
    {"recipient": "other"}, {"scope": "tool_calls"}, {"metadata": {"room_id": "public_group"}},
    {"metadata": {"input_source": "system_trigger"}},
])
def test_private_customer_source_is_never_a_public_or_other_recipient_exception(kwargs):
    _, _, items, payload, review = setup_case(**kwargs)
    assert payload["external_action_attribution_context"] == {}
    assert not approved(items, payload, review)[0]


def test_safe_ordinary_text_needs_no_attribution_and_real_agent_delivery_still_works():
    _, _, items, payload, review = setup_case()
    items[0]["text"] = review["item_reviews"][0]["suggested_safe_text"] = "等回复就好。"
    review["item_reviews"][0].pop("external_action_attributions")
    assert approved(items, payload, review)[0]
    items[0]["text"] = review["item_reviews"][0]["suggested_safe_text"] = "我已经发消息问过了。"
    assert not approved(items, payload, review)[0]
    items[0]["action_evidence"] = {"contact_started": True}
    assert approved(items, payload, review)[0]


def test_normalized_attribution_cannot_be_replayed_for_changed_text_or_source():
    _, _, items, payload, review = setup_case()
    passed, normalized = approved(items, payload, review)
    assert passed
    context = deepcopy(payload["external_action_attribution_context"])
    context["sources"][0]["text"] = "我还没发预算邮件。"
    assert not item_reviews_approved(normalized, items, attribution_context=context)
    normalized[0]["suggested_safe_text"] = items[0]["text"] = SOURCE
    assert not item_reviews_approved(normalized, items, attribution_context=payload["external_action_attribution_context"])


def test_private_remark_and_ambiguous_public_name_cannot_become_actor_identity():
    message = UserMessage(conversation_id="own", sender_id="customer", sender_name="我", text=SOURCE)
    context = build_action_attribution_context(message, {"sender_profile": {"public_name": "我"}}, enabled=True)
    assert context["target_actor_literals"] == ["你"]


def test_explicit_public_output_cannot_reuse_a_private_reply_attribution():
    message, action, items, _, review = setup_case()
    items[0]["channel"] = "group"
    payload = build_reply_self_review_payload(message=message, action=action, review_items=items,
        context_payload={}, review_scope="reply_to_user")
    assert payload["external_action_attribution_context"] == {}
    assert not approved(items, payload, review)[0]


@pytest.mark.parametrize("bad_value", [{}, "current_customer", None, True])
def test_attribution_container_type_is_a_review_contract_error(bad_value):
    *_, review = setup_case()
    review["item_reviews"][0]["external_action_attributions"] = bad_value
    _, errors = parse_reply_self_review(json.dumps(review, ensure_ascii=False))
    assert any("external_action_attributions must be array" in error for error in errors)


def test_model_payload_declares_fixed_actor_and_integer_array_spans_with_valid_unrelated_example():
    _, _, _, payload, _ = setup_case()
    contract = payload["external_action_attribution_contract"]
    schema = contract["json_schema"]
    assert schema["type"] == "array"
    item = schema["items"]
    assert item["type"] == "object" and item["additionalProperties"] is False
    assert set(item["required"]) == set(attribution()) == set(item["properties"])
    assert item["properties"]["actor"]["type"] == "string"
    assert item["properties"]["actor"]["enum"] == ["current_customer"]
    for field in ("reviewed_text", "source_ref", "source_quote"):
        assert item["properties"][field]["type"] == "string"
    for field in ("claim_span", "actor_span", "source_claim_span", "source_actor_span"):
        span = item["properties"][field]
        assert span["type"] == "array" and span["minItems"] == span["maxItems"] == 2
        assert span["items"] == {"type": "integer", "minimum": 0}
    assert contract["span_encoding"]["valid_example"] == [0, 1]
    assert "[0,1)" in contract["span_encoding"]["invalid_examples"]
    example = contract["format_example_only"]
    example_review = example["item_review"]
    assert "预算" not in json.dumps(example, ensure_ascii=False)
    _, _, items, example_payload, review = setup_case(
        source=example["source_text"], text=example_review["suggested_safe_text"])
    review["item_reviews"] = [example_review]
    assert approved(items, example_payload, review)[0]
    assert json.loads(json.dumps(contract, ensure_ascii=False)) == contract


@pytest.mark.parametrize("field", ["actor", "claim_span", "actor_span", "source_claim_span", "source_actor_span"])
def test_customer_id_actor_and_interval_string_spans_are_rejected_without_coercion(field):
    _, _, items, payload, review = setup_case()
    proof = review["item_reviews"][0]["external_action_attributions"][0]
    proof[field] = "review_customer" if field == "actor" else f"[{proof[field][0]},{proof[field][1]})"
    original = deepcopy(review)
    passed, normalized = approved(items, payload, review)
    assert not passed
    assert "customer_visible_contract:unverified_external_action_attribution" in normalized[0]["violations"]
    assert normalized[0]["external_action_attributions"][0][field] == proof[field]
    assert review == original


def test_runtime_attribution_contract_rejection_feeds_back_then_natural_reply_has_no_business_effects():
    message, action, _, _, malformed = setup_case()
    # 复现实际失败格式，不将客户 ID 或区间字符串自动修成可授权的证明。
    proof = malformed["item_reviews"][0]["external_action_attributions"][0]
    proof["actor"] = "review_customer"
    for field in ("claim_span", "actor_span", "source_claim_span", "source_actor_span"):
        proof[field] = f"[{proof[field][0]},{proof[field][1]})"
    malformed_json = json.dumps(malformed, ensure_ascii=False)
    final_text = "嗯，那就等对方回复。"
    action.stop_reason = {"can_stop": True, "why": "仅回应当前工作话题", "pending_work": [],
                          "depends_on_tool_results": False}
    corrected = deepcopy(action)
    corrected.reply_to_user = final_text
    corrected.reasoning_summary = "根据真实拒绝反馈重新自然回应，没有声称任何已执行动作。"
    main = StaticAgentClient([json.dumps(action.to_dict(), ensure_ascii=False),
                              json.dumps(corrected.to_dict(), ensure_ascii=False)])
    accepted = {"approved": True, "needs_human": False, "reasoning_summary": "工作话题自然回应",
                "violations": [], "item_reviews": [{"item_id": "reply_to_user", "approved": True,
                    "suggested_safe_text": final_text, "reasoning_summary": "没有外发声明", "violations": []}]}
    reviewer = StaticAgentClient([malformed_json, json.dumps(accepted, ensure_ascii=False)])
    store = InMemoryAgentStore()
    store.upsert_customer(CustomerProfile(customer_id=message.sender_id, display_name="客户"))
    business_fields = ("games", "invite_drafts", "outbound_message_drafts", "scheduled_tasks", "waiting_demands",
                       "room_reservations")
    before = {field: deepcopy(getattr(store, field)) for field in business_fields}
    trace = InMemoryTraceRecorder()
    runtime = AgentRuntime(llm_client=main, store=store, trace_recorder=trace,
                           reply_self_review_enabled=True, reply_self_review_client=reviewer)
    result = runtime.handle_user_message(message, trace_id="attribution_safe_recovery")
    assert result.final_reply == final_text and result.status == "completed"
    assert len(main.calls) == len(reviewer.calls) == 2
    assert before == {field: getattr(store, field) for field in business_fields}
    # 会话目标和版本的正常审计允许保留，任何业务实体的状态变迁都不允许发生。
    assert all(transition.entity_type in {"task_context", "conversation_version"} for transition in store.transitions), [
        transition.to_dict() for transition in store.transitions]
    events = trace.get_trace("attribution_safe_recovery")
    assert not any(event.step == "tool_gateway_received" for event in events)
    reviews = [event.content for event in events if event.step == "customer_visible_content_review_result"]
    assert reviews[0]["raw_approved"] is True and reviews[0]["approved"] is False
    assert reviews[1]["approved"] is True
    assert "customer_visible_contract:unverified_external_action_attribution" in reviews[0]["item_reviews"][0]["violations"]
    retry_payload = json.loads(main.calls[1]["messages"][1]["content"])
    feedback = retry_payload["previous_tool_results"][0]
    reference = feedback["result_reference"]
    assert reference["source"] == "turn_tool_evidence"
    evidence = retry_payload["turn_tool_evidence"][reference["index"]]
    assert evidence["name"] == feedback["name"] == "customer_visible_content_review"
    assert evidence["result"]["approved"] is False
    assert "unverified_external_action_attribution" in json.dumps(evidence)
    raw_review = next(event for event in events if event.step == "customer_visible_content_review_response")
    assert raw_review.content["content"] == malformed_json
    replies = [turn.content for turn in store.recent_turns(message.conversation_id) if turn.role.value == "assistant"]
    assert replies == [final_text]
    request = json.loads(reviewer.calls[0]["messages"][1]["content"])
    assert request["external_action_attribution_contract"]["json_schema"]["items"]["properties"]["actor"]["enum"] == ["current_customer"]
