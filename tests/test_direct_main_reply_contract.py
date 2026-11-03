"""核对主回复、上下文和审查合同同向；离线测试不冒充真实模型语义通过。"""

from pathlib import Path

import pytest

from mahjong_agent_runtime import AgentAction, AgentContextBuilder, CustomerProfile, InMemoryAgentStore, ToolGateway, UserMessage
from mahjong_agent_runtime.customer_visible_review import build_reply_self_review_payload


ROOT = Path(__file__).resolve().parents[1]


def main_context():
    """可靠画像与客户本次条件一同送给主模型，最终话术无须依赖另一次润色获取事实。"""
    store = InMemoryAgentStore()
    store.upsert_customer(CustomerProfile(customer_id="customer", display_name="客户",
        preferred_games=["hangzhou_mahjong"], preferred_stakes=["0.5"], smoke_preference="no_smoke"))
    message = UserMessage(conversation_id="direct-reply", sender_id="customer", sender_name="客户",
                          text="帮我找六点半无烟的", message_id="direct-input")
    payload = AgentContextBuilder(store, ToolGateway(store)).build(message, trace_id="direct-reply").payload
    return message, payload


def test_main_context_assigns_final_wording_and_state_to_the_agent_with_all_facts():
    message, payload = main_context()
    assert payload["current_message"]["text"] == message.text
    assert payload["sender_profile"]["preferred_stakes"] == ["0.5"]
    assert "最终客户可见回复" in payload["output_contract"]["field_types"]["reply_to_user"]
    guidance = payload["planning_contract"]["tool_selection_hints"]["final_customer_reply"]
    assert "已明确且未变化" in guidance and "不依赖后续润色" in guidance
    assert "waiting_user" in payload["output_contract"]["stop_reason_contract"]["waiting_decision"]


@pytest.mark.parametrize("text", ["七点三缺一，你打吗？", "七点三缺一，0.5无烟杭麻，你打吗？"])
def test_real_option_review_distinguishes_omission_and_minor_echo_from_fact_errors(text):
    """省略和偶尔复述都不改变真实局事实；判定器仍需检查不同条件和真正的新问题。"""
    message, context = main_context()
    action = AgentAction(goal="介绍真实现成局并等客户决定", objective_status="waiting_user", reasoning_summary="",
                         reply_to_user=text, objective_state={"known_facts": {"stake": "0.5", "smoke_preference": "no_smoke"}})
    context["active_game_visible_summaries"] = [{"user_visible_summary": "七点三缺一", "stake": "0.5", "smoke_preference": "no_smoke"}]
    payload = build_reply_self_review_payload(message=message, action=action, context_payload=context,
        review_scope="reply_to_user", review_items=[{"item_id": "reply_to_user", "recipient_id": "customer", "text": text}])
    fidelity = payload["semantic_fidelity_contract"]
    assert fidelity["allow_known_user_echo_omission"] is True
    rules = "\n".join(fidelity["rules"])
    assert "包括真实局选项" in rules and "已明确且未变化" in rules
    assert "偶尔复述" in rules and "风格瑕疵" in rules
    assert "你打吗" in rules and "waiting_user" in rules
    assert payload["review_items"][0]["text"] == text


def test_recipient_boundaries_remain_distinct_from_known_condition_omission():
    """不能把发起人已知条件当成另一候选人也知道，更不能借省略授权披露关系。"""
    message, context = main_context()
    action = AgentAction(goal="候选邀约", objective_status="completed", reasoning_summary="", reply_to_user="")
    payload = build_reply_self_review_payload(message=message, action=action, context_payload=context,
        review_scope="tool_message", review_items=[{"item_id": "candidate", "recipient_id": "other", "text": "七点0.5无烟，打吗？"}])
    assert payload["semantic_fidelity_contract"]["allow_known_user_echo_omission"] is False
    rules = "\n".join(payload["semantic_fidelity_contract"]["rules"])
    assert "实际收件人" in rules
    assert "对他尚未知" in rules


def test_main_and_review_prompts_share_direct_reply_and_nonfixed_confirmation_rules():
    main = (ROOT / "src/mahjong_agent_runtime/prompts/agent_runtime_system.md").read_text()
    review = (ROOT / "src/mahjong_agent_runtime/prompts/agent_runtime_reply_self_review.md").read_text()
    assert "不依赖后续润色" in main
    assert "你打吗" in main and "你打吗" in review
    assert "已明确且未变化" in main and "已明确且未变化" in review
    assert "偶尔复述" in review
    assert "无具体局选项的服务提议可省去" not in review


def test_current_room_query_contract_uses_backend_default_instead_of_forcing_a_time_question():
    _, payload = main_context()
    guidance = payload["planning_contract"]["tool_selection_hints"]["game_lookup_vs_room_availability"]
    assert "check_room_availability({})" in guidance and "4小时" in guidance
    assert "不代表客户预约" in guidance and "部分明确" in guidance


def test_search_feedback_does_not_reintroduce_a_fixed_closing_or_treat_accurate_echo_as_fact_error():
    """工具回灌合同与主提示同向，不能在下一步又把允许的话术标成事实错误。"""
    from mahjong_agent_runtime.domains.tools.shared import current_game_search_reply_contract

    contract = current_game_search_reply_contract({"stake": "0.5"}, [])
    assert "no fixed closing" in contract["customer_visible_rule"]
    assert "style issue" in contract["customer_visible_rule"]
    assert "七点三缺一，0.5无烟杭麻，打吗" in contract["style_only_echo_examples"]
    assert "七点三缺一，0.5无烟杭麻，打吗" not in contract["bad_reply_examples"]
