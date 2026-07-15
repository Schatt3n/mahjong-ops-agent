from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from mahjong_agent_runtime.copywriting import build_customer_visible_text_generation_payload
from mahjong_agent_runtime.customer_visible_contract import (
    FORBIDDEN_CUSTOMER_SERVICE_PHRASES,
    PREFERRED_CANDIDATE_INVITE_PHRASES,
    PREFERRED_OPERATION_ACK_PHRASES,
    PREFERRED_REQUESTER_CURRENT_GAME_PHRASES,
    compact_customer_visible_text,
    customer_visible_contract_snapshot,
    customer_visible_text_contract_violations,
)
from mahjong_agent_runtime.copywriting import parse_customer_visible_text_generation
from mahjong_agent_runtime.models import AgentAction, UserMessage


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_customer_visible_contract.py"


def load_contract_verifier_module():
    spec = importlib.util.spec_from_file_location("verify_customer_visible_contract_for_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_customer_visible_contract_verifier_passes_current_prompts() -> None:
    module = load_contract_verifier_module()

    assert module.verify_prompts() == []


def test_customer_visible_copywriting_payload_uses_shared_contract_terms() -> None:
    payload = build_customer_visible_text_generation_payload(
        message=UserMessage(conversation_id="c1", sender_id="zhang", sender_name="张哥", text="现在有人吗"),
        action=AgentAction(
            goal="测试客户可见话术合同",
            objective_status="completed",
            reasoning_summary="",
            reply_to_user="七点三缺一，打吗？",
            stop_reason={"can_stop": True, "why": "test", "pending_work": [], "depends_on_tool_results": False},
        ),
        items=[{"item_id": "reply_to_user", "source": "reply_to_user", "text": "七点三缺一，打吗？"}],
        context_payload={},
        generation_scope="reply_to_user",
    )

    style_contract = payload["style_quality_contract"]
    assert style_contract["forbidden_customer_service_phrases"] == list(FORBIDDEN_CUSTOMER_SERVICE_PHRASES)
    assert style_contract["preferred_short_phrases"] == [
        *PREFERRED_REQUESTER_CURRENT_GAME_PHRASES,
        *PREFERRED_CANDIDATE_INVITE_PHRASES,
        *PREFERRED_OPERATION_ACK_PHRASES,
    ]
    style_examples_text = "\n".join(item["good"] for item in payload["style_examples"])
    assert "七点三缺一，可以不？" in style_examples_text
    assert "还没有，还差俩" in style_examples_text
    assert "两个人，18.30 星月的局，371 她，打吗？" in style_examples_text
    assert payload["current_request"] == {"text": "现在有人吗", "quoted_text": ""}
    assert "Only items[].text is allowed as factual source" in payload["semantic_source_of_truth"]
    assert "current_request is only used to decide" in payload["semantic_source_of_truth"]
    assert payload["reply_relevance_contract"]["applies_when"] == "generation_scope=reply_to_user"
    assert "shortest text that fully answers current_request" in payload["reply_relevance_contract"]["rule"]
    assert "Preserving a coherent customer decision summary overrides brevity" in payload["reply_relevance_contract"]["priority"]
    assert "treat those clauses as one coherent option summary" in payload["reply_relevance_contract"]["coherent_option_summary_rule"]
    assert "tone only" in payload["style_examples_boundary"]
    assert "Never copy facts from examples" in payload["style_examples_boundary"]


def test_customer_visible_contract_snapshot_is_grouped_for_eval_and_docs() -> None:
    snapshot = customer_visible_contract_snapshot()

    assert snapshot["forbidden_customer_service_phrases"] == FORBIDDEN_CUSTOMER_SERVICE_PHRASES
    assert snapshot["preferred_requester_current_game_phrases"] == PREFERRED_REQUESTER_CURRENT_GAME_PHRASES
    assert snapshot["preferred_candidate_invite_phrases"] == PREFERRED_CANDIDATE_INVITE_PHRASES
    assert snapshot["preferred_operation_ack_phrases"] == PREFERRED_OPERATION_ACK_PHRASES


def test_customer_visible_text_contract_violations_group_shared_terms() -> None:
    violations = customer_visible_text_contract_violations("我是智能助手，已经生成草稿，asap_when_full 后等审批。")

    assert "implementation_identity_term:智能助手" in violations
    assert "internal_process_term:草稿" in violations
    assert "internal_process_term:审批" in violations
    assert "internal_enum:asap_when_full" in violations


def test_customer_visible_text_contract_normalizes_spacing_width_and_case() -> None:
    assert compact_customer_visible_text("Ａ I / 智能 助手 / asap when full") == "ai智能助手asapwhenfull"

    violations = customer_visible_text_contract_violations("我是Ａ I，不要写成智能 助手，asap when full 后等审 批。")

    assert "implementation_identity_term:AI" in violations
    assert "implementation_identity_term:智能助手" in violations
    assert "internal_process_term:审批" in violations
    assert "internal_enum:asap_when_full" in violations


def test_customer_visible_text_generation_rejects_rewrite_with_contract_terms() -> None:
    _, errors = parse_customer_visible_text_generation(
        """
        {
          "reasoning_summary": "模型误保留了内部词。",
          "item_rewrites": [
            {
              "item_id": "reply_to_user",
              "final_text": "我是智能助手，已经生成草稿，等审批。",
              "semantic_preserved": true,
              "used_facts": [],
              "withheld_facts": [],
              "style_checks": ["误判安全"],
              "change_summary": "错误改写"
            }
          ]
        }
        """,
        [{"item_id": "reply_to_user", "source": "reply_to_user", "text": "好，我帮你看看。"}],
    )

    assert any("violates customer-visible contract" in error for error in errors)
    assert any("智能助手" in error and "草稿" in error and "审批" in error for error in errors)
