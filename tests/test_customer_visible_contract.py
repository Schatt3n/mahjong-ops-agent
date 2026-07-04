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
    customer_visible_contract_snapshot,
)
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


def test_customer_visible_contract_snapshot_is_grouped_for_eval_and_docs() -> None:
    snapshot = customer_visible_contract_snapshot()

    assert snapshot["forbidden_customer_service_phrases"] == FORBIDDEN_CUSTOMER_SERVICE_PHRASES
    assert snapshot["preferred_requester_current_game_phrases"] == PREFERRED_REQUESTER_CURRENT_GAME_PHRASES
    assert snapshot["preferred_candidate_invite_phrases"] == PREFERRED_CANDIDATE_INVITE_PHRASES
    assert snapshot["preferred_operation_ack_phrases"] == PREFERRED_OPERATION_ACK_PHRASES
