"""审查传输投影只消除精确副本，完整事实可逆且原证据不变。"""

from copy import deepcopy
import json

import pytest

from mahjong_agent_runtime import AgentAction, UserMessage
from mahjong_agent_runtime.customer_visible_review import (
    build_reply_self_review_payload, expand_review_transport_references, project_review_transport_payload,
)
from mahjong_agent_runtime.domains.query_contract import game_query_strategy_contract


def source_payload():
    contract = game_query_strategy_contract()
    failed = {"name": "create_game", "called": False, "allowed": False, "call_id": "denied",
              "error": "rejected", "deduplicated": False,
              "result": {"instruction": "原限制仍有效", "known_facts": {"seat_count": 3}}}
    read = {"name": "search_current_games", "called": True, "allowed": True, "call_id": "read",
            "error": None, "deduplicated": False, "result": {"matches": [], "requirement": {"stake": "1"},
                "customer_reply_contract": {"evidence_scope": {"supports_narrower_empty_claim": True},
                    "game_query_strategy_contract": contract}}}
    return {"game_query_strategy_contract": contract, "previous_tool_results": [failed, read],
            "turn_tool_evidence": [failed, read], "recent_conversation_tail": [
                {"role": "tool", "content": json.dumps([read], ensure_ascii=False)},
                {"role": "user", "content": json.dumps([read], ensure_ascii=False)}]}


def test_review_transport_roundtrip_preserves_order_failure_history_and_source():
    source = source_payload()
    snapshot = deepcopy(source)
    projected = project_review_transport_payload(source)
    assert source == snapshot
    assert len(json.dumps(projected, ensure_ascii=False)) < len(json.dumps(source, ensure_ascii=False)) * .7
    assert [row["result_reference"]["index"] for row in projected["previous_tool_results"]] == [0, 1]
    assert projected["turn_tool_evidence"][0] == source["turn_tool_evidence"][0]
    assert projected["turn_tool_evidence"][1]["result"]["customer_reply_contract"]["game_query_strategy_contract"] == {
        "contract_reference": "game_query_strategy_contract"}
    assert projected["recent_conversation_tail"][1] == source["recent_conversation_tail"][1]
    assert expand_review_transport_references(projected) == source
    assert project_review_transport_payload(source) == projected


def test_review_builder_projects_after_privacy_filtering_without_mutating_context():
    source = source_payload()
    context = {"previous_tool_results": source["previous_tool_results"],
               "turn_tool_evidence": source["turn_tool_evidence"],
               "recent_conversation": source["recent_conversation_tail"]}
    snapshot = deepcopy(context)
    message = UserMessage(conversation_id="own", sender_id="customer", sender_name="本人", text="只看一下")
    action = AgentAction(goal="告知查询结果", objective_status="completed", reasoning_summary="如实告知", reply_to_user="暂时没查到。")
    payload = build_reply_self_review_payload(message=message, action=action, context_payload=context,
        review_scope="reply_to_user", review_items=[{"recipient_id": "customer", "text": action.reply_to_user}])
    assert context == snapshot
    assert payload["previous_tool_results"][0]["result_reference"]["index"] == 0
    expanded = expand_review_transport_references(payload)
    assert expanded["previous_tool_results"] == context["previous_tool_results"]
    assert expanded["turn_tool_evidence"] == context["turn_tool_evidence"]


def test_distinct_history_contracts_and_boolean_number_facts_are_not_collapsed():
    source = source_payload()
    source["previous_tool_results"] = deepcopy(source["previous_tool_results"])
    source["previous_tool_results"][0]["result"]["known_facts"]["seat_count"] = True
    source["turn_tool_evidence"][0]["result"]["known_facts"]["seat_count"] = 1
    old = json.loads(source["recent_conversation_tail"][0]["content"])
    old[0]["result"]["customer_reply_contract"]["game_query_strategy_contract"]["query_result_reporting"] = "旧版限定范围"
    source["recent_conversation_tail"][0]["content"] = json.dumps(old, ensure_ascii=False)
    projected = project_review_transport_payload(source)
    assert "result_reference" not in projected["previous_tool_results"][0]
    assert json.loads(projected["recent_conversation_tail"][0]["content"]) == old
    assert expand_review_transport_references(projected) == source


@pytest.mark.parametrize("shape", ["list", "object", "json_object"])
def test_structured_tool_history_is_losslessly_projected_without_touching_user_text(shape):
    source = source_payload()
    result = deepcopy(source["turn_tool_evidence"][1])
    source["recent_conversation_tail"][0]["content"] = (
        [result] if shape == "list" else result if shape == "object" else json.dumps(result, ensure_ascii=False))
    projected = project_review_transport_payload(source)
    assert expand_review_transport_references(projected) == source
    assert projected["recent_conversation_tail"][0] != source["recent_conversation_tail"][0]
    assert projected["recent_conversation_tail"][1] == source["recent_conversation_tail"][1]


@pytest.mark.parametrize("fault", ["index", "boolean_index", "source", "call_id", "shell", "contract", "missing_anchor", "cycle"])
def test_invalid_transport_references_cannot_be_used_as_full_gate_evidence(fault):
    projected = project_review_transport_payload(source_payload())
    reference = projected["previous_tool_results"][0]["result_reference"]
    if fault == "index":
        reference["index"] = 99
    elif fault == "boolean_index":
        reference["index"] = False
    elif fault == "source":
        reference["source"] = "another_conversation"
    elif fault == "call_id":
        reference["call_id"] = "unrelated"
    elif fault == "shell":
        projected["previous_tool_results"][0]["allowed"] = True
    elif fault == "contract":
        projected["turn_tool_evidence"][1]["result"]["customer_reply_contract"]["game_query_strategy_contract"] = {
            "contract_reference": "another_contract"}
    elif fault == "missing_anchor":
        del projected["game_query_strategy_contract"]
    else:
        projected["turn_tool_evidence"][0] = deepcopy(projected["previous_tool_results"][0])
    with pytest.raises(ValueError):
        expand_review_transport_references(projected)
