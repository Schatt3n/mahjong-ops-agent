"""固定审查对照集保留整句语义差异，测试只校验资产合同，不冒充真实模型判定。"""
from __future__ import annotations

import json
from pathlib import Path
import re


DATASET = Path(__file__).resolve().parents[1] / "eval" / "golden" / "customer_visible_review_contract.jsonl"
CASE_IDS = {
    "review_follow_up_offer_question", "review_ask_people_offer_question", "review_conditional_follow_up_offer",
    "review_grounded_empty_query", "review_unarranged_follow_up_promise", "review_fabricated_contact_completed",
    "review_offer_with_false_sent_claim", "review_offer_with_unconditional_follow_up",
}


def read_records():
    return [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]


def test_customer_visible_review_dataset_has_fixed_unique_source_ids_and_both_verdicts():
    records = read_records()
    assert len(records) == 8
    assert {record["case_id"] for record in records} == CASE_IDS
    assert len({record["case_id"] for record in records}) == len(records)
    assert sum(record["expected_approved"] is True for record in records) == 4
    assert sum(record["expected_approved"] is False for record in records) == 4
    for record in records:
        assert set(record) == {"case_id", "text", "expected_approved", "requirement", "source"}
        assert type(record["expected_approved"]) is bool
        assert isinstance(record["text"], str) and record["text"]
        assert record["requirement"] == {"game_type": "red_center_mahjong", "stake_range": {"min": 1, "max": 3}, "smoke_preference": "any"}
        assert record["source"]["not_original_human_transcript"] is True
        assert record["source"]["kind"] in {"observed_failure_regression", "synthetic_regression_fixture"}
        assert set(record).isdisjoint({"tool_results", "state", "persisted_state", "approved", "raw_response"})


def test_customer_visible_review_dataset_preserves_observed_question_and_independent_clause_contrasts():
    records = {record["case_id"]: record for record in read_records()}
    assert records["review_follow_up_offer_question"]["text"] == "下午这会儿暂时没有合适的局，要我帮你留意着，有合适的叫你？"
    assert records["review_follow_up_offer_question"]["expected_approved"] is True
    assert records["review_follow_up_offer_question"]["source"]["verbatim_observed_text"] is True
    assert records["review_follow_up_offer_question"]["source"]["origin_scenario_id"] == "profile_range_empty_inquiry"
    assert records["review_unarranged_follow_up_promise"]["text"] == "我帮你留意着，有合适的叫你。"
    assert records["review_unarranged_follow_up_promise"]["expected_approved"] is False
    assert records["review_fabricated_contact_completed"]["text"] == "已经问好人了。"
    assert records["review_offer_with_false_sent_claim"]["text"] == "要我问吗？我已发消息。"
    assert records["review_offer_with_false_sent_claim"]["expected_approved"] is False
    assert records["review_offer_with_unconditional_follow_up"]["expected_approved"] is False
    assert records["review_grounded_empty_query"]["expected_approved"] is True


def test_customer_visible_review_dataset_has_no_credentials_or_local_paths():
    serialized = DATASET.read_text()
    assert not re.search(r"(?:sk-[A-Za-z0-9]{16,}|Bearer\s+\S+|/Users/|/var/folders/)", serialized)
