"""用户新话术口径必须显式版本化，保留旧禁词记录和所有事实、工具与安全断言。"""

from copy import deepcopy
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_product_contracts as policy
import run_real_owner_chat_live_eval as owner


def records_and_revision():
    """读取原资产及追加声明，不用当前模型输出生成或反写期望。"""
    records = policy.read_product_records()
    record = next(item for item in records if item.get("kind") == policy.REPLY_PRESENTATION_REVISION_KIND)
    return records, record, record["revisions"][0]


def test_presentation_revision_keeps_original_bytes_and_factual_requirements():
    records, record, revision = records_and_revision()
    raw = policy.DATASET_PATH.read_bytes()
    assert hashlib.sha256(raw[:record["original_dataset_prefix_bytes"]]).hexdigest() == record["original_dataset_prefix_sha256"]
    old = revision["previous_expectation"]
    assert "0.5" in old["forbidden_reply_contains"] and "无烟" in old["forbidden_reply_contains"]
    case = next(item for item in owner.golden_reply_cases() if item["case_id"] == revision["target_case_id"])
    assert set(old["forbidden_reply_contains"]) - set(case["expected"]["forbidden_reply_contains"]) == {"0.5", "无烟"}
    for key in set(old) - {"forbidden_reply_contains"}:
        assert case["expected"][key] == old[key]
    assert case["product_contract_revision_ids"] == [revision["revision_id"]]
    scene = next(item for item in owner.live_eval_scenarios() if item.scenario_id == revision["live_scenario_id"])
    assert scene.required_tool_names == ["search_current_games"]
    assert scene.expected_tool_result_paths["search_current_games"]["result.requirement.stake"] == "0.5"
    assert "打吗" in scene.required_reply_any[2]
    assert "候选人" in scene.forbidden_reply_contains
    assert "系统" in scene.forbidden_reply_contains


@pytest.mark.parametrize("fault", ["provenance", "prefix", "case_hash", "fact", "privacy", "time", "duplicates"])
def test_presentation_revision_rejects_unrelated_relaxation_or_missing_source(fault):
    records, record, revision = records_and_revision()
    if fault == "provenance":
        record["source"]["user_quotes"] = []
    elif fault == "prefix":
        record["original_dataset_prefix_sha256"] = "0" * 64
    elif fault == "case_hash":
        revision["source_case_sha256"] = "0" * 64
    elif fault == "fact":
        revision["current_expectation"]["search_requirement_should_include"]["stake"] = "1"
    elif fault == "privacy":
        revision["current_expectation"]["forbidden_reply_contains"].remove("候选人")
    elif fault == "time":
        revision["current_scenario_contract"]["required_reply_any"][0] = ["八点"]
    else:
        record["revisions"].append(deepcopy(revision))
    with pytest.raises(ValueError):
        policy.validate_reply_presentation_policy(records)


def test_catalog_and_effective_expectation_cannot_drift_apart():
    records, _, revision = records_and_revision()
    current = next(item for item in owner.live_eval_scenarios() if item.scenario_id == revision["live_scenario_id"])
    # 当前catalog已经应用修订，不能再次应用并把不匹配的前态默认为可信旧基线。
    with pytest.raises(ValueError, match="catalog drifted"):
        policy.apply_reply_presentation_to_scenarios([current], records)
    cases = owner.golden_reply_cases()
    with pytest.raises(ValueError, match="golden link"):
        policy.apply_reply_presentation_to_cases(cases, records)
