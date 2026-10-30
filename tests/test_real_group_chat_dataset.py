from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_real_group_chat_dataset.py"
FLOW_EVAL_SCRIPT = ROOT / "scripts" / "run_real_group_chat_flow_eval.py"


def load_validator_module():
    spec = importlib.util.spec_from_file_location("validate_real_group_chat_dataset", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_flow_eval_module():
    spec = importlib.util.spec_from_file_location("run_real_group_chat_flow_eval", FLOW_EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {FLOW_EVAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_group_chat_datasets_are_anonymized_and_contract_valid() -> None:
    module = load_validator_module()

    records, errors = module.validate_datasets(module.DEFAULT_DATASET_PATHS)

    assert errors == []
    assert sum(record["quality_tier"] == "gold" for record in records) >= 12
    assert sum(record["quality_tier"] == "adversarial" for record in records) >= 4


def test_real_group_chat_gold_covers_key_production_behaviors() -> None:
    module = load_validator_module()
    records, errors = module.validate_datasets(module.DEFAULT_DATASET_PATHS)
    assert errors == []

    gold = [record for record in records if record["quality_tier"] == "gold"]
    case_types = {record["case_type"] for record in gold}
    assert {
        "owner_board_parse",
        "owner_board_snapshot",
        "owner_board_increment",
        "board_state_diff",
        "member_query",
        "fragmented_input",
        "quoted_state_update",
        "quick_filter",
        "message_revoke",
        "quoted_requirement_update",
    } <= case_types

    fragmented = next(record for record in gold if record["id"] == "real_group_fragmented_constraint_relaxation_001")
    assert fragmented["expected"]["search_requirement"]["accepted_stakes"] == ["1", "0.5"]
    urgent = next(record for record in gold if record["id"] == "real_group_fragmented_urgent_371_001")
    assert urgent["expected"]["current_players"] == 3
    assert urgent["expected"]["missing_players"] == 1


def test_validator_rejects_raw_identifiers_and_current_model_labels(tmp_path) -> None:
    module = load_validator_module()
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text(
        '{"schema_version":1,"kind":"real_group_chat_golden","id":"bad","quality_tier":"gold",'
        '"review_status":"approved","case_type":"query","source":{"channel":"wechat",'
        '"capture_date":"2026-07-22","room_alias":"room_alpha","source_refs":["sha256:1234567890abcdef"],'
        '"anonymized":true},"messages":[{"offset_seconds":0,"role":"customer","text":"x",'
        '"sender_id":"123456789012345678"}],"expected":{},"semantic_action":"process_business"}\n',
        encoding="utf-8",
    )

    _, errors = module.validate_datasets((bad_path,))
    text = "\n".join(errors)

    assert "raw observation/model-label keys" in text
    assert "raw long numeric identifier" in text


def test_adversarial_cases_cannot_be_promoted_without_resolving_open_questions() -> None:
    module = load_validator_module()
    records, errors = module.validate_datasets(module.DEFAULT_DATASET_PATHS)
    assert errors == []

    adversarial = [record for record in records if record["quality_tier"] == "adversarial"]
    assert all(record["review_status"] == "pending_domain_review" for record in adversarial)
    assert all(record["open_questions"] for record in adversarial)


def test_real_group_chat_gold_runs_through_deterministic_production_components() -> None:
    module = load_flow_eval_module()

    report = module.RealGroupChatFlowEvaluator(llm_client=None).evaluate(
        module.read_jsonl(module.DEFAULT_DATASET)
    )

    assert report["summary"] == {
        **report["summary"],
        "total": 12,
        "executed": 9,
        "passed": 9,
        "failed": 0,
        "skipped": 3,
        "pass_rate": 1.0,
    }


def test_real_group_chat_domain_codes_remain_separate_from_stakes() -> None:
    module = load_flow_eval_module()
    report = module.RealGroupChatFlowEvaluator(llm_client=None).evaluate(
        module.read_jsonl(module.DEFAULT_DATASET)
    )
    compact = next(item for item in report["cases"] if item["case_id"] == "real_group_compact_multi_board_001")
    items = compact["actual"]["board_items"]

    assert items[0]["game_type"] == "杭麻"
    assert items[0]["ruleset"] == "财敲"
    assert items[2]["game_type"] == "红中麻将"
    assert items[3]["rule_code"] == "368"
    assert items[3]["stake"] is None


def test_today_real_group_chat_gold_runs_through_production_components() -> None:
    module = load_flow_eval_module()
    dataset = ROOT / "eval" / "golden" / "real_group_chat_20260723.jsonl"

    report = module.RealGroupChatFlowEvaluator(llm_client=None).evaluate(
        module.read_jsonl(dataset),
        dataset_path=dataset,
    )

    assert report["dataset"] == str(dataset)
    assert report["summary"] == {
        **report["summary"],
        "total": 4,
        "executed": 4,
        "passed": 4,
        "failed": 0,
        "skipped": 0,
        "pass_rate": 1.0,
    }

    quoted = next(
        item for item in report["cases"]
        if item["case_id"] == "real_group_quoted_requirement_progress_20260723"
    )
    items = quoted["actual"]["board_items"]
    assert len(items) == 1
    assert {
        key: items[0].get(key)
        for key in (
            "participant_code",
            "current_players",
            "missing_players",
            "start_time",
            "stake",
            "smoking",
        )
    } == {
        "participant_code": "371",
        "current_players": 3,
        "missing_players": 1,
        "start_time": "23:00",
        "stake": "1",
        "smoking": "有烟",
    }


def test_20260726_real_group_chat_gold_runs_through_production_components() -> None:
    module = load_flow_eval_module()
    dataset = ROOT / "eval" / "golden" / "real_group_chat_20260726.jsonl"

    report = module.RealGroupChatFlowEvaluator(llm_client=None).evaluate(
        module.read_jsonl(dataset),
        dataset_path=dataset,
    )

    assert report["dataset"] == str(dataset)
    assert report["summary"] == {
        **report["summary"],
        "total": 7,
        "executed": 5,
        "passed": 5,
        "failed": 0,
        "skipped": 2,
        "pass_rate": 1.0,
    }

    compact = next(
        item for item in report["cases"]
        if item["case_id"] == "real_group_compact_cq_low_smoke_duration_20260726"
    )
    assert {
        key: compact["actual"]["board_items"][0].get(key)
        for key in (
            "game_type",
            "ruleset",
            "participant_code",
            "current_players",
            "missing_players",
            "stake",
            "smoking",
            "duration_hours",
        )
    } == {
        "game_type": "杭麻",
        "ruleset": "财敲",
        "participant_code": "272",
        "current_players": 2,
        "missing_players": 2,
        "stake": "0.5",
        "smoking": "少烟",
        "duration_hours": 5.0,
    }

    rule_code = next(
        item for item in report["cases"]
        if item["case_id"] == "real_group_red_rule_code_368_20260726"
    )
    parsed = rule_code["actual"]["board_items"][0]
    assert parsed["game_type"] == "红中麻将"
    assert parsed["rule_code"] == "368"
    assert parsed["stake"] is None

    progress = next(
        item for item in report["cases"]
        if item["case_id"] == "real_group_quoted_progress_272_to_371_20260726"
    )
    assert len(progress["actual"]["board_items"]) == 1
    assert progress["actual"]["board_items"][0]["participant_code"] == "371"

    resolved = next(
        item for item in report["cases"]
        if item["case_id"] == "real_group_quoted_full_scheduled_20260726"
    )
    assert resolved["actual"]["board_items"][0]["status"] == "full"
    assert resolved["actual"]["board_items"][0]["start_time"] == "22:00"


def test_real_group_chat_eval_accepts_semantically_equivalent_domain_labels() -> None:
    module = load_flow_eval_module()

    assert module.subset_errors(
        {"stakes": "1", "smoking": "无烟"},
        {"stakes": "1块", "smoking": "no"},
        path="classification.extracted_features",
    ) == []


@pytest.mark.parametrize("value", ["一块", "一元", "一 块", "1", "1.0", "1块", "1元"])
def test_group_eval_compares_complete_monetary_labels_by_existing_amount_protocol(value) -> None:
    module = load_flow_eval_module()
    assert module.normalize_number(value) == "1"
    assert module.subset_errors({"stakes": "1"}, {"stakes": value}, path="classification.extracted_features") == []


@pytest.mark.parametrize("value,expected", [("两块", "2"), ("十元", "10"), ("0.5", "0.5"), ("0.5块", "0.5")])
def test_group_eval_keeps_different_amounts_distinct(value, expected) -> None:
    module = load_flow_eval_module()
    assert module.normalize_number(value) == expected
    assert module.subset_errors({"stakes": "1"}, {"stakes": value}, path="classification.extracted_features")


@pytest.mark.parametrize("value", [
    "1-3", "1-256", "一到三块", "一块都行", "随便一块", "十一十块", "opaque", "一元块",
])
def test_group_eval_never_extracts_a_single_amount_from_ranges_caps_or_unrecognized_labels(value) -> None:
    module = load_flow_eval_module()
    assert module.normalize_number(value) != "1"
    # 既有单位清理可保留未知文本，但不能扫描其中的中文金额片段来接受整个表达。
    assert module.normalize_number(value) == value.strip().removesuffix("块").removesuffix("元")
    assert module.subset_errors({"stakes": "1"}, {"stakes": value}, path="classification.extracted_features")


@pytest.mark.parametrize("classification_stake,accepted", [("一块", True), ("一元", True), ("两块", False)])
def test_original_one_yuan_group_case_accepts_equivalent_classifier_text_but_rejects_wrong_amount(
    classification_stake, accepted,
) -> None:
    module = load_flow_eval_module()
    dataset = ROOT / "eval" / "golden" / "real_group_chat_20260726.jsonl"
    original_bytes = dataset.read_bytes()
    case = next(row for row in module.read_jsonl(dataset)
                if row["id"] == "real_group_member_query_one_yuan_no_smoke_20260726")
    assert case["messages"][0]["text"] == "一块无烟有没有"
    actual = {
        "board_items": [],
        "session_outcomes": [{"classification": {
            "intent": "query", "channel_action": "group_reply",
            "extracted_features": {"stakes": classification_stake, "smoking": "无烟", "game_type": None, "time": None},
        }}],
        "main_agent": {
            "status": "waiting_user", "final_reply": "一块无烟的，想打什么玩法？几点打？",
            "tool_calls": ["search_current_games"],
            "tool_results": [{"name": "search_current_games", "called": True, "allowed": True,
                              "result": {"requirement": {"stake": "1", "smoke_preference": "无烟"}}}],
        },
    }
    errors = module.RealGroupChatFlowEvaluator._check_case(
        case, expected=case["expected"], actual=actual, before_snapshots=[], pipeline=None,
    )
    if accepted:
        assert errors == []
    else:
        assert errors == [f"classification.extracted_features.stakes: expected '1', got {classification_stake!r}"]
    assert dataset.read_bytes() == original_bytes



@pytest.mark.parametrize("amounts,accepted", [
    (["1块", "0.5"], True), (["一元", "0.5块"], True),
    (["两块", "0.5"], False), (["0.5", "1块"], False), (["1块"], False),
    (["1-3", "0.5"], False), (["1-256", "0.5"], False), (["opaque", "0.5"], False),
])
def test_group_eval_accepted_stakes_normalizes_only_complete_amount_elements(amounts, accepted) -> None:
    module = load_flow_eval_module()
    before = list(amounts)
    errors = module.subset_errors(
        {"accepted_stakes": ["1", "0.5"]}, {"accepted_stakes": amounts},
        path="classification.extracted_features",
    )
    assert (errors == []) is accepted
    assert amounts == before
    assert len(amounts) == len(before)


@pytest.mark.parametrize("path", [
    "classification.extracted_features.time[0]",
    "classification.extracted_features.rule_code[0]",
    "classification.extracted_features.participant_code[0]",
    "classification.extracted_features.accepted_stakes_extra[0]",
    "classification.extracted_features.accepted_stakes[0]tail",
    "classification.extracted_features.accepted_stakes[]",
    "classification.extracted_features.accepted_stakes[-1]",
    "classification.extracted_features.accepted_stakes[0][1]",
])
def test_group_eval_does_not_normalize_non_amount_or_malformed_list_paths(path) -> None:
    module = load_flow_eval_module()
    assert module.normalize_semantic_value("1块", path=path) == "1块"


@pytest.mark.parametrize("first_amount,accepted", [("1块", True), ("一元", True), ("两块", False)])
def test_original_fragmented_group_case_accepts_equivalent_amount_options_after_recovered_query(
    first_amount, accepted,
) -> None:
    module = load_flow_eval_module()
    dataset = ROOT / "eval" / "golden" / "real_group_chat_20260722.jsonl"
    original_bytes = dataset.read_bytes()
    case = next(row for row in module.read_jsonl(dataset)
                if row["id"] == "real_group_fragmented_constraint_relaxation_001")
    # 原始人工期望保留金额偏好顺序，不能改成参考当前模型输出的新标准。
    assert case["expected"]["search_requirement"]["accepted_stakes"] == ["1", "0.5"]
    assert case["expected"]["preference_order"] == ["1", "0.5"]
    actual = {
        "board_items": [], "session_count": 1, "main_agent_invocation_count": 1,
        "session_outcomes": [{"classification": {
            "intent": "query", "channel_action": "group_reply",
            "extracted_features": {"accepted_stakes": [first_amount, "0.5"], "smoking": "无烟"},
        }}],
        "main_agent": {
            "status": "waiting_user", "tool_calls": ["search_current_games"],
            "tool_results": [
                {"name": "search_current_games", "called": False, "allowed": False, "error": "invalid arguments"},
                {"name": "search_current_games", "called": True, "allowed": True, "error": None,
                 "result": {"requirement": {"stakes": [1, 0.5], "smoke_preference": "无烟"}}},
            ],
        },
    }
    successful_queries = [item for item in actual["main_agent"]["tool_results"]
                          if item["name"] == "search_current_games" and item["called"] and item["allowed"] and not item["error"]]
    assert len(successful_queries) == 1
    errors = module.RealGroupChatFlowEvaluator._check_case(
        case, expected=case["expected"], actual=actual, before_snapshots=[], pipeline=None,
    )
    if accepted:
        assert errors == []
    else:
        assert errors == [f"classification.extracted_features.accepted_stakes[0]: expected '1', got {first_amount!r}"]
    assert dataset.read_bytes() == original_bytes
