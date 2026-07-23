from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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


def test_real_group_chat_eval_accepts_semantically_equivalent_domain_labels() -> None:
    module = load_flow_eval_module()

    assert module.subset_errors(
        {"stakes": "1", "smoking": "无烟"},
        {"stakes": "1块", "smoking": "no"},
        path="classification.extracted_features",
    ) == []
