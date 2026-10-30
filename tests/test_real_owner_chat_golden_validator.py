from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_real_owner_chat_golden.py"


def load_validator_module():
    spec = importlib.util.spec_from_file_location("validate_real_owner_chat_golden", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_owner_chat_golden_dataset_validator_passes_current_dataset() -> None:
    module = load_validator_module()

    records = module.read_default_records()
    errors = module.validate_dataset(records)

    assert errors == []
    assert {
        "owner_chat_multi_episode_timeline_20260705_20260718_001",
        "owner_b_chat_multi_episode_timeline_20260627_20260719_001",
    }.issubset({record["id"] for record in records})
    assert len(module.DEFAULT_TIMELINE_PATHS) >= 2


def _semantic_expectation_record():
    return {
        "kind": "real_owner_chat_golden", "id": "semantic_contract_fixture",
        "messages": [{"turn": 1, "role": "customer", "text": "1块的吧时间都行"}],
        "eval_cases": [{
            "id": "semantic_case", "input_turn": 1,
            "live_scenario_id": "flexible_time_empty_pool_offers_help",
            "expected": {
                "reply_reference": "要我帮你问问有没有人吗？",
                "required_meaning": [{"id": "offer_help", "description": "提议帮忙问人并等待客户决定。"}],
                "forbidden_meaning": [{"id": "no_fake_contact", "description": "无真实外发证据不得声称已联系。"}],
            },
        }],
    }


def test_semantic_expectation_dataset_links_all_live_cases_without_rewriting_legacy() -> None:
    module = load_validator_module()
    records = module.read_jsonl(module.DEFAULT_DATASET_PATH)
    record = next(item for item in records if item["id"] == "owner_chat_flexible_time_and_help_offer_20261029_001")
    assert record["source"]["trace_id"] == "trace_2ea5c2650520"
    assert record["source"]["observed_date"] == "2026-10-29"
    assert [message["text"] for message in record["messages"]] == ["下午有人吗", "下午想几点打打多大", "1块的吧时间都行"]
    assert {case["live_scenario_id"] for case in record["eval_cases"]} == {
        "flexible_time_followup_matches_evening_game", "flexible_time_empty_pool_offers_help",
        "inventory_only_does_not_offer_recruitment", "complete_query_is_not_creation_consent",
    }
    for case in record["eval_cases"]:
        assert case["expected"]["reply_reference_is_example"] is True
        if "input" in case:
            assert case["input"]["derived"] is True
    for legacy in records[:2]:
        assert module._validate_semantic_eval_expectations(legacy, set()) == []


@pytest.mark.parametrize("field,value,error", [
    ("reply_reference", " ", "reply_reference must be a non-empty string"),
    ("reply_reference", ["示例"], "reply_reference must be a non-empty string"),
    ("required_meaning", [], "required_meaning must be a non-empty list"),
    ("forbidden_meaning", "不能伪造", "forbidden_meaning must be a non-empty list"),
    ("required_meaning", ["必须提供服务提议"], "required_meaning item must be an object"),
    ("required_meaning", [{"id": "", "description": "不能为空ID"}], "required_meaning id must be a non-empty string"),
    ("forbidden_meaning", [{"id": "no_fake_contact", "description": ""}], "forbidden_meaning description must be a non-empty string"),
])
def test_semantic_expectation_validator_rejects_malformed_meanings(field, value, error) -> None:
    module = load_validator_module()
    record = _semantic_expectation_record()
    record["eval_cases"][0]["expected"][field] = value
    assert error in "\n".join(module.validate_dataset([record]))


def test_semantic_expectation_validator_rejects_duplicate_ids_and_unknown_live_reference() -> None:
    module = load_validator_module()
    record = _semantic_expectation_record()
    case = record["eval_cases"][0]
    case["expected"]["forbidden_meaning"][0]["id"] = "offer_help"
    case["live_scenario_id"] = "nonexistent_live_case"
    record["eval_cases"].append(deepcopy(case))
    errors = "\n".join(module.validate_dataset([record]))
    assert "duplicate meaning id: offer_help" in errors
    assert "duplicate semantic eval case id: semantic_case" in errors
    assert "live_scenario_id not found: nonexistent_live_case" in errors


def test_semantic_expectation_validator_requires_complete_contract_when_live_link_declared() -> None:
    module = load_validator_module()
    record = _semantic_expectation_record()
    case = record["eval_cases"][0]
    case["live_scenario_id"] = ""
    case["expected"] = None
    errors = "\n".join(module.validate_dataset([record]))
    assert "live_scenario_id must be a non-empty string" in errors
    assert "semantic expected must be an object" in errors


def _synthetic_semantic_record():
    module = load_validator_module()
    record = deepcopy(next(record for record in module.read_jsonl(module.DEFAULT_DATASET_PATH)
                           if record["kind"] == "owner_chat_live_semantic_regression"))
    record["eval_cases"] = record["eval_cases"][:1]
    return record


@pytest.mark.parametrize("path,value,error", [
    (("schema_version",), True, "synthetic schema_version must be integer 1"),
    (("source", "kind"), "real_owner_chat", "synthetic source must declare"),
    (("source", "not_original_human_transcript"), "true", "synthetic source must declare"),
    (("eval_cases",), 1, "synthetic eval_cases must be a non-empty list"),
    (("eval_cases",), [], "synthetic eval_cases must be a non-empty list"),
    (("eval_cases",), ["bad"], "synthetic eval case must be an object"),
    (("eval_cases", 0, "source"), {}, "synthetic case source is required"),
    (("eval_cases", 0, "input"), None, "input must be an object"),
    (("eval_cases", 0, "input", "text"), " ", "input.text must be a non-empty string"),
    (("eval_cases", 0, "input", "sender_id"), None, "input.sender_id must be a non-empty string"),
    (("eval_cases", 0, "input", "quoted_message"), {"text": ""}, "input.quoted_message.text must be a non-empty string"),
    (("eval_cases", 0, "context"), [], "context must be an object"),
    (("eval_cases", 0, "context", "fixture_ref"), "", "context.fixture_ref must match the live scenario setup"),
    (("eval_cases", 0, "context", "fixture_ref"), "scripts/run_real_owner_chat_live_eval.py::setup_no_game_profile", "context.fixture_ref must match the live scenario setup"),
    (("eval_cases", 0, "context", "known"), [], "context.known must be a non-empty object"),
    (("eval_cases", 0, "context", "unknown"), [1], "context.unknown must be a non-empty list of strings"),
    (("eval_cases", 0, "existing_hard_assertions_snapshot"), {}, "existing_hard_assertions_snapshot must be a non-empty object"),
])
def test_synthetic_semantic_record_rejects_missing_or_forged_context(path, value, error) -> None:
    module = load_validator_module()
    record = _synthetic_semantic_record()
    target = record
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    assert error in "\n".join(module.validate_dataset([record]))


def test_synthetic_semantics_cannot_skip_contract_or_duplicate_source_identity() -> None:
    module = load_validator_module()
    record = _synthetic_semantic_record()
    case = record["eval_cases"][0]
    case.pop("live_scenario_id")
    case.pop("expected")
    errors = "\n".join(module.validate_dataset([record]))
    assert "live_scenario_id must be a non-empty string" in errors
    assert "semantic expected must be an object" in errors
    first = _synthetic_semantic_record()
    second = deepcopy(first)
    second["id"] += "_second"
    errors = "\n".join(module.validate_dataset([first, second]))
    assert "duplicate dataset eval case id" in errors
    assert "missing or duplicate record id" in "\n".join(module.validate_dataset([first, deepcopy(first)]))


def test_synthetic_record_does_not_invent_human_style_evidence() -> None:
    module = load_validator_module()
    record = _synthetic_semantic_record()
    record["eval_cases"][0]["expected"]["good_examples"] = ["合成参考不能当原始真人记录"]
    assert module._collect_style_evidence_texts([record]) == []


def test_real_owner_chat_golden_validator_rejects_broken_turn_references() -> None:
    module = load_validator_module()
    transcript = {
        "kind": "real_owner_chat_golden",
        "id": "broken_transcript",
        "source": {"source_image_files": ["one.png"]},
        "messages": [
            {
                "turn": 1,
                "role": "customer",
                "text": "帮我约个6.30无烟的",
                "source_image": "one.png",
            }
        ],
        "business_facts": [{"id": "broken_fact", "evidence_turns": [2], "fact": "bad"}],
        "eval_cases": [{"id": "broken_case", "context_turns": [1, 3], "input_turn": 4}],
    }

    errors = module.validate_dataset([transcript])

    assert "broken_fact" in "\n".join(errors)
    assert "invalid context_turns [3]" in "\n".join(errors)
    assert "invalid input_turn 4" in "\n".join(errors)


def test_real_owner_chat_golden_validator_rejects_broken_timeline_metadata() -> None:
    module = load_validator_module()
    transcript = {
        "kind": "real_owner_chat_golden",
        "id": "broken_timeline",
        "source": {"source_image_files": ["one.png"]},
        "messages": [
            {"turn": 1, "role": "customer", "text": "七点无烟还有吗", "source_image": "one.png"},
            {"turn": 2, "role": "boss", "text": "有的", "source_image": "one.png"},
        ],
        "episodes": [
            {
                "id": "episode_1",
                "start_turn": 1,
                "end_turn": 1,
                "business_exchange_rounds": 0,
                "outcome": "",
            }
        ],
        "timeline_metrics": {
            "message_count": 3,
            "customer_message_count": 2,
            "boss_message_count": 1,
            "episode_count": 2,
            "round_definition": "",
        },
        "replay_scenarios": [
            {
                "id": "replay_1",
                "events": [{"actor": "customer", "text": "七点无烟还有吗", "source_turn": 9}],
                "expected": {},
            }
        ],
    }

    errors = module.validate_dataset([transcript])
    text = "\n".join(errors)

    assert "episode turn ranges must partition all messages in order" in text
    assert "business_exchange_rounds must be a positive integer" in text
    assert "timeline_metrics.message_count must be 2" in text
    assert "timeline_metrics.customer_message_count must be 1" in text
    assert "timeline_metrics.round_definition is required" in text
    assert "source_turn not found: 9" in text
    assert "expected is required" in text


def test_real_owner_chat_golden_validator_rejects_broken_source_event_references() -> None:
    module = load_validator_module()
    transcript = {
        "kind": "real_owner_chat_golden",
        "id": "broken_source_event",
        "source": {"source_image_files": ["one.png"]},
        "messages": [{"turn": 1, "role": "customer", "text": "有五毛的吗", "source_image": "one.png"}],
        "source_events": [
            {
                "id": "recalled_1",
                "event_type": "message_recalled",
                "actor": "customer",
                "content_available": "false",
                "source_image": "missing.png",
            }
        ],
        "business_facts": [
            {"id": "fact", "evidence_turns": [1], "fact": "ok", "source_event_ids": ["missing_event"]}
        ],
        "eval_cases": [
            {
                "id": "case",
                "context_turns": [],
                "input_turn": 1,
                "source_event_refs": ["missing_event"],
            }
        ],
    }

    errors = module.validate_dataset([transcript])
    text = "\n".join(errors)

    assert "content_available must be boolean" in text
    assert "source_image not listed" in text
    assert "source_event_id not found: missing_event" in text
    assert "source_event_ref not found: missing_event" in text


def test_real_owner_chat_golden_validator_requires_supplement_contract() -> None:
    module = load_validator_module()
    transcript = {
        "kind": "real_owner_chat_golden",
        "id": "parent",
        "source": {"source_image_files": ["one.png"]},
        "messages": [{"turn": 1, "role": "customer", "text": "x", "source_image": "one.png"}],
        "business_facts": [{"id": "fact", "evidence_turns": [1], "fact": "ok"}],
        "eval_cases": [],
    }
    supplement = {
        "kind": "real_owner_chat_eval_supplement",
        "id": "supplement",
        "parent_id": "parent",
        "hidden_context": [{"after_turn": 99, "event": "bad"}],
        "customer_profile_assumptions": {"profile_facts": ["只有一条不完整画像"]},
        "business_facts": [{"id": "profile_defaults_fill_missing_slots"}],
        "eval_cases": [{"id": "initial_request_uses_profile_defaults_and_searches_pool", "hidden_context_refs": ["missing"]}],
    }

    errors = module.validate_dataset([transcript, supplement])
    text = "\n".join(errors)

    assert "missing required business facts" in text
    assert "missing required eval cases" in text
    assert "after_turn not found" in text
    assert "missing profile assumption" in text
    assert "hidden_context_ref not found" in text


def test_real_owner_chat_golden_validator_checks_copywriting_style_examples(monkeypatch) -> None:
    module = load_validator_module()
    transcript = {
        "kind": "real_owner_chat_golden",
        "id": "parent",
        "source": {"source_image_files": ["one.png"]},
        "messages": [
            {"turn": 1, "role": "customer", "text": "帮我约个6.30无烟的", "source_image": "one.png"},
            {"turn": 2, "role": "boss", "text": "七点三缺一", "source_image": "one.png"},
            {"turn": 3, "role": "boss", "text": "可以不", "source_image": "one.png"},
        ],
        "business_facts": [{"id": "fact", "evidence_turns": [1], "fact": "ok"}],
        "eval_cases": [],
    }
    monkeypatch.setattr(
        module,
        "REAL_OWNER_WECHAT_STYLE_EXAMPLES",
        (
            {
                "scenario": "matched game",
                "good": "七点三缺一，可以不？",
                "bad": "已为您查询到一个局，请问是否加入？",
                "style_note": "短句",
            },
        ),
    )

    errors = module._validate_real_owner_style_examples([transcript])

    assert errors == []


def test_real_owner_chat_golden_validator_rejects_unbacked_copywriting_style_example(monkeypatch) -> None:
    module = load_validator_module()
    monkeypatch.setattr(
        module,
        "REAL_OWNER_WECHAT_STYLE_EXAMPLES",
        (
            {
                "scenario": "fake",
                "good": "这句不是老板说过的话",
                "bad": "已为您处理",
                "style_note": "bad",
            },
        ),
    )

    errors = module._validate_real_owner_style_examples([])

    assert "not backed by real owner dataset" in "\n".join(errors)


@pytest.mark.parametrize("history,error", [
    (None, "hard_assertions_history must be a non-empty list"),
    ({}, "hard_assertions_history must be a non-empty list"),
    ([], "hard_assertions_history must be a non-empty list"),
    (["bad"], "hard assertion revision must be an object"),
    ([{}], "hard assertion revision_id must be a non-empty string"),
    ([{"revision_id": "r", "reason": " ", "source_run_id": "full8", "previous_snapshot": {"required_tool_names": []}}], "hard assertion reason must be a non-empty string"),
    ([{"revision_id": "r", "reason": "澄清合法", "source_run_id": None, "previous_snapshot": {"required_tool_names": []}}], "hard assertion source_run_id must be a non-empty string"),
    ([{"revision_id": "r", "reason": "澄清合法", "source_run_id": "full8", "previous_snapshot": {}}], "hard assertion previous_snapshot must be a non-empty object"),
    ([{"revision_id": "r", "reason": "澄清合法", "source_run_id": "full8", "previous_snapshot": []}], "hard assertion previous_snapshot must be a non-empty object"),
])
def test_hard_assertion_history_validator_rejects_untraceable_revision(history, error) -> None:
    module = load_validator_module()
    record = _synthetic_semantic_record()
    record["eval_cases"][0]["hard_assertions_history"] = history
    assert error in "\n".join(module.validate_dataset([record]))


def test_hard_assertion_history_retains_original_contract_without_matching_current_fixture() -> None:
    """历史的必查要求与现合同允许澄清可以并存，但修订 ID 不可重复。"""
    module = load_validator_module()
    record = _synthetic_semantic_record()
    case = record["eval_cases"][0]
    revision = {"revision_id": "allow-clarification", "reason": "人工期望允许澄清未知底注。",
                "source_run_id": "full8", "previous_snapshot": {"required_tool_names": ["search_current_games"]}}
    case["hard_assertions_history"] = [revision]
    assert module._validate_hard_assertions_history(record, case) == []
    case["hard_assertions_history"].append(deepcopy(revision))
    assert "duplicate hard assertion revision_id" in "\n".join(module._validate_hard_assertions_history(record, case))


def test_profile_query_sources_cover_clarification_and_information_sufficient_query_cases() -> None:
    module = load_validator_module()
    records = module.read_jsonl(module.DEFAULT_DATASET_PATH)
    synthetic = next(record for record in records if record["kind"] == "owner_chat_live_semantic_regression")
    assert len(synthetic["eval_cases"]) == 19
    cases = {case["live_scenario_id"]: case for case in synthetic["eval_cases"]}
    for prefix in ("single", "custom"):
        complete = cases[f"{prefix}_profile_game_complete_query"]
        assert complete["input"]["text"] == "我一个人，1块，烟不限，现在有局吗？"
        assert complete["existing_hard_assertions_snapshot"]["required_tool_names"] == ["search_current_games"]
        assert complete["existing_hard_assertions_snapshot"]["expected_status_any"] == ["waiting_user"]
        assert complete["expected"]["reply_reference_is_example"] is True
    for scenario_id in ("single_profile_game_used_for_query", "custom_profile_game_preserved"):
        case = cases[scenario_id]
        assert case["expected"]["reply_reference"] == "你打多大的？"
        assert case["hard_assertions_history"][0]["previous_snapshot"]["required_tool_names"] == ["search_current_games"]
    assert module.validate_dataset(records) == []


@pytest.mark.parametrize("field,value,error", [
    ("derivation_kind", "real_observed_success", "derivation_kind must be hypothetical_real_failure_derivation"),
    ("source_run_id", " ", "hypothetical derivation source_run_id must be a non-empty string"),
    ("source_scenario_id", None, "hypothetical derivation source_scenario_id must be a non-empty string"),
    ("hypothesis", [], "hypothetical derivation hypothesis must be a non-empty string"),
])
@pytest.mark.parametrize("target", ["record", "case"])
def test_inventory_hypothetical_source_validator_rejects_untraceable_derivation(field, value, error, target) -> None:
    module = load_validator_module()
    record = deepcopy(next(record for record in module.read_jsonl(module.DEFAULT_DATASET_PATH)
                           if record["id"] == "owner_chat_inventory_scope_regression_20261029_001"))
    source = record["source"] if target == "record" else record["eval_cases"][0]["source"]
    source[field] = value
    assert error in "\n".join(module.validate_dataset([record]))


def test_inventory_hypothetical_source_is_synthetic_and_refers_to_exact_live_fixtures() -> None:
    module = load_validator_module()
    records = module.read_jsonl(module.DEFAULT_DATASET_PATH)
    record = next(record for record in records if record["id"] == "owner_chat_inventory_scope_regression_20261029_001")
    assert len(record["eval_cases"]) == 2
    for source in [record["source"], *(case["source"] for case in record["eval_cases"])]:
        assert source["kind"] == "synthetic_regression_fixture"
        assert source["not_original_human_transcript"] is True
        assert source["derivation_kind"] == "hypothetical_real_failure_derivation"
    assert module.validate_dataset(records) == []


def _capacity_revision_record(records):
    """按证据 ID 选历史修订，后续追加记录不能改变既有断链/类型回归的目标。"""
    return next(record for record in records if record["id"] == "owner_chat_party_capacity_regression_20261029_001")


@pytest.mark.parametrize("field,value,error", [
    ("revision_id", "", "revision_id must be a non-empty string"),
    ("target_case_id", "unknown-case", "target must be a unique prior case"),
    ("live_scenario_id", "inventory_only_does_not_offer_recruitment", "target must be a unique prior case"),
    ("source_run_id", None, "source_run_id must be a non-empty string"),
    ("reason", " ", "reason must be a non-empty string"),
    ("previous_snapshot", {}, "previous_snapshot must be a non-empty object"),
    ("current_snapshot", [], "current_snapshot must be a non-empty object"),
])
def test_hard_assertion_revision_chain_rejects_invalid_or_unknown_target(field, value, error) -> None:
    module = load_validator_module()
    records = module.read_jsonl(module.DEFAULT_DATASET_PATH)
    _capacity_revision_record(records)["hard_assertion_revisions"][0][field] = value
    with pytest.raises(ValueError, match=error):
        module.resolve_hard_assertion_snapshots(records)


def test_hard_assertion_revisions_are_strict_append_only_and_validate_latest_code() -> None:
    module = load_validator_module()
    records = module.read_jsonl(module.DEFAULT_DATASET_PATH)
    before = deepcopy(records)
    resolved = module.resolve_hard_assertion_snapshots(records)
    assert records == before
    revision = _capacity_revision_record(records)["hard_assertion_revisions"][0]
    key = (revision["target_case_id"], revision["live_scenario_id"])
    assert resolved[key] == revision["current_snapshot"]
    assert module.validate_dataset(records) == []
    duplicate = deepcopy(records)
    _capacity_revision_record(duplicate)["hard_assertion_revisions"].append(deepcopy(revision))
    with pytest.raises(ValueError, match="duplicate hard assertion revision_id"):
        module.resolve_hard_assertion_snapshots(duplicate)
    broken = deepcopy(records)
    _capacity_revision_record(broken)["hard_assertion_revisions"][0]["previous_snapshot"]["expected_active_game_count"] = True
    with pytest.raises(ValueError, match="previous_snapshot does not match"):
        module.resolve_hard_assertion_snapshots(broken)
    wrong_latest = deepcopy(records)
    _capacity_revision_record(wrong_latest)["hard_assertion_revisions"][0]["current_snapshot"]["max_reply_chars"] = 1
    assert "latest hard snapshot max_reply_chars must match" in "\n".join(module.validate_dataset(wrong_latest))
    absent_revision = deepcopy([record for record in records if record["id"] != "owner_chat_party_capacity_regression_20261029_001"])
    assert "latest hard snapshot expected_tool_result_paths must match" in "\n".join(module.validate_dataset(absent_revision))


def test_hard_assertion_revision_chain_can_extend_but_cannot_reference_future_records() -> None:
    module = load_validator_module()
    records = module.read_jsonl(module.DEFAULT_DATASET_PATH)
    first = deepcopy(_capacity_revision_record(records)["hard_assertion_revisions"][0])
    second = deepcopy(first)
    second["revision_id"] = "second-revision"
    second["previous_snapshot"] = deepcopy(first["current_snapshot"])
    second["current_snapshot"] = deepcopy(first["current_snapshot"])
    second["current_snapshot"]["max_reply_chars"] = 79
    _capacity_revision_record(records)["hard_assertion_revisions"].append(second)
    final = module.resolve_hard_assertion_snapshots(records)
    assert final[(first["target_case_id"], first["live_scenario_id"])]["max_reply_chars"] == 79
    original_order = module.read_jsonl(module.DEFAULT_DATASET_PATH)
    with pytest.raises(ValueError, match="target must be a unique prior case"):
        module.resolve_hard_assertion_snapshots([original_order[-1], *original_order[:-1]])



def _proactive_help_review_record():
    module = load_validator_module()
    return deepcopy(next(record for record in module.read_jsonl(module.DEFAULT_DATASET_PATH)
                         if record["kind"] == "owner_chat_hard_assertion_revision"))


@pytest.mark.parametrize("field,value,error", [
    ("schema_version", True, "hard assertion review schema_version must be integer 1"),
    ("eval_cases", [], "hard assertion review must be metadata-only"),
    ("messages", [], "hard assertion review must be metadata-only"),
    ("input", {}, "hard assertion review must be metadata-only"),
    ("expected", {}, "hard assertion review must be metadata-only"),
    ("hard_assertion_revisions", [], "hard assertion review requires non-empty hard_assertion_revisions"),
])
def test_metadata_only_hard_review_rejects_transcript_or_missing_revision(field, value, error) -> None:
    module = load_validator_module()
    record = _proactive_help_review_record()
    record[field] = value
    assert error in "\n".join(module._validate_hard_assertion_review_record(record))


@pytest.mark.parametrize("field,value,error", [
    ("kind", "real_chat", "hard assertion review source must declare"),
    ("not_original_human_transcript", False, "hard assertion review source must declare"),
    ("statement", " ", "source statement must be a non-empty string"),
    ("basis", [], "source basis must be a non-empty string"),
    ("source_run_id", None, "source source_run_id must be a non-empty string"),
    ("observed_date", "", "source observed_date must be a non-empty string"),
])
def test_metadata_only_hard_review_requires_explicit_product_source(field, value, error) -> None:
    module = load_validator_module()
    record = _proactive_help_review_record()
    record["source"][field] = value
    assert error in "\n".join(module._validate_hard_assertion_review_record(record))


def test_metadata_review_flag_revision_is_checked_against_catalog_without_adding_case() -> None:
    module = load_validator_module()
    records = module.read_jsonl(module.DEFAULT_DATASET_PATH)
    assert module.validate_dataset(records) == []
    review = next(record for record in records if record["kind"] == "owner_chat_hard_assertion_revision")
    revision = review["hard_assertion_revisions"][0]
    revision["current_snapshot"].pop("require_help_offer")
    with pytest.raises(ValueError, match="must change the preceding snapshot"):
        module.resolve_hard_assertion_snapshots(records)
    records = module.read_jsonl(module.DEFAULT_DATASET_PATH)
    review = next(record for record in records if record["kind"] == "owner_chat_hard_assertion_revision")
    review["hard_assertion_revisions"][0]["current_snapshot"]["require_help_offer"] = False
    assert "latest hard snapshot require_help_offer must match" in "\n".join(module.validate_dataset(records))
