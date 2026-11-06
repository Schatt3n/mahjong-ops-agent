"""真实空池证据可替代范围过滤；OR不能拼接多次调用、模糊布尔或放宽其他硬边界。"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, ToolCall, ToolGateway, UserMessage
from mahjong_agent_runtime.models import ToolResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_product_contracts as policy
import run_real_owner_chat_live_eval as owner
import validate_real_owner_chat_golden as source_validator


RECORD_ID = "owner_chat_unfiltered_empty_scope_revision_20261102_001"
SCOPE = "result.customer_reply_contract.evidence_scope."


def scene_and_query(requirement=None):
    """前置画像和搜索均走真实Store/Gateway；模型语义在本离线协议测试中明确为替身。"""
    store = InMemoryAgentStore()
    scene = next(item for item in owner.live_eval_scenarios(apply_current_topic_policy=False) if item.scenario_id == "profile_range_empty_inquiry")
    scene.setup(store)
    tool = ToolGateway(store).execute(ToolCall(name="search_current_games", arguments={"requirement": requirement or {}, "limit": 10}, reason="离线证据协议"),
        trace_id="or-evidence", conversation_id=scene.message.conversation_id, sender_id=scene.message.sender_id,
        sender_name=scene.message.sender_name, source_message_id=scene.message.message_id, step_index=1)
    assert tool.called and tool.allowed and not tool.error
    return scene, store, tool


def check(scene, store, tools):
    """只评测既有硬断言，预置语义通过不计作真实模型能力证明。"""
    result = SimpleNamespace(status="waiting_user", final_reply="下午暂时没现成的，要我帮你问问吗？", tool_results=tools)
    validation = owner.validate_result(result, scene, [], store, {"passed": True, "error": None})
    return next(item for item in validation["checks"] if item["name"] == "search_current_games_result_should_match_one_declared_alternative")


@pytest.mark.parametrize("requirement", [{}, {"game_type": "red_center_mahjong", "stake_range": {"min": 1, "max": 3}}])
def test_same_actual_successful_gateway_result_can_prove_either_declared_path(requirement):
    scene, store, tool = scene_and_query(requirement)
    assert check(scene, store, [tool])["passed"] is True


def test_unfiltered_backend_scope_does_not_require_a_literal_empty_requirement_object():
    scene, store, tool = scene_and_query({"requesting_party": {"seat_count": 1}})
    assert tool.result["requirement"]
    assert tool.result["customer_reply_contract"]["evidence_scope"]["kind"] == "unfiltered_inventory"
    assert check(scene, store, [tool])["passed"] is True


@pytest.mark.parametrize("fault", ["filtered", "applied", "not_pool_empty", "not_narrower", "integer_pool", "integer_narrower",
    "string_pool", "string_narrower", "missing_pool", "missing_narrower", "nonempty_matches", "nonempty_alternatives",
    "only_self_report", "called_false", "allowed_false", "tool_error", "absent"])
def test_broad_empty_proof_rejects_missing_filtered_nonempty_or_unexecuted_evidence(fault):
    scene, store, tool = scene_and_query()
    evidence = tool.result["customer_reply_contract"]["evidence_scope"]
    if fault == "filtered":
        evidence["kind"] = "submitted_filters_only"
    elif fault == "applied":
        evidence["applied_filter_fields"] = ["game_type"]
    elif fault in {"not_pool_empty", "integer_pool", "string_pool"}:
        evidence["unfiltered_available_pool_empty"] = {"not_pool_empty": False, "integer_pool": 1, "string_pool": "true"}[fault]
    elif fault in {"not_narrower", "integer_narrower", "string_narrower"}:
        evidence["supports_narrower_empty_claim"] = {"not_narrower": False, "integer_narrower": 1, "string_narrower": "true"}[fault]
    elif fault == "missing_pool":
        evidence.pop("unfiltered_available_pool_empty")
    elif fault == "missing_narrower":
        evidence.pop("supports_narrower_empty_claim")
    elif fault == "nonempty_matches":
        tool.result["matches"] = [{"game": {"game_id": "actual-nonempty"}}]
    elif fault == "nonempty_alternatives":
        tool.result["alternatives"] = [{"game": {"game_id": "actual-alternative"}}]
    elif fault == "only_self_report":
        tool.result.pop("customer_reply_contract")
        tool.result["model_claim"] = "全店未过滤且为空，可推出下午为空"
    elif fault == "called_false":
        tool.called = False
    elif fault == "allowed_false":
        tool.allowed = False
    elif fault == "tool_error":
        tool.error = "query failed"
    assert check(scene, store, [] if fault == "absent" else [tool])["passed"] is False


def test_real_filtered_empty_result_does_not_support_a_broad_empty_claim():
    scene, store, tool = scene_and_query({"game_type": "hangzhou_mahjong"})
    assert tool.result["matches"] == [] and tool.result["alternatives"] == []
    assert check(scene, store, [tool])["passed"] is False


def test_two_successful_results_cannot_lend_each_other_missing_proof_fields():
    scene, store, first = scene_and_query()
    second = deepcopy(first)
    first.result["customer_reply_contract"]["evidence_scope"]["supports_narrower_empty_claim"] = False
    second.result["customer_reply_contract"]["evidence_scope"]["unfiltered_available_pool_empty"] = False
    assert check(scene, store, [first, second])["passed"] is False
    assert check(scene, store, [second, first])["passed"] is False


def test_or_preserves_the_existing_latest_successful_result_selection():
    scene, store, first = scene_and_query()
    second = deepcopy(first)
    second.result["matches"] = [{"game": {"game_id": "later-nonempty"}}]
    assert check(scene, store, [first, second])["passed"] is False


@pytest.mark.parametrize("invalid", [[], [{}], [None], [{"result.matches": []}, []], "any matching output"])
def test_alternative_protocol_rejects_empty_or_malformed_branches(invalid):
    scene, store, tool = scene_and_query()
    scene.expected_tool_result_paths = {"search_current_games": invalid}
    with pytest.raises(ValueError):
        check(scene, store, [tool])


def test_legacy_dictionary_check_names_and_non_boolean_equivalence_are_unchanged():
    scene, store, tool = scene_and_query({"stake_range": {"min": 1, "max": 3}})
    scene.expected_tool_result_paths = {"search_current_games": {"result.requirement.stake_range": {"min": 1, "max": 3}}}
    result = SimpleNamespace(status="waiting_user", final_reply="下午暂时没现成的，要我帮你问问吗？", tool_results=[tool])
    validation = owner.validate_result(result, scene, [], store, {"passed": True, "error": None})
    old = next(c for c in validation["checks"] if c["name"] == "search_current_games_result_should_keep_result.requirement.stake_range")
    assert old["passed"] is True and old["actual"] == old["expected"]
    assert owner.eval_values_equal("no_smoke", "no_smoking", field_path="result.requirement.smoke_preference")


@pytest.mark.parametrize("expected,actual,valid", [(True, True, True), (True, 1, False), (False, False, True), (False, 0, False)])
def test_boolean_proof_comparison_is_type_strict(expected, actual, valid):
    assert owner.eval_values_equal(actual, expected, field_path="generic.proof") is valid


def catalog():
    """实际catalog包含新声明字段；缺失字段不能以旧接口默认值蒙混通过。"""
    return {scene.scenario_id: {"required_reply_any": scene.required_reply_any,
        "forbidden_reply_contains": scene.forbidden_reply_contains, "expected_tool_result_paths": scene.expected_tool_result_paths}
        for scene in owner.live_eval_scenarios(apply_current_topic_policy=False)}


def test_versioned_scope_policy_keeps_original_prefix_and_all_unrelated_obligations():
    records = policy.read_product_records()
    record = next(r for r in records if r.get("id") == RECORD_ID)
    raw = policy.DATASET_PATH.read_bytes()
    assert hashlib.sha256(raw[:record["original_dataset_prefix_bytes"]]).hexdigest() == record["original_dataset_prefix_sha256"]
    revision = record["case_expectation_revisions"][0]
    assert revision["changed_criterion_ids"] == ["accept_range_as_known"]
    before, after = revision["previous_expectation"], deepcopy(revision["current_expectation"])
    next(c for c in after["required_meaning"] if c["id"] == "accept_range_as_known")["description"] = next(
        c["description"] for c in before["required_meaning"] if c["id"] == "accept_range_as_known")
    assert before == after
    resolved = policy.resolve_product_contract_revisions(records, scenario_contracts=catalog())
    assert resolved["revision_ids"]["owner:" + revision["target_case_id"]][-1] == revision["revision_id"]
    snapshots = source_validator.resolve_hard_assertion_snapshots(records)
    assert snapshots[(revision["target_case_id"], revision["live_scenario_id"])]["require_help_offer"] is True
    assert source_validator._validate_current_hard_assertion_snapshots(snapshots) == []


@pytest.mark.parametrize("fault", ["source_case", "previous_expectation", "old_field", "missing_actual_field", "undeclared_catalog", "undeclared_semantics", "duplicate"])
def test_versioned_policy_rejects_source_drift_or_unrelated_relaxation(fault):
    records = policy.read_product_records()
    record = next(r for r in records if r.get("id") == RECORD_ID)
    revision = record["case_expectation_revisions"][0]
    actual = catalog()
    if fault == "source_case":
        revision["source_case_sha256"] = "0" * 64
    elif fault == "previous_expectation":
        revision["previous_expectation"]["reply_reference"] = "不能倒改过去的标准"
    elif fault == "old_field":
        revision["previous_scenario_contract"]["expected_tool_result_paths"]["search_current_games"]["result.requirement.stake_range"]["max"] = 4
    elif fault == "missing_actual_field":
        actual[revision["live_scenario_id"]].pop("expected_tool_result_paths")
    elif fault == "undeclared_catalog":
        revision["current_scenario_contract"]["forbidden_reply_contains"] = []
    elif fault == "undeclared_semantics":
        revision["current_expectation"]["forbidden_meaning"].pop()
    else:
        records.append(deepcopy(record))
    with pytest.raises(ValueError):
        policy.resolve_product_contract_revisions(records, scenario_contracts=actual)
