from __future__ import annotations

import json

import pytest

from mahjong_agent_runtime.action_contract import parse_action_with_repairs
from mahjong_agent_runtime.models import ToolResult


def test_missing_plan_step_ids_are_repaired_as_structural_metadata():
    payload = {
        "goal": "完成未来预约",
        "objective_status": "completed",
        "reasoning_summary": "局已创建，等待招募窗口。",
        "objective_plan": [
            {"step_id": "1", "title": "创建局", "status": "done", "depends_on": []},
            {"title": "等待窗口", "status": "done", "depends_on": ["1"]},
        ],
        "reply_to_user": "好，记上了。",
        "tool_calls": [],
        "needs_human": False,
        "stop_reason": {
            "can_stop": True,
            "why": "本轮已经完成",
            "pending_work": [],
            "depends_on_tool_results": True,
        },
    }

    action, errors, repairs = parse_action_with_repairs(json.dumps(payload, ensure_ascii=False))

    assert errors == []
    assert action.objective_plan[1]["step_id"] == "2"
    assert repairs == [
        {
            "field": "objective_plan[2].step_id",
            "from": None,
            "to": "2",
            "reason": "plan step identifiers are structural metadata and can be assigned deterministically",
        }
    ]


def test_single_or_null_plan_dependencies_are_repaired_to_arrays():
    payload = {
        "goal": "安全回复客户",
        "objective_status": "completed",
        "reasoning_summary": "不披露其他客户的私聊。",
        "objective_plan": [
            {"step_id": "1", "title": "理解请求", "status": "done", "depends_on": None},
            {"step_id": "2", "title": "安全回复", "status": "done", "depends_on": "1"},
        ],
        "reply_to_user": "别人的私聊我不方便转述。",
        "tool_calls": [],
        "needs_human": False,
        "stop_reason": {
            "can_stop": True,
            "why": "已安全回复",
            "pending_work": [],
            "depends_on_tool_results": False,
        },
    }

    action, errors, repairs = parse_action_with_repairs(json.dumps(payload, ensure_ascii=False))

    assert errors == []
    assert action.objective_plan[0]["depends_on"] == []
    assert action.objective_plan[1]["depends_on"] == ["1"]
    assert [item["field"] for item in repairs] == [
        "objective_plan[1].depends_on",
        "objective_plan[2].depends_on",
    ]


def _tool_payload(dependencies=None, *, call_id="current"):
    return {
        "goal": "继续查询", "objective_status": "needs_tool", "reasoning_summary": "读取已知条件的结果。",
        "reply_to_user": "", "needs_human": False,
        "tool_calls": [{"name": "search_current_games", "call_id": call_id,
                        "depends_on": ["prior"] if dependencies is None else dependencies,
                        "arguments": {"requirement": {"stake": "1"}, "limit": 5}, "reason": "读取局池"}],
        "stop_reason": {"can_stop": False, "why": "等待工具结果", "pending_work": ["查询"], "depends_on_tool_results": True},
    }


def _successful_history(*, call_id="prior"):
    return ToolResult(name="search_customers", called=True, allowed=True, call_id=call_id, result={"candidates": []})


def test_single_call_removes_only_proven_successful_latest_historical_dependency():
    payload = _tool_payload()
    result = _successful_history()
    action, errors, repairs = parse_action_with_repairs(
        json.dumps(payload), turn_tool_evidence=[result], previous_step_tool_results=[result],
    )
    assert errors == []
    assert action.tool_calls[0].depends_on == []
    assert action.tool_calls[0].name == payload["tool_calls"][0]["name"]
    assert action.tool_calls[0].arguments == payload["tool_calls"][0]["arguments"]
    assert repairs[0]["satisfied_historical_call_ids"] == ["prior"]
    assert repairs[0]["from"] == ["prior"] and repairs[0]["to"] == []


@pytest.mark.parametrize("called,allowed,error", [(False, True, None), (True, False, None), (True, True, "failed")])
def test_failed_historical_dependency_is_not_repaired_even_if_plan_claims_done(called, allowed, error):
    payload = _tool_payload()
    payload["objective_plan"] = [{"step_id": "prior", "title": "自述完成", "status": "done", "depends_on": []}]
    result = ToolResult(name="search_customers", called=called, allowed=allowed, error=error, call_id="prior")
    _, errors, repairs = parse_action_with_repairs(
        json.dumps(payload), turn_tool_evidence=[result], previous_step_tool_results=[result],
    )
    assert "depends on unknown call_id values: prior" in " ".join(errors)
    assert repairs == []


def test_unknown_historical_dependency_remains_invalid_without_execution_proof():
    _, errors, repairs = parse_action_with_repairs(
        json.dumps(_tool_payload()), turn_tool_evidence=[], previous_step_tool_results=[],
    )
    assert "depends on unknown call_id values: prior" in " ".join(errors)
    assert repairs == []


def test_reused_historical_id_is_ambiguous_even_when_latest_execution_succeeded():
    first, second = _successful_history(), _successful_history()
    payload = _tool_payload()
    payload["objective_plan"] = [{"step_id": "prior", "title": "计划里的同名步骤", "status": "done", "depends_on": []}]
    _, errors, repairs = parse_action_with_repairs(
        json.dumps(payload), turn_tool_evidence=[first, second], previous_step_tool_results=[second],
    )
    assert "depends on unknown call_id values: prior" in " ".join(errors)
    assert repairs == []


def test_old_success_cannot_skip_a_more_recent_failed_batch_or_missing_history():
    success = _successful_history()
    failed_batch = ToolResult(name="agent_action_contract", called=False, allowed=False, error="invalid")
    for history, latest in (([success], [failed_batch]), ([], [success]), ([success], [])):
        _, errors, repairs = parse_action_with_repairs(
            json.dumps(_tool_payload()), turn_tool_evidence=history, previous_step_tool_results=latest,
        )
        assert "depends on unknown call_id values: prior" in " ".join(errors)
        assert repairs == []


def test_same_batch_dependency_takes_priority_over_a_successful_historical_name():
    payload = _tool_payload()
    payload["tool_calls"].append({**payload["tool_calls"][0], "call_id": "prior", "depends_on": []})
    success = _successful_history()
    action, errors, repairs = parse_action_with_repairs(
        json.dumps(payload), turn_tool_evidence=[success], previous_step_tool_results=[success],
    )
    assert errors == [] and repairs == []
    assert action.tool_calls[0].depends_on == ["prior"]


def test_self_dependency_is_not_removed_by_historical_success_or_plan_namespace_repair():
    payload = _tool_payload(call_id="prior")
    payload["objective_plan"] = [{"step_id": "prior", "title": "同名计划步骤", "status": "done", "depends_on": []}]
    success = _successful_history()
    _, errors, repairs = parse_action_with_repairs(
        json.dumps(payload), turn_tool_evidence=[success], previous_step_tool_results=[success],
    )
    assert "must not depend on itself" in " ".join(errors)
    assert repairs == []


def test_partial_dependency_repair_does_not_hide_remaining_unknown_dependency():
    success = _successful_history()
    _, errors, repairs = parse_action_with_repairs(
        json.dumps(_tool_payload(["prior", "missing"])),
        turn_tool_evidence=[success], previous_step_tool_results=[success],
    )
    assert "depends on unknown call_id values: missing" in " ".join(errors)
    assert repairs[0]["to"] == ["missing"]


def test_numeric_self_reference_cannot_be_cleared_by_plan_step_repair():
    payload = _tool_payload([3], call_id="3")
    payload["objective_plan"] = [{"step_id": "3", "title": "同名计划步骤", "status": "done", "depends_on": []}]
    _, errors, repairs = parse_action_with_repairs(json.dumps(payload))
    assert "tool_calls[1].depends_on items must be non-empty strings" in errors
    assert repairs == []


def test_numeric_reference_to_failed_history_cannot_be_cleared_by_plan_step_repair():
    payload = _tool_payload([3])
    payload["objective_plan"] = [{"step_id": "3", "title": "自述已完成", "status": "done", "depends_on": []}]
    failed = ToolResult(name="search_customers", call_id="3", called=True, allowed=False, error="failed")
    _, errors, repairs = parse_action_with_repairs(
        json.dumps(payload), turn_tool_evidence=[failed], previous_step_tool_results=[failed],
    )
    assert "tool_calls[1].depends_on items must be non-empty strings" in errors
    assert repairs == []


def _premature_action_parts(metadata_key, *, revision_in_prefix=False):
    base = _tool_payload([])
    prefix = {
        "goal": base["goal"], "objective_status": base["objective_status"],
        "reasoning_summary": base["reasoning_summary"],
        "objective_state": {"current_phase": "search", "known_facts": {}, "missing_facts": [], "blockers": []},
        "objective_plan": [],
    }
    tail = {
        "plan_revision_reason": "根据真实查询结果继续。", "reply_to_user": "",
        "tool_calls": base["tool_calls"], "needs_human": False, "stop_reason": base["stop_reason"],
    }
    if revision_in_prefix:
        prefix["plan_revision_reason"] = tail.pop("plan_revision_reason")
    prefix[metadata_key] = {"requested_information": ["找局"], "allowed_response_facts": ["查询条件"]} if metadata_key == "reply_scope" else None
    return prefix, tail


def _join_premature_action(prefix, tail, *, leading=""):
    return leading + json.dumps(prefix, ensure_ascii=False) + ", " + json.dumps(tail, ensure_ascii=False)[1:]


@pytest.mark.parametrize("metadata_key,revision_in_prefix", [("reply_scope", False), ("active_game_id", False), ("reply_scope", True)])
def test_extra_data_repair_preserves_full_action_and_moves_only_declared_state_metadata(metadata_key, revision_in_prefix):
    prefix, tail = _premature_action_parts(metadata_key, revision_in_prefix=revision_in_prefix)
    raw = _join_premature_action(prefix, tail, leading=" \n")
    action, errors, repairs = parse_action_with_repairs(raw)
    assert errors == []
    assert action.objective_state == {**prefix["objective_state"], metadata_key: prefix[metadata_key]}
    assert action.plan_revision_reason == "根据真实查询结果继续。"
    assert action.stop_reason == tail["stop_reason"]
    assert action.tool_calls[0].name == tail["tool_calls"][0]["name"]
    assert action.tool_calls[0].arguments == tail["tool_calls"][0]["arguments"]
    assert action.tool_calls[0].call_id == tail["tool_calls"][0]["call_id"]
    offset = len(" \n" + json.dumps(prefix, ensure_ascii=False)) - 1
    assert repairs[0]["removed_character_offset"] == offset and raw[offset] == "}"
    assert repairs[1]["field"] == f"objective_state.{metadata_key}"


def test_premature_action_repair_ignores_braces_and_field_text_inside_quoted_arguments():
    prefix, tail = _premature_action_parts("reply_scope")
    note = '保留原文 } , "tool_calls": [] 和 {，包含引号与反斜杠 \\。'
    prefix["reasoning_summary"] = note
    tail["tool_calls"][0]["arguments"]["requirement"]["note"] = note
    action, errors, repairs = parse_action_with_repairs(_join_premature_action(prefix, tail))
    assert errors == [] and repairs
    assert action.reasoning_summary == note
    assert action.tool_calls[0].arguments["requirement"]["note"] == note


@pytest.mark.parametrize("extra", ["\n{}", "\n说明：接着执行。", ', "goal": "另一个动作"}', "}"])
def test_premature_action_repair_rejects_unconsumed_json_or_text_tail(extra):
    prefix, tail = _premature_action_parts("reply_scope")
    _, errors, repairs = parse_action_with_repairs(_join_premature_action(prefix, tail) + extra)
    assert errors and repairs == []


def test_complete_action_followed_by_another_action_is_not_merged_or_truncated():
    prefix, tail = _premature_action_parts("reply_scope")
    complete = json.dumps({**prefix, **tail}, ensure_ascii=False)
    for raw in (complete + "\n" + complete, complete + ", " + json.dumps(tail)[1:]):
        _, errors, repairs = parse_action_with_repairs(raw)
        assert errors and repairs == []


@pytest.mark.parametrize("metadata_key,inner_value", [("reply_scope", {"requested_information": ["不同含义"]}), ("active_game_id", "another_game")])
def test_premature_action_repair_rejects_conflicting_state_metadata(metadata_key, inner_value):
    prefix, tail = _premature_action_parts(metadata_key)
    prefix["objective_state"][metadata_key] = inner_value
    _, errors, repairs = parse_action_with_repairs(_join_premature_action(prefix, tail))
    assert "conflicts with objective_state" in " ".join(errors)
    assert repairs == []


def test_premature_action_repair_accepts_identical_state_metadata_without_overwriting_it():
    prefix, tail = _premature_action_parts("reply_scope")
    prefix["objective_state"]["reply_scope"] = dict(prefix["reply_scope"])
    action, errors, _ = parse_action_with_repairs(_join_premature_action(prefix, tail))
    assert errors == []
    assert action.objective_state["reply_scope"] == prefix["reply_scope"]


@pytest.mark.parametrize("location", ["prefix", "tail", "arguments", "legal_json"])
def test_action_parser_rejects_duplicate_keys_instead_of_overwriting_any_action_part(location):
    prefix, tail = _premature_action_parts("reply_scope")
    raw = _join_premature_action(prefix, tail)
    if location == "prefix":
        raw = raw.replace('"goal":', '"goal": "另一个目标", "goal":', 1)
    elif location == "tail":
        raw = raw.replace('"tool_calls":', '"tool_calls": [], "tool_calls":', 1)
    elif location == "arguments":
        raw = raw.replace('"stake": "1"', '"stake": "1", "stake": "2"', 1)
    else:
        raw = json.dumps({**prefix, **tail}, ensure_ascii=False).replace('"tool_calls":', '"tool_calls": [], "tool_calls":', 1)
    _, errors, repairs = parse_action_with_repairs(raw)
    assert "duplicate JSON key" in " ".join(errors)
    assert repairs == []


@pytest.mark.parametrize("change", ["missing_metadata", "nonobject_state", "wrong_scope_type", "different_action_header", "two_premature_braces", "nonarray_plan", "nonarray_tools"])
def test_extra_data_without_the_unambiguous_misplaced_state_shape_is_not_repaired(change):
    prefix, tail = _premature_action_parts("reply_scope")
    if change == "missing_metadata":
        prefix.pop("reply_scope")
    elif change == "nonobject_state":
        prefix["objective_state"] = []
    elif change == "wrong_scope_type":
        prefix["reply_scope"] = "不能猜成对象"
    elif change == "different_action_header":
        tail = {"goal": "另一个动作", **tail}
    elif change == "nonarray_plan":
        prefix["objective_plan"] = 3
    elif change == "nonarray_tools":
        tail["tool_calls"] = 3
    raw = _join_premature_action(prefix, tail)
    if change == "two_premature_braces":
        raw = json.dumps(prefix, ensure_ascii=False) + "}, " + json.dumps(tail, ensure_ascii=False)[1:]
    _, errors, repairs = parse_action_with_repairs(raw)
    assert errors and repairs == []


def test_valid_action_json_is_unchanged_and_premature_shape_still_requires_action_schema():
    prefix, tail = _premature_action_parts("reply_scope")
    state = prefix["objective_state"]
    state["reply_scope"] = prefix.pop("reply_scope")
    action, errors, repairs = parse_action_with_repairs(json.dumps({**prefix, **tail}, ensure_ascii=False))
    assert errors == [] and repairs == []
    assert action.objective_state == state and action.tool_calls[0].arguments == tail["tool_calls"][0]["arguments"]
    prefix["reply_scope"] = state["reply_scope"]
    tail["tool_calls"] = []
    _, errors, _ = parse_action_with_repairs(_join_premature_action(prefix, tail))
    assert "needs_tool requires at least one tool_call" in errors
