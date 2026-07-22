from __future__ import annotations

import json

from mahjong_agent_runtime.action_contract import parse_action_with_repairs


def terminal_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "goal": "记录用户补充条件",
        "objective_status": "needs_tool",
        "reasoning_summary": "约束已写入，短句确认后停止。",
        "reply_to_user": "好的，记下了。",
        "tool_calls": [],
        "needs_human": False,
        "stop_reason": {
            "can_stop": True,
            "why": "约束已经写入。",
            "pending_work": [],
            "depends_on_tool_results": True,
        },
    }
    payload.update(overrides)
    return payload


def test_parse_action_repairs_status_when_all_other_fields_are_terminal() -> None:
    action, errors, repairs = parse_action_with_repairs(json.dumps(terminal_payload(), ensure_ascii=False))

    assert errors == []
    assert action.objective_status == "completed"
    assert action.reply_to_user == "好的，记下了。"
    assert repairs == [
        {
            "field": "objective_status",
            "from": "needs_tool",
            "to": "completed",
            "reason": "all other contract fields describe a terminal reply with no tool work",
        }
    ]


def test_parse_action_accepts_optional_self_assessment() -> None:
    payload = terminal_payload(
        objective_status="completed",
        self_assessment={"progress": "advancing", "should_escalate": False},
    )

    action, errors, repairs = parse_action_with_repairs(json.dumps(payload, ensure_ascii=False))

    assert errors == []
    assert repairs == []
    assert action.self_assessment is not None
    assert action.self_assessment.progress == "advancing"
    assert action.self_assessment.should_escalate is False


def test_parse_action_keeps_self_assessment_optional() -> None:
    payload = terminal_payload(objective_status="completed")

    action, errors, repairs = parse_action_with_repairs(json.dumps(payload, ensure_ascii=False))

    assert errors == []
    assert repairs == []
    assert action.self_assessment is None


def test_parse_action_rejects_invalid_self_assessment() -> None:
    payload = terminal_payload(
        objective_status="completed",
        self_assessment={"progress": "stuck", "should_escalate": "yes"},
    )

    _, errors, repairs = parse_action_with_repairs(json.dumps(payload, ensure_ascii=False))

    assert repairs == []
    assert "self_assessment.progress is invalid" in errors
    assert "self_assessment.should_escalate must be boolean" in errors


def test_parse_action_does_not_repair_when_tool_work_is_still_pending() -> None:
    payload = terminal_payload(
        reply_to_user="",
        tool_calls=[
            {
                "name": "search_current_games",
                "arguments": {"requirement": {}},
                "reason": "需要先查询当前局。",
            }
        ],
        stop_reason={
            "can_stop": False,
            "why": "等待查询结果。",
            "pending_work": ["查询当前局"],
            "depends_on_tool_results": True,
        },
    )

    action, errors, repairs = parse_action_with_repairs(json.dumps(payload, ensure_ascii=False))

    assert errors == []
    assert action.objective_status == "needs_tool"
    assert len(action.tool_calls) == 1
    assert repairs == []


def test_parse_action_does_not_hide_ambiguous_invalid_contract() -> None:
    payload = terminal_payload(reply_to_user="", stop_reason={"can_stop": True, "why": "", "pending_work": [], "depends_on_tool_results": False})

    action, errors, repairs = parse_action_with_repairs(json.dumps(payload, ensure_ascii=False))

    assert action.objective_status == "needs_human"
    assert repairs == []
    assert "needs_tool requires at least one tool_call" in errors
    assert "needs_tool requires empty reply_to_user" not in errors


def test_parse_action_accepts_valid_tool_dependency_graph() -> None:
    payload = terminal_payload(
        reply_to_user="",
        tool_calls=[
            {
                "name": "check_room_availability",
                "arguments": {"start_at": "2026-07-20T14:00:00+08:00", "end_at": "2026-07-20T18:00:00+08:00"},
                "reason": "check rooms",
                "call_id": "rooms",
                "depends_on": [],
            },
            {
                "name": "search_current_games",
                "arguments": {"requirement": {}},
                "reason": "search games",
                "call_id": "games",
                "depends_on": [],
            },
        ],
        stop_reason={
            "can_stop": False,
            "why": "waiting for reads",
            "pending_work": ["check rooms", "search games"],
            "depends_on_tool_results": True,
        },
    )

    action, errors, _ = parse_action_with_repairs(json.dumps(payload, ensure_ascii=False))

    assert errors == []
    assert [call.call_id for call in action.tool_calls] == ["rooms", "games"]
    assert action.tool_calls[1].depends_on == []


def test_parse_action_repairs_omitted_dependency_but_still_requires_call_id() -> None:
    payload = terminal_payload(
        reply_to_user="",
        tool_calls=[
            {
                "name": "search_current_games",
                "arguments": {"requirement": {}},
                "reason": "search games",
                "call_id": "games",
                "depends_on": [],
            },
            {
                "name": "search_customers",
                "arguments": {"requirement": {}},
                "reason": "search customers",
            },
        ],
        stop_reason={
            "can_stop": False,
            "why": "waiting for reads",
            "pending_work": ["search"],
            "depends_on_tool_results": True,
        },
    )

    _, errors, repairs = parse_action_with_repairs(json.dumps(payload, ensure_ascii=False))

    assert "tool_calls[2].call_id is required in dependency graph mode" in errors
    assert "tool_calls[2].depends_on is required in dependency graph mode" not in errors
    assert repairs[-1] == {
        "field": "tool_calls[2].depends_on",
        "from": None,
        "to": [],
        "reason": "an omitted tool dependency is structurally equivalent to an independent call",
    }


def test_parse_action_repairs_omitted_dependency_for_independent_call() -> None:
    payload = terminal_payload(
        reply_to_user="",
        tool_calls=[
            {
                "name": "search_current_games",
                "arguments": {"requirement": {}},
                "reason": "search games",
                "call_id": "games",
                "depends_on": [],
            },
            {
                "name": "search_customers",
                "arguments": {"requirement": {}},
                "reason": "search customers",
                "call_id": "customers",
            },
        ],
        stop_reason={
            "can_stop": False,
            "why": "waiting for reads",
            "pending_work": ["search"],
            "depends_on_tool_results": True,
        },
    )

    action, errors, repairs = parse_action_with_repairs(json.dumps(payload, ensure_ascii=False))

    assert errors == []
    assert action.tool_calls[1].depends_on == []
    assert repairs[-1]["field"] == "tool_calls[2].depends_on"


def test_parse_action_accepts_diagnostic_call_id_without_enabling_graph_mode() -> None:
    payload = terminal_payload(
        reply_to_user="",
        tool_calls=[
            {
                "name": "search_current_games",
                "arguments": {"requirement": {}},
                "reason": "search games",
                "call_id": "games",
            }
        ],
        stop_reason={
            "can_stop": False,
            "why": "waiting for read",
            "pending_work": ["search"],
            "depends_on_tool_results": True,
        },
    )

    action, errors, _ = parse_action_with_repairs(json.dumps(payload, ensure_ascii=False))

    assert errors == []
    assert action.tool_calls[0].call_id == "games"
    assert action.tool_calls[0].depends_on is None


def test_parse_action_rejects_unknown_and_cyclic_tool_dependencies() -> None:
    unknown_payload = terminal_payload(
        reply_to_user="",
        tool_calls=[
            {
                "name": "search_current_games",
                "arguments": {"requirement": {}},
                "reason": "search games",
                "call_id": "games",
                "depends_on": ["missing"],
            }
        ],
        stop_reason={
            "can_stop": False,
            "why": "waiting for reads",
            "pending_work": ["search"],
            "depends_on_tool_results": True,
        },
    )
    cycle_payload = terminal_payload(
        reply_to_user="",
        tool_calls=[
            {
                "name": "search_current_games",
                "arguments": {"requirement": {}},
                "reason": "search games",
                "call_id": "games",
                "depends_on": ["customers"],
            },
            {
                "name": "search_customers",
                "arguments": {"requirement": {}},
                "reason": "search customers",
                "call_id": "customers",
                "depends_on": ["games"],
            },
        ],
        stop_reason={
            "can_stop": False,
            "why": "waiting for reads",
            "pending_work": ["search"],
            "depends_on_tool_results": True,
        },
    )

    _, unknown_errors, _ = parse_action_with_repairs(json.dumps(unknown_payload, ensure_ascii=False))
    _, cycle_errors, _ = parse_action_with_repairs(json.dumps(cycle_payload, ensure_ascii=False))

    assert unknown_errors == ["tool call games depends on unknown call_id values: missing"]
    assert cycle_errors == ["tool dependency graph contains a cycle"]
