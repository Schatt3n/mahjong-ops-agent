"""Structured planning and output contracts supplied to the main model."""

from __future__ import annotations

from typing import Any


def output_contract() -> dict[str, Any]:
    return {
        "format": "json_object",
        "required_keys": [
            "goal",
            "objective_status",
            "reasoning_summary",
            "objective_state",
            "objective_plan",
            "plan_revision_reason",
            "reply_to_user",
            "tool_calls",
            "needs_human",
            "stop_reason",
        ],
        "objective_status_values": ["needs_tool", "waiting_user", "completed", "needs_human", "unknown"],
        "field_types": {
            "goal": "string",
            "objective_status": "string",
            "reasoning_summary": "string",
            "objective_state": "object; structured current task state, including known facts, missing facts, current phase, active IDs, blockers",
            "objective_plan": "array; ordered plan steps. each step should include step_id, title, status, tool, depends_on, decision_rule",
            "plan_revision_reason": "string; why this plan is created or changed after reading current message/tool results",
            "reply_to_user": "string",
            "tool_calls": "array",
            "needs_human": "boolean",
            "stop_reason": "object",
            "badcase": "null; deprecated side-channel, call record_badcase tool instead",
        },
        "objective_state_contract": {
            "current_phase": "recommended string: understand_intent | query_existing_games | collect_missing_info | create_game | search_customers | draft_invites | record_feedback | answer_user | wait_user | human_review",
            "known_facts": (
                "recommended JSON object mapping stable keys to values; facts already safe to use for this objective. "
                "Every object member must use key:value syntax, for example "
                "{\"requested_game\": \"hangzhou_mahjong\", \"known_player_count\": 2}. "
                "Never write set-like JSON such as {\"fact A\", \"fact B\"}; use an array when no keys are needed."
            ),
            "missing_facts": "recommended array of strings; facts still needed before state writes or drafts",
            "active_game_id": "optional string|null",
            "blockers": "recommended array of strings",
            "reply_scope": (
                "recommended object for terminal replies: requested_information, allowed_response_facts, "
                "background_facts_to_withhold. Context facts may support reasoning without becoming customer-visible."
            ),
        },
        "objective_plan_contract": {
            "step_status_values": ["pending", "in_progress", "done", "blocked", "skipped"],
            "required_step_keys": ["step_id", "title", "status"],
            "recommended_step_keys": ["tool", "depends_on", "decision_rule"],
            "tool_step_rule": "Any step that needs system state should map to one available tool. Use objective_status=needs_tool while such steps are still in_progress.",
            "revision_rule": "After previous_tool_results are present, mark completed tool steps done, update known facts/blockers, and choose the next step instead of restarting from scratch.",
        },
        "stop_reason_contract": {
            "can_stop": "required boolean; false when objective_status=needs_tool, true for terminal statuses",
            "why": "required non-empty string explaining why the agent can stop now or why it must continue with tools",
            "pending_work": "required array of strings; non-empty when can_stop=false",
            "depends_on_tool_results": "required boolean; true if the decision depends on previous_tool_results or system state",
        },
        "tool_call_contract": {
            "call_id": (
                "optional non-empty string in legacy single-call mode; required and unique for every call when an "
                "action contains an explicit dependency graph"
            ),
            "depends_on": (
                "optional array in legacy mode; array of prerequisite call_id strings in dependency graph mode. "
                "Omitted/null is normalized to []; use [] only when the call is truly independent"
            ),
            "name": "required non-empty string",
            "arguments": "required object, validated again by ToolGateway schema",
            "reason": "required non-empty string explaining why this tool is needed now",
            "idempotency_key": "optional string|null; backend still derives authoritative idempotency key",
        },
        "invariants": [
            "objective_status=needs_tool requires at least one tool_call",
            "objective_status=needs_tool requires empty reply_to_user",
            "objective_status=waiting_user|completed|needs_human|unknown must not include tool_calls",
            "objective_status=waiting_user|completed|needs_human|unknown requires non-empty reply_to_user",
            "objective_status=needs_human requires needs_human=true",
            "needs_human=true requires objective_status=needs_human",
            "objective_status=needs_tool requires stop_reason.can_stop=false and non-empty pending_work",
            "objective_status=waiting_user|completed|needs_human|unknown requires stop_reason.can_stop=true",
            "invalid contract means backend will not execute any tool",
            "multiple independent parallel_safe read-only calls should declare unique call_id and depends_on=[]; dependent calls must list their prerequisites",
            "missing graph metadata keeps backward-compatible sequential execution; the model cannot mark a tool parallel-safe because that permission comes from ToolGateway",
            "badcase must be null; badcase/eval writes must use record_badcase tool_call",
            "terminal reply must answer only current_message or an explicitly unresolved confirmation; do not append adjacent active-game facts, shortage, time, or calls to action that the user did not ask for",
            "casual-chat replies may use business state for continuity but must not surface that state unless current_message explicitly refers to it",
        ],
    }


def planning_contract() -> dict[str, Any]:
    return {
        "purpose": "把每轮用户输入转成一个可执行目标，然后用工具结果持续修订计划。",
        "loop_rule": (
            "每一轮先更新 objective_state，再给出 objective_plan。"
            "如果计划中的下一步依赖系统事实，必须通过 tool_calls 调用工具；工具返回后基于 previous_tool_results 修订计划。"
        ),
        "state_progression": [
            "理解意图和上下文",
            "确认已知槽位、画像默认值和缺失槽位",
            "需要事实时查询当前局、房态或候选人",
            "需要写入时创建/更新局、记录候选人反馈或生成待审批草稿",
            "根据工具结果决定继续调用工具、追问用户、短句回复或转人工",
        ],
        "do_not": [
            "不要只用一句自然语言承诺代替应执行的工具步骤",
            "不要在工具结果回来后丢掉上一轮已确认的计划和槽位",
            "不要把计划、工具名或后台细节暴露给客户",
        ],
    }


__all__ = ["output_contract", "planning_contract"]
