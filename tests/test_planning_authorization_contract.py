"""已授权目标的多轮续办和工具阶段输出协议不能被单轮话术习惯覆盖。"""

from __future__ import annotations

import json
from pathlib import Path

from mahjong_agent_runtime.action_contract import parse_action_with_repairs

ROOT = Path(__file__).resolve().parents[1]


def test_prompt_keeps_authorization_within_the_same_unfinished_goal():
    """补齐缺项可沿原授权继续，改目标和纯咨询不能继承不适用的写入许可。"""
    prompt = (ROOT / "src/mahjong_agent_runtime/prompts/agent_runtime_system.md").read_text()
    assert "同一尚未完成目标中的明确委托" in prompt
    assert "不要求最新一条消息重复" in prompt
    assert "授权不能补足未知人数、时间等事实" in prompt
    assert "用户改变目标、撤销委托或明确限定只查询时" in prompt
    assert "单纯咨询且此前没有明确组局委托时，仍需先征询是否帮忙" in prompt
    assert "已完成步骤和已有安排不重复启动" in prompt


def test_nonterminal_tool_reply_is_still_rejected_instead_of_silently_dropped():
    """提示词澄清不能变成宽松执行：矛盾的主回复仍由原合同拒绝，工具参数不被改写。"""
    payload = {
        "goal": "为已授权组局准备邀约", "objective_status": "needs_tool",
        "reasoning_summary": "候选人已找到，下一步仍需真实工具。",
        "objective_state": {"current_phase": "draft_invites", "known_facts": {}, "missing_facts": [], "blockers": []},
        "objective_plan": [{"step_id": "draft", "title": "准备邀约", "status": "in_progress", "tool": "create_invite_drafts", "depends_on": []}],
        "plan_revision_reason": "按候选人结果继续。", "reply_to_user": "好，我帮你问问。",
        "tool_calls": [{"call_id": "c1", "name": "create_invite_drafts", "arguments": {"game_id": "game-1", "invitations": []}, "reason": "准备已授权邀约", "depends_on": []}],
        "needs_human": False,
        "stop_reason": {"can_stop": False, "why": "还需准备邀约", "pending_work": ["准备邀约"], "depends_on_tool_results": True},
        "badcase": None,
    }
    action, errors, repairs = parse_action_with_repairs(json.dumps(payload))
    assert "needs_tool requires empty reply_to_user" in errors
    assert action.tool_calls == []
    assert not any(repair.get("field") == "reply_to_user" for repair in repairs)
    payload["reply_to_user"] = ""
    action, errors, _ = parse_action_with_repairs(json.dumps(payload))
    assert errors == []
    assert action.tool_calls[0].arguments == {"game_id": "game-1", "invitations": []}
    prompt = (ROOT / "src/mahjong_agent_runtime/prompts/agent_runtime_system.md").read_text()
    assert '本次提交 `tool_calls`，就用 `objective_status=needs_tool` 且 `reply_to_user=""`' in prompt


def test_confirmation_semantics_require_consistency_without_known_condition_echo():
    """保留首次人工描述及澄清理由；问缺项可简短，但不得改写事实或假设人数时间。"""
    records = [json.loads(line) for line in (ROOT / "eval/golden/real_owner_chat_golden.jsonl").read_text().splitlines()]
    record = next(item for item in records if item["kind"] == "owner_chat_live_semantic_regression")
    case = next(item for item in record["eval_cases"] if item["live_scenario_id"] == "confirmation_does_not_answer_missing_party")
    criterion = next(item for item in case["expected"]["required_meaning"] if item["id"] == "retain_known_profile_and_conditions")
    assert "不要求回复复述这些条件" in criterion["description"]
    assert "不能改写或重新追问这些条件" in criterion["description"]
    revision = case["semantic_review_revisions"][-1]
    assert revision["previous_description"] == "沿用唯一国标玩法及已说0.5无烟，不重问这些已知条件。"
    assert revision["business_hard_assertions_changed"] is False
    assert case["existing_hard_assertions_snapshot"]["expected_status_any"] == ["waiting_user"]
    assert case["existing_hard_assertions_snapshot"]["expected_active_game_count"] == 0
    assert case["existing_hard_assertions_snapshot"]["forbidden_tool_names"] == ["create_game", "search_customers", "create_invite_drafts"]


def test_people_source_roles_remain_consistent_across_query_and_creation():
    """只检查通用规划合同，保留工具/状态硬验收；不把文本存在当作真实模型已理解。"""
    from mahjong_agent_runtime.domains.context_builders.contracts import planning_contract
    prompt = (ROOT / "src/mahjong_agent_runtime/prompts/agent_runtime_system.md").read_text()
    assert "先独立于工具解释原话的人数角色与来源" in prompt
    assert "先独立陈述桌况再委托补齐或组织，通常是在报告己方当前进度" in prompt
    assert "寻找对象、参加条件或他人引语" in prompt
    assert "仍须结合上下文判断" in prompt
    assert "工具字段含义不决定原话角色" in prompt
    assert "空查询不提供己方人数新事实" in prompt
    assert "来源澄清、新输入或重新核实原话可合法纠正理解" in prompt
    assert "当前采用的数量、玩法或时间范围若未被旧检索覆盖" in prompt
    hints = planning_contract()["tool_selection_hints"]
    assert "未知己方人数不阻断只读检索" in hints["read_vs_write_preconditions"]
    assert "工具字段含义不决定原话角色" in hints["read_vs_write_preconditions"]
    assert "独立陈述桌况再委托补齐通常是己方进度" in hints["read_vs_write_preconditions"]
    assert "仍须结合上下文" in hints["read_vs_write_preconditions"]
    consistency = hints["source_interpretation_consistency"]
    assert "同一来源的已知己方人数须贯穿查询和建局" in consistency
    assert "先解释人数角色，再分别投影查询座位与建局参与者" in consistency
    assert "来源澄清、新输入或重新核实原话可合法重判" in consistency
    assert "采用的条件未被旧检索覆盖时先重查再写入" in consistency
    assert "真正未知或存在歧义时澄清，不硬猜人数" in consistency


def test_previous_decision_is_continuity_support_not_fact_or_authorization():
    """上一步提案帮助发现计划漂移，但不能自授权、冒充事实或锁死已发现的误解。"""
    from mahjong_agent_runtime.domains.context_builders.contracts import planning_contract
    prompt = (ROOT / "src/mahjong_agent_runtime/prompts/agent_runtime_system.md").read_text()
    assert "根字段 `previous_decision`" in prompt
    assert "authority=model_proposal_only" in prompt
    assert "不能把先前提案升级为用户事实或授权" in prompt
    assert "用户原话、数据库状态和真实工具结果优先" in prompt
    assert "发现前次解释有误时，回看来源纠正并说明受影响步骤" in prompt
    authority = planning_contract()["tool_selection_hints"]["previous_decision_authority"]
    assert "根字段 previous_decision" in authority
    assert "authority=model_proposal_only" in authority
    assert "提案不能凭空增加用户事实、授权或已执行动作" in authority
    assert "发现错误可回看原话有据纠正" in authority


def test_same_goal_facts_are_updated_by_field_instead_of_reset_by_new_message():
    """事实有效性取决于目标和新证据，不能把本条未重复当成撤销，也不能据事实自动授权。"""
    from mahjong_agent_runtime.domains.context_builders.contracts import output_contract, planning_contract
    prompt = (ROOT / "src/mahjong_agent_runtime/prompts/agent_runtime_system.md").read_text()
    assert "同一目标的有效事实按字段增量更新" in prompt
    assert "补充只更新实际改变的字段" in prompt
    assert "未重提不撤销、不清空、不套用默认值" in prompt
    assert "用户新改、撤销或目标变化优先，事实与动作授权分开判断" in prompt
    assert "current 表示同一任务仍适用的用户陈述，不限于最新消息" in prompt
    assert "须保留真实 source_message_id" in prompt
    retention = planning_contract()["tool_selection_hints"]["known_fact_retention"]
    assert "同一任务仍成立的用户事实合集，不限于 current_message" in retention
    assert "按字段增量更新" in retention
    assert "未重提不撤销、不清空、不套用默认值" in retention
    assert "用户新改、撤销或目标变化优先，事实与动作授权分开判断" in retention
    assertion = output_contract()["objective_state_contract"]["fact_assertions"]
    assert "still-applicable user assertion in this task" in assertion
    assert "including prior user turns with their original source_message_id" in assertion
    assert "not current_message-only" in assertion


def test_persisted_occupancy_does_not_erase_precreation_user_party_facts():
    """建局前人数是有来源的需求输入，实际入局和占座仍只能由持久化 Game 证明。"""
    prompt = (ROOT / "src/mahjong_agent_runtime/prompts/agent_runtime_system.md").read_text()
    assert "`Game.participants/parties` 是已落库局实际人数与占座的唯一事实源" in prompt
    assert "用户对己方人数的有效陈述可作为检索和建局输入" in prompt
    assert "不因尚未入库而失效，也不证明已加入任何现成局" in prompt
