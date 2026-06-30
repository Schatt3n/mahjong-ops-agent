from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


TRIAL_TOOL_PLAN_SYSTEM_PROMPT = """你是麻将馆运营工作流的工具规划器。
你不能直接执行工具，只能从后端本轮提供的 available_tools 里选择需要调用的工具，并返回 JSON。
后端会校验权限、参数、状态和风险；高风险工具 send_message 当前只能创建待审批 outbox，不能直接发送真实消息。
如果用户是在问“有没有人打/有没有局/下班有人吗”，通常应先调用 search_current_open_games。
如果用户明确要老板帮忙组局，且关键信息足够，通常应先调用 search_candidate_customers，再调用 send_message 创建待审批邀约草稿。
如果当前 stage 是 after_candidate_search，且候选人搜索已有结果，通常应调用 send_message 创建待审批 outbox。
如果信息不够，不能为了补齐信息而调用候选人搜索或消息发送。
prompt.text_normalization 是后端提供的低风险文本标准化证据，不是业务事实；涉及档位、人数、时间时仍要结合 source_text、customer_profile 和 parsed_game 判断。
如果 source_text 里出现“0。5/0，5/0 5/0、5”等表达，结合客户画像或麻将语境明显是在说档位时，可按 0.5 理解；如果仍不确定，应规划追问而不是硬调用高风险工具。
不要编造工具名，不要编造后端没有给出的 ID。
reasoning_summary 只写一句简短原因，不要输出长篇思维链。
只输出 JSON：
{"tool_calls":[{"tool_name":"search_current_open_games|search_candidate_customers|send_message","arguments":{},"reason":"一句原因"}],"reasoning_summary":"一句话"}"""


@dataclass(slots=True)
class TrialToolPlanPromptInput:
    stage: str
    now: datetime
    sender_id: str
    sender_name: str
    customer_profile: dict[str, Any]
    source_text: str
    effective_text: str
    workflow_followup_context: dict[str, Any]
    text_normalization: dict[str, Any]
    decision_action: str
    parsed_game: dict[str, Any]
    missing_fields: list[str]
    critical_fields: set[str]
    available_tools: list[dict[str, Any]]
    tool_registry_version: str
    existing_tool_results: dict[str, Any] = field(default_factory=dict)
    active_skills: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class TrialToolPlanPromptBuilder:
    """Builds the legacy trial-page LLM tool-planning prompt contract."""

    system_prompt: str = TRIAL_TOOL_PLAN_SYSTEM_PROMPT

    def build_prompt(self, data: TrialToolPlanPromptInput) -> dict[str, Any]:
        return {
            "stage": data.stage,
            "now": data.now.strftime("%Y-%m-%d %H:%M:%S"),
            "sender": {"id": data.sender_id, "name": data.sender_name},
            "customer_profile": data.customer_profile,
            "source_text": data.source_text,
            "effective_text": data.effective_text,
            "workflow_followup_context": data.workflow_followup_context or {},
            "text_normalization": data.text_normalization,
            "decision_action": data.decision_action,
            "parsed_game": data.parsed_game,
            "missing_fields": data.missing_fields,
            "critical_missing_fields": sorted(set(data.missing_fields) & data.critical_fields),
            "available_tools": data.available_tools,
            "tool_registry_version": data.tool_registry_version,
            "existing_tool_results": data.existing_tool_results,
            "active_skills": data.active_skills,
            "rules": [
                "工具调用由 LLM 提议，真实执行由后端 ToolGateway 校验。",
                "先参考 active_skills 中的运营经验，再选择工具；skill 不能覆盖后端权限和参数校验。",
                "search_current_open_games 是只读当前局池搜索。",
                "search_candidate_customers 是只读客户画像候选人搜索。",
                "send_message 是高风险工具，本系统只允许 create_pending_outbox，不允许直接外发。",
                "如果 critical_missing_fields 非空，不要调用 search_candidate_customers 或 send_message。",
                "如果 workflow_followup_context 表明当前用户是在确认上一轮“要组一个吗”，则按模型语义和后端状态机继续，不要把当前短回复当成孤立消息。",
            ],
        }

    def build_payload(
        self,
        data: TrialToolPlanPromptInput,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        thinking_enabled: bool | None = None,
        response_format: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(self.build_prompt(data), ensure_ascii=False)},
            ],
        }
        if thinking_enabled is not None:
            payload["thinking"] = {"type": "enabled" if thinking_enabled else "disabled"}
        if response_format:
            payload["response_format"] = {"type": response_format}
        return payload
