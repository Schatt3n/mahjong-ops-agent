from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence


@dataclass(slots=True)
class TrialToolRequestFactory:
    """Builds stable request contracts for legacy trial-page tool calls."""

    def current_open_games(
        self,
        *,
        called: bool,
        requested_by: str,
        tool_plan_source: Any,
        decision_action: str,
        call_reason: str,
        sender_id: str,
        game_type: str | None,
        level_options: list[str],
        smoke_preference: str | None,
        smoke_options: list[str],
        time_window: Sequence[datetime] | None,
        source_text: str,
    ) -> dict[str, Any]:
        return {
            "tool_name": "search_current_open_games",
            "called": called,
            "requested_by": requested_by,
            "tool_plan_source": tool_plan_source,
            "decision_action": decision_action,
            "call_reason": call_reason,
            "query": {
                "sender_id": sender_id,
                "game_type": game_type,
                "level_options": level_options,
                "smoke_preference": smoke_preference,
                "smoke_options": smoke_options,
                "time_window": [value.isoformat() for value in time_window] if time_window else None,
                "source_text": source_text,
            },
            "hard_filters": [
                "只返回当前未结束且未满的局",
                "不返回当前发送人自己发起的局",
                "不返回已过期局",
                "不返回档位/烟况/玩法硬冲突的局",
            ],
        }

    def candidate_customers(
        self,
        *,
        requested_by: str,
        tool_plan_source: Any,
        game_id: str,
        game_type: str,
        game_label: str,
        level: str,
        start_at: datetime | None,
        rules: list[str],
        missing_count: int,
        organizer_id: str,
        candidate_composition_preference: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "tool_name": "search_candidate_customers",
            "called": True,
            "requested_by": requested_by,
            "tool_plan_source": tool_plan_source,
            "risk_level": "low",
            "approval_required": False,
            "call_reason": "没有匹配到可拼局，且组局关键信息已补齐，需要搜索可邀约候选人。",
            "query": {
                "game_id": game_id,
                "game_type": game_type,
                "game_label": game_label,
                "level": level,
                "start_at": start_at.isoformat() if start_at else None,
                "rules": rules,
                "missing_count": missing_count,
                "organizer_id": organizer_id,
                "candidate_composition_preference": candidate_composition_preference,
            },
            "hard_filters": [
                "不返回当前局发起人",
                "不返回勿扰客户",
                "不返回画像硬冲突且低分客户",
                "疲劳度、最近邀约和响应率进入排序",
                "性别等候选组合偏好是推荐排序约束，不是外发话术内容",
            ],
        }

    def pending_outbox_message(
        self,
        *,
        called: bool,
        requested_by: str,
        tool_plan_source: Any,
        game_id: str,
        recipient_count: int,
    ) -> dict[str, Any]:
        return {
            "tool_name": "send_message",
            "called": called,
            "requested_by": requested_by,
            "tool_plan_source": tool_plan_source,
            "risk_level": "high",
            "approval_required": True,
            "direct_send_allowed": False,
            "execution_mode": "create_pending_outbox",
            "call_reason": "候选人已找到，但当前安全边界不允许自动外发，只能创建待审批发送请求。",
            "query": {
                "game_id": game_id,
                "recipient_count": recipient_count,
                "channel_policy": "老板审批后复制/发送",
            },
            "hard_filters": [
                "禁止直接发送真实消息",
                "所有消息必须先进入待审批 outbox",
                "候选人必须来自 search_candidate_customers 工具结果",
                "消息草稿必须通过隐私和话术 guard",
                "不能透露缺口、发起人、推荐评分和默认杭麻/财敲细分",
            ],
        }
