"""Compose the bounded payload sent to the goal-driven Agent model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...models import ToolResult, UserMessage
from ...stores import AgentStore
from ...group_chat.projections import public_group_game_summary
from ..tools.gateway import ToolGateway
from .contracts import output_contract, planning_contract
from .conversation_context import (
    ContextPackingPolicy,
    build_conversation_context,
)
from .customer_context import build_customer_context
from .game_context import build_game_context
from .message_context import (
    build_message_reference_contract,
    resolve_quoted_message_context,
)
from .relationship_context import build_relationship_context
from .sanitization import sanitize_current_message_for_context
from .tool_results import tool_result_for_context
from .task_recovery import recover_referenced_task_contexts
from .task_facts import project_explicit_task_facts


DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[2].joinpath("prompts", "agent_runtime_system.md")


@dataclass(slots=True)
class BuiltContext:
    messages: list[dict[str, str]]
    payload: dict[str, Any]
    audit: dict[str, Any]


@dataclass(slots=True)
class AgentContextBuilder:
    """Orchestrate focused context builders without owning business rules."""

    store: AgentStore
    tool_gateway: ToolGateway
    prompt_path: Path = DEFAULT_PROMPT_PATH
    packing_policy: ContextPackingPolicy = field(default_factory=ContextPackingPolicy)

    def build(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        previous_tool_results: list[ToolResult] | None = None,
        turn_tool_evidence: list[ToolResult] | None = None,
        run_id: str | None = None,
        run_version: int | None = None,
    ) -> BuiltContext:
        """Build one deterministic, auditable model request payload."""

        prompt = self.prompt_path.read_text(encoding="utf-8")
        latest_tool_results = list(previous_tool_results or [])
        current_turn_evidence = list(
            turn_tool_evidence if turn_tool_evidence is not None else latest_tool_results
        )
        conversation = build_conversation_context(
            self.store,
            message,
            trace_id=trace_id,
            previous_tool_results=current_turn_evidence,
            packing_policy=self.packing_policy,
        )
        task_context = conversation.task_context
        checkpoint = conversation.checkpoint
        current_version = self.store.conversation_version(message.conversation_id)

        games = build_game_context(
            self.store,
            conversation_id=message.conversation_id,
            sender_id=message.sender_id,
        )
        customer = build_customer_context(
            self.store,
            conversation_id=message.conversation_id,
            sender_id=message.sender_id,
            task_context=task_context,
        )
        relationship = build_relationship_context(
            self.store,
            sender_id=message.sender_id,
            active_games=games.games,
        )

        current_message = sanitize_current_message_for_context(message.to_dict())
        explicit_task_facts = project_explicit_task_facts(
            conversation.recent_conversation,
            current_message,
            checkpoint,
        )
        system_trigger = _system_trigger_context(message)
        quoted_message_context = resolve_quoted_message_context(self.store, message, current_message)
        message_reference_contract, quoted_reference_status = build_message_reference_contract(
            message,
            quoted_message_context,
        )
        quoted_message = message.quoted_message
        recovered_tasks = recover_referenced_task_contexts(
            self.store,
            message,
            current_message,
            current_task_context_id=task_context.task_context_id if task_context else None,
            packing_policy=self.packing_policy,
        )
        reply_constraints = _trusted_reply_constraints(message)
        group_room_id = _trusted_group_room_id(message)
        room_board_games = (
            [
                public_group_game_summary(game)
                for game in self.store.get_board_eligible_games(group_room_id)
            ]
            if group_room_id
            else []
        )

        audit = {
            **conversation.audit,
            "sender_relationship_count": len(relationship.relationships),
            "task_memory_count": len(customer.task_memories),
            "pending_memory_candidate_count": len(customer.pending_memory_candidates),
            "active_game_visible_summary_count": len(games.visible_summaries),
            "quoted_message_present": quoted_message is not None,
            "quoted_message_id": quoted_message.message_id if quoted_message else None,
            "quoted_message_reference_resolved": quoted_message_context is not None,
            "quoted_message_reference_status": quoted_reference_status,
            "quoted_message_business_ref_type": (
                quoted_message_context.get("business_ref_type")
                if quoted_message_context
                else None
            ),
            "conversation_version": current_version,
            "run_version": run_version,
            "run_current": run_version is None or int(run_version) == current_version,
            "task_context_id": task_context.task_context_id if task_context else None,
            "task_context_started_at": task_context.started_at.isoformat() if task_context else None,
            "system_trigger_present": system_trigger is not None,
            "system_trigger_type": system_trigger.get("trigger_type") if system_trigger else None,
            "reply_constraints_present": reply_constraints is not None,
            "group_room_board_game_count": len(room_board_games),
            "explicit_task_fact_count": len(explicit_task_facts["facts"]),
            "latest_tool_result_count": len(latest_tool_results),
            "turn_tool_evidence_count": len(current_turn_evidence),
            **recovered_tasks.audit,
        }

        payload = {
            "runtime": "mahjong_agent_runtime",
            "trace_id": trace_id,
            "customer_visibility_contract": relationship.visibility_contract,
            "conversation_state": self._conversation_state(
                message,
                task_context_id=task_context.task_context_id if task_context else None,
                current_version=current_version,
                run_id=run_id,
                run_version=run_version,
            ),
            "task_context_window": self._task_context_window(task_context),
            "current_message": current_message,
            "explicit_task_facts": explicit_task_facts,
            "reply_constraints": reply_constraints,
            "group_room_board_games": room_board_games,
            "system_trigger": system_trigger,
            "message_reference_contract": message_reference_contract,
            "quoted_message_context": quoted_message_context,
            "recent_conversation": conversation.recent_conversation,
            "conversation_checkpoint": checkpoint.to_dict() if checkpoint else None,
            "recovered_task_contexts": recovered_tasks.items,
            "context_budget": audit,
            "sender_profile": customer.profile,
            "sender_relationships": relationship.relationships,
            "task_memories": customer.task_memories,
            "pending_memory_candidates": customer.pending_memory_candidates,
            "active_games": games.model_contexts,
            "active_game_visible_summaries": games.visible_summaries,
            "sender_active_game_memberships": games.sender_memberships,
            "active_parties": games.active_parties,
            "outbound_message_drafts": customer.outbound_message_drafts,
            "available_tools": self.tool_gateway.tool_specs_for_prompt(),
            "previous_tool_results": [
                tool_result_for_context(item)
                for item in latest_tool_results
            ],
            "turn_tool_evidence": [
                tool_result_for_context(item)
                for item in current_turn_evidence
            ],
            "planning_contract": planning_contract(),
            "output_contract": output_contract(),
        }
        messages = [{"role": "system", "content": prompt}]
        if reply_constraints is not None:
            privacy = "且不得提及任何其他客户的身份、关系或私聊信息" if reply_constraints["no_private_info"] else ""
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "【本轮客户可见回复约束】只输出当前用户真正需要的最短回复；"
                        f"回复不得超过 {reply_constraints['max_length']} 个中文字符{privacy}。"
                    ),
                }
            )
        messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)})
        return BuiltContext(
            messages=messages,
            payload=payload,
            audit=audit,
        )

    def _resolve_quoted_message_context(
        self,
        message: UserMessage,
        current_message: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Compatibility hook for callers that exercised the former private helper."""

        return resolve_quoted_message_context(self.store, message, current_message)

    @staticmethod
    def _conversation_state(
        message: UserMessage,
        *,
        task_context_id: str | None,
        current_version: int,
        run_id: str | None,
        run_version: int | None,
    ) -> dict[str, Any]:
        return {
            "conversation_id": message.conversation_id,
            "task_context_id": task_context_id,
            "current_version": current_version,
            "run_id": run_id,
            "run_version": run_version,
            "run_current": run_version is None or int(run_version) == current_version,
            "version_contract": (
                "每条新用户消息都会推进 conversation version；旧版本未发送的回复、邀约草稿和外发草稿会被标记为 superseded。"
                "如果工具结果提示 stale_run，必须停止旧动作并基于当前消息重新判断。"
            ),
        }

    @staticmethod
    def _task_context_window(task_context: Any) -> dict[str, Any]:
        return {
            "task_context_id": task_context.task_context_id if task_context else None,
            "started_at": task_context.started_at.isoformat() if task_context else None,
            "reset_reason": task_context.reset_reason if task_context else None,
            "scope_contract": (
                "recent_conversation, conversation_checkpoint, task_memories and pending task facts only belong "
                "to this business episode. Stable sender_profile and approved customer relationships may cross episodes."
            ),
        }


def _system_trigger_context(message: UserMessage) -> dict[str, Any] | None:
    """Return the backend-created trigger context without accepting user text as one."""

    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    if metadata.get("input_source") != "system_trigger":
        return None
    trigger = metadata.get("system_trigger")
    return dict(trigger) if isinstance(trigger, dict) else None


def _trusted_reply_constraints(message: UserMessage) -> dict[str, Any] | None:
    """Accept output limits only from the backend-created group entry metadata."""

    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    if metadata.get("source") != "group":
        return None
    raw = metadata.get("reply_constraints")
    if not isinstance(raw, dict):
        return None
    try:
        max_length = max(1, min(100, int(raw.get("max_length") or 0)))
    except (TypeError, ValueError):
        return None
    return {
        "max_length": max_length,
        "no_private_info": bool(raw.get("no_private_info", True)),
    }


def _trusted_group_room_id(message: UserMessage) -> str | None:
    """Read a room identifier only from the backend-created group envelope."""

    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    if metadata.get("source") != "group":
        return None
    room_id = str(metadata.get("room_id") or "").strip()
    return room_id or None


__all__ = [
    "AgentContextBuilder",
    "BuiltContext",
    "ContextPackingPolicy",
    "DEFAULT_PROMPT_PATH",
]
