from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import ConversationTurn, ToolResult, UserMessage
from .store import InMemoryAgentStore
from .tools import ToolGateway


DEFAULT_PROMPT_PATH = Path(__file__).with_name("prompts").joinpath("agent_runtime_system.md")


@dataclass(slots=True)
class BuiltContext:
    messages: list[dict[str, str]]
    payload: dict[str, Any]
    audit: dict[str, Any]


@dataclass(slots=True)
class ContextPackingPolicy:
    max_turns_considered: int = 60
    max_recent_conversation_tokens: int = 4_000

    def pack_turns(self, turns: list[ConversationTurn]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        considered = list(turns)[-self.max_turns_considered :]
        included_reversed: list[dict[str, Any]] = []
        estimated_tokens = 0
        omitted_for_budget = 0
        for turn in reversed(considered):
            payload = turn.to_dict()
            turn_tokens = estimate_tokens(payload)
            if included_reversed and estimated_tokens + turn_tokens > self.max_recent_conversation_tokens:
                omitted_for_budget += 1
                continue
            included_reversed.append(payload)
            estimated_tokens += turn_tokens
        included = list(reversed(included_reversed))
        omitted_before_window = max(0, len(turns) - len(considered))
        audit = {
            "total_turns_available": len(turns),
            "included_turn_count": len(included),
            "omitted_turn_count": omitted_before_window + omitted_for_budget,
            "omitted_before_window": omitted_before_window,
            "omitted_for_budget": omitted_for_budget,
            "estimated_recent_conversation_tokens": estimated_tokens,
        }
        return included, audit


@dataclass(slots=True)
class AgentContextBuilder:
    store: InMemoryAgentStore
    tool_gateway: ToolGateway
    prompt_path: Path = DEFAULT_PROMPT_PATH
    packing_policy: ContextPackingPolicy = field(default_factory=ContextPackingPolicy)

    def build(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        previous_tool_results: list[ToolResult] | None = None,
        run_id: str | None = None,
        run_version: int | None = None,
    ) -> BuiltContext:
        prompt = self.prompt_path.read_text(encoding="utf-8")
        recent_conversation, audit = self.packing_policy.pack_turns(
            self.store.recent_turns(message.conversation_id, self.packing_policy.max_turns_considered)
        )
        profile = self.store.customers.get(message.sender_id)
        checkpoint = self.store.get_conversation_checkpoint(message.conversation_id)
        current_version = self.store.conversation_version(message.conversation_id)
        active_games = self.store.active_games(message.conversation_id)
        active_game_visible_summaries = [active_game_visible_summary(item) for item in active_games]
        sender_relationships = self.store.relationship_context_for_sender(message.sender_id, active_games)
        current_message = message.to_dict()
        quoted_message_context = self._resolve_quoted_message_context(message, current_message)
        audit = {
            **audit,
            "conversation_checkpoint_present": checkpoint is not None,
            "conversation_checkpoint_source_trace_id": checkpoint.source_trace_id if checkpoint else None,
            "sender_relationship_count": len(sender_relationships),
            "active_game_visible_summary_count": len(active_game_visible_summaries),
            "quoted_message_reference_resolved": quoted_message_context is not None,
            "quoted_message_business_ref_type": quoted_message_context.get("business_ref_type") if quoted_message_context else None,
            "conversation_version": current_version,
            "run_version": run_version,
            "run_current": run_version is None or int(run_version) == current_version,
        }
        payload = {
            "runtime": "mahjong_agent_runtime",
            "trace_id": trace_id,
            "conversation_state": {
                "conversation_id": message.conversation_id,
                "current_version": current_version,
                "run_id": run_id,
                "run_version": run_version,
                "run_current": run_version is None or int(run_version) == current_version,
                "version_contract": (
                    "每条新用户消息都会推进 conversation version；旧版本未发送的回复、邀约草稿和外发草稿会被标记为 superseded。"
                    "如果工具结果提示 stale_run，必须停止旧动作并基于当前消息重新判断。"
                ),
            },
            "current_message": current_message,
            "quoted_message_context": quoted_message_context,
            "recent_conversation": recent_conversation,
            "conversation_checkpoint": checkpoint.to_dict() if checkpoint else None,
            "context_budget": audit,
            "sender_profile": profile.to_dict() if profile else None,
            "sender_relationships": sender_relationships,
            "active_games": [item.to_dict() for item in active_games],
            "active_game_visible_summaries": active_game_visible_summaries,
            "active_parties": [
                {
                    "game_id": game.game_id,
                    "parties": [party.to_dict() for party in game.parties],
                    "seat_claims": game.seat_claims(),
                    "seat_summary": game.seat_summary(),
                }
                for game in active_games
            ],
            "outbound_message_drafts": [item.to_dict() for item in self.store.outbound_message_drafts.values()],
            "available_tools": self.tool_gateway.tool_specs_for_prompt(),
            "previous_tool_results": [item.to_dict() for item in previous_tool_results or []],
            "output_contract": output_contract(),
        }
        return BuiltContext(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
            ],
            payload=payload,
            audit=audit,
        )

    def _resolve_quoted_message_context(
        self,
        message: UserMessage,
        current_message: dict[str, Any],
    ) -> dict[str, Any] | None:
        quoted = message.quoted_message
        if quoted is None or not quoted.message_id:
            return None
        resolver = getattr(self.store, "resolve_message_reference", None)
        if not callable(resolver):
            return None
        reference = resolver(
            conversation_id=quoted.conversation_id or message.conversation_id,
            message_id=quoted.message_id,
        )
        if reference is None:
            return None
        reference_payload = reference.to_dict()
        quoted_payload = dict(current_message.get("quoted_message") or quoted.to_dict())
        quoted_payload["business_ref_type"] = quoted_payload.get("business_ref_type") or reference.business_ref_type
        quoted_payload["business_ref_id"] = quoted_payload.get("business_ref_id") or reference.business_ref_id
        quoted_payload["conversation_id"] = quoted_payload.get("conversation_id") or reference.conversation_id
        quoted_payload["text"] = quoted_payload.get("text") or reference.text
        quoted_payload["metadata"] = {
            **dict(quoted_payload.get("metadata") or {}),
            "resolved_message_reference": {
                "business_ref_type": reference.business_ref_type,
                "business_ref_id": reference.business_ref_id,
                "channel": reference.channel,
                "recipient_id": reference.recipient_id,
                "recipient_name": reference.recipient_name,
                "source": reference.metadata.get("source"),
            },
        }
        current_message["quoted_message"] = quoted_payload
        return reference_payload


def output_contract() -> dict[str, Any]:
    return {
        "format": "json_object",
        "required_keys": [
            "goal",
            "objective_status",
            "reasoning_summary",
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
            "reply_to_user": "string",
            "tool_calls": "array",
            "needs_human": "boolean",
            "stop_reason": "object",
            "badcase": "null; deprecated side-channel, call record_badcase tool instead",
        },
        "stop_reason_contract": {
            "can_stop": "required boolean; false when objective_status=needs_tool, true for terminal statuses",
            "why": "required non-empty string explaining why the agent can stop now or why it must continue with tools",
            "pending_work": "required array of strings; non-empty when can_stop=false",
            "depends_on_tool_results": "required boolean; true if the decision depends on previous_tool_results or system state",
        },
        "tool_call_contract": {
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
            "badcase must be null; badcase/eval writes must use record_badcase tool_call",
        ],
    }


def active_game_visible_summary(game: Any) -> dict[str, Any]:
    requirement = dict(getattr(game, "requirement", {}) or {})
    public_requirement_keys = (
        "user_visible_summary",
        "game_type",
        "stake",
        "base_stake",
        "cap_score",
        "stake_label",
        "smoke_preference",
        "start_time_kind",
        "start_time",
        "duration_kind",
        "duration_hours",
        "known_player_count",
        "needed_seats",
    )
    return {
        "game_id": game.game_id,
        "status": game.status.value,
        "user_visible_summary": str(requirement.get("user_visible_summary") or ""),
        "seat_summary": game.seat_summary(),
        "public_requirement": {
            key: requirement.get(key)
            for key in public_requirement_keys
            if requirement.get(key) is not None
        },
    }


def estimate_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return max(1, len(text) // 4)
