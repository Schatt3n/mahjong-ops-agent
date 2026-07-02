from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import ConversationTurnV3, ToolResultV3, UserMessageV3
from .store import InMemoryAgentStoreV3
from .tools import ToolGatewayV3


DEFAULT_PROMPT_PATH_V3 = Path(__file__).with_name("prompts").joinpath("agent_v3_system.md")


@dataclass(slots=True)
class BuiltContextV3:
    messages: list[dict[str, str]]
    payload: dict[str, Any]
    audit: dict[str, Any]


@dataclass(slots=True)
class ContextPackingPolicyV3:
    max_turns_considered: int = 60
    max_recent_conversation_tokens: int = 4_000

    def pack_turns(self, turns: list[ConversationTurnV3]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
class AgentContextBuilderV3:
    store: InMemoryAgentStoreV3
    tool_gateway: ToolGatewayV3
    prompt_path: Path = DEFAULT_PROMPT_PATH_V3
    packing_policy: ContextPackingPolicyV3 = field(default_factory=ContextPackingPolicyV3)

    def build(
        self,
        message: UserMessageV3,
        *,
        trace_id: str,
        previous_tool_results: list[ToolResultV3] | None = None,
    ) -> BuiltContextV3:
        prompt = self.prompt_path.read_text(encoding="utf-8")
        recent_conversation, audit = self.packing_policy.pack_turns(
            self.store.recent_turns(message.conversation_id, self.packing_policy.max_turns_considered)
        )
        profile = self.store.customers.get(message.sender_id)
        payload = {
            "runtime": "mahjong_agent_v3",
            "trace_id": trace_id,
            "current_message": message.to_dict(),
            "recent_conversation": recent_conversation,
            "context_budget": audit,
            "sender_profile": profile.to_dict() if profile else None,
            "active_games": [item.to_dict() for item in self.store.active_games(message.conversation_id)],
            "available_tools": self.tool_gateway.tool_specs_for_prompt(),
            "previous_tool_results": [item.to_dict() for item in previous_tool_results or []],
            "output_contract": output_contract_v3(),
        }
        return BuiltContextV3(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
            ],
            payload=payload,
            audit=audit,
        )


def output_contract_v3() -> dict[str, Any]:
    return {
        "format": "json_object",
        "required_keys": [
            "goal",
            "objective_status",
            "reasoning_summary",
            "reply_to_user",
            "tool_calls",
            "needs_human",
        ],
        "objective_status_values": ["needs_tool", "waiting_user", "completed", "needs_human", "unknown"],
        "field_types": {
            "goal": "string",
            "objective_status": "string",
            "reasoning_summary": "string",
            "reply_to_user": "string",
            "tool_calls": "array",
            "needs_human": "boolean",
            "badcase": "null; deprecated side-channel, call record_badcase tool instead",
        },
        "invariants": [
            "objective_status=needs_tool requires at least one tool_call",
            "objective_status=waiting_user|completed|needs_human must not include tool_calls",
            "objective_status=needs_human requires needs_human=true",
            "invalid contract means backend will not execute any tool",
            "badcase must be null; badcase/eval writes must use record_badcase tool_call",
        ],
    }


def estimate_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return max(1, len(text) // 4)
