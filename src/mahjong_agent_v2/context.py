from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import ConversationTurnV2, ToolResultV2, UserMessageV2
from .store import InMemoryAgentStoreV2
from .tools import ToolGatewayV2, public_game_payload


DEFAULT_V2_PROMPT_PATH = Path(__file__).with_name("prompts").joinpath("agent_v2_system.md")


@dataclass(slots=True)
class BuiltContextV2:
    messages: list[dict[str, str]]
    payload: dict[str, Any]
    audit: dict[str, Any]


@dataclass(slots=True)
class ContextPackingPolicyV2:
    """Deterministic context-window packing for V2.

    This is not mahjong semantic logic. It only decides how much historical
    context can fit into the prompt budget and records what was omitted.
    """

    max_turns_considered: int = 50
    max_recent_conversation_tokens: int = 3_000

    def pack_turns(self, turns: list[ConversationTurnV2]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        considered = list(turns)[-self.max_turns_considered :]
        included_reversed: list[dict[str, Any]] = []
        estimated_tokens = 0
        omitted_for_budget = 0
        for turn in reversed(considered):
            payload = turn.to_dict()
            turn_tokens = estimate_context_tokens(payload)
            if included_reversed and estimated_tokens + turn_tokens > self.max_recent_conversation_tokens:
                omitted_for_budget += 1
                continue
            included_reversed.append(payload)
            estimated_tokens += turn_tokens
        included = list(reversed(included_reversed))
        omitted_before_window = max(0, len(turns) - len(considered))
        audit = {
            "max_turns_considered": self.max_turns_considered,
            "max_recent_conversation_tokens": self.max_recent_conversation_tokens,
            "total_turns_available": len(turns),
            "turns_considered": len(considered),
            "included_turn_count": len(included),
            "omitted_turn_count": omitted_before_window + omitted_for_budget,
            "omitted_before_window": omitted_before_window,
            "omitted_for_budget": omitted_for_budget,
            "estimated_recent_conversation_tokens": estimated_tokens,
        }
        return included, audit


@dataclass(slots=True)
class ContextBuilderV2:
    store: InMemoryAgentStoreV2
    tool_gateway: ToolGatewayV2
    prompt_path: Path = DEFAULT_V2_PROMPT_PATH
    packing_policy: ContextPackingPolicyV2 = field(default_factory=ContextPackingPolicyV2)

    def build(
        self,
        message: UserMessageV2,
        *,
        trace_id: str,
        previous_tool_results: list[ToolResultV2] | None = None,
    ) -> BuiltContextV2:
        prompt = self.prompt_path.read_text(encoding="utf-8")
        sender_profile = self.store.customers.get(message.sender_id)
        available_turns = self.store.recent_turns(message.conversation_id, self.packing_policy.max_turns_considered)
        recent_conversation, context_audit = self.packing_policy.pack_turns(available_turns)
        payload = {
            "runtime": "agent_runtime_v2",
            "trace_id": trace_id,
            "current_message": message.to_dict(),
            "recent_conversation": recent_conversation,
            "context_budget": context_audit,
            "sender_profile": sender_profile.to_dict() if sender_profile else None,
            "active_games": [public_game_payload(game) for game in self.store.active_games(message.conversation_id)],
            "available_tools": self.tool_gateway.tool_specs_for_prompt(),
            "previous_tool_results": [result.to_dict() for result in previous_tool_results or []],
            "output_contract": {
                "format": "json_object",
                "required_keys": [
                    "goal",
                    "reasoning_summary",
                    "reply_to_user",
                    "tool_calls",
                    "needs_human",
                    "objective_status",
                ],
            },
        }
        return BuiltContextV2(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
            ],
            payload=payload,
            audit=context_audit,
        )


def estimate_context_tokens(value: Any) -> int:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    return max(1, len(text) // 4)
