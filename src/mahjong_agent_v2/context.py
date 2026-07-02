from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import ToolResultV2, UserMessageV2
from .store import InMemoryAgentStoreV2
from .tools import ToolGatewayV2, public_game_payload


DEFAULT_V2_PROMPT_PATH = Path(__file__).with_name("prompts").joinpath("agent_v2_system.md")


@dataclass(slots=True)
class BuiltContextV2:
    messages: list[dict[str, str]]
    payload: dict[str, Any]


@dataclass(slots=True)
class ContextBuilderV2:
    store: InMemoryAgentStoreV2
    tool_gateway: ToolGatewayV2
    prompt_path: Path = DEFAULT_V2_PROMPT_PATH
    recent_turn_limit: int = 12

    def build(
        self,
        message: UserMessageV2,
        *,
        trace_id: str,
        previous_tool_results: list[ToolResultV2] | None = None,
    ) -> BuiltContextV2:
        prompt = self.prompt_path.read_text(encoding="utf-8")
        sender_profile = self.store.customers.get(message.sender_id)
        payload = {
            "runtime": "agent_runtime_v2",
            "trace_id": trace_id,
            "current_message": message.to_dict(),
            "recent_conversation": [
                turn.to_dict() for turn in self.store.recent_turns(message.conversation_id, self.recent_turn_limit)
            ],
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
                ],
            },
        }
        return BuiltContextV2(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
            ],
            payload=payload,
        )
