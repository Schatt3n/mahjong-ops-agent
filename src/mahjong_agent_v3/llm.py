"""Compatibility LLM imports for historical ``mahjong_agent_v3`` users."""

from __future__ import annotations

from mahjong_agent_runtime.llm import (
    AgentLLMClient as AgentLLMClientV3,
    AgentLLMConfig as AgentLLMConfigV3,
    OpenAICompatibleAgentClient as OpenAICompatibleAgentClientV3,
    StaticAgentClient as StaticAgentClientV3,
    content_from_response,
    default_base_url,
    env_float,
    env_int,
    http_error_note,
)

__all__ = [
    "AgentLLMClientV3",
    "AgentLLMConfigV3",
    "OpenAICompatibleAgentClientV3",
    "StaticAgentClientV3",
    "content_from_response",
    "default_base_url",
    "env_float",
    "env_int",
    "http_error_note",
]
