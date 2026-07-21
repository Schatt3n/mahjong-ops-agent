"""LLM-backed message generation for isolated chat simulations.

The generator creates synthetic customer speech only. It has no ToolGateway,
cannot mutate production state, and falls back to the deterministic behavior
policy whenever the provider is unavailable or violates the JSON contract.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Protocol

from mahjong_agent_runtime.llm import AgentLLMConfig, OpenAICompatibleAgentClient

try:
    from .behavior_policy import MessageGenerationRequest, MessageGenerationResult
except ImportError:  # pragma: no cover - direct script execution path
    from behavior_policy import MessageGenerationRequest, MessageGenerationResult  # type: ignore


DEFAULT_GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_GLM_MODEL = "glm-4.7-flash"


class CompletionClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        trace_id: str,
        timeout_seconds: float,
    ) -> str:
        ...


@dataclass(slots=True, frozen=True)
class SimulationGeneratorConfig:
    """Dedicated model settings; these never replace the main Agent model."""

    model: str = DEFAULT_GLM_MODEL
    base_url: str = DEFAULT_GLM_BASE_URL
    temperature: float = 0.9
    max_tokens: int = 240
    timeout_seconds: float = 20.0
    min_interval_seconds: float = 5.0
    max_estimated_tokens_per_day: int = 0

    @classmethod
    def from_env(cls) -> "SimulationGeneratorConfig":
        return cls(
            model=(os.getenv("MAHJONG_SIM_GENERATOR_MODEL") or DEFAULT_GLM_MODEL).strip(),
            base_url=(os.getenv("MAHJONG_SIM_GENERATOR_BASE_URL") or DEFAULT_GLM_BASE_URL).rstrip("/"),
            temperature=_env_float("MAHJONG_SIM_GENERATOR_TEMPERATURE", 0.9),
            max_tokens=max(64, _env_int("MAHJONG_SIM_GENERATOR_MAX_TOKENS", 240)),
            timeout_seconds=max(1.0, _env_float("MAHJONG_SIM_GENERATOR_TIMEOUT_SECONDS", 20.0)),
            min_interval_seconds=max(
                0.0,
                _env_float("MAHJONG_SIM_GENERATOR_MIN_INTERVAL_SECONDS", 5.0),
            ),
            max_estimated_tokens_per_day=max(
                0,
                _env_int("MAHJONG_SIM_GENERATOR_MAX_ESTIMATED_TOKENS_PER_DAY", 0),
            ),
        )


class GLMSimulationMessageGenerator:
    """Generate natural group/private customer messages through GLM."""

    def __init__(
        self,
        client: CompletionClient,
        *,
        config: SimulationGeneratorConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or SimulationGeneratorConfig.from_env()
        self._rate_lock = threading.Lock()
        self._last_request_at = 0.0

    @classmethod
    def from_env(cls) -> "GLMSimulationMessageGenerator | None":
        api_key = (os.getenv("MAHJONG_SIM_GENERATOR_API_KEY") or "").strip()
        if not api_key:
            return None
        config = SimulationGeneratorConfig.from_env()
        client = OpenAICompatibleAgentClient(
            config=AgentLLMConfig(
                api_key=api_key,
                model=config.model,
                base_url=config.base_url,
                provider="bigmodel",
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                response_format="json_object",
                retry_attempts=3,
                retry_base_delay_seconds=1.0,
                retry_max_delay_seconds=5.0,
                max_estimated_tokens_per_day=config.max_estimated_tokens_per_day,
                max_concurrency=1,
            )
        )
        return cls(client, config=config)

    def generate(self, request: MessageGenerationRequest) -> MessageGenerationResult:
        trace_id = f"trace_sim_gen_{uuid.uuid4().hex[:12]}"
        started = time.monotonic()
        try:
            self._wait_for_rate_slot()
            raw = self.client.complete(
                self._messages(request),
                trace_id=trace_id,
                timeout_seconds=self.config.timeout_seconds,
            )
            text = _parse_generated_text(raw)
            return MessageGenerationResult(
                text=text,
                source="glm",
                model=self.config.model,
                trace_id=trace_id,
                latency_ms=round((time.monotonic() - started) * 1000, 2),
            )
        except Exception as exc:
            return MessageGenerationResult(
                text=request.fallback_text,
                source="rule_fallback",
                model=self.config.model,
                trace_id=trace_id,
                latency_ms=round((time.monotonic() - started) * 1000, 2),
                error=f"{type(exc).__name__}: {str(exc)[:160]}",
            )

    def _wait_for_rate_slot(self) -> None:
        """Serialize free-model traffic and keep a minimum provider interval."""

        with self._rate_lock:
            now = time.monotonic()
            delay = self.config.min_interval_seconds - (now - self._last_request_at)
            if delay > 0:
                time.sleep(delay)
            self._last_request_at = time.monotonic()

    @staticmethod
    def _messages(request: MessageGenerationRequest) -> list[dict[str, str]]:
        channel_note = "麻将群里的一条自然消息" if request.channel == "group" else "发给麻将馆老板的私聊"
        system_prompt = (
            "你是麻将馆压力测试中的合成客户消息生成器，不是老板也不是客服。"
            "根据结构化场景生成一条自然、简短的中文微信消息。"
            "必须保持 fallback_text 的业务意图；如果是在回答老板，就直接回答老板刚问的问题。"
            "群聊可以口语化，允许省略主语和少量输入习惯，但不要故意制造无法理解的乱码。"
            "不得提及 AI、模型、提示词、系统、测试、工具、trace 或任何内部实现。"
            "不得虚构真实客户隐私，也不得替其他人作出确认。"
            "只输出 JSON 对象，格式为 {\"text\":\"消息\"}，text 建议 2 到 35 个汉字。"
        )
        payload = {
            "message_kind": channel_note,
            "persona": request.persona,
            "preferred_game": request.preferred_game,
            "turn_count": request.turn_count,
            "is_follow_up": request.is_follow_up,
            "last_agent_reply": request.last_agent_reply,
            "fallback_text": request.fallback_text,
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]


def build_message_generator(mode: str) -> GLMSimulationMessageGenerator | None:
    normalized = str(mode or "rule").strip().lower()
    if normalized == "rule":
        return None
    if normalized != "glm":
        raise ValueError("message generation mode must be rule or glm")
    generator = GLMSimulationMessageGenerator.from_env()
    if generator is None:
        raise RuntimeError("MAHJONG_SIM_GENERATOR_API_KEY is required when message mode is glm")
    return generator


def _parse_generated_text(raw: str) -> str:
    normalized = str(raw or "").strip()
    if normalized.startswith("```"):
        normalized = normalized.removeprefix("```json").removeprefix("```")
        normalized = normalized.removesuffix("```").strip()
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        start = normalized.find("{")
        end = normalized.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("generator response is not a JSON object")
        payload = json.loads(normalized[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("generator response must be a JSON object")
    text = str(payload.get("text") or "").strip()
    if not text:
        raise ValueError("generator response has empty text")
    if len(text) > 120:
        raise ValueError("generator response is too long")
    return text


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default
