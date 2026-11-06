"""统一生产、模拟与真实评测的话术审查配置，独立预算不改变主规划模型。"""

from __future__ import annotations

from dataclasses import replace
import math
import os
from typing import Any

from .llm import AgentLLMConfig, OpenAICompatibleAgentClient, default_base_url, optional_positive_env_float


def _review_integer(name: str, default: int, *, minimum: int = 1) -> int:
    """显式预算必须合法，配置错误不能静默退回另一套实验参数。"""
    raw = os.getenv(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _review_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    """校验独立审查浮点配置，拒绝非有限值及负数。"""
    raw = os.getenv(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be finite and >= {minimum}") from exc
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def reply_review_timeout_seconds() -> float:
    """审查使用独立端到端期限；此预算同时覆盖排队、HTTP读取及有限重试。"""
    value = optional_positive_env_float("MAHJONG_REPLY_REVIEW_LLM_TIMEOUT_SECONDS")
    return 180.0 if value is None else value


def build_reply_review_client(base_config: AgentLLMConfig | None = None) -> OpenAICompatibleAgentClient | None:
    """复制基础连接配置并应用审查专用选项，绝不修改主模型客户端或其配置。

    默认复用模型身份，采用低推理强度和独立完成预算；提供方只接收其支持的
    参数。未提供基础配置时读取标准环境，仍允许完整独立的审查连接配置。
    """
    base = base_config or AgentLLMConfig.from_env()
    model = os.getenv("MAHJONG_REPLY_REVIEW_LLM_MODEL") or (base.model if base else None)
    if not model:
        return None
    provider = (os.getenv("MAHJONG_REPLY_REVIEW_LLM_PROVIDER")
                or (base.provider if base else os.getenv("MAHJONG_LLM_PROVIDER", "openai_compatible"))).strip().lower()
    api_key = (os.getenv("MAHJONG_REPLY_REVIEW_LLM_API_KEY") or (base.api_key if base else None)
               or os.getenv("MAHJONG_LLM_API_KEY") or os.getenv("MAHJONG_DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY"))
    if not api_key:
        raise ValueError("reply review model is configured without an API credential")
    base_url = (os.getenv("MAHJONG_REPLY_REVIEW_LLM_BASE_URL")
                or (base.base_url if base and provider == base.provider else default_base_url(provider))).rstrip("/")
    if base is None:
        base = AgentLLMConfig(api_key=api_key, model=model, provider=provider, base_url=base_url)
    # 单次上限不能继承主模型的较短值；默认由独立总期限为传输重试均分预算。
    config = replace(
        base, api_key=api_key, model=model, provider=provider, base_url=base_url,
        temperature=_review_float("MAHJONG_REPLY_REVIEW_LLM_TEMPERATURE", 0.0),
        max_tokens=_review_integer("MAHJONG_REPLY_REVIEW_LLM_MAX_COMPLETION_TOKENS", 16384),
        response_format="json_object",
        thinking_mode=os.getenv("MAHJONG_REPLY_REVIEW_LLM_THINKING_MODE", "enabled"),
        reasoning_effort=os.getenv("MAHJONG_REPLY_REVIEW_LLM_REASONING_EFFORT", "low"),
        retry_attempts=_review_integer("MAHJONG_REPLY_REVIEW_LLM_RETRY_ATTEMPTS", 3),
        retry_base_delay_seconds=_review_float("MAHJONG_REPLY_REVIEW_LLM_RETRY_BASE_DELAY_SECONDS", 0.5),
        retry_max_delay_seconds=_review_float("MAHJONG_REPLY_REVIEW_LLM_RETRY_MAX_DELAY_SECONDS", 4.0),
        max_concurrency=_review_integer("MAHJONG_REPLY_REVIEW_LLM_MAX_CONCURRENCY", base.max_concurrency),
        max_estimated_tokens_per_day=_review_integer(
            "MAHJONG_REPLY_REVIEW_LLM_MAX_ESTIMATED_TOKENS_PER_DAY", base.max_estimated_tokens_per_day, minimum=0),
        attempt_timeout_seconds=optional_positive_env_float("MAHJONG_REPLY_REVIEW_LLM_ATTEMPT_TIMEOUT_SECONDS"),
    )
    return OpenAICompatibleAgentClient(config=config)


def reply_review_runtime_config(client: Any, timeout_seconds: float) -> dict[str, Any]:
    """只投影可核验的审查请求预算；认证、地址及任意客户端字段均不可进入报告。"""
    config = getattr(client, "config", None)
    provider = getattr(config, "provider", None)
    thinking = getattr(config, "thinking_mode", None) if provider == "deepseek" else "disabled" if provider in {"zai", "glm", "bigmodel"} else None
    return {
        "provider": provider,
        "model": getattr(config, "model", None),
        "thinking_mode": thinking,
        "reasoning_effort": getattr(config, "reasoning_effort", None) if thinking == "enabled" else None,
        "max_completion_tokens": getattr(config, "max_tokens", None),
        "response_format": getattr(config, "response_format", None),
        "total_timeout_seconds": timeout_seconds,
        "attempt_timeout_seconds": getattr(config, "attempt_timeout_seconds", None),
        "retry_attempts": getattr(config, "retry_attempts", None),
        "max_concurrency": getattr(config, "max_concurrency", None),
    }
