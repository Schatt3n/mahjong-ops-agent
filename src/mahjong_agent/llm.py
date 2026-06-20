from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from .models import Message


@dataclass(slots=True)
class LLMConfig:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    provider: str = "openai"
    timeout_seconds: float = 8.0
    temperature: float = 0.1

    @classmethod
    def from_env(cls) -> "LLMConfig | None":
        provider = os.getenv("MAHJONG_LLM_PROVIDER", "").strip().lower()
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        api_key = os.getenv("MAHJONG_LLM_API_KEY") or dashscope_key or os.getenv("OPENAI_API_KEY")
        if not provider:
            provider = "qwen" if dashscope_key else "openai"
        defaults = _provider_defaults(provider)
        model = os.getenv("MAHJONG_LLM_MODEL") or defaults.get("model")
        if not api_key or not model:
            return None
        return cls(
            api_key=api_key,
            model=model,
            base_url=os.getenv("MAHJONG_LLM_BASE_URL", defaults.get("base_url", "https://api.openai.com/v1")).rstrip("/"),
            provider=provider,
            timeout_seconds=float(os.getenv("MAHJONG_LLM_TIMEOUT_SECONDS", "8")),
            temperature=float(os.getenv("MAHJONG_LLM_TEMPERATURE", "0.1")),
        )


@dataclass(slots=True)
class LLMResolution:
    is_mahjong_related: bool
    intent: str = "uncertain"
    confidence: float = 0.0
    normalized_text: str | None = None
    reply_text: str | None = None
    needs_human_review: bool = False
    facts: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class LLMResolver(Protocol):
    def resolve(self, message: Message, context: dict[str, Any] | None = None) -> LLMResolution:
        ...


class OpenAICompatibleLLMResolver:
    """Small OpenAI-compatible JSON resolver.

    It is deliberately narrow: the model can interpret a message and propose a
    normalized text, but it cannot mutate state or send messages directly.
    """

    system_prompt = """你是棋牌室运营 Agent 的语义解析器。
只判断用户消息和麻将馆运营是否相关，并把本地行话归一化。
你不能直接安排座位，不能承诺已发送邀约，不能处理资金结算。

已知本店玩法：
- cq = 杭麻里的财敲，不是重庆麻将。
- 371 = 三缺一，272 = 二缺二，173 = 一缺三。
- 川麻216 = 川麻 2-16 档，底注 2，封顶 16。
- 川麻1-32 = 川麻 1-32 档，底注 1，封顶 32。
- 半块、半、五毛 = 0.5 档。
- 不抽、不抽烟、无烟 = 无烟局；不要把“不抽”单独理解为抽水。
- 可识别玩法包括杭麻/财敲、川麻、幺鸡、素鸡、幺鸡47、红中麻将、捉鸡麻将、湖南麻将、重庆麻将。

只输出 JSON，不要输出解释文字。格式：
{
  "is_mahjong_related": true,
  "intent": "find_players|join_game|cancel_or_full|update_game|irrelevant|uncertain",
  "confidence": 0.0,
  "normalized_text": "可选，把用户意思改写成规则解析器容易理解的一句话",
  "reply_text": "可选，信息不足时给用户的自然追问",
  "needs_human_review": false,
  "facts": {"reason": "简短原因"}
}

如果明确涉及抽水、赌资、结算输赢、代收代付、借码、上分下分，必须 needs_human_review=true。
如果不确定，intent 用 uncertain，confidence 不要超过 0.55。"""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @classmethod
    def from_env(cls) -> "OpenAICompatibleLLMResolver | None":
        config = LLMConfig.from_env()
        return cls(config) if config else None

    def resolve(self, message: Message, context: dict[str, Any] | None = None) -> LLMResolution:
        message_payload = self._message_payload(message, context)
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "message": message_payload,
                            "context": context or {},
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return LLMResolution(
                is_mahjong_related=False,
                intent="uncertain",
                confidence=0.0,
                needs_human_review=True,
                notes=[f"LLM 调用失败：{type(exc).__name__}: {exc}"],
            )

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return self._parse_resolution(content)

    def _message_payload(self, message: Message, context: dict[str, Any] | None) -> dict[str, Any]:
        current_message = (context or {}).get("current_message")
        if isinstance(current_message, dict):
            return {
                "text": current_message.get("text", ""),
                "sender_ref": current_message.get("sender_ref"),
                "sender_display_name": current_message.get("sender_display_name"),
                "channel_type": current_message.get("channel_type", message.channel_type.value),
                "modalities": current_message.get("modalities", []),
                "source": current_message.get("source", {}),
            }
        return {
            "text": message.text,
            "sender_ref": "unscoped_message_without_context",
            "sender_display_name": message.sender_name,
            "channel_type": message.channel_type.value,
            "metadata_keys": sorted(message.metadata.keys()),
        }

    def _parse_resolution(self, content: str) -> LLMResolution:
        try:
            raw = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.S)
            if not match:
                return LLMResolution(
                    is_mahjong_related=False,
                    intent="uncertain",
                    confidence=0.0,
                    needs_human_review=True,
                    notes=["LLM 未返回可解析 JSON。"],
                )
            try:
                raw = json.loads(match.group(0))
            except json.JSONDecodeError:
                return LLMResolution(
                    is_mahjong_related=False,
                    intent="uncertain",
                    confidence=0.0,
                    needs_human_review=True,
                    notes=["LLM 返回的 JSON 片段无法解析。"],
                )

        confidence = raw.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        return LLMResolution(
            is_mahjong_related=bool(raw.get("is_mahjong_related")),
            intent=str(raw.get("intent") or "uncertain"),
            confidence=max(0.0, min(confidence, 1.0)),
            normalized_text=_optional_str(raw.get("normalized_text")),
            reply_text=_optional_str(raw.get("reply_text")),
            needs_human_review=bool(raw.get("needs_human_review")),
            facts=raw.get("facts") if isinstance(raw.get("facts"), dict) else {},
            notes=["LLM 语义解析已执行。"],
        )


def _optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _provider_defaults(provider: str) -> dict[str, str]:
    if provider in {"qwen", "dashscope", "aliyun", "bailian"}:
        return {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-plus",
        }
    return {
        "base_url": "https://api.openai.com/v1",
    }
