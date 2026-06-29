from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .budget import LLMBudgetDecision, LLMBudgetManager, LLMUsage, usage_from_response
from .models import Message


@dataclass(slots=True)
class LLMConfig:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    provider: str = "openai"
    timeout_seconds: float = 8.0
    temperature: float = 0.1
    max_completion_tokens: int = 512
    thinking_enabled: bool | None = None
    response_format: str | None = None
    parse_retry_enabled: bool = True
    parse_retry_max_tokens: int = 256

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
            max_completion_tokens=int(os.getenv("MAHJONG_LLM_MAX_COMPLETION_TOKENS", defaults.get("max_completion_tokens", "512"))),
            thinking_enabled=_env_bool("MAHJONG_LLM_THINKING_ENABLED", _optional_bool(defaults.get("thinking_enabled"))),
            response_format=os.getenv("MAHJONG_LLM_RESPONSE_FORMAT", defaults.get("response_format")),
            parse_retry_enabled=_env_bool("MAHJONG_LLM_PARSE_RETRY_ENABLED", True) is True,
            parse_retry_max_tokens=int(os.getenv("MAHJONG_LLM_PARSE_RETRY_MAX_TOKENS", "256")),
        )


@dataclass(slots=True)
class LLMResolution:
    is_mahjong_related: bool
    intent: str = "uncertain"
    proposed_action: str = "unknown"
    confidence: float = 0.0
    normalized_text: str | None = None
    reply_text: str | None = None
    needs_human_review: bool = False
    facts: dict[str, Any] = field(default_factory=dict)
    slots: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    usage: LLMUsage | None = None
    budget: dict[str, Any] = field(default_factory=dict)


class LLMResolver(Protocol):
    def resolve(self, message: Message, context: dict[str, Any] | None = None) -> LLMResolution:
        ...


class OpenAICompatibleLLMResolver:
    """Small OpenAI-compatible JSON resolver.

    It is deliberately narrow: the model can interpret a message and propose a
    normalized text, but it cannot mutate state or send messages directly.
    """

    system_prompt = """你是棋牌室运营 Agent 的语义解析器和回复建议助手。
你的任务是理解用户消息、结合上下文和客户画像补全低风险语义，并输出后端可继续处理的结构化 JSON。
你不能直接安排座位，不能承诺已发送邀约，不能处理资金结算，不能声称已经确认房间。

已知本店玩法：
- cq = 杭麻里的财敲，不是重庆麻将。
- 财敲是杭麻的一种细分玩法/变体，不是和杭麻并列的大类。
- 当前默认地区是杭州；用户只说“麻将/打牌/有局”且没有明确玩法时，优先按杭麻理解。若门店地区是四川，则优先按川麻理解。
- 杭州门店里，川麻/换三张/定缺/幺鸡等非默认玩法通常会被用户明说；用户没说时不要主动问“杭麻还是川麻”，按杭麻或客户画像中的杭麻细分理解。
- 客户画像优先级高于地区默认。比如客户画像显示长期只打红中，则用户只说“今晚组一桌/有人打吗”时，可以按红中理解。
- 若客户画像里有稳定高频偏好，可以在 normalized_text 里用“按画像推断/建议确认”的方式表达；但不要伪造用户没有说过的事实。
- 对金额档位、人数、烟况这类可能影响纠纷的字段，画像只能作为建议或默认问法；除非用户原话明确，否则不要把它当成 100% 确认。
- “帮我组一桌/组一桌”只表示客户想让老板帮忙组局，不代表三缺一、二缺二或一缺三；如果用户没说现在几个人，normalized_text 里不要新增 371/三缺一/缺几。
- 如果用户给出的开局时间早于当前时间，不能在 normalized_text 里直接改成“明天”；只能标记为需要确认，例如“用户说下午两点但当前时间已过，需确认是否明天或改时间”。
- 371 = 三缺一，272 = 二缺二，173 = 一缺三。
- 川麻216 = 川麻 2-16 档，底注 2，封顶 16。
- 川麻1-32 = 川麻 1-32 档，底注 1，封顶 32。
- 半块、半、五毛 = 0.5 档。
- 不抽、不抽烟、无烟 = 无烟局；不要把“不抽”单独理解为抽水。
- 人齐开/尽快开/时间可以商量/能早点开就早点开，表示开局时间策略是人齐后尽快开，不是缺少开局时间。
- 通宵表示时长策略是通宵，不是缺少时长。
- 可识别玩法包括杭麻/财敲、川麻、幺鸡、素鸡、幺鸡47、红中麻将、捉鸡麻将、湖南麻将、重庆麻将。
- context.text_normalization 里的 normalized_text 和 changes 是后端提供的低风险文本标准化证据，只能作为理解辅助，不能替代原文事实。
- 如果原文里出现“0。5/0，5/0 5/0、5”等表达，且结合客户画像或麻将语境明显是在说档位，按 0.5 理解；如果仍不确定，在 reply_text 里自然追问。
- context.workflow_followup_context 如果存在，说明当前消息可能是在回复上一轮老板建议。必须结合 previous_system_suggested_reply、previous_user_text、previous_tool_results 判断当前短消息的真实含义。
- 如果上一轮老板建议“要组一个吗/要不要帮你组一个”，当前用户回复“可以/好/行/要/帮我组”，应理解为确认新组局；若继承上一轮条件后仍缺关键信息，则 proposed_action=ask_clarification，否则 proposed_action=create_game。

只输出最小 JSON，不要解释，不要输出空字段。schema：
{"is_mahjong_related":true,"intent":"find_players|join_game|cancel_or_full|update_game|irrelevant|uncertain","proposed_action":"search_existing_games|create_game|ask_clarification|cancel_game|join_game|human_review|ignore|unknown","confidence":0.0,"normalized_text":"可选短句","reply_text":"可选短回复","needs_human_review":false,"reasoning_summary":"一句话","slots":{}}

slots 只放本轮有证据或由上下文高置信继承的字段，允许扁平值或 {"value":...,"confidence":...,"source":"explicit|profile|region_default|inferred"}。
常用 slot 键：query_mode、game_type、variant、level、level_options、start_time、start_time_mode、duration_hours、duration_mode、known_players、missing_count、smoke。

槽位规则：
- 用户错别字、标点错误、语音转写错误但语义明确时，可以在 slots 和 normalized_text 中修正，例如“0，5/0,5/0 5”按 0.5，“人气开”按“人齐开/快开局”。
- 只有原文明确或业务上下文高置信时，slot confidence 才能 >= 0.75。
- 人数/缺口必须特别谨慎：“帮我组一桌”不能推断三缺一；如果 known_players 或 missing_count 不是 explicit，不要在 normalized_text 中写成已确认事实。
- start_time 如果早于当前时间、上午下午不明确、或只是“下班/晚上”这类范围，needs_confirmation=true，不要当作确定时间。
- 如果用户表达“尽快开/人齐开/时间可商量/能早点就早点”，start_time_mode=people_ready，start_time.value=null，needs_confirmation=false；不要追问固定几点，除非当前房态或参与人明确要求固定时间。
- 如果用户表达“通宵”，duration_mode=overnight；不要追问“打几个小时”。
- 不确定时不要硬填 unknown 以外的值；可以给 reply_text 追问。

如果明确涉及抽水、赌资、结算输赢、代收代付、借码、上分下分，必须 needs_human_review=true。
如果不确定，intent 用 uncertain，confidence 不要超过 0.55。"""

    def __init__(
        self,
        config: LLMConfig,
        budget_manager: LLMBudgetManager | None = None,
        audit_logger: Callable[[str, str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.budget_manager = budget_manager or LLMBudgetManager.from_env()
        self.audit_logger = audit_logger

    @classmethod
    def from_env(cls) -> "OpenAICompatibleLLMResolver | None":
        config = LLMConfig.from_env()
        return cls(config) if config else None

    def resolve(self, message: Message, context: dict[str, Any] | None = None) -> LLMResolution:
        message_payload = self._message_payload(message, context)
        trace_id = self._trace_id(message, context)
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_completion_tokens,
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
        if self.config.thinking_enabled is not None:
            payload["thinking"] = {"type": "enabled" if self.config.thinking_enabled else "disabled"}
        if self.config.response_format:
            payload["response_format"] = {"type": self.config.response_format}
        budget_key = str(
            message.metadata.get("budget_key")
            or message.metadata.get("tenant_id")
            or "default"
        )
        budget_decision = self.budget_manager.reserve(
            key=budget_key,
            model=self.config.model,
            prompt=payload,
            max_completion_tokens=self.config.max_completion_tokens,
        )
        if not budget_decision.allowed:
            self._audit(
                trace_id,
                "llm_budget_denied",
                {
                    "stage": "semantic_resolution",
                    "provider": self.config.provider,
                    "model": self.config.model,
                    "budget": budget_decision.to_dict(),
                },
            )
            return LLMResolution(
                is_mahjong_related=False,
                intent="uncertain",
                confidence=0.0,
                needs_human_review=True,
                notes=[f"LLM 预算不足，已停止模型调用：{budget_decision.reason}"],
                budget=budget_decision.to_dict(),
            )

        self._audit(
            trace_id,
            "llm_request",
            {
                "stage": "semantic_resolution",
                "provider": self.config.provider,
                "model": self.config.model,
                "base_url": self.config.base_url,
                "timeout_seconds": self.config.timeout_seconds,
                "budget": budget_decision.to_dict(),
                "payload": payload,
            },
        )

        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )

        usage: LLMUsage | None = None
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
                usage = usage_from_response(data, self.config.model)
                self.budget_manager.commit(budget_decision.reservation_id, usage)
        except urllib.error.HTTPError as exc:
            self._audit(
                trace_id,
                "llm_error",
                {
                    "stage": "semantic_resolution",
                    "provider": self.config.provider,
                    "model": self.config.model,
                    "error": _http_error_note(exc),
                    "budget": budget_decision.to_dict(),
                },
            )
            return LLMResolution(
                is_mahjong_related=False,
                intent="uncertain",
                confidence=0.0,
                needs_human_review=True,
                notes=[f"LLM 调用失败：{_http_error_note(exc)}"],
                budget=budget_decision.to_dict(),
            )
        except TimeoutError as exc:
            self._audit(
                trace_id,
                "llm_timeout",
                {
                    "stage": "semantic_resolution",
                    "provider": self.config.provider,
                    "model": self.config.model,
                    "timeout_seconds": self.config.timeout_seconds,
                    "error": f"{type(exc).__name__}: {exc}",
                    "budget": budget_decision.to_dict(),
                },
            )
            return LLMResolution(
                is_mahjong_related=False,
                intent="uncertain",
                confidence=0.0,
                needs_human_review=True,
                notes=[f"LLM 调用超过 {self.config.timeout_seconds} 秒，已中断并转人工。"],
                budget=budget_decision.to_dict(),
            )
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            self._audit(
                trace_id,
                "llm_error",
                {
                    "stage": "semantic_resolution",
                    "provider": self.config.provider,
                    "model": self.config.model,
                    "error": f"{type(exc).__name__}: {exc}",
                    "budget": budget_decision.to_dict(),
                },
            )
            return LLMResolution(
                is_mahjong_related=False,
                intent="uncertain",
                confidence=0.0,
                needs_human_review=True,
                notes=[f"LLM 调用失败：{type(exc).__name__}: {exc}"],
                budget=budget_decision.to_dict(),
            )

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        self._audit(
            trace_id,
            "llm_response",
            {
                "stage": "semantic_resolution",
                "provider": self.config.provider,
                "model": self.config.model,
                "raw_response": data,
                "content": content,
                "usage": usage.to_dict() if usage else None,
            },
        )
        resolution = self._parse_resolution(content, usage=usage, budget=budget_decision)
        if self._should_retry_parse(data, resolution):
            retry_resolution = self._retry_parse_resolution(
                trace_id=trace_id,
                budget_key=budget_key,
                original_content=content,
                original_finish_reason=self._finish_reason(data),
                message_payload=message_payload,
                context=context or {},
            )
            if retry_resolution is not None:
                if retry_resolution.proposed_action == "unknown" and resolution.proposed_action != "unknown":
                    resolution.notes.append("LLM 解析重试未给出有效动作，保留截断输出中可安全恢复的顶层动作。")
                else:
                    resolution = retry_resolution
        self._audit(
            trace_id,
            "llm_parsed",
            {
                "stage": "semantic_resolution",
                "provider": self.config.provider,
                "model": self.config.model,
                "resolution": {
                    "is_mahjong_related": resolution.is_mahjong_related,
                    "intent": resolution.intent,
                    "proposed_action": resolution.proposed_action,
                    "confidence": resolution.confidence,
                    "normalized_text": resolution.normalized_text,
                    "reply_text": resolution.reply_text,
                    "needs_human_review": resolution.needs_human_review,
                    "slots": resolution.slots,
                    "facts": resolution.facts,
                    "notes": resolution.notes,
                    "budget": resolution.budget,
                },
            },
        )
        if usage:
            resolution.notes.append(
                f"LLM usage prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}"
            )
        resolution.notes.append(f"LLM budget={budget_decision.reason}")
        return resolution

    def _should_retry_parse(self, data: dict[str, Any], resolution: LLMResolution) -> bool:
        if not self.config.parse_retry_enabled:
            return False
        markers = ("LLM 未返回可解析 JSON", "LLM 返回的 JSON 片段无法解析", "LLM JSON 被截断")
        if any(any(marker in note for marker in markers) for note in resolution.notes):
            return True
        return self._finish_reason(data) == "length"

    def _finish_reason(self, data: dict[str, Any]) -> str:
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            return ""
        return str((choices[0] or {}).get("finish_reason") or "")

    def _retry_parse_resolution(
        self,
        *,
        trace_id: str,
        budget_key: str,
        original_content: str,
        original_finish_reason: str,
        message_payload: dict[str, Any],
        context: dict[str, Any],
    ) -> LLMResolution | None:
        max_tokens = max(64, min(self.config.parse_retry_max_tokens, self.config.max_completion_tokens))
        compact_context = {
            "current_message": context.get("current_message"),
            "text_normalization": context.get("text_normalization"),
            "workflow_followup_context": context.get("workflow_followup_context"),
            "customer_profile_summary": context.get("customer_profile_summary"),
            "conversation_compressed_history": (
                (context.get("conversation_summary") or {}).get("compressed_history")
                if isinstance(context.get("conversation_summary"), dict)
                else {}
            ),
            "recent_open_games": (context.get("game_state_snapshot") or {}).get("recent_open_games", [])[:3]
            if isinstance(context.get("game_state_snapshot"), dict)
            else [],
        }
        retry_payload = {
            "model": self.config.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是语义 JSON 修复器。只返回最小 JSON，不要解释。"
                        "schema: {\"is_mahjong_related\":bool,\"intent\":str,\"proposed_action\":str,"
                        "\"confidence\":0-1,\"normalized_text\":str|null,\"reply_text\":str|null,"
                        "\"needs_human_review\":bool,\"reasoning_summary\":str,\"slots\":{}}。"
                        "如果原输出已包含 proposed_action/confidence，优先保留；不要输出长 slots。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "reason": "previous_response_parse_failed",
                            "finish_reason": original_finish_reason,
                            "message": message_payload,
                            "compact_context": compact_context,
                            "malformed_model_output": original_content[:4000],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        if self.config.thinking_enabled is not None:
            retry_payload["thinking"] = {"type": "enabled" if self.config.thinking_enabled else "disabled"}
        if self.config.response_format:
            retry_payload["response_format"] = {"type": self.config.response_format}

        budget_decision = self.budget_manager.reserve(
            key=f"{budget_key}:parse_retry",
            model=self.config.model,
            prompt=retry_payload,
            max_completion_tokens=max_tokens,
        )
        if not budget_decision.allowed:
            self._audit(
                trace_id,
                "llm_retry_budget_denied",
                {
                    "stage": "semantic_resolution",
                    "provider": self.config.provider,
                    "model": self.config.model,
                    "budget": budget_decision.to_dict(),
                },
            )
            return None

        self._audit(
            trace_id,
            "llm_retry_request",
            {
                "stage": "semantic_resolution",
                "provider": self.config.provider,
                "model": self.config.model,
                "base_url": self.config.base_url,
                "timeout_seconds": self.config.timeout_seconds,
                "budget": budget_decision.to_dict(),
                "payload": retry_payload,
            },
        )
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(retry_payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
                usage = usage_from_response(data, self.config.model)
                self.budget_manager.commit(budget_decision.reservation_id, usage)
        except TimeoutError as exc:
            self._audit(
                trace_id,
                "llm_retry_timeout",
                {
                    "stage": "semantic_resolution",
                    "provider": self.config.provider,
                    "model": self.config.model,
                    "timeout_seconds": self.config.timeout_seconds,
                    "error": f"{type(exc).__name__}: {exc}",
                    "budget": budget_decision.to_dict(),
                },
            )
            return None
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
            self._audit(
                trace_id,
                "llm_retry_error",
                {
                    "stage": "semantic_resolution",
                    "provider": self.config.provider,
                    "model": self.config.model,
                    "error": f"{type(exc).__name__}: {exc}",
                    "budget": budget_decision.to_dict(),
                },
            )
            return None

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        self._audit(
            trace_id,
            "llm_retry_response",
            {
                "stage": "semantic_resolution",
                "provider": self.config.provider,
                "model": self.config.model,
                "raw_response": data,
                "content": content,
                "usage": usage.to_dict() if usage else None,
            },
        )
        resolution = self._parse_resolution(content, usage=usage, budget=budget_decision)
        if any("无法解析" in note or "未返回可解析" in note for note in resolution.notes):
            self._audit(
                trace_id,
                "llm_retry_parse_failed",
                {
                    "stage": "semantic_resolution",
                    "provider": self.config.provider,
                    "model": self.config.model,
                    "notes": resolution.notes,
                },
            )
            return None
        resolution.notes.append("LLM 解析失败后已使用紧凑 schema 重试成功。")
        self._audit(
            trace_id,
            "llm_retry_parsed",
            {
                "stage": "semantic_resolution",
                "provider": self.config.provider,
                "model": self.config.model,
                "resolution": {
                    "intent": resolution.intent,
                    "proposed_action": resolution.proposed_action,
                    "confidence": resolution.confidence,
                    "notes": resolution.notes,
                    "budget": resolution.budget,
                },
            },
        )
        return resolution

    def _message_payload(self, message: Message, context: dict[str, Any] | None) -> dict[str, Any]:
        current_message = (context or {}).get("current_message")
        if isinstance(current_message, dict):
            return {
                "text": current_message.get("text", ""),
                "text_normalization": (context or {}).get("text_normalization", {}),
                "workflow_followup_context": (context or {}).get("workflow_followup_context", {}),
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

    def _trace_id(self, message: Message, context: dict[str, Any] | None) -> str:
        runtime = (context or {}).get("runtime")
        if isinstance(runtime, dict) and runtime.get("trace_id"):
            return str(runtime["trace_id"])
        return str(message.metadata.get("trace_id") or "trace_missing")

    def _audit(self, trace_id: str, event: str, payload: dict[str, Any]) -> None:
        if self.audit_logger is None:
            return
        self.audit_logger(trace_id, event, payload)

    def _parse_resolution(
        self,
        content: str,
        usage: LLMUsage | None = None,
        budget: LLMBudgetDecision | None = None,
    ) -> LLMResolution:
        budget_dict = budget.to_dict() if budget else {}
        try:
            raw = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.S)
            if not match:
                salvaged = _salvage_truncated_resolution(content, usage=usage, budget=budget)
                if salvaged:
                    return salvaged
                return LLMResolution(
                    is_mahjong_related=False,
                    intent="uncertain",
                    confidence=0.0,
                    needs_human_review=True,
                    notes=["LLM 未返回可解析 JSON。"],
                    usage=usage,
                    budget=budget_dict,
                )
            try:
                raw = json.loads(match.group(0))
            except json.JSONDecodeError:
                salvaged = _salvage_truncated_resolution(content, usage=usage, budget=budget)
                if salvaged:
                    return salvaged
                return LLMResolution(
                    is_mahjong_related=False,
                    intent="uncertain",
                    confidence=0.0,
                    needs_human_review=True,
                    notes=["LLM 返回的 JSON 片段无法解析。"],
                    usage=usage,
                    budget=budget_dict,
                )

        confidence = raw.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        facts = raw.get("facts") if isinstance(raw.get("facts"), dict) else {}
        slots = raw.get("slots") if isinstance(raw.get("slots"), dict) else {}
        if not slots and isinstance(facts.get("slots"), dict):
            slots = facts["slots"]
        reasoning_summary = _optional_str(raw.get("reasoning_summary")) or _optional_str(raw.get("reason"))
        if reasoning_summary:
            facts = {**facts, "reasoning_summary": reasoning_summary}

        return LLMResolution(
            is_mahjong_related=bool(raw.get("is_mahjong_related")),
            intent=str(raw.get("intent") or "uncertain"),
            proposed_action=_normalize_proposed_action(raw.get("proposed_action"), raw.get("intent")),
            confidence=max(0.0, min(confidence, 1.0)),
            normalized_text=_optional_str(raw.get("normalized_text")),
            reply_text=_optional_str(raw.get("reply_text")),
            needs_human_review=bool(raw.get("needs_human_review")),
            facts=facts,
            slots=slots,
            notes=["LLM 语义解析已执行。"],
            usage=usage,
            budget=budget_dict,
        )


def _salvage_truncated_resolution(
    content: str,
    *,
    usage: LLMUsage | None,
    budget: LLMBudgetDecision | None,
) -> LLMResolution | None:
    """Recover only top-level action fields from a truncated JSON object.

    The recovered action still goes through backend confidence, state-machine,
    permission, and critical-slot validation. We intentionally do not recover
    nested slots from malformed JSON.
    """

    intent = _extract_json_string_field(content, "intent") or "uncertain"
    proposed_action = _extract_json_string_field(content, "proposed_action")
    if not proposed_action and intent == "uncertain":
        return None
    confidence = _extract_json_float_field(content, "confidence")
    if confidence is None:
        confidence = 0.0
    is_related = _extract_json_bool_field(content, "is_mahjong_related")
    if is_related is None:
        is_related = _normalize_proposed_action(proposed_action, intent) != "unknown"
    needs_human_review = bool(_extract_json_bool_field(content, "needs_human_review") or False)
    reasoning_summary = _extract_json_string_field(content, "reasoning_summary")
    facts = {"reasoning_summary": reasoning_summary} if reasoning_summary else {}
    budget_dict = budget.to_dict() if budget else {}
    return LLMResolution(
        is_mahjong_related=bool(is_related),
        intent=intent,
        proposed_action=_normalize_proposed_action(proposed_action, intent),
        confidence=max(0.0, min(float(confidence), 1.0)),
        normalized_text=_extract_json_string_field(content, "normalized_text"),
        reply_text=_extract_json_string_field(content, "reply_text"),
        needs_human_review=needs_human_review,
        facts=facts,
        slots={},
        notes=["LLM JSON 被截断，已仅采纳可解析的顶层动作字段。"],
        usage=usage,
        budget=budget_dict,
    )


def _extract_json_string_field(content: str, field_name: str) -> str | None:
    pattern = rf'"{re.escape(field_name)}"\s*:\s*"((?:\\.|[^"\\])*)"'
    match = re.search(pattern, content, flags=re.S)
    if not match:
        return None
    try:
        value = json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return None
    return _optional_str(value)


def _extract_json_float_field(content: str, field_name: str) -> float | None:
    pattern = rf'"{re.escape(field_name)}"\s*:\s*(-?\d+(?:\.\d+)?)'
    match = re.search(pattern, content)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_json_bool_field(content: str, field_name: str) -> bool | None:
    pattern = rf'"{re.escape(field_name)}"\s*:\s*(true|false)'
    match = re.search(pattern, content, flags=re.I)
    if not match:
        return None
    return match.group(1).lower() == "true"


def _optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _normalize_proposed_action(value: Any, intent: Any = None) -> str:
    action = str(value or "").strip().lower()
    aliases = {
        "search_existing": "search_existing_games",
        "search_current_open_games": "search_existing_games",
        "find_existing_game": "search_existing_games",
        "create_new_game": "create_game",
        "find_players": "create_game",
        "queue_invites": "create_game",
        "clarify": "ask_clarification",
        "ask_followup": "ask_clarification",
        "human": "human_review",
        "manual_review": "human_review",
        "silent": "ignore",
        "no_reply": "ignore",
    }
    action = aliases.get(action, action)
    allowed = {
        "search_existing_games",
        "create_game",
        "ask_clarification",
        "cancel_game",
        "join_game",
        "human_review",
        "ignore",
        "unknown",
    }
    if action in allowed:
        return action

    intent_value = str(intent or "").strip().lower()
    intent_mapping = {
        "find_players": "create_game",
        "join_game": "join_game",
        "cancel_or_full": "cancel_game",
        "update_game": "create_game",
        "irrelevant": "ignore",
        "uncertain": "unknown",
    }
    return intent_mapping.get(intent_value, "unknown")


def _provider_defaults(provider: str) -> dict[str, str]:
    if provider in {"qwen", "dashscope", "aliyun", "bailian"}:
        return {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-plus",
        }
    if provider in {"zai", "z.ai", "zhipu", "bigmodel", "glm"}:
        return {
            "base_url": "https://api.z.ai/api/paas/v4",
            "model": "glm-4.7-flash",
            "max_completion_tokens": "1024",
        }
    if provider in {"deepseek", "deepseek-ai"}:
        return {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "max_completion_tokens": "1024",
            "thinking_enabled": "false",
            "response_format": "json_object",
        }
    return {
        "base_url": "https://api.openai.com/v1",
    }


def _http_error_note(exc: urllib.error.HTTPError) -> str:
    body = exc.read().decode("utf-8", errors="replace")
    message = ""
    try:
        raw = json.loads(body)
        error = raw.get("error") if isinstance(raw, dict) else None
        if isinstance(error, dict):
            code = error.get("code")
            text = error.get("message")
            message = f" code={code}, message={text}" if code or text else ""
    except json.JSONDecodeError:
        message = f" body={body[:300]}" if body else ""
    return f"HTTPError {exc.code} {exc.reason}{message}"


def _optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _env_bool(name: str, default: bool | None) -> bool | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    if raw.strip().lower() in {"none", "unset", "default"}:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on", "enabled"}
