from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Protocol

from .budget import LLMBudgetManager, usage_from_response
from .llm import LLMConfig


AuditLogger = Callable[[str, str, dict[str, Any]], None]


class UrlOpenResponse(Protocol):
    def __enter__(self) -> "UrlOpenResponse":
        ...

    def __exit__(self, exc_type, exc, tb) -> None:
        ...

    def read(self) -> bytes:
        ...


UrlOpen = Callable[..., UrlOpenResponse]


ORGANIZER_FOLLOWUP_SYSTEM_PROMPT = """你是麻将馆老板的协商消息起草助手。
当前候选人对原局提出了新条件，例如改时间、改时长、改烟况。
你的任务是判断是否需要向本局发起人确认，并起草一条给发起人的待审批消息。
只能生成草稿，不能直接发送，不能修改局状态，不能确认候选人入局。
消息要短、自然、像老板微信手写。
不要输出内部字段、推荐分、工具名。
reasoning_summary 只能写一句简短判断依据。
只输出 JSON：
{"should_create_message":true,"message_text":"发给发起人的一句话","risk_level":"low|medium|high","reasoning_summary":"一句话原因","notes":["可选简短说明"]}"""


@dataclass(slots=True)
class OrganizerFollowupDraftService:
    """Draft and guard organizer followup messages without side effects.

    This service may call an LLM to draft a pending message, but it never writes
    state, creates approvals, or sends messages. Tool validation and persistence
    remain owned by the followup adapter and backend.
    """

    llm_config: LLMConfig | None = None
    budget_manager: LLMBudgetManager | None = None
    audit_logger: AuditLogger | None = None
    urlopen: UrlOpen = urllib.request.urlopen

    def fallback_message(
        self,
        *,
        classification: dict[str, Any],
        candidate_name: str,
        organizer_name: str,
    ) -> str:
        prefix = f"{organizer_name}，"
        requested_start = classification.get("requested_start_time_label")
        if requested_start:
            return f"{prefix}{candidate_name}最快{requested_start}到，你们{requested_start}开可以吗？"
        requested_duration = classification.get("requested_duration_hours")
        if requested_duration:
            duration_text = _duration_text(requested_duration)
            return f"{prefix}{candidate_name}想打{duration_text}，你们这桌能不能打{duration_text}？"
        return f"{prefix}{candidate_name}这边条件有点不一样，我先跟你确认下能不能对上？"

    def guard_message(
        self,
        text: str,
        *,
        fallback: str,
        classification: dict[str, Any],
        organizer_name: str,
    ) -> str:
        cleaned = _truncate_text(text.strip(), 300)
        if not cleaned:
            return fallback
        if re.search(r"已经(发|通知|确认|改|安排)|已(发|通知|确认|改|安排)|直接来|我已经", cleaned):
            return fallback
        requested_start = classification.get("requested_start_time_label")
        requested_duration = classification.get("requested_duration_hours")
        if requested_start and str(requested_start) not in cleaned:
            return fallback
        if requested_duration:
            duration_text = _duration_text(requested_duration)
            if duration_text not in cleaned:
                return fallback
        if organizer_name and organizer_name not in cleaned:
            cleaned = f"{organizer_name}，{cleaned}"
        if not re.search(r"可以吗|行吗|能不能|能对上|可不可以|确认", cleaned):
            return fallback
        return cleaned

    def draft(
        self,
        *,
        trace_id: str,
        classification: dict[str, Any],
        candidate_text: str,
        suggested_candidate_reply: str,
        outbox_item: dict[str, Any],
        game: dict[str, Any],
        fallback: str,
        now: datetime,
    ) -> dict[str, Any]:
        if not self.llm_config or not self.budget_manager:
            return {
                "should_create_message": True,
                "text": fallback,
                "source": "rules",
                "model": None,
                "reasoning_summary": "未配置 LLM，使用规则兜底。",
            }
        max_tokens = min(self.llm_config.max_completion_tokens, 180)
        payload = self._payload(
            classification=classification,
            candidate_text=candidate_text,
            suggested_candidate_reply=suggested_candidate_reply,
            outbox_item=outbox_item,
            game=game,
            fallback=fallback,
            now=now,
            max_tokens=max_tokens,
        )
        budget_decision = self.budget_manager.reserve(
            key="boss_trial_organizer_followup",
            model=self.llm_config.model,
            prompt=payload,
            max_completion_tokens=max_tokens,
        )
        if not budget_decision.allowed:
            self._audit(
                trace_id,
                "llm_budget_denied",
                {
                    "stage": "organizer_followup_draft",
                    "provider": self.llm_config.provider,
                    "model": self.llm_config.model,
                    "budget": budget_decision.to_dict(),
                },
            )
            return {
                "should_create_message": True,
                "text": fallback,
                "source": "rules",
                "model": self.llm_config.model,
                "reasoning_summary": f"LLM 预算不足，使用规则兜底：{budget_decision.reason}",
                "budget": budget_decision.to_dict(),
            }

        self._audit(
            trace_id,
            "llm_request",
            {
                "stage": "organizer_followup_draft",
                "provider": self.llm_config.provider,
                "model": self.llm_config.model,
                "base_url": self.llm_config.base_url,
                "timeout_seconds": self.llm_config.timeout_seconds,
                "budget": budget_decision.to_dict(),
                "payload": payload,
            },
        )
        request = urllib.request.Request(
            f"{self.llm_config.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.llm_config.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        try:
            with self.urlopen(request, timeout=self.llm_config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            self._audit(
                trace_id,
                "llm_error",
                {
                    "stage": "organizer_followup_draft",
                    "provider": self.llm_config.provider,
                    "model": self.llm_config.model,
                    "error": f"{type(exc).__name__}: {exc}",
                    "budget": budget_decision.to_dict(),
                },
            )
            return {
                "should_create_message": True,
                "text": fallback,
                "source": "rules",
                "model": self.llm_config.model,
                "reasoning_summary": f"LLM 起草失败，使用规则兜底：{type(exc).__name__}",
                "budget": budget_decision.to_dict(),
            }

        actual_usage = usage_from_response(data, self.llm_config.model)
        self.budget_manager.commit(budget_decision.reservation_id, actual_usage)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        self._audit(
            trace_id,
            "llm_response",
            {
                "stage": "organizer_followup_draft",
                "provider": self.llm_config.provider,
                "model": self.llm_config.model,
                "raw_response": data,
                "content": content,
                "usage": actual_usage.to_dict() if actual_usage else None,
            },
        )
        parsed = _parse_json_object(content)
        result = {
            "should_create_message": bool(parsed.get("should_create_message", True)),
            "text": str(parsed.get("message_text") or parsed.get("reply_text") or "").strip() or fallback,
            "source": "llm",
            "model": self.llm_config.model,
            "risk_level": str(parsed.get("risk_level") or "low"),
            "reasoning_summary": str(parsed.get("reasoning_summary") or parsed.get("reason") or "").strip(),
            "notes": parsed.get("notes") if isinstance(parsed.get("notes"), list) else [],
            "budget": budget_decision.to_dict(),
        }
        self._audit(
            trace_id,
            "llm_parsed",
            {
                "stage": "organizer_followup_draft",
                "provider": self.llm_config.provider,
                "model": self.llm_config.model,
                "parsed": result,
            },
        )
        return result

    def _payload(
        self,
        *,
        classification: dict[str, Any],
        candidate_text: str,
        suggested_candidate_reply: str,
        outbox_item: dict[str, Any],
        game: dict[str, Any],
        fallback: str,
        now: datetime,
        max_tokens: int,
    ) -> dict[str, Any]:
        prompt = {
            "task": "候选人提出新条件后，决定是否向发起人创建待审批确认消息。",
            "now": now.strftime("%Y-%m-%d %H:%M:%S"),
            "organizer": {
                "customer_id": game.get("organizer_id"),
                "customer_name": game.get("organizer_name"),
            },
            "candidate": {
                "customer_id": outbox_item.get("customer_id"),
                "customer_name": outbox_item.get("customer_name"),
                "reply_text": candidate_text,
            },
            "original_invite": {
                "message_text": outbox_item.get("message_text"),
                "game_id": outbox_item.get("game_id"),
            },
            "backend_classification": classification,
            "game_state": {
                "summary": game.get("live_summary") or (game.get("parsed") or {}).get("summary"),
                "parsed": game.get("parsed") or {},
                "participants": game.get("participants") or [],
            },
            "candidate_reply_to_send": suggested_candidate_reply,
            "fallback_message": fallback,
            "rules": [
                "如果 candidate_negotiation 为真，通常需要给发起人创建确认消息。",
                "发给发起人的消息要问新条件是否可以，不要说已经确认。",
                "如果候选人改时间，要问发起人这个时间能不能开。",
                "如果候选人改时长，要问发起人这桌能不能接受这个时长。",
                "不要让发起人误以为候选人已经入局。",
            ],
        }
        payload: dict[str, Any] = {
            "model": self.llm_config.model if self.llm_config else "",
            "temperature": min(self.llm_config.temperature, 0.3) if self.llm_config else 0.1,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": ORGANIZER_FOLLOWUP_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        }
        if self.llm_config and self.llm_config.thinking_enabled is not None:
            payload["thinking"] = {"type": "enabled" if self.llm_config.thinking_enabled else "disabled"}
        if self.llm_config and self.llm_config.response_format:
            payload["response_format"] = {"type": self.llm_config.response_format}
        return payload

    def _audit(self, trace_id: str, event: str, payload: dict[str, Any]) -> None:
        if self.audit_logger:
            self.audit_logger(trace_id, event, payload)


def _duration_text(value: Any) -> str:
    number = float(value)
    if number.is_integer():
        return f"{int(number)}小时"
    return f"{number:g}小时"


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


def _parse_json_object(content: Any) -> dict[str, Any]:
    text = str(content or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
