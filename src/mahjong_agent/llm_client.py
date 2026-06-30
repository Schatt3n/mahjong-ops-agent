from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .budget import LLMBudgetManager, usage_from_response
from .llm import LLMConfig


class UrlOpenResponse(Protocol):
    def __enter__(self) -> "UrlOpenResponse":
        ...

    def __exit__(self, exc_type, exc, tb) -> None:
        ...

    def read(self) -> bytes:
        ...


UrlOpen = Callable[..., UrlOpenResponse]
AuditLogger = Callable[[str, str, dict[str, Any]], None]


@dataclass(slots=True)
class OpenAICompatibleSemanticLLMClient:
    """OpenAI-compatible client for the controlled SemanticResolver contract.

    It only returns the model content for SemanticResolver to parse. It does not
    interpret actions, call tools, mutate state, or generate outbound messages.
    """

    config: LLMConfig
    budget_manager: LLMBudgetManager | None = None
    audit_logger: AuditLogger | None = None
    urlopen: UrlOpen = urllib.request.urlopen

    @classmethod
    def from_env(
        cls,
        *,
        budget_manager: LLMBudgetManager | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> "OpenAICompatibleSemanticLLMClient | None":
        config = LLMConfig.from_env()
        if config is None:
            return None
        return cls(config=config, budget_manager=budget_manager, audit_logger=audit_logger)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        trace_id: str,
        timeout_seconds: float,
    ) -> str:
        budget_manager = self.budget_manager or LLMBudgetManager.from_env()
        payload = self._payload(messages)
        budget_key = "controlled_workflow"
        budget_decision = budget_manager.reserve(
            key=budget_key,
            model=self.config.model,
            prompt=payload,
            max_completion_tokens=self.config.max_completion_tokens,
        )
        self._audit(
            trace_id,
            "semantic_llm_budget",
            {
                "provider": self.config.provider,
                "model": self.config.model,
                "budget": budget_decision.to_dict(),
            },
        )
        if not budget_decision.allowed:
            raise RuntimeError(f"LLM budget denied: {budget_decision.reason}")

        self._audit(
            trace_id,
            "semantic_llm_request",
            {
                "provider": self.config.provider,
                "model": self.config.model,
                "base_url": self.config.base_url,
                "timeout_seconds": timeout_seconds,
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
        try:
            with self.urlopen(request, timeout=timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except TimeoutError:
            raise
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"LLM HTTP error: {_http_error_note(exc)}") from exc
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM request failed: {type(exc).__name__}: {exc}") from exc

        usage = usage_from_response(data, self.config.model)
        budget_manager.commit(budget_decision.reservation_id, usage)
        content = _content_from_response(data)
        self._audit(
            trace_id,
            "semantic_llm_response",
            {
                "provider": self.config.provider,
                "model": self.config.model,
                "content": content,
                "usage": usage.to_dict() if usage else None,
                "raw_response": data,
            },
        )
        return content

    def _payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_completion_tokens,
            "messages": messages,
        }
        if self.config.thinking_enabled is not None:
            payload["thinking"] = {"type": "enabled" if self.config.thinking_enabled else "disabled"}
        if self.config.response_format:
            payload["response_format"] = {"type": self.config.response_format}
        return payload

    def _audit(self, trace_id: str, event: str, payload: dict[str, Any]) -> None:
        if self.audit_logger:
            self.audit_logger(trace_id, event, payload)


def _content_from_response(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM response has no choices.")
    message = (choices[0] or {}).get("message")
    if not isinstance(message, dict):
        raise RuntimeError("LLM response choice has no message.")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("LLM response message content is not a string.")
    return content


def _http_error_note(exc: urllib.error.HTTPError) -> str:
    body = exc.read().decode("utf-8", errors="replace")
    if not body:
        return f"HTTPError {exc.code} {exc.reason}"
    try:
        raw = json.loads(body)
    except json.JSONDecodeError:
        return f"HTTPError {exc.code} {exc.reason} body={body[:300]}"
    error = raw.get("error") if isinstance(raw, dict) else None
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        return f"HTTPError {exc.code} {exc.reason} code={code}, message={message}"
    return f"HTTPError {exc.code} {exc.reason} body={body[:300]}"
