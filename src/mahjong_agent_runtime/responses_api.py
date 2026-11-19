"""解析 Responses 完成结果，将最终正文和可审计的元数据分开。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any


_RESPONSE_STATUSES = ("completed", "failed", "in_progress", "cancelled", "queued", "incomplete")
_INCOMPLETE_REASONS = ("max_output_tokens", "content_filter")


@dataclass(frozen=True, slots=True)
class ResponsesOutput:
    """单次 Responses 解析结果及脱敏诊断，空正文和无正文分别表示。"""

    text: str | None
    error: str | None
    diagnostics: dict[str, Any]


def _text_sha256(text: str) -> str:
    """以与主 Agent 来源证明相同的编码计算正文哈希，不替换异常 Unicode。"""
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _responses_usage(value: Any) -> dict[str, Any]:
    """把供应商允许列表内的计数映射到 Runtime 通用用量字段。"""
    if not isinstance(value, dict):
        return {}
    usage = {}
    for original, normalized in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"),
                                 ("total_tokens", "total_tokens")):
        count = value.get(original)
        if type(count) is int and 0 <= count <= 10**12:
            usage[normalized] = count
    for original, normalized, field in (("input_tokens_details", "prompt_tokens_details", "cached_tokens"),
                                        ("output_tokens_details", "completion_tokens_details", "reasoning_tokens")):
        details = value.get(original)
        count = details.get(field) if isinstance(details, dict) else None
        if type(count) is int and 0 <= count <= 10**12:
            usage[normalized] = {field: count}
    return usage


def _final_text_blocks(output: list[Any]) -> tuple[list[str], int, str | None]:
    """提取已结束的助手文本块，避免把推理、工具或中间说明送入动作解析器。"""
    text_blocks: list[str] = []
    refusal_count = 0
    malformed = "LLM Responses output has an invalid message or content block"
    for item in output:
        if not isinstance(item, dict) or not isinstance(item.get("type"), str):
            return [], refusal_count, malformed
        if item["type"] != "message" or item.get("phase") == "commentary":
            continue
        if item.get("role") != "assistant" or item.get("phase") not in (None, "final_answer"):
            return [], refusal_count, malformed
        if item.get("status") != "completed":
            return [], refusal_count, "LLM Responses output message is not completed"
        content = item.get("content")
        if not isinstance(content, list):
            return [], refusal_count, malformed
        for block in content:
            if isinstance(block, dict) and block.get("type") == "refusal":
                refusal_count += 1
            elif isinstance(block, dict) and block.get("type") == "output_text" and isinstance(block.get("text"), str):
                text_blocks.append(block["text"])
            else:
                return [], refusal_count, malformed
    if refusal_count:
        return [], refusal_count, "LLM Responses output was refused"
    if not text_blocks:
        return [], refusal_count, "LLM Responses output has no text content"
    return text_blocks, refusal_count, None


def parse_responses_output(raw: Any) -> ResponsesOutput:
    """解析非流式 Responses；状态或正文不合格时保留诊断并阻止业务动作。"""
    payload = raw if isinstance(raw, dict) else {}
    status = payload.get("status")
    details = payload.get("incomplete_details")
    reason = details.get("reason") if isinstance(details, dict) else None
    output = payload.get("output")
    if payload.get("object") not in (None, "response") or not isinstance(output, list):
        blocks, refusals, error = [], 0, "LLM Responses payload has no valid output array"
    elif status != "completed" or details is not None or payload.get("error") is not None:
        blocks, refusals, error = [], 0, "LLM Responses response is not completed"
    else:
        blocks, refusals, error = _final_text_blocks(output)
    text = "".join(blocks) if error is None else None
    return ResponsesOutput(text, error, {
        "api": "responses",
        "response_status": status if status in _RESPONSE_STATUSES else None,
        "incomplete_reason": reason if reason in _INCOMPLETE_REASONS else None,
        "output_count": len(output) if isinstance(output, list) else None,
        "refusal_count": refusals,
        "choice_count": None,
        "finish_reason": None,
        "reasoning_fields": {},
        "usage": _responses_usage(payload.get("usage")),
        "content": {
            "present": text is not None,
            "kind": "output_text_blocks",
            "text_block_count": len(blocks),
            "length": len(text) if text is not None else None,
            "empty": text == "" if text is not None else None,
            "whitespace_only": bool(text) and not text.strip() if text is not None else None,
            "sha256": _text_sha256(text) if text is not None else None,
        },
    })


def responses_content_matches(attempt: dict[str, Any], text: str) -> bool:
    """校验原始文本块的提取证据，防止无正文或其他协议借用完整响应身份。"""
    if (attempt.get("api") != "responses" or attempt.get("response_status") != "completed"
            or attempt.get("incomplete_reason") is not None or attempt.get("choice_count") is not None
            or type(attempt.get("output_count")) is not int or attempt["output_count"] < 1
            or type(attempt.get("refusal_count")) is not int or attempt["refusal_count"] != 0):
        return False
    content = attempt.get("content")
    if (not isinstance(content, dict) or type(content.get("text_block_count")) is not int
            or content["text_block_count"] < 1):
        return False
    expected = {
        "present": True,
        "kind": "output_text_blocks",
        "length": len(text),
        "empty": text == "",
        "whitespace_only": bool(text) and not text.strip(),
        "sha256": _text_sha256(text),
    }
    return all(type(content.get(key)) is type(value) and content[key] == value for key, value in expected.items())
