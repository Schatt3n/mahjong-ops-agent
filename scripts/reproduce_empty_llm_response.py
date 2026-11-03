#!/usr/bin/env python3
"""有限重放原失败提示词，只诊断模型空响应，不执行业务动作或覆盖原始证据。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mahjong_agent_runtime.env import load_dotenv_defaults  # noqa: E402
from mahjong_agent_runtime.llm import OpenAICompatibleAgentClient  # noqa: E402


STAGES = {
    "main": ("llm_prompt", "llm_response"),
    "review": ("customer_visible_content_review_prompt", "customer_visible_content_review_response"),
}
SAFE_CONFIG_FIELDS = (
    "provider", "model", "temperature", "max_tokens", "response_format", "thinking_mode", "reasoning_effort",
    "retry_attempts", "attempt_timeout_seconds", "max_concurrency",
)


def message_digest(messages: list[dict[str, Any]]) -> str:
    """摘要覆盖完整消息结构及内容字符串，不改写任何提示词或上下文。"""
    encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_source_prompt(trace_path: Path, *, stage: str, event_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按显式零基序号读取原空响应的请求，仅返回原消息和不含正文的来源摘要。"""
    if stage not in STAGES or type(event_index) is not int or event_index < 0:
        raise ValueError("stage must be main/review and event-index must be a nonnegative integer")
    raw = trace_path.read_bytes()
    payload = json.loads(raw)
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list) or event_index >= len(events):
        raise ValueError("source trace does not contain the selected event index")
    prompt_step, response_step = STAGES[stage]
    event = events[event_index]
    content = event.get("content") if isinstance(event, dict) else None
    if not isinstance(content, dict) or event.get("step") != prompt_step:
        raise ValueError("selected event is not the prompt for the requested stage")
    messages = content.get("messages")
    if (not isinstance(messages, list) or not messages or any(
            not isinstance(message, dict) or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str) for message in messages)):
        raise ValueError("source prompt requires the original nonempty role/content message array")
    response_index = None
    for index in range(event_index + 1, len(events)):
        following = events[index]
        if not isinstance(following, dict):
            raise ValueError("source trace contains a malformed event")
        if following.get("step") == prompt_step:
            break
        if following.get("step") == response_step:
            response_index = index
            break
    if response_index is None:
        raise ValueError("selected prompt has no response before the next prompt of the same stage")
    response = events[response_index].get("content")
    original = response.get("content") if isinstance(response, dict) else None
    if not isinstance(original, str) or original.strip():
        raise ValueError("selected source response must be an actual empty or whitespace-only string")
    return deepcopy(messages), {
        "source_trace_sha256": hashlib.sha256(raw).hexdigest(), "stage": stage,
        "source_prompt_event_index": event_index, "source_response_event_index": response_index,
        "source_response_classification": "empty" if original == "" else "whitespace",
        "messages_sha256": message_digest(messages), "message_count": len(messages),
    }


def write_private_json(path: Path, payload: dict[str, Any]) -> None:
    """仅新建当前运行的本地证据文件，不覆盖旧结果，限制文件权限避免无关读取。"""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")


def summarize_transport_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """分别汇总每次HTTP尝试的终止原因和真实用量，不混淆repeat与传输重试。"""
    finishes: dict[str, int] = {}
    usage: dict[str, int] = {}
    http_attempt_count = 0
    pending = [record.get("diagnostics", {}) for record in records]
    while pending:
        diagnostic = pending.pop()
        if not isinstance(diagnostic, dict):
            continue
        pending.extend(diagnostic[key] for key in ("primary", "fallback") if isinstance(diagnostic.get(key), dict))
        for attempt in diagnostic.get("attempts", []):
            if not isinstance(attempt, dict):
                continue
            http_attempt_count += 1
            reason = attempt.get("finish_reason")
            reason = reason if isinstance(reason, str) else "(not_reported)"
            finishes[reason] = finishes.get(reason, 0) + 1
            for field, value in attempt.get("usage", {}).items():
                if type(value) is int:
                    usage[field] = usage.get(field, 0) + value
                elif isinstance(value, dict):
                    for nested, amount in value.items():
                        if type(amount) is int:
                            name = f"{field}.{nested}"
                            usage[name] = usage.get(name, 0) + amount
    return {"http_attempt_count": http_attempt_count, "finish_reason_counts": finishes, "usage_totals": usage}


def reproduce_prompt(
    client: Any, messages: list[dict[str, Any]], source: dict[str, Any], *, output_root: Path,
    repeat: int = 6, workers: int = 3, timeout_seconds: float = 90,
) -> tuple[Path, dict[str, Any]]:
    """通过生产client有限重放原请求；所有返回只落本地证据，不进入Agent或工具网关。

    每次调用保留独立原content及安全诊断。空响应不会自行再次请求；传输层有限重试
    沿用当前client配置，故每个repeat与其内部HTTP尝试必须分别解释。
    """
    from mahjong_agent_runtime.llm import completion_diagnostics

    if type(repeat) is not int or not 1 <= repeat <= 12 or type(workers) is not int or not 1 <= workers <= 3:
        raise ValueError("repeat must be 1..12 and workers must be 1..3")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be finite and positive")
    if message_digest(messages) != source.get("messages_sha256"):
        raise ValueError("messages no longer match the selected original source prompt")
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(tempfile.mkdtemp(prefix=f"{stamp}_", dir=output_root))
    config = getattr(client, "config", None)
    safe_config = {field: getattr(config, field, None) for field in SAFE_CONFIG_FIELDS}
    # fallback也可能返回正文，扣留范围须覆盖所有已配置提供方；循环保护只防止
    # 注入client错误地自引用，不改变生产client的回退或调用顺序。
    configured_keys: set[str] = set()
    visited_clients: set[int] = set()
    current = client
    while current is not None and id(current) not in visited_clients:
        visited_clients.add(id(current))
        key = getattr(getattr(current, "config", None), "api_key", None)
        if isinstance(key, str) and key:
            configured_keys.add(key)
        current = getattr(current, "fallback_client", None)

    def attempt(index: int) -> dict[str, Any]:
        """一次调用保存一个独立证据文件，异常只保存类型及安全诊断，禁止记录认证文本。"""
        trace_id = f"empty-response-reproduction:{run_dir.name}:{index:02d}"
        started = time.perf_counter()
        record: dict[str, Any] = {"attempt": index, "trace_id": trace_id, "timeout_seconds": timeout_seconds,
                                  "messages_sha256": source["messages_sha256"]}
        try:
            result = client.complete(deepcopy(messages), trace_id=trace_id, timeout_seconds=timeout_seconds)
            record["diagnostics"] = completion_diagnostics(result)
            if not isinstance(result, str):
                raise TypeError("production client must return text")
            raw_content = str(result)
            record.update(classification="empty" if raw_content == "" else "whitespace" if not raw_content.strip() else "non_empty",
                          content=raw_content, content_sha256=hashlib.sha256(raw_content.encode("utf-8")).hexdigest())
            # 内容通常不含认证；若提供方异常回显配置密钥，保留摘要并扣留正文，不能把认证写入证据。
            if any(key in raw_content for key in configured_keys):
                record.update(content=None, content_withheld="configured_credential_detected")
        except Exception as exc:
            record.update(classification="error", error_type=type(exc).__name__, diagnostics=completion_diagnostics(exc))
        record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        name = f"attempt_{index:02d}.json"
        write_private_json(run_dir / name, record)
        return {"attempt": index, "record_file": name, "classification": record["classification"],
                "elapsed_ms": record["elapsed_ms"], "diagnostics": record["diagnostics"]}

    # 先写来源和配置；即使进程被中断，已完成attempt仍有完整证据，原始失败始终保持独立。
    write_private_json(run_dir / "source.json", {**source, "current_model_config": safe_config,
        "configuration_source": "current_environment_without_replay_overrides", "repeat": repeat, "workers": workers,
        "timeout_seconds": timeout_seconds, "historical_configuration_verified": False,
        "scope": "model_calls_only_no_business_tools_or_delivery"})
    with ThreadPoolExecutor(max_workers=workers) as executor:
        attempts = list(executor.map(attempt, range(1, repeat + 1)))
    summary = {"run_id": run_dir.name, "status": "completed", "source": source, "current_model_config": safe_config,
               "repeat": repeat, "workers": workers, "timeout_seconds": timeout_seconds,
               "counts": {kind: sum(item["classification"] == kind for item in attempts)
                          for kind in ("empty", "whitespace", "non_empty", "error")},
               "attempts": attempts,
               "transport_summary": summarize_transport_diagnostics(attempts),
               "notice": "原失败证据保持不变；本次只诊断模型响应，不证明业务恢复或替代完整验收。"}
    write_private_json(run_dir / "summary.json", summary)
    return run_dir, summary


def main(argv: list[str] | None = None) -> int:
    """显式指定失败提示词和有限调用数；加载当前配置后仅启动独立模型诊断。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", type=Path, required=True, help="original local raw trace JSON; never modified")
    parser.add_argument("--stage", choices=tuple(STAGES), required=True, help="main planning or customer-visible review")
    parser.add_argument("--event-index", type=int, required=True, help="zero-based original prompt event index")
    parser.add_argument("--repeat", type=int, default=6, help="finite production calls, 1..12 (default 6)")
    parser.add_argument("--workers", type=int, default=3, help="concurrent calls, 1..3 (default 3)")
    parser.add_argument("--timeout-seconds", type=float, help="per-call total budget; defaults to current MAHJONG_AGENT_LLM_TIMEOUT_SECONDS")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runtime_data" / "empty_llm_reproductions")
    parser.add_argument("--dotenv-path", type=Path, default=ROOT / ".env")
    parser.add_argument("--no-dotenv", action="store_true")
    args = parser.parse_args(argv)
    if not args.no_dotenv:
        load_dotenv_defaults(args.dotenv_path)
    try:
        messages, source = load_source_prompt(args.trace_path, stage=args.stage, event_index=args.event_index)
        client = OpenAICompatibleAgentClient.from_env()
        if client is None:
            raise ValueError("current model and API key must be configured")
        timeout = args.timeout_seconds if args.timeout_seconds is not None else float(os.getenv("MAHJONG_AGENT_LLM_TIMEOUT_SECONDS", "45"))
        run_dir, summary = reproduce_prompt(client, messages, source, output_root=args.output_root,
            repeat=args.repeat, workers=args.workers, timeout_seconds=timeout)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        # 不输出异常原文，避免路径、模型正文或环境配置进入终端摘要。
        parser.exit(2, f"Replay setup failed: {type(exc).__name__}; verify source event, finite limits and current configuration.\n")
    print(json.dumps({"run_directory": str(run_dir), "counts": summary["counts"],
                      "transport_summary": summary["transport_summary"]}, ensure_ascii=False, indent=2))
    return 1 if summary["counts"]["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
