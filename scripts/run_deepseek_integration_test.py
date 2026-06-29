#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mahjong_agent import (  # noqa: E402
    ChannelType,
    LLMBudgetLimits,
    LLMBudgetManager,
    LLMConfig,
    Message,
    OpenAICompatibleLLMResolver,
)


TZ = ZoneInfo("Asia/Shanghai")
LOG_PATH = ROOT / "logs" / "deepseek_integration.log"
DEFAULT_TEXT = "老板，今天下班有人打麻将吗？0.5或者1都行，烟也都可"


def load_dotenv(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def deepseek_api_key() -> tuple[str | None, str | None]:
    for key in ("MAHJONG_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY", "MAHJONG_LLM_API_KEY"):
        value = os.getenv(key)
        if value:
            return value, key
    return None, None


def redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if "key" in key.lower() or "authorization" in key.lower() or "token" in key.lower():
                redacted[key] = "<redacted>"
            else:
                redacted[key] = redact_payload(item)
        return redacted
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    return value


def write_integration_log(event: str, payload: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": datetime.now(TZ).isoformat(),
        "event": event,
        **redact_payload(payload),
    }
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def audit_logger(trace_id: str, event: str, payload: dict[str, Any]) -> None:
    write_integration_log(
        event,
        {
            "trace_id": trace_id,
            "provider": payload.get("provider"),
            "model": payload.get("model"),
            "stage": payload.get("stage"),
            "usage": payload.get("usage"),
            "budget": payload.get("budget"),
            "error": payload.get("error"),
        },
    )


def build_resolver(args: argparse.Namespace) -> tuple[OpenAICompatibleLLMResolver, str]:
    load_dotenv(ROOT / ".env")
    api_key, key_source = deepseek_api_key()
    if not api_key:
        raise RuntimeError(
            "DeepSeek integration test requires MAHJONG_DEEPSEEK_API_KEY, "
            "DEEPSEEK_API_KEY, or MAHJONG_LLM_API_KEY."
        )
    config = LLMConfig(
        api_key=api_key,
        provider="deepseek",
        model=args.model,
        base_url=args.base_url.rstrip("/"),
        timeout_seconds=args.timeout_seconds,
        temperature=0.1,
        max_completion_tokens=args.max_completion_tokens,
        thinking_enabled=False,
        response_format="json_object",
    )
    budget = LLMBudgetManager(
        LLMBudgetLimits(
            max_calls_per_day=args.max_calls,
            max_tokens_per_day=args.max_tokens,
            max_cost_per_day=args.max_cost,
            max_tokens_per_call=args.max_tokens_per_call,
            input_price_per_1k=args.input_price_per_1k,
            output_price_per_1k=args.output_price_per_1k,
        )
    )
    return OpenAICompatibleLLMResolver(config, budget_manager=budget, audit_logger=audit_logger), str(key_source)


def run_semantic_smoke(args: argparse.Namespace) -> int:
    try:
        resolver, key_source = build_resolver(args)
    except RuntimeError as exc:
        print(f"SKIP DeepSeek integration: {exc}")
        return 2

    trace_id = f"deepseek_it_{datetime.now(TZ).strftime('%Y%m%d%H%M%S')}"
    message = Message(
        text=args.text,
        sender_id="deepseek_it_user",
        sender_name="DeepSeek集成测试用户",
        channel_id="deepseek_integration",
        channel_type=ChannelType.MANUAL,
        metadata={"trace_id": trace_id, "budget_key": "deepseek_integration"},
    )
    context = {
        "known_local_aliases": {
            "cq": "杭麻财敲",
            "371": "三缺一",
            "272": "二缺二",
            "173": "一缺三",
            "半块": "0.5档",
            "五毛": "0.5档",
        },
        "integration_test": {
            "provider_must_be": "deepseek",
            "model_must_be": args.model,
            "no_side_effects": True,
        },
    }

    print("DeepSeek integration test:")
    print("  provider=deepseek")
    print(f"  model={resolver.config.model}")
    print(f"  base_url={resolver.config.base_url}")
    print(f"  api_key_source={key_source}")
    print("  api_key=<configured>")
    print(f"  trace_id={trace_id}")

    resolution = resolver.resolve(message, context=context)
    usage = resolution.usage.to_dict() if resolution.usage else None
    write_integration_log(
        "integration_result",
        {
            "trace_id": trace_id,
            "provider": resolver.config.provider,
            "model": resolver.config.model,
            "related": resolution.is_mahjong_related,
            "intent": resolution.intent,
            "confidence": resolution.confidence,
            "human_review": resolution.needs_human_review,
            "usage": usage,
            "budget": resolution.budget,
            "notes": resolution.notes,
        },
    )

    print("Result:")
    print(f"  related={resolution.is_mahjong_related}")
    print(f"  intent={resolution.intent}")
    print(f"  confidence={resolution.confidence}")
    print(f"  normalized_text={resolution.normalized_text}")
    print(f"  reply_text={resolution.reply_text}")
    print(f"  human_review={resolution.needs_human_review}")
    print(f"  usage={usage}")
    print(f"  log={LOG_PATH}")

    if not resolution.notes:
        print("FAIL: DeepSeek did not return resolver notes; call may not have completed normally.")
        return 1
    if resolver.config.provider != "deepseek":
        print("FAIL: integration test did not use provider=deepseek.")
        return 1
    if usage is None:
        print("FAIL: DeepSeek response did not include token usage; cannot prove a real provider response.")
        return 1
    if resolution.needs_human_review and not resolution.is_mahjong_related:
        print("FAIL: DeepSeek did not produce a usable mahjong semantic result.")
        return 1
    if resolution.confidence < args.min_confidence:
        print(f"FAIL: confidence {resolution.confidence} < min_confidence {args.min_confidence}.")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real DeepSeek integration smoke test.")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--model", default=os.getenv("MAHJONG_DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.getenv("MAHJONG_DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--timeout-seconds", type=float, default=float(os.getenv("MAHJONG_DEEPSEEK_TIMEOUT_SECONDS", "30")))
    parser.add_argument("--max-completion-tokens", type=int, default=int(os.getenv("MAHJONG_DEEPSEEK_MAX_COMPLETION_TOKENS", "1024")))
    parser.add_argument("--max-calls", type=int, default=int(os.getenv("MAHJONG_DEEPSEEK_MAX_CALLS", "2")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MAHJONG_DEEPSEEK_MAX_TOKENS", "12000")))
    parser.add_argument("--max-tokens-per-call", type=int, default=int(os.getenv("MAHJONG_DEEPSEEK_MAX_TOKENS_PER_CALL", "8000")))
    parser.add_argument("--max-cost", type=float, default=float(os.getenv("MAHJONG_DEEPSEEK_MAX_COST", "1.0")))
    parser.add_argument("--input-price-per-1k", type=float, default=float(os.getenv("MAHJONG_DEEPSEEK_INPUT_PRICE_PER_1K", "0")))
    parser.add_argument("--output-price-per-1k", type=float, default=float(os.getenv("MAHJONG_DEEPSEEK_OUTPUT_PRICE_PER_1K", "0")))
    parser.add_argument("--min-confidence", type=float, default=float(os.getenv("MAHJONG_DEEPSEEK_MIN_CONFIDENCE", "0.45")))
    args = parser.parse_args()
    return run_semantic_smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
