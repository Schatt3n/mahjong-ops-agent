from __future__ import annotations

import html
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_dotenv_defaults(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


from mahjong_agent_runtime import (  # noqa: E402
    AgentAction,
    AgentRuntime,
    AgentRuntimeResult,
    AgentLLMConfig,
    CustomerRelationship,
    CustomerProfile,
    JsonlTraceRecorder,
    OpenAICompatibleAgentClient,
    PendingInputBatch,
    PendingInputBatchStatus,
    PendingInputScheduler,
    QuotedMessageRef,
    SQLiteAgentStore,
    TokenBudget,
    ToolCall,
    ToolGateway,
    ToolResult,
    UserMessage,
    aggregate_pending_input_batch,
)
from mahjong_agent_runtime.summary import ContextSummaryManager, ContextSummaryPolicy  # noqa: E402
from mahjong_agent_runtime.models import InviteStatus  # noqa: E402
from mahjong_agent_runtime.tracing import validate_trace  # noqa: E402
from test_observability import observability_payload, run_fixed_suite  # noqa: E402


PORT = int(os.getenv("MAHJONG_AGENT_PORT", "8790"))
TRACE_PATH = Path(os.getenv("MAHJONG_AGENT_TRACE_PATH") or ROOT / "logs" / "agent_runtime_trace.log")
DB_PATH = Path(os.getenv("MAHJONG_AGENT_DB_PATH") or ROOT / "data" / "agent_runtime.sqlite3")
HERMES_RAW_LOG_PATH = Path(
    os.getenv("MAHJONG_HERMES_RAW_LOG_PATH") or ROOT / "logs" / "hermes_weixin_raw.jsonl"
)
ASTRBOT_RAW_LOG_PATH = Path(
    os.getenv("MAHJONG_ASTRBOT_RAW_LOG_PATH") or ROOT / "logs" / "astrbot_weixin_raw.jsonl"
)
WECHATY_RAW_LOG_PATH = Path(
    os.getenv("MAHJONG_WECHATY_RAW_LOG_PATH") or ROOT / "logs" / "wechaty_weixin_raw.jsonl"
)
WECHATY_INPUT_GATE_PROMPT_PATH = (
    SRC / "mahjong_agent_runtime" / "prompts" / "wechaty_input_gate.md"
)
WECHATY_CASUAL_CHAT_PROMPT_PATH = (
    SRC / "mahjong_agent_runtime" / "prompts" / "wechaty_casual_chat_reply.md"
)
DEFAULT_WECHATY_AGENT_WHITELIST: set[str] = set()
WECHAT_DISPLAY_QUOTE_PATTERN = re.compile(
    r"^\s*[「『](?P<quoted>.+?)[」』]\s*\n(?P<separator>(?:[-—–_]\s*){3,})\n(?P<reply>.+?)\s*$",
    re.DOTALL,
)

MAX_REQUEST_BYTES = int(os.getenv("MAHJONG_AGENT_MAX_REQUEST_BYTES", str(1024 * 1024)))
MAX_CONCURRENT_REQUESTS = max(1, int(os.getenv("MAHJONG_AGENT_MAX_CONCURRENT_REQUESTS", "8")))
REQUESTS_PER_MINUTE = max(1, int(os.getenv("MAHJONG_AGENT_REQUESTS_PER_MINUTE", "120")))
REQUEST_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)
_API_TOKEN: str | None = None
_API_TOKEN_LOCK = threading.Lock()


class RequestRateLimiter:
    """Small per-client sliding-window limiter for the local HTTP boundary."""

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, *, now: float | None = None) -> bool:
        actual_now = time.monotonic() if now is None else now
        cutoff = actual_now - self.window_seconds
        with self._lock:
            recent = [item for item in self._events.get(key, []) if item > cutoff]
            if len(recent) >= self.limit:
                self._events[key] = recent
                return False
            recent.append(actual_now)
            self._events[key] = recent
            return True


REQUEST_RATE_LIMITER = RequestRateLimiter(REQUESTS_PER_MINUTE)


def runtime_api_token() -> str:
    """Load an explicit token or create a private persistent local token."""

    global _API_TOKEN
    if _API_TOKEN:
        return _API_TOKEN
    with _API_TOKEN_LOCK:
        if _API_TOKEN:
            return _API_TOKEN
        load_dotenv_defaults(ROOT / ".env")
        configured = os.getenv("MAHJONG_AGENT_API_TOKEN", "").strip()
        if configured:
            _API_TOKEN = configured
            return configured
        token_path = Path(os.getenv("MAHJONG_AGENT_API_TOKEN_PATH") or ROOT / "data" / "runtime_api_token")
        token_path.parent.mkdir(parents=True, exist_ok=True)
        if token_path.exists():
            token = token_path.read_text(encoding="utf-8").strip()
        else:
            token = secrets.token_urlsafe(32)
            token_path.write_text(token + "\n", encoding="utf-8")
            token_path.chmod(0o600)
        _API_TOKEN = token
        return token


RUNTIME: AgentRuntime | None = None
INPUT_SCHEDULER: PendingInputScheduler | None = None


def build_runtime() -> AgentRuntime:
    load_dotenv_defaults(ROOT / ".env")
    llm_client = OpenAICompatibleAgentClient.from_env()
    if llm_client is None:
        raise RuntimeError("MAHJONG_LLM_API_KEY and MAHJONG_LLM_MODEL are required for AgentRuntime.")
    customer_visible_text_generation_client = build_customer_visible_text_generation_client()
    reply_self_review_client = build_reply_self_review_client()
    store = SQLiteAgentStore(DB_PATH)
    configured_rooms = split_env_list(os.getenv("MAHJONG_ROOM_IDS", ""))
    if configured_rooms:
        store.configure_rooms(configured_rooms)
    seed_customers(store)
    trace = JsonlTraceRecorder(TRACE_PATH)
    gateway = ToolGateway(store=store, trace_recorder=trace)
    summary_manager = None
    if env_bool("MAHJONG_CONTEXT_SUMMARY_ENABLED", True):
        summary_manager = ContextSummaryManager(
            store=store,
            llm_client=build_context_summary_client() or llm_client,
            trace_recorder=trace,
            policy=ContextSummaryPolicy(
                min_turns_before_summary=env_int("MAHJONG_CONTEXT_SUMMARY_MIN_TURNS", 12),
                min_turns_since_last_summary=env_int("MAHJONG_CONTEXT_SUMMARY_MIN_TURNS_SINCE_LAST", 6),
                max_recent_tokens_before_summary=env_int("MAHJONG_CONTEXT_SUMMARY_TOKEN_THRESHOLD", 3_000),
                max_turns_considered=env_int("MAHJONG_CONTEXT_SUMMARY_MAX_TURNS_CONSIDERED", 80),
                max_summary_input_tokens=env_int("MAHJONG_CONTEXT_SUMMARY_MAX_INPUT_TOKENS", 6_000),
                max_summary_chars=env_int("MAHJONG_CONTEXT_SUMMARY_MAX_CHARS", 800),
                max_open_questions=env_int("MAHJONG_CONTEXT_SUMMARY_MAX_OPEN_QUESTIONS", 10),
                min_confidence=env_float("MAHJONG_CONTEXT_SUMMARY_MIN_CONFIDENCE", 0.6),
                timeout_seconds=env_float("MAHJONG_CONTEXT_SUMMARY_TIMEOUT_SECONDS", 30.0),
            ),
        )
    main_budget = TokenBudget(
        max_tokens_per_call=env_int("MAHJONG_AGENT_MAX_TOKENS_PER_CALL", env_int("MAHJONG_LLM_MAX_TOKENS_PER_CALL", 32_000)),
        max_calls_per_turn=env_int("MAHJONG_AGENT_MAX_CALLS_PER_TURN", 8),
    )
    return AgentRuntime(
        llm_client=llm_client,
        store=store,
        tool_gateway=gateway,
        trace_recorder=trace,
        token_budget=main_budget,
        customer_visible_text_generation_token_budget=TokenBudget(
            max_tokens_per_call=env_int(
                "MAHJONG_TEXT_GENERATION_MAX_TOKENS_PER_CALL",
                env_int("MAHJONG_AGENT_REVIEW_MAX_TOKENS_PER_CALL", main_budget.max_tokens_per_call),
            ),
            max_calls_per_turn=env_int("MAHJONG_TEXT_GENERATION_MAX_CALLS_PER_TURN", 8),
        ),
        review_token_budget=TokenBudget(
            max_tokens_per_call=env_int("MAHJONG_AGENT_REVIEW_MAX_TOKENS_PER_CALL", main_budget.max_tokens_per_call),
            max_calls_per_turn=env_int("MAHJONG_AGENT_REVIEW_MAX_CALLS_PER_TURN", 8),
        ),
        max_steps=env_int("MAHJONG_AGENT_MAX_STEPS", 8),
        repeated_observation_limit=env_int("MAHJONG_AGENT_REPEATED_OBSERVATION_LIMIT", 2),
        consecutive_no_progress_limit=env_int("MAHJONG_AGENT_NO_PROGRESS_LIMIT", 2),
        max_progress_replans=env_int("MAHJONG_AGENT_MAX_PROGRESS_REPLANS", 1),
        max_cycle_period=env_int("MAHJONG_AGENT_MAX_CYCLE_PERIOD", 3),
        llm_timeout_seconds=float(env_int("MAHJONG_AGENT_LLM_TIMEOUT_SECONDS", 45)),
        context_summary_preemptive_ratio=env_float("MAHJONG_CONTEXT_SUMMARY_PREEMPTIVE_RATIO", 0.85),
        customer_visible_text_generation_enabled=env_bool("MAHJONG_TEXT_GENERATION_ENABLED", True),
        customer_visible_text_generation_client=customer_visible_text_generation_client,
        reply_self_review_enabled=env_bool("MAHJONG_AGENT_REPLY_SELF_REVIEW_ENABLED", True),
        reply_self_review_client=reply_self_review_client,
        context_summary_manager=summary_manager,
    )


def build_customer_visible_text_generation_client() -> OpenAICompatibleAgentClient | None:
    model = os.getenv("MAHJONG_TEXT_GENERATION_LLM_MODEL")
    if not model:
        return None
    provider = (os.getenv("MAHJONG_TEXT_GENERATION_LLM_PROVIDER") or os.getenv("MAHJONG_LLM_PROVIDER") or "openai_compatible").strip().lower()
    api_key = os.getenv("MAHJONG_TEXT_GENERATION_LLM_API_KEY") or os.getenv("MAHJONG_LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("MAHJONG_TEXT_GENERATION_LLM_MODEL is set, but no text generation or default LLM API key is available.")
    config = AgentLLMConfig(
        api_key=api_key,
        model=model,
        provider=provider,
        base_url=(os.getenv("MAHJONG_TEXT_GENERATION_LLM_BASE_URL") or os.getenv("MAHJONG_LLM_BASE_URL") or default_base_url(provider)).rstrip("/"),
        temperature=env_float("MAHJONG_TEXT_GENERATION_LLM_TEMPERATURE", env_float("MAHJONG_LLM_TEMPERATURE", 0.4)),
        max_tokens=env_int("MAHJONG_TEXT_GENERATION_LLM_MAX_COMPLETION_TOKENS", 1024),
    )
    return OpenAICompatibleAgentClient(config=config)


def build_reply_self_review_client() -> OpenAICompatibleAgentClient | None:
    model = os.getenv("MAHJONG_REPLY_REVIEW_LLM_MODEL")
    if not model:
        return None
    provider = (os.getenv("MAHJONG_REPLY_REVIEW_LLM_PROVIDER") or os.getenv("MAHJONG_LLM_PROVIDER") or "openai_compatible").strip().lower()
    api_key = os.getenv("MAHJONG_REPLY_REVIEW_LLM_API_KEY") or os.getenv("MAHJONG_LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("MAHJONG_REPLY_REVIEW_LLM_MODEL is set, but no reply review or default LLM API key is available.")
    config = AgentLLMConfig(
        api_key=api_key,
        model=model,
        provider=provider,
        base_url=(os.getenv("MAHJONG_REPLY_REVIEW_LLM_BASE_URL") or os.getenv("MAHJONG_LLM_BASE_URL") or default_base_url(provider)).rstrip("/"),
        temperature=env_float("MAHJONG_REPLY_REVIEW_LLM_TEMPERATURE", env_float("MAHJONG_LLM_TEMPERATURE", 0.0)),
        max_tokens=env_int("MAHJONG_REPLY_REVIEW_LLM_MAX_COMPLETION_TOKENS", 1024),
    )
    return OpenAICompatibleAgentClient(config=config)


def build_context_summary_client() -> OpenAICompatibleAgentClient | None:
    model = os.getenv("MAHJONG_CONTEXT_SUMMARY_LLM_MODEL")
    if not model:
        return None
    provider = (os.getenv("MAHJONG_CONTEXT_SUMMARY_LLM_PROVIDER") or os.getenv("MAHJONG_LLM_PROVIDER") or "openai_compatible").strip().lower()
    api_key = os.getenv("MAHJONG_CONTEXT_SUMMARY_LLM_API_KEY") or os.getenv("MAHJONG_LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("MAHJONG_CONTEXT_SUMMARY_LLM_MODEL is set, but no summary or default LLM API key is available.")
    config = AgentLLMConfig(
        api_key=api_key,
        model=model,
        provider=provider,
        base_url=(os.getenv("MAHJONG_CONTEXT_SUMMARY_LLM_BASE_URL") or default_base_url(provider)).rstrip("/"),
        temperature=env_float("MAHJONG_CONTEXT_SUMMARY_LLM_TEMPERATURE", env_float("MAHJONG_LLM_TEMPERATURE", 0.1)),
        max_tokens=env_int("MAHJONG_CONTEXT_SUMMARY_LLM_MAX_COMPLETION_TOKENS", 1200),
    )
    return OpenAICompatibleAgentClient(config=config)


def get_runtime() -> AgentRuntime:
    global RUNTIME
    if RUNTIME is None:
        RUNTIME = build_runtime()
    return RUNTIME


def trace_payload(runtime: AgentRuntime, trace_id: str) -> dict:
    events = runtime.trace_recorder.get_trace(trace_id)
    return {
        "trace_id": trace_id,
        "trace_log_path": str(TRACE_PATH),
        "events": [item.to_dict() for item in events],
        "completeness": validate_trace(events),
    }


def runtime_manifest(runtime: AgentRuntime) -> dict:
    return {
        "runtime": "mahjong_agent_runtime",
        "main_chain": "agent_runtime",
        "implementation_package": "mahjong_agent_runtime",
        "status": "ok",
        "legacy_reference_only": True,
        "legacy_entrypoints": {
            "legacy_analyze_endpoint": "not_exposed",
            "default_runtime_entrypoint": "scripts/run_agent_app.py",
        },
        "endpoints": {
            "message": ["/api/message"],
            "state": ["/api/state"],
            "traces": ["/api/traces"],
            "logs": ["/api/logs"],
            "badcases": ["/api/badcases"],
            "hermes_raw": ["/api/channels/hermes/raw"],
            "astrbot_raw": ["/api/channels/astrbot/raw"],
            "wechaty_raw": ["/api/channels/wechaty/raw"],
            "reset_state": ["/api/reset-state"],
            "test_observability": ["/tests", "/api/test-observability", "/api/test-observability/run"],
            "runtime": ["/api/runtime"],
            "health": ["/api/health"],
        },
        "available_tools": [item["name"] for item in runtime.tool_gateway.tool_specs_for_prompt()],
        "runtime_config": runtime_config(runtime),
    }


def tail_trace_log(limit: int = 200) -> list[str]:
    if not TRACE_PATH.exists():
        return []
    lines = TRACE_PATH.read_text(encoding="utf-8").splitlines()
    return lines[-max(1, int(limit)) :]


def tail_jsonl(path: Path, limit: int = 100) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-max(1, int(limit)) :]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def record_channel_raw_message(
    *,
    payload: dict,
    channel: str,
    log_path: Path,
    endpoint_path: str,
) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    fallback_message_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    source_message_id = str(payload.get("message_id") or payload.get("source_message_id") or fallback_message_id)
    source_message_hash = hashlib.sha256(f"{channel}:{source_message_id}".encode("utf-8")).hexdigest()[:12]
    trace_id = str(payload.get("trace_id") or f"trace_{channel}_{source_message_hash}")
    record = {
        "trace_id": trace_id,
        "source": f"{channel}_weixin",
        "source_message_id": source_message_id,
        "received_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "payload": payload,
    }
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    JsonlTraceRecorder(TRACE_PATH).record(
        trace_id,
        f"{channel}_raw_message_received",
        {
            "direction": "input",
            "path": endpoint_path,
            "source_message_id": source_message_id,
            "raw_log_path": str(log_path),
            "payload": payload,
        },
    )
    return record


def record_hermes_raw_message(payload: dict) -> dict:
    return record_channel_raw_message(
        payload=payload,
        channel="hermes",
        log_path=HERMES_RAW_LOG_PATH,
        endpoint_path="/api/channels/hermes/raw",
    )


def record_astrbot_raw_message(payload: dict) -> dict:
    return record_channel_raw_message(
        payload=payload,
        channel="astrbot",
        log_path=ASTRBOT_RAW_LOG_PATH,
        endpoint_path="/api/channels/astrbot/raw",
    )


def record_wechaty_raw_message(payload: dict) -> dict:
    return record_channel_raw_message(
        payload=payload,
        channel="wechaty",
        log_path=WECHATY_RAW_LOG_PATH,
        endpoint_path="/api/channels/wechaty/raw",
    )


def _wechaty_nested_text(payload: dict, *path: str) -> str:
    value = payload
    for key in path:
        if not isinstance(value, dict):
            return ""
        value = value.get(key)
    return str(value or "").strip()


WECHATY_MESSAGE_TYPE_MODALITIES = {
    1: "file",
    2: "voice",
    5: "sticker",
    6: "image",
    7: "text",
    8: "location",
    14: "link",
    15: "video",
}


MEDIA_KIND_HINTS = {
    "audio": "voice",
    "voice": "voice",
    "silk": "voice",
    "amr": "voice",
    "image": "image",
    "img": "image",
    "photo": "image",
    "picture": "image",
    "sticker": "sticker",
    "emoticon": "sticker",
    "emoji": "sticker",
    "video": "video",
    "file": "file",
    "attachment": "file",
}


TRANSCRIPT_FIELD_SOURCES = [
    ("audio_transcript", "audio_transcript"),
    ("voice_transcript", "voice_transcript"),
    ("asr_text", "asr_text"),
    ("transcript", "transcript"),
    ("recognized_text", "recognized_text"),
    ("image_ocr_text", "image_ocr_text"),
    ("ocr_text", "ocr_text"),
]


SAFE_USER_MESSAGE_METADATA_KEYS = {
    "channel",
    "platform_name",
    "source",
    "message_type",
    "source_message_id",
    "is_room",
    "self_message",
    "reply_target_id",
    "conversation_target_type",
    "has_text",
    "text_source",
    "modalities",
    "media_candidates",
    "raw_observation_summary",
    "media_requires_transcription",
    "media_requires_ocr",
    "transcript_confidence",
    "ocr_confidence",
    "language",
}


SAFE_QUOTED_MESSAGE_METADATA_KEYS = {
    "source",
    "raw_chatusr",
    "platform_message_id",
    "platformMessageId",
    "source_message_id",
    "sourceMessageId",
    "message_type",
    "text_source",
    "channel",
}


def safe_text_preview(value: object, limit: int = 120) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def safe_metadata_media_candidates(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    safe_items: list[dict] = []
    for item in value[:12]:
        if not isinstance(item, dict):
            continue
        safe_item = {
            "path": safe_text_preview(item.get("path"), 160),
            "kind": safe_text_preview(item.get("kind"), 40),
            "value_type": safe_text_preview(item.get("value_type"), 40),
        }
        text_preview = safe_text_preview(item.get("text_preview"), 120)
        if text_preview:
            safe_item["text_preview"] = text_preview
        safe_items.append({key: val for key, val in safe_item.items() if val not in {"", None}})
    return safe_items


def safe_metadata_observation_summary(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    summary: dict[str, int] = {}
    for key in ("quote_candidate_count", "media_candidate_count"):
        try:
            summary[key] = max(int(value.get(key) or 0), 0)
        except (TypeError, ValueError):
            continue
    return summary


def sanitize_user_message_metadata(metadata: object) -> dict:
    if not isinstance(metadata, dict):
        return {}
    sanitized: dict[str, object] = {}
    for key, value in metadata.items():
        if key not in SAFE_USER_MESSAGE_METADATA_KEYS:
            continue
        if key == "modalities":
            if isinstance(value, list):
                sanitized[key] = [safe_text_preview(item, 40) for item in value[:12] if str(item or "").strip()]
            continue
        if key == "media_candidates":
            sanitized[key] = safe_metadata_media_candidates(value)
            continue
        if key == "raw_observation_summary":
            sanitized[key] = safe_metadata_observation_summary(value)
            continue
        if isinstance(value, bool) or value is None:
            sanitized[key] = value
        elif isinstance(value, (int, float)):
            sanitized[key] = value
        elif isinstance(value, str):
            sanitized[key] = safe_text_preview(value, 160)
    return sanitized


def sanitize_quoted_message_metadata(metadata: object) -> dict:
    if not isinstance(metadata, dict):
        return {}
    sanitized: dict[str, object] = {}
    for key, value in metadata.items():
        if key not in SAFE_QUOTED_MESSAGE_METADATA_KEYS:
            continue
        if isinstance(value, bool) or value is None:
            sanitized[key] = value
        elif isinstance(value, (int, float)):
            sanitized[key] = value
        elif isinstance(value, str):
            sanitized[key] = safe_text_preview(value, 160)
    return sanitized


def text_from_wechaty_payload(payload: dict) -> tuple[str, str]:
    direct_text = str(payload.get("text") or payload.get("raw_text") or "").strip()
    if direct_text:
        return direct_text, "text"
    raw_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    for source in (payload, metadata, raw_payload):
        if not isinstance(source, dict):
            continue
        for key, source_name in TRANSCRIPT_FIELD_SOURCES:
            value = str(source.get(key) or "").strip()
            if value:
                return value, source_name
    return "", ""


def parse_wechat_display_quote_text(text: str) -> tuple[str, QuotedMessageRef] | None:
    match = WECHAT_DISPLAY_QUOTE_PATTERN.match(str(text or ""))
    if not match:
        return None
    quoted_text = str(match.group("quoted") or "").strip()
    reply_text = str(match.group("reply") or "").strip()
    if not quoted_text or not reply_text:
        return None
    quote_id = f"display_quote_{hashlib.sha256(quoted_text.encode('utf-8')).hexdigest()[:12]}"
    return (
        reply_text,
        QuotedMessageRef(
            message_id=quote_id,
            text=quoted_text,
            metadata=sanitize_quoted_message_metadata({"source": "wechat_display_quote"}),
        ),
    )


def media_kind_from_hint(value: object) -> str:
    lowered = str(value or "").lower()
    for hint, kind in MEDIA_KIND_HINTS.items():
        if hint in lowered:
            return kind
    return ""


def compact_media_candidates(raw_observation: dict) -> list[dict]:
    raw_candidates = raw_observation.get("media_candidates")
    if not isinstance(raw_candidates, list):
        return []
    compacted: list[dict] = []
    for candidate in raw_candidates[:12]:
        if not isinstance(candidate, dict):
            continue
        path = str(candidate.get("path") or "").strip()
        raw_kind = str(candidate.get("kind") or candidate.get("type") or "").strip()
        value = candidate.get("value")
        kind = media_kind_from_hint(raw_kind) or media_kind_from_hint(path) or media_kind_from_hint(value)
        item = {
            "path": safe_text_preview(path, 160),
            "kind": safe_text_preview(kind or "unknown_media", 40),
            "value_type": type(value).__name__,
        }
        if isinstance(value, str):
            item["text_preview"] = safe_text_preview(value)
        compacted.append(item)
    return compacted


def infer_wechaty_modalities(payload: dict, *, text: str, media_candidates: list[dict]) -> list[str]:
    modalities: list[str] = []
    if text:
        modalities.append("text")
    raw_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
    raw_message_type = payload.get("message_type", raw_payload.get("type"))
    try:
        message_type = int(raw_message_type)
    except (TypeError, ValueError):
        message_type = None
    if message_type is not None:
        modality = WECHATY_MESSAGE_TYPE_MODALITIES.get(message_type)
        if modality and modality not in modalities:
            modalities.append(modality)
    for candidate in media_candidates:
        kind = str(candidate.get("kind") or "").strip()
        if kind and kind not in {"unknown_media", "text"} and kind not in modalities:
            modalities.append(kind)
    if not modalities and raw_message_type not in {None, "", 0, "0"}:
        modalities.append("unknown_media")
    return modalities


def has_wechaty_ocr_text(payload: dict) -> bool:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    for key in ("image_ocr_text", "ocr_text", "recognized_text"):
        if str(metadata.get(key) or payload.get(key) or "").strip():
            return True
    return False


def build_wechaty_message_metadata(payload: dict, *, text: str, text_source: str) -> dict:
    raw_observation = payload.get("raw_observation") if isinstance(payload.get("raw_observation"), dict) else {}
    raw_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
    media_candidates = compact_media_candidates(raw_observation)
    modalities = infer_wechaty_modalities(payload, text=text, media_candidates=media_candidates)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    has_transcript = bool(text and text_source and text_source != "text")
    self_message = bool(payload.get("self_message"))
    sender_id = str(payload.get("sender_id") or raw_payload.get("talkerId") or "").strip()
    listener_id = str(raw_payload.get("listenerId") or "").strip()
    reply_target_id = listener_id if self_message and listener_id else sender_id
    return {
        **sanitize_user_message_metadata(metadata),
        "channel": "wechaty",
        "platform_name": str(payload.get("platform_name") or "wechaty"),
        "message_type": payload.get("message_type"),
        "source_message_id": str(payload.get("source_message_id") or payload.get("message_id") or ""),
        "is_room": bool(payload.get("is_room")),
        "self_message": self_message,
        "reply_target_id": reply_target_id,
        "conversation_target_type": "room" if bool(payload.get("is_room")) else "contact",
        "has_text": bool(text),
        "text_source": text_source or None,
        "modalities": modalities,
        "media_candidates": media_candidates,
        "raw_observation_summary": {
            "quote_candidate_count": len(raw_observation.get("quote_candidates") or [])
            if isinstance(raw_observation.get("quote_candidates"), list)
            else 0,
            "media_candidate_count": len(media_candidates),
        },
        "media_requires_transcription": any(item in modalities for item in ("voice", "audio")) and not has_transcript,
        "media_requires_ocr": "image" in modalities and not has_wechaty_ocr_text(payload),
    }


def _first_wechat_refermsg_xml(value: object, *, depth: int = 0) -> str:
    if depth > 4 or value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        return text if "<refermsg" in text.lower() else ""
    if isinstance(value, dict):
        for child in value.values():
            found = _first_wechat_refermsg_xml(child, depth=depth + 1)
            if found:
                return found
    if isinstance(value, list):
        for child in value[:20]:
            found = _first_wechat_refermsg_xml(child, depth=depth + 1)
            if found:
                return found
    return ""


def _element_text(root: ET.Element, *paths: str) -> str:
    for path in paths:
        node = root.find(path)
        if node is not None and node.text:
            return html.unescape(str(node.text)).strip()
    return ""


def _runtime_wechat_conversation_id(value: str) -> str | None:
    normalized = str(value or "").strip()
    if normalized.startswith(("wechaty:contact:", "wechaty:room:")):
        return normalized
    return None


def parse_wechat_refermsg_quoted_message_ref(value: object) -> QuotedMessageRef | None:
    raw_text = str(value or "").strip()
    if "<refermsg" not in raw_text.lower():
        return None
    xml_text = html.unescape(raw_text)
    roots: list[ET.Element] = []
    for candidate in (xml_text, f"<root>{xml_text}</root>"):
        try:
            roots.append(ET.fromstring(candidate))
        except ET.ParseError:
            continue
    for root in roots:
        refermsg = root.find(".//refermsg")
        if refermsg is None:
            continue
        message_id = _element_text(refermsg, "svrid", "msgid", "msgId", "messageId", "id")
        text = _element_text(refermsg, "content", "displaycontent", "displayContent", "text")
        if not message_id and not text:
            continue
        raw_chatusr = _element_text(refermsg, "chatusr", "conversationId", "conversation_id")
        metadata = {"source": "wechat_refermsg_xml"}
        if raw_chatusr:
            metadata["raw_chatusr"] = raw_chatusr
        return QuotedMessageRef(
            message_id=message_id,
            sender_id=_element_text(refermsg, "fromusr", "senderId", "sender_id") or None,
            sender_name=_element_text(refermsg, "displayname", "senderName", "sender_name") or None,
            text=text,
            conversation_id=_runtime_wechat_conversation_id(raw_chatusr),
            metadata=sanitize_quoted_message_metadata(metadata),
        )
    return None


def parse_quoted_message_ref(payload: dict) -> QuotedMessageRef | None:
    raw = payload.get("quoted_message")
    if raw is None:
        raw = payload.get("quoted")
    if raw is None:
        raw_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        raw = raw_payload.get("quoted_message") or raw_payload.get("quoted") or raw_payload.get("quote")
    if raw is None:
        raw_observation = payload.get("raw_observation") if isinstance(payload.get("raw_observation"), dict) else {}
        candidates = raw_observation.get("quote_candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                path = str(candidate.get("path") or "").lower()
                if not any(token in path for token in ("quote", "quoted", "refer", "reference")):
                    value = candidate.get("value")
                    if not isinstance(value, str) or "<refermsg" not in value.lower():
                        continue
                else:
                    value = candidate.get("value")
                if isinstance(value, (dict, str)):
                    raw = value
                    break
    if raw is None:
        raw_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        raw = _first_wechat_refermsg_xml(raw_payload)
    if isinstance(raw, str):
        return parse_wechat_refermsg_quoted_message_ref(raw)
    if not isinstance(raw, dict):
        return None

    message_id = str(
        raw.get("message_id")
        or raw.get("source_message_id")
        or raw.get("id")
        or raw.get("msgId")
        or raw.get("msg_id")
        or raw.get("messageId")
        or raw.get("sourceMessageId")
        or ""
    ).strip()
    text = str(
        raw.get("text")
        or raw.get("raw_text")
        or raw.get("content")
        or raw.get("message_text")
        or raw.get("messageText")
        or raw.get("body")
        or ""
    ).strip()
    if not message_id and not text:
        return None

    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    return QuotedMessageRef(
        message_id=message_id,
        sender_id=str(raw.get("sender_id") or raw.get("senderId") or "") or None,
        sender_name=str(raw.get("sender_name") or raw.get("senderName") or "") or None,
        text=text,
        conversation_id=str(raw.get("conversation_id") or raw.get("conversationId") or "") or None,
        business_ref_type=str(raw.get("business_ref_type") or raw.get("businessRefType") or "") or None,
        business_ref_id=str(raw.get("business_ref_id") or raw.get("businessRefId") or "") or None,
        metadata=sanitize_quoted_message_metadata(metadata),
    )


def build_api_user_message(payload: dict) -> tuple[UserMessage | None, list[str]]:
    required_fields = ("conversation_id", "sender_id", "sender_name", "text")
    missing_fields = [field for field in required_fields if not str(payload.get(field) or "").strip()]
    if missing_fields:
        return None, missing_fields
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return (
        UserMessage(
            conversation_id=str(payload["conversation_id"]).strip(),
            sender_id=str(payload["sender_id"]).strip(),
            sender_name=str(payload["sender_name"]).strip(),
            text=str(payload["text"]).strip(),
            message_id=str(payload.get("message_id") or "").strip() or None,
            quoted_message=parse_quoted_message_ref(payload),
            metadata=sanitize_user_message_metadata(metadata),
        ),
        [],
    )


def link_delivered_message_reference(runtime: AgentRuntime, payload: dict) -> dict:
    conversation_id = str(payload.get("conversation_id") or payload.get("conversationId") or "").strip()
    message_id = str(
        payload.get("platform_message_id")
        or payload.get("platformMessageId")
        or payload.get("delivered_message_id")
        or payload.get("deliveredMessageId")
        or payload.get("message_id")
        or payload.get("messageId")
        or ""
    ).strip()
    source_message_id = str(
        payload.get("source_message_id")
        or payload.get("sourceMessageId")
        or payload.get("source_reference_message_id")
        or payload.get("sourceReferenceMessageId")
        or payload.get("draft_id")
        or payload.get("draftId")
        or ""
    ).strip()
    business_ref_type = str(payload.get("business_ref_type") or payload.get("businessRefType") or "").strip()
    business_ref_id = str(payload.get("business_ref_id") or payload.get("businessRefId") or "").strip()
    trace_seed = json.dumps(
        {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "source_message_id": source_message_id,
            "business_ref_type": business_ref_type,
            "business_ref_id": business_ref_id,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    trace_id = str(payload.get("trace_id") or payload.get("traceId") or "").strip()
    if not trace_id:
        trace_id = f"trace_message_ref_{hashlib.sha256(trace_seed.encode('utf-8')).hexdigest()[:12]}"
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    try:
        reference = runtime.store.link_message_reference(
            conversation_id=conversation_id,
            message_id=message_id,
            source_message_id=source_message_id or None,
            business_ref_type=business_ref_type or None,
            business_ref_id=business_ref_id or None,
            channel=str(payload.get("channel") or payload.get("platform_name") or payload.get("platformName") or "").strip() or None,
            text=str(payload.get("text") or payload.get("message_text") or payload.get("messageText") or "").strip() or None,
            metadata={
                **metadata,
                "source": metadata.get("source") or "delivered_message_reference",
                "platform_message_id": message_id,
            },
        )
    except ValueError as error:
        result = {
            "ok": False,
            "trace_id": trace_id,
            "error": str(error),
            "conversation_id": conversation_id,
            "message_id": message_id,
            "source_message_id": source_message_id,
            "business_ref_type": business_ref_type,
            "business_ref_id": business_ref_id,
        }
        runtime.trace_recorder.record(trace_id, "message_reference_link_failed", result, level="WARN")
        return result
    result = {"ok": True, "trace_id": trace_id, "reference": reference.to_dict()}
    runtime.trace_recorder.record(trace_id, "message_reference_linked", result)
    return result


def split_env_list(raw: str) -> list[str]:
    normalized = str(raw or "").replace("\n", ",").replace(";", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def wechaty_identity_values(payload: dict) -> list[str]:
    raw_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
    talker = payload.get("talker") if isinstance(payload.get("talker"), dict) else {}
    values = [
        payload.get("sender_id"),
        payload.get("sender_name"),
        raw_payload.get("talkerId"),
        raw_payload.get("fromId"),
        _wechaty_nested_text(talker, "id"),
        _wechaty_nested_text(talker, "name"),
        _wechaty_nested_text(talker, "alias"),
        _wechaty_nested_text(talker, "payload", "id"),
        _wechaty_nested_text(talker, "payload", "name"),
        _wechaty_nested_text(talker, "payload", "alias"),
        _wechaty_nested_text(talker, "payload", "weixin"),
    ]
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def wechaty_agent_whitelist_hits(payload: dict) -> list[str]:
    allowed = {item.lower() for item in configured_wechaty_whitelist()}
    if not allowed:
        return []
    return [item for item in wechaty_identity_values(payload) if item.lower() in allowed]


def configured_wechaty_whitelist() -> set[str]:
    """Read private WeChat identities from local configuration, never source."""

    values = set(split_env_list(os.getenv("MAHJONG_WECHATY_AGENT_WHITELIST", "")))
    path = Path(
        os.getenv("MAHJONG_WECHATY_AGENT_WHITELIST_PATH")
        or ROOT / "data" / "wechaty_whitelist.local.json"
    )
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_items = payload.get("identities", []) if isinstance(payload, dict) else payload
            values.update(str(item).strip() for item in raw_items if str(item).strip())
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return values


def build_wechaty_input_gate_client() -> OpenAICompatibleAgentClient | None:
    model = os.getenv("MAHJONG_WECHATY_INPUT_GATE_LLM_MODEL")
    if not model:
        return None
    provider = (
        os.getenv("MAHJONG_WECHATY_INPUT_GATE_LLM_PROVIDER")
        or os.getenv("MAHJONG_LLM_PROVIDER")
        or "openai_compatible"
    ).strip().lower()
    api_key = (
        os.getenv("MAHJONG_WECHATY_INPUT_GATE_LLM_API_KEY")
        or os.getenv("MAHJONG_LLM_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
    )
    if not api_key:
        raise RuntimeError("MAHJONG_WECHATY_INPUT_GATE_LLM_MODEL is set, but no input gate or default LLM API key is available.")
    config = AgentLLMConfig(
        api_key=api_key,
        model=model,
        provider=provider,
        base_url=(
            os.getenv("MAHJONG_WECHATY_INPUT_GATE_LLM_BASE_URL")
            or os.getenv("MAHJONG_LLM_BASE_URL")
            or default_base_url(provider)
        ).rstrip("/"),
        temperature=env_float("MAHJONG_WECHATY_INPUT_GATE_LLM_TEMPERATURE", 0.0),
        max_tokens=env_int("MAHJONG_WECHATY_INPUT_GATE_MAX_COMPLETION_TOKENS", 500),
    )
    return OpenAICompatibleAgentClient(config=config)


def parse_wechaty_input_gate_response(raw_response: str) -> tuple[dict, list[str]]:
    errors: list[str] = []
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return (
            {
                "action": "ignore",
                "should_route": False,
                "should_wait": False,
                "category": "uncertain",
                "confidence": 0.0,
                "reasoning_summary": "入口分流模型没有返回合法 JSON。",
                "evidence": [],
            },
            [f"invalid json: {exc}"],
        )
    if not isinstance(payload, dict):
        return (
            {
                "action": "ignore",
                "should_route": False,
                "should_wait": False,
                "category": "uncertain",
                "confidence": 0.0,
                "reasoning_summary": "入口分流模型返回值不是对象。",
                "evidence": [],
            },
            ["input gate response must be an object"],
        )
    allowed_categories = {
        "operational",
        "followup_answer",
        "candidate_reply",
        "casual_chat",
        "non_mahjong",
        "uncertain",
    }
    if not isinstance(payload.get("should_route"), bool):
        errors.append("should_route must be boolean")
    category = str(payload.get("category") or "").strip()
    if category not in allowed_categories:
        errors.append(f"category invalid {category!r}")
        category = "uncertain"
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        errors.append("confidence must be number")
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reasoning_summary = str(payload.get("reasoning_summary") or "").strip()
    if not reasoning_summary:
        errors.append("reasoning_summary required")
        reasoning_summary = "入口分流结果缺少原因。"
    evidence_raw = payload.get("evidence")
    evidence = [str(item).strip() for item in evidence_raw if str(item).strip()][:3] if isinstance(evidence_raw, list) else []
    if not isinstance(evidence_raw, list):
        errors.append("evidence must be array")
    allowed_actions = {"process_business", "process_casual", "wait_for_more_input", "ignore"}
    action = str(payload.get("action") or "").strip()
    if not action:
        if payload.get("should_route") is True:
            action = "process_business"
        elif category in {"casual_chat", "non_mahjong"}:
            action = "process_casual"
        else:
            action = "ignore"
    if action not in allowed_actions:
        errors.append(f"action invalid {action!r}")
        action = "ignore"
    return (
        {
            "action": action,
            "should_route": action == "process_business",
            "should_wait": action == "wait_for_more_input",
            "category": category,
            "confidence": confidence,
            "reasoning_summary": reasoning_summary,
            "evidence": evidence,
        },
        errors,
    )


def build_wechaty_input_gate_payload(
    message: UserMessage,
    runtime: AgentRuntime,
    *,
    input_batch: PendingInputBatch | None = None,
    quiet_period_elapsed: bool = False,
) -> dict:
    recent_turns = [
        item.to_dict()
        for item in runtime.store.recent_turns(
            message.conversation_id,
            env_int("MAHJONG_WECHATY_INPUT_GATE_RECENT_TURNS", 8),
        )
    ]
    active_games = [item.to_dict() for item in runtime.store.active_games(message.conversation_id)]
    profile = runtime.store.customers.get(message.sender_id)
    return {
        "runtime": "mahjong_agent_runtime",
        "gate": "wechaty_input_gate",
        "current_message": message.to_dict(),
        "input_window": {
            "enabled": input_batch is not None,
            "quiet_period_elapsed": bool(quiet_period_elapsed),
            "quiet_period_seconds": env_float("MAHJONG_INPUT_QUIET_PERIOD_SECONDS", 30.0),
            "batch_id": input_batch.batch_id if input_batch else None,
            "batch_version": input_batch.version if input_batch else None,
            "fragment_count": len(input_batch.fragments) if input_batch else 1,
            "fragments": [
                {
                    "message_id": str(item.get("message_id") or ""),
                    "text": str(item.get("text") or ""),
                    "sent_at": str(item.get("sent_at") or ""),
                }
                for item in (input_batch.fragments if input_batch else [message.to_dict()])
            ],
        },
        "recent_conversation": recent_turns,
        "sender_profile": profile.to_dict() if profile else None,
        "active_games": active_games,
        "policy": {
            "purpose": "只判断微信消息是否进入麻将馆运营主流程。",
            "route_when": [
                "麻将馆运营相关",
                "组局/找人/咨询现有局/加入/取消/改时间/候选人回复",
                "对上一轮麻将运营追问的短答",
            ],
            "do_not_route_when": ["日常闲聊", "与麻将馆运营无关", "纯表情或无意义内容"],
            "no_user_reply_from_gate": True,
            "adaptive_wait": (
                "信息可能仍在分段输入且尚未静默时，可以 wait_for_more_input；"
                "静默期已结束后禁止再次等待，必须处理、闲聊或忽略。"
            ),
        },
        "output_contract": {
            "format": "json_object",
            "required_keys": ["action", "should_route", "category", "confidence", "reasoning_summary", "evidence"],
            "actions": ["process_business", "process_casual", "wait_for_more_input", "ignore"],
            "categories": ["operational", "followup_answer", "candidate_reply", "casual_chat", "non_mahjong", "uncertain"],
        },
    }


def run_wechaty_input_gate(
    message: UserMessage,
    *,
    trace_id: str,
    runtime: AgentRuntime,
    input_batch: PendingInputBatch | None = None,
    quiet_period_elapsed: bool = False,
) -> dict:
    if not env_bool("MAHJONG_WECHATY_INPUT_GATE_ENABLED", True):
        return {
            "enabled": False,
            "action": "process_business",
            "should_route": True,
            "should_wait": False,
            "category": "disabled",
            "confidence": 1.0,
            "reasoning_summary": "Wechaty input gate disabled by env.",
            "evidence": [],
            "errors": [],
        }
    client = build_wechaty_input_gate_client() or runtime.llm_client
    payload = build_wechaty_input_gate_payload(
        message,
        runtime,
        input_batch=input_batch,
        quiet_period_elapsed=quiet_period_elapsed,
    )
    messages = [
        {"role": "system", "content": WECHATY_INPUT_GATE_PROMPT_PATH.read_text(encoding="utf-8")},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]
    runtime.trace_recorder.record(trace_id, "wechaty_input_gate_prompt", {"messages": messages})
    started = time.perf_counter()
    try:
        raw_response = client.complete(
            messages,
            trace_id=trace_id,
            timeout_seconds=env_float("MAHJONG_WECHATY_INPUT_GATE_TIMEOUT_SECONDS", 12.0),
        )
    except Exception as exc:
        fail_open = env_bool("MAHJONG_WECHATY_INPUT_GATE_FAIL_OPEN", False)
        decision = {
            "enabled": True,
            "action": "process_business" if fail_open else "ignore",
            "should_route": fail_open,
            "should_wait": False,
            "category": "uncertain",
            "confidence": 0.0,
            "reasoning_summary": f"入口分流模型调用失败：{type(exc).__name__}",
            "evidence": [],
            "errors": [str(exc)],
            "fail_open": fail_open,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
        runtime.trace_recorder.record(trace_id, "wechaty_input_gate_error", decision, level="ERROR")
        return decision
    decision, errors = parse_wechaty_input_gate_response(raw_response)
    decision = {
        "enabled": True,
        **decision,
        "errors": errors,
        "raw_response": raw_response,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
    if quiet_period_elapsed and decision.get("action") == "wait_for_more_input":
        business_like = decision.get("category") in {
            "operational",
            "followup_answer",
            "candidate_reply",
            "uncertain",
        }
        decision.update(
            {
                "action": "process_business" if business_like else "process_casual",
                "should_route": business_like,
                "should_wait": False,
            }
        )
        decision.setdefault("normalizations", []).append(
            "wait_for_more_input was converted because quiet period already elapsed"
        )
    runtime.trace_recorder.record(trace_id, "wechaty_input_gate_response", {"content": raw_response, "elapsed_ms": decision["elapsed_ms"]})
    runtime.trace_recorder.record(trace_id, "wechaty_input_gate_decision", decision, level="WARN" if errors else "INFO")
    if errors and not env_bool("MAHJONG_WECHATY_INPUT_GATE_FAIL_OPEN", False):
        decision.update({"action": "ignore", "should_route": False, "should_wait": False})
    return decision


def build_wechaty_casual_chat_payload(
    message: UserMessage,
    runtime: AgentRuntime,
    *,
    gate_decision: dict,
) -> dict:
    recent_turns = [
        item.to_dict()
        for item in runtime.store.recent_turns(
            message.conversation_id,
            env_int("MAHJONG_WECHATY_CASUAL_CHAT_RECENT_TURNS", 8),
        )
    ]
    active_games = [item.to_dict() for item in runtime.store.active_games(message.conversation_id)]
    profile = runtime.store.customers.get(message.sender_id)
    return {
        "runtime": "mahjong_agent_runtime",
        "task": "wechaty_casual_chat_reply",
        "current_message": message.to_dict(),
        "input_gate_decision": gate_decision,
        "recent_conversation": recent_turns,
        "sender_profile": profile.to_dict() if profile else None,
        "sender_relationships": runtime.store.relationship_context_for_sender(message.sender_id, runtime.store.active_games()),
        "active_games": active_games,
        "policy": {
            "purpose": "只处理未进入麻将运营主流程的闲聊或非运营消息。",
            "do_not_modify_state": True,
            "do_not_call_business_tools": True,
            "customer_visible_review_required": True,
            "reply_style": "像麻将馆老板的微信短回复，简短自然，不客服化。",
            "privacy": [
                "不能透露系统、模型、Agent、工具、日志、后台、测试通道等实现信息。",
                "不能透露其他用户、候选人、局内人员、客户画像或隐私信息。",
                "不能把普通闲聊伪装成已经开始组局，例如不要说“我帮你问问”。",
            ],
        },
        "output_contract": {
            "format": "json_object",
            "required_keys": ["should_reply", "reply_to_user", "reasoning_summary", "needs_human"],
            "field_types": {
                "should_reply": "boolean",
                "reply_to_user": "string; empty when should_reply=false",
                "reasoning_summary": "string",
                "needs_human": "boolean",
            },
        },
    }


def parse_wechaty_casual_chat_response(raw_response: str) -> tuple[dict, list[str]]:
    errors: list[str] = []
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return {
            "should_reply": False,
            "reply_to_user": "",
            "reasoning_summary": "",
            "needs_human": True,
        }, [f"casual chat response is not valid JSON: {exc.msg}"]
    if not isinstance(payload, dict):
        return {
            "should_reply": False,
            "reply_to_user": "",
            "reasoning_summary": "",
            "needs_human": True,
        }, ["casual chat response JSON root must be object"]
    for key in ["should_reply", "reply_to_user", "reasoning_summary", "needs_human"]:
        if key not in payload:
            errors.append(f"missing casual chat response key: {key}")
    if "should_reply" in payload and not isinstance(payload.get("should_reply"), bool):
        errors.append("should_reply must be boolean")
    if "reply_to_user" in payload and not isinstance(payload.get("reply_to_user"), str):
        errors.append("reply_to_user must be string")
    if "reasoning_summary" in payload and not isinstance(payload.get("reasoning_summary"), str):
        errors.append("reasoning_summary must be string")
    if "needs_human" in payload and not isinstance(payload.get("needs_human"), bool):
        errors.append("needs_human must be boolean")
    if payload.get("should_reply") and not str(payload.get("reply_to_user") or "").strip():
        errors.append("should_reply=true requires non-empty reply_to_user")
    if not payload.get("should_reply") and str(payload.get("reply_to_user") or "").strip():
        errors.append("should_reply=false requires empty reply_to_user")
    return {
        "should_reply": bool(payload.get("should_reply")),
        "reply_to_user": str(payload.get("reply_to_user") or "").strip(),
        "reasoning_summary": str(payload.get("reasoning_summary") or ""),
        "needs_human": bool(payload.get("needs_human")),
    }, errors


def wechaty_message_idempotency_key(message: UserMessage) -> str:
    return f"conversation:{message.conversation_id}:sender:{message.sender_id}:message:{message.message_id}"


def review_result_safe_text(review_result: ToolResult, *, item_id: str) -> str:
    item_reviews = review_result.result.get("item_reviews")
    if not isinstance(item_reviews, list) or bool(review_result.result.get("needs_human")):
        return ""
    for item in item_reviews:
        if not isinstance(item, dict):
            continue
        if str(item.get("item_id") or "") == item_id:
            return str(item.get("suggested_safe_text") or "").strip()
    return ""


def handle_wechaty_casual_chat(
    message: UserMessage,
    *,
    trace_id: str,
    runtime: AgentRuntime,
    gate_decision: dict,
) -> AgentRuntimeResult:
    with runtime._conversation_lock(message.conversation_id):
        message_key = wechaty_message_idempotency_key(message)
        cached = runtime.store.idempotent_message_result(message_key)
        if cached is not None:
            runtime.trace_recorder.record(trace_id, "user_input", {"message": message.to_dict(), "route": "casual_chat"})
            runtime.trace_recorder.record(
                trace_id,
                "message_deduplicated",
                {
                    "message_id": message.message_id,
                    "message_idempotency_key": message_key,
                    "original_trace_id": cached.trace_id,
                },
            )
            runtime.trace_recorder.record(trace_id, "final_output", {"reply": cached.final_reply, "reason": "message_deduplicated"})
            return cached

        runtime.store.append_user_turn(message, trace_id)
        runtime.trace_recorder.record(trace_id, "user_input", {"message": message.to_dict(), "route": "casual_chat"})
        context_payload = build_wechaty_casual_chat_payload(message, runtime, gate_decision=gate_decision)
        messages = [
            {"role": "system", "content": WECHATY_CASUAL_CHAT_PROMPT_PATH.read_text(encoding="utf-8")},
            {"role": "user", "content": json.dumps(context_payload, ensure_ascii=False, sort_keys=True)},
        ]
        runtime.trace_recorder.record(trace_id, "wechaty_casual_chat_prompt", {"messages": messages})
        turn_budget = TokenBudget(
            max_tokens_per_call=env_int(
                "MAHJONG_WECHATY_CASUAL_CHAT_MAX_TOKENS_PER_CALL",
                env_int("MAHJONG_AGENT_MAX_TOKENS_PER_CALL", env_int("MAHJONG_LLM_MAX_TOKENS_PER_CALL", 32_000)),
            ),
            max_calls_per_turn=env_int("MAHJONG_WECHATY_CASUAL_CHAT_MAX_CALLS_PER_TURN", 3),
        )
        budget = turn_budget.reserve(messages)
        runtime.trace_recorder.record(trace_id, "wechaty_casual_chat_budget_checked", budget.to_dict())
        actions: list[AgentAction] = []
        tool_results: list[ToolResult] = []
        final_reply = ""
        if not budget.allowed:
            runtime.trace_recorder.record(
                trace_id,
                "final_output",
                {"reply": "", "reason": "casual_chat_budget_exhausted", "budget_reason": budget.reason},
                level="WARN",
            )
            result = AgentRuntimeResult(trace_id=trace_id, conversation_id=message.conversation_id, final_reply="")
            runtime.store.remember_message_result(message_key, result)
            return result

        started = time.perf_counter()
        try:
            raw_response = runtime.llm_client.complete(
                messages,
                trace_id=trace_id,
                timeout_seconds=env_float("MAHJONG_WECHATY_CASUAL_CHAT_TIMEOUT_SECONDS", 12.0),
            )
        except Exception as exc:
            runtime.trace_recorder.record(
                trace_id,
                "wechaty_casual_chat_error",
                {"error_type": type(exc).__name__, "error": str(exc), "elapsed_ms": int((time.perf_counter() - started) * 1000)},
                level="ERROR",
            )
            runtime.trace_recorder.record(trace_id, "final_output", {"reply": "", "reason": "casual_chat_llm_error"}, level="WARN")
            result = AgentRuntimeResult(trace_id=trace_id, conversation_id=message.conversation_id, final_reply="")
            runtime.store.remember_message_result(message_key, result)
            return result

        runtime.trace_recorder.record(
            trace_id,
            "wechaty_casual_chat_response",
            {"content": raw_response, "elapsed_ms": int((time.perf_counter() - started) * 1000)},
        )
        reply_payload, errors = parse_wechaty_casual_chat_response(raw_response)
        runtime.trace_recorder.record(
            trace_id,
            "wechaty_casual_chat_decision",
            {"decision": reply_payload, "errors": errors},
            level="WARN" if errors else "INFO",
        )
        if errors or not reply_payload["should_reply"] or reply_payload["needs_human"]:
            runtime.trace_recorder.record(
                trace_id,
                "final_output",
                {
                    "reply": "",
                    "reason": "casual_chat_no_safe_reply",
                    "errors": errors,
                    "needs_human": reply_payload["needs_human"],
                },
                level="WARN" if errors or reply_payload["needs_human"] else "INFO",
            )
            result = AgentRuntimeResult(trace_id=trace_id, conversation_id=message.conversation_id, final_reply="")
            runtime.store.remember_message_result(message_key, result)
            return result

        action = AgentAction(
            goal="处理微信闲聊，不进入麻将运营主流程",
            objective_status="completed",
            reasoning_summary=reply_payload["reasoning_summary"],
            reply_to_user=reply_payload["reply_to_user"],
            tool_calls=[],
            needs_human=False,
            stop_reason={
                "can_stop": True,
                "why": "闲聊回复生成后只需完成客户可见内容审查。",
                "pending_work": [],
                "depends_on_tool_results": False,
            },
        )
        actions.append(action)
        review_items = [
            {
                "item_id": "casual_chat.reply_to_user",
                "source": "casual_chat_reply",
                "recipient_id": message.sender_id,
                "recipient_name": message.sender_name,
                "text": reply_payload["reply_to_user"],
            }
        ]
        review_result = runtime._run_customer_visible_content_review(
            message=message,
            trace_id=trace_id,
            action=action,
            review_items=review_items,
            context_payload=context_payload,
            turn_budget=TokenBudget(
                max_tokens_per_call=env_int("MAHJONG_AGENT_REVIEW_MAX_TOKENS_PER_CALL", turn_budget.max_tokens_per_call),
                max_calls_per_turn=env_int("MAHJONG_AGENT_REVIEW_MAX_CALLS_PER_TURN", 8),
            ),
            review_scope="casual_chat_reply",
        )
        if review_result is not None:
            tool_results.append(review_result)
            runtime.trace_recorder.record(trace_id, "tool_result", review_result.to_dict())
            runtime.store.append_tool_turn(message.conversation_id, json.dumps([review_result.to_dict()], ensure_ascii=False), trace_id)
        if review_result is None:
            runtime.trace_recorder.record(
                trace_id,
                "final_output",
                {"reply": "", "reason": "casual_chat_review_not_available"},
                level="WARN",
            )
        elif bool(review_result.result.get("approved")) and review_result.error is None:
            final_reply = reply_payload["reply_to_user"]
            action.reply_to_user = final_reply
            runtime.store.append_assistant_turn(
                message.conversation_id,
                final_reply,
                trace_id,
                metadata={
                    "delivery_status": "pending_operator_send",
                    "message_type": "casual_chat",
                    "input_gate_category": str(gate_decision.get("category") or ""),
                },
            )
            runtime.trace_recorder.record(
                trace_id,
                "final_output",
                {"reply": final_reply, "objective_status": "casual_chat_completed"},
            )
        else:
            safe_text = review_result_safe_text(review_result, item_id="casual_chat.reply_to_user") if review_result else ""
            if safe_text:
                final_reply = safe_text
                action.reply_to_user = final_reply
                runtime.store.append_assistant_turn(
                    message.conversation_id,
                    final_reply,
                    trace_id,
                    metadata={
                        "delivery_status": "pending_operator_send",
                        "message_type": "casual_chat",
                        "input_gate_category": str(gate_decision.get("category") or ""),
                        "rewritten_by_review": True,
                    },
                )
                runtime.trace_recorder.record(
                    trace_id,
                    "final_output",
                    {"reply": final_reply, "objective_status": "casual_chat_review_rewritten"},
                    level="WARN",
                )
            else:
                runtime.trace_recorder.record(
                    trace_id,
                    "final_output",
                    {"reply": "", "reason": "casual_chat_review_rejected"},
                    level="WARN",
                )
        result = AgentRuntimeResult(
            trace_id=trace_id,
            conversation_id=message.conversation_id,
            final_reply=final_reply,
            actions=actions,
            tool_results=tool_results,
        )
        runtime.store.remember_message_result(message_key, result)
        return result


def build_wechaty_user_message(payload: dict) -> tuple[UserMessage | None, dict]:
    text, text_source = text_from_wechaty_payload(payload)
    display_quote = parse_wechat_display_quote_text(text)
    display_quoted_message = None
    if display_quote:
        text, display_quoted_message = display_quote
    message_metadata = build_wechaty_message_metadata(payload, text=text, text_source=text_source)
    self_message = bool(payload.get("self_message"))
    route_scope = os.getenv("MAHJONG_WECHATY_ROUTE_SCOPE", "self_only").strip().lower()
    if route_scope not in {"self_only", "incoming_only", "all"}:
        route_scope = "self_only"
    whitelist_hits = wechaty_agent_whitelist_hits(payload)
    whitelisted = bool(whitelist_hits)
    audit = {
        "channel": "wechaty",
        "routed_to_agent": False,
        "reason": "",
        "message_id": str(payload.get("message_id") or payload.get("source_message_id") or ""),
        "conversation_id": str(payload.get("conversation_id") or ""),
        "sender_id": str(payload.get("sender_id") or ""),
        "is_room": bool(payload.get("is_room")),
        "self_message": self_message,
        "message_type": payload.get("message_type"),
        "modalities": list(message_metadata.get("modalities") or []),
        "text_source": message_metadata.get("text_source"),
        "media_requires_transcription": bool(message_metadata.get("media_requires_transcription")),
        "media_requires_ocr": bool(message_metadata.get("media_requires_ocr")),
        "raw_observation_summary": dict(message_metadata.get("raw_observation_summary") or {}),
        "route_scope": route_scope,
        "agent_whitelisted": whitelisted,
        "agent_whitelist_hits": whitelist_hits,
    }
    if not env_bool("MAHJONG_WECHATY_AUTO_ROUTE_TO_AGENT", True):
        audit["reason"] = "auto_route_disabled"
        return None, audit
    if not text:
        audit["reason"] = "non_text_without_transcript_or_ocr" if message_metadata.get("modalities") else "empty_text"
        audit["metadata"] = message_metadata
        return None, audit
    if route_scope == "self_only" and not self_message and not whitelisted:
        audit["reason"] = "non_self_message_in_self_only_scope"
        return None, audit
    if route_scope == "incoming_only" and self_message:
        audit["reason"] = "self_message_in_incoming_only_scope"
        return None, audit

    raw_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
    room = payload.get("room") if isinstance(payload.get("room"), dict) else {}
    talker = payload.get("talker") if isinstance(payload.get("talker"), dict) else {}
    conversation_id = str(payload.get("conversation_id") or "").strip()
    sender_id = str(payload.get("sender_id") or raw_payload.get("talkerId") or "").strip()
    message_id = str(payload.get("message_id") or payload.get("source_message_id") or raw_payload.get("id") or "").strip()
    if not conversation_id:
        room_id = str(room.get("id") or raw_payload.get("roomId") or "").strip()
        conversation_id = f"wechaty:room:{room_id}" if room_id else f"wechaty:contact:{sender_id}"
    if not conversation_id or conversation_id in {"wechaty:contact:", "wechaty:room:"}:
        audit["reason"] = "missing_conversation_id"
        return None, audit
    if not sender_id:
        audit["reason"] = "missing_sender_id"
        return None, audit
    if not message_id:
        audit["reason"] = "missing_message_id"
        return None, audit

    sender_name = (
        str(payload.get("sender_name") or "").strip()
        or _wechaty_nested_text(talker, "alias")
        or _wechaty_nested_text(talker, "name")
        or _wechaty_nested_text(talker, "payload", "alias")
        or _wechaty_nested_text(talker, "payload", "name")
        or sender_id
    )
    message = UserMessage(
        conversation_id=conversation_id,
        sender_id=sender_id,
        sender_name=sender_name,
        text=text,
        message_id=message_id,
        quoted_message=parse_quoted_message_ref(payload) or display_quoted_message,
        metadata=message_metadata,
    )
    audit.update(
        {
            "routed_to_agent": True,
            "reason": "valid_text_message",
            "message_id": message_id,
            "conversation_id": conversation_id,
            "sender_id": sender_id,
            "sender_name": sender_name,
            "text": text,
            "metadata": message_metadata,
            "quoted_message": message.quoted_message.to_dict() if message.quoted_message else None,
        }
    )
    return message, audit


def input_quiet_period_seconds() -> float:
    """Return the inactivity window used to close one fragmented utterance."""

    return max(0.1, env_float("MAHJONG_INPUT_QUIET_PERIOD_SECONDS", 30.0))


def input_aggregation_enabled(channel: str) -> bool:
    """Allow channels to adopt aggregation independently during rollout."""

    if channel == "wechaty":
        return env_bool("MAHJONG_WECHATY_INPUT_AGGREGATION_ENABLED", True)
    return env_bool("MAHJONG_API_INPUT_AGGREGATION_ENABLED", False)


def buffer_input_fragment(
    runtime: AgentRuntime,
    message: UserMessage,
    *,
    trace_id: str,
) -> tuple[PendingInputBatch, bool]:
    """Persist one fragment and reset the durable quiet-period deadline."""

    deadline = datetime.now().astimezone() + timedelta(seconds=input_quiet_period_seconds())
    batch, transition, added = runtime.store.upsert_pending_input_fragment(
        message,
        trace_id=trace_id,
        quiet_deadline=deadline,
    )
    runtime.trace_recorder.record(
        trace_id,
        "input_fragment_buffered" if added else "input_fragment_deduplicated",
        {
            "batch": batch.to_dict(),
            "transition": transition.to_dict() if transition else None,
            "source_message_id": message.message_id,
            "quiet_period_seconds": input_quiet_period_seconds(),
        },
    )
    return batch, added


def dispatch_pending_input_batch(
    runtime: AgentRuntime,
    batch: PendingInputBatch,
    *,
    trace_id: str,
    quiet_period_elapsed: bool,
    trigger: str,
    audit: dict | None = None,
) -> dict:
    """Let the model close or extend an input window, then enter one real flow.

    The model owns the semantic decision. The backend only persists the batch,
    compare-and-set claims its exact version, and records the terminal status.
    """

    route_audit = dict(audit or {})
    aggregate = aggregate_pending_input_batch(
        batch,
        quiet_period_elapsed=quiet_period_elapsed,
        trigger=trigger,
    )
    gate_decision = run_wechaty_input_gate(
        aggregate,
        trace_id=trace_id,
        runtime=runtime,
        input_batch=batch,
        quiet_period_elapsed=quiet_period_elapsed,
    )
    runtime.store.record_pending_input_decision(
        batch_id=batch.batch_id,
        expected_version=batch.version,
        decision=gate_decision,
    )
    route_audit["input_gate"] = gate_decision
    route_audit["input_batch"] = batch.to_dict()
    route_audit["quiet_period_elapsed"] = quiet_period_elapsed

    if gate_decision.get("action") == "wait_for_more_input" and not quiet_period_elapsed:
        route_audit.update(
            {
                "routed_to_agent": False,
                "reason": "model_waiting_for_more_input",
                "waiting_for_more_input": True,
            }
        )
        runtime.trace_recorder.record(trace_id, "input_batch_waiting", route_audit)
        return {
            "routed_to_agent": False,
            "waiting_for_more_input": True,
            "input_status": PendingInputBatchStatus.PENDING.value,
            "input_batch": batch.to_dict(),
            "audit": route_audit,
            "agent_result": None,
        }

    claimed, claim_transition = runtime.store.claim_pending_input_batch(
        batch_id=batch.batch_id,
        expected_version=batch.version,
        trace_id=trace_id,
    )
    if claimed is None:
        route_audit.update(
            {
                "routed_to_agent": False,
                "reason": "input_batch_version_superseded_before_dispatch",
                "waiting_for_more_input": True,
            }
        )
        runtime.trace_recorder.record(trace_id, "input_batch_dispatch_superseded", route_audit, level="WARN")
        return {
            "routed_to_agent": False,
            "waiting_for_more_input": True,
            "input_status": "superseded",
            "input_batch": batch.to_dict(),
            "audit": route_audit,
            "agent_result": None,
        }

    runtime.trace_recorder.record(
        trace_id,
        "input_batch_claimed",
        {
            "batch": claimed.to_dict(),
            "transition": claim_transition.to_dict() if claim_transition else None,
            "trigger": trigger,
        },
    )
    action = str(gate_decision.get("action") or "ignore")
    result: AgentRuntimeResult | None = None
    terminal_status = PendingInputBatchStatus.IGNORED
    try:
        if action == "process_business":
            route_audit.update({"routed_to_agent": True, "reason": "input_batch_ready_for_business"})
            runtime.trace_recorder.record(trace_id, "input_batch_routed_to_agent", route_audit)
            result = runtime.handle_user_message(aggregate, trace_id=trace_id)
            terminal_status = PendingInputBatchStatus.COMPLETED
        elif action == "process_casual" and env_bool("MAHJONG_WECHATY_CASUAL_CHAT_REPLY_ENABLED", True):
            route_audit.update({"routed_to_agent": False, "reason": "wechaty_input_gate_routed_to_casual_chat"})
            runtime.trace_recorder.record(trace_id, "input_batch_routed_to_casual_chat", route_audit)
            if route_audit.get("channel") == "wechaty":
                runtime.trace_recorder.record(trace_id, "wechaty_raw_message_routed_to_casual_chat", route_audit)
            result = handle_wechaty_casual_chat(
                aggregate,
                trace_id=trace_id,
                runtime=runtime,
                gate_decision=gate_decision,
            )
            terminal_status = PendingInputBatchStatus.COMPLETED
        else:
            route_audit.update({"routed_to_agent": False, "reason": "input_batch_ignored"})
            runtime.trace_recorder.record(trace_id, "input_batch_ignored", route_audit)

        finished, finish_transition = runtime.store.finish_pending_input_batch(
            batch_id=claimed.batch_id,
            expected_version=claimed.version,
            status=terminal_status,
            trace_id=trace_id,
            decision=gate_decision,
        )
        if finished is None:
            runtime.trace_recorder.record(
                trace_id,
                "input_batch_result_superseded",
                {
                    "batch_id": claimed.batch_id,
                    "batch_version": claimed.version,
                    "generated_reply_suppressed": bool(result and result.final_reply),
                },
                level="WARN",
            )
            result_payload = result.to_dict() if result else None
            if result_payload is not None:
                result_payload["final_reply"] = ""
            return {
                "routed_to_agent": action == "process_business",
                "waiting_for_more_input": True,
                "input_status": "superseded",
                "input_batch": claimed.to_dict(),
                "audit": route_audit,
                "agent_result": result_payload,
            }
        runtime.trace_recorder.record(
            trace_id,
            "input_batch_finished",
            {
                "batch": finished.to_dict(),
                "transition": finish_transition.to_dict() if finish_transition else None,
            },
        )
        result_payload = result.to_dict() if result else None
        response = {
            "routed_to_agent": action == "process_business",
            "waiting_for_more_input": False,
            "input_status": finished.status.value,
            "input_batch": finished.to_dict(),
            "audit": route_audit,
            "agent_result": result_payload,
        }
        if action == "process_casual" and result_payload is not None:
            response["casual_chat_result"] = result_payload
        return response
    except Exception as exc:
        runtime.store.finish_pending_input_batch(
            batch_id=claimed.batch_id,
            expected_version=claimed.version,
            status=PendingInputBatchStatus.FAILED,
            trace_id=trace_id,
            decision={**gate_decision, "error_type": type(exc).__name__, "error": str(exc)},
        )
        runtime.trace_recorder.record(
            trace_id,
            "input_batch_dispatch_failed",
            {"batch_id": claimed.batch_id, "error_type": type(exc).__name__, "error": str(exc)},
            level="ERROR",
        )
        raise


def route_user_message_with_aggregation(
    runtime: AgentRuntime,
    message: UserMessage,
    *,
    trace_id: str,
    channel: str,
    audit: dict | None = None,
) -> dict:
    """Buffer a fragment and dispatch only when the model closes the window."""

    batch, added = buffer_input_fragment(runtime, message, trace_id=trace_id)
    if not added:
        return {
            "routed_to_agent": False,
            "waiting_for_more_input": batch.status == PendingInputBatchStatus.PENDING,
            "input_status": "duplicate",
            "input_batch": batch.to_dict(),
            "audit": {**dict(audit or {}), "reason": "duplicate_source_message"},
            "agent_result": None,
        }
    return dispatch_pending_input_batch(
        runtime,
        batch,
        trace_id=trace_id,
        quiet_period_elapsed=False,
        trigger=f"{channel}_message_arrived",
        audit=audit,
    )


def route_wechaty_raw_to_agent(payload: dict, *, trace_id: str) -> dict:
    message, audit = build_wechaty_user_message(payload)
    trace_recorder = JsonlTraceRecorder(TRACE_PATH)
    if message is None:
        trace_recorder.record(trace_id, "wechaty_raw_message_not_routed", audit)
        return {"routed_to_agent": False, "audit": audit, "agent_result": None}
    runtime = get_runtime()
    if input_aggregation_enabled("wechaty"):
        return route_user_message_with_aggregation(
            runtime,
            message,
            trace_id=trace_id,
            channel="wechaty",
            audit=audit,
        )
    gate_decision = run_wechaty_input_gate(message, trace_id=trace_id, runtime=runtime)
    audit["input_gate"] = gate_decision
    action = str(gate_decision.get("action") or "ignore")
    if action == "process_casual" and env_bool("MAHJONG_WECHATY_CASUAL_CHAT_REPLY_ENABLED", True):
        audit.update({"routed_to_agent": False, "reason": "wechaty_input_gate_routed_to_casual_chat"})
        runtime.trace_recorder.record(trace_id, "wechaty_raw_message_routed_to_casual_chat", audit)
        result = handle_wechaty_casual_chat(message, trace_id=trace_id, runtime=runtime, gate_decision=gate_decision)
        return {
            "routed_to_agent": False,
            "audit": audit,
            "agent_result": result.to_dict(),
            "casual_chat_result": result.to_dict(),
        }
    if action != "process_business":
        audit.update({"routed_to_agent": False, "reason": "wechaty_input_gate_not_routed"})
        runtime.trace_recorder.record(trace_id, "wechaty_raw_message_not_routed", audit)
        return {"routed_to_agent": False, "audit": audit, "agent_result": None}
    runtime.trace_recorder.record(trace_id, "wechaty_raw_message_routed_to_agent", audit)
    result = runtime.handle_user_message(message, trace_id=trace_id)
    return {"routed_to_agent": True, "audit": audit, "agent_result": result.to_dict()}


def request_local_json(path: str, *, payload: dict | None = None, timeout_seconds: float = 3.0) -> dict:
    """Call the local Wechaty bridge without introducing a third-party client."""

    base_url = os.getenv("MAHJONG_WECHATY_OUTBOUND_BASE_URL", "http://127.0.0.1:8791").rstrip("/")
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = Request(
        f"{base_url}{path}",
        data=body,
        method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json; charset=utf-8"} if payload is not None else {},
    )
    with urlopen(request, timeout=max(0.1, timeout_seconds)) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def handle_invite_draft_action(runtime: AgentRuntime, payload: dict) -> dict:
    """Apply a human approval decision and deliver an approved invite once."""

    draft_id = str(payload.get("draft_id") or "").strip()
    action = str(payload.get("action") or "").strip().lower()
    trace_id = str(payload.get("trace_id") or f"trace_invite_{os.urandom(6).hex()}")
    if not draft_id:
        raise ValueError("draft_id is required")
    if action not in {"approve_send", "reject"}:
        raise ValueError("action must be approve_send or reject")
    draft = runtime.store.invite_drafts.get(draft_id)
    if draft is None:
        raise ValueError(f"invite draft not found: {draft_id}")
    if action == "reject":
        updated, transition = runtime.store.update_invite_delivery_status(
            draft_id=draft_id,
            status=InviteStatus.SUPERSEDED,
            trace_id=trace_id,
            reason="human_rejected_invite",
        )
        runtime.trace_recorder.record(trace_id, "invite_delivery_rejected", transition.to_dict())
        return {"ok": True, "sent": False, "draft": updated.to_dict(), "transition": transition.to_dict()}
    if draft.status == InviteStatus.SENT:
        return {"ok": True, "sent": True, "deduplicated": True, "draft": draft.to_dict()}
    metadata = draft.metadata if isinstance(draft.metadata, dict) else {}
    if metadata.get("content_review_approved") is not True:
        raise ValueError("invite draft has no backend-issued content review approval")
    health = request_local_json(
        "/health",
        timeout_seconds=env_float("MAHJONG_WECHATY_OUTBOUND_TIMEOUT_SECONDS", 3.0),
    )
    if not bool(health.get("send_channel_enabled")):
        raise RuntimeError("wechaty send channel is paused")
    response = request_local_json(
        "/send",
        payload={
            "to": draft.customer_id,
            "text": draft.message_text,
            "draft_id": draft.draft_id,
            "business_ref_type": "invite_draft",
            "business_ref_id": draft.draft_id,
            "source": "human_approved_invite",
            "source_trace_id": trace_id,
        },
        timeout_seconds=env_float("MAHJONG_WECHATY_OUTBOUND_TIMEOUT_SECONDS", 5.0),
    )
    if not bool(response.get("ok")):
        raise RuntimeError(f"wechaty invite delivery failed: {response}")
    updated, transition = runtime.store.update_invite_delivery_status(
        draft_id=draft_id,
        status=InviteStatus.SENT,
        trace_id=trace_id,
        reason="human_approved_invite_sent",
    )
    runtime.trace_recorder.record(
        trace_id,
        "invite_delivery_sent",
        {"draft": updated.to_dict(), "transition": transition.to_dict(), "bridge_response": response},
    )
    return {
        "ok": True,
        "sent": True,
        "deduplicated": bool(response.get("deduplicated")),
        "draft": updated.to_dict(),
        "transition": transition.to_dict(),
        "bridge_response": response,
    }


def deliver_delayed_wechaty_reply(
    runtime: AgentRuntime,
    batch: PendingInputBatch,
    route_result: dict,
    *,
    trace_id: str,
) -> dict:
    """Send a delayed result only when both Wechaty outbound switches allow it."""

    result = route_result.get("agent_result") if isinstance(route_result.get("agent_result"), dict) else {}
    reply = str(result.get("final_reply") or "").strip()
    if not reply:
        delivery = {"sent": False, "reason": "no_customer_visible_reply"}
        runtime.trace_recorder.record(trace_id, "delayed_reply_delivery_skipped", delivery)
        return delivery
    aggregate = aggregate_pending_input_batch(batch, quiet_period_elapsed=True, trigger="quiet_period_elapsed")
    metadata = aggregate.metadata if isinstance(aggregate.metadata, dict) else {}
    if batch.source_channel != "wechaty":
        delivery = {"sent": False, "reason": "source_channel_has_no_delayed_sender", "channel": batch.source_channel}
        runtime.trace_recorder.record(trace_id, "delayed_reply_delivery_skipped", delivery)
        return delivery
    if metadata.get("conversation_target_type") == "room":
        delivery = {"sent": False, "reason": "delayed_room_send_not_supported_by_current_bridge"}
        runtime.trace_recorder.record(trace_id, "delayed_reply_delivery_skipped", delivery, level="WARN")
        return delivery
    target = str(metadata.get("reply_target_id") or batch.sender_id).strip()
    try:
        health = request_local_json("/health", timeout_seconds=env_float("MAHJONG_WECHATY_OUTBOUND_TIMEOUT_SECONDS", 3.0))
        if not bool(health.get("send_channel_enabled")) or not bool(health.get("auto_send_reply")):
            delivery = {
                "sent": False,
                "reason": "wechaty_outbound_switch_disabled",
                "send_channel_enabled": bool(health.get("send_channel_enabled")),
                "auto_send_reply": bool(health.get("auto_send_reply")),
            }
            runtime.trace_recorder.record(trace_id, "delayed_reply_delivery_skipped", delivery)
            return delivery
        response = request_local_json(
            "/send",
            payload={
                "to": target,
                "text": reply,
                "source": "delayed_input_batch_reply",
                "source_trace_id": trace_id,
                "source_message_id": aggregate.message_id,
            },
            timeout_seconds=env_float("MAHJONG_WECHATY_OUTBOUND_TIMEOUT_SECONDS", 3.0),
        )
        delivery = {"sent": bool(response.get("ok")), "target": target, "response": response}
        runtime.trace_recorder.record(
            trace_id,
            "delayed_reply_delivered" if delivery["sent"] else "delayed_reply_delivery_failed",
            delivery,
            level="INFO" if delivery["sent"] else "ERROR",
        )
        return delivery
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        delivery = {
            "sent": False,
            "reason": "wechaty_outbound_request_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        runtime.trace_recorder.record(trace_id, "delayed_reply_delivery_failed", delivery, level="ERROR")
        return delivery


def handle_due_pending_input_batch(batch: PendingInputBatch, trace_id: str) -> None:
    """Scheduler callback: re-evaluate a quiet batch and deliver its final reply."""

    runtime = get_runtime()
    route_result = dispatch_pending_input_batch(
        runtime,
        batch,
        trace_id=trace_id,
        quiet_period_elapsed=True,
        trigger="quiet_period_elapsed",
        audit={
            "channel": batch.source_channel or "unknown",
            "conversation_id": batch.conversation_id,
            "sender_id": batch.sender_id,
        },
    )
    delivery = deliver_delayed_wechaty_reply(runtime, batch, route_result, trace_id=trace_id)
    runtime.trace_recorder.record(
        trace_id,
        "delayed_input_batch_completed",
        {"route_result": route_result, "delivery": delivery},
    )


def conversation_id_from_trace(runtime: AgentRuntime, trace_id: str) -> str:
    if not trace_id:
        return ""
    for event in runtime.trace_recorder.get_trace(trace_id):
        if event.step == "user_input":
            message = event.content.get("message")
            if isinstance(message, dict) and message.get("conversation_id"):
                return str(message["conversation_id"])
    return ""


def trace_facts(runtime: AgentRuntime, trace_id: str) -> dict:
    facts: dict[str, dict] = {"input": {}, "actual": {}}
    if not trace_id:
        return facts
    for event in runtime.trace_recorder.get_trace(trace_id):
        if event.step == "user_input":
            message = event.content.get("message")
            if isinstance(message, dict):
                facts["input"] = {"message": message}
        if event.step == "final_output":
            facts["actual"] = {"reply": event.content.get("reply", ""), "final_output": dict(event.content)}
    return facts


def build_manual_badcase_payload(runtime: AgentRuntime, payload: dict, *, source_trace_id: str) -> dict:
    facts = trace_facts(runtime, source_trace_id)
    expected = payload.get("expected") if isinstance(payload.get("expected"), dict) else {"note": str(payload.get("expected") or "")}
    return {
        "reason": str(payload.get("reason") or "人工标记回复不符合预期"),
        "input": payload.get("input") if isinstance(payload.get("input"), dict) else facts.get("input", {}),
        "actual": payload.get("actual") if isinstance(payload.get("actual"), dict) else facts.get("actual", {}),
        "expected": expected,
        "tags": list(dict.fromkeys([*(str(item) for item in payload.get("tags") or []), "agent_runtime", "manual_review"])),
        "source": "manual_operator",
        "metadata": {
            "source_trace_id": source_trace_id,
            "operator_note": str(payload.get("note") or ""),
            "source_trace_completeness": trace_payload(runtime, source_trace_id)["completeness"] if source_trace_id else {},
        },
    }


def record_manual_badcase(runtime: AgentRuntime, payload: dict) -> dict:
    source_trace_id = str(payload.get("trace_id") or payload.get("source_trace_id") or "").strip()
    audit_trace_id = str(payload.get("audit_trace_id") or f"trace_manual_badcase_{os.urandom(6).hex()}")
    conversation_id = str(payload.get("conversation_id") or conversation_id_from_trace(runtime, source_trace_id) or "manual_review")
    badcase_payload = build_manual_badcase_payload(runtime, payload, source_trace_id=source_trace_id)
    call = ToolCall(name="record_badcase", arguments=badcase_payload, reason="manual operator reported badcase")
    runtime.trace_recorder.record(
        audit_trace_id,
        "manual_badcase_input",
        {
            "source_trace_id": source_trace_id,
            "conversation_id": conversation_id,
            "payload": badcase_payload,
        },
    )
    runtime.trace_recorder.record(audit_trace_id, "tool_called", {"call": call.to_dict(), "step_index": 1})
    result = runtime.tool_gateway.execute(
        call,
        trace_id=audit_trace_id,
        conversation_id=conversation_id,
        sender_id=str(payload.get("operator_id") or "operator"),
        sender_name=str(payload.get("operator_name") or "老板/测试者"),
        step_index=1,
    )
    runtime.trace_recorder.record(audit_trace_id, "tool_result", result.to_dict())
    runtime.trace_recorder.record(
        audit_trace_id,
        "manual_badcase_recorded",
        {
            "source_trace_id": source_trace_id,
            "tool_result": result.to_dict(),
        },
        level="WARN" if result.error else "INFO",
    )
    return {
        "audit_trace_id": audit_trace_id,
        "source_trace_id": source_trace_id,
        "tool_result": result.to_dict(),
        "trace": trace_payload(runtime, audit_trace_id),
    }


def reset_runtime_state(runtime: AgentRuntime, payload: dict) -> dict:
    trace_id = str(payload.get("trace_id") or f"trace_operator_reset_{os.urandom(6).hex()}")
    include_customers = bool(payload.get("include_customers"))
    include_badcases = bool(payload.get("include_badcases"))
    runtime.trace_recorder.record(
        trace_id,
        "operator_reset_state_requested",
        {
            "include_customers": include_customers,
            "include_badcases": include_badcases,
            "reason": str(payload.get("reason") or "operator requested clearing runtime state and memory"),
        },
        level="WARN",
    )
    deleted = runtime.store.clear_runtime_state(
        include_customers=include_customers,
        include_badcases=include_badcases,
    )
    with runtime._conversation_locks_guard:
        runtime._conversation_locks.clear()
    runtime.trace_recorder.record(
        trace_id,
        "operator_reset_state_completed",
        {
            "deleted": deleted,
            "preserved": {
                "customers": not include_customers,
                "badcases": not include_badcases,
                "trace_log": True,
            },
        },
        level="WARN",
    )
    return {
        "trace_id": trace_id,
        "deleted": deleted,
        "preserved": {
            "customers": not include_customers,
            "badcases": not include_badcases,
            "trace_log": True,
        },
        "trace": trace_payload(runtime, trace_id),
    }


def seed_customers(store: SQLiteAgentStore) -> None:
    """Load optional private seed data from an ignored local JSON file."""

    path = Path(os.getenv("MAHJONG_CUSTOMER_SEED_PATH") or ROOT / "data" / "customer_seed.local.json")
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid customer seed file: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"customer seed root must be an object: {path}")
    profile_fields = {
        "customer_id",
        "display_name",
        "public_name",
        "private_remark",
        "gender",
        "preferred_games",
        "preferred_stakes",
        "preferred_time_tags",
        "profile_facts",
        "smoke_preference",
        "response_score",
        "fatigue_score",
        "no_contact",
        "notes",
    }
    for raw_profile in payload.get("profiles", []):
        if not isinstance(raw_profile, dict):
            continue
        store.upsert_customer(CustomerProfile(**{key: value for key, value in raw_profile.items() if key in profile_fields}))
    relationship_fields = {"customer_a_id", "customer_b_id", "played_together_count", "avoid_playing", "notes"}
    for raw_relationship in payload.get("relationships", []):
        if not isinstance(raw_relationship, dict):
            continue
        store.upsert_customer_relationship(
            CustomerRelationship(
                **{key: value for key, value in raw_relationship.items() if key in relationship_fields}
            )
        )


class AgentRuntimeHandler(BaseHTTPRequestHandler):
    server_version = "MahjongAgentRuntime/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._html(index_html(), set_auth_cookie=True)
            return
        if parsed.path == "/tests":
            self._html(test_observability_html(), set_auth_cookie=True)
            return
        if parsed.path == "/api/health":
            self._json({"ok": True, "runtime": "mahjong_agent_runtime"})
            return
        if not self._authorized():
            self._json({"error": "unauthorized"}, status=401)
            return
        if parsed.path == "/api/runtime":
            runtime = get_runtime()
            self._json(runtime_manifest(runtime))
            return
        if parsed.path == "/api/state":
            runtime = get_runtime()
            self._json(
                {
                    "games": [item.to_dict() for item in runtime.store.games.values()],
                    "invite_drafts": [item.to_dict() for item in runtime.store.invite_drafts.values()],
                    "outbound_message_drafts": [item.to_dict() for item in runtime.store.outbound_message_drafts.values()],
                    "conversation_checkpoints": [
                        item.to_dict() for item in runtime.store.conversation_checkpoints.values()
                    ],
                    "conversation_versions": dict(runtime.store.conversation_versions),
                    "pending_input_batches": [
                        item.to_dict() for item in runtime.store.pending_input_batches.values()
                    ],
                    "customers": [item.to_dict() for item in runtime.store.customers.values()],
                    "runtime_config": runtime_config(runtime),
                }
            )
            return
        if parsed.path == "/api/traces":
            runtime = get_runtime()
            trace_id = (parse_qs(parsed.query).get("trace_id") or [""])[0]
            self._json(trace_payload(runtime, trace_id))
            return
        if parsed.path == "/api/logs":
            limit = int((parse_qs(parsed.query).get("limit") or ["200"])[0] or "200")
            self._json({"runtime": "mahjong_agent_runtime", "trace_log_path": str(TRACE_PATH), "tail": tail_trace_log(limit)})
            return
        if parsed.path == "/api/channels/hermes/raw":
            limit = int((parse_qs(parsed.query).get("limit") or ["100"])[0] or "100")
            self._json(
                {
                    "runtime": "mahjong_agent_runtime",
                    "raw_log_path": str(HERMES_RAW_LOG_PATH),
                    "records": tail_jsonl(HERMES_RAW_LOG_PATH, limit),
                }
            )
            return
        if parsed.path == "/api/channels/astrbot/raw":
            limit = int((parse_qs(parsed.query).get("limit") or ["100"])[0] or "100")
            self._json(
                {
                    "runtime": "mahjong_agent_runtime",
                    "raw_log_path": str(ASTRBOT_RAW_LOG_PATH),
                    "records": tail_jsonl(ASTRBOT_RAW_LOG_PATH, limit),
                }
            )
            return
        if parsed.path == "/api/channels/wechaty/raw":
            limit = int((parse_qs(parsed.query).get("limit") or ["100"])[0] or "100")
            self._json(
                {
                    "runtime": "mahjong_agent_runtime",
                    "raw_log_path": str(WECHATY_RAW_LOG_PATH),
                    "records": tail_jsonl(WECHATY_RAW_LOG_PATH, limit),
                }
            )
            return
        if parsed.path == "/api/badcases":
            runtime = get_runtime()
            self._json({"records": list(runtime.store.badcases)})
            return
        if parsed.path == "/api/test-observability":
            self._json(observability_payload())
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if not self._authorized():
            self._json({"error": "unauthorized"}, status=401)
            return
        client_key = self.client_address[0] if self.client_address else "unknown"
        if not REQUEST_RATE_LIMITER.allow(client_key):
            self._json({"error": "rate_limit_exceeded"}, status=429)
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            self._json({"error": "invalid_content_length"}, status=400)
            return
        if length < 0 or length > MAX_REQUEST_BYTES:
            self._json(
                {"error": "request_too_large", "max_request_bytes": MAX_REQUEST_BYTES},
                status=413,
            )
            return
        if not REQUEST_SEMAPHORE.acquire(blocking=False):
            self._json({"error": "server_busy"}, status=503)
            return
        try:
            self._dispatch_POST()
        except json.JSONDecodeError:
            self._json({"error": "invalid_json"}, status=400)
        finally:
            REQUEST_SEMAPHORE.release()

    def _dispatch_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/message":
            runtime = get_runtime()
            payload = self._read_json()
            message, missing_fields = build_api_user_message(payload)
            if message is None:
                self._json(
                    {
                        "error": "missing_required_fields",
                        "missing_fields": missing_fields,
                        "message": "conversation_id、sender_id、sender_name 和 text 必须由调用方明确提供。",
                    },
                    status=400,
                )
                return
            if message.message_id is None:
                idempotency_key = self.headers.get("Idempotency-Key", "").strip()
                if not idempotency_key:
                    self._json(
                        {
                            "error": "message_id_required",
                            "message": "message_id 或 Idempotency-Key 请求头至少提供一个。",
                        },
                        status=400,
                    )
                    return
                message = UserMessage(
                    conversation_id=message.conversation_id,
                    sender_id=message.sender_id,
                    sender_name=message.sender_name,
                    text=message.text,
                    message_id=f"api_{hashlib.sha256(idempotency_key.encode('utf-8')).hexdigest()[:24]}",
                    quoted_message=message.quoted_message,
                    metadata=message.metadata,
                )
            aggregate_fragments = bool(payload.get("aggregate_fragments")) or input_aggregation_enabled("api")
            if aggregate_fragments:
                trace_id = str(payload.get("trace_id") or f"trace_api_{os.urandom(6).hex()}")
                route_result = route_user_message_with_aggregation(
                    runtime,
                    message,
                    trace_id=trace_id,
                    channel="api",
                    audit={
                        "channel": "api",
                        "conversation_id": message.conversation_id,
                        "sender_id": message.sender_id,
                    },
                )
                agent_result = route_result.get("agent_result")
                if isinstance(agent_result, dict):
                    self._json(
                        {
                            **agent_result,
                            "input_status": route_result.get("input_status"),
                            "waiting_for_more_input": bool(route_result.get("waiting_for_more_input")),
                            "input_batch": route_result.get("input_batch"),
                        }
                    )
                else:
                    self._json(
                        {
                            "trace_id": trace_id,
                            "conversation_id": message.conversation_id,
                            "final_reply": "",
                            "actions": [],
                            "tool_results": [],
                            "state_transitions": [],
                            "input_status": route_result.get("input_status"),
                            "waiting_for_more_input": bool(route_result.get("waiting_for_more_input")),
                            "input_batch": route_result.get("input_batch"),
                        }
                    )
                return
            result = runtime.handle_user_message(message, trace_id=payload.get("trace_id"))
            self._json(result.to_dict())
            return
        if parsed.path == "/api/message-references/link":
            runtime = get_runtime()
            payload = self._read_json()
            self._json(link_delivered_message_reference(runtime, payload))
            return
        if parsed.path == "/api/channels/hermes/raw":
            payload = self._read_json()
            record = record_hermes_raw_message(payload)
            self._json(
                {
                    "ok": True,
                    "trace_id": record["trace_id"],
                    "raw_log_path": str(HERMES_RAW_LOG_PATH),
                    "record": record,
                }
            )
            return
        if parsed.path == "/api/channels/astrbot/raw":
            payload = self._read_json()
            record = record_astrbot_raw_message(payload)
            self._json(
                {
                    "ok": True,
                    "trace_id": record["trace_id"],
                    "raw_log_path": str(ASTRBOT_RAW_LOG_PATH),
                    "record": record,
                }
            )
            return
        if parsed.path == "/api/channels/wechaty/raw":
            payload = self._read_json()
            record = record_wechaty_raw_message(payload)
            route_result = route_wechaty_raw_to_agent(payload, trace_id=record["trace_id"])
            self._json(
                {
                    "ok": True,
                    "trace_id": record["trace_id"],
                    "raw_log_path": str(WECHATY_RAW_LOG_PATH),
                    "record": record,
                    "route_result": route_result,
                }
            )
            return
        if parsed.path == "/api/invite-drafts/action":
            runtime = get_runtime()
            payload = self._read_json()
            try:
                self._json(handle_invite_draft_action(runtime, payload))
            except ValueError as exc:
                self._json({"ok": False, "error": str(exc)}, status=400)
            except (RuntimeError, HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                self._json(
                    {"ok": False, "error_type": type(exc).__name__, "error": str(exc)},
                    status=503,
                )
            return
        if parsed.path == "/api/badcases":
            runtime = get_runtime()
            payload = self._read_json()
            self._json(record_manual_badcase(runtime, payload))
            return
        if parsed.path == "/api/reset-state":
            runtime = get_runtime()
            payload = self._read_json()
            self._json(reset_runtime_state(runtime, payload))
            return
        if parsed.path == "/api/test-observability/run":
            payload = self._read_json()
            suite = str(payload.get("suite") or "").strip()
            try:
                result = run_fixed_suite(suite)
            except ValueError as exc:
                self._json({"ok": False, "error": str(exc)}, status=400)
                return
            status = 200 if result["ok"] else (409 if result.get("return_code") == 409 else 500)
            self._json(result, status=status)
            return
        self.send_error(404)

    def log_message(self, fmt: str, *args) -> None:
        print(fmt % args)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _authorized(self) -> bool:
        expected = runtime_api_token()
        authorization = self.headers.get("Authorization", "")
        bearer = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
        header_token = self.headers.get("X-Mahjong-Agent-Token", "").strip()
        cookie_token = ""
        for item in self.headers.get("Cookie", "").split(";"):
            key, separator, value = item.strip().partition("=")
            if separator and key == "mahjong_agent_token":
                cookie_token = value
                break
        return any(
            token and hmac.compare_digest(token, expected)
            for token in (bearer, header_token, cookie_token)
        )

    def _json(self, payload: dict, *, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html: str, *, set_auth_cookie: bool = False) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        if set_auth_cookie:
            self.send_header(
                "Set-Cookie",
                f"mahjong_agent_token={runtime_api_token()}; HttpOnly; SameSite=Strict; Path=/",
            )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def runtime_config(runtime: AgentRuntime) -> dict:
    llm_config = getattr(getattr(runtime, "llm_client", None), "config", None)
    return {
        "runtime": "mahjong_agent_runtime",
        "implementation_package": "mahjong_agent_runtime",
        "llm": {
            "provider": getattr(llm_config, "provider", ""),
            "model": getattr(llm_config, "model", ""),
            "base_url": getattr(llm_config, "base_url", ""),
            "max_completion_tokens": getattr(llm_config, "max_tokens", None),
        },
        "max_steps": runtime.max_steps,
        "progress_monitor": {
            "repeated_observation_limit": runtime.repeated_observation_limit,
            "consecutive_no_progress_limit": runtime.consecutive_no_progress_limit,
            "max_replan_attempts": runtime.max_progress_replans,
            "max_cycle_period": runtime.max_cycle_period,
        },
        "max_tokens_per_call": runtime.token_budget.max_tokens_per_call,
        "max_calls_per_turn": runtime.token_budget.max_calls_per_turn,
        "customer_visible_text_generation_enabled": runtime.customer_visible_text_generation_enabled,
        "customer_visible_text_generation_model": getattr(
            getattr(runtime.customer_visible_text_generation_client, "config", None), "model", None
        )
        or getattr(llm_config, "model", ""),
        "customer_visible_text_generation_max_tokens_per_call": runtime.customer_visible_text_generation_token_budget.max_tokens_per_call,
        "customer_visible_text_generation_max_calls_per_turn": runtime.customer_visible_text_generation_token_budget.max_calls_per_turn,
        "review_max_tokens_per_call": runtime.review_token_budget.max_tokens_per_call,
        "review_max_calls_per_turn": runtime.review_token_budget.max_calls_per_turn,
        "customer_visible_content_review_enabled": runtime.reply_self_review_enabled,
        "customer_visible_content_review_model": getattr(getattr(runtime.reply_self_review_client, "config", None), "model", None)
        or getattr(llm_config, "model", ""),
        "reply_self_review_enabled": runtime.reply_self_review_enabled,
        "reply_self_review_model": getattr(getattr(runtime.reply_self_review_client, "config", None), "model", None)
        or getattr(llm_config, "model", ""),
        "context_summary_enabled": runtime.context_summary_manager is not None,
        "context_summary_model": getattr(getattr(getattr(runtime.context_summary_manager, "llm_client", None), "config", None), "model", None)
        if runtime.context_summary_manager is not None
        else "",
        "context_summary_policy": asdict(runtime.context_summary_manager.policy)
        if runtime.context_summary_manager is not None
        else None,
        "wechaty_route_scope": os.getenv("MAHJONG_WECHATY_ROUTE_SCOPE", "self_only"),
        "wechaty_agent_whitelist_count": len(configured_wechaty_whitelist()),
        "wechaty_input_gate_enabled": env_bool("MAHJONG_WECHATY_INPUT_GATE_ENABLED", True),
        "wechaty_input_gate_model": os.getenv("MAHJONG_WECHATY_INPUT_GATE_LLM_MODEL")
        or getattr(getattr(runtime.llm_client, "config", None), "model", None),
        "wechaty_input_gate_fail_open": env_bool("MAHJONG_WECHATY_INPUT_GATE_FAIL_OPEN", False),
        "wechaty_input_gate_prompt": str(WECHATY_INPUT_GATE_PROMPT_PATH),
        "input_aggregation": {
            "wechaty_enabled": input_aggregation_enabled("wechaty"),
            "api_enabled_by_default": input_aggregation_enabled("api"),
            "quiet_period_seconds": input_quiet_period_seconds(),
            "scheduler_poll_seconds": env_float("MAHJONG_INPUT_SCHEDULER_POLL_SECONDS", 0.5),
            "batch_scope": "conversation_id + sender_id",
        },
        "wechaty_casual_chat_reply_enabled": env_bool("MAHJONG_WECHATY_CASUAL_CHAT_REPLY_ENABLED", True),
        "wechaty_casual_chat_reply_prompt": str(WECHATY_CASUAL_CHAT_PROMPT_PATH),
        "trace_log": str(TRACE_PATH),
        "hermes_raw_log": str(HERMES_RAW_LOG_PATH),
        "astrbot_raw_log": str(ASTRBOT_RAW_LOG_PATH),
        "wechaty_raw_log": str(WECHATY_RAW_LOG_PATH),
        "sqlite_db": str(DB_PATH),
    }


def default_base_url(provider: str) -> str:
    if provider == "deepseek":
        return "https://api.deepseek.com"
    if provider in {"zai", "glm", "bigmodel"}:
        return "https://api.z.ai/api/paas/v4"
    return "https://api.openai.com/v1"


def test_observability_html() -> str:
    return """
<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>麻将 Agent 测试与回放</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f6f8f6;color:#17211b}
main{max-width:1120px;margin:auto;padding:28px 24px 56px}
a{color:#176b52;text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:30px;margin:10px 0 6px}h2{font-size:21px;margin:30px 0 12px}h3{font-size:17px;margin:18px 0 8px}
p{line-height:1.65}.muted{color:#657269}.small{font-size:13px}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
button{font:inherit;border:1px solid #1f765c;background:#1f765c;color:white;border-radius:7px;padding:9px 14px;cursor:pointer}
button.secondary{background:white;color:#17211b;border-color:#b8c4bc}button.live{background:#8a4b08;border-color:#8a4b08}
button:disabled{opacity:.55;cursor:wait}.toolbar{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin:18px 0}
.summary{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));border:1px solid #d5ddd7;background:white;border-radius:8px;overflow:hidden}
.summary>div{padding:16px;border-right:1px solid #e1e7e2}.summary>div:last-child{border-right:0}.summary strong{display:block;font-size:24px;margin-top:5px}
.band{border-top:1px solid #d5ddd7;padding-top:6px;margin-top:26px}.method{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.panel{border:1px solid #d5ddd7;background:white;border-radius:8px;padding:16px}.panel p:first-child{margin-top:0}.panel p:last-child{margin-bottom:0}
.checks{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.check{border:1px solid #dde4df;background:white;border-radius:6px;padding:10px 12px}
.pass{color:#176b52}.fail{color:#b42318}.pill{display:inline-block;border-radius:999px;padding:3px 8px;background:#edf2ee;font-size:12px}
.timeline{border-left:2px solid #afc8bb;margin-left:7px;padding-left:18px}.timeline p{margin:8px 0}
details{border:1px solid #d5ddd7;background:white;border-radius:7px;margin:10px 0}summary{cursor:pointer;padding:12px 14px;font-weight:600}
pre{white-space:pre-wrap;overflow:auto;margin:0;border-top:1px solid #e1e7e2;padding:14px;font-size:12px;max-height:520px;background:#fbfcfb}
table{width:100%;border-collapse:collapse;background:white;border:1px solid #d5ddd7}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #e2e7e3;font-size:13px}th{background:#f0f4f1}
#runOutput{min-height:48px}.source{word-break:break-all}
@media(max-width:760px){.summary,.method,.checks{grid-template-columns:1fr}.summary>div{border-right:0;border-bottom:1px solid #e1e7e2}}
</style>
<main>
  <a href="/">返回运行控制台</a>
  <h1>测试与回放</h1>
  <p class="muted">这里展示测试数据怎么构造、候选人回复怎么模拟、模型调用了什么工具，以及数据库最终发生了什么变化。</p>
  <div class="toolbar">
    <button onclick="runSuite('deterministic',this)">重跑确定性并发测试</button>
    <button class="secondary" onclick="runSuite('focused_unit',this)">重跑聚焦单元测试</button>
    <button class="live" onclick="runLive(this)">调用真实 DeepSeek 回放</button>
    <button class="secondary" onclick="loadReports()">刷新报告</button>
    <span id="loadStatus" class="muted small"></span>
  </div>
  <div id="runOutput" class="panel small">尚未从页面触发测试。读取已有报告不产生模型费用。</div>

  <div class="summary" id="summary"></div>

  <section class="band">
    <h2>测试数据是怎么造的</h2>
    <div class="method">
      <div class="panel">
        <h3>确定性测试</h3>
        <p>先在临时 SQLite 中创建多个局，并把同一个客户以“暂定参与”放进这些局；再用多个线程同时调用生产工具 <span class="mono">record_candidate_reply</span>，模拟不同局的最后一位候选人同时确认。</p>
        <p>这里不调用模型，专门验证事务、并发、幂等和最终落库状态。判断通过靠数据库事实，不靠字符串猜测。</p>
        <p class="small source" id="deterministicSource"></p>
      </div>
      <div class="panel">
        <h3>真实模型测试</h3>
        <p>先创建现成局、参与者和最近对话，再把候选人的“也可以”构造成真实 <span class="mono">UserMessage</span> 送入主 Agent。DeepSeek 必须自己选择 <span class="mono">record_candidate_reply</span>，后端执行并把结果回喂模型。</p>
        <p>给候选人的邀约同样先生成 <span class="mono">invite_draft</span>，经过客户可见文本审查后落库。外发测试使用假的微信适配器记录调用，断言第一次只发送一次、重复审批被幂等去重；不会误扰真实用户。</p>
        <p class="small source" id="liveSource"></p>
      </div>
    </div>
  </section>

  <section class="band">
    <h2>并发共享候选人：数据库证据</h2>
    <p class="muted">观察同一个人同时候选多个局时，是否只有第一个成局方案保留他，其他局是否自动释放并重新计算缺口。</p>
    <div id="focusedScenario"></div>
  </section>

  <section class="band">
    <h2>真实 DeepSeek：决策回放</h2>
    <div id="liveReplay"></div>
  </section>

  <section class="band">
    <h2>全部测试套件</h2>
    <div id="allSuites"></div>
  </section>
</main>
<script>
const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const pretty = value => JSON.stringify(value, null, 2);
function reportPayload(data,key){return data?.reports?.[key]?.payload || null}
function statusClass(status){return status === 'passed' ? 'pass' : 'fail'}
function renderChecks(checks){
  return `<div class="checks">${(checks||[]).map(c => `<div class="check"><strong class="${c.passed?'pass':'fail'}">${c.passed?'通过':'失败'}</strong> ${esc(c.name)}<div class="small muted">期望：${esc(pretty(c.expected))}<br>实际：${esc(pretty(c.actual))}</div></div>`).join('')}</div>`;
}
function renderSummary(data){
  const det=reportPayload(data,'deterministic'); const live=reportPayload(data,'live_deepseek'); const unit=reportPayload(data,'focused_unit');
  const detCases=(det?.deterministic||[]); const detPassed=detCases.filter(x=>x.status==='passed').length;
  summary.innerHTML=`<div>确定性场景<strong class="${det?.status==='passed'?'pass':'fail'}">${detPassed}/${detCases.length}</strong></div><div>真实模型场景<strong class="${live?.status==='passed'?'pass':'fail'}">${live?.passed_count||0}/${live?.reports?.length||0}</strong></div><div>聚焦单测<strong class="${unit?.status==='passed'?'pass':'fail'}">${esc(unit?.status||'未运行')}</strong></div>`;
}
function renderFocused(data){
  const det=reportPayload(data,'deterministic');
  const scenario=(det?.deterministic||[]).find(x=>x.name==='shared_participant_first_ready_wins_race');
  if(!scenario){focusedScenario.innerHTML='<div class="panel">尚无报告，请点击“重跑确定性并发测试”。</div>';return}
  const evidence=scenario.metrics?.evidence||{}; const outcome=evidence.outcome||{}; const winner=outcome.winner||{}; const losers=outcome.losers||[];
  const shared=(winner.participants||[]).find(x=>x.customer_id==='shared_customer');
  focusedScenario.innerHTML=`
    <div class="timeline">
      <p><strong>1. 构造</strong> ${esc(evidence.fixture?.option_count)} 个互斥候选局，<span class="mono">shared_customer</span> 同时暂定参加全部局。</p>
      <p><strong>2. 并发动作</strong> ${esc(evidence.simulated_candidate_replies?.length)} 个候选人同时回复“确认”，每个动作都调用生产入口 <span class="mono">${esc(evidence.production_entrypoint)}</span>。</p>
      <p><strong>3. 胜出</strong> <span class="mono">${esc(winner.game_id)}</span> 状态为 <span class="pill">${esc(winner.status)}</span>，共享用户状态为 <span class="pill">${esc(shared?.status)}</span>。</p>
      <p><strong>4. 释放</strong> 其余 ${losers.length} 个局释放共享用户，释放记录 ${esc(outcome.released_shared_participation_count)} 条。</p>
    </div>
    ${renderChecks(scenario.checks)}
    <details><summary>查看胜出局完整状态</summary><pre>${esc(pretty(winner))}</pre></details>
    <details><summary>查看所有模拟候选人回复</summary><pre>${esc(pretty(evidence.simulated_candidate_replies))}</pre></details>
    <details><summary>查看失败局和状态迁移</summary><pre>${esc(pretty({losers, state_transitions:evidence.state_transitions}))}</pre></details>`;
}
function renderLive(data){
  const live=reportPayload(data,'live_deepseek'); const report=live?.reports?.[0];
  if(!report){liveReplay.innerHTML='<div class="panel">尚无真实模型报告。点击按钮会调用 DeepSeek，并产生少量模型费用。</div>';return}
  const firstAction=(report.decision_trace||[]).find(x=>x.step==='action_proposed')?.content||{};
  const toolResult=(report.tool_result_summaries||[]).find(x=>x.name==='record_candidate_reply')||{};
  liveReplay.innerHTML=`
    <table><tbody>
      <tr><th>候选人输入</th><td>${esc(report.input?.text)}</td></tr>
      <tr><th>模型选择</th><td>${esc((firstAction.tool_calls||[]).map(x=>x.name).join(', ')||report.tool_names?.join(', '))}</td></tr>
      <tr><th>工具参数</th><td><span class="mono">${esc(pretty(firstAction.tool_calls?.[0]?.arguments||{}))}</span></td></tr>
      <tr><th>落库结果</th><td>局状态 ${esc(toolResult.game?.status)}，已占 ${esc(toolResult.game?.seat_summary?.claimed_seats)} 席，剩余 ${esc(toolResult.game?.seat_summary?.remaining_seats)} 席</td></tr>
      <tr><th>最终回复</th><td><strong>${esc(report.final_reply)}</strong></td></tr>
      <tr><th>Trace ID</th><td class="mono">${esc(report.trace_id)}</td></tr>
    </tbody></table>
    ${renderChecks((report.checks||[]).filter((_,i)=>i<12))}
    <details><summary>查看模型决策、工具结果与审查证据</summary><pre>${esc(pretty({decision_trace:report.decision_trace,tool_results:report.tool_result_summaries,trace_steps:report.trace_steps}))}</pre></details>`;
}
function renderSuites(data){
  const det=reportPayload(data,'deterministic');
  const rows=(det?.deterministic||[]).map(x=>`<tr><td>${esc(x.name)}</td><td class="${statusClass(x.status)}">${esc(x.status)}</td><td>${esc(x.operation_count)}</td><td>${esc(x.elapsed_ms)} ms</td></tr>`).join('');
  allSuites.innerHTML=`<table><thead><tr><th>场景</th><th>状态</th><th>操作数</th><th>耗时</th></tr></thead><tbody>${rows||'<tr><td colspan="4">尚无报告</td></tr>'}</tbody></table><details><summary>查看报告文件与原始 JSON</summary><pre>${esc(pretty(data.reports))}</pre></details>`;
}
async function loadReports(){
  loadStatus.textContent='读取中';
  const res=await fetch('/api/test-observability'); const data=await res.json();
  renderSummary(data); renderFocused(data); renderLive(data); renderSuites(data);
  deterministicSource.textContent=`测试代码：${data.test_design.deterministic.fixture_source}`;
  liveSource.textContent=`测试代码：${data.test_design.live_deepseek.fixture_source}\nGolden dataset：${data.test_design.live_deepseek.golden_source}`;
  loadStatus.textContent=`最近刷新：${new Date().toLocaleTimeString()}`;
}
async function runSuite(suite,button){
  button.disabled=true; runOutput.textContent=`正在运行 ${suite}，请稍候...`;
  try{
    const res=await fetch('/api/test-observability/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({suite})});
    const body=await res.json(); runOutput.textContent=`${body.ok?'通过':'失败'}，耗时 ${body.elapsed_ms} ms\n${body.stdout_tail||''}\n${body.stderr_tail||''}`;
    await loadReports();
  }finally{button.disabled=false}
}
function runLive(button){
  if(!confirm('这会真实调用 DeepSeek API，并产生少量费用。继续吗？')) return;
  runSuite('live_deepseek',button);
}
loadReports();
</script>
"""


def index_html() -> str:
    return """
<!doctype html>
<meta charset="utf-8">
<title>Mahjong Agent Runtime</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:32px;background:#f8faf8;color:#1f2a24}
main{max-width:980px;margin:auto}
textarea,input,button{font:inherit}
input,textarea{width:100%;box-sizing:border-box;border:1px solid #b9c7bd;border-radius:8px;padding:12px;background:white}
textarea{min-height:140px}
button{border:1px solid #2f7d62;background:#2f7d62;color:white;border-radius:8px;padding:10px 16px;cursor:pointer}
button.secondary{border-color:#b9c7bd;background:white;color:#1f2a24}
button.danger{border-color:#b42318;background:#b42318;color:white}
pre{white-space:pre-wrap;background:white;border:1px solid #d6ded8;border-radius:8px;padding:16px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0}
.toolbar label{display:flex;gap:6px;align-items:center}
.toolbar input[type=checkbox]{width:auto}
.live{border:1px solid #d6ded8;background:#eef5f0;border-radius:8px;padding:12px;margin:16px 0}
.live-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.status{font-size:14px;color:#5d6c62}
.small{font-size:13px;max-height:320px;overflow:auto}
.draft-list{display:grid;gap:10px;margin:12px 0}
.draft-item{background:white;border:1px solid #d6ded8;border-radius:8px;padding:12px}
.draft-meta{font-size:13px;color:#5d6c62;margin-bottom:8px}
.draft-text{white-space:pre-wrap;margin-bottom:10px}
</style>
<main>
  <h1>Mahjong Agent Runtime</h1>
  <p>当前主链路：模型决定工具，后端只做合同、权限、幂等、状态和审计。</p>
  <p><a href="/tests">打开测试与回放页面</a></p>
  <div class="live">
    <div class="toolbar">
      <label><input id="autoRefresh" type="checkbox" checked onchange="toggleAutoRefresh()">实时刷新</label>
      <button class="secondary" onclick="refreshLive()">立即刷新</button>
      <span class="status" id="liveStatus">等待刷新</span>
    </div>
    <div class="live-grid">
      <div>
        <h2>最近 runtime 日志</h2>
        <pre class="small" id="runtimeLogs"></pre>
      </div>
      <div>
        <h2>最近 Wechaty 消息</h2>
        <pre class="small" id="wechatyRaw"></pre>
      </div>
    </div>
  </div>
  <div class="grid">
    <input id="conversationId" value="runtime_trial" placeholder="conversationId">
    <input id="senderId" value="zhang" placeholder="senderId">
  </div>
  <p><input id="senderName" value="张哥" placeholder="senderName"></p>
  <p><textarea id="text">通宵1块有人吗？没有就帮我组一个</textarea></p>
  <button onclick="sendMessage()">发送</button>
  <button onclick="loadState()">刷新状态</button>
  <button class="secondary" onclick="loadHermesRaw()">Hermes 原始消息</button>
  <button class="secondary" onclick="loadAstrBotRaw()">AstrBot 原始消息</button>
  <button class="secondary" onclick="loadWechatyRaw()">Wechaty 原始消息</button>
  <button onclick="recordBadcase()">标记 badcase</button>
  <button class="danger" onclick="resetState()">清空状态和记忆</button>
  <h2>结果</h2>
  <pre id="output"></pre>
  <h2>待审批邀约</h2>
  <div id="inviteDrafts" class="draft-list"></div>
  <h2>微信手动发送</h2>
  <p><input id="wechatTarget" value="xml31323" placeholder="微信号、备注名、昵称或联系人ID"></p>
  <p><textarea id="wechatText" placeholder="默认会填入上一轮建议回复"></textarea></p>
  <button onclick="sendWechatText()">发送给微信联系人</button>
  <button class="secondary" onclick="loadWechatChannelStatus()">微信通道状态</button>
  <button class="danger" onclick="setWechatSendChannel(false)">暂停微信发送总闸</button>
  <button class="secondary" onclick="setWechatSendChannel(true)">开启微信发送总闸</button>
  <button class="danger" onclick="setWechatAutoSend(false)">暂停自动外发</button>
  <button class="secondary" onclick="setWechatAutoSend(true)">开启自动外发</button>
  <pre id="wechatSendOutput"></pre>
  <h2>人工 badcase</h2>
  <p><input id="badcaseReason" value="回复不符合预期" placeholder="badcase 原因"></p>
  <p><textarea id="badcaseExpected" placeholder="期望行为或回复"></textarea></p>
  <pre id="badcaseOutput"></pre>
  <h2>Hermes 原始消息</h2>
  <pre id="hermesRaw"></pre>
  <h2>AstrBot 原始消息</h2>
  <pre id="astrbotRaw"></pre>
  <h2>状态</h2>
  <pre id="state"></pre>
</main>
<script>
let liveTimer = null;
let latestResult = null;
let pendingMessageAttempt = null;
const WECHATY_OUTBOUND_BASE = 'http://127.0.0.1:8791';

function createMessageId(){
  if(globalThis.crypto && typeof globalThis.crypto.randomUUID === 'function'){
    return `web_${globalThis.crypto.randomUUID()}`;
  }
  return `web_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}
async function sendMessage(){
  const requestSignature = JSON.stringify([
    conversationId.value,
    senderId.value,
    senderName.value,
    text.value
  ]);
  if(!pendingMessageAttempt || pendingMessageAttempt.signature !== requestSignature){
    pendingMessageAttempt = {signature: requestSignature, messageId: createMessageId()};
  }
  const payload = {
    conversation_id: conversationId.value,
    sender_id: senderId.value,
    sender_name: senderName.value,
    text: text.value,
    message_id: pendingMessageAttempt.messageId,
    aggregate_fragments: true
  };
  const res = await fetch('/api/message',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const body = await res.json();
  if(res.ok) pendingMessageAttempt = null;
  latestResult = body;
  window.lastTraceId = body.trace_id;
  output.textContent = JSON.stringify(body, null, 2);
  if(body.final_reply) wechatText.value = body.final_reply;
  await loadState();
  await refreshLive();
}
async function recordBadcase(){
  const payload = {
    source_trace_id: window.lastTraceId || '',
    reason: badcaseReason.value,
    expected: { note: badcaseExpected.value }
  };
  const res = await fetch('/api/badcases',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  badcaseOutput.textContent = JSON.stringify(await res.json(), null, 2);
  await loadState();
}
async function loadState(){
  const res = await fetch('/api/state');
  const data = await res.json();
  state.textContent = JSON.stringify(data, null, 2);
  renderInviteDrafts(data.invite_drafts || []);
}
function renderInviteDrafts(drafts){
  inviteDrafts.replaceChildren();
  const pending = drafts.filter(item => item.status === 'pending_approval');
  if(!pending.length){
    const empty = document.createElement('div');
    empty.className = 'status';
    empty.textContent = '暂无待审批邀约。';
    inviteDrafts.appendChild(empty);
    return;
  }
  for(const draft of pending){
    const item = document.createElement('div');
    item.className = 'draft-item';
    const meta = document.createElement('div');
    meta.className = 'draft-meta';
    meta.textContent = `${draft.display_name || draft.customer_id} · ${draft.game_id} · ${draft.metadata?.content_review_approved ? '已过对外内容审查' : '未过对外内容审查'}`;
    const message = document.createElement('div');
    message.className = 'draft-text';
    message.textContent = draft.message_text;
    const approve = document.createElement('button');
    approve.textContent = '通过并发送';
    approve.disabled = draft.metadata?.content_review_approved !== true;
    approve.onclick = () => decideInviteDraft(draft.draft_id, 'approve_send');
    const reject = document.createElement('button');
    reject.className = 'danger';
    reject.textContent = '拒绝';
    reject.onclick = () => decideInviteDraft(draft.draft_id, 'reject');
    item.append(meta, message, approve, document.createTextNode(' '), reject);
    inviteDrafts.appendChild(item);
  }
}
async function decideInviteDraft(draftId, action){
  const verb = action === 'approve_send' ? '发送这条邀约' : '拒绝这条邀约';
  if(!confirm(`确认${verb}？`)) return;
  const res = await fetch('/api/invite-drafts/action',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({draft_id:draftId, action})
  });
  const data = await res.json();
  wechatSendOutput.textContent = JSON.stringify(data, null, 2);
  await loadState();
}
async function loadRuntimeLogs(){
  const res = await fetch('/api/logs?limit=30');
  const data = await res.json();
  runtimeLogs.textContent = (data.tail || []).map(compactTraceLine).join('\\n');
}
function compactTraceLine(line){
  const text = String(line || '');
  const match = text.match(/^(.*?)-(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2})-(\\w+): (.*)$/);
  if(!match) return text.slice(0, 900);
  let payload = match[4];
  try{
    const json = JSON.parse(payload);
    const step = json.step || json.direction || json.path || '-';
    const reply = json.reply || json.final_reply || json.suggested_safe_text || '';
    const reason = json.reason || json.reasoning_summary || json.error || '';
    const trace = match[1];
    return `${match[2]} ${match[3]} ${trace} ${step}${reply ? ' | reply=' + reply : ''}${reason ? ' | ' + reason : ''}`;
  }catch{
    return text.slice(0, 900);
  }
}
async function loadHermesRaw(){
  const res = await fetch('/api/channels/hermes/raw?limit=50');
  hermesRaw.textContent = JSON.stringify(await res.json(), null, 2);
}
async function loadAstrBotRaw(){
  const res = await fetch('/api/channels/astrbot/raw?limit=50');
  astrbotRaw.textContent = JSON.stringify(await res.json(), null, 2);
}
async function loadWechatyRaw(){
  const res = await fetch('/api/channels/wechaty/raw?limit=20');
  const data = await res.json();
  wechatyRaw.textContent = JSON.stringify(data, null, 2);
}
async function loadWechatChannelStatus(){
  try{
    const res = await fetch(`${WECHATY_OUTBOUND_BASE}/health`);
    const data = await res.json();
    wechatSendOutput.textContent = JSON.stringify({
      ok: data.ok,
      send_channel_enabled: data.send_channel_enabled,
      auto_send_reply: data.auto_send_reply,
      known_contact_count: data.known_contact_count
    }, null, 2);
  }catch(err){
    wechatSendOutput.textContent = '微信发送通道不可用：' + err;
  }
}
async function setWechatAutoSend(enabled){
  const ok = confirm(enabled ? '确认开启自动外发？' : '确认暂停自动外发？');
  if(!ok) return;
  try{
    const res = await fetch(`${WECHATY_OUTBOUND_BASE}/auto-send`,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({enabled})
    });
    wechatSendOutput.textContent = JSON.stringify(await res.json(), null, 2);
  }catch(err){
    wechatSendOutput.textContent = '切换失败：' + err;
  }
}
async function setWechatSendChannel(enabled){
  const ok = confirm(enabled ? '确认开启微信发送总闸？开启后手动发送可用，自动外发仍受自动外发开关控制。' : '确认暂停微信发送总闸？暂停后手动发送和自动外发都会被挡住。');
  if(!ok) return;
  try{
    const res = await fetch(`${WECHATY_OUTBOUND_BASE}/send-channel`,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({enabled})
    });
    wechatSendOutput.textContent = JSON.stringify(await res.json(), null, 2);
  }catch(err){
    wechatSendOutput.textContent = '切换失败：' + err;
  }
}
async function sendWechatText(){
  const payload = {to: wechatTarget.value, text: wechatText.value || latestResult?.final_reply || ''};
  if(!payload.text.trim()){
    alert('没有可发送文本。');
    return;
  }
  const ok = confirm(`确认发给 ${payload.to}？\\n\\n${payload.text}`);
  if(!ok) return;
  try{
    const res = await fetch(`${WECHATY_OUTBOUND_BASE}/send`,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload)
    });
    wechatSendOutput.textContent = JSON.stringify(await res.json(), null, 2);
  }catch(err){
    wechatSendOutput.textContent = '发送失败：' + err;
  }
}
async function refreshLive(){
  liveStatus.textContent = '刷新中...';
  try{
    await Promise.all([loadRuntimeLogs(), loadWechatyRaw(), loadState()]);
    liveStatus.textContent = '已刷新 ' + new Date().toLocaleTimeString();
  }catch(err){
    liveStatus.textContent = '刷新失败：' + err;
  }
}
function toggleAutoRefresh(){
  if(liveTimer){
    clearInterval(liveTimer);
    liveTimer = null;
  }
  if(autoRefresh.checked){
    liveTimer = setInterval(refreshLive, 2500);
  }
}
async function resetState(){
  const ok = confirm('确认清空当前测试状态和记忆？会删除局、草稿、对话上下文、checkpoint、幂等缓存和消息结果；默认保留客户画像、badcase/eval 和日志。');
  if(!ok) return;
  const res = await fetch('/api/reset-state',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({reason:'operator clicked reset state and memory'})
  });
  output.textContent = JSON.stringify(await res.json(), null, 2);
  window.lastTraceId = '';
  await refreshLive();
}
toggleAutoRefresh();
refreshLive();
</script>
"""


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    global INPUT_SCHEDULER
    runtime = get_runtime()
    INPUT_SCHEDULER = PendingInputScheduler(
        store=runtime.store,
        handler=handle_due_pending_input_batch,
        trace_recorder=runtime.trace_recorder,
        poll_interval_seconds=env_float("MAHJONG_INPUT_SCHEDULER_POLL_SECONDS", 0.5),
        batch_limit=env_int("MAHJONG_INPUT_SCHEDULER_BATCH_LIMIT", 50),
    )
    INPUT_SCHEDULER.start()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), AgentRuntimeHandler)
    print(f"Mahjong Agent Runtime listening on http://127.0.0.1:{PORT}")
    print(f"Trace log: {TRACE_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if INPUT_SCHEDULER is not None:
            INPUT_SCHEDULER.stop()
        server.server_close()
        print("Mahjong Agent Runtime stopped.")


if __name__ == "__main__":
    main()
