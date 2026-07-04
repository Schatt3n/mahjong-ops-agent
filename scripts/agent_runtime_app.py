from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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
    QuotedMessageRef,
    SQLiteAgentStore,
    TokenBudget,
    ToolCall,
    ToolGateway,
    ToolResult,
    UserMessage,
)
from mahjong_agent_runtime.summary import ContextSummaryManager, ContextSummaryPolicy  # noqa: E402
from mahjong_agent_runtime.tracing import validate_trace  # noqa: E402


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
DEFAULT_WECHATY_AGENT_WHITELIST = {
    "@a848814d1f5ac34c2926032824f9c369b9596f7d4f8295a6936310a2630bd477",
    "@844362549504da622551a39c8569ab565dd821d90291f1ead5befff9c1856aa4",
    "@bcf90150509c71de7409b5204b82d919c423c5a9d9958f1ac8563a8f6ff0a097",
    "@ae3faab1468870edf94552f9802092efbdd3910943edcc0cbfe2c3afd65134b5",
    "@5657a9459a503bf10c1360f24e491963",
    "xml31323",
    "刘峻甫-21M-高分子-宜宾",
    "陈子贤",
    "Ech0",
    "刘臻",
    "噜噜小王！",
}


RUNTIME: AgentRuntime | None = None


def build_runtime() -> AgentRuntime:
    load_dotenv_defaults(ROOT / ".env")
    llm_client = OpenAICompatibleAgentClient.from_env()
    if llm_client is None:
        raise RuntimeError("MAHJONG_LLM_API_KEY and MAHJONG_LLM_MODEL are required for AgentRuntime.")
    customer_visible_text_generation_client = build_customer_visible_text_generation_client()
    reply_self_review_client = build_reply_self_review_client()
    store = SQLiteAgentStore(DB_PATH)
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
        max_tokens_per_call=env_int("MAHJONG_AGENT_MAX_TOKENS_PER_CALL", env_int("MAHJONG_LLM_MAX_TOKENS_PER_CALL", 24_000)),
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
        llm_timeout_seconds=float(env_int("MAHJONG_AGENT_LLM_TIMEOUT_SECONDS", 45)),
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


def parse_quoted_message_ref(payload: dict) -> QuotedMessageRef | None:
    raw = payload.get("quoted_message")
    if raw is None:
        raw = payload.get("quoted")
    if raw is None:
        raw_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        raw = raw_payload.get("quoted_message") or raw_payload.get("quoted") or raw_payload.get("quote")
    if not isinstance(raw, dict):
        return None

    message_id = str(
        raw.get("message_id")
        or raw.get("source_message_id")
        or raw.get("id")
        or raw.get("msgId")
        or ""
    ).strip()
    text = str(raw.get("text") or raw.get("raw_text") or raw.get("content") or "").strip()
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
        metadata=dict(metadata),
    )


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
    allowed = {item.lower() for item in DEFAULT_WECHATY_AGENT_WHITELIST}
    allowed.update(item.lower() for item in split_env_list(os.getenv("MAHJONG_WECHATY_AGENT_WHITELIST", "")))
    if not allowed:
        return []
    return [item for item in wechaty_identity_values(payload) if item.lower() in allowed]


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
                "should_route": False,
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
                "should_route": False,
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
    return (
        {
            "should_route": bool(payload.get("should_route")) if isinstance(payload.get("should_route"), bool) else False,
            "category": category,
            "confidence": confidence,
            "reasoning_summary": reasoning_summary,
            "evidence": evidence,
        },
        errors,
    )


def build_wechaty_input_gate_payload(message: UserMessage, runtime: AgentRuntime) -> dict:
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
        },
        "output_contract": {
            "format": "json_object",
            "required_keys": ["should_route", "category", "confidence", "reasoning_summary", "evidence"],
            "categories": ["operational", "followup_answer", "candidate_reply", "casual_chat", "non_mahjong", "uncertain"],
        },
    }


def run_wechaty_input_gate(message: UserMessage, *, trace_id: str, runtime: AgentRuntime) -> dict:
    if not env_bool("MAHJONG_WECHATY_INPUT_GATE_ENABLED", True):
        return {
            "enabled": False,
            "should_route": True,
            "category": "disabled",
            "confidence": 1.0,
            "reasoning_summary": "Wechaty input gate disabled by env.",
            "evidence": [],
            "errors": [],
        }
    client = build_wechaty_input_gate_client() or runtime.llm_client
    payload = build_wechaty_input_gate_payload(message, runtime)
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
            "should_route": fail_open,
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
    runtime.trace_recorder.record(trace_id, "wechaty_input_gate_response", {"content": raw_response, "elapsed_ms": decision["elapsed_ms"]})
    runtime.trace_recorder.record(trace_id, "wechaty_input_gate_decision", decision, level="WARN" if errors else "INFO")
    if errors and not env_bool("MAHJONG_WECHATY_INPUT_GATE_FAIL_OPEN", False):
        decision["should_route"] = False
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
                env_int("MAHJONG_AGENT_MAX_TOKENS_PER_CALL", env_int("MAHJONG_LLM_MAX_TOKENS_PER_CALL", 24_000)),
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
    text = str(payload.get("text") or payload.get("raw_text") or "").strip()
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
        "route_scope": route_scope,
        "agent_whitelisted": whitelisted,
        "agent_whitelist_hits": whitelist_hits,
    }
    if not env_bool("MAHJONG_WECHATY_AUTO_ROUTE_TO_AGENT", True):
        audit["reason"] = "auto_route_disabled"
        return None, audit
    if not text:
        audit["reason"] = "empty_text"
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
        quoted_message=parse_quoted_message_ref(payload),
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
            "quoted_message": message.quoted_message.to_dict() if message.quoted_message else None,
        }
    )
    return message, audit


def route_wechaty_raw_to_agent(payload: dict, *, trace_id: str) -> dict:
    message, audit = build_wechaty_user_message(payload)
    trace_recorder = JsonlTraceRecorder(TRACE_PATH)
    if message is None:
        trace_recorder.record(trace_id, "wechaty_raw_message_not_routed", audit)
        return {"routed_to_agent": False, "audit": audit, "agent_result": None}
    runtime = get_runtime()
    gate_decision = run_wechaty_input_gate(message, trace_id=trace_id, runtime=runtime)
    audit["input_gate"] = gate_decision
    if not gate_decision.get("should_route"):
        category = str(gate_decision.get("category") or "").strip()
        if (
            env_bool("MAHJONG_WECHATY_CASUAL_CHAT_REPLY_ENABLED", True)
            and category in {"casual_chat", "non_mahjong"}
            and not gate_decision.get("errors")
        ):
            audit["routed_to_agent"] = False
            audit["reason"] = "wechaty_input_gate_routed_to_casual_chat"
            runtime.trace_recorder.record(trace_id, "wechaty_raw_message_routed_to_casual_chat", audit)
            result = handle_wechaty_casual_chat(message, trace_id=trace_id, runtime=runtime, gate_decision=gate_decision)
            return {"routed_to_agent": False, "audit": audit, "agent_result": result.to_dict(), "casual_chat_result": result.to_dict()}
        audit["routed_to_agent"] = False
        audit["reason"] = "wechaty_input_gate_not_routed"
        runtime.trace_recorder.record(trace_id, "wechaty_raw_message_not_routed", audit)
        return {"routed_to_agent": False, "audit": audit, "agent_result": None}
    runtime.trace_recorder.record(trace_id, "wechaty_raw_message_routed_to_agent", audit)
    result = runtime.handle_user_message(message, trace_id=trace_id)
    return {"routed_to_agent": True, "audit": audit, "agent_result": result.to_dict()}


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
    profiles = [
        CustomerProfile(
            customer_id="zhang",
            display_name="张哥",
            gender="男",
            preferred_games=["hangzhou_mahjong", "sichuan_mahjong"],
            preferred_stakes=["0.5", "1"],
            smoke_preference="any",
            response_score=0.9,
            notes="常客，杭麻和川麻都打。",
        ),
        CustomerProfile(
            customer_id="ran",
            display_name="冉姐",
            gender="女",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["0.5", "1"],
            smoke_preference="any",
            response_score=0.85,
        ),
        CustomerProfile(
            customer_id="he",
            display_name="何哥",
            gender="男",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["1"],
            smoke_preference="any",
            response_score=0.8,
        ),
        CustomerProfile(
            customer_id="wang01",
            display_name="王哥",
            gender="男",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["0.5", "1"],
            smoke_preference="any",
            response_score=0.75,
            notes="常客，杭麻0.5/1都可以。",
        ),
        CustomerProfile(
            customer_id="@a848814d1f5ac34c2926032824f9c369b9596f7d4f8295a6936310a2630bd477",
            display_name="xml31323",
            preferred_games=[],
            preferred_stakes=[],
            preferred_time_tags=[],
            smoke_preference=None,
            response_score=0.7,
            notes="微信测试联系人，偏好待补充。Wechaty contact id 已确认。",
        ),
        CustomerProfile(
            customer_id="@844362549504da622551a39c8569ab565dd821d90291f1ead5befff9c1856aa4",
            display_name="刘峻甫-21M-高分子-宜宾",
            gender="男",
            preferred_games=["sichuan_mahjong"],
            preferred_stakes=[],
            preferred_time_tags=["anytime"],
            smoke_preference="smoke_ok",
            response_score=0.85,
            notes="用户确认：好哥们儿，抽烟，打川麻，随时可以打。",
        ),
        CustomerProfile(
            customer_id="@bcf90150509c71de7409b5204b82d919c423c5a9d9958f1ac8563a8f6ff0a097",
            display_name="陈子贤",
            gender="男",
            preferred_games=["sichuan_mahjong"],
            preferred_stakes=[],
            preferred_time_tags=["anytime"],
            smoke_preference="smoke_ok",
            response_score=0.85,
            notes="用户确认：好哥们儿，抽烟，打川麻，随时可以打。",
        ),
        CustomerProfile(
            customer_id="@ae3faab1468870edf94552f9802092efbdd3910943edcc0cbfe2c3afd65134b5",
            display_name="Ech0",
            gender="女",
            preferred_games=[],
            preferred_stakes=[],
            preferred_time_tags=[],
            smoke_preference=None,
            response_score=0.8,
            notes="用户确认：姐姐，白名单测试联系人。偏好待补充。",
        ),
        CustomerProfile(
            customer_id="@5657a9459a503bf10c1360f24e491963",
            display_name="刘臻",
            gender="男",
            preferred_games=[],
            preferred_stakes=[],
            preferred_time_tags=[],
            smoke_preference=None,
            response_score=0.8,
            notes="用户确认：好哥们儿，白名单测试联系人。Wechaty alias/name 日志已确认。",
        ),
    ]
    for profile in profiles:
        store.upsert_customer(profile)
    relationships = [
        CustomerRelationship(
            customer_a_id="zhang",
            customer_b_id="wang01",
            played_together_count=0,
            avoid_playing=False,
            notes="暂无共同打牌记录。",
        ),
        CustomerRelationship(
            customer_a_id="zhang",
            customer_b_id="ran",
            played_together_count=0,
            avoid_playing=True,
            notes="张哥不和冉姐同桌。",
        ),
    ]
    for relationship in relationships:
        store.upsert_customer_relationship(relationship)


class AgentRuntimeHandler(BaseHTTPRequestHandler):
    server_version = "MahjongAgentRuntime/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._html(index_html())
            return
        if parsed.path in {"/api/runtime", "/api/health"}:
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
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/message":
            runtime = get_runtime()
            payload = self._read_json()
            quoted_message = parse_quoted_message_ref(payload)
            message = UserMessage(
                conversation_id=str(payload.get("conversation_id") or "runtime_trial"),
                sender_id=str(payload.get("sender_id") or "zhang"),
                sender_name=str(payload.get("sender_name") or "张哥"),
                text=str(payload.get("text") or ""),
                message_id=str(payload.get("message_id") or "") or None,
                quoted_message=quoted_message,
            )
            if message.message_id is None:
                message = UserMessage(
                    conversation_id=message.conversation_id,
                    sender_id=message.sender_id,
                    sender_name=message.sender_name,
                    text=message.text,
                    quoted_message=quoted_message,
                )
            result = runtime.handle_user_message(message, trace_id=payload.get("trace_id"))
            self._json(result.to_dict())
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
        self.send_error(404)

    def log_message(self, fmt: str, *args) -> None:
        print(fmt % args)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
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
        "wechaty_agent_whitelist": sorted(
            {*DEFAULT_WECHATY_AGENT_WHITELIST, *split_env_list(os.getenv("MAHJONG_WECHATY_AGENT_WHITELIST", ""))}
        ),
        "wechaty_input_gate_enabled": env_bool("MAHJONG_WECHATY_INPUT_GATE_ENABLED", True),
        "wechaty_input_gate_model": os.getenv("MAHJONG_WECHATY_INPUT_GATE_LLM_MODEL")
        or getattr(getattr(runtime.llm_client, "config", None), "model", None),
        "wechaty_input_gate_fail_open": env_bool("MAHJONG_WECHATY_INPUT_GATE_FAIL_OPEN", False),
        "wechaty_input_gate_prompt": str(WECHATY_INPUT_GATE_PROMPT_PATH),
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
</style>
<main>
  <h1>Mahjong Agent Runtime</h1>
  <p>当前主链路：模型决定工具，后端只做合同、权限、幂等、状态和审计。</p>
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
const WECHATY_OUTBOUND_BASE = 'http://127.0.0.1:8791';

async function sendMessage(){
  const payload = {
    conversation_id: conversationId.value,
    sender_id: senderId.value,
    sender_name: senderName.value,
    text: text.value
  };
  const res = await fetch('/api/message',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const body = await res.json();
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
  state.textContent = JSON.stringify(await res.json(), null, 2);
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
    server = ThreadingHTTPServer(("127.0.0.1", PORT), AgentRuntimeHandler)
    print(f"Mahjong Agent Runtime listening on http://127.0.0.1:{PORT}")
    print(f"Trace log: {TRACE_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Mahjong Agent Runtime stopped.")


if __name__ == "__main__":
    main()
