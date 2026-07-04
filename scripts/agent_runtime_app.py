from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
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
    AgentRuntime,
    AgentLLMConfig,
    CustomerRelationship,
    CustomerProfile,
    JsonlTraceRecorder,
    OpenAICompatibleAgentClient,
    SQLiteAgentStore,
    TokenBudget,
    ToolCall,
    ToolGateway,
    UserMessage,
)
from mahjong_agent_runtime.summary import ContextSummaryManager, ContextSummaryPolicy  # noqa: E402
from mahjong_agent_runtime.tracing import validate_trace  # noqa: E402


PORT = int(os.getenv("MAHJONG_AGENT_PORT", "8790"))
TRACE_PATH = Path(os.getenv("MAHJONG_AGENT_TRACE_PATH") or ROOT / "logs" / "agent_runtime_trace.log")
DB_PATH = Path(os.getenv("MAHJONG_AGENT_DB_PATH") or ROOT / "data" / "agent_runtime.sqlite3")


RUNTIME: AgentRuntime | None = None


def build_runtime() -> AgentRuntime:
    load_dotenv_defaults(ROOT / ".env")
    llm_client = OpenAICompatibleAgentClient.from_env()
    if llm_client is None:
        raise RuntimeError("MAHJONG_LLM_API_KEY and MAHJONG_LLM_MODEL are required for AgentRuntime.")
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
        review_token_budget=TokenBudget(
            max_tokens_per_call=env_int("MAHJONG_AGENT_REVIEW_MAX_TOKENS_PER_CALL", main_budget.max_tokens_per_call),
            max_calls_per_turn=env_int("MAHJONG_AGENT_REVIEW_MAX_CALLS_PER_TURN", 8),
        ),
        max_steps=env_int("MAHJONG_AGENT_MAX_STEPS", 8),
        llm_timeout_seconds=float(env_int("MAHJONG_AGENT_LLM_TIMEOUT_SECONDS", 45)),
        reply_self_review_enabled=env_bool("MAHJONG_AGENT_REPLY_SELF_REVIEW_ENABLED", True),
        reply_self_review_client=reply_self_review_client,
        context_summary_manager=summary_manager,
    )


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
            message = UserMessage(
                conversation_id=str(payload.get("conversation_id") or "runtime_trial"),
                sender_id=str(payload.get("sender_id") or "zhang"),
                sender_name=str(payload.get("sender_name") or "张哥"),
                text=str(payload.get("text") or ""),
                message_id=str(payload.get("message_id") or "") or None,
            )
            if message.message_id is None:
                message = UserMessage(
                    conversation_id=message.conversation_id,
                    sender_id=message.sender_id,
                    sender_name=message.sender_name,
                    text=message.text,
                )
            result = runtime.handle_user_message(message, trace_id=payload.get("trace_id"))
            self._json(result.to_dict())
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
        "trace_log": str(TRACE_PATH),
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
</style>
<main>
  <h1>Mahjong Agent Runtime</h1>
  <p>当前主链路：模型决定工具，后端只做合同、权限、幂等、状态和审计。</p>
  <div class="grid">
    <input id="conversationId" value="runtime_trial" placeholder="conversationId">
    <input id="senderId" value="zhang" placeholder="senderId">
  </div>
  <p><input id="senderName" value="张哥" placeholder="senderName"></p>
  <p><textarea id="text">通宵1块有人吗？没有就帮我组一个</textarea></p>
  <button onclick="sendMessage()">发送</button>
  <button onclick="loadState()">刷新状态</button>
  <button onclick="recordBadcase()">标记 badcase</button>
  <button class="danger" onclick="resetState()">清空状态和记忆</button>
  <h2>结果</h2>
  <pre id="output"></pre>
  <h2>人工 badcase</h2>
  <p><input id="badcaseReason" value="回复不符合预期" placeholder="badcase 原因"></p>
  <p><textarea id="badcaseExpected" placeholder="期望行为或回复"></textarea></p>
  <pre id="badcaseOutput"></pre>
  <h2>状态</h2>
  <pre id="state"></pre>
</main>
<script>
async function sendMessage(){
  const payload = {
    conversation_id: conversationId.value,
    sender_id: senderId.value,
    sender_name: senderName.value,
    text: text.value
  };
  const res = await fetch('/api/message',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const body = await res.json();
  window.lastTraceId = body.trace_id;
  output.textContent = JSON.stringify(body, null, 2);
  await loadState();
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
  await loadState();
}
loadState();
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
