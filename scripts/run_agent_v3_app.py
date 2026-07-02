from __future__ import annotations

import json
import os
import sys
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


from mahjong_agent_v3 import (  # noqa: E402
    AgentRuntimeV3,
    CustomerProfileV3,
    InMemoryAgentStoreV3,
    JsonlTraceRecorderV3,
    OpenAICompatibleAgentClientV3,
    TokenBudgetV3,
    ToolGatewayV3,
    UserMessageV3,
)
from mahjong_agent_v3.tracing import validate_trace_v3  # noqa: E402


PORT = int(os.getenv("MAHJONG_AGENT_V3_PORT", "8791"))
TRACE_PATH = ROOT / "logs" / "agent_runtime_v3_trace.log"


RUNTIME: AgentRuntimeV3 | None = None


def build_runtime() -> AgentRuntimeV3:
    load_dotenv_defaults(ROOT / ".env")
    llm_client = OpenAICompatibleAgentClientV3.from_env()
    if llm_client is None:
        raise RuntimeError("MAHJONG_LLM_API_KEY and MAHJONG_LLM_MODEL are required for AgentRuntimeV3.")
    store = InMemoryAgentStoreV3()
    seed_customers(store)
    trace = JsonlTraceRecorderV3(TRACE_PATH)
    gateway = ToolGatewayV3(store=store, trace_recorder=trace)
    return AgentRuntimeV3(
        llm_client=llm_client,
        store=store,
        tool_gateway=gateway,
        trace_recorder=trace,
        token_budget=TokenBudgetV3(
            max_tokens_per_call=env_int("MAHJONG_AGENT_V3_MAX_TOKENS_PER_CALL", env_int("MAHJONG_LLM_MAX_TOKENS_PER_CALL", 24_000)),
            max_calls_per_turn=env_int("MAHJONG_AGENT_V3_MAX_CALLS_PER_TURN", 8),
        ),
        max_steps=env_int("MAHJONG_AGENT_V3_MAX_STEPS", 8),
        llm_timeout_seconds=float(env_int("MAHJONG_AGENT_V3_LLM_TIMEOUT_SECONDS", 45)),
    )


def get_runtime() -> AgentRuntimeV3:
    global RUNTIME
    if RUNTIME is None:
        RUNTIME = build_runtime()
    return RUNTIME


def seed_customers(store: InMemoryAgentStoreV3) -> None:
    profiles = [
        CustomerProfileV3(
            customer_id="zhang",
            display_name="张哥",
            gender="男",
            preferred_games=["hangzhou_mahjong", "sichuan_mahjong"],
            preferred_stakes=["0.5", "1"],
            smoke_preference="any",
            response_score=0.9,
            notes="常客，杭麻和川麻都打。",
        ),
        CustomerProfileV3(
            customer_id="ran",
            display_name="冉姐",
            gender="女",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["0.5", "1"],
            smoke_preference="any",
            response_score=0.85,
        ),
        CustomerProfileV3(
            customer_id="he",
            display_name="何哥",
            gender="男",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["1"],
            smoke_preference="any",
            response_score=0.8,
        ),
    ]
    for profile in profiles:
        store.upsert_customer(profile)


class AgentV3Handler(BaseHTTPRequestHandler):
    server_version = "MahjongAgentV3/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._html(index_html())
            return
        if parsed.path == "/api/v3/state":
            runtime = get_runtime()
            self._json(
                {
                    "games": [item.to_dict() for item in runtime.store.games.values()],
                    "invite_drafts": [item.to_dict() for item in runtime.store.invite_drafts.values()],
                    "customers": [item.to_dict() for item in runtime.store.customers.values()],
                    "runtime_config": runtime_config(runtime),
                }
            )
            return
        if parsed.path == "/api/v3/traces":
            runtime = get_runtime()
            trace_id = (parse_qs(parsed.query).get("trace_id") or [""])[0]
            events = runtime.trace_recorder.get_trace(trace_id)
            self._json({"trace_id": trace_id, "trace_log_path": str(TRACE_PATH), "events": [item.to_dict() for item in events], "completeness": validate_trace_v3(events)})
            return
        if parsed.path == "/api/v3/badcases":
            runtime = get_runtime()
            self._json({"records": list(runtime.store.badcases)})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/v3/message":
            runtime = get_runtime()
            payload = self._read_json()
            message = UserMessageV3(
                conversation_id=str(payload.get("conversation_id") or "boss_v3"),
                sender_id=str(payload.get("sender_id") or "zhang"),
                sender_name=str(payload.get("sender_name") or "张哥"),
                text=str(payload.get("text") or ""),
                message_id=str(payload.get("message_id") or "") or None,
            )
            if message.message_id is None:
                message = UserMessageV3(
                    conversation_id=message.conversation_id,
                    sender_id=message.sender_id,
                    sender_name=message.sender_name,
                    text=message.text,
                )
            result = runtime.handle_user_message(message, trace_id=payload.get("trace_id"))
            self._json(result.to_dict())
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


def runtime_config(runtime: AgentRuntimeV3) -> dict:
    llm_config = getattr(getattr(runtime, "llm_client", None), "config", None)
    return {
        "runtime": "mahjong_agent_v3",
        "llm": {
            "provider": getattr(llm_config, "provider", ""),
            "model": getattr(llm_config, "model", ""),
            "base_url": getattr(llm_config, "base_url", ""),
            "max_completion_tokens": getattr(llm_config, "max_tokens", None),
        },
        "max_steps": runtime.max_steps,
        "max_tokens_per_call": runtime.token_budget.max_tokens_per_call,
        "max_calls_per_turn": runtime.token_budget.max_calls_per_turn,
        "trace_log": str(TRACE_PATH),
    }


def index_html() -> str:
    return """
<!doctype html>
<meta charset="utf-8">
<title>Mahjong Agent V3</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:32px;background:#f8faf8;color:#1f2a24}
main{max-width:980px;margin:auto}
textarea,input,button{font:inherit}
input,textarea{width:100%;box-sizing:border-box;border:1px solid #b9c7bd;border-radius:8px;padding:12px;background:white}
textarea{min-height:140px}
button{border:1px solid #2f7d62;background:#2f7d62;color:white;border-radius:8px;padding:10px 16px;cursor:pointer}
pre{white-space:pre-wrap;background:white;border:1px solid #d6ded8;border-radius:8px;padding:16px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
</style>
<main>
  <h1>Mahjong Agent V3</h1>
  <p>独立 V3 主链路：模型决定工具，后端只做合同、权限、幂等、状态和审计。</p>
  <div class="grid">
    <input id="conversationId" value="boss_v3" placeholder="conversationId">
    <input id="senderId" value="zhang" placeholder="senderId">
  </div>
  <p><input id="senderName" value="张哥" placeholder="senderName"></p>
  <p><textarea id="text">通宵1块有人吗？没有就帮我组一个</textarea></p>
  <button onclick="sendMessage()">发送到 V3</button>
  <button onclick="loadState()">刷新状态</button>
  <h2>结果</h2>
  <pre id="output"></pre>
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
  const res = await fetch('/api/v3/message',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  output.textContent = JSON.stringify(await res.json(), null, 2);
  await loadState();
}
async function loadState(){
  const res = await fetch('/api/v3/state');
  state.textContent = JSON.stringify(await res.json(), null, 2);
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


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), AgentV3Handler)
    print(f"Mahjong Agent V3 listening on http://127.0.0.1:{PORT}")
    print(f"Trace log: {TRACE_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Mahjong Agent V3 stopped.")


if __name__ == "__main__":
    main()
