from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mahjong_agent_v2 import (  # noqa: E402
    AgentRuntimeV2,
    CustomerProfileV2,
    JsonlEvalRecorderV2,
    JsonlTraceRecorderV2,
    OpenAICompatibleAgentClientV2,
    SQLiteAgentStoreV2,
    ToolGatewayV2,
    UserMessageV2,
)


PORT = int(os.getenv("MAHJONG_AGENT_V2_PORT", "8791"))
TRACE_PATH = ROOT / "logs" / "agent_runtime_v2_trace.jsonl"
BADCASE_PATH = ROOT / "eval" / "badcases" / "agent_runtime_v2_badcases.jsonl"
DB_PATH = Path(os.getenv("MAHJONG_AGENT_V2_DB_PATH") or ROOT / "data" / "agent_runtime_v2.sqlite3")


def build_runtime() -> AgentRuntimeV2:
    llm_client = OpenAICompatibleAgentClientV2.from_env()
    if llm_client is None:
        raise RuntimeError("MAHJONG_LLM_API_KEY and MAHJONG_LLM_MODEL are required for AgentRuntimeV2.")
    store = SQLiteAgentStoreV2(DB_PATH)
    seed_customers(store)
    tool_gateway = ToolGatewayV2(
        store=store,
        eval_recorder=JsonlEvalRecorderV2(BADCASE_PATH),
    )
    return AgentRuntimeV2(
        llm_client=llm_client,
        store=store,
        tool_gateway=tool_gateway,
        trace_recorder=JsonlTraceRecorderV2(TRACE_PATH),
    )


def seed_customers(store) -> None:
    profiles = [
        CustomerProfileV2(
            customer_id="zhang",
            display_name="张哥",
            gender="男",
            preferred_games=["hangzhou_mahjong", "sichuan_mahjong"],
            preferred_stakes=["0.5", "1"],
            smoke_preference="any",
            response_score=0.9,
            notes="常客，杭麻和川麻都打。",
        ),
        CustomerProfileV2(
            customer_id="ran",
            display_name="冉姐",
            gender="女",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["0.5", "1"],
            smoke_preference="any",
            response_score=0.85,
        ),
        CustomerProfileV2(
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


RUNTIME = build_runtime()


class AgentV2Handler(BaseHTTPRequestHandler):
    server_version = "MahjongAgentV2/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._html(index_html())
            return
        if parsed.path == "/api/v2/state":
            self._json(
                {
                    "games": [game.to_dict() for game in RUNTIME.store.games.values()],
                    "invite_drafts": [draft.to_dict() for draft in RUNTIME.store.invite_drafts.values()],
                    "customers": [customer.to_dict() for customer in RUNTIME.store.customers.values()],
                    "db_path": str(DB_PATH),
                }
            )
            return
        if parsed.path == "/api/v2/traces":
            query = parse_qs(parsed.query)
            trace_id = (query.get("trace_id") or [""])[0]
            self._json({"trace_id": trace_id, "events": [event.to_dict() for event in RUNTIME.trace_recorder.get_trace(trace_id)]})
            return
        if parsed.path == "/api/v2/badcases":
            self._json({"path": str(BADCASE_PATH), "records": read_jsonl(BADCASE_PATH)})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/v2/message":
            payload = self._read_json()
            message = UserMessageV2(
                conversation_id=str(payload.get("conversation_id") or "boss_trial_v2"),
                sender_id=str(payload.get("sender_id") or "zhang"),
                sender_name=str(payload.get("sender_name") or "张哥"),
                text=str(payload.get("text") or ""),
            )
            result = RUNTIME.handle_user_message(message, trace_id=payload.get("trace_id"))
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


def index_html() -> str:
    return """
<!doctype html>
<meta charset="utf-8">
<title>Mahjong Agent Runtime V2</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:32px;max-width:960px}
textarea,input,button{font:inherit;font-size:16px}
textarea{width:100%;height:120px}
input{width:100%;padding:8px;margin:6px 0}
button{padding:10px 16px}
pre{white-space:pre-wrap;background:#f6f7f5;padding:16px;border:1px solid #d8ded6}
</style>
<h1>Mahjong Agent Runtime V2</h1>
<p>独立 V2 主链路：LLM 决策工具，后端只做工具网关、状态、幂等、预算、审计。</p>
<input id="conversation_id" value="boss_trial_v2" placeholder="conversation_id">
<input id="sender_id" value="zhang" placeholder="sender_id">
<input id="sender_name" value="张哥" placeholder="sender_name">
<textarea id="text">通宵有人吗</textarea>
<button onclick="send()">发送到 V2 Agent</button>
<pre id="output"></pre>
<script>
async function send(){
  const body={
    conversation_id:document.getElementById('conversation_id').value,
    sender_id:document.getElementById('sender_id').value,
    sender_name:document.getElementById('sender_name').value,
    text:document.getElementById('text').value
  };
  const res=await fetch('/api/v2/message',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  document.getElementById('output').textContent=JSON.stringify(await res.json(),null,2);
}
</script>
"""


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), AgentV2Handler)
    print(f"Mahjong Agent Runtime V2 listening on http://127.0.0.1:{PORT}")
    print(f"Trace log: {TRACE_PATH}")
    print(f"Badcase log: {BADCASE_PATH}")
    print(f"SQLite state: {DB_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Mahjong Agent Runtime V2 stopped.")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


if __name__ == "__main__":
    main()
