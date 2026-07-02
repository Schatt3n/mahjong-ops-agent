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


RUNTIME: AgentRuntimeV2 | None = None


def get_runtime() -> AgentRuntimeV2:
    global RUNTIME
    if RUNTIME is None:
        RUNTIME = build_runtime()
    return RUNTIME


class AgentV2Handler(BaseHTTPRequestHandler):
    server_version = "MahjongAgentV2/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._html(index_html())
            return
        if parsed.path == "/api/v2/state":
            runtime = get_runtime()
            self._json(
                {
                    "games": [game.to_dict() for game in runtime.store.games.values()],
                    "invite_drafts": [draft.to_dict() for draft in runtime.store.invite_drafts.values()],
                    "customers": [customer.to_dict() for customer in runtime.store.customers.values()],
                    "db_path": str(DB_PATH),
                }
            )
            return
        if parsed.path == "/api/v2/traces":
            runtime = get_runtime()
            query = parse_qs(parsed.query)
            trace_id = (query.get("trace_id") or [""])[0]
            self._json({"trace_id": trace_id, "events": [event.to_dict() for event in runtime.trace_recorder.get_trace(trace_id)]})
            return
        if parsed.path == "/api/v2/badcases":
            self._json({"path": str(BADCASE_PATH), "records": read_jsonl(BADCASE_PATH)})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/v2/message":
            runtime = get_runtime()
            payload = self._read_json()
            message_kwargs = {
                "conversation_id": str(payload.get("conversation_id") or "boss_trial_v2"),
                "sender_id": str(payload.get("sender_id") or "zhang"),
                "sender_name": str(payload.get("sender_name") or "张哥"),
                "text": str(payload.get("text") or ""),
            }
            if payload.get("message_id"):
                message_kwargs["message_id"] = str(payload["message_id"])
            message = UserMessageV2(**message_kwargs)
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


def index_html() -> str:
    return """
<!doctype html>
<meta charset="utf-8">
<title>Mahjong Agent Runtime V2</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f7f8f5;color:#1d2620}
header{padding:18px 24px;border-bottom:1px solid #d9dfd7;background:#fff}
h1{font-size:22px;margin:0}
h2{font-size:17px;margin:0 0 12px}
h3{font-size:15px;margin:14px 0 8px}
main{display:grid;grid-template-columns:320px minmax(420px,1fr) 360px;gap:16px;padding:16px}
section{background:#fff;border:1px solid #d9dfd7;border-radius:8px;padding:16px;min-width:0}
textarea,input,button{font:inherit;font-size:15px}
textarea{width:100%;height:150px;resize:vertical}
input,textarea{box-sizing:border-box;border:1px solid #bac6bb;border-radius:6px;padding:9px;margin:6px 0;background:#fff}
input{width:100%}
button{padding:10px 14px;border:1px solid #1f6f55;border-radius:6px;background:#26745a;color:#fff;cursor:pointer}
button.secondary{background:#fff;color:#1f6f55}
button.danger{background:#fff;color:#9b1c1c;border-color:#b64b4b}
.row{display:flex;gap:8px;flex-wrap:wrap}
.card{border:1px solid #dce2db;border-radius:8px;padding:12px;margin:10px 0;background:#fbfcfa}
.muted{color:#607064}
.pill{display:inline-block;border-radius:999px;background:#e9f1ec;padding:3px 9px;margin:2px 4px 2px 0;font-size:13px}
.dangerText{color:#a22626}
pre{white-space:pre-wrap;word-break:break-word;background:#f4f6f2;padding:12px;border:1px solid #d8ded6;border-radius:6px;max-height:360px;overflow:auto}
.traceList{max-height:420px;overflow:auto}
.traceEvent{border-left:3px solid #6a8b77;padding:8px 10px;margin:8px 0;background:#f8faf7}
@media(max-width:1100px){main{grid-template-columns:1fr}.traceList{max-height:none}}
</style>
<header>
  <h1>Mahjong Agent Runtime V2</h1>
</header>
<main>
  <section>
    <h2>输入</h2>
    <input id="conversation_id" value="boss_trial_v2" placeholder="conversation_id">
    <input id="message_id" value="" placeholder="message_id，可选；重复投递测试用">
    <input id="sender_id" value="zhang" placeholder="sender_id">
    <input id="sender_name" value="张哥" placeholder="sender_name">
    <textarea id="text">通宵有人吗</textarea>
    <div class="row">
      <button onclick="send()">发送</button>
      <button class="secondary" onclick="loadState()">刷新状态</button>
      <button class="secondary" onclick="loadBadcases()">刷新 badcase</button>
    </div>
    <h3>最近 traceId</h3>
    <input id="trace_id" placeholder="trace_id">
    <button class="secondary" onclick="loadTrace()">查看 trace</button>
  </section>
  <section>
    <h2>本轮结果</h2>
    <div id="reply" class="card muted">暂无</div>
    <h3>模型决策</h3>
    <div id="decisions"></div>
    <h3>工具调用</h3>
    <div id="tools"></div>
    <h3>状态变化</h3>
    <div id="transitions"></div>
  </section>
  <section>
    <h2>状态</h2>
    <div id="state"></div>
    <h3>Trace</h3>
    <div id="trace" class="traceList"></div>
    <h3>Badcase</h3>
    <div id="badcases"></div>
  </section>
</main>
<script>
let lastTraceId="";
function esc(v){
  return String(v ?? '').replace(/[&<>"']/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s]));
}
function pretty(v){return esc(JSON.stringify(v,null,2));}
function pill(text){return `<span class="pill">${esc(text)}</span>`}
async function send(){
  const body={
    conversation_id:document.getElementById('conversation_id').value,
    message_id:document.getElementById('message_id').value,
    sender_id:document.getElementById('sender_id').value,
    sender_name:document.getElementById('sender_name').value,
    text:document.getElementById('text').value
  };
  const res=await fetch('/api/v2/message',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await res.json();
  renderRun(data);
  await loadState();
  await loadBadcases();
  await loadTrace(data.trace_id);
}
function renderRun(data){
  lastTraceId=data.trace_id || "";
  document.getElementById('trace_id').value=lastTraceId;
  document.getElementById('reply').innerHTML=`<b>${esc(data.final_reply || '无回复')}</b><div class="muted">${esc(lastTraceId)}</div>`;
  document.getElementById('decisions').innerHTML=(data.decisions||[]).map((d,i)=>`
    <div class="card">
      ${pill('#'+(i+1))}${pill(d.goal||'goal')}
      <div>${esc(d.reasoning_summary||'')}</div>
      <div class="muted">${esc(d.reply_to_user||'')}</div>
      ${(d.tool_calls||[]).map(c=>pill(c.name)).join('')}
      ${d.needs_human ? pill('needs_human') : ''}
    </div>`).join('') || '<div class="muted">暂无</div>';
  document.getElementById('tools').innerHTML=(data.tool_results||[]).map(t=>`
    <div class="card">
      ${pill(t.name)}${pill(t.called?'called':'not_called')}${pill(t.allowed?'allowed':'blocked')}
      ${t.deduplicated ? pill('deduplicated') : ''}
      ${t.error ? `<div class="dangerText">${esc(t.error)}</div>` : ''}
      <pre>${pretty(t.result || {})}</pre>
    </div>`).join('') || '<div class="muted">暂无</div>';
  document.getElementById('transitions').innerHTML=(data.state_transitions||[]).map(t=>`
    <div class="card">
      ${pill(t.entity_type)}${pill(t.entity_id)}${pill((t.from_status||'null')+' -> '+t.to_status)}
      <div>${esc(t.reason||'')}</div>
    </div>`).join('') || '<div class="muted">暂无</div>';
}
async function loadState(){
  const data=await (await fetch('/api/v2/state')).json();
  document.getElementById('state').innerHTML=`
    <div class="muted">${esc(data.db_path||'')}</div>
    <div>${pill('客户 '+(data.customers||[]).length)}${pill('局 '+(data.games||[]).length)}${pill('草稿 '+(data.invite_drafts||[]).length)}</div>
    <h3>当前局</h3>${(data.games||[]).map(g=>`<div class="card">${pill(g.status)}<b>${esc(g.game_id)}</b><pre>${pretty(g.requirement||{})}</pre></div>`).join('') || '<div class="muted">暂无</div>'}
    <h3>邀约草稿</h3>${(data.invite_drafts||[]).map(d=>`<div class="card">${pill(d.status)}<b>${esc(d.display_name)}</b><div>${esc(d.message_text)}</div></div>`).join('') || '<div class="muted">暂无</div>'}
  `;
}
async function loadTrace(id){
  const traceId=id || document.getElementById('trace_id').value || lastTraceId;
  if(!traceId){document.getElementById('trace').innerHTML='<div class="muted">暂无</div>';return;}
  const data=await (await fetch('/api/v2/traces?trace_id='+encodeURIComponent(traceId))).json();
  document.getElementById('trace').innerHTML=(data.events||[]).map(e=>`
    <div class="traceEvent">
      ${pill(e.step)}${pill(e.level)}
      <div class="muted">${esc(e.time)}</div>
      <pre>${pretty(e.content||{})}</pre>
    </div>`).join('') || '<div class="muted">暂无</div>';
}
async function loadBadcases(){
  const data=await (await fetch('/api/v2/badcases')).json();
  document.getElementById('badcases').innerHTML=`
    <div class="muted">${esc(data.path||'')}</div>
    ${(data.records||[]).slice(-6).reverse().map(b=>`<div class="card">${pill(b.badcase_id)}<div>${esc(b.reason)}</div><pre>${pretty(b)}</pre></div>`).join('') || '<div class="muted">暂无</div>'}
  `;
}
loadState();loadBadcases();
</script>
"""


def main() -> None:
    get_runtime()
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
