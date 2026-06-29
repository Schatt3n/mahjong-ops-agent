from __future__ import annotations

import json
import os
import pathlib
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mahjong_agent import (
    AgentResponder,
    AgentRuntime,
    ChannelType,
    CustomerProfile,
    DurableAgentProcessor,
    IncomingEnvelope,
    Message,
    OpenAICompatibleLLMResolver,
    PlayPreference,
    RuntimeConfig,
    SQLiteDurableStore,
)


TZ = ZoneInfo("Asia/Shanghai")
HOST = "127.0.0.1"
PORT = 8788


def runtime_timeout_seconds() -> float:
    explicit = os.getenv("MAHJONG_AGENT_TIMEOUT_SECONDS")
    if explicit:
        return float(explicit)
    llm_timeout = os.getenv("MAHJONG_LLM_TIMEOUT_SECONDS")
    if llm_timeout:
        return float(llm_timeout) + 5.0
    return 3.0


def build_responder() -> AgentResponder:
    responder = AgentResponder(
        invite_limit=5,
        llm_resolver=OpenAICompatibleLLMResolver.from_env(),
    )
    for customer in [
        CustomerProfile(
            id="host",
            display_name="张哥",
            preferred_levels=[],
            play_preferences=[
                PlayPreference(
                    game_type="hangzhou_mahjong",
                    preferred_levels=["0.5"],
                    preferred_rulesets=["hangzhou_mahjong"],
                    preferred_variants=["caiqiao"],
                    preferred_play_options=["财敲"],
                ),
                PlayPreference(
                    game_type="sichuan_mahjong",
                    preferred_levels=["1-32"],
                    preferred_rulesets=["sichuan_mahjong"],
                    preferred_play_options=["换三张"],
                ),
            ],
            tags=["杭麻", "川麻", "换三张"],
            smoke_free_preference=True,
            usual_start_hours=[19, 20],
        ),
        CustomerProfile(
            id="amy",
            display_name="Amy",
            preferred_levels=["0.5"],
            tags=["无烟"],
            smoke_free_preference=True,
            usual_start_hours=[17, 18, 19],
            usual_weekdays=[0, 1, 2, 3, 4, 5],
            max_games_per_day=1,
            min_hours_between_games=6,
            invite_cooldown_hours=6,
        ),
        CustomerProfile(
            id="chen",
            display_name="陈姐",
            preferred_levels=["0.5", "1"],
            tags=["无烟", "熟人局"],
            smoke_free_preference=True,
            usual_start_hours=[17, 18],
            usual_weekdays=[1, 2, 3, 4, 5],
            max_games_per_day=1,
            min_hours_between_games=6,
            invite_cooldown_hours=6,
        ),
        CustomerProfile(
            id="ben",
            display_name="Ben",
            preferred_levels=["2"],
            tags=["可吸烟"],
            smoke_free_preference=False,
            usual_start_hours=[20, 21],
            max_games_per_day=2,
            min_hours_between_games=4,
            invite_cooldown_hours=4,
            fatigue_sensitivity=0.8,
        ),
        CustomerProfile(
            id="lin",
            display_name="林姐",
            preferred_levels=["1"],
            tags=["无烟"],
            smoke_free_preference=True,
            usual_start_hours=[19, 20],
            max_games_per_day=3,
            min_hours_between_games=2,
            invite_cooldown_hours=2,
            daily_invite_limit=5,
            fatigue_sensitivity=0.35,
        ),
    ]:
        responder.core.upsert_customer(customer)
    return responder


PROCESSOR = DurableAgentProcessor(
    AgentRuntime(
        build_responder(),
        RuntimeConfig(
            log_path=ROOT / "logs" / "chatroom_events.jsonl",
            timeout_seconds=runtime_timeout_seconds(),
        ),
    ),
    SQLiteDurableStore(ROOT / "data" / "chatroom.sqlite3"),
)
TRANSCRIPT: list[dict] = []
LAST_DECISION: dict | None = None


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>棋牌室 Agent 聊天室模拟器</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --line: #d9e1ec;
      --text: #172033;
      --muted: #64748b;
      --accent: #176b87;
      --accent-strong: #0f5369;
      --ok: #1f7a4c;
      --warn: #a15c05;
      --danger: #a93434;
      --agent: #eef7f8;
      --user: #eaf1ff;
      --shadow: 0 8px 24px rgba(20, 34, 55, 0.08);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }

    header {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }

    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }

    .status {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      white-space: nowrap;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--ok);
    }

    main {
      height: calc(100vh - 64px);
      display: grid;
      grid-template-columns: minmax(420px, 1fr) 430px;
      gap: 16px;
      padding: 16px;
    }

    .panel {
      min-width: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .chat {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: 0;
    }

    .toolbar {
      display: grid;
      grid-template-columns: 150px 120px 1fr auto;
      gap: 10px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }

    select,
    input,
    button {
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }

    select,
    input {
      padding: 0 10px;
      min-width: 0;
    }

    button {
      padding: 0 12px;
      cursor: pointer;
      font-weight: 650;
    }

    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }

    button.primary:hover {
      background: var(--accent-strong);
    }

    button.ghost {
      background: #f8fafc;
      color: var(--text);
    }

    .presets {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }

    .preset {
      height: 32px;
      padding: 0 10px;
      font-size: 13px;
      font-weight: 600;
      background: #f8fafc;
    }

    .timeline {
      min-height: 0;
      overflow: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .message {
      max-width: 78%;
      display: grid;
      gap: 4px;
    }

    .message.user {
      align-self: flex-end;
    }

    .message.agent,
    .message.system {
      align-self: flex-start;
    }

    .meta {
      color: var(--muted);
      font-size: 12px;
    }

    .bubble {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      line-height: 1.5;
      word-break: break-word;
    }

    .user .bubble {
      background: var(--user);
    }

    .agent .bubble {
      background: var(--agent);
    }

    .system .bubble {
      background: #f8fafc;
      color: var(--muted);
      font-size: 13px;
    }

    .side {
      min-height: 0;
      overflow: auto;
      display: grid;
      gap: 12px;
      align-content: start;
    }

    .section {
      padding: 14px;
    }

    .section h2 {
      margin: 0 0 10px;
      font-size: 14px;
      letter-spacing: 0;
    }

    .kv {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 8px 10px;
      color: var(--muted);
    }

    .kv strong {
      color: var(--text);
      font-weight: 650;
      overflow-wrap: anywhere;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      height: 24px;
      border-radius: 999px;
      padding: 0 8px;
      font-size: 12px;
      font-weight: 700;
      background: #e6f3f5;
      color: var(--accent-strong);
    }

    .pill.warn {
      background: #fff2df;
      color: var(--warn);
    }

    .pill.danger {
      background: #ffe8e8;
      color: var(--danger);
    }

    .list {
      display: grid;
      gap: 8px;
    }

    .item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }

    .item-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 6px;
      font-weight: 700;
    }

    .small {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }

    .draft {
      margin-top: 6px;
      color: var(--text);
      line-height: 1.5;
    }

    .empty {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }

    .footer {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 12px;
      border-top: 1px solid var(--line);
    }

    @media (max-width: 960px) {
      main {
        height: auto;
        grid-template-columns: 1fr;
      }

      .chat {
        min-height: 620px;
      }

      .toolbar {
        grid-template-columns: 1fr;
      }

      .message {
        max-width: 92%;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>棋牌室 Agent 聊天室模拟器</h1>
    <div class="status"><span class="dot"></span><span id="health">本地运行中</span></div>
  </header>
  <main>
    <section class="panel chat">
      <div>
        <div class="toolbar">
          <select id="sender">
            <option value="host|张哥">张哥</option>
            <option value="amy|Amy">Amy</option>
            <option value="chen|陈姐">陈姐</option>
            <option value="ben|Ben">Ben</option>
            <option value="lin|林姐">林姐</option>
            <option value="passerby|路人">路人</option>
          </select>
          <select id="messageType">
            <option value="text">文字</option>
            <option value="audio">语音</option>
            <option value="image">图片/OCR</option>
            <option value="sticker">表情包</option>
          </select>
          <input id="text" autocomplete="off" placeholder="输入群聊消息，例如：今晚5点 0.5 三缺一 无烟">
          <button class="primary" id="send">发送</button>
        </div>
        <div class="presets">
          <button class="preset" data-sender="host|张哥" data-text="今晚5点 0.5 三缺一 无烟 打四小时">清晰三缺一</button>
          <button class="preset" data-sender="host|张哥" data-text="川麻216三等一">川麻三等一</button>
          <button class="preset" data-sender="host|张哥" data-text="cq371 0.5 19.30 无烟">财敲19.30</button>
          <button class="preset" data-sender="host|张哥" data-text="今晚7点 川麻1-32换三张 371">川麻换三张</button>
          <button class="preset" data-sender="passerby|路人" data-text="今天下班有人打麻将吗">潜客文字</button>
          <button class="preset" data-sender="passerby|路人" data-kind="audio" data-text="下班想搓一把，有局吗">语音意向</button>
          <button class="preset" data-sender="passerby|路人" data-kind="image" data-text="群截图：今晚7点 0.5 三缺一 无烟">图片 OCR</button>
          <button class="preset" data-sender="passerby|路人" data-kind="sticker" data-text="麻将表情包：🀄 约吗">表情包</button>
          <button class="preset" data-sender="host|张哥" data-text="0.5 5点开 371 无烟">模糊五点</button>
          <button class="preset" data-sender="amy|Amy" data-text="我来">Amy 报名</button>
          <button class="preset" data-sender="host|张哥" data-text="组好了不用找了">发起人组好</button>
          <button class="preset" data-sender="passerby|路人" data-text="今天路上有点堵">无关闲聊</button>
          <button class="preset" data-sender="host|张哥" data-text="这桌输赢结算你帮我代收一下">敏感转人工</button>
        </div>
      </div>
      <div class="timeline" id="timeline"></div>
      <div class="footer">
        <button class="ghost" id="reset">重置模拟器</button>
        <span class="empty">所有消息都进入同一个模拟群：group_main</span>
      </div>
    </section>

    <aside class="side">
      <section class="panel section">
        <h2>本轮决策</h2>
        <div id="decision" class="empty">还没有消息。</div>
      </section>
      <section class="panel section">
        <h2>运行监控</h2>
        <div id="monitor" class="empty">暂无指标。</div>
      </section>
      <section class="panel section">
        <h2>草稿</h2>
        <div id="drafts" class="empty">暂无草稿。</div>
      </section>
      <section class="panel section">
        <h2>组局状态</h2>
        <div id="games" class="empty">暂无组局。</div>
      </section>
      <section class="panel section">
        <h2>邀约状态</h2>
        <div id="invitations" class="empty">暂无邀约。</div>
      </section>
      <section class="panel section">
        <h2>客户画像</h2>
        <div id="customers" class="empty">暂无客户。</div>
      </section>
    </aside>
  </main>

  <script>
    const timeline = document.querySelector("#timeline");
    const sender = document.querySelector("#sender");
    const messageType = document.querySelector("#messageType");
    const input = document.querySelector("#text");
    const send = document.querySelector("#send");
    const reset = document.querySelector("#reset");

    function splitSender(value) {
      const [senderId, senderName] = value.split("|");
      return { senderId, senderName };
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function pill(text, kind = "") {
      return `<span class="pill ${kind}">${escapeHtml(text)}</span>`;
    }

    function gameTypeLabel(value) {
      if (value === "sichuan_mahjong") return "川麻";
      if (value === "chongqing_mahjong") return "重庆麻将";
      if (value === "hangzhou_mahjong") return "杭麻";
      if (value === "hongzhong_mahjong") return "红中麻将";
      if (value === "zhuoji_mahjong") return "捉鸡麻将";
      if (value === "hunan_mahjong") return "湖南麻将";
      return "麻将";
    }

    function variantLabel(value) {
      if (value === "caiqiao") return "财敲";
      if (value === "yaoji") return "幺鸡";
      if (value === "suji") return "素鸡";
      if (value === "yaoji_47") return "幺鸡47";
      return value || "";
    }

    function stakeLabel(game) {
      if (!game.level) return "档位未定";
      if (game.base_score !== null && game.base_score !== undefined && game.cap_score !== null && game.cap_score !== undefined) {
        return `${game.level}档(底注${game.base_score}/封顶${game.cap_score})`;
      }
      return `${game.level}档`;
    }

    function playPreferenceText(item) {
      if (!item.play_preferences || !item.play_preferences.length) return "";
      return item.play_preferences.map((pref) => {
        const labels = [gameTypeLabel(pref.game_type)];
        if (pref.preferred_variants?.length) labels.push(pref.preferred_variants.map(variantLabel).join("/"));
        if (pref.preferred_levels?.length) labels.push(pref.preferred_levels.join("/"));
        if (pref.preferred_play_options?.length) labels.push(pref.preferred_play_options.join("/"));
        return labels.filter(Boolean).join(" ");
      }).join("；");
    }

    function payloadForMessage(type, text) {
      if (type === "audio") {
        return { text: "[语音]", metadata: { message_type: "audio", audio_transcript: text } };
      }
      if (type === "image") {
        return { text: "[图片]", metadata: { message_type: "image", image_ocr_text: text } };
      }
      if (type === "sticker") {
        return { text: "[表情包]", metadata: { message_type: "sticker", sticker_description: text } };
      }
      return { text, metadata: { message_type: "text" } };
    }

    function renderTranscript(items) {
      if (!items.length) {
        timeline.innerHTML = `<div class="message system"><div class="bubble">先点一个预设，或者自己输入一句群聊消息。</div></div>`;
        return;
      }
      timeline.innerHTML = items.map((item) => {
        const who = item.kind === "agent" ? "Agent" : item.sender_name;
        const role = item.kind === "agent" ? "agent" : item.kind === "system" ? "system" : "user";
        return `
          <div class="message ${role}">
            <div class="meta">${escapeHtml(who)} · ${escapeHtml(item.time)}</div>
            <div class="bubble">${escapeHtml(item.text)}</div>
          </div>
        `;
      }).join("");
      timeline.scrollTop = timeline.scrollHeight;
    }

    function renderDecision(decision) {
      const el = document.querySelector("#decision");
      if (!decision) {
        el.className = "empty";
        el.innerHTML = "还没有消息。";
        return;
      }
      const kind = decision.action === "human_review" ? "danger" : decision.needs_human_review ? "warn" : "";
      const notes = decision.notes && decision.notes.length ? decision.notes.join("；") : "无";
      el.className = "kv";
      el.innerHTML = `
        <span>动作</span><strong>${pill(decision.action, kind)}</strong>
        <span>置信度</span><strong>${escapeHtml(decision.confidence)}</strong>
        <span>是否回复</span><strong>${decision.should_reply ? "是" : "否"}</strong>
        <span>人工确认</span><strong>${decision.needs_human_review ? "需要" : "不需要"}</strong>
        <span>回复内容</span><strong>${escapeHtml(decision.reply_text || "静默")}</strong>
        <span>证据/备注</span><strong>${escapeHtml(notes)}</strong>
        <span>耗时</span><strong>${escapeHtml(decision.runtime?.latency_ms ?? "-")} ms</strong>
        <span>异常</span><strong>${escapeHtml(decision.runtime?.error || "无")}</strong>
      `;
    }

    function renderMonitor(runtime) {
      const el = document.querySelector("#monitor");
      if (!runtime || !runtime.metrics) {
        el.className = "empty";
        el.innerHTML = "暂无指标。";
        return;
      }
      const metrics = runtime.metrics;
      const llm = runtime.llm || {};
      const durable = runtime.durable || {};
      const durableCounts = durable.counts || {};
      const messageStatuses = durable.message_statuses || {};
      const contextCount = Object.keys(runtime.contexts || {}).length;
      el.className = "kv";
      el.innerHTML = `
        <span>消息总数</span><strong>${escapeHtml(metrics.total_messages)}</strong>
        <span>异常次数</span><strong>${escapeHtml(metrics.total_errors)}</strong>
        <span>超时次数</span><strong>${escapeHtml(metrics.total_timeouts)}</strong>
        <span>转人工</span><strong>${escapeHtml(metrics.total_human_reviews)}</strong>
        <span>静默</span><strong>${escapeHtml(metrics.total_ignored)}</strong>
        <span>平均耗时</span><strong>${escapeHtml(metrics.latency_ms_avg)} ms</strong>
        <span>最大耗时</span><strong>${escapeHtml(metrics.latency_ms_max)} ms</strong>
        <span>上下文数</span><strong>${escapeHtml(contextCount)}</strong>
        <span>LLM</span><strong>${llm.enabled ? `启用 · ${escapeHtml(llm.model || "-")}` : "未启用"}</strong>
        <span>持久消息</span><strong>${escapeHtml(durableCounts.inbound_messages ?? 0)}</strong>
        <span>审计事件</span><strong>${escapeHtml(durableCounts.audit_events ?? 0)}</strong>
        <span>Outbox</span><strong>${escapeHtml(durableCounts.outbox_events ?? 0)}</strong>
        <span>已处理</span><strong>${escapeHtml(messageStatuses.processed ?? 0)}</strong>
      `;
    }

    function renderDrafts(decision) {
      const el = document.querySelector("#drafts");
      if (!decision || (!decision.draft_group_post && !decision.invitation_drafts.length)) {
        el.className = "empty";
        el.innerHTML = "暂无草稿。";
        return;
      }
      el.className = "list";
      const group = decision.draft_group_post
        ? `<div class="item"><div class="item-title">群发草稿</div><div class="draft">${escapeHtml(decision.draft_group_post)}</div></div>`
        : "";
      const invites = decision.invitation_drafts.map((item) => `
        <div class="item">
          <div class="item-title"><span>${escapeHtml(item.customer_name)}</span>${pill(item.status)}</div>
          <div class="small">${escapeHtml(item.customer_id)} · ${escapeHtml(item.game_id)}</div>
          <div class="draft">${escapeHtml(item.message_text)}</div>
        </div>
      `).join("");
      el.innerHTML = group + invites;
    }

    function renderGames(games) {
      const el = document.querySelector("#games");
      if (!games.length) {
        el.className = "empty";
        el.innerHTML = "暂无组局。";
        return;
      }
      el.className = "list";
      el.innerHTML = games.map((game) => `
        <div class="item">
          <div class="item-title"><span>${escapeHtml(game.organizer_name)}</span>${pill(game.status)}</div>
          <div class="small">${escapeHtml(game.id)}</div>
          <div class="draft">
            ${escapeHtml(game.start_at || "时间未定")} · ${escapeHtml(gameTypeLabel(game.game_type))}${game.variant ? ` · ${escapeHtml(variantLabel(game.variant))}` : ""} · ${escapeHtml(stakeLabel(game))} ·
            ${escapeHtml(game.current_player_count ?? "?")}缺${escapeHtml(game.missing_count ?? "?")} ·
            剩余 ${escapeHtml(game.open_slots ?? "?")} 位
          </div>
          <div class="small">${escapeHtml(game.play_options.join("、") || "无玩法选项")}</div>
          <div class="small">${escapeHtml(game.rules.join("、") || "无特殊规则")}</div>
        </div>
      `).join("");
    }

    function renderInvitations(invitations) {
      const el = document.querySelector("#invitations");
      if (!invitations.length) {
        el.className = "empty";
        el.innerHTML = "暂无邀约。";
        return;
      }
      el.className = "list";
      el.innerHTML = invitations.map((item) => `
        <div class="item">
          <div class="item-title"><span>${escapeHtml(item.customer_name)}</span>${pill(item.status)}</div>
          <div class="small">${escapeHtml(item.id)} · ${escapeHtml(item.game_id)}</div>
          <div class="draft">${escapeHtml(item.message_text)}</div>
        </div>
      `).join("");
    }

    function renderCustomers(customers) {
      const el = document.querySelector("#customers");
      if (!customers.length) {
        el.className = "empty";
        el.innerHTML = "暂无客户。";
        return;
      }
      el.className = "list";
      el.innerHTML = customers.map((item) => `
        <div class="item">
          <div class="item-title"><span>${escapeHtml(item.display_name)}</span><span class="small">${escapeHtml(item.id)}</span></div>
          <div class="small">常打：${escapeHtml(item.preferred_levels.join("、") || "未记录")} · 标签：${escapeHtml(item.tags.join("、") || "无")}</div>
          <div class="small">${escapeHtml(playPreferenceText(item) || "未记录细分玩法偏好")}</div>
          <div class="small">频率：每日最多 ${escapeHtml(item.max_games_per_day)} 场 · 间隔 ${escapeHtml(item.min_hours_between_games)}h · 邀约冷却 ${escapeHtml(item.invite_cooldown_hours)}h</div>
          <div class="small">${item.metadata?.last_lead_modalities ? `最近意向：${escapeHtml(item.metadata.last_lead_modalities.join("、"))} · score ${escapeHtml(item.metadata.last_lead_score ?? "-")}` : ""}</div>
        </div>
      `).join("");
    }

    function renderState(state) {
      renderTranscript(state.transcript);
      renderDecision(state.last_decision);
      renderMonitor(state.runtime);
      renderDrafts(state.last_decision);
      renderGames(state.games);
      renderInvitations(state.invitations);
      renderCustomers(state.customers);
    }

    async function api(path, payload) {
      const options = payload
        ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }
        : {};
      const response = await fetch(path, options);
      if (!response.ok) {
        throw new Error(await response.text());
      }
      return response.json();
    }

    async function refresh() {
      renderState(await api("/api/state"));
    }

    async function sendMessage() {
      const text = input.value.trim();
      if (!text) return;
      const { senderId, senderName } = splitSender(sender.value);
      const payload = payloadForMessage(messageType.value, text);
      input.value = "";
      const state = await api("/api/message", {
        text: payload.text,
        metadata: payload.metadata,
        sender_id: senderId,
        sender_name: senderName,
        channel_id: "group_main",
        channel_type: "wechat_group",
        now: "2026-06-16T12:00:00+08:00"
      });
      renderState(state);
      input.focus();
    }

    send.addEventListener("click", sendMessage);
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") sendMessage();
    });
    reset.addEventListener("click", async () => {
      renderState(await api("/api/reset", {}));
      input.focus();
    });
    document.querySelectorAll(".preset").forEach((button) => {
      button.addEventListener("click", async () => {
        sender.value = button.dataset.sender;
        messageType.value = button.dataset.kind || "text";
        input.value = button.dataset.text;
        await sendMessage();
      });
    });

    refresh().catch((error) => {
      document.querySelector("#health").textContent = "连接失败";
      console.error(error);
    });
  </script>
</body>
</html>
"""


class ChatroomHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/chatroom":
            self._html(HTML)
            return
        if self.path == "/api/state":
            self._json(current_state())
            return
        if self.path == "/health":
            snapshot = PROCESSOR.snapshot()
            self._json({"ok": True, "runtime": snapshot["metrics"], "durable": snapshot["durable"], "llm": llm_status()})
            return
        self._json({"error": "not_found"}, status=404)

    def do_POST(self) -> None:
        global PROCESSOR, TRANSCRIPT, LAST_DECISION

        if self.path == "/api/reset":
            PROCESSOR.shutdown()
            store = SQLiteDurableStore(ROOT / "data" / "chatroom.sqlite3")
            store.reset_all()
            PROCESSOR = DurableAgentProcessor(
                AgentRuntime(
                    build_responder(),
                    RuntimeConfig(
                        log_path=ROOT / "logs" / "chatroom_events.jsonl",
                        timeout_seconds=runtime_timeout_seconds(),
                    ),
                ),
                store,
            )
            TRANSCRIPT = []
            LAST_DECISION = None
            self._json(current_state())
            return

        if self.path != "/api/message":
            self._json({"error": "not_found"}, status=404)
            return

        try:
            body = self._read_json()
            message = Message(
                text=str(body["text"]),
                sender_id=str(body.get("sender_id", "unknown")),
                sender_name=str(body.get("sender_name", body.get("sender_id", "unknown"))),
                channel_id=str(body.get("channel_id", "group_main")),
                channel_type=ChannelType(str(body.get("channel_type", ChannelType.WECHAT_GROUP.value))),
                metadata=dict(body.get("metadata") or {}),
            )
            now = parse_now(body.get("now"))
            envelope = IncomingEnvelope(
                message=message,
                tenant_id=str(body.get("tenant_id", "default")),
                source_message_id=str(body["source_message_id"]) if body.get("source_message_id") else message.id,
                sequence=int(body["sequence"]) if body.get("sequence") is not None else None,
            )
            result = PROCESSOR.process(envelope, now=now)
            decision = result.runtime_result.decision if result.runtime_result else None
        except Exception as exc:
            self._json({"error": type(exc).__name__, "message": str(exc)}, status=400)
            return

        TRANSCRIPT.append(
            {
                "kind": "user",
                "sender_id": message.sender_id,
                "sender_name": message.sender_name,
                "text": display_message_text(message),
                "time": datetime.now(TZ).strftime("%H:%M:%S"),
            }
        )
        if decision and decision.should_reply and decision.reply_text:
            TRANSCRIPT.append(
                {
                    "kind": "agent",
                    "sender_id": "agent",
                    "sender_name": "Agent",
                    "text": decision.reply_text,
                    "time": datetime.now(TZ).strftime("%H:%M:%S"),
                }
            )
        else:
            TRANSCRIPT.append(
                {
                    "kind": "system",
                    "sender_id": "system",
                    "sender_name": "系统",
                    "text": (
                        f"Agent 静默：{'; '.join(decision.notes) if decision.notes else decision.action.value}"
                        if decision
                        else "消息已持久化，正在等待前序消息。"
                    ),
                    "time": datetime.now(TZ).strftime("%H:%M:%S"),
                }
            )
        LAST_DECISION = result.to_dict()
        self._json(current_state())

    def log_message(self, format: str, *args) -> None:
        print(format % args)

    def _read_json(self) -> dict:
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def parse_now(value: object) -> datetime | None:
    if not value:
        return None
    if not isinstance(value, str):
        raise ValueError("now must be an ISO datetime string")
    return datetime.fromisoformat(value).astimezone(TZ)


def display_message_text(message: Message) -> str:
    message_type = str(message.metadata.get("message_type") or "text")
    if message_type == "audio":
        return f"[语音] {message.metadata.get('audio_transcript') or message.metadata.get('asr_text') or message.text}"
    if message_type == "image":
        return (
            f"[图片] "
            f"{message.metadata.get('image_ocr_text') or message.metadata.get('ocr_text') or message.metadata.get('image_description') or message.metadata.get('vision_description') or message.text}"
        )
    if message_type == "sticker":
        return (
            f"[表情包] "
            f"{message.metadata.get('sticker_description') or message.metadata.get('sticker_text') or message.metadata.get('emoji_text') or message.text}"
        )
    return message.text


def current_state() -> dict:
    store = PROCESSOR.runtime.responder.core.store
    runtime_snapshot = PROCESSOR.snapshot()
    runtime_snapshot["llm"] = llm_status()
    return {
        "transcript": TRANSCRIPT,
        "last_decision": LAST_DECISION,
        "runtime": runtime_snapshot,
        "games": [
            {
                "id": game.id,
                "status": game.status.value,
                "organizer_id": game.organizer_id,
                "organizer_name": game.organizer_name,
                "channel_id": game.channel_id,
                "game_type": game.game_type,
                "ruleset": game.ruleset,
                "variant": game.variant,
                "level": game.level,
                "base_score": game.base_score,
                "cap_score": game.cap_score,
                "start_at": game.start_at.strftime("%m-%d %H:%M") if game.start_at else None,
                "duration_hours": game.duration_hours,
                "play_options": game.play_options,
                "current_player_count": game.current_player_count,
                "missing_count": game.missing_count,
                "open_slots": game.open_slots,
                "rules": game.rules,
                "ambiguities": game.ambiguities,
                "reserved_customer_ids": game.reserved_customer_ids,
            }
            for game in sorted(store.games.values(), key=lambda item: item.created_at, reverse=True)
        ],
        "invitations": [
            {
                "id": invitation.id,
                "game_id": invitation.game_id,
                "customer_id": invitation.customer_id,
                "customer_name": invitation.customer_name,
                "status": invitation.status.value,
                "message_text": invitation.message_text,
            }
            for invitation in sorted(store.invitations.values(), key=lambda item: item.created_at, reverse=True)
        ],
        "customers": [
            {
                "id": customer.id,
                "display_name": customer.display_name,
                "preferred_levels": customer.preferred_levels,
                "play_preferences": [
                    {
                        "game_type": preference.game_type,
                        "preferred_levels": preference.preferred_levels,
                        "preferred_rulesets": preference.preferred_rulesets,
                        "preferred_variants": preference.preferred_variants,
                        "preferred_play_options": preference.preferred_play_options,
                        "avoid_play_options": preference.avoid_play_options,
                    }
                    for preference in customer.play_preferences
                ],
                "tags": customer.tags,
                "smoke_free_preference": customer.smoke_free_preference,
                "usual_start_hours": customer.usual_start_hours,
                "max_games_per_day": customer.max_games_per_day,
                "min_hours_between_games": customer.min_hours_between_games,
                "invite_cooldown_hours": customer.invite_cooldown_hours,
                "daily_invite_limit": customer.daily_invite_limit,
                "fatigue_sensitivity": customer.fatigue_sensitivity,
                "metadata": customer.metadata,
            }
            for customer in sorted(store.customers.values(), key=lambda item: item.display_name)
        ],
    }


def llm_status() -> dict:
    resolver = PROCESSOR.runtime.responder.llm_resolver
    config = getattr(resolver, "config", None)
    budget_manager = getattr(resolver, "budget_manager", None)
    return {
        "enabled": resolver is not None,
        "model": getattr(config, "model", None),
        "base_url": getattr(config, "base_url", None),
        "budget": budget_manager.snapshot() if budget_manager else None,
    }


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), ChatroomHandler)
    print(f"Mahjong chatroom simulator listening on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMahjong chatroom simulator stopped.")
    finally:
        PROCESSOR.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
