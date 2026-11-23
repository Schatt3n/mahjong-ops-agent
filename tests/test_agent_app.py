from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import pytest

from mahjong_agent_runtime import (
    AgentRuntime,
    CustomerProfile,
    InMemoryTraceRecorder,
    SQLiteAgentStore,
    StaticAgentClient,
    ToolResult,
    ToolGateway,
    UserMessage,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "agent_runtime_app.py"
MAIN_SCRIPT = ROOT / "scripts" / "run_agent_app.py"


def load_app_module():
    spec = importlib.util.spec_from_file_location("agent_runtime_app_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def local_agent_http(tmp_path, monkeypatch):
    """用真实 HTTP 与 SQLite 隔离验证入口；所有模型输出由本地测试客户端提供。"""
    from http.server import ThreadingHTTPServer
    from types import SimpleNamespace

    app = load_app_module()
    store_path = tmp_path / "http_runtime.sqlite3"
    store = SQLiteAgentStore(store_path)
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient([])
    runtime = AgentRuntime(
        llm_client=client,
        store=store,
        tool_gateway=ToolGateway(store=store, trace_recorder=trace),
        trace_recorder=trace,
    )
    monkeypatch.setattr(app, "_API_TOKEN", "local-channel-test-token")
    monkeypatch.setattr(app, "get_runtime", lambda: runtime)
    monkeypatch.setattr(app, "WECHATY_RAW_LOG_PATH", tmp_path / "wechaty_raw.jsonl")
    monkeypatch.setattr(app, "AGENT_WECHAT_RAW_LOG_PATH", tmp_path / "agent_wechat_raw.jsonl")
    monkeypatch.setattr(app, "TRACE_PATH", tmp_path / "trace.jsonl")
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("MAHJONG_AGENT_WECHAT_OUTBOUND_ALLOWLIST_PATH", raising=False)
    monkeypatch.setenv("MAHJONG_API_INPUT_AGGREGATION_ENABLED", "false")
    server = ThreadingHTTPServer(("127.0.0.1", 0), app.AgentRuntimeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(
            app=app,
            runtime=runtime,
            client=client,
            store_path=store_path,
            server=server,
            base=f"http://127.0.0.1:{server.server_port}",
            headers={
                "Authorization": "Bearer local-channel-test-token",
                "Content-Type": "application/json",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("channel", ["hermes", "astrbot"])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_retired_channel_http_endpoints_return_404_without_side_effects(
    local_agent_http, monkeypatch, tmp_path, channel, method,
) -> None:
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    context = local_agent_http
    raw_writes = []

    def record_unexpected_raw_message(**kwargs):
        raw_writes.append(kwargs)
        raise AssertionError("退役入口不得调用原始消息持久化")

    monkeypatch.setattr(context.app, "record_channel_raw_message", record_unexpected_raw_message)
    request = Request(
        f"{context.base}/api/channels/{channel}/raw",
        method=method,
        data=b'{"message_id":"retired_message","text":"hello"}' if method == "POST" else None,
        headers=context.headers,
    )
    with pytest.raises(HTTPError) as exc:
        urlopen(request, timeout=5)
    assert exc.value.code == 404
    # 退役路径沿用 HTTP 服务器的未知路由响应，不额外保留兼容端点。
    assert "Error code: 404" in exc.value.read().decode("utf-8")
    assert raw_writes == []
    assert context.client.calls == []
    assert context.runtime.trace_recorder.events == []
    assert context.runtime.store.list_channel_observations() == []
    assert list(tmp_path.glob("*.jsonl")) == []



@pytest.fixture
def local_wechat_bridge(local_agent_http, monkeypatch):
    """在真实回环端口模拟适配器，只提供合成收件人和发送回执。"""
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from types import SimpleNamespace

    not_found = (404, {"error": "not found"})
    state = SimpleNamespace(calls=[], receipts={}, reject="", uncertain=False,
                            contact_refresh=not_found, contact_policy=not_found,
                            permission_roster=not_found, group_permission=not_found,
                            active_conversations=not_found, conversation=not_found)

    class StubWechatBridge(BaseHTTPRequestHandler):
        def reply(self, status, data):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            state.calls.append((self.command, self.path, self.headers.get("X-Mahjong-Agent-Token"), None))
            if self.path == "/health":
                self.reply(200, {"ok": True, "adapter": "agent-wechat", "send_channel_enabled": False})
                return
            if self.path.partition("?")[0] == "/contact-policy":
                self.reply(*state.contact_policy)
                return
            if self.path.partition("?")[0] == "/permission-roster":
                self.reply(*state.permission_roster)
                return
            if self.path == "/console-conversations":
                self.reply(*state.active_conversations)
                return
            if self.path.partition("?")[0] == "/console-conversation":
                self.reply(*state.conversation)
                return
            self.reply(200, {"ok": True, "candidates": [
                {"id": "wxid_fixture", "kind": "direct", "displayName": "合成备注", "backend_only": "debug"},
            ]})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.calls.append((self.command, self.path, self.headers.get("X-Mahjong-Agent-Token"), payload))
            if self.path in ("/contacts/refresh", "/contact-policy/refresh"):
                self.reply(*state.contact_refresh)
                return
            if self.path == "/contact-policy":
                self.reply(*state.contact_policy)
                return
            if self.path == "/group-permission":
                self.reply(*state.group_permission)
                return
            if self.path != "/send":
                self.reply(200, {"ok": True, "enabled": payload["enabled"]})
                return
            if state.reject:
                self.reply(409, {"ok": False, "error": state.reject})
                return
            if state.uncertain:
                self.reply(200, {"ok": False, "state": "needs_review", "thread_id": payload["to"]})
                return
            anchor = payload["source_trace_id"]
            indexed = anchor in state.receipts
            receipt = state.receipts.setdefault(anchor, f"server-fixture-{len(state.receipts) + 1}")
            self.reply(200, {"ok": True, "state": "accepted", "thread_id": payload["to"],
                             "server_message_id": receipt, "local_message_id": "local-fixture" if indexed else ""})

    server = ThreadingHTTPServer(("127.0.0.1", 0), StubWechatBridge)
    monkeypatch.setenv("MAHJONG_WECHATY_OUTBOUND_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("MAHJONG_WECHATY_OUTBOUND_TOKEN", "fixture-bridge-control-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_owner_console_lists_recent_or_scheduled_whitelisted_wechat_conversations(
    local_agent_http, local_wechat_bridge, monkeypatch,
):
    """两小时或未完成定时任务取并集，任务不能使未授权、错误账户或已完成会话入列。"""
    from datetime import timedelta
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from mahjong_agent_runtime.models import ScheduledAgentTask, now

    context = local_agent_http
    store = context.runtime.store
    at = now()
    old = (at - timedelta(hours=2, minutes=1)).isoformat()
    recent = (at - timedelta(hours=1, minutes=59)).isoformat()
    future = (at + timedelta(minutes=1)).isoformat()
    local_wechat_bridge.active_conversations = (200, {
        "ok": True, "account_id": "fixture-account", "items": [
            {"id": "wxid_recent", "kind": "direct", "displayName": "近期私聊", "last_at": recent, "token": "hidden"},
            {"id": "wxid_inactive", "kind": "direct", "displayName": "已完成任务", "last_at": old},
            {"id": "wxid_other_account", "kind": "direct", "displayName": "账户不同", "last_at": old},
            {"id": "wxid_future", "kind": "direct", "displayName": "未来时间", "last_at": future},
            {"id": "123@chatroom", "kind": "group", "displayName": "待发看板", "last_at": None},
            {"id": "456@chatroom", "kind": "direct", "displayName": "错误会话类型", "last_at": recent},
        ],
    })
    for thread_id, account in (("wxid_recent", "fixture-account"), ("wxid_inactive", "fixture-account"),
                               ("wxid_other_account", "different-account"), ("wxid_unlisted", None)):
        task = ScheduledAgentTask(
            task_id=f"task-{thread_id}", task_type="conversation_follow_up", aggregate_type="conversation_goal",
            aggregate_id=f"goal-{thread_id}", conversation_id=f"wechaty:contact:{thread_id}",
            subject_id=thread_id, subject_name="合成用户", due_at=at, idempotency_key=f"job-{thread_id}",
            payload={"account_id": account, "goal": "不应该被投影的任务详情"},
        )
        store.save_follow_up_task(task, trace_id="owner-console")
    finished = store.claim_scheduled_task("task-wxid_inactive", at=at)
    store.complete_scheduled_task(finished.task_id, trace_id="owner-console",
                                  expected_attempts=finished.attempts, expected_lease_until=finished.lease_until)
    task, _ = store.ensure_group_board_publish_task(room_id="123@chatroom", due_at=at, trace_id="owner-console")
    claimed = store.claim_scheduled_task(task.task_id, at=at)
    assert claimed is not None

    endpoint = f"{context.base}/api/wechat/conversations"
    for headers, code in (({}, 401), ({**context.headers, "Host": "outside.invalid"}, 403)):
        with pytest.raises(HTTPError) as error:
            urlopen(Request(endpoint, headers=headers), timeout=5)
        assert error.value.code == code
    assert local_wechat_bridge.calls == []
    monkeypatch.delenv("MAHJONG_WECHATY_OUTBOUND_TOKEN")
    with urlopen(Request(endpoint, headers=context.headers), timeout=5) as response:
        assert response.headers["Cache-Control"] == "no-store"
        payload = context.app.json.load(response)
    assert payload == {"ok": True, "items": [
        {"id": "wxid_recent", "kind": "direct", "displayName": "近期私聊", "last_at": recent, "pending_tasks": 1},
        {"id": "123@chatroom", "kind": "group", "displayName": "待发看板", "last_at": None, "pending_tasks": 1},
    ]}
    assert local_wechat_bridge.calls == [("GET", "/console-conversations", "local-channel-test-token", None)]
    assert context.client.calls == []


def test_owner_console_chat_history_relays_exact_target_and_minimal_text(local_agent_http, local_wechat_bridge):
    """仅同源精确会话可读取聊天页，内部送达证明和错误的消息身份不交给浏览器。"""
    from urllib.error import HTTPError
    from urllib.parse import parse_qs, urlencode, urlsplit
    from urllib.request import Request, urlopen

    context = local_agent_http
    endpoint = f"{context.base}/api/wechat/conversation"
    valid = {"thread_id": "wxid_fixture", "kind": "direct", "cursor": "native+cursor=", "limit": "3"}
    for params in ({**valid, "thread_id": "123@chatroom"}, {**valid, "cursor": "\x00"}, {**valid, "limit": "61"}):
        with pytest.raises(HTTPError) as error:
            urlopen(Request(f"{endpoint}?{urlencode(params)}", headers=context.headers), timeout=5)
        assert error.value.code == 400
    assert local_wechat_bridge.calls == []
    stamp = "2026-11-19T15:00:00+08:00"
    incoming = {"id": "local-1", "at": stamp, "direction": "inbound", "author": "contact",
                "speaker": "群成员", "text": "合成文本"}
    outgoing = {"id": "local-2", "at": stamp, "direction": "outbound", "author": "agent",
                "speaker": None, "text": "已经确认发送的回复"}
    target = {"id": "wxid_group_member", "kind": "direct", "displayName": "精确好友"}
    local_wechat_bridge.conversation = (200, {
        "ok": True, "id": valid["thread_id"], "kind": "direct", "account_id": "hidden-account",
        "items": [{**incoming, "rawPayload": {"private": "hidden"}, "direct_recipient": target},
                  {**outgoing, "server_id": "hidden-native-id"}, {**incoming, "id": "bad", "author": "agent"}],
        "nextCursor": "next+page=",
    })
    request = Request(f"{endpoint}?{urlencode(valid)}", headers=context.headers)
    with urlopen(request, timeout=5) as response:
        assert response.headers["Cache-Control"] == "no-store"
        assert context.app.json.load(response) == {"ok": True, "id": valid["thread_id"], "kind": "direct",
                                                   "items": [incoming, outgoing], "nextCursor": "next+page="}
    method, path, token, body = local_wechat_bridge.calls.pop()
    assert (method, urlsplit(path).path, token, body) == ("GET", "/console-conversation", "fixture-bridge-control-token", None)
    assert parse_qs(urlsplit(path).query) == {key: [value] for key, value in valid.items()}
    local_wechat_bridge.conversation[1]["id"] = "different-peer"
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=5)
    assert error.value.code == 502

    group = {**valid, "thread_id": "123@chatroom", "kind": "group", "limit": "4"}
    local_wechat_bridge.conversation = (200, {
        "ok": True, "id": group["thread_id"], "kind": "group", "nextCursor": None,
        "items": [
            {**incoming, "direct_recipient": {**target, "phone": "hidden"}},
            {**outgoing, "direct_recipient": target},
            {**incoming, "id": "local-3", "direct_recipient": {**target, "id": "another@chatroom"}},
        ],
    })
    request = Request(f"{endpoint}?{urlencode(group)}", headers=context.headers)
    with urlopen(request, timeout=5) as response:
        assert context.app.json.load(response)["items"] == [
            {**incoming, "direct_recipient": target}, outgoing, {**incoming, "id": "local-3"},
        ]
    assert context.client.calls == []


def test_wechat_console_uses_configured_authenticated_bridge_for_search_and_controls(
    local_agent_http, local_wechat_bridge,
) -> None:
    """页面同源鉴权，查询参数编码后到达配置端口，令牌与内部字段留在服务端。"""
    import json
    from urllib.error import HTTPError
    from urllib.parse import parse_qs, urlencode, urlsplit
    from urllib.request import Request, urlopen

    context = local_agent_http
    query = "李&张+#"
    route = f"/api/wechat/recipients?{urlencode({'query': query})}"
    with pytest.raises(HTTPError) as unauthorized:
        urlopen(f"{context.base}{route}", timeout=5)
    assert unauthorized.value.code == 401
    assert local_wechat_bridge.calls == []
    with urlopen(Request(f"{context.base}{route}", headers=context.headers), timeout=5) as response:
        body = json.load(response)
    assert body == {"ok": True, "candidates": [
        {"id": "wxid_fixture", "kind": "direct", "displayName": "合成备注",
         "remark": None, "nickname": None, "alias": None},
    ]}
    method, path, token, _ = local_wechat_bridge.calls.pop()
    assert (method, urlsplit(path).path, parse_qs(urlsplit(path).query)) == ("GET", "/resolve", {"query": [query]})
    assert token == "fixture-bridge-control-token"
    with urlopen(Request(f"{context.base}/api/wechat/health", headers=context.headers), timeout=5) as response:
        assert json.load(response)["send_channel_enabled"] is False
    for endpoint in ("send-channel", "auto-send"):
        req = Request(f"{context.base}/api/wechat/{endpoint}", headers=context.headers,
                      data=b'{"enabled": false, "path": "/send"}')
        with urlopen(req, timeout=5) as response:
            assert json.load(response) == {"ok": True, "enabled": False}
        assert local_wechat_bridge.calls[-1] == ("POST", f"/{endpoint}", token, {"enabled": False})


def test_wechat_console_refreshes_only_the_selected_private_contact(local_agent_http, local_wechat_bridge, monkeypatch):
    """刷新接口沿用控制面鉴权，核对读回身份并仅投影浏览器需要的资料。"""
    import json
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    context = local_agent_http
    route = f"{context.base}/api/wechat/recipients/refresh"
    payload = {"recipient_id": "wxid_fixture", "remark": "不得代写的备注", "path": "/send"}

    def refresh(body, headers=context.headers, *, endpoint=route):
        request = Request(endpoint, headers=headers, data=json.dumps(body).encode())
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.code, json.load(response)

    assert refresh(payload, headers={})[0] == 401
    for invalid in (None, "", "昵称 带空格", "123@chatroom", "12@im.chatroom", "..", "bad/id", "bad\x00id", "a" * 257):
        assert refresh({"recipient_id": invalid})[0] == 400
    assert local_wechat_bridge.calls == []

    assert refresh(payload)[0] == 501
    local_wechat_bridge.contact_refresh = (202, {"ok": False, "pending": True, "operation_id": "private-op"})
    status, pending = refresh(payload)
    assert status == 202 and pending["pending"] and "operation_id" not in pending

    recipient = {"id": "wxid_fixture", "kind": "direct", "displayName": "新备注", "remark": "新备注",
                 "nickname": "旧昵称", "alias": "account-alias"}
    result = {"ok": True, "projection": "partial", "recipient": {**recipient, "backend_only": "hidden"}}
    local_wechat_bridge.contact_refresh = (200, result)
    assert refresh(payload) == (200, {"ok": True, "projection": "partial", "recipient": recipient})
    result["recipient"]["id"] = "wxid_other"
    assert refresh(payload)[0] == 502
    assert local_wechat_bridge.calls == [
        ("POST", "/contacts/refresh", "fixture-bridge-control-token", {"recipient_id": "wxid_fixture"}),
    ] * 4
    monkeypatch.delenv("MAHJONG_WECHATY_OUTBOUND_TOKEN")
    local_wechat_bridge.contact_refresh = (404, {"error": "未找到可刷新的微信联系人。"})
    assert refresh(payload)[0] == 404
    assert local_wechat_bridge.calls[-1] == (
        "POST", "/contacts/refresh", "local-channel-test-token", {"recipient_id": "wxid_fixture"},
    )
    policy_refresh = f"{context.base}/api/wechat/contact-policy/refresh"
    calls_before = len(local_wechat_bridge.calls)
    assert refresh(payload, headers={}, endpoint=policy_refresh)[0] == 401
    assert refresh(payload, headers={**context.headers, "Host": "external.invalid"}, endpoint=policy_refresh)[0] == 403
    assert refresh(payload, headers={**context.headers, "Content-Type": "text/plain"}, endpoint=policy_refresh)[0] == 415
    assert len(local_wechat_bridge.calls) == calls_before
    local_wechat_bridge.contact_refresh = (200, {**result, "recipient": recipient})
    assert refresh(payload, endpoint=policy_refresh) == (200, {
        "ok": True, "projection": "partial", "recipient": recipient,
    })
    assert local_wechat_bridge.calls[-1] == (
        "POST", "/contact-policy/refresh", "local-channel-test-token", {"recipient_id": "wxid_fixture"},
    )
    assert context.runtime.store.recent_turns("wechaty:contact:wxid_fixture") == []
    assert context.client.calls == []


def test_wechat_contact_policy_relays_only_exact_contact_permissions(local_agent_http, local_wechat_bridge, monkeypatch):
    """联系人授权沿用同源鉴权，搜索和写入只去适配器的固定控制入口。"""
    import json
    from urllib.error import HTTPError
    from urllib.parse import parse_qs, urlencode, urlsplit
    from urllib.request import Request, urlopen

    context = local_agent_http
    route = f"{context.base}/api/wechat/contact-policy"
    query = "李&张+#"
    allow = {"recipient_id": "wxid_fixture", "mode": "allowlist"}
    candidate = {"id": "wxid_fixture", "kind": "direct", "displayName": "合成备注", "remark": "合成备注",
                 "nickname": "示例昵称", "alias": "fixture-alias", "mode": "unassigned"}
    entry = {"id": "wxid_other_fixture", "displayName": "仅观察联系人", "mode": "observe"}

    def request(body=None, *, headers=context.headers, search_query=None):
        url = route if search_query is None else f"{route}?{urlencode({'query': search_query})}"
        request = Request(url, headers=headers, data=None if body is None else json.dumps(body).encode())
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.code, json.load(response)

    assert request(headers={}, search_query=query)[0] == 401
    assert request(allow, headers={})[0] == 401
    off_host = {**context.headers, "Host": "outside.example.invalid"}
    assert request(headers=off_host, search_query=query)[0] == 403
    assert request(allow, headers=off_host)[0] == 403
    assert request(allow, headers={**context.headers, "Content-Type": "text/plain"})[0] == 415
    assert request(search_query="字" * 129)[0] == 400
    for invalid in ({}, {**allow, "recipient_id": "123@chatroom"}, {**allow, "recipient_id": "123@im.chatroom"},
                    {**allow, "recipient_id": "a/b"}, {**allow, "recipient_id": ".."},
                    {**allow, "recipient_id": "nickname whitespace"}, {**allow, "mode": "send"}):
        assert request(invalid)[0] == 400
    assert local_wechat_bridge.calls == []

    local_wechat_bridge.contact_policy = (200, {
        "ok": True, "managed": True, "account_id": "private-account", "path": "/private/path",
        "candidates": [{**candidate, "internal_native_profile": "hidden"},
                       {**candidate, "id": "123@chatroom", "kind": "group"}],
        "entries": [{**entry, "backend_only": "hidden"}, {**entry, "id": "123@chatroom"}],
    })
    status, body = request(search_query=query)
    assert status == 200
    assert body["ok"] and body["managed"]
    assert body["candidates"] == [candidate] and body["entries"] == [entry]
    assert "account_id" not in body and "path" not in body
    method, path, token, payload = local_wechat_bridge.calls.pop()
    assert (method, urlsplit(path).path, parse_qs(urlsplit(path).query)) == ("GET", "/contact-policy", {"query": [query]})
    assert token == "fixture-bridge-control-token" and payload is None

    monkeypatch.delenv("MAHJONG_WECHATY_OUTBOUND_TOKEN")
    for mode in ("allowlist", "observe", "managed_only"):
        selected = {**candidate, "mode": mode}
        flags = {"managed": True, "auto_reply": False} if mode == "managed_only" else {}
        if flags:
            selected["automation"] = flags
        selected_entry = {"id": selected["id"], "displayName": selected["displayName"], "mode": mode}
        local_wechat_bridge.contact_policy = (200, {
            "ok": True, "managed": True, "entries": [selected_entry],
            "recipient": {**selected, "native_auth": "hidden"},
        })
        status, body = request({**allow, "mode": mode, **flags,
                                "account_id": "ignored", "path": "/send", "enabled": True})
        assert status == 200 and body["ok"] and body["managed"]
        assert body["entries"] == [selected_entry] and body["recipient"] == selected
        assert local_wechat_bridge.calls.pop() == (
            "POST", "/contact-policy", "local-channel-test-token", {**allow, "mode": mode, **flags},
        )
    assert request({**allow, "mode": "managed_only", "managed": False})[0] == 400
    assert request({**allow, "mode": "managed_only", "auto_reply": True})[0] == 400
    assert local_wechat_bridge.calls == []
    assert context.client.calls == []
    assert context.runtime.store.list_channel_observations() == []


def test_wechat_permission_page_and_group_relay_keep_inventory_local(local_agent_http, local_wechat_bridge, monkeypatch):
    """独立页和分页目录受本机鉴权约束，群保存仅传精确身份并核对回执。"""
    import json
    from urllib.error import HTTPError
    from urllib.parse import parse_qs, urlencode, urlsplit
    from urllib.request import Request, urlopen

    context = local_agent_http
    room_id = "123@chatroom"
    account_id = "fixture-account"
    group = {"id": room_id, "kind": "group", "displayName": "合成微信群", "mode": "observe", "override": "observe"}
    monkeypatch.setenv("MAHJONG_WECHATY_MANAGED_ROOM_IDS", room_id)
    monkeypatch.setattr(context.app, "agent_wechat_group_reply_policy", lambda: {
        "account_id": account_id, "room_ids": [room_id],
    })

    def request(uri, *, headers=context.headers, body=None):
        try:
            response = urlopen(Request(context.base + uri, headers=headers,
                                       data=None if body is None else json.dumps(body).encode()), timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.code, json.load(response)

    with urlopen(context.base + "/wechat-permissions", timeout=5) as response:
        assert "微信权限管理" in response.read().decode()
        assert response.headers["Set-Cookie"].startswith("mahjong_agent_token=")
    off_host = {**context.headers, "Host": "outside.example.invalid"}
    assert request("/wechat-permissions", headers=off_host)[0] == 403
    roster = "/api/wechat/permission-roster"
    save = "/api/wechat/group-permission"
    assert request(roster, headers={})[0] == 401
    assert request(roster, headers=off_host)[0] == 403
    assert request(roster + "?view=invalid")[0] == 400
    assert request(roster + "?limit=101")[0] == 400
    assert request(save, body={"recipient_id": "wxid_fixture", "mode": "allowlist"})[0] == 400
    assert request(save, body={"recipient_id": room_id, "mode": "auto"})[0] == 400
    assert local_wechat_bridge.calls == []

    local_wechat_bridge.permission_roster = (200, {
        "ok": True, "managed": True, "account_id": account_id, "nextCursor": "page+next=",
        "items": [{**group, "native_owner": "hidden"}, {**group, "id": "wxid_fixture", "kind": "direct"}],
    })
    params = {"view": "groups", "query": "合成&群", "cursor": "page+first=", "limit": "40"}
    status, body = request(f"{roster}?{urlencode(params)}")
    assert status == 200 and body["nextCursor"] == "page+next="
    assert len(body["items"]) == 1 and "account_id" not in body
    item = body["items"][0]
    assert item["id"] == room_id and item["mode"] == "observe" and "native_owner" not in item
    assert item["group_rules"] == {"managed": True, "auto_reply": True, "observe": False}
    assert item["automation"] == {"managed": False, "auto_reply": False}
    method, path, token, sent = local_wechat_bridge.calls.pop()
    assert (method, urlsplit(path).path, sent) == ("GET", "/permission-roster", None)
    assert parse_qs(urlsplit(path).query) == {key: [value] for key, value in params.items()}
    assert token == "fixture-bridge-control-token"

    monkeypatch.delenv("MAHJONG_WECHATY_OUTBOUND_TOKEN")
    local_wechat_bridge.group_permission = (200, {
        "ok": True, "account_id": account_id, "item": {**group, "mode": "unassigned", "override": None},
    })
    status, body = request(save, body={"recipient_id": room_id, "mode": "inherit", "enabled": True, "path": "/send"})
    assert status == 200 and body["item"]["override"] is None
    assert body["item"]["group_rules"] == {"managed": True, "auto_reply": True, "observe": False}
    assert body["item"]["automation"] == {"managed": True, "auto_reply": True}
    assert local_wechat_bridge.calls == [
        ("POST", "/group-permission", "local-channel-test-token", {"recipient_id": room_id, "mode": "inherit"}),
    ]
    local_wechat_bridge.calls.clear()
    settings = {"recipient_id": room_id, "mode": "inherit", "managed": False, "auto_reply": True}
    local_wechat_bridge.group_permission = (200, {
        "ok": True, "account_id": account_id, "item": {
            **group, "mode": "unassigned", "override": None, "automation": {"managed": False, "auto_reply": True},
        },
    })
    assert request(save, body={**settings, "auto_reply": "yes"})[0] == 400
    assert request(save, body={**settings, "mode": "observe"})[0] == 400
    status, body = request(save, body={**settings, "enabled_at": "ignored", "enabled": True})
    assert status == 200 and body["item"]["automation"] == {"managed": False, "auto_reply": True}
    assert local_wechat_bridge.calls == [("POST", "/group-permission", "local-channel-test-token", settings)]
    assert context.client.calls == []


def test_wechat_console_records_only_accepted_direct_messages_and_deduplicates_retries(
    local_agent_http, local_wechat_bridge,
) -> None:
    """命中发送总闸不会写上下文，成功的私聊用真实回执 ID 去重，群聊不写入私聊。"""
    import json
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    context = local_agent_http
    draft = {"recipient_id": "wxid_fixture", "kind": "direct", "text": "明天有空吗？", "request_id": "fixture-attempt"}
    def send(payload):
        request = Request(f"{context.base}/api/wechat/send", headers=context.headers,
                          data=json.dumps(payload, ensure_ascii=False).encode())
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    conversation_id = "wechaty:contact:wxid_fixture"
    store = context.runtime.store
    for reason, code, label in (
        ("send channel is disabled", "wechat_send_channel_disabled", "发送总闸"),
        ("auto send switch is disabled", "wechat_auto_send_disabled", "自动外发"),
    ):
        local_wechat_bridge.reject = reason
        with pytest.raises(HTTPError) as rejection:
            send(draft)
        assert rejection.value.code == 409
        response = json.load(rejection.value)
        assert response["sent"] is False and response["error"] == code
        assert label in response["message"]
    assert store.recent_turns(conversation_id) == []

    local_wechat_bridge.reject = ""
    local_wechat_bridge.uncertain = True
    with pytest.raises(HTTPError) as pending:
        send(draft)
    assert pending.value.code == 502
    assert json.load(pending.value)["delivery_uncertain"] is True
    assert store.recent_turns(conversation_id) == []
    local_wechat_bridge.uncertain = False

    first = send(draft)
    assert first["ok"] and first["sent"] and first["context_recorded"]
    assert send(draft) == first
    bridge_payload = local_wechat_bridge.calls[-1][3]
    assert bridge_payload == {"to": draft["recipient_id"], "text": draft["text"], "target_type": "direct",
                              "exact_target": True, "source_trace_id": first["trace_id"]}
    group = send({**draft, "kind": "group", "recipient_id": "123@chatroom"})
    assert group["sent"] and not group["context_recorded"]
    assert not store.recent_turns("wechaty:room:123@chatroom")
    turns = store.recent_turns(conversation_id)
    assert len(turns) == 1
    assert turns[0].content == draft["text"]
    assert turns[0].metadata["reply_author"] == "owner"
    assert turns[0].metadata["source_message_id"] == "agent-wechat:server:server-fixture-1"
    synced = context.app.record_agent_wechat_self_message(context.runtime, {
        "message_origin": "owner", "self_message": True, "message_id": "local-fixture",
        "source_message_id": "local-fixture", "conversation_id": conversation_id, "text": draft["text"],
        "payload": {"direction": "outgoing", "threadId": draft["recipient_id"], "id": "local-fixture",
                    "server_id": "server-fixture-1"},
    }, trace_id="fixture_sync")
    assert synced["audit"]["deduplicated"] is True
    assert len(store.recent_turns(conversation_id)) == 1

    from mahjong_agent_runtime.domains.context_builders import AgentContextBuilder

    message = UserMessage(conversation_id=conversation_id, sender_id=draft["recipient_id"], sender_name="合成客户",
                          text="可以。", metadata={"channel": "agent-wechat"})
    builder = AgentContextBuilder(store, ToolGateway(store))
    built = builder.build(message, trace_id="fixture_followup").payload
    assert [turn["content"] for turn in built["recent_conversation"]] == [draft["text"]]
    other = UserMessage(conversation_id="wechaty:contact:other", sender_id="other", sender_name="其他客户",
                        text="在吗？", metadata={"channel": "agent-wechat"})
    other_payload = builder.build(other, trace_id="other").payload
    assert not other_payload["recent_conversation"]
    assert "owner_conversation_context" not in other_payload
    assert context.client.calls == []


def test_runtime_http_manifest_and_console_only_advertise_supported_channels(local_agent_http) -> None:
    from urllib.request import Request, urlopen

    context = local_agent_http
    with urlopen(Request(f"{context.base}/api/runtime", headers=context.headers), timeout=5) as response:
        manifest = context.app.json.load(response)
    with urlopen(f"{context.base}/", timeout=5) as response:
        owner = response.read().decode("utf-8")
        assert "mahjong_agent_token=" in response.headers["Set-Cookie"]
        assert "HttpOnly" in response.headers["Set-Cookie"]
    with urlopen(f"{context.base}/developer", timeout=5) as response:
        developer = response.read().decode("utf-8")
    for retired_channel in ("hermes", "astrbot"):
        assert retired_channel not in context.app.json.dumps(manifest).lower()
        assert retired_channel not in owner.lower()
        assert retired_channel not in developer.lower()
    assert manifest["endpoints"]["wechaty_raw"] == ["/api/channels/wechaty/raw"]
    assert manifest["endpoints"]["message"] == ["/api/message"]
    assert "/api/channels/wechaty/raw" in developer and "/api/channels/agent-wechat/raw" in developer
    assert "/api/message" in developer and "/api/message" not in owner
    assert context.client.calls == []


def test_console_state_projects_open_work_and_raw_state_stays_protected(local_agent_http) -> None:
    """真实 HTTP 投影只返回日常运营字段，原始快照继续保留诊断证据。"""
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from mahjong_agent_runtime.models import PendingInputBatchStatus, now

    context = local_agent_http
    store = context.runtime.store
    game, _ = store.create_game(
        conversation_id="internal-console-test-conversation",
        organizer_id="internal-console-test-organizer",
        organizer_name="测试牌局发起人",
        requirement={"game_type": "hangzhou_mahjong", "base_stake": 0.5},
        known_players=[],
        trace_id="test_console_state",
    )
    store.upsert_customer(CustomerProfile(
        customer_id=game.organizer_id, display_name=game.organizer_name, private_remark="仅原始快照可见的备注",
    ))
    drafts, _ = store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[
            {"customer_id": "test-pending-invite", "display_name": "待审批的客户", "message_text": "打牌吗？",
             "metadata": {"content_review_approved": True, "content_review_trace_id": "internal-review-trace"}},
            {"customer_id": "test-sent-invite", "display_name": "已发邀约的客户", "message_text": "有空一起打牌吗？"},
        ],
        trace_id="test_console_state",
    )
    store.update_invite_delivery_status(
        draft_id=drafts[1].draft_id, status="sent", reason="test_seed", trace_id="test_console_state",
    )

    at = now()
    for status in ("pending", "processing", "completed"):
        batch, _, _ = store.upsert_pending_input_fragment(
            UserMessage(conversation_id=f"test-input-{status}", sender_id="test-fragment-sender",
                        sender_name="测试消息发送人", text="仅原始快照可见的聊天片段", message_id=f"test-fragment-{status}"),
            quiet_deadline=at, trace_id="test_console_state",
        )
        task, _ = store.ensure_group_board_publish_task(
            room_id=f"test-task-{status}", due_at=at, trace_id="test_console_state",
        )
        if status != "pending":
            store.claim_pending_input_batch(batch_id=batch.batch_id, expected_version=batch.version,
                                           trace_id="test_console_state")
            claimed = store.claim_scheduled_task(task.task_id, at=at)
            assert claimed is not None and claimed.lease_until is not None
        if status == "completed":
            store.finish_pending_input_batch(batch_id=batch.batch_id, expected_version=batch.version,
                                            status=PendingInputBatchStatus.COMPLETED, trace_id="test_console_state")
            store.complete_scheduled_task(task.task_id, trace_id="test_console_state",
                                          expected_attempts=claimed.attempts, expected_lease_until=claimed.lease_until)

    endpoint = f"{context.base}/api/state"
    console_endpoint = f"{endpoint}?view=console"
    for path in (endpoint, console_endpoint):
        with pytest.raises(HTTPError) as exc:
            urlopen(path, timeout=5)
        assert exc.value.code == 401

    with urlopen(Request(console_endpoint, headers=context.headers), timeout=5) as response:
        console = context.app.json.load(response)
    with urlopen(Request(endpoint, headers=context.headers), timeout=5) as response:
        raw = context.app.json.load(response)

    assert set(console) == {"summary", "active_game_schedule", "invite_drafts"}
    assert console["summary"] == {"active_games": 1, "pending_invites": 1,
                                  "pending_input_batches": 2, "pending_scheduled_tasks": 2}
    schedule, = console["active_game_schedule"]
    assert schedule["requirement"]["game_type"] == "hangzhou_mahjong"
    assert schedule["requirement"]["base_stake"] == 0.5
    assert schedule["seat_summary"]["remaining_seats"] == 3
    assert {"game_id", "conversation_id", "organizer_id", "participants", "parties", "seat_claims"}.isdisjoint(schedule)
    pending, = console["invite_drafts"]
    assert pending["draft_id"] == drafts[0].draft_id
    assert pending["message_text"] == "打牌吗？"
    assert pending["metadata"] == {"content_review_approved": True}

    assert raw["games"][0]["game_id"] == game.game_id
    assert len(raw["invite_drafts"]) == 2
    raw_text = context.app.json.dumps(raw, ensure_ascii=False)
    console_text = context.app.json.dumps(console, ensure_ascii=False)
    for diagnostic_field in ("仅原始快照可见的备注", "仅原始快照可见的聊天片段", "internal-review-trace"):
        assert diagnostic_field in raw_text
        assert diagnostic_field not in console_text


def test_console_state_debug_details_start_collapsed_and_empty(local_agent_http) -> None:
    """老板首页保留运营概况，完整快照只在开发者页按需呈现。"""
    import re
    from urllib.request import urlopen

    with urlopen(f"{local_agent_http.base}/", timeout=5) as response:
        owner = response.read().decode("utf-8")
    with urlopen(f"{local_agent_http.base}/developer", timeout=5) as response:
        html = response.read().decode("utf-8")
    assert "老板工作台" in owner and 'href="/developer"' in owner
    assert 'id="wechatTarget"' in owner and 'id="gameSchedule"' in owner and 'id="inviteDrafts"' in owner
    assert 'id="stateDebug"' not in owner and "/api/logs" not in owner and "/api/reset-state" not in owner
    assert 'id="wechatTarget"' not in html and 'id="inviteDrafts"' not in html
    assert "<!-- OWNER_SCRIPT -->" not in owner and "<!-- DEVELOPER_SCRIPT -->" not in html
    details = re.search(r'<details\b[^>]*\bid="stateDebug"[^>]*>', html)
    assert details and not re.search(r'\sopen(?:[\s=>]|$)', details.group())
    assert "<summary>原始 JSON（调试）</summary>" in html
    assert re.search(r'<pre\b[^>]*\bid="stateRaw"[^>]*>\s*</pre>', html)
    assert "/api/state?view=console" in owner and "/api/state?view=console" in html
    with urlopen(f"{local_agent_http.base}/console.css", timeout=5) as response:
        assert response.headers.get_content_type() == "text/css"
        assert b".console-panel" in response.read()


def test_wechaty_raw_http_requires_auth_and_preserves_received_evidence(local_agent_http, monkeypatch) -> None:
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    context = local_agent_http
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "incoming_only")
    payload = {
        "conversation_id": "wechaty:contact:http_test",
        "sender_id": "http_owner",
        "sender_name": "老板",
        "message_id": "wechaty_http_raw_001",
        "text": "原始消息证据",
        "self_message": True,
        "message_type": 7,
    }
    body = context.app.json.dumps(payload, ensure_ascii=False).encode()
    endpoint = f"{context.base}/api/channels/wechaty/raw"
    for method in ("GET", "POST"):
        for headers in ({}, {"Authorization": "Bearer invalid-test-token"}):
            with pytest.raises(HTTPError) as exc:
                urlopen(Request(endpoint, data=body if method == "POST" else None,
                                method=method, headers=headers), timeout=5)
            assert exc.value.code == 401
    assert not context.app.WECHATY_RAW_LOG_PATH.exists()
    assert context.runtime.store.list_channel_observations() == []

    with urlopen(Request(endpoint, data=body, headers=context.headers), timeout=5) as response:
        received = context.app.json.load(response)
    assert received["ok"] is True
    assert received["record"]["payload"] == payload
    assert received["route_result"]["routed_to_agent"] is False
    assert received["route_result"]["audit"]["reason"] == "self_message_in_incoming_only_scope"
    with urlopen(Request(endpoint, headers=context.headers), timeout=5) as response:
        history = context.app.json.load(response)
    assert history["records"] == [received["record"]]
    assert context.app.json.loads(context.app.WECHATY_RAW_LOG_PATH.read_text()) == received["record"]
    # 重新打开事实库，确认 HTTP 成功不是仅写在进程内存中的回执。
    persisted = SQLiteAgentStore(context.store_path).list_channel_observations(channel="wechaty")
    assert len(persisted) == 1
    assert persisted[0]["source_message_id"] == payload["message_id"]
    assert persisted[0]["payload"]["raw_payload"] == payload
    assert persisted[0]["route_mode"] == "filtered"
    assert context.client.calls == []


def test_main_message_http_still_requires_auth_and_runs_agent(local_agent_http) -> None:
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    context = local_agent_http
    payload = {
        "conversation_id": "http_main_message",
        "sender_id": "http_customer",
        "sender_name": "张哥",
        "message_id": "http_main_001",
        "text": "你好",
    }
    body = context.app.json.dumps(payload, ensure_ascii=False).encode()
    endpoint = f"{context.base}/api/message"
    for headers in ({}, {"Authorization": "Bearer invalid-test-token"}):
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(endpoint, data=body, headers=headers), timeout=5)
        assert exc.value.code == 401
    assert context.client.calls == []
    assert context.runtime.store.recent_turns(payload["conversation_id"]) == []
    context.client.outputs.append(context.app.json.dumps({
        "goal": "回应问候",
        "objective_status": "completed",
        "reasoning_summary": "用户问候，不涉及业务副作用。",
        "reply_to_user": "你好。",
        "tool_calls": [],
        "needs_human": False,
        "stop_reason": {
            "can_stop": True,
            "why": "已回应问候。",
            "pending_work": [],
            "depends_on_tool_results": False,
        },
        "badcase": None,
    }, ensure_ascii=False))
    with urlopen(Request(endpoint, data=body, headers=context.headers), timeout=5) as response:
        result = context.app.json.load(response)
    assert result["conversation_id"] == payload["conversation_id"]
    assert result["final_reply"] == "你好。"
    assert result["status"] == "completed"
    assert len(context.client.calls) == 1
    turns = SQLiteAgentStore(context.store_path).recent_turns(payload["conversation_id"])
    assert [(turn.role, turn.content) for turn in turns] == [("user", "你好"), ("assistant", "你好。")]


def test_simulation_http_review_and_export_require_auth_and_preserve_evidence(tmp_path, monkeypatch) -> None:
    import json
    from http.server import ThreadingHTTPServer
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen
    app = load_app_module()
    import simulation_observability
    monkeypatch.setattr(app, "_API_TOKEN", "local-simulation-test-token")
    monkeypatch.setattr(simulation_observability, "DEFAULT_RUNTIME_DIR", tmp_path)
    run_id = "20261025T120000Z_http_review"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    report_path = run_dir / "report.json"
    original = json.dumps({"llm_mode": "real", "transcript": [{
        "sequence": 1, "conversation_id": "sim:group:g1", "thread_id": "script_1",
        "user": {"customer_id": "u1", "text": "我来"}, "agent": {"reply": "好"},
    }]}, ensure_ascii=False).encode()
    report_path.write_bytes(original)
    server = ThreadingHTTPServer(("127.0.0.1", 0), app.AgentRuntimeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    headers = {"Authorization": "Bearer local-simulation-test-token", "Content-Type": "application/json"}
    try:
        with pytest.raises(HTTPError) as exc:
            urlopen(f"{base}/api/simulations/export?run_id={run_id}")
        assert exc.value.code == 401
        body = json.dumps({"run_id": run_id, "sequence": 1, "verdict": "uncertain", "note": "需核对局号"}).encode()
        request = Request(f"{base}/api/simulations/reviews", data=body, headers=headers)
        with urlopen(request) as response:
            assert json.load(response)["ok"] is True
        with urlopen(Request(f"{base}/api/simulations/export?run_id={run_id}&format=jsonl", headers=headers)) as response:
            assert 'attachment; filename=' in response.headers["Content-Disposition"]
            row = json.loads(response.read())
        assert row["agent"]["reply"] == "好"
        assert row["human_review"]["note"] == "需核对局号"
        assert row["simulation_run"]["llm_mode"] == "real"
        assert report_path.read_bytes() == original
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(f"{base}/api/simulations/export?run_id=invalid", headers=headers))
        assert exc.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_manual_badcase_is_recorded_through_tool_gateway(tmp_path) -> None:
    app = load_app_module()
    store = SQLiteAgentStore(tmp_path / "agent_runtime_manual_badcase.sqlite3")
    trace = InMemoryTraceRecorder()
    runtime = AgentRuntime(
        llm_client=StaticAgentClient(
            [
                app.json.dumps(
                    {
                        "goal": "测试回复",
                        "objective_status": "completed",
                        "reasoning_summary": "模型直接回复。",
                        "reply_to_user": "好的，我先帮你留意下。",
                        "tool_calls": [],
                        "needs_human": False,
                        "stop_reason": {
                            "can_stop": True,
                            "why": "测试场景模拟模型提前停止，后续由人工 badcase 入口归档。",
                            "pending_work": [],
                            "depends_on_tool_results": False,
                        },
                        "badcase": None,
                    },
                    ensure_ascii=False,
                )
            ]
        ),
        store=store,
        tool_gateway=ToolGateway(store=store, trace_recorder=trace),
        trace_recorder=trace,
    )
    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_manual_badcase",
            sender_id="zhang",
            sender_name="张哥",
            text="组",
            message_id="msg_manual_badcase",
        ),
        trace_id="trace_manual_source",
    )

    response = app.record_manual_badcase(
        runtime,
        {
            "source_trace_id": result.trace_id,
            "audit_trace_id": "trace_manual_audit",
            "reason": "回复停在留意，没有继续规划",
            "expected": {"behavior": "应该继续规划或追问关键缺口"},
            "operator_id": "tester",
            "operator_name": "测试者",
        },
    )

    assert response["tool_result"]["called"] is True
    assert response["tool_result"]["allowed"] is True
    assert len(store.badcases) == 1
    badcase = store.badcases[0]
    assert badcase["reason"] == "回复停在留意，没有继续规划"
    assert badcase["input"]["message"]["text"] == "组"
    assert badcase["actual"]["reply"] == "好的，我先帮你留意下。"
    assert badcase["metadata"]["source_trace_id"] == "trace_manual_source"
    steps = [event.step for event in trace.get_trace("trace_manual_audit")]
    assert steps == [
        "manual_badcase_input",
        "tool_called",
        "tool_gateway_received",
        "tool_idempotency_checked",
        "tool_definition_checked",
        "tool_schema_checked",
        "tool_permission_checked",
        "tool_authorization_checked",
        "tool_idempotency_claimed",
        "tool_gateway_completed",
        "tool_result",
        "manual_badcase_recorded",
    ]


def test_app_defaults_to_main_trial_port(monkeypatch) -> None:
    monkeypatch.delenv("MAHJONG_AGENT_PORT", raising=False)

    app = load_app_module()

    assert app.PORT == 8790


def test_main_agent_app_entrypoint_exists_without_versioned_operator_name() -> None:
    text = MAIN_SCRIPT.read_text(encoding="utf-8")

    assert "from agent_runtime_app import MahjongAgentApplication" in text
    assert "MahjongAgentApplication.from_environment().run()" in text


def test_main_app_imports_stable_runtime_package() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "from mahjong_agent_runtime import" in text
    assert "from mahjong_agent_v3 import" not in text
    assert "from mahjong_agent_v3.tracing" not in text


def test_app_uses_public_runtime_coordination_and_review_boundaries() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "runtime.conversation_lock(" in text
    assert "runtime.action_processor.review_customer_visible_content(" in text
    assert "runtime._conversation_lock" not in text
    assert "runtime._conversation_locks" not in text
    assert "runtime._run_customer_visible_content_review" not in text


def test_application_settings_are_loaded_before_process_objects(monkeypatch, tmp_path) -> None:
    app = load_app_module()
    monkeypatch.setattr(app, "load_dotenv_defaults", lambda _path: None)
    monkeypatch.setenv("MAHJONG_AGENT_PORT", "9012")
    monkeypatch.setenv("MAHJONG_AGENT_DB_PATH", str(tmp_path / "configured.sqlite3"))

    settings = app.ApplicationSettings.from_environment()

    assert settings.host == "127.0.0.1"
    assert settings.port == 9012
    assert settings.database_path == tmp_path / "configured.sqlite3"


def test_invalid_numeric_application_settings_use_safe_defaults(monkeypatch) -> None:
    app = load_app_module()
    monkeypatch.setattr(app, "load_dotenv_defaults", lambda _path: None)
    monkeypatch.setenv("MAHJONG_AGENT_PORT", "not-a-number")
    monkeypatch.setenv("MAHJONG_AGENT_MAX_REQUEST_BYTES", "not-a-number")
    monkeypatch.setenv("MAHJONG_AGENT_MAX_CONCURRENT_REQUESTS", "not-a-number")
    monkeypatch.setenv("MAHJONG_AGENT_REQUESTS_PER_MINUTE", "not-a-number")

    settings = app.ApplicationSettings.from_environment()

    assert settings.port == 8790
    assert settings.max_request_bytes == 1024 * 1024
    assert settings.max_concurrent_requests == 8
    assert settings.requests_per_minute == 120


def test_main_reuses_an_explicitly_injected_runtime(monkeypatch) -> None:
    app = load_app_module()
    injected_runtime = object()
    calls: list[object] = []

    class Application:
        @classmethod
        def from_environment(cls, *, runtime=None):
            calls.append(runtime)
            return cls()

        def run(self) -> None:
            calls.append("run")

    monkeypatch.setattr(app, "MahjongAgentApplication", Application)
    app.RUNTIME = injected_runtime

    app.main()

    assert calls == [injected_runtime, "run"]


def test_mahjong_agent_application_owns_the_process_lifecycle() -> None:
    app = load_app_module()
    events: list[str] = []

    class Record:
        def to_dict(self) -> dict:
            return {"id": "maintenance"}

    class Store:
        def ensure_waiting_demand_expiration_task(self, **_kwargs):
            events.append("maintenance.initialize")
            return Record(), None

    class TraceRecorder:
        def record(self, *_args, **_kwargs) -> None:
            events.append("maintenance.trace")

    class Runtime:
        store = Store()
        trace_recorder = TraceRecorder()

    class Scheduler:
        def __init__(self, name: str) -> None:
            self.name = name

        def start(self) -> None:
            events.append(f"{self.name}.start")

        def stop(self) -> None:
            events.append(f"{self.name}.stop")

    class Server:
        def serve_forever(self) -> None:
            events.append("server.serve")

        def server_close(self) -> None:
            events.append("server.close")

    settings = app.ApplicationSettings(
        host="127.0.0.1",
        port=8790,
        database_path=Path("runtime.sqlite3"),
        trace_path=Path("trace.jsonl"),
        wechaty_raw_log_path=Path("wechaty.jsonl"),
        max_request_bytes=1024,
        max_concurrent_requests=2,
        requests_per_minute=10,
    )
    application = app.MahjongAgentApplication(
        settings=settings,
        runtime=Runtime(),
        input_scheduler=Scheduler("input"),
        scheduled_task_scheduler=Scheduler("scheduled"),
        server=Server(),
        agent_run_recovery_scheduler=Scheduler("recovery"),
    )

    assert app.RUNTIME is application.runtime
    assert app.INPUT_SCHEDULER is application.input_scheduler
    assert app.SCHEDULED_TASK_SCHEDULER is application.scheduled_task_scheduler
    assert app.AGENT_RUN_RECOVERY_SCHEDULER is application.agent_run_recovery_scheduler
    assert app.PORT == settings.port

    application.run()

    assert events == [
        "maintenance.initialize",
        "maintenance.trace",
        "input.start",
        "scheduled.start",
        "recovery.start",
        "server.serve",
        "input.stop",
        "scheduled.stop",
        "recovery.stop",
        "server.close",
    ]


def test_application_close_stops_serve_forever_from_another_thread() -> None:
    app = load_app_module()

    class Record:
        def to_dict(self) -> dict:
            return {}

    class Store:
        def ensure_waiting_demand_expiration_task(self, **_kwargs):
            return Record(), None

    class TraceRecorder:
        def record(self, *_args, **_kwargs) -> None:
            return None

    class Runtime:
        store = Store()
        trace_recorder = TraceRecorder()

    class Scheduler:
        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class BlockingServer:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.shutdown_requested = threading.Event()
            self.closed = False

        def serve_forever(self) -> None:
            self.started.set()
            self.shutdown_requested.wait(timeout=2.0)

        def shutdown(self) -> None:
            self.shutdown_requested.set()

        def server_close(self) -> None:
            self.closed = True

    settings = app.ApplicationSettings(
        host="127.0.0.1",
        port=8790,
        database_path=Path("runtime.sqlite3"),
        trace_path=Path("trace.jsonl"),
        wechaty_raw_log_path=Path("wechaty.jsonl"),
        max_request_bytes=1024,
        max_concurrent_requests=2,
        requests_per_minute=10,
    )
    server = BlockingServer()
    application = app.MahjongAgentApplication(
        settings=settings,
        runtime=Runtime(),
        input_scheduler=Scheduler(),
        scheduled_task_scheduler=Scheduler(),
        server=server,
    )
    worker = threading.Thread(target=application.run)
    worker.start()
    assert server.started.wait(timeout=1.0)

    application.close()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert server.closed is True


def test_application_close_cannot_interleave_scheduler_startup() -> None:
    app = load_app_module()
    events: list[str] = []
    input_start_entered = threading.Event()
    release_input_start = threading.Event()
    server_stop_requested = threading.Event()
    close_returned = threading.Event()

    class Record:
        def to_dict(self) -> dict:
            return {}

    class Store:
        def ensure_waiting_demand_expiration_task(self, **_kwargs):
            events.append("maintenance")
            return Record(), None

    class TraceRecorder:
        def record(self, *_args, **_kwargs) -> None:
            return None

    class Runtime:
        store = Store()
        trace_recorder = TraceRecorder()

    class Scheduler:
        def __init__(self, name: str, *, block_start: bool = False) -> None:
            self.name = name
            self.block_start = block_start

        def start(self) -> None:
            events.append(f"{self.name}.start")
            if self.block_start:
                input_start_entered.set()
                assert release_input_start.wait(timeout=1.0)

        def stop(self) -> None:
            events.append(f"{self.name}.stop")

    class Server:
        def serve_forever(self) -> None:
            events.append("server.serve")
            server_stop_requested.wait(timeout=1.0)

        def shutdown(self) -> None:
            events.append("server.shutdown")
            server_stop_requested.set()

        def server_close(self) -> None:
            events.append("server.close")

    settings = app.ApplicationSettings(
        host="127.0.0.1",
        port=8790,
        database_path=Path("runtime.sqlite3"),
        trace_path=Path("trace.jsonl"),
        wechaty_raw_log_path=Path("wechaty.jsonl"),
        max_request_bytes=1024,
        max_concurrent_requests=2,
        requests_per_minute=10,
    )
    application = app.MahjongAgentApplication(
        settings=settings,
        runtime=Runtime(),
        input_scheduler=Scheduler("input", block_start=True),
        scheduled_task_scheduler=Scheduler("scheduled"),
        server=Server(),
    )
    runner = threading.Thread(target=application.run)
    closer = threading.Thread(target=lambda: (application.close(), close_returned.set()))
    runner.start()
    assert input_start_entered.wait(timeout=1.0)
    closer.start()
    assert not close_returned.wait(timeout=0.05)

    release_input_start.set()
    runner.join(timeout=1.0)
    closer.join(timeout=1.0)

    assert not runner.is_alive()
    assert not closer.is_alive()
    assert events.index("scheduled.start") < events.index("input.stop")
    assert events.index("scheduled.start") < events.index("scheduled.stop")
    assert events.count("server.close") == 1


def test_run_interrupt_before_serve_does_not_shutdown_an_unstarted_server(monkeypatch) -> None:
    app = load_app_module()

    class Record:
        def to_dict(self) -> dict:
            return {}

    class Store:
        def ensure_waiting_demand_expiration_task(self, **_kwargs):
            return Record(), None

    class TraceRecorder:
        def record(self, *_args, **_kwargs) -> None:
            return None

    class Runtime:
        store = Store()
        trace_recorder = TraceRecorder()

    class Scheduler:
        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class Server:
        serve_calls = 0
        shutdown_calls = 0
        close_calls = 0

        def serve_forever(self) -> None:
            self.serve_calls += 1

        def shutdown(self) -> None:
            self.shutdown_calls += 1

        def server_close(self) -> None:
            self.close_calls += 1

    settings = app.ApplicationSettings(
        host="127.0.0.1",
        port=8790,
        database_path=Path("runtime.sqlite3"),
        trace_path=Path("trace.jsonl"),
        wechaty_raw_log_path=Path("wechaty.jsonl"),
        max_request_bytes=1024,
        max_concurrent_requests=2,
        requests_per_minute=10,
    )
    server = Server()
    application = app.MahjongAgentApplication(
        settings=settings,
        runtime=Runtime(),
        input_scheduler=Scheduler(),
        scheduled_task_scheduler=Scheduler(),
        server=server,
    )
    real_print = print
    interrupted = False

    def interrupt_first_print(*args, **kwargs) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        real_print(*args, **kwargs)

    monkeypatch.setattr("builtins.print", interrupt_first_print)

    application.run()

    assert server.serve_calls == 0
    assert server.shutdown_calls == 0
    assert server.close_calls == 1


def test_rejected_second_run_does_not_close_the_active_run() -> None:
    app = load_app_module()

    class Record:
        def to_dict(self) -> dict:
            return {}

    class Store:
        def ensure_waiting_demand_expiration_task(self, **_kwargs):
            return Record(), None

    class TraceRecorder:
        def record(self, *_args, **_kwargs) -> None:
            return None

    class Runtime:
        store = Store()
        trace_recorder = TraceRecorder()

    class Scheduler:
        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class Server:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.stop_requested = threading.Event()
            self.shutdown_calls = 0

        def serve_forever(self) -> None:
            self.started.set()
            self.stop_requested.wait(timeout=2.0)

        def shutdown(self) -> None:
            self.shutdown_calls += 1
            self.stop_requested.set()

        def server_close(self) -> None:
            return None

    settings = app.ApplicationSettings(
        host="127.0.0.1",
        port=8790,
        database_path=Path("runtime.sqlite3"),
        trace_path=Path("trace.jsonl"),
        wechaty_raw_log_path=Path("wechaty.jsonl"),
        max_request_bytes=1024,
        max_concurrent_requests=2,
        requests_per_minute=10,
    )
    server = Server()
    application = app.MahjongAgentApplication(
        settings=settings,
        runtime=Runtime(),
        input_scheduler=Scheduler(),
        scheduled_task_scheduler=Scheduler(),
        server=server,
    )
    first_run = threading.Thread(target=application.run)
    first_run.start()
    assert server.started.wait(timeout=1.0)

    with pytest.raises(RuntimeError, match="cannot run from running"):
        application.run()

    assert first_run.is_alive()
    assert server.shutdown_calls == 0

    application.close()
    first_run.join(timeout=1.0)
    assert not first_run.is_alive()
    assert server.shutdown_calls == 1


def test_external_close_and_pre_serve_interrupt_do_not_deadlock(monkeypatch) -> None:
    app = load_app_module()
    print_entered = threading.Event()
    release_print = threading.Event()

    class Record:
        def to_dict(self) -> dict:
            return {}

    class Store:
        def ensure_waiting_demand_expiration_task(self, **_kwargs):
            return Record(), None

    class TraceRecorder:
        def record(self, *_args, **_kwargs) -> None:
            return None

    class Runtime:
        store = Store()
        trace_recorder = TraceRecorder()

    class Scheduler:
        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class Server:
        serve_calls = 0
        shutdown_calls = 0

        def serve_forever(self) -> None:
            self.serve_calls += 1

        def shutdown(self) -> None:
            self.shutdown_calls += 1

        def server_close(self) -> None:
            return None

    settings = app.ApplicationSettings(
        host="127.0.0.1",
        port=8790,
        database_path=Path("runtime.sqlite3"),
        trace_path=Path("trace.jsonl"),
        wechaty_raw_log_path=Path("wechaty.jsonl"),
        max_request_bytes=1024,
        max_concurrent_requests=2,
        requests_per_minute=10,
    )
    server = Server()
    application = app.MahjongAgentApplication(
        settings=settings,
        runtime=Runtime(),
        input_scheduler=Scheduler(),
        scheduled_task_scheduler=Scheduler(),
        server=server,
    )
    real_print = print
    interrupted = False

    def blocking_interrupt(*args, **kwargs) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            print_entered.set()
            assert release_print.wait(timeout=1.0)
            raise KeyboardInterrupt
        real_print(*args, **kwargs)

    monkeypatch.setattr("builtins.print", blocking_interrupt)
    runner = threading.Thread(target=application.run)
    runner.start()
    assert print_entered.wait(timeout=1.0)

    closer = threading.Thread(target=application.close)
    closer.start()
    closer.join(timeout=1.0)
    release_print.set()
    runner.join(timeout=1.0)

    assert not runner.is_alive()
    assert not closer.is_alive()
    assert server.serve_calls == 0
    assert server.shutdown_calls == 0


def test_direct_start_failure_rolls_back_started_workers() -> None:
    app = load_app_module()
    events: list[str] = []

    class Record:
        def to_dict(self) -> dict:
            return {}

    class Store:
        def ensure_waiting_demand_expiration_task(self, **_kwargs):
            return Record(), None

    class TraceRecorder:
        def record(self, *_args, **_kwargs) -> None:
            return None

    class Runtime:
        store = Store()
        trace_recorder = TraceRecorder()

    class Scheduler:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def start(self) -> None:
            events.append(f"{self.name}.start")
            if self.fail:
                raise RuntimeError("scheduler failed")

        def stop(self) -> None:
            events.append(f"{self.name}.stop")

    class Server:
        def server_close(self) -> None:
            events.append("server.close")

    settings = app.ApplicationSettings(
        host="127.0.0.1",
        port=8790,
        database_path=Path("runtime.sqlite3"),
        trace_path=Path("trace.jsonl"),
        wechaty_raw_log_path=Path("wechaty.jsonl"),
        max_request_bytes=1024,
        max_concurrent_requests=2,
        requests_per_minute=10,
    )
    application = app.MahjongAgentApplication(
        settings=settings,
        runtime=Runtime(),
        input_scheduler=Scheduler("input"),
        scheduled_task_scheduler=Scheduler("scheduled", fail=True),
        server=Server(),
    )

    with pytest.raises(RuntimeError, match="scheduler failed"):
        application.start()

    assert events == [
        "input.start",
        "scheduled.start",
        "input.stop",
        "scheduled.stop",
        "server.close",
    ]


def test_rejected_second_start_does_not_close_the_started_application() -> None:
    app = load_app_module()
    events: list[str] = []

    class Record:
        def to_dict(self) -> dict:
            return {}

    class Store:
        def ensure_waiting_demand_expiration_task(self, **_kwargs):
            return Record(), None

    class TraceRecorder:
        def record(self, *_args, **_kwargs) -> None:
            return None

    class Runtime:
        store = Store()
        trace_recorder = TraceRecorder()

    class Scheduler:
        def __init__(self, name: str) -> None:
            self.name = name

        def start(self) -> None:
            events.append(f"{self.name}.start")

        def stop(self) -> None:
            events.append(f"{self.name}.stop")

    class Server:
        def server_close(self) -> None:
            events.append("server.close")

    settings = app.ApplicationSettings(
        host="127.0.0.1",
        port=8790,
        database_path=Path("runtime.sqlite3"),
        trace_path=Path("trace.jsonl"),
        wechaty_raw_log_path=Path("wechaty.jsonl"),
        max_request_bytes=1024,
        max_concurrent_requests=2,
        requests_per_minute=10,
    )
    application = app.MahjongAgentApplication(
        settings=settings,
        runtime=Runtime(),
        input_scheduler=Scheduler("input"),
        scheduled_task_scheduler=Scheduler("scheduled"),
        server=Server(),
    )
    application.start()

    with pytest.raises(RuntimeError, match="cannot start from started"):
        application.start()

    assert events == ["input.start", "scheduled.start"]

    application.close()
    assert events == [
        "input.start",
        "scheduled.start",
        "input.stop",
        "scheduled.stop",
        "server.close",
    ]


def test_operator_console_exposes_test_observability_page() -> None:
    app = load_app_module()

    page = app.test_observability_html()
    manifest_text = app.developer_html()

    assert "测试与回放" in page
    assert "重跑确定性并发测试" in page
    assert "调用真实 DeepSeek 回放" in page
    assert "record_candidate_reply" in page
    assert "/api/test-observability/run" in page
    assert 'href="/tests"' in manifest_text and 'href="/developer"' in page


def test_operator_console_exposes_simulation_observability_page() -> None:
    app = load_app_module()

    page = app.simulation_dashboard_html()
    manifest_text = app.developer_html()

    assert "模拟聊天室 · 对话审阅" in page
    assert "/api/simulations" in page
    assert "业务对话" in page
    assert "闲聊插入" in page
    assert "回到业务" in page
    assert "模拟消息从哪里来" in page
    assert "Agent 实际执行" in page
    assert "导出完整对话 MD" in page
    assert "模型识别尚未独立验收" in page
    assert 'href="/simulation"' in manifest_text and 'href="/developer"' in page


def test_observability_runner_rejects_non_allowlisted_commands() -> None:
    app = load_app_module()

    try:
        app.run_fixed_suite("rm -rf /")
    except ValueError as exc:
        assert "unsupported test suite" in str(exc)
    else:
        raise AssertionError("arbitrary commands must not reach the test runner")


def test_main_app_does_not_expose_legacy_trial_defaults() -> None:
    app = load_app_module()
    owner, developer = app.index_html(), app.developer_html()
    assert "boss_trial" not in owner and "boss_trial" not in developer
    assert "runtime_trial" not in owner and "runtime_trial" in developer


def test_api_message_builder_requires_explicit_sender_identity() -> None:
    app = load_app_module()

    message, missing_fields = app.build_api_user_message({"text": "现在有人吗"})

    assert message is None
    assert missing_fields == ["conversation_id", "sender_id", "sender_name"]


def test_api_message_builder_preserves_explicit_identity_and_quote() -> None:
    app = load_app_module()

    message, missing_fields = app.build_api_user_message(
        {
            "conversation_id": "runtime_trial",
            "sender_id": "wang02",
            "sender_name": "王哥",
            "text": "可以",
            "quoted_message": {
                "message_id": "wechat_msg_001",
                "text": "七点三缺一，打吗？",
                "business_ref_type": "outbound_message_draft",
                "business_ref_id": "draft_001",
            },
        }
    )

    assert missing_fields == []
    assert message is not None
    assert message.conversation_id == "runtime_trial"
    assert message.sender_id == "wang02"
    assert message.sender_name == "王哥"
    assert message.text == "可以"
    assert message.quoted_message is not None
    assert message.quoted_message.message_id == "wechat_msg_001"
    assert message.quoted_message.business_ref_type == "outbound_message_draft"
    assert message.quoted_message.business_ref_id == "draft_001"


def test_runtime_manifest_identifies_current_main_chain(tmp_path) -> None:
    app = load_app_module()
    store = SQLiteAgentStore(tmp_path / "agent_runtime_manifest.sqlite3")
    trace = InMemoryTraceRecorder()
    runtime = AgentRuntime(
        llm_client=StaticAgentClient([]),
        store=store,
        tool_gateway=ToolGateway(store=store, trace_recorder=trace),
        trace_recorder=trace,
    )

    manifest = app.runtime_manifest(runtime)

    assert manifest["runtime"] == "mahjong_agent_runtime"
    assert manifest["main_chain"] == "agent_runtime"
    assert manifest["implementation_package"] == "mahjong_agent_runtime"
    assert "compatibility_packages" not in manifest
    assert manifest["legacy_reference_only"] is True
    assert manifest["legacy_entrypoints"]["legacy_analyze_endpoint"] == "not_exposed"
    assert manifest["legacy_entrypoints"]["default_runtime_entrypoint"] == "scripts/run_agent_app.py"
    assert "/api/message" in manifest["endpoints"]["message"]
    assert "legacy_endpoint_aliases" not in manifest
    assert "search_current_games" in manifest["available_tools"]
    assert "update_context_checkpoint" in manifest["available_tools"]
    assert manifest["runtime_config"]["progress_monitor"] == {
        "repeated_observation_limit": 2,
        "consecutive_no_progress_limit": 2,
        "max_replan_attempts": 1,
        "max_cycle_period": 3,
    }
    assert "/api/reset-state" in manifest["endpoints"]["reset_state"]
    assert "/api/analyze" not in app.json.dumps(manifest, ensure_ascii=False)


def test_operator_reset_state_clears_runtime_state_but_preserves_assets(tmp_path) -> None:
    app = load_app_module()
    store = SQLiteAgentStore(tmp_path / "agent_runtime_reset.sqlite3")
    trace = InMemoryTraceRecorder()
    runtime = AgentRuntime(
        llm_client=StaticAgentClient(
            [
                app.json.dumps(
                    {
                        "goal": "测试回复",
                        "objective_status": "completed",
                        "reasoning_summary": "模型直接回复。",
                        "reply_to_user": "收到。",
                        "tool_calls": [],
                        "needs_human": False,
                        "stop_reason": {
                            "can_stop": True,
                            "why": "测试场景模拟模型回复。",
                            "pending_work": [],
                            "depends_on_tool_results": False,
                        },
                        "badcase": None,
                    },
                    ensure_ascii=False,
                )
            ]
        ),
        store=store,
        tool_gateway=ToolGateway(store=store, trace_recorder=trace),
        trace_recorder=trace,
    )
    store.upsert_customer(CustomerProfile(customer_id="zhang", display_name="张哥"))
    runtime.handle_user_message(
        UserMessage(
            conversation_id="reset_conversation",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_reset_001",
        ),
        trace_id="trace_before_reset",
    )
    game, _ = store.create_game(
        conversation_id="reset_conversation",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong"},
        known_players=[],
        trace_id="trace_before_reset",
    )
    store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[{"customer_id": "ran", "display_name": "冉姐", "message_text": "冉姐，打吗？"}],
        trace_id="trace_before_reset",
    )
    store.upsert_conversation_checkpoint(
        conversation_id="reset_conversation",
        summary="用户正在组局",
        facts={"stake": "1"},
        open_questions=[],
        trace_id="trace_before_reset",
    )
    store.remember_result("tool_key", ToolResult(name="search_current_games", called=True, allowed=True))
    store.record_badcase({"reason": "保留样本"}, trace_id="trace_before_reset", conversation_id="reset_conversation")

    response = app.reset_runtime_state(runtime, {"trace_id": "trace_reset_operator"})

    assert response["deleted"]["games"] == 1
    assert response["deleted"]["invite_drafts"] == 1
    assert response["deleted"]["conversation_turns"] > 0
    assert response["deleted"]["conversation_checkpoints"] == 1
    assert response["deleted"]["idempotency_ledger"] > 0
    assert response["deleted"]["message_results"] == 1
    assert response["deleted"]["customers"] == 0
    assert response["deleted"]["badcases"] == 0
    assert store.games == {}
    assert store.invite_drafts == {}
    assert store.conversation_checkpoints == {}
    assert store.idempotent_result("tool_key") is None
    assert len(store.customers) == 1
    assert len(store.badcases) == 1
    steps = [event.step for event in trace.get_trace("trace_reset_operator")]
    assert steps == ["operator_reset_state_requested", "operator_reset_state_completed"]


def test_developer_html_has_reset_state_button() -> None:
    app = load_app_module()
    html = app.developer_html()

    assert "清空状态和记忆" in html
    assert "/api/reset-state" in html
    assert "客户画像、badcase/eval 和日志" in html


def test_log_tail_exposes_trace_log_path(tmp_path, monkeypatch) -> None:
    app = load_app_module()
    trace_path = tmp_path / "agent_runtime_trace.log"
    trace_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    monkeypatch.setattr(app, "TRACE_PATH", trace_path)

    assert app.tail_trace_log(2) == ["line2", "line3"]


def test_human_approved_invite_is_sent_once_and_persisted(tmp_path, monkeypatch) -> None:
    app = load_app_module()
    store = SQLiteAgentStore(tmp_path / "invite_delivery.sqlite3")
    trace = InMemoryTraceRecorder()
    runtime = AgentRuntime(
        llm_client=StaticAgentClient([]),
        store=store,
        tool_gateway=ToolGateway(store=store, trace_recorder=trace),
        trace_recorder=trace,
    )
    game, _ = store.create_game(
        conversation_id="invite_delivery_conversation",
        organizer_id="customer_a",
        organizer_name="客户A",
        requirement={"game_type": "hangzhou_mahjong", "start_time_kind": "asap_when_full"},
        known_players=[{"customer_id": "customer_a", "display_name": "客户A"}],
        trace_id="trace_invite_delivery_setup",
    )
    drafts, _ = store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[
            {
                "customer_id": "customer_b",
                "display_name": "客户B",
                "message_text": "0.5无烟，打吗？",
                "metadata": {"content_review_approved": True, "content_review_trace_id": "trace_review"},
            }
        ],
        trace_id="trace_invite_delivery_setup",
    )
    calls: list[tuple[str, dict | None]] = []

    def fake_request(path: str, *, payload=None, timeout_seconds=3.0):
        calls.append((path, payload))
        return {"send_channel_enabled": True} if path == "/health" else {"ok": True}

    monkeypatch.setattr(app, "request_local_json", fake_request)
    first = app.handle_invite_draft_action(
        runtime,
        {"draft_id": drafts[0].draft_id, "action": "approve_send", "trace_id": "trace_human_approve"},
    )
    second = app.handle_invite_draft_action(
        runtime,
        {"draft_id": drafts[0].draft_id, "action": "approve_send", "trace_id": "trace_human_retry"},
    )

    assert first["sent"] is True
    assert second["deduplicated"] is True
    assert store.invite_drafts[drafts[0].draft_id].status.value == "sent"
    assert [path for path, _ in calls] == ["/health", "/send"]


def test_agent_wechat_raw_endpoint_records_its_own_channel(local_agent_http, monkeypatch) -> None:
    from urllib.request import Request, urlopen

    context = local_agent_http
    monkeypatch.delenv("MAHJONG_AGENT_WECHAT_ROUTE_SCOPE", raising=False)
    monkeypatch.delenv("MAHJONG_WECHATY_ROUTE_SCOPE", raising=False)
    monkeypatch.delenv("MAHJONG_WECHATY_AUTO_ROUTE_TO_AGENT", raising=False)
    payload = {
        "conversation_id": "wechaty:contact:wxid_agent_wechat",
        "sender_id": "wxid_agent_wechat",
        "sender_name": "阿强",
        "message_id": "agent_wechat_raw_001",
        "text": "原始消息证据",
        "self_message": True,
        "message_type": 7,
    }
    body = context.app.json.dumps(payload, ensure_ascii=False).encode()
    endpoint = f"{context.base}/api/channels/agent-wechat/raw"
    with urlopen(Request(endpoint, data=body, headers=context.headers), timeout=5) as response:
        received = context.app.json.load(response)
    assert received["ok"] is True
    assert received["channel"] == "agent-wechat"
    assert received["route_result"]["audit"]["channel"] == "agent-wechat"
    assert received["route_result"]["audit"]["route_scope"] == "incoming_only"
    assert received["route_result"]["audit"]["reason"] == "self_message_in_incoming_only_scope"
    # 原始证据必须落在 agent-wechat 自己的日志里，不能混进 wechaty 证据文件。
    assert context.app.json.loads(
        context.app.AGENT_WECHAT_RAW_LOG_PATH.read_text()
    ) == received["record"]
    assert not context.app.WECHATY_RAW_LOG_PATH.exists()

    persisted = SQLiteAgentStore(context.store_path).list_channel_observations(
        channel="agent-wechat"
    )
    assert len(persisted) == 1
    # 会话命名空间与原生微信一致，观察记录保留真实通道。
    assert persisted[0]["conversation_id"].startswith("wechaty:contact:")
    assert persisted[0]["channel"] == "agent-wechat"
    assert persisted[0]["route_mode"] == "filtered"
    assert context.runtime.store.list_channel_observations(channel="wechaty") == []

    with urlopen(Request(f"{endpoint}?limit=10", headers=context.headers), timeout=5) as response:
        history = context.app.json.load(response)
    assert history["channel"] == "agent-wechat"
    assert history["records"] == [received["record"]]
    with urlopen(
        Request(f"{context.base}/api/channels/agent-wechat/observations", headers=context.headers),
        timeout=5,
    ) as response:
        observations = context.app.json.load(response)
    assert observations["channel"] == "agent-wechat"
    assert observations["count"] == 1
    assert context.client.calls == []


def test_agent_wechat_contact_policy_only_routes_authorized_private_messages(
    local_agent_http, tmp_path, monkeypatch,
) -> None:
    """只观察与未分配私聊须留存真实入站证据，且不能进入 Agent 上下文和出站。"""
    import json
    from urllib.request import Request, urlopen

    context = local_agent_http
    policy = tmp_path / "contact_permissions.local.json"
    policy.write_text(json.dumps({
        "account_id": "fixture-account",
        "managed_direct": True,
        "recipients": [
            {"thread_id": "wxid_allowed", "label": "已授权"},
            {"thread_id": "wxid_manual", "label": "仅手动", "managed": False, "auto_reply": False},
        ],
        "managed_only": [
            {"thread_id": "wxid_managed", "label": "仅托管", "aliases": ["wxid_managed_alias"]},
            {"thread_id": "wxid_observed", "label": "已撤回托管"},
            {"thread_id": "wxid_manual", "label": "只允许手动发送"},
        ],
        "observed": [{"thread_id": "wxid_observed", "label": "仅观察"}],
    }), encoding="utf-8")
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_ACCOUNT", "fixture-account")
    monkeypatch.delenv("MAHJONG_AGENT_WECHAT_OUTBOUND_ALLOWLIST_PATH", raising=False)
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_ROUTE_SCOPE", "incoming_only")
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_INPUT_AGGREGATION_ENABLED", "false")
    monkeypatch.setenv("MAHJONG_WECHATY_INPUT_GATE_ENABLED", "false")

    def post(thread_id, message_id, *, self_message=False, account_id="fixture-account", conversation_id=None):
        payload = {
            "conversation_id": conversation_id or f"wechaty:contact:{thread_id}",
            "message_id": message_id, "source_message_id": message_id, "sender_id": thread_id,
            "text": "合成消息", "self_message": self_message, "message_type": 7,
            "payload": {"id": message_id, "threadId": thread_id, "account_id": account_id,
                        "direction": "outgoing" if self_message else "incoming"},
        }
        if self_message:
            payload["message_origin"] = "owner"
        request = Request(f"{context.base}/api/channels/agent-wechat/raw", headers=context.headers,
                          data=json.dumps(payload).encode())
        with urlopen(request, timeout=5) as response:
            return json.load(response)["route_result"]

    for thread_id, message_id, self_message, account_id in (
        ("wxid_observed", "observed-incoming", False, "fixture-account"),
        ("wxid_manual", "manual-only", False, "fixture-account"),
        ("wxid_other", "default-observe", False, "fixture-account"),
        ("wxid_observed", "observed-owner", True, "fixture-account"),
        ("wxid_managed_alias", "managed-alias-is-not-an-id", False, "fixture-account"),
        ("wxid_managed", "managed-wrong-account", False, "unexpected-account"),
        ("wxid_allowed", "wrong-account", False, "unexpected-account"),
        ("wxid_allowed", "missing-account", False, ""),
    ):
        result = post(thread_id, message_id, self_message=self_message, account_id=account_id)
        assert result["observe_only"] is True and result["routed_to_agent"] is False
        assert result["agent_result"] is None and not result.get("scoped_group_reply")
        assert not context.runtime.store.recent_turns(f"wechaty:contact:{thread_id}")
    forged = post("wxid_allowed", "wrong-conversation", conversation_id="wechaty:contact:wxid_observed")
    assert forged["observe_only"] is True and not forged["routed_to_agent"]
    assert not context.runtime.store.recent_turns("wechaty:contact:wxid_observed")
    assert not context.client.calls
    observations = context.runtime.store.list_channel_observations(channel="agent-wechat")
    assert {item["source_message_id"] for item in observations} == {
        "observed-incoming", "manual-only", "default-observe", "observed-owner", "wrong-account", "missing-account",
        "wrong-conversation", "managed-alias-is-not-an-id", "managed-wrong-account",
    }
    assert all(item["route_mode"] == "observe_only" and item["text"] == "合成消息" for item in observations)
    assert context.app.AGENT_WECHAT_RAW_LOG_PATH.read_text(encoding="utf-8").count("\n") == len(observations)

    approved_reply = json.dumps({
        "goal": "回应测试消息", "objective_status": "completed", "reasoning_summary": "测试消息无业务副作用。",
        "reply_to_user": "你好。", "tool_calls": [], "needs_human": False,
        "stop_reason": {"can_stop": True, "why": "已回应。", "pending_work": [], "depends_on_tool_results": False},
        "badcase": None,
    })
    for thread_id in ("wxid_allowed", "wxid_managed"):
        context.client.outputs.append(approved_reply)
        result = post(thread_id, f"incoming-{thread_id}")
        assert result["routed_to_agent"] is True and result["audit"]["reason"] == "valid_text_message"
    assert len(context.client.calls) == 2


@pytest.mark.parametrize("policy_change", [
    {"managed_direct": False, "managed_only": []},
    {"managed_direct": True, "managed_only": None},
    {"managed_direct": True, "managed_only": [{"thread_id": "123@chatroom", "label": "示例群"}]},
    {"managed_direct": True, "managed_only": [{"thread_id": "../invalid", "label": "无效标识"}]},
    {"managed_direct": True, "managed_only": [{"thread_id": "wxid_managed", "label": "示例", "aliases": "昵称"}]},
])
def test_agent_wechat_managed_only_invalid_policy_is_observe_only(tmp_path, monkeypatch, policy_change):
    """仅托管需有效的账号名单，损坏配置不能向其他私聊放开入站权限。"""
    import json

    app = load_app_module()
    policy = tmp_path / "private-contact-policy.json"
    policy.write_text(json.dumps({
        "account_id": "fixture-account", "recipients": [{"thread_id": "wxid_allowed", "label": "示例联系人"}],
        **policy_change,
    }), encoding="utf-8")
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_OUTBOUND_ALLOWLIST_PATH", str(policy))
    assert app.agent_wechat_private_observe_only("wxid_allowed", "fixture-account")


def test_agent_wechat_contact_policy_revocation_ignores_already_queued_batch(
    local_agent_http, tmp_path, monkeypatch,
) -> None:
    """已在队列里的私聊撤回权限后，按当前权限落下忽略终态。"""
    import json

    from mahjong_agent_runtime.models import PendingInputBatchStatus, now

    context = local_agent_http
    app = context.app
    store = context.runtime.store
    policy = tmp_path / "private-contact-policy.json"
    contact = {"thread_id": "wxid_allowed", "label": "示例联系人"}
    policy_document = {"account_id": "fixture-account", "managed_direct": True,
                       "recipients": [contact], "observed": []}
    policy.write_text(json.dumps(policy_document), encoding="utf-8")
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_OUTBOUND_ALLOWLIST_PATH", str(policy))
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_ACCOUNT", "fixture-account")
    monkeypatch.setattr(app, "run_wechaty_input_gate", lambda *args, **kwargs: pytest.fail("仅观察不得调用分流"))
    monkeypatch.setattr(app, "request_local_json", lambda *args, **kwargs: pytest.fail("仅观察不得调用发送接口"))
    message = UserMessage(conversation_id="wechaty:contact:wxid_allowed", sender_id="wxid_allowed",
                          sender_name="示例联系人", text="合成消息", message_id="queued-private-message",
                          metadata={"channel": "agent-wechat", "native_reply_source": {"account_id": "fixture-account"}})
    due_at = now()
    batch, _, _ = store.upsert_pending_input_fragment(message, quiet_deadline=due_at, trace_id="initial-queue")
    queued = store.queue_pending_input_batch(
        batch_id=batch.batch_id, expected_version=batch.version, due_at=due_at,
        decision={"action": "process_business", "dispatch_mode": "background_after_immediate_ack"},
    )
    assert queued is not None and queued.source_channel == "agent-wechat"
    policy.write_text(json.dumps({**policy_document, "recipients": [], "observed": [contact]}), encoding="utf-8")

    app.handle_due_pending_input_batch(queued, "after-revocation")

    finished = store.pending_input_batch(message.conversation_id, message.sender_id)
    assert finished is not None and finished.status == PendingInputBatchStatus.IGNORED
    assert finished.decision["action"] == "ignore" and finished.decision["agent_result"] is None
    assert context.client.calls == []
    assert store.recent_turns(message.conversation_id) == []


def test_unauthorized_native_group_queue_is_ignored_before_business(local_agent_http, monkeypatch):
    """旧版本误放进通用队列的原生群批次保留证据，关闭业务派发和外发。"""
    from mahjong_agent_runtime.models import PendingInputBatchStatus, now

    context = local_agent_http
    app = context.app
    store = context.runtime.store
    monkeypatch.setattr(app, "run_wechaty_input_gate", lambda *args, **kwargs: pytest.fail("群批次不应调用分流"))
    monkeypatch.setattr(app, "request_local_json", lambda *args, **kwargs: pytest.fail("群批次不应调用发送接口"))
    for kind in ("room", "contact"):
        message = UserMessage(conversation_id=f"wechaty:{kind}:123@chatroom", sender_id="wxid_sample",
                              sender_name="示例联系人", text="合成消息", message_id=f"queued-group-{kind}",
                              metadata={"channel": "agent-wechat"})
        due_at = now()
        batch, _, _ = store.upsert_pending_input_fragment(message, quiet_deadline=due_at, trace_id="old-group")
        queued = store.queue_pending_input_batch(
            batch_id=batch.batch_id, expected_version=batch.version, due_at=due_at,
            decision={"action": "process_business", "dispatch_mode": "background_after_immediate_ack"},
        )
        assert queued is not None
        app.handle_due_pending_input_batch(queued, "retire-old-group")
        finished = store.pending_input_batch(message.conversation_id, message.sender_id)
        assert finished.status == PendingInputBatchStatus.IGNORED
        assert finished.decision["decision_source"] == "native_group_outside_group_route"
        assert not store.recent_turns(message.conversation_id)
    assert not context.client.calls


def test_agent_wechat_owner_reply_enters_only_its_private_context(local_agent_http, monkeypatch) -> None:
    from datetime import datetime, timedelta
    from urllib.request import Request, urlopen

    from mahjong_agent_runtime.domains.context_builders.conversation_context import (
        ContextPackingPolicy,
        build_conversation_context,
    )

    context = local_agent_http
    app = context.app
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_ROUTE_SCOPE", "all")
    conversation_id = "wechaty:contact:customer"
    store = context.runtime.store
    started_at = datetime.now(app.DEFAULT_TZ) - timedelta(minutes=2)
    task, _ = store.activate_task_context(
        conversation_id=conversation_id, customer_id="customer",
        trace_id="current_task", reason="new_request", activity_at=started_at, started_at=started_at,
        force_new=False, archive_previous=False,
    )
    # 回复在摘要之前已发出，摘要完成后才经原生微信同步到本机。
    checkpoint, _ = store.upsert_conversation_checkpoint(
        conversation_id=conversation_id, summary="前文摘要", facts={}, open_questions=[],
        trace_id="summary", task_context_id=task.task_context_id,
    )

    def post(source_id, *, origin="owner", thread_id="customer", conversation=conversation_id,
             text="我留了八点的桌。", self_message=True, direction="outgoing"):
        payload = {
            "message_id": source_id, "source_message_id": source_id, "conversation_id": conversation,
            "sender_id": "owner_account", "self_message": self_message, "message_origin": origin,
            "is_room": thread_id.endswith("@chatroom"), "message_type": 7, "text": text,
            "payload": {
                "id": source_id, "direction": direction, "threadId": thread_id,
                "createdAt": (task.started_at + timedelta(seconds=1)).isoformat(),
            },
        }
        request = Request(
            f"{context.base}/api/channels/agent-wechat/raw", headers=context.headers,
            data=app.json.dumps(payload).encode(),
        )
        with urlopen(request, timeout=5) as response:
            result = app.json.load(response)["route_result"]
        assert result["routed_to_agent"] is False and result["agent_result"] is None
        return result["audit"]

    assert post("owner-1")["reason"] == "owner_reply_recorded"
    assert post("owner-1", text="重复投递不能覆盖原文")["deduplicated"] is True
    assert post("bot-echo", origin="local_send")["reason"] == "self_message_ignored"
    assert post("missing-self", self_message=False)["reason"] == "self_message_ignored"
    assert post("invalid-direction", direction="incoming")["reason"] == "self_message_ignored"
    assert post("incorrect-thread", thread_id="elsewhere")["reason"] == "self_message_ignored"
    assert post("room-owner", thread_id="room@chatroom",
                conversation="wechaty:room:room@chatroom")["reason"] == "group_owner_reply_observed"
    assert context.client.calls == []

    reopened = SQLiteAgentStore(context.store_path)
    turns = reopened.recent_turns(conversation_id)
    assert len(turns) == 1
    assert turns[0].occurred_at == task.started_at + timedelta(seconds=1)
    assert turns[0].occurred_at < checkpoint.updated_at
    assert reopened.recent_turns("wechaty:room:room@chatroom") == []
    for chat, sender in ((conversation_id, "customer"), ("wechaty:contact:elsewhere", "elsewhere")):
        bundle = build_conversation_context(
            reopened, UserMessage(conversation_id=chat, sender_id=sender, sender_name=sender, text="好的"),
            trace_id="next", previous_tool_results=None, packing_policy=ContextPackingPolicy(),
        )
        if chat == conversation_id:
            assert [turn["role"] for turn in bundle.recent_conversation] == ["assistant"]
            assert bundle.recent_conversation[0]["content"] == "我留了八点的桌。"
            metadata = bundle.recent_conversation[0]["metadata"]
            assert datetime.fromisoformat(metadata["owner_reply_recorded_at"]) >= checkpoint.updated_at
            assert {key: value for key, value in metadata.items() if key != "owner_reply_recorded_at"} == {
                "channel": "agent-wechat", "source_message_id": "owner-1", "reply_author": "owner",
                "delivery_status": "delivered", "task_context_id": task.task_context_id,
            }
        else:
            assert bundle.recent_conversation == []


def test_agent_wechat_owner_reply_is_idempotent_in_memory() -> None:
    from types import SimpleNamespace

    from mahjong_agent_runtime import InMemoryAgentStore

    app = load_app_module()
    runtime = SimpleNamespace(store=InMemoryAgentStore())
    conversation = "wechaty:contact:customer"
    payload = {
        "message_id": "owner-media", "source_message_id": "owner-media",
        "conversation_id": conversation, "self_message": True, "message_origin": "owner",
        "payload": {"id": "owner-media", "direction": "outgoing", "threadId": "customer"},
    }
    for index in range(2):
        result = app.record_agent_wechat_self_message(runtime, payload, trace_id=f"trace-{index}")
        assert result["audit"]["deduplicated"] is bool(index)
    turns = runtime.store.recent_turns(conversation)
    assert len(turns) == 1
    assert turns[0].trace_id == "trace-0"
    assert turns[0].content == "[老板发送了一条非文字消息]"


def test_agent_wechat_durable_receipt_unblocks_other_conversations(local_agent_http, monkeypatch) -> None:
    """用真实 HTTP、SQLite 与 Agent 路由复现前一会话的模型请求迟迟未返回。"""
    from time import monotonic, sleep
    from urllib.request import Request, urlopen

    context = local_agent_http
    app = context.app
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_ROUTE_SCOPE", "incoming_only")
    monkeypatch.setenv("MAHJONG_WECHATY_AUTO_ROUTE_TO_AGENT", "true")
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_INPUT_AGGREGATION_ENABLED", "false")
    monkeypatch.setenv("MAHJONG_WECHATY_INPUT_GATE_ENABLED", "false")
    events = {key: threading.Event() for key in ("a-first", "a-next", "b", "c")}
    blockers = {key: threading.Event() for key in ("a-first", "b")}
    calls = []
    reply = app.json.dumps({
        "goal": "回应问候", "objective_status": "completed",
        "reasoning_summary": "客户问候，不涉及业务副作用。", "reply_to_user": "你好。",
        "tool_calls": [], "needs_human": False,
        "stop_reason": {"can_stop": True, "why": "已回应问候。", "pending_work": [],
                        "depends_on_tool_results": False},
        "badcase": None,
    })

    def complete(_client, _messages, *, trace_id, timeout_seconds):
        calls.append(trace_id)
        events[trace_id].set()
        if trace_id in blockers:
            assert blockers[trace_id].wait(timeout=10), "测试未释放被阻塞的模型请求"
        return reply

    monkeypatch.setattr(type(context.client), "complete", complete)
    worker = app.ChannelIngressWorker(
        store=context.runtime.store,
        coordination=context.runtime.coordination_manager,
        handler=lambda observation: app.process_queued_channel_raw(context.runtime, observation),
        max_workers=2,
    )
    context.server.channel_ingress_worker = worker
    worker.start()

    def post(message_id, conversation, *, text="你好", trace_id=None):
        request = Request(
            f"{context.base}/api/channels/agent-wechat/raw",
            headers={**context.headers, "Prefer": "respond-async"},
            data=app.json.dumps({
                "conversation_id": f"wechaty:contact:{conversation}",
                "sender_id": conversation, "message_id": message_id, "trace_id": trace_id or message_id,
                "text": text, "message_type": 7,
            }).encode(),
        )
        with urlopen(request, timeout=2) as response:
            assert response.status == 202
            return app.json.load(response)

    try:
        assert post("a-first", "a")["accepted"] is True
        assert events["a-first"].wait(timeout=2)
        assert post("a-next", "a")["accepted"] is True
        assert post("b", "b")["accepted"] is True
        assert events["b"].wait(timeout=2)
        assert post("c", "c")["accepted"] is True
        duplicate = post("a-next", "a", text="不能覆盖首次原文", trace_id="retry")
        assert duplicate["deduplicated"] is True
        assert duplicate["trace_id"] == "a-next"
        assert "route_result" not in duplicate
        # 另一个 SQLite 连接能读到确认后的原始消息和待处理状态。
        reopened = SQLiteAgentStore(context.store_path)
        saved = {row["source_message_id"]: row for row in reopened.list_channel_observations()}
        assert set(saved) == {"a-first", "a-next", "b", "c"}
        assert saved["a-next"]["route_mode"] == saved["c"]["route_mode"] == "ingress_queue"
        assert saved["a-next"]["text"] == "你好"
        assert not events["a-next"].is_set()
        assert not events["c"].is_set()

        blockers["b"].set()
        assert events["c"].wait(timeout=2)
        assert not events["a-next"].is_set()
        blockers["a-first"].set()
        assert events["a-next"].wait(timeout=2)
        deadline = monotonic() + 2
        while any(row["route_mode"] == "ingress_queue" for row in reopened.list_channel_observations()):
            assert monotonic() < deadline
            sleep(0.01)
        assert calls == ["a-first", "b", "c", "a-next"]
        assert {row["route_status"] for row in reopened.list_channel_observations()} == {"processed"}
        assert len(app.AGENT_WECHAT_RAW_LOG_PATH.read_text().splitlines()) == 4
    finally:
        for event in blockers.values():
            event.set()
        worker.stop()


def test_agent_wechat_pending_receipt_survives_store_reopen(local_agent_http, monkeypatch) -> None:
    """模拟确认之后、路由之前停止；重新打开数据库后仍能消费并保留失败证据。"""
    context = local_agent_http
    app = context.app
    payload = {
        "sender_id": "test-sender", "message_id": "durable-receipt-test",
        "text": "用于恢复验证的消息", "message_type": 7,
    }
    receipt = app.queue_agent_wechat_raw(context.runtime, payload)
    reopened = SQLiteAgentStore(context.store_path)
    pending = reopened.next_channel_ingress(set())
    assert pending["trace_id"] == receipt["trace_id"]
    assert pending["conversation_id"] == "wechaty:contact:test-sender"

    def failing_route(_payload, *, trace_id):
        assert trace_id == receipt["trace_id"]
        raise TimeoutError("测试路由超时")

    monkeypatch.setattr(app, "channel_route_handler", lambda _channel: failing_route)
    app.process_queued_channel_raw(context.runtime, pending)
    assert reopened.next_channel_ingress(set()) is None
    observation, = reopened.list_channel_observations(channel="agent-wechat")
    assert observation["route_status"] == "route_failed"
    assert observation["route_reason"] == "TimeoutError"
    assert app.queue_agent_wechat_raw(context.runtime, payload)["deduplicated"] is True
    assert reopened.next_channel_ingress(set()) is None


@pytest.mark.parametrize("late_fragment", [False, True], ids=["send-once", "superseded-before-send"])
def test_native_private_casual_burst_answers_once_with_all_fragments(
    local_agent_http, tmp_path, monkeypatch, late_fragment,
):
    """补充片段同批生成；审查结束前又入站时，旧稿不会交给发送接口。"""
    import json
    from datetime import timedelta

    from mahjong_agent_runtime import PendingInputScheduler

    context = local_agent_http
    app, runtime = context.app, context.runtime
    store = runtime.store
    account, contact = "fixture-account", "wxid_fixture"
    conversation_id = f"wechaty:contact:{contact}"
    path = tmp_path / "permissions.json"
    path.write_text(json.dumps({
        "account_id": account, "managed_direct": True, "recipients": [{"thread_id": contact, "label": "测试联系人"}],
    }))
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_OUTBOUND_ALLOWLIST_PATH", str(path))
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_ACCOUNT", account)
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_ROUTE_SCOPE", "incoming_only")
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_INPUT_AGGREGATION_ENABLED", "true")
    calls, deliveries = [], []
    decision = {"action": "process_casual", "category": "casual_chat"}
    monkeypatch.setattr(app, "run_wechaty_input_gate", lambda *args, **kwargs: decision)
    response = "还好这次没碰上。"
    context.client.outputs.append(json.dumps({
        "should_reply": True, "reply_to_user": response,
        "reasoning_summary": "承接这一轮的实际情况。", "needs_human": False,
    }))
    runtime.action_processor.reply_self_review_enabled = True
    context.client.outputs.append(json.dumps({
        "approved": True, "needs_human": False, "reasoning_summary": "未反转前文事实。", "violations": [],
        "item_reviews": [{"item_id": "casual_chat.reply_to_user", "approved": True,
                          "suggested_safe_text": response, "reasoning_summary": "安全。", "violations": []}],
    }))

    def send(path, *, payload=None, timeout_seconds):
        if path == "/health":
            return {"send_channel_enabled": True, "auto_send_reply": True}
        assert path == "/send"
        deliveries.append(payload)
        return {"ok": True, "state": "accepted"}

    monkeypatch.setattr(app, "request_local_json", send)
    original_handler = app.handle_wechaty_casual_chat

    def reply(message, *, trace_id, runtime, gate_decision):
        calls.append(message)
        result = original_handler(message, trace_id=trace_id, runtime=runtime, gate_decision=gate_decision)
        if late_fragment:
            queue_fragment(4, "问过了，明天有人来看")
        return result

    def queue_fragment(i, body):
        message_id = f"fragment-{i}"
        payload = {
            "channel": "agent-wechat", "conversation_id": conversation_id, "message_id": message_id,
            "source_message_id": message_id, "sender_id": contact, "is_room": False,
            "text": body, "self_message": False, "message_type": 7,
            "native_reply_source": {"account_id": account, "thread_id": contact, "cursor": str(i)},
            "payload": {"id": message_id, "threadId": contact, "account_id": account, "direction": "incoming"},
        }
        assert app.queue_agent_wechat_raw(runtime, payload, deferred_reply=True)["accepted"]

    monkeypatch.setattr(app, "handle_wechaty_casual_chat", reply)
    for i, body in enumerate(("新装的雨棚听说会漏水", "好在今天没下雨", "我明天才去检查"), start=1):
        queue_fragment(i, body)

    scheduler = PendingInputScheduler(store=store, handler=app.handle_due_pending_input_batch,
                                      trace_recorder=runtime.trace_recorder)
    for i in range(1, 4):
        queued = store.next_channel_ingress(set())
        app.process_queued_channel_raw(runtime, queued)
        batch = store.pending_input_batch(conversation_id, contact)
        assert batch is not None and len(batch.fragments) == i
        assert batch.decision["dispatch_mode"] == app.NATIVE_CASUAL_QUIET_MODE
        assert not calls and not deliveries
        if i < 3:
            assert scheduler.run_due_once(at=batch.quiet_deadline + timedelta(seconds=1)) == 0

    assert scheduler.run_due_once(at=batch.quiet_deadline - timedelta(milliseconds=1)) == 0
    assert scheduler.run_due_once(at=batch.quiet_deadline) == 1
    assert scheduler.run_due_once(at=batch.quiet_deadline) == 0
    assert [message.text for message in calls] == ["新装的雨棚听说会漏水\n好在今天没下雨\n我明天才去检查"]
    if late_fragment:
        assert not deliveries
        assert store.next_channel_ingress(set())["source_message_id"] == "fragment-4"
    else:
        assert [item["text"] for item in deliveries] == [response]
        assert deliveries[0]["native_reply_source"]["cursor"] == "3"
        assert store.recent_turns(conversation_id)[-1].metadata["delivery_status"] == "delivered"


def test_agent_wechat_deferred_reply_recovers_and_preserves_conversation_order(local_agent_http, monkeypatch) -> None:
    """回复及重试时间落库；适配器短暂不可用时，其他会话继续处理。"""
    from time import time
    from types import SimpleNamespace
    from urllib.error import URLError
    from urllib.request import Request, urlopen

    from mahjong_agent_runtime.stores.sqlite import channel_observations

    context = local_agent_http
    app = context.app
    context.server.channel_ingress_worker = SimpleNamespace(wake=lambda: None)
    routes = []
    deliveries = []

    def route(payload, *, trace_id):
        routes.append(payload["message_id"])
        return {
            "routed_to_agent": True,
            "agent_result": {"final_reply": "收到。", "tool_results": ["只在 Runtime 保存"]},
        }

    def deliver(path, *, payload, timeout_seconds):
        assert path == "/deliver-ingress-reply"
        assert timeout_seconds > 60
        deliveries.append(payload)
        if len(deliveries) == 1:
            raise URLError("模拟适配器暂时无法连接")
        return {"ok": True, "delivery_state": "accepted"}

    monkeypatch.setattr(app, "channel_route_handler", lambda _channel: route)
    monkeypatch.setattr(app, "request_local_json", deliver)

    def post(source_id, sender, *, deferred=True, self_message=False):
        payload = {"message_id": source_id, "sender_id": sender, "self_message": self_message, "text": "你好"}
        headers = {**context.headers, "Prefer": "respond-async"}
        if deferred:
            headers["X-Mahjong-Agent-Wechat-Deferred-Reply"] = "1"
        request = Request(
            f"{context.base}/api/channels/agent-wechat/raw", headers=headers,
            data=app.json.dumps(payload).encode(),
        )
        with urlopen(request, timeout=2) as response:
            assert response.status == 202
            return app.json.load(response)

    assert post("first", "customer")["deferred_reply"] is True
    assert post("later", "customer")["deferred_reply"] is True
    assert "deferred_reply" not in post("other", "elsewhere", deferred=False)
    assert post("self", "owner", self_message=True)["deferred_reply"] is True

    runtime = context.runtime
    app.process_queued_channel_raw(runtime, runtime.store.next_channel_ingress(set()))
    assert routes == ["first"]
    assert not deliveries

    reopened = SQLiteAgentStore(context.store_path)
    recovered_runtime = SimpleNamespace(store=reopened, trace_recorder=runtime.trace_recorder)
    ready = reopened.next_channel_ingress(set())
    assert ready["route_mode"] == "ingress_reply_pending"
    assert ready["source_message_id"] == "first"
    app.process_queued_channel_raw(recovered_runtime, ready)
    skipped = reopened.next_channel_ingress(set())
    assert skipped["source_message_id"] == "other"
    app.process_queued_channel_raw(recovered_runtime, skipped)
    owner = reopened.next_channel_ingress(set())
    assert owner["source_message_id"] == "self"
    app.process_queued_channel_raw(recovered_runtime, owner)
    assert reopened.next_channel_ingress(set()) is None
    assert reopened.channel_ingress_record("agent-wechat", "first") is None

    monkeypatch.setattr(channel_observations, "time", lambda: time() + 10)
    ready = reopened.next_channel_ingress(set())
    assert ready["route_mode"] == "ingress_reply_pending"
    app.process_queued_channel_raw(recovered_runtime, ready)
    assert reopened.next_channel_ingress(set())["source_message_id"] == "later"
    assert routes == ["first", "other", "self"]
    assert deliveries[0] == deliveries[1]
    assert deliveries[0]["result"]["route_result"] == {"agent_result": {"final_reply": "收到。"}}
    persisted = next(row for row in reopened.list_channel_observations() if row["source_message_id"] == "first")
    assert persisted["route_mode"] == "agent"
    assert persisted["payload"]["deferred_reply"] == {"delivery_state": "accepted", "attempts": 2}
    duplicate = post("first", "customer")
    assert duplicate["deduplicated"] is True and duplicate["deferred_reply"] is True
    assert reopened.channel_ingress_record("agent-wechat", "first") is None


@pytest.mark.parametrize("http_status", [400, 413])
def test_agent_wechat_permanent_reply_rejection_unblocks_same_conversation(
    local_agent_http, monkeypatch, http_status,
) -> None:
    """确定的请求拒绝保留路由结果，后续同一会话继续消费。"""
    from urllib.error import HTTPError

    app = local_agent_http.app
    runtime = local_agent_http.runtime
    routes = []
    deliveries = []

    def route(payload, *, trace_id):
        source = payload["message_id"]
        routes.append(source)
        return {"routed_to_agent": True, "agent_result": {"final_reply": f"收到 {source}。"}}

    def deliver(path, *, payload, timeout_seconds):
        assert path == "/deliver-ingress-reply"
        source = payload["payload"]["message_id"]
        deliveries.append(source)
        if source == "first":
            raise HTTPError("http://localhost/deliver-ingress-reply", http_status, "invalid input", {}, None)
        return {"ok": True, "delivery_state": "accepted"}

    monkeypatch.setattr(app, "channel_route_handler", lambda _channel: route)
    monkeypatch.setattr(app, "request_local_json", deliver)
    for source in ("first", "later"):
        app.queue_agent_wechat_raw(
            runtime, {"message_id": source, "sender_id": "customer", "text": source},
            deferred_reply=True,
        )

    store = runtime.store
    for expected_source, expected_mode in (
        ("first", "ingress_queue"),
        ("first", "ingress_reply_pending"),
        ("later", "ingress_queue"),
        ("later", "ingress_reply_pending"),
    ):
        pending = store.next_channel_ingress(set())
        assert pending is not None
        assert (pending["source_message_id"], pending["route_mode"]) == (expected_source, expected_mode)
        app.process_queued_channel_raw(runtime, pending)

    assert store.next_channel_ingress(set()) is None
    assert routes == deliveries == ["first", "later"]
    first = next(row for row in store.list_channel_observations() if row["source_message_id"] == "first")
    assert first["route_mode"] == "agent"
    assert first["payload"]["route_result"]["agent_result"]["final_reply"] == "收到 first。"
    assert first["payload"]["deferred_reply"] == {
        "delivery_state": "withheld",
        "attempts": 1,
        "reason": "adapter_rejected_callback",
        "http_status": http_status,
    }
    trace_event = next(
        event for event in runtime.trace_recorder.get_trace(first["trace_id"])
        if event.step == "agent_wechat_ingress_reply_completed"
    )
    assert trace_event.content == {
        "delivery_state": "withheld", "reason": "adapter_rejected_callback", "http_status": http_status,
    }


def test_agent_wechat_customer_message_routes_with_incoming_only_default(
    local_agent_http, monkeypatch
) -> None:
    from urllib.request import Request, urlopen

    context = local_agent_http
    monkeypatch.delenv("MAHJONG_AGENT_WECHAT_ROUTE_SCOPE", raising=False)
    monkeypatch.delenv("MAHJONG_WECHATY_ROUTE_SCOPE", raising=False)
    monkeypatch.delenv("MAHJONG_WECHATY_AUTO_ROUTE_TO_AGENT", raising=False)
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_INPUT_AGGREGATION_ENABLED", "false")
    monkeypatch.setenv("MAHJONG_WECHATY_INPUT_GATE_ENABLED", "false")
    context.client.outputs.append(context.app.json.dumps({
        "goal": "回应问候",
        "objective_status": "completed",
        "reasoning_summary": "客户问候，不涉及业务副作用。",
        "reply_to_user": "你好。",
        "tool_calls": [],
        "needs_human": False,
        "stop_reason": {
            "can_stop": True,
            "why": "已回应问候。",
            "pending_work": [],
            "depends_on_tool_results": False,
        },
        "badcase": None,
    }, ensure_ascii=False))
    payload = {
        "conversation_id": "wechaty:contact:wxid_agent_customer",
        "sender_id": "wxid_agent_customer",
        "sender_name": "阿强",
        "message_id": "agent_wechat_customer_001",
        "text": "你好",
        "self_message": False,
        "message_type": 7,
    }
    body = context.app.json.dumps(payload, ensure_ascii=False).encode()
    endpoint = f"{context.base}/api/channels/agent-wechat/raw"
    with urlopen(Request(endpoint, data=body, headers=context.headers), timeout=5) as response:
        received = context.app.json.load(response)
    assert received["route_result"]["audit"]["route_scope"] == "incoming_only"
    assert received["route_result"]["audit"]["routed_to_agent"] is True
    assert received["route_result"]["audit"]["reason"] == "valid_text_message"
    assert len(context.client.calls) == 1


def test_wechaty_customer_message_keeps_self_only_default(local_agent_http, monkeypatch) -> None:
    from urllib.request import Request, urlopen

    context = local_agent_http
    monkeypatch.delenv("MAHJONG_AGENT_WECHAT_ROUTE_SCOPE", raising=False)
    monkeypatch.delenv("MAHJONG_WECHATY_ROUTE_SCOPE", raising=False)
    monkeypatch.delenv("MAHJONG_WECHATY_AUTO_ROUTE_TO_AGENT", raising=False)
    payload = {
        "conversation_id": "wechaty:contact:wxid_wechaty_customer",
        "sender_id": "wxid_wechaty_customer",
        "sender_name": "张哥",
        "message_id": "wechaty_customer_001",
        "text": "你好",
        "self_message": False,
        "message_type": 7,
    }
    body = context.app.json.dumps(payload, ensure_ascii=False).encode()
    endpoint = f"{context.base}/api/channels/wechaty/raw"
    with urlopen(Request(endpoint, data=body, headers=context.headers), timeout=5) as response:
        received = context.app.json.load(response)
    assert received["route_result"]["audit"]["route_scope"] == "self_only"
    assert received["route_result"]["audit"]["routed_to_agent"] is False
    assert received["route_result"]["audit"]["reason"] == "non_self_message_in_self_only_scope"
    assert context.client.calls == []


def test_channel_manifest_advertises_generic_and_agent_wechat_entries(local_agent_http) -> None:
    from urllib.request import Request, urlopen

    context = local_agent_http
    with urlopen(Request(f"{context.base}/api/runtime", headers=context.headers), timeout=5) as response:
        manifest = context.app.json.load(response)
    assert manifest["endpoints"]["wechaty_raw"] == ["/api/channels/wechaty/raw"]
    assert manifest["endpoints"]["channel_raw"] == ["/api/channels/{channel}/raw"]
    assert manifest["endpoints"]["agent_wechat_raw"] == ["/api/channels/agent-wechat/raw"]


def test_unknown_channel_raw_endpoint_stays_404(local_agent_http) -> None:
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    context = local_agent_http
    for channel in ("acme", "wechaty-plus", "Agent-WeChat"):
        with pytest.raises(HTTPError) as exc:
            urlopen(
                Request(
                    f"{context.base}/api/channels/{channel}/raw",
                    data=b'{"message_id":"x","text":"hello"}',
                    headers=context.headers,
                ),
                timeout=5,
            )
        assert exc.value.code == 404
    assert context.runtime.store.list_channel_observations() == []


@pytest.mark.parametrize(("channel", "nickname", "expected"), [
    ("agent-wechat", "公开昵称", "公开昵称"),
    ("agent-wechat", None, "群成员"),
    ("agent-wechat", {"remark": "私有备注"}, "群成员"),
    ("wechaty", "其它公开名称", "旧适配器公开名称"),
])
def test_managed_group_public_sender_name_excludes_native_private_remark(local_agent_http, channel, nickname, expected):
    """群业务和共享身份只接受原生公开昵称，控制台显示名称不进入可外发上下文。"""
    context = local_agent_http
    payload = {
        "is_room": True,
        "room": {"id": "123@chatroom"},
        "sender_name": "私有备注",
        "talker": {
            "name": "私有备注" if channel == "agent-wechat" else "旧适配器公开名称",
            "payload": {"nickname": nickname, "remark": "私有备注", "roomDisplayName": "本群名片"},
        },
    }
    message = UserMessage(conversation_id="wechaty:room:123@chatroom", sender_id="wxid_sample",
                          sender_name="私有备注", text="合成消息", message_id="sample-message")

    public_message = context.app.build_managed_group_message(context.runtime, payload, message, channel=channel)
    identity = context.runtime.store.get_channel_identity(channel, message.sender_id)

    assert public_message.sender_name == identity.public_name == expected
    assert "私有备注" not in str(public_message.to_dict())
    assert "本群名片" not in str(identity.to_dict())


def test_agent_wechat_group_permission_observes_a_previously_managed_group(local_agent_http, tmp_path, monkeypatch):
    """观察群保留原始证据，明确托管或旧的定向回复授权都不能覆盖撤权。"""
    import json
    from urllib.request import Request, urlopen

    from mahjong_agent_runtime.group_chat.stake_confirmation import (
        BOARD_CANDIDATE_UPDATE_TASK_TYPE, BOARD_STAKE_CONFIRMATION_TASK_TYPE, BOARD_STAKE_UPDATE_TASK_TYPE,
    )
    from mahjong_agent_runtime.models import ScheduledAgentTask, now

    app = local_agent_http.app
    runtime = local_agent_http.runtime
    room_id = "123@chatroom"
    policy = tmp_path / "contact_permissions.local.json"
    policy.write_text(json.dumps({
        "account_id": "sample-group-account", "recipients": [], "observed": [],
        "group_permissions": [{"thread_id": room_id, "label": "示例群", "mode": "observe"}],
    }), encoding="utf-8")
    reply_policy = tmp_path / "group-replies.json"
    reply_policy.write_text(json.dumps({
        "account_id": "sample-group-account", "room_ids": [room_id], "enabled_at": "2025-01-01T00:00:00Z",
    }), encoding="utf-8")
    monkeypatch.delenv("MAHJONG_AGENT_WECHAT_OUTBOUND_ALLOWLIST_PATH", raising=False)
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_GROUP_REPLY_POLICY_PATH", str(reply_policy))
    monkeypatch.setenv("MAHJONG_WECHATY_MANAGED_ROOM_IDS", room_id)
    monkeypatch.setattr(app, "get_group_message_handler", lambda *args, **kw: pytest.fail("观察群不应执行业务"))
    monkeypatch.setattr(app, "agent_wechat_group_reply_topic", lambda *args, **kw: pytest.fail("观察群不应分类"))
    monkeypatch.setattr(app, "request_local_json", lambda *args, **kw: pytest.fail("观察群不应调用外发适配器"))

    assert not app.agent_wechat_private_observe_only("wxid_sample", "sample-group-account")
    assert app.agent_wechat_group_permission_rule("321@chatroom", "sample-group-account") == {}
    assert app.agent_wechat_group_permission_rule(room_id, "unexpected-account") is None

    payload = {
        "is_room": True, "room": {"id": room_id, "topic": "示例群"},
        "conversation_id": f"wechaty:room:{room_id}", "message_id": "observe-group-1",
        "source_message_id": "observe-group-1", "sender_id": "wxid_sample", "text": "合成消息",
        "message_type": 7,
        "payload": {"threadId": room_id, "account_id": "sample-group-account", "direction": "incoming",
                    "native_type": 1, "createdAt": "2026-01-02T12:00:00Z"},
    }
    request = Request(f"{local_agent_http.base}/api/channels/agent-wechat/raw", headers=local_agent_http.headers,
                      data=json.dumps(payload).encode())
    with urlopen(request, timeout=5) as response:
        result = json.load(response)["route_result"]
    assert result["observe_only"] is True and result["routed_to_agent"] is False and result["agent_result"] is None
    assert not result.get("scoped_group_reply")
    observation, = runtime.store.list_channel_observations(channel="agent-wechat")
    assert observation["route_mode"] == "observe_only" and observation["text"] == "合成消息"
    assert app.AGENT_WECHAT_RAW_LOG_PATH.exists() and not local_agent_http.client.calls
    assert runtime.store.get_group_room_policy(room_id) is None

    # 已经排队的原托管看板只暂停，群授权文件与数据库的托管配置不被控制台覆盖。
    runtime.store.upsert_group_room_policy(app.GroupRoomPolicy(room_id=room_id))
    task, _ = runtime.store.ensure_group_board_publish_task(room_id=room_id, due_at=now(), trace_id="pending-board")
    with pytest.raises(app.ScheduledTaskDeferred, match="group_board_policy_paused"):
        app.handle_due_scheduled_agent_task(task, "after-group-revocation")
    for task_type in (BOARD_STAKE_CONFIRMATION_TASK_TYPE, BOARD_STAKE_UPDATE_TASK_TYPE, BOARD_CANDIDATE_UPDATE_TASK_TYPE):
        notification = ScheduledAgentTask(
            task_id=task_type, task_type=task_type, aggregate_type="game", aggregate_id="sample-game",
            conversation_id="wechaty:contact:wxid_sample", subject_id="wxid_sample", subject_name="示例",
            due_at=now(), idempotency_key=task_type, payload={"room_id": room_id},
        )
        with pytest.raises(app.ScheduledTaskDeferred, match="group_board_policy_paused"):
            app.handle_due_scheduled_agent_task(notification, "after-group-revocation")
    assert runtime.store.get_group_room_policy(room_id).outbound_enabled
    messenger = app.WechatyGroupMessenger(runtime)
    assert messenger.send_private_message("wxid_sample", "合成消息", metadata={"origin_room_id": room_id}).startswith("suppressed:")
    with pytest.raises(app.ScheduledTaskDeferred, match="group_permission_observe_only"):
        messenger.send_group_message(room_id, "合成看板", metadata={"message_type": "game_board"})


def test_unauthorized_native_group_cannot_enter_private_routing(local_agent_http, tmp_path, monkeypatch):
    """没有群授权文件时，原生群默认只留证据；精确旧观察名单仍可做只读分析。"""
    import json
    from urllib.request import Request, urlopen

    context = local_agent_http
    app = context.app
    room_id = "123@chatroom"
    for key in ("MAHJONG_WECHATY_MANAGED_ROOMS_PATH", "MAHJONG_WECHATY_OBSERVE_ONLY_ROOMS_PATH",
                "MAHJONG_AGENT_WECHAT_GROUP_REPLY_POLICY_PATH", "MAHJONG_AGENT_WECHAT_TOPIC_OBSERVE_PATH"):
        monkeypatch.setenv(key, str(tmp_path / "missing-legacy-file.json"))
    for key in ("MAHJONG_WECHATY_MANAGED_ROOM_IDS", "MAHJONG_WECHATY_MANAGED_ROOM_TOPICS",
                "MAHJONG_WECHATY_OBSERVE_ONLY_ROOM_IDS", "MAHJONG_WECHATY_OBSERVE_ONLY_ROOM_TOPICS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_ROUTE_SCOPE", "incoming_only")
    monkeypatch.setenv("MAHJONG_WECHATY_AUTO_ROUTE_TO_AGENT", "true")
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_INPUT_AGGREGATION_ENABLED", "true")
    monkeypatch.setattr(app, "route_user_message_with_aggregation", lambda *a, **kw: pytest.fail("群聊不应进入私聊聚合"))
    classified = []

    def analyze(message, **kwargs):
        classified.append(message.message_id)
        return {"action": "process_business"}

    monkeypatch.setattr(app, "run_wechaty_input_gate", analyze)

    def post(message_id, *, is_room=True):
        kind = "room" if is_room else "contact"
        payload = {
            "is_room": is_room, "room": {"id": room_id} if is_room else None,
            "conversation_id": f"wechaty:{kind}:{room_id}", "message_id": message_id,
            "source_message_id": message_id, "sender_id": "wxid_sample", "text": "合成消息", "message_type": 7,
            "payload": {"threadId": room_id, "account_id": "sample-account", "direction": "incoming"},
        }
        request = Request(f"{context.base}/api/channels/agent-wechat/raw", headers=context.headers,
                          data=json.dumps(payload).encode())
        with urlopen(request, timeout=5) as response:
            return json.load(response)["route_result"]

    for result in (post("unconfigured-group"), post("mislabeled-group", is_room=False)):
        assert result["observe_only"] and not result["routed_to_agent"]
        assert result["agent_result"] is None and not result.get("observation")
    assert classified == []

    monkeypatch.setenv("MAHJONG_WECHATY_OBSERVE_ONLY_ROOM_IDS", room_id)
    named = post("named-observer")
    assert named["observe_only"] and named["agent_result"] is None
    assert named["observation"]["state_mutation_allowed"] is False
    assert named["observation"]["outbound_allowed"] is False
    assert classified == ["named-observer"] and not context.client.calls
    assert len(context.runtime.store.list_channel_observations(channel="agent-wechat")) == 3


def test_agent_wechat_group_auto_reply_has_independent_scope(tmp_path, monkeypatch):
    """单群授权只允许经审查的当期业务回复，未知群和旧消息保留为观察。"""
    import json
    from types import SimpleNamespace

    app = load_app_module()
    room_id = "123@chatroom"
    policy = tmp_path / "group-replies.json"
    policy.write_text(json.dumps({
        "account_id": "sample-account",
        "room_ids": [room_id],
        "enabled_at": "2026-01-02T11:00:00Z",
        "replay_message_id": "msg-1",
    }))
    private_policy = tmp_path / "private-contact-policy.json"
    private_policy.write_text(json.dumps({
        "account_id": "sample-account", "managed_direct": True, "recipients": [], "observed": [],
        "group_permissions": [{"thread_id": "321@chatroom", "label": "手动发送群", "mode": "allowlist"}],
    }), encoding="utf-8")
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_OUTBOUND_ALLOWLIST_PATH", str(private_policy))
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_ACCOUNT", "sample-account")
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_GROUP_REPLY_POLICY_PATH", str(policy))
    monkeypatch.setattr(app, "TRACE_PATH", tmp_path / "trace.jsonl")
    monkeypatch.setattr(app, "route_scope_for_channel", lambda _: "self_only")
    monkeypatch.setattr(app, "wechaty_agent_whitelist_hits", lambda _: [])
    monkeypatch.setattr(app, "wechaty_managed_room_hits", lambda _: [])
    monkeypatch.setattr(app, "wechaty_observe_only_room_hits", lambda _: [])
    monkeypatch.setattr(app, "get_runtime", lambda: SimpleNamespace(trace_recorder=InMemoryTraceRecorder()))
    monkeypatch.setattr(app, "run_wechaty_input_gate", lambda *args, **kwargs: pytest.fail("定向群无需重复分流"))
    intent = "query"
    reviewed = []

    def classify(*args, **kwargs):
        return {"intent": intent}

    def review(message, **kwargs):
        reviewed.append(message.message_id)
        return "你大概几点方便？"

    monkeypatch.setattr(app, "agent_wechat_group_reply_topic", classify)
    monkeypatch.setattr(app, "reviewed_agent_wechat_group_reply", review)

    def route(room=room_id, trace="scope", **raw_overrides):
        return app.route_channel_raw_to_agent({
            "is_room": True,
            "room": {"id": room},
            "message_id": "msg-1",
            "source_message_id": "msg-1",
            "sender_id": "sample-sender",
            "text": "老板明天有局吗？",
            "payload": {
                "threadId": room, "account_id": "sample-account", "direction": "incoming",
                "native_type": 1, "createdAt": "2026-01-02T12:00:00Z", **raw_overrides,
            },
        }, trace_id=trace, channel="agent-wechat")

    result = route()
    assert result["observe_only"] is True
    assert result["agent_result"] is None
    assert result["observation"]["state_mutation_allowed"] is False
    assert result["scoped_group_reply"] == {
        "room_id": room_id, "source_message_id": "msg-1", "text": "你大概几点方便？",
    }
    intent = "chitchat"
    assert route()["scoped_group_reply"] == {}
    for denied in (
        route(room="321@chatroom"),  # 只加入外发白名单，不附带定向回复授权。
        route(account_id="other"),
        route(threadId="321@chatroom"),
        route(direction="outgoing"),
        route(createdAt="2026-01-01T12:00:00Z"),
    ):
        assert denied["observe_only"] is True
        assert not denied.get("scoped_group_reply")
    assert reviewed == ["msg-1"]


def test_agent_wechat_group_management_and_auto_reply_take_effect_independently(local_agent_http, tmp_path, monkeypatch):
    """真实入站在分别授权的时间点进入对应路径，使用公开审查回复且不新授予看板外发。"""
    import json
    from types import SimpleNamespace
    from urllib.request import Request, urlopen

    context = local_agent_http
    app = context.app
    room_id = "123@chatroom"
    account = "sample-account"
    policy_path = tmp_path / "contact_permissions.local.json"
    scoped_policy_path = tmp_path / "legacy-group-replies.json"
    scoped_policy_path.write_text(json.dumps({
        "account_id": account, "room_ids": [room_id], "enabled_at": "2026-01-02T10:00:00Z",
    }), encoding="utf-8")
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_OUTBOUND_ALLOWLIST_PATH", str(policy_path))
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_GROUP_REPLY_POLICY_PATH", str(scoped_policy_path))
    monkeypatch.setenv("MAHJONG_WECHATY_MANAGED_ROOMS_PATH", str(tmp_path / "missing-managed.json"))
    monkeypatch.setenv("MAHJONG_WECHATY_MANAGED_ROOM_IDS", "321@chatroom")
    monkeypatch.setenv("MAHJONG_WECHATY_MANAGED_ROOM_TOPICS", "")
    monkeypatch.setenv("MAHJONG_WECHATY_MANAGED_ROOM_OUTBOUND_ENABLED", "true")
    monkeypatch.setattr(app, "route_scope_for_channel", lambda _: "self_only")
    monkeypatch.setattr(app, "request_local_json", lambda *args, **kwargs: pytest.fail("不能调用真实外发适配器"))
    handled = []
    classified = []
    reviewed = []

    def classify(*, message, **kwargs):
        classified.append(message.message_id)
        return {"intent": "query", "topic_action": "none", "public_summary": {}}

    topic_classifier = SimpleNamespace(classify=classify, max_candidates=5)
    monkeypatch.setattr(app, "SemanticTopicClassifier", lambda _: topic_classifier)

    def set_group_permissions(managed, auto_reply):
        policy_path.write_text(json.dumps({
            "account_id": account, "recipients": [], "observed": [],
            "group_permissions": [{
                "thread_id": room_id, "label": "示例群", "mode": "inherit", "managed": managed, "auto_reply": auto_reply,
                "managed_at": "2026-01-02T11:00:00Z", "auto_reply_at": "2026-01-02T12:00:00Z",
            }],
        }), encoding="utf-8")

    def group_handler(_, *, suppress_group_reply=False):
        def handle(message, *, trace_id):
            handled.append((message.message_id, suppress_group_reply))
            app.TopicRoutingService(context.runtime.store, topic_classifier).classify(message, trace_id=trace_id)
            return SimpleNamespace(action="agent_loop", game_id=None, reply="托管内部结果", detail={})

        return SimpleNamespace(handle=handle)

    def review(message, **kwargs):
        reviewed.append(message.message_id)
        return "你大概几点方便？"

    monkeypatch.setattr(app, "get_group_message_handler", group_handler)
    monkeypatch.setattr(app, "reviewed_agent_wechat_group_reply", review)

    def post(message_id, created_at, *, conversation_room=room_id):
        payload = {
            "is_room": True, "room": {"id": room_id}, "conversation_id": f"wechaty:room:{conversation_room}",
            "message_id": message_id, "source_message_id": message_id, "sender_id": "wxid_sample",
            "text": "老板明天有局吗？", "message_type": 7,
            "payload": {"threadId": room_id, "account_id": account, "direction": "incoming",
                        "native_type": 1, "createdAt": created_at},
        }
        request = Request(f"{context.base}/api/channels/agent-wechat/raw", headers=context.headers,
                          data=json.dumps(payload).encode())
        with urlopen(request, timeout=5) as response:
            return json.load(response)["route_result"]

    set_group_permissions(True, True)
    early = post("before-both", "2026-01-02T10:59:00Z")
    assert early["observe_only"] is True and not early.get("scoped_group_reply")
    assert context.runtime.store.get_group_room_policy(room_id) is None
    assert handled == reviewed == []

    mid = post("manage-only", "2026-01-02T11:30:00Z")
    assert mid["managed_group"] is True and not mid.get("scoped_group_reply")
    assert handled == [("manage-only", True)] and not reviewed
    stored_policy = context.runtime.store.get_group_room_policy(room_id)
    assert stored_policy.managed and not stored_policy.outbound_enabled and stored_policy.channel == "agent-wechat"

    both = post("both", "2026-01-02T12:30:00Z")
    assert both["managed_group"] and both["agent_result"] is None
    assert both["scoped_group_reply"] == {
        "room_id": room_id, "source_message_id": "both", "text": "你大概几点方便？",
    }
    forged = post("wrong-group-conversation", "2026-01-02T12:31:00Z", conversation_room="321@chatroom")
    assert forged["observe_only"] and not forged.get("scoped_group_reply")

    monkeypatch.setenv("MAHJONG_WECHATY_MANAGED_ROOM_IDS", room_id)
    set_group_permissions(False, True)
    reply = post("explicit-reply-only", "2026-01-02T12:31:00Z")
    assert not reply.get("managed_group") and reply["scoped_group_reply"]["source_message_id"] == "explicit-reply-only"
    set_group_permissions(True, False)
    manage = post("explicit-manage-only", "2026-01-02T12:32:00Z")
    assert manage["managed_group"] and not manage.get("scoped_group_reply")
    set_group_permissions(False, False)
    neither = post("explicit-neither", "2026-01-02T12:33:00Z")
    assert neither["observe_only"] and not neither.get("scoped_group_reply")
    assert handled == [("manage-only", True), ("both", True), ("explicit-manage-only", True)]
    assert classified == ["manage-only", "both", "explicit-reply-only", "explicit-manage-only"]
    assert reviewed == ["both", "explicit-reply-only"]
    assert context.app.AGENT_WECHAT_RAW_LOG_PATH.read_text(encoding="utf-8").count("\n") == 7
