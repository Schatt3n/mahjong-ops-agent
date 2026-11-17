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


def test_runtime_http_manifest_and_console_only_advertise_supported_channels(local_agent_http) -> None:
    from urllib.request import Request, urlopen

    context = local_agent_http
    with urlopen(Request(f"{context.base}/api/runtime", headers=context.headers), timeout=5) as response:
        manifest = context.app.json.load(response)
    with urlopen(f"{context.base}/", timeout=5) as response:
        html = response.read().decode("utf-8")
        assert "mahjong_agent_token=" in response.headers["Set-Cookie"]
        assert "HttpOnly" in response.headers["Set-Cookie"]
    for retired_channel in ("hermes", "astrbot"):
        assert retired_channel not in context.app.json.dumps(manifest).lower()
        assert retired_channel not in html.lower()
    assert manifest["endpoints"]["wechaty_raw"] == ["/api/channels/wechaty/raw"]
    assert manifest["endpoints"]["message"] == ["/api/message"]
    assert "/api/channels/wechaty/raw" in html
    assert "/api/message" in html
    assert context.client.calls == []


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
    manifest_text = app.index_html()

    assert "测试与回放" in page
    assert "重跑确定性并发测试" in page
    assert "调用真实 DeepSeek 回放" in page
    assert "record_candidate_reply" in page
    assert "/api/test-observability/run" in page
    assert 'href="/tests"' in manifest_text


def test_operator_console_exposes_simulation_observability_page() -> None:
    app = load_app_module()

    page = app.simulation_dashboard_html()
    manifest_text = app.index_html()

    assert "模拟聊天室 · 对话审阅" in page
    assert "/api/simulations" in page
    assert "业务对话" in page
    assert "闲聊插入" in page
    assert "回到业务" in page
    assert "模拟消息从哪里来" in page
    assert "Agent 实际执行" in page
    assert "导出完整对话 MD" in page
    assert "模型识别尚未独立验收" in page
    assert 'href="/simulation"' in manifest_text


def test_observability_runner_rejects_non_allowlisted_commands() -> None:
    app = load_app_module()

    try:
        app.run_fixed_suite("rm -rf /")
    except ValueError as exc:
        assert "unsupported test suite" in str(exc)
    else:
        raise AssertionError("arbitrary commands must not reach the test runner")


def test_main_app_does_not_expose_legacy_trial_defaults() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "boss_trial" not in text
    assert "runtime_trial" in text


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


def test_index_html_has_reset_state_button() -> None:
    app = load_app_module()
    html = app.index_html()

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
        route(room="321@chatroom"),
        route(account_id="other"),
        route(threadId="321@chatroom"),
        route(direction="outgoing"),
        route(createdAt="2026-01-01T12:00:00Z"),
    ):
        assert denied["observe_only"] is True
        assert not denied.get("scoped_group_reply")
    assert reviewed == ["msg-1"]
