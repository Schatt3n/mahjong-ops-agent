"""Layer 4: bidirectional HTTP adapter and per-user inboxes."""

from __future__ import annotations

import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mahjong_agent_runtime import (  # noqa: E402
    AgentRuntime,
    InMemoryTraceRecorder,
    OpenAICompatibleAgentClient,
    SQLiteAgentStore,
    ToolGateway,
    UserMessage,
)
from mahjong_agent_runtime.env import load_dotenv_defaults  # noqa: E402

try:
    from .behavior_policy import SimulationAction
    from .sim_factory import VirtualUser
    from .sim_state import (
        InboxMessage,
        InMemorySimulationStateBackend,
        ReplyGate,
        SimulationStateBackend,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from behavior_policy import SimulationAction  # type: ignore
    from sim_factory import VirtualUser  # type: ignore
    from sim_state import (  # type: ignore
        InboxMessage,
        InMemorySimulationStateBackend,
        ReplyGate,
        SimulationStateBackend,
    )


SQLITE_LOCK_MARKERS = (
    "database is locked",
    "database table is locked",
    "database schema is locked",
)
MENTION_PATTERN = re.compile(r"@([\u4e00-\u9fa5a-zA-Z0-9]+)")


@dataclass(slots=True)
class RequestOutcome:
    action: SimulationAction
    sent: bool
    latency_ms: float = 0.0
    status_code: int = 0
    response: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    sent_at: float | None = None
    inbox_deliveries: int = 0
    next_speaker_only: str | None = None

    @property
    def sqlite_lock_error(self) -> bool:
        searchable = f"{self.error} {json.dumps(self.response, ensure_ascii=False)}".lower()
        return any(marker in searchable for marker in SQLITE_LOCK_MARKERS)


class StaticAgentLLMClient:
    """Seeded mock: one ToolCall followed by a varied terminal response.

    The same seed remains reproducible for regression tests, while periodic
    simulations use a fresh seed so both user utterances and Agent replies vary
    between runs without spending provider tokens.
    """

    def __init__(self, *, seed: int = 42) -> None:
        self._lock = threading.Lock()
        self.call_count = 0
        self._random = random.Random(seed)

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        del trace_id, timeout_seconds
        with self._lock:
            self.call_count += 1
        payload = self._context_payload(messages)
        previous_tool_results = payload.get("previous_tool_results") or []
        action = self._terminal_action(payload) if previous_tool_results else self._tool_action()
        return json.dumps(action, ensure_ascii=False)

    @staticmethod
    def _context_payload(messages: list[dict[str, str]]) -> dict[str, Any]:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            try:
                payload = json.loads(message.get("content") or "{}")
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return {}

    @staticmethod
    def _tool_action() -> dict[str, Any]:
        return {
            "goal": "查询当前可用麻将局",
            "objective_status": "needs_tool",
            "reasoning_summary": "模拟压测固定先读取当前局池。",
            "objective_state": {
                "current_phase": "query_existing_games",
                "known_facts": {},
                "missing_facts": [],
                "blockers": [],
            },
            "objective_plan": [
                {
                    "step_id": "search_pool",
                    "title": "查询当前局池",
                    "status": "in_progress",
                    "tool": "search_current_games",
                    "depends_on": [],
                    "decision_rule": "读取工具结果后结束本轮模拟。",
                }
            ],
            "plan_revision_reason": "模拟客户端使用固定只读工具调用覆盖后端链路。",
            "reply_to_user": "",
            "tool_calls": [
                {
                    "call_id": "search_pool",
                    "depends_on": [],
                    "name": "search_current_games",
                    "arguments": {"requirement": {}, "limit": 5},
                    "reason": "压测 Tool Gateway 和 SQLite 读取链路。",
                }
            ],
            "needs_human": False,
            "stop_reason": {
                "can_stop": False,
                "why": "必须等待局池查询结果。",
                "pending_work": ["查询当前局池"],
                "depends_on_tool_results": True,
            },
            "badcase": None,
        }

    def _terminal_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        current_message = payload.get("current_message") or {}
        sender_name = str(current_message.get("sender_name") or "").strip()
        prefix = f"@{sender_name} " if sender_name else ""
        reply = prefix + self._random.choice(
            (
                "你几点方便？",
                "大概几点能到？",
                "你这边几人？",
                "打多大的？",
                "烟有要求吗？",
            )
        )
        return {
            "goal": "查询当前可用麻将局",
            "objective_status": "completed",
            "reasoning_summary": "局池查询已完成，本轮模拟结束。",
            "objective_state": {
                "current_phase": "answer_user",
                "known_facts": {"pool_checked": True},
                "missing_facts": [],
                "blockers": [],
            },
            "objective_plan": [
                {
                    "step_id": "search_pool",
                    "title": "查询当前局池",
                    "status": "done",
                    "tool": "search_current_games",
                    "depends_on": [],
                    "decision_rule": "工具查询完成。",
                }
            ],
            "plan_revision_reason": "收到只读工具结果后生成固定模拟回复。",
            "reply_to_user": reply,
            "tool_calls": [],
            "needs_human": False,
            "stop_reason": {
                "can_stop": True,
                "why": "工具结果已经返回并生成回复。",
                "pending_work": [],
                "depends_on_tool_results": True,
            },
            "badcase": None,
        }


def required_llm_mode(environ: dict[str, str] | None = None) -> str:
    """Make model cost selection explicit; there is intentionally no default."""

    environment = os.environ if environ is None else environ
    mode = str(environment.get("SIM_LLM_MODE") or "").strip().lower()
    if mode not in {"mock", "real"}:
        raise RuntimeError("SIM_LLM_MODE is required and must be exactly 'mock' or 'real'.")
    return mode


def build_runtime(store: SQLiteAgentStore, mode: str, *, seed: int = 42) -> tuple[AgentRuntime, Any]:
    """Build a mock runtime unless the caller explicitly selected real mode.

    Real mode also exercises customer-visible copywriting and privacy review;
    mock mode keeps those model tasks disabled because its small seeded client
    implements only the main Agent contract.
    """

    if mode not in {"mock", "real"}:
        raise RuntimeError("mode must be exactly 'mock' or 'real'")
    if mode == "mock":
        llm_client: Any = StaticAgentLLMClient(seed=seed)
    else:
        load_dotenv_defaults(ROOT / ".env")
        llm_client = OpenAICompatibleAgentClient.from_env()
        if llm_client is None:
            raise RuntimeError(
                "SIM_LLM_MODE=real requires MAHJONG_LLM_API_KEY and MAHJONG_LLM_MODEL."
            )
    runtime = AgentRuntime(
        llm_client=llm_client,
        store=store,
        tool_gateway=ToolGateway(store),
        trace_recorder=InMemoryTraceRecorder(),
        customer_visible_text_generation_enabled=mode == "real",
        customer_visible_text_generation_client=llm_client if mode == "real" else None,
        reply_self_review_enabled=mode == "real",
        reply_self_review_client=llm_client if mode == "real" else None,
        context_summary_manager=None,
    )
    return runtime, llm_client


class SimulationHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], runtime: AgentRuntime) -> None:
        super().__init__(address, SimulationRequestHandler)
        self.runtime = runtime


class SimulationRequestHandler(BaseHTTPRequestHandler):
    """Isolated HTTP ingress; the pressure client never calls Runtime directly."""

    server: SimulationHTTPServer

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if self.path != "/api/message":
            self._json({"error": "not_found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            required = ("conversation_id", "sender_id", "sender_name", "message_id", "text")
            missing = [key for key in required if not str(payload.get(key) or "").strip()]
            if missing:
                self._json({"error": "missing_required_fields", "missing_fields": missing}, status=400)
                return
            message = UserMessage(
                conversation_id=str(payload["conversation_id"]),
                sender_id=str(payload["sender_id"]),
                sender_name=str(payload["sender_name"]),
                message_id=str(payload["message_id"]),
                text=str(payload["text"]),
                metadata=dict(payload.get("metadata") or {}),
            )
            result = self.server.runtime.handle_user_message(
                message,
                trace_id=str(payload.get("trace_id") or "") or None,
            )
            self._json(result.to_dict())
        except json.JSONDecodeError:
            self._json({"error": "invalid_json"}, status=400)
        except Exception as exc:  # Expose backend errors to the simulator report.
            self._json({"error": type(exc).__name__, "detail": str(exc)}, status=500)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        del format, args

    def _json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@contextmanager
def running_http_backend(runtime: AgentRuntime) -> Iterator[str]:
    server = SimulationHTTPServer(("127.0.0.1", 0), runtime)
    thread = threading.Thread(target=server.serve_forever, name="simulation-http", daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


class SimulationAdapter:
    """Send WeChat-shaped HTTP payloads and fan replies into virtual inboxes."""

    def __init__(
        self,
        *,
        base_url: str,
        users: list[VirtualUser],
        request_timeout_seconds: float = 30.0,
        state_backend: SimulationStateBackend | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout_seconds = max(0.1, float(request_timeout_seconds))
        self._user_ids = [user.customer_id for user in users]
        self._user_ids_by_name: dict[str, list[str]] = {}
        for user in users:
            self._user_ids_by_name.setdefault(user.display_name, []).append(user.customer_id)
        self.state_backend = state_backend or InMemorySimulationStateBackend(self._user_ids)

    def inbox_for(self, customer_id: str) -> list[InboxMessage]:
        return self.state_backend.inbox_for(customer_id)

    def inbox_sizes(self) -> dict[str, int]:
        return self.state_backend.inbox_sizes()

    def next_speaker_only(
        self,
        conversation_id: str,
        thread_id: str | None = None,
    ) -> str | None:
        """Return the user holding the next turn in one logical topic."""

        gate = self.state_backend.reply_gate(conversation_id, thread_id)
        return gate.expected_user_id if gate is not None else None

    def expired_speaker_locks(
        self,
        timeout_seconds: float,
        *,
        now: float | None = None,
    ) -> list[ReplyGate]:
        """Snapshot topic gates old enough for the timeout safety valve."""

        return self.state_backend.expired_reply_gates(timeout_seconds, now=now)

    def seconds_until_lock_timeout(
        self,
        timeout_seconds: float,
        *,
        now: float | None = None,
    ) -> float | None:
        return self.state_backend.seconds_until_reply_gate_timeout(
            timeout_seconds,
            now=now,
        )

    def release_speaker_lock(
        self,
        conversation_id: str,
        thread_id: str | None = None,
        *,
        expected_user_id: str | None = None,
    ) -> bool:
        """Release one topic gate, optionally checking its current owner."""

        return self.state_backend.release_reply_gate(
            conversation_id,
            thread_id,
            expected_user_id=expected_user_id,
        )

    def send(self, action: SimulationAction, *, deadline: float) -> RequestOutcome:
        payload = action.to_wechat_payload()
        self.state_backend.publish_event(
            "user_message",
            {
                "conversation_id": action.conversation_id,
                "thread_id": action.thread_id,
                "source_message_id": action.message_id,
                "sender_id": action.sender_id,
                "sender_name": action.sender_name,
                "channel": action.channel,
                "text": action.text,
                "dialog_phase": action.dialog_phase,
            },
        )
        request = urllib.request.Request(
            f"{self.base_url}/api/message",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        started = time.perf_counter()
        sent_at = time.monotonic()
        timeout = min(self.request_timeout_seconds, max(0.1, deadline - time.monotonic()))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                parsed = json.loads(body) if body else {}
                result = RequestOutcome(
                    action=action,
                    sent=True,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    status_code=int(response.status),
                    response=parsed if isinstance(parsed, dict) else {"raw": parsed},
                    sent_at=sent_at,
                )
                result.inbox_deliveries = self._deliver_reply(result)
                return result
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body) if body else {}
            except json.JSONDecodeError:
                parsed = {"raw": body}
            return RequestOutcome(
                action=action,
                sent=True,
                latency_ms=(time.perf_counter() - started) * 1000,
                status_code=int(exc.code),
                response=parsed if isinstance(parsed, dict) else {"raw": parsed},
                error=f"HTTP {exc.code}",
                sent_at=sent_at,
            )
        except Exception as exc:
            return RequestOutcome(
                action=action,
                sent=True,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
                sent_at=sent_at,
            )

    def _deliver_reply(self, outcome: RequestOutcome) -> int:
        reply = str(outcome.response.get("final_reply") or "")
        trace_id = str(outcome.response.get("trace_id") or "")
        outcome.next_speaker_only = self._update_next_speaker(
            outcome.action.conversation_id,
            outcome.action.thread_id,
            reply,
            source_message_id=outcome.action.message_id,
        )
        recipients = self._user_ids if outcome.action.channel == "group" else [outcome.action.sender_id]
        received_at = time.time()
        self.state_backend.append_inboxes(
            [
                InboxMessage(
                    recipient_id=recipient_id,
                    sender="mahjong_agent",
                    text=reply,
                    trace_id=trace_id,
                    source_message_id=outcome.action.message_id,
                    channel=outcome.action.channel,
                    received_at=received_at,
                    conversation_id=outcome.action.conversation_id,
                    thread_id=outcome.action.thread_id,
                )
                for recipient_id in recipients
            ]
        )
        self.state_backend.publish_event(
            "agent_message",
            {
                "conversation_id": outcome.action.conversation_id,
                "thread_id": outcome.action.thread_id,
                "source_message_id": outcome.action.message_id,
                "trace_id": trace_id,
                "channel": outcome.action.channel,
                "text": reply,
                "recipient_ids": recipients,
                "next_speaker_only": outcome.next_speaker_only,
            },
        )
        return len(recipients)

    def _update_next_speaker(
        self,
        conversation_id: str,
        thread_id: str,
        reply: str,
        *,
        source_message_id: str,
    ) -> str | None:
        """Translate an Agent ``@nickname`` into a topic-scoped reply gate."""

        mentioned_user_id: str | None = None
        for nickname in MENTION_PATTERN.findall(reply):
            matches = self._user_ids_by_name.get(nickname, [])
            if len(matches) == 1:
                mentioned_user_id = matches[0]
                break

        if mentioned_user_id is None:
            # A broadcast releases only this topic. Other group topics retain
            # their own gates and can continue independently.
            self.state_backend.release_reply_gate(conversation_id, thread_id)
            return None
        self.state_backend.set_reply_gate(
            ReplyGate(
                conversation_id=conversation_id,
                thread_id=thread_id,
                expected_user_id=mentioned_user_id,
                source_message_id=source_message_id,
                acquired_at=time.time(),
            )
        )
        return mentioned_user_id
