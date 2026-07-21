"""Short-lived state backends for the synthetic chat simulator.

SQLite remains the source of truth for games and customers.  This module only
stores disposable simulation coordination data: delivered inbox messages,
fine-grained reply gates, and an auditable event stream.  Keeping the contract
small lets deterministic tests use memory while local multi-worker runs use
Redis without changing the simulator's business behavior.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Protocol


@dataclass(slots=True, frozen=True)
class InboxMessage:
    recipient_id: str
    sender: str
    text: str
    trace_id: str
    source_message_id: str
    channel: str
    received_at: float
    conversation_id: str = ""
    thread_id: str = ""


@dataclass(slots=True, frozen=True)
class ReplyGate:
    """Only ``expected_user_id`` may answer one Agent reply in one topic."""

    conversation_id: str
    thread_id: str
    expected_user_id: str
    source_message_id: str
    acquired_at: float

    @property
    def scope(self) -> str:
        return coordination_scope(self.conversation_id, self.thread_id)


def coordination_scope(conversation_id: str, thread_id: str | None) -> str:
    """Build the narrowest stable scheduler scope available for one message."""

    topic = str(thread_id or "").strip() or "legacy"
    return f"{conversation_id}::{topic}"


class SimulationStateBackend(Protocol):
    """Storage contract used by the adapter; implementations are disposable."""

    backend_name: str
    namespace: str

    def append_inbox(self, message: InboxMessage) -> None: ...

    def append_inboxes(self, messages: Iterable[InboxMessage]) -> None: ...

    def inbox_for(self, customer_id: str) -> list[InboxMessage]: ...

    def inbox_sizes(self) -> dict[str, int]: ...

    def set_reply_gate(self, gate: ReplyGate) -> None: ...

    def reply_gate(self, conversation_id: str, thread_id: str | None) -> ReplyGate | None: ...

    def expired_reply_gates(
        self,
        timeout_seconds: float,
        *,
        now: float | None = None,
    ) -> list[ReplyGate]: ...

    def seconds_until_reply_gate_timeout(
        self,
        timeout_seconds: float,
        *,
        now: float | None = None,
    ) -> float | None: ...

    def release_reply_gate(
        self,
        conversation_id: str,
        thread_id: str | None,
        *,
        expected_user_id: str | None = None,
    ) -> bool: ...

    def publish_event(self, event_type: str, payload: dict[str, Any]) -> str: ...

    def recent_events(self, limit: int = 100) -> list[dict[str, Any]]: ...


class InMemorySimulationStateBackend:
    """Thread-safe deterministic backend for unit tests and small local runs."""

    backend_name = "memory"

    def __init__(self, user_ids: Iterable[str], *, namespace: str = "memory") -> None:
        self.namespace = namespace
        self._user_ids = tuple(dict.fromkeys(str(item) for item in user_ids))
        self._inboxes: dict[str, list[InboxMessage]] = {
            customer_id: [] for customer_id in self._user_ids
        }
        self._reply_gates: dict[str, ReplyGate] = {}
        self._events: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def append_inbox(self, message: InboxMessage) -> None:
        self.append_inboxes([message])

    def append_inboxes(self, messages: Iterable[InboxMessage]) -> None:
        with self._lock:
            for message in messages:
                self._inboxes.setdefault(message.recipient_id, []).append(message)

    def inbox_for(self, customer_id: str) -> list[InboxMessage]:
        with self._lock:
            return list(self._inboxes.get(customer_id, []))

    def inbox_sizes(self) -> dict[str, int]:
        with self._lock:
            return {
                customer_id: len(self._inboxes.get(customer_id, []))
                for customer_id in self._user_ids
            }

    def set_reply_gate(self, gate: ReplyGate) -> None:
        with self._lock:
            self._reply_gates[gate.scope] = gate

    def reply_gate(self, conversation_id: str, thread_id: str | None) -> ReplyGate | None:
        with self._lock:
            return self._reply_gates.get(coordination_scope(conversation_id, thread_id))

    def expired_reply_gates(
        self,
        timeout_seconds: float,
        *,
        now: float | None = None,
    ) -> list[ReplyGate]:
        current = time.time() if now is None else float(now)
        with self._lock:
            return [
                gate
                for gate in self._reply_gates.values()
                if current - gate.acquired_at >= timeout_seconds
            ]

    def seconds_until_reply_gate_timeout(
        self,
        timeout_seconds: float,
        *,
        now: float | None = None,
    ) -> float | None:
        current = time.time() if now is None else float(now)
        with self._lock:
            if not self._reply_gates:
                return None
            return max(
                0.0,
                min(
                    timeout_seconds - (current - gate.acquired_at)
                    for gate in self._reply_gates.values()
                ),
            )

    def release_reply_gate(
        self,
        conversation_id: str,
        thread_id: str | None,
        *,
        expected_user_id: str | None = None,
    ) -> bool:
        scope = coordination_scope(conversation_id, thread_id)
        with self._lock:
            gate = self._reply_gates.get(scope)
            if gate is None or (
                expected_user_id is not None and gate.expected_user_id != expected_user_id
            ):
                return False
            self._reply_gates.pop(scope, None)
            return True

    def publish_event(self, event_type: str, payload: dict[str, Any]) -> str:
        event_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        with self._lock:
            self._events.append(
                {
                    "event_id": event_id,
                    "event_type": event_type,
                    "recorded_at": time.time(),
                    **payload,
                }
            )
        return event_id

    def recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        resolved_limit = max(0, int(limit))
        if resolved_limit == 0:
            return []
        with self._lock:
            return list(self._events[-resolved_limit:])


class RedisSimulationStateBackend:
    """Redis-backed simulation state using List, Hash, ZSet, and Stream.

    One namespace belongs to one simulation run.  Reply gates are indexed by
    ``conversation_id + thread_id`` rather than by the whole group, so an @
    question in one topic cannot stall unrelated topics in the same room.
    """

    backend_name = "redis"
    _RELEASE_SCRIPT = """
local raw = redis.call('HGET', KEYS[1], ARGV[1])
if not raw then return 0 end
if ARGV[2] ~= '' then
  local gate = cjson.decode(raw)
  if gate['expected_user_id'] ~= ARGV[2] then return 0 end
end
redis.call('HDEL', KEYS[1], ARGV[1])
redis.call('ZREM', KEYS[2], ARGV[1])
return 1
"""

    def __init__(
        self,
        redis_url: str,
        user_ids: Iterable[str],
        *,
        namespace: str,
        client: Any | None = None,
        event_max_length: int = 10_000,
        retention_seconds: int = 86_400,
    ) -> None:
        if not namespace.strip():
            raise ValueError("Redis simulation namespace must not be empty")
        if client is None:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "Redis simulation state requires `pip install -e '.[distributed]'`"
                ) from exc
            client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.client = client
        self.client.ping()
        self.namespace = namespace.strip()
        self.event_max_length = max(100, int(event_max_length))
        self.retention_seconds = max(60, int(retention_seconds))
        self._user_ids = tuple(dict.fromkeys(str(item) for item in user_ids))
        self._prefix = f"mahjong-agent:simulation:{self.namespace}"
        self._gates_key = f"{self._prefix}:reply-gates"
        self._gate_index_key = f"{self._prefix}:reply-gate-acquired-at"
        self._events_key = f"{self._prefix}:events"

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _inbox_key(self, customer_id: str) -> str:
        return f"{self._prefix}:inbox:{self._digest(customer_id)}"

    def append_inbox(self, message: InboxMessage) -> None:
        self.append_inboxes([message])

    def append_inboxes(self, messages: Iterable[InboxMessage]) -> None:
        values = list(messages)
        if not values:
            return
        pipeline = self.client.pipeline(transaction=False)
        touched_keys: set[str] = set()
        for message in values:
            key = self._inbox_key(message.recipient_id)
            pipeline.rpush(
                key,
                json.dumps(asdict(message), ensure_ascii=False, sort_keys=True),
            )
            touched_keys.add(key)
        for key in touched_keys:
            pipeline.expire(key, self.retention_seconds)
        pipeline.execute()

    def inbox_for(self, customer_id: str) -> list[InboxMessage]:
        values = self.client.lrange(self._inbox_key(customer_id), 0, -1)
        return [InboxMessage(**json.loads(value)) for value in values]

    def inbox_sizes(self) -> dict[str, int]:
        pipeline = self.client.pipeline(transaction=False)
        for customer_id in self._user_ids:
            pipeline.llen(self._inbox_key(customer_id))
        return dict(zip(self._user_ids, (int(value) for value in pipeline.execute())))

    def set_reply_gate(self, gate: ReplyGate) -> None:
        payload = json.dumps(asdict(gate), ensure_ascii=False, sort_keys=True)
        pipeline = self.client.pipeline(transaction=True)
        pipeline.hset(self._gates_key, gate.scope, payload)
        pipeline.zadd(self._gate_index_key, {gate.scope: gate.acquired_at})
        pipeline.expire(self._gates_key, self.retention_seconds)
        pipeline.expire(self._gate_index_key, self.retention_seconds)
        pipeline.execute()

    def reply_gate(self, conversation_id: str, thread_id: str | None) -> ReplyGate | None:
        raw = self.client.hget(
            self._gates_key,
            coordination_scope(conversation_id, thread_id),
        )
        return ReplyGate(**json.loads(raw)) if raw else None

    def expired_reply_gates(
        self,
        timeout_seconds: float,
        *,
        now: float | None = None,
    ) -> list[ReplyGate]:
        current = time.time() if now is None else float(now)
        scopes = self.client.zrangebyscore(
            self._gate_index_key,
            "-inf",
            current - timeout_seconds,
        )
        if not scopes:
            return []
        values = self.client.hmget(self._gates_key, scopes)
        return [ReplyGate(**json.loads(raw)) for raw in values if raw]

    def seconds_until_reply_gate_timeout(
        self,
        timeout_seconds: float,
        *,
        now: float | None = None,
    ) -> float | None:
        oldest = self.client.zrange(self._gate_index_key, 0, 0, withscores=True)
        if not oldest:
            return None
        current = time.time() if now is None else float(now)
        return max(0.0, timeout_seconds - (current - float(oldest[0][1])))

    def release_reply_gate(
        self,
        conversation_id: str,
        thread_id: str | None,
        *,
        expected_user_id: str | None = None,
    ) -> bool:
        scope = coordination_scope(conversation_id, thread_id)
        released = self.client.eval(
            self._RELEASE_SCRIPT,
            2,
            self._gates_key,
            self._gate_index_key,
            scope,
            expected_user_id or "",
        )
        return bool(released)

    def publish_event(self, event_type: str, payload: dict[str, Any]) -> str:
        fields = {
            "event_type": event_type,
            "recorded_at": str(time.time()),
            "payload": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        }
        pipeline = self.client.pipeline(transaction=False)
        pipeline.xadd(
            self._events_key,
            fields,
            maxlen=self.event_max_length,
            approximate=True,
        )
        pipeline.expire(self._events_key, self.retention_seconds)
        event_id, _ = pipeline.execute()
        return str(event_id)

    def recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        resolved_limit = max(0, int(limit))
        if resolved_limit == 0:
            return []
        entries = self.client.xrevrange(self._events_key, count=resolved_limit)
        result: list[dict[str, Any]] = []
        for event_id, fields in reversed(entries):
            payload = json.loads(fields.get("payload") or "{}")
            result.append(
                {
                    "event_id": str(event_id),
                    "event_type": str(fields.get("event_type") or ""),
                    "recorded_at": float(fields.get("recorded_at") or 0.0),
                    **(payload if isinstance(payload, dict) else {"payload": payload}),
                }
            )
        return result


def build_simulation_state_backend(
    backend_name: str,
    *,
    user_ids: Iterable[str],
    redis_url: str = "redis://127.0.0.1:6379/0",
    namespace: str | None = None,
) -> SimulationStateBackend:
    """Create the explicitly selected backend; never silently downgrade Redis."""

    selected = str(backend_name or "memory").strip().lower()
    resolved_namespace = namespace or f"run-{uuid.uuid4().hex}"
    if selected == "memory":
        return InMemorySimulationStateBackend(user_ids, namespace=resolved_namespace)
    if selected == "redis":
        return RedisSimulationStateBackend(
            redis_url,
            user_ids,
            namespace=resolved_namespace,
        )
    raise ValueError("simulation state backend must be 'memory' or 'redis'")
