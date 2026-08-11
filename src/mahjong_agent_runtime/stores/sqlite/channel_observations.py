"""实现 SQLite 存储层的通道observations数据访问与事务操作。

"""

from __future__ import annotations

from typing import Any

from .serialization import _dumps, _loads, _now_iso


class SQLiteChannelObservationsStoreMixin:
    """持久化原始渠道证据及后续语义路由决策。

    观察归档有意与会话轮次分离：仅观察的群聊流量必须保持可用以供分析，而不被当作 Agent 记忆，也不改变任何业务聚合。
    """

    __slots__ = ()

    def upsert_channel_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """新增或更新通道observation，使重复写入收敛到同一条记录。"""
        record = _normalize_observation(observation)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runtime_channel_observations(
                    channel, source_message_id, trace_id, conversation_id,
                    room_id, room_topic, sender_id, sender_name, message_text,
                    message_type, is_room, self_message, route_status,
                    route_mode, route_reason, semantic_action, semantic_category,
                    semantic_confidence, business_message_detected, payload,
                    received_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(channel, source_message_id) DO UPDATE SET
                    trace_id=excluded.trace_id,
                    conversation_id=excluded.conversation_id,
                    room_id=excluded.room_id,
                    room_topic=excluded.room_topic,
                    sender_id=excluded.sender_id,
                    sender_name=excluded.sender_name,
                    message_text=excluded.message_text,
                    message_type=excluded.message_type,
                    is_room=excluded.is_room,
                    self_message=excluded.self_message,
                    route_status=excluded.route_status,
                    route_mode=excluded.route_mode,
                    route_reason=excluded.route_reason,
                    semantic_action=excluded.semantic_action,
                    semantic_category=excluded.semantic_category,
                    semantic_confidence=excluded.semantic_confidence,
                    business_message_detected=excluded.business_message_detected,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    record["channel"],
                    record["source_message_id"],
                    record["trace_id"],
                    record["conversation_id"],
                    record["room_id"],
                    record["room_topic"],
                    record["sender_id"],
                    record["sender_name"],
                    record["text"],
                    record["message_type"],
                    int(record["is_room"]),
                    int(record["self_message"]),
                    record["route_status"],
                    record["route_mode"],
                    record["route_reason"],
                    record["semantic_action"],
                    record["semantic_category"],
                    record["semantic_confidence"],
                    int(record["business_message_detected"]),
                    _dumps(record["payload"]),
                    record["received_at"],
                    record["updated_at"],
                ),
            )
        return record

    def list_channel_observations(
        self,
        *,
        channel: str | None = None,
        room_id: str | None = None,
        room_topic_keyword: str | None = None,
        semantic_action: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """列出通道observations；结果保持稳定的领域顺序。"""
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in (
            ("channel", channel),
            ("room_id", room_id),
            ("semantic_action", semantic_action),
        ):
            normalized = str(value or "").strip()
            if normalized:
                clauses.append(f"{column} = ?")
                parameters.append(normalized)
        topic_keyword = str(room_topic_keyword or "").strip()
        if topic_keyword:
            clauses.append("room_topic LIKE ?")
            parameters.append(f"%{topic_keyword}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(max(1, min(int(limit), 1_000)))
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT *
                FROM runtime_channel_observations
                {where}
                ORDER BY received_at DESC, rowid DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [_observation_from_row(row) for row in rows]


def _normalize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    channel = str(observation.get("channel") or "").strip()
    source_message_id = str(observation.get("source_message_id") or "").strip()
    if not channel:
        raise ValueError("channel observation requires channel")
    if not source_message_id:
        raise ValueError("channel observation requires source_message_id")
    received_at = str(observation.get("received_at") or _now_iso())
    try:
        confidence = float(observation.get("semantic_confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    payload = observation.get("payload")
    return {
        "channel": channel,
        "source_message_id": source_message_id,
        "trace_id": str(observation.get("trace_id") or ""),
        "conversation_id": str(observation.get("conversation_id") or ""),
        "room_id": str(observation.get("room_id") or ""),
        "room_topic": str(observation.get("room_topic") or ""),
        "sender_id": str(observation.get("sender_id") or ""),
        "sender_name": str(observation.get("sender_name") or ""),
        "text": str(observation.get("text") or ""),
        "message_type": str(observation.get("message_type") or ""),
        "is_room": bool(observation.get("is_room")),
        "self_message": bool(observation.get("self_message")),
        "route_status": str(observation.get("route_status") or "received"),
        "route_mode": str(observation.get("route_mode") or "pending"),
        "route_reason": str(observation.get("route_reason") or ""),
        "semantic_action": str(observation.get("semantic_action") or ""),
        "semantic_category": str(observation.get("semantic_category") or ""),
        "semantic_confidence": max(0.0, min(confidence, 1.0)),
        "business_message_detected": bool(observation.get("business_message_detected")),
        "payload": dict(payload) if isinstance(payload, dict) else {},
        "received_at": received_at,
        "updated_at": _now_iso(),
    }


def _observation_from_row(row: Any) -> dict[str, Any]:
    return {
        "channel": str(row["channel"]),
        "source_message_id": str(row["source_message_id"]),
        "trace_id": str(row["trace_id"]),
        "conversation_id": str(row["conversation_id"]),
        "room_id": str(row["room_id"]),
        "room_topic": str(row["room_topic"]),
        "sender_id": str(row["sender_id"]),
        "sender_name": str(row["sender_name"]),
        "text": str(row["message_text"]),
        "message_type": str(row["message_type"]),
        "is_room": bool(row["is_room"]),
        "self_message": bool(row["self_message"]),
        "route_status": str(row["route_status"]),
        "route_mode": str(row["route_mode"]),
        "route_reason": str(row["route_reason"]),
        "semantic_action": str(row["semantic_action"]),
        "semantic_category": str(row["semantic_category"]),
        "semantic_confidence": float(row["semantic_confidence"]),
        "business_message_detected": bool(row["business_message_detected"]),
        "payload": _loads(str(row["payload"])),
        "received_at": str(row["received_at"]),
        "updated_at": str(row["updated_at"]),
    }


__all__ = ["SQLiteChannelObservationsStoreMixin"]
