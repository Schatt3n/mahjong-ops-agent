from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .models import DEFAULT_TZ_V2, new_id


class EvalRecorderV2(Protocol):
    def record_badcase(
        self,
        payload: dict[str, Any],
        *,
        trace_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class InMemoryEvalRecorderV2:
    records: list[dict[str, Any]] = field(default_factory=list)

    def record_badcase(
        self,
        payload: dict[str, Any],
        *,
        trace_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        record = badcase_record(payload, trace_id=trace_id, conversation_id=conversation_id)
        self.records.append(record)
        return record


@dataclass(slots=True)
class JsonlEvalRecorderV2:
    path: Path | str

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_badcase(
        self,
        payload: dict[str, Any],
        *,
        trace_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        record = badcase_record(payload, trace_id=trace_id, conversation_id=conversation_id)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record


def badcase_record(payload: dict[str, Any], *, trace_id: str, conversation_id: str) -> dict[str, Any]:
    now = datetime.now(DEFAULT_TZ_V2)
    return {
        "schema_version": "agent_runtime_v2.badcase.v1",
        "badcase_id": str(payload.get("badcase_id") or new_id("badcasev2")),
        "trace_id": trace_id,
        "conversation_id": conversation_id,
        "recorded_at": now.isoformat(),
        "reason": str(payload.get("reason") or ""),
        "input": _jsonable(payload.get("input") or {}),
        "actual": _jsonable(payload.get("actual") or {}),
        "expected": _jsonable(payload.get("expected") or {}),
        "tags": [str(item) for item in payload.get("tags") or []],
        "source": str(payload.get("source") or "llm"),
        "metadata": _jsonable(payload.get("metadata") or {}),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    return str(value)
