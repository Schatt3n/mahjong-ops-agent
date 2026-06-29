from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_SKILL_LIBRARY_PATH = Path(__file__).resolve().parents[2] / "skills" / "mahjong_operations_skills.jsonl"


def load_skill_records(path: Path | None = None) -> list[dict[str, Any]]:
    skill_path = path or DEFAULT_SKILL_LIBRARY_PATH
    if not skill_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in skill_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("kind") == "operation_skill":
            records.append(value)
    return records


def select_relevant_skills(
    *,
    stage: str,
    text: str = "",
    records: list[dict[str, Any]] | None = None,
    path: Path | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    all_records = records if records is not None else load_skill_records(path)
    matched: list[tuple[int, int, dict[str, Any]]] = []
    fallback: list[tuple[int, int, dict[str, Any]]] = []
    normalized_text = text.lower()
    for index, record in enumerate(all_records):
        stages = [str(item) for item in record.get("stages") or []]
        if stage not in stages and "all" not in stages:
            continue
        score = 10
        trigger_hits = 0
        for trigger in record.get("triggers") or []:
            trigger_text = str(trigger).strip().lower()
            if not trigger_text:
                continue
            if trigger_text in normalized_text or re.search(re.escape(trigger_text), normalized_text):
                trigger_hits += 1
        score += trigger_hits * 5
        priority = int(record.get("priority") or 50)
        item = (score + max(0, 100 - priority), -index, record)
        if trigger_hits > 0:
            matched.append(item)
        else:
            fallback.append(item)
    scored = matched or fallback
    scored.sort(reverse=True)
    return [_compact_skill(record) for _, _, record in scored[: max(0, limit)]]


def _compact_skill(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "name": record.get("name"),
        "version": record.get("version", "v1"),
        "instructions": list(record.get("instructions") or [])[:4],
        "risk_controls": list(record.get("risk_controls") or [])[:3],
    }
