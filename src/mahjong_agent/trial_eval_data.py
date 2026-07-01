from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BOSS_REPLY_FEW_SHOTS: list[dict[str, Any]] = [
    {
        "name": "明确组局，信息基本够",
        "source": "真实聊天脱敏改写：客户给出财敲、人数、档位、烟况和开局时间。",
        "customer_message": "可以帮忙摇下人吗，272财敲0.5，无烟，7点开4小时",
        "parsed": "财敲，0.5，二缺二，19:00，4小时，无烟",
        "reply_text": "可以，我先按财敲0.5、7点开、二缺二、无烟帮你问人。有合适的我先给你确认。",
    },
    {
        "name": "组一桌但人数未知",
        "source": "老板反馈：组一桌不等于三缺一，人数要确认。",
        "customer_message": "下午两点 0.5 无烟杭麻，帮我组一桌",
        "parsed": "时间、玩法、档位、烟况明确，但当前人数未知",
        "conditions": "仅适用时间未过期；时间已过先确认，不能说“先帮你看”。",
        "reply_text": "可以，我先帮你看。你一个人吗？",
    },
    {
        "name": "缺关键字段，先追问",
        "source": "真实聊天脱敏改写：群里有人只表达川麻意向。",
        "customer_message": "川麻132 晚上有人吗",
        "parsed": "川麻，1-32，晚上，人数和烟况不明确",
        "reply_text": "可以，我帮你看看。你大概几点能到、现在几个人、有烟无烟有要求吗？",
    },
    {
        "name": "客户可接受备选玩法",
        "source": "真实聊天脱敏改写：客户先要川麻，后续补充杭麻也可以。",
        "customer_message": "川麻没人的话，杭麻0.5无烟的也行",
        "parsed": "优先川麻，备选杭麻0.5无烟",
        "reply_text": "收到，我先按川麻帮你看；如果川麻不好凑，我再帮你看杭麻0.5无烟的局。",
    },
    {
        "name": "已有相近局，先协商时间",
        "source": "真实聊天脱敏改写：客户能接受相近开局时间，需要跟另一桌确认。",
        "customer_message": "七点半0.5帮我问下吧",
        "parsed": "0.5，19:30，人数未完全明确",
        "reply_text": "好，我先帮你问下7点半左右0.5的局。如果那边时间能对上，我再跟你确认。",
    },
    {
        "name": "群内弱意图，问清信息",
        "source": "真实聊天脱敏改写：群里有人问现在有没有三缺一。",
        "customer_message": "有没有三缺一的局啊现在",
        "parsed": "想找局，时间为现在，玩法、档位、烟况不明确",
        "reply_text": "我帮你看下。你想打杭麻还是川麻，0.5还是1，有烟无烟有要求吗？",
    },
]


_JSONL_LOCK = threading.Lock()


@dataclass(frozen=True)
class TrialEvalDataPaths:
    golden: Path
    boss_trial_golden: Path
    badcase: Path
    few_shot: Path
    skills: Path

    def as_strings(self) -> dict[str, str]:
        return {
            "golden": str(self.golden),
            "boss_trial_golden": str(self.boss_trial_golden),
            "badcase": str(self.badcase),
            "few_shot": str(self.few_shot),
            "skills": str(self.skills),
        }


class TrialEvalDataStore:
    def __init__(self, paths: TrialEvalDataPaths) -> None:
        self.paths = paths

    def overview(self) -> dict[str, Any]:
        return {
            "paths": self.paths.as_strings(),
            "counts": {
                "golden": count_jsonl_records(self.paths.golden),
                "boss_trial_golden": count_jsonl_records(self.paths.boss_trial_golden),
                "badcase": count_jsonl_records(self.paths.badcase),
                "few_shot": count_jsonl_records(self.paths.few_shot),
                "skills": count_jsonl_records(self.paths.skills),
            },
            "recent": {
                "golden": read_jsonl_records(self.paths.golden, limit=3),
                "boss_trial_golden": read_jsonl_records(self.paths.boss_trial_golden, limit=3),
                "badcase": read_jsonl_records(self.paths.badcase, limit=3),
                "few_shot": read_jsonl_records(self.paths.few_shot, limit=3),
                "skills": read_jsonl_records(self.paths.skills, limit=3),
            },
            "runner": "PYTHONPATH=src python scripts/run_scenario_eval.py",
        }

    def path_for_case_type(self, case_type: str) -> Path:
        if case_type == "badcase":
            return self.paths.badcase
        if case_type == "golden":
            return self.paths.golden
        if case_type == "few_shot":
            return self.paths.few_shot
        raise ValueError("case_type 必须是 badcase、golden 或 few_shot")

    def append_case(self, case_type: str, record: dict[str, Any]) -> Path:
        path = self.path_for_case_type(case_type)
        append_jsonl_record(path, record)
        return path

    def few_shot_examples(self) -> list[dict[str, Any]]:
        dynamic: list[dict[str, Any]] = []
        for record in read_jsonl_records(self.paths.few_shot, limit=20):
            if record.get("kind") != "few_shot":
                continue
            customer_message = str(record.get("customer_message") or "").strip()
            reply_text = str(record.get("reply_text") or "").strip()
            if not customer_message or not reply_text:
                continue
            item = {
                "name": str(record.get("name") or "老板认可话术"),
                "source": f"试用台采集：{record.get('id')}",
                "customer_message": customer_message,
                "parsed": str(record.get("parsed") or ""),
                "reply_text": reply_text,
            }
            conditions = str(record.get("conditions") or "").strip()
            if conditions:
                item["conditions"] = conditions
            dynamic.append(item)
        return [*BOSS_REPLY_FEW_SHOTS, *dynamic[-8:]]


def read_jsonl_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    if limit is None:
        return records
    return records[-limit:]


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _JSONL_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def count_jsonl_records(path: Path) -> int:
    return len(read_jsonl_records(path))


def eval_case_tags(payload: dict[str, Any], analysis: dict[str, Any], case_type: str) -> list[str]:
    raw_tags = payload.get("tags") or []
    if isinstance(raw_tags, str):
        raw_tags = re.split(r"[,，、\s]+", raw_tags)
    tags = [str(item).strip() for item in raw_tags if str(item).strip()]
    tags.append(case_type)
    decision = analysis.get("decision") if isinstance(analysis.get("decision"), dict) else {}
    action = decision.get("action")
    if action:
        tags.append(str(action))
    if analysis.get("used_short_memory"):
        tags.append("short_memory")
    parsed = analysis.get("parsed") if isinstance(analysis.get("parsed"), dict) else {}
    game_label = parsed.get("game_label")
    if game_label:
        tags.append(str(game_label))
    return list(dict.fromkeys(tags))


def few_shot_parsed_text(parsed: dict[str, Any], analysis: dict[str, Any]) -> str:
    fields = [
        parsed.get("game_label"),
        parsed.get("level"),
        parsed.get("start_time"),
    ]
    if parsed.get("missing_count") is not None:
        fields.append(f"缺{parsed.get('missing_count')}")
    rules = parsed.get("rules") if isinstance(parsed.get("rules"), list) else []
    fields.extend(rules[:3])
    missing = analysis.get("missing_fields") or []
    if missing:
        fields.append("待确认：" + "、".join(str(item) for item in missing))
    return "，".join(str(item) for item in fields if item) or "系统未解析出完整组局条件"
