#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


QUOTE_PATH_TOKENS = ("quote", "quoted", "refer", "reference")
WECHAT_DISPLAY_QUOTE_PATTERN = re.compile(
    r"^\s*[「『](?P<quoted>.+?)[」』]\s*\n(?P<separator>(?:[-—–_]\s*){3,})\n(?P<reply>.+?)\s*$",
    re.DOTALL,
)
DEFAULT_INPUT = Path("logs/wechaty_weixin_raw.jsonl")
DEFAULT_OUTPUT = Path("runtime_data/wechaty_quote_payload_analysis.json")


def iter_jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                yield {
                    "_line_number": line_number,
                    "_decode_error": str(exc),
                    "_raw_preview": text[:240],
                }
                continue
            if isinstance(value, dict):
                value["_line_number"] = line_number
                yield value
            else:
                yield {
                    "_line_number": line_number,
                    "_decode_error": "json line is not an object",
                    "_raw_preview": repr(value)[:240],
                }


def value_preview(value: Any, *, max_chars: int = 240) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def looks_like_wechat_refermsg(value: Any) -> bool:
    return isinstance(value, str) and "<refermsg" in value.lower()


def looks_like_wechat_display_quote(value: Any) -> bool:
    return isinstance(value, str) and bool(WECHAT_DISPLAY_QUOTE_PATTERN.match(value))


def find_quote_like_fields(value: Any, *, prefix: str = "$") -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            lowered = child_path.lower()
            if (
                any(token in lowered for token in QUOTE_PATH_TOKENS)
                or looks_like_wechat_refermsg(child)
                or looks_like_wechat_display_quote(child)
            ):
                kind = "field"
                if looks_like_wechat_refermsg(child):
                    kind = "wechat_refermsg_xml"
                elif looks_like_wechat_display_quote(child):
                    kind = "wechat_display_quote"
                candidates.append({"path": child_path, "kind": kind, "value_preview": value_preview(child)})
            candidates.extend(find_quote_like_fields(child, prefix=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            candidates.extend(find_quote_like_fields(child, prefix=f"{prefix}[{index}]"))
    return candidates


def unwrap_wechaty_log_envelope(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return record
    if not (record.get("source") or record.get("received_at") or record.get("trace_id")):
        return record
    if payload.get("channel") == "wechaty" or payload.get("conversation_id") or payload.get("source_message_id"):
        return payload
    return record


def _message_summary(record: dict[str, Any]) -> dict[str, Any]:
    line_number = record.get("_line_number")
    record = unwrap_wechaty_log_envelope(record)
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    talker = record.get("talker") if isinstance(record.get("talker"), dict) else {}
    room = record.get("room") if isinstance(record.get("room"), dict) else None
    return {
        "line_number": line_number if line_number is not None else record.get("_line_number"),
        "source_message_id": record.get("source_message_id") or record.get("message_id") or payload.get("id"),
        "conversation_id": record.get("conversation_id"),
        "sender_id": record.get("sender_id") or talker.get("id"),
        "sender_name": record.get("sender_name") or talker.get("name"),
        "is_room": bool(record.get("is_room")),
        "room_id": room.get("id") if isinstance(room, dict) else None,
        "room_name": room.get("name") if isinstance(room, dict) else None,
        "text": record.get("text") or record.get("raw_text") or payload.get("text") or "",
        "self_message": bool(record.get("self_message")),
    }


def analyze_records(records: Iterable[dict[str, Any]], *, max_records: int | None = None) -> dict[str, Any]:
    total_records = 0
    decode_errors: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    path_counts: Counter[str] = Counter()

    for record in records:
        if max_records is not None and total_records >= max_records:
            break
        total_records += 1
        if record.get("_decode_error"):
            decode_errors.append(
                {
                    "line_number": record.get("_line_number"),
                    "error": record.get("_decode_error"),
                    "raw_preview": record.get("_raw_preview"),
                }
            )
            continue
        message_record = unwrap_wechaty_log_envelope(record)
        candidates = find_quote_like_fields(message_record)
        for candidate in candidates:
            path_counts[candidate["path"]] += 1
        if candidates:
            candidate_records.append(
                {
                    **_message_summary(record),
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                }
            )

    return {
        "total_records": total_records,
        "candidate_record_count": len(candidate_records),
        "decode_error_count": len(decode_errors),
        "path_counts": dict(path_counts.most_common()),
        "candidate_records": candidate_records,
        "decode_errors": decode_errors,
        "next_sample_types": [
            "候选人引用邀约消息回复：可以",
            "候选人引用邀约消息回复：今天不来",
            "用户引用老板问询消息补充：0.5 或 1 都行",
            "用户引用闲聊消息回复：哈哈 / 对 / 不对",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze raw WeChaty JSONL logs for quote/reference payload fields.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"input file not found: {args.input}")

    report = analyze_records(iter_jsonl_records(args.input), max_records=args.max_records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "total_records": report["total_records"],
                "candidate_record_count": report["candidate_record_count"],
                "decode_error_count": report["decode_error_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
