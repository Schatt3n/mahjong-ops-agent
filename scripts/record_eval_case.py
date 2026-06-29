from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mahjong_agent import ReplyAction  # noqa: E402


TZ = ZoneInfo("Asia/Shanghai")
DATASETS = {
    "golden": ROOT / "eval" / "golden_dataset.jsonl",
    "badcase": ROOT / "eval" / "badcases.jsonl",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="追加评估样本到 golden dataset 或 badcase 队列")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--id", required=True, help="样本唯一 id，例如 weak_intent_002")
    parser.add_argument("--name", required=True, help="样本名称")
    parser.add_argument("--text", required=True, help="输入消息文本")
    parser.add_argument("--sender-id", default="eval_user")
    parser.add_argument("--sender-name")
    parser.add_argument("--expected-action", choices=[item.value for item in ReplyAction])
    parser.add_argument("--contains", action="append", default=[], help="回复中应该包含的片段，可传多次")
    parser.add_argument("--should-reply", choices=["true", "false"])
    parser.add_argument("--tag", action="append", default=[], help="标签，可传多次")
    parser.add_argument("--metadata-json", default="{}", help="消息 metadata JSON")
    parser.add_argument("--note", default="")
    parser.add_argument("--replace", action="store_true", help="如果 id 已存在，则替换原样本")
    args = parser.parse_args()

    path = DATASETS[args.dataset]
    metadata = parse_json_object(args.metadata_json, "--metadata-json")
    record = build_record(args, metadata)
    write_record(path, record, replace=args.replace)
    print(f"written {args.dataset} case: {record['id']} -> {path}")
    return 0


def parse_json_object(raw: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} 不是合法 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} 必须是 JSON object")
    return value


def build_record(args: argparse.Namespace, metadata: dict[str, Any]) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    if args.expected_action:
        expected["action"] = args.expected_action
    if args.contains:
        expected["contains"] = args.contains
    if args.should_reply is not None:
        expected["should_reply"] = args.should_reply == "true"

    record: dict[str, Any] = {
        "schema_version": 1,
        "kind": args.dataset,
        "id": args.id,
        "name": args.name,
        "tags": args.tag,
        "text": args.text,
        "sender_id": args.sender_id,
        "metadata": metadata,
        "created_at": datetime.now(TZ).isoformat(),
    }
    if args.sender_name:
        record["sender_name"] = args.sender_name
    if expected:
        record["expected"] = expected
    if args.note:
        record["note"] = args.note
    return record


def write_record(path: pathlib.Path, record: dict[str, Any], *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_records(path)
    ids = [str(item.get("id")) for item in existing]
    if record["id"] in ids and not replace:
        raise SystemExit(f"id 已存在: {record['id']}。需要覆盖时加 --replace")

    if replace:
        existing = [item for item in existing if item.get("id") != record["id"]]
    existing.append(record)
    with path.open("w", encoding="utf-8") as fh:
        for item in existing:
            fh.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_records(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


if __name__ == "__main__":
    raise SystemExit(main())
