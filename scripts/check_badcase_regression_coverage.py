#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BADCASE_PATHS = [
    ROOT / "eval" / "badcases" / "badcases.jsonl",
]
DATASET_PATHS = {
    "scenario_golden": ROOT / "eval" / "golden" / "scenario_golden.jsonl",
    "boss_trial_golden": ROOT / "eval" / "golden" / "boss_trial_golden.jsonl",
    "real_owner_chat_golden": ROOT / "eval" / "golden" / "real_owner_chat_golden.jsonl",
    "controlled_workflow_regression": ROOT / "eval" / "regression" / "controlled_workflow_regression.jsonl",
    "agent_runtime_regression": ROOT / "eval" / "regression" / "agent_runtime_regression.jsonl",
}
FIXED_STATUSES = {"fixed", "closed", "resolved"}
LIVE_EVAL_SCRIPT = ROOT / "scripts" / "run_real_owner_chat_live_eval.py"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} JSON root must be object")
        records.append(record)
    return records


def dataset_ids() -> dict[str, set[str]]:
    ids = {
        dataset_type: {str(record.get("id")) for record in load_jsonl(path)}
        for dataset_type, path in DATASET_PATHS.items()
    }
    ids["live_eval"] = set(re.findall(r'scenario_id="([^"]+)"', LIVE_EVAL_SCRIPT.read_text(encoding="utf-8")))
    return ids


def validate_pytest_ref(ref_id: str) -> str | None:
    path_text, sep, test_name = ref_id.partition("::")
    if not sep or not path_text or not test_name:
        return "pytest ref must use tests/path.py::test_name"
    path = ROOT / path_text
    if not path.exists():
        return f"pytest file does not exist: {path_text}"
    text = path.read_text(encoding="utf-8")
    if not re.search(rf"def\s+{re.escape(test_name)}\s*\(", text):
        return f"pytest function not found: {ref_id}"
    return None


def validate_refs(record: dict[str, Any], ids_by_type: dict[str, set[str]]) -> list[str]:
    errors: list[str] = []
    refs = record.get("regression_refs")
    if not isinstance(refs, list) or not refs:
        return ["fixed badcase is missing non-empty regression_refs"]
    for index, ref in enumerate(refs):
        if not isinstance(ref, dict):
            errors.append(f"regression_refs[{index}] must be object")
            continue
        ref_type = str(ref.get("type") or "")
        ref_id = str(ref.get("id") or "")
        if ref_type in ids_by_type:
            if ref_id not in ids_by_type[ref_type]:
                errors.append(f"unknown {ref_type} id: {ref_id}")
            continue
        if ref_type == "pytest":
            error = validate_pytest_ref(ref_id)
            if error:
                errors.append(error)
            continue
        errors.append(f"unknown regression ref type: {ref_type or '<empty>'}")
    return errors


def record_id(record: dict[str, Any]) -> str:
    return str(record.get("id") or record.get("badcase_id") or "<missing-id>")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def audit_badcases() -> tuple[int, int, list[str]]:
    ids_by_type = dataset_ids()
    fixed = 0
    open_count = 0
    errors: list[str] = []
    for path in BADCASE_PATHS:
        for record in load_jsonl(path):
            status = str(record.get("triage_status") or record.get("status") or "").strip().lower()
            label = f"{display_path(path)}:{record_id(record)}"
            if status not in FIXED_STATUSES:
                open_count += 1
                errors.append(f"{label}: badcase is not closed; triage_status/status={status or '<empty>'}")
                continue
            fixed += 1
            record_errors = validate_refs(record, ids_by_type)
            for error in record_errors:
                errors.append(f"{label}: {error}")
    return fixed, open_count, errors


def main() -> int:
    fixed, open_count, errors = audit_badcases()
    if errors:
        print("FAIL badcase regression coverage")
        for error in errors:
            print(f"- {error}")
        print(f"\nfixed={fixed}, open={open_count}, failed={len(errors)}")
        return 1
    print(f"PASS badcase regression coverage: fixed={fixed}, open={open_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
