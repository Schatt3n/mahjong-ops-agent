#!/usr/bin/env python3
"""核对每个已归档 badcase 是否拥有可执行回归测试。"""

from __future__ import annotations

import json
import hashlib
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BADCASE_PATHS = [
    ROOT / "eval" / "badcases" / "badcases.jsonl",
]
DATASET_PATHS = {
    "real_owner_chat_golden": ROOT / "eval" / "golden" / "real_owner_chat_golden.jsonl",
    "agent_runtime_regression": ROOT / "eval" / "regression" / "agent_runtime_regression.jsonl",
}
FIXED_STATUSES = {"fixed", "closed", "resolved"}
LIVE_EVAL_SCRIPT = ROOT / "scripts" / "run_real_owner_chat_live_eval.py"
REFERENCE_REVISION_PATH = ROOT / "eval" / "badcases" / "regression_reference_revisions.jsonl"
REFERENCE_SOURCE_PATH = "eval/badcases/badcases.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """加载jsonl，并恢复为领域层可使用的结构。"""
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
    """校验pytestref；失败时返回或抛出可审计的原因。"""
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
    """校验refs；失败时返回或抛出可审计的原因。"""
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


def source_record_digest(record: dict[str, Any]) -> str:
    """绑定完整原记录的精确JSON内容，引用迁移不能顺便修改期望、状态或原始观测。"""
    return hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def regression_reference_revisions() -> dict[str, list[dict[str, str]]]:
    """核验追加式测试引用迁移；只替换明确列出的pytest引用，其他原引用按顺序保留。

    迁移依据是代码重构，不替历史badcase改写架构期望或重新宣称模型语义通过。
    原记录摘要、完整旧引用、逐项替换及真实目标函数缺一不可；重复和孤儿修订均失败。
    """
    revisions = load_jsonl(REFERENCE_REVISION_PATH)
    originals = load_jsonl(ROOT / REFERENCE_SOURCE_PATH)
    effective: dict[str, list[dict[str, str]]] = {}
    revision_ids: set[str] = set()
    for revision in revisions:
        revision_id = revision.get("id")
        target_id = revision.get("source_record_id")
        source = revision.get("source")
        if (type(revision.get("schema_version")) is not int or revision["schema_version"] != 1
                or revision.get("kind") != "badcase_regression_reference_revision"
                or not isinstance(revision_id, str) or not revision_id.strip() or revision_id in revision_ids
                or not isinstance(target_id, str) or not target_id.strip() or target_id in effective
                or revision.get("source_path") != REFERENCE_SOURCE_PATH
                or not isinstance(source, dict) or source.get("kind") != "code_refactor"
                or not isinstance(source.get("reason"), str) or not source["reason"].strip()):
            raise ValueError("invalid, duplicate or non-refactor regression reference revision")
        matches = [record for record in originals if record_id(record) == target_id]
        if len(matches) != 1 or revision.get("source_record_sha256") != source_record_digest(matches[0]):
            raise ValueError(f"{target_id}: source record identity or digest differs from the original")
        previous = matches[0].get("regression_refs")
        if (not isinstance(previous, list) or not previous or revision.get("previous_regression_refs") != previous):
            raise ValueError(f"{target_id}: previous regression references differ from the original")
        replacements = revision.get("replacements")
        if not isinstance(replacements, list) or not replacements:
            raise ValueError(f"{target_id}: missing explicit reference replacements")
        by_old: dict[str, list[dict[str, str]]] = {}
        for replacement in replacements:
            if not isinstance(replacement, dict):
                raise ValueError(f"{target_id}: reference replacement must be an object")
            old, new = replacement.get("old_ref"), replacement.get("new_refs")
            old_key = json.dumps(old, sort_keys=True)
            if (not isinstance(old, dict) or set(old) != {"type", "id"} or old.get("type") != "pytest"
                    or old not in previous or old_key in by_old or not isinstance(new, list) or not new
                    or not isinstance(replacement.get("coverage_intent"), str) or not replacement["coverage_intent"].strip()):
                raise ValueError(f"{target_id}: old reference or coverage intent is missing, changed or duplicated")
            for ref in new:
                if (not isinstance(ref, dict) or set(ref) != {"type", "id"} or ref.get("type") != "pytest"
                        or not isinstance(ref.get("id"), str) or validate_pytest_ref(ref["id"]) is not None):
                    raise ValueError(f"{target_id}: replacement pytest reference is not executable")
            if len({json.dumps(ref, sort_keys=True) for ref in new}) != len(new):
                raise ValueError(f"{target_id}: duplicate replacement references")
            by_old[old_key] = new
        expected = []
        for ref in previous:
            for replacement in by_old.get(json.dumps(ref, sort_keys=True), [ref]):
                if replacement not in expected:
                    expected.append(replacement)
        if revision.get("current_regression_refs") != expected:
            raise ValueError(f"{target_id}: current references drop or add an undeclared regression obligation")
        effective[target_id] = deepcopy(expected)
        revision_ids.add(revision_id)
    return effective


def record_id(record: dict[str, Any]) -> str:
    """记录id，形成后续决策和审计所需的持久证据。"""
    return str(record.get("id") or record.get("badcase_id") or "<missing-id>")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def audit_badcases() -> tuple[int, int, list[str]]:
    """逐条检查当前引用；历史记录只读，引用修订失配仍作为覆盖门禁失败报告。"""
    ids_by_type = dataset_ids()
    fixed = 0
    open_count = 0
    errors: list[str] = []
    try:
        revised_refs = regression_reference_revisions()
    except (ValueError, OSError) as exc:
        revised_refs = {}
        errors.append(f"{display_path(REFERENCE_REVISION_PATH)}: {exc}")
    for path in BADCASE_PATHS:
        for record in load_jsonl(path):
            status = str(record.get("triage_status") or record.get("status") or "").strip().lower()
            label = f"{display_path(path)}:{record_id(record)}"
            if status not in FIXED_STATUSES:
                open_count += 1
                errors.append(f"{label}: badcase is not closed; triage_status/status={status or '<empty>'}")
                continue
            fixed += 1
            effective_record = record
            if path.resolve() == (ROOT / REFERENCE_SOURCE_PATH).resolve() and record_id(record) in revised_refs:
                effective_record = {**record, "regression_refs": revised_refs[record_id(record)]}
            record_errors = validate_refs(effective_record, ids_by_type)
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
