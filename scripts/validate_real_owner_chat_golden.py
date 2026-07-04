#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = ROOT / "eval" / "golden" / "real_owner_chat_golden.jsonl"
REQUIRED_EVAL_CASE_IDS = {
    "initial_request_uses_profile_defaults_and_searches_pool",
    "who_are_the_players_can_show_public_nickname_only",
    "group_duration_mismatch_records_exit",
    "human_likeness_reply_should_be_short_and_decision_focused",
}
REQUIRED_BUSINESS_FACT_IDS = {
    "profile_defaults_fill_missing_slots",
    "public_nickname_allowed_private_remark_forbidden",
    "group_invite_then_duration_mismatch",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: json line must be an object")
        payload["_line_number"] = line_number
        records.append(payload)
    return records


def validate_dataset(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    records_by_id = {str(record.get("id") or ""): record for record in records}
    for record in records:
        kind = str(record.get("kind") or "")
        if kind == "real_owner_chat_golden":
            errors.extend(_validate_transcript_record(record))
        elif kind == "real_owner_chat_eval_supplement":
            errors.extend(_validate_supplement_record(record, records_by_id))
        else:
            errors.append(_record_error(record, f"unsupported kind: {kind!r}"))
    return errors


def _validate_transcript_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return [_record_error(record, "messages must be a non-empty list")]

    source_files = set((record.get("source") or {}).get("source_image_files") or [])
    turns = [message.get("turn") for message in messages if isinstance(message, dict)]
    expected_turns = list(range(1, len(messages) + 1))
    if turns != expected_turns:
        errors.append(_record_error(record, "message turns must be consecutive from 1"))

    for message in messages:
        if not isinstance(message, dict):
            errors.append(_record_error(record, "message must be an object"))
            continue
        turn = message.get("turn")
        role = message.get("role")
        if role not in {"customer", "boss"}:
            errors.append(_record_error(record, f"turn {turn}: role must be customer or boss"))
        if not str(message.get("text") or "").strip():
            errors.append(_record_error(record, f"turn {turn}: text is required"))
        source_image = str(message.get("source_image") or "")
        if source_files and source_image not in source_files:
            errors.append(_record_error(record, f"turn {turn}: source_image not listed in source.source_image_files"))

    valid_turns = set(expected_turns)
    errors.extend(_validate_business_facts(record, valid_turns))
    errors.extend(_validate_eval_case_turn_refs(record, valid_turns))
    return errors


def _validate_supplement_record(record: dict[str, Any], records_by_id: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    parent_id = str(record.get("parent_id") or "")
    parent = records_by_id.get(parent_id)
    if not parent:
        return [_record_error(record, f"parent_id not found: {parent_id}")]

    parent_messages = parent.get("messages") if isinstance(parent.get("messages"), list) else []
    parent_turns = {message.get("turn") for message in parent_messages if isinstance(message, dict)}
    business_fact_ids = {str(item.get("id") or "") for item in record.get("business_facts") or [] if isinstance(item, dict)}
    eval_case_ids = {str(item.get("id") or "") for item in record.get("eval_cases") or [] if isinstance(item, dict)}

    missing_facts = sorted(REQUIRED_BUSINESS_FACT_IDS - business_fact_ids)
    if missing_facts:
        errors.append(_record_error(record, f"missing required business facts: {missing_facts}"))

    missing_eval_cases = sorted(REQUIRED_EVAL_CASE_IDS - eval_case_ids)
    if missing_eval_cases:
        errors.append(_record_error(record, f"missing required eval cases: {missing_eval_cases}"))

    hidden_context = record.get("hidden_context")
    if not isinstance(hidden_context, list) or not hidden_context:
        errors.append(_record_error(record, "hidden_context must be a non-empty list"))
    else:
        for item in hidden_context:
            if not isinstance(item, dict):
                errors.append(_record_error(record, "hidden_context item must be an object"))
                continue
            after_turn = item.get("after_turn")
            if after_turn not in parent_turns:
                errors.append(_record_error(record, f"hidden_context after_turn not found in parent transcript: {after_turn}"))
            if not str(item.get("event") or "").strip():
                errors.append(_record_error(record, f"hidden_context after_turn {after_turn}: event is required"))

    profile_facts = ((record.get("customer_profile_assumptions") or {}).get("profile_facts") or [])
    profile_text = "\n".join(str(item) for item in profile_facts)
    for expected in ("95% 情况打 0.5", "95% 情况是一个人", "常打杭麻"):
        if expected not in profile_text:
            errors.append(_record_error(record, f"missing profile assumption: {expected}"))

    for case in record.get("eval_cases") or []:
        if not isinstance(case, dict):
            errors.append(_record_error(record, "eval case must be an object"))
            continue
        for ref in case.get("hidden_context_refs") or []:
            if ref not in business_fact_ids:
                errors.append(_record_error(record, f"eval case {case.get('id')}: hidden_context_ref not found: {ref}"))
    return errors


def _validate_business_facts(record: dict[str, Any], valid_turns: set[int]) -> list[str]:
    errors: list[str] = []
    for fact in record.get("business_facts") or []:
        if not isinstance(fact, dict):
            errors.append(_record_error(record, "business fact must be an object"))
            continue
        evidence_turns = fact.get("evidence_turns") or []
        if not evidence_turns:
            errors.append(_record_error(record, f"business fact {fact.get('id')}: evidence_turns is required"))
        missing = [turn for turn in evidence_turns if turn not in valid_turns]
        if missing:
            errors.append(_record_error(record, f"business fact {fact.get('id')}: invalid evidence_turns {missing}"))
    return errors


def _validate_eval_case_turn_refs(record: dict[str, Any], valid_turns: set[int]) -> list[str]:
    errors: list[str] = []
    for case in record.get("eval_cases") or []:
        if not isinstance(case, dict):
            errors.append(_record_error(record, "eval case must be an object"))
            continue
        case_id = case.get("id")
        context_turns = case.get("context_turns") or []
        invalid_context_turns = [turn for turn in context_turns if turn not in valid_turns]
        if invalid_context_turns:
            errors.append(_record_error(record, f"eval case {case_id}: invalid context_turns {invalid_context_turns}"))
        input_turn = case.get("input_turn")
        if input_turn is not None and input_turn not in valid_turns:
            errors.append(_record_error(record, f"eval case {case_id}: invalid input_turn {input_turn}"))
    return errors


def _record_error(record: dict[str, Any], message: str) -> str:
    record_id = record.get("id") or f"line:{record.get('_line_number')}"
    return f"{record_id}: {message}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate real owner chat golden dataset consistency.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    args = parser.parse_args(argv)

    records = read_jsonl(args.dataset)
    errors = validate_dataset(records)
    payload = {
        "dataset": str(args.dataset),
        "record_count": len(records),
        "error_count": len(errors),
        "errors": errors,
        "status": "passed" if not errors else "failed",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
