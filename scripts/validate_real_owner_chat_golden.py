#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mahjong_agent_runtime.copywriting import REAL_OWNER_WECHAT_STYLE_EXAMPLES  # noqa: E402

DEFAULT_DATASET_PATH = ROOT / "eval" / "golden" / "real_owner_chat_golden.jsonl"
DEFAULT_TIMELINE_PATH = ROOT / "eval" / "golden" / "real_owner_chat_timeline_20260705_20260718.json"
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


def read_json_document(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON document must be an object")
    payload["_source_path"] = str(path)
    return payload


def read_default_records() -> list[dict[str, Any]]:
    return [*read_jsonl(DEFAULT_DATASET_PATH), read_json_document(DEFAULT_TIMELINE_PATH)]


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
    errors.extend(_validate_real_owner_style_examples(records))
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
    errors.extend(_validate_timeline_extensions(record, messages, expected_turns))
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


def _validate_timeline_extensions(
    record: dict[str, Any],
    messages: list[dict[str, Any]],
    expected_turns: list[int],
) -> list[str]:
    """Validate optional episode and replay metadata for multi-day transcripts."""

    episodes = record.get("episodes")
    metrics = record.get("timeline_metrics")
    replay_scenarios = record.get("replay_scenarios")
    if episodes is None and metrics is None and replay_scenarios is None:
        return []

    errors: list[str] = []
    valid_turns = set(expected_turns)
    episode_items = episodes if isinstance(episodes, list) else []
    if not episode_items:
        errors.append(_record_error(record, "episodes must be a non-empty list when timeline metadata is present"))
    else:
        episode_ids: set[str] = set()
        covered_turns: list[int] = []
        for episode in episode_items:
            if not isinstance(episode, dict):
                errors.append(_record_error(record, "episode must be an object"))
                continue
            episode_id = str(episode.get("id") or "")
            if not episode_id:
                errors.append(_record_error(record, "episode id is required"))
            elif episode_id in episode_ids:
                errors.append(_record_error(record, f"duplicate episode id: {episode_id}"))
            episode_ids.add(episode_id)

            start_turn = episode.get("start_turn")
            end_turn = episode.get("end_turn")
            if not isinstance(start_turn, int) or not isinstance(end_turn, int) or start_turn > end_turn:
                errors.append(_record_error(record, f"episode {episode_id}: invalid turn range"))
                continue
            episode_turns = list(range(start_turn, end_turn + 1))
            missing = [turn for turn in episode_turns if turn not in valid_turns]
            if missing:
                errors.append(_record_error(record, f"episode {episode_id}: invalid turns {missing}"))
            covered_turns.extend(episode_turns)

            exchange_rounds = episode.get("business_exchange_rounds")
            if not isinstance(exchange_rounds, int) or exchange_rounds <= 0:
                errors.append(
                    _record_error(record, f"episode {episode_id}: business_exchange_rounds must be a positive integer")
                )
            if not str(episode.get("outcome") or "").strip():
                errors.append(_record_error(record, f"episode {episode_id}: outcome is required"))

        if covered_turns != expected_turns:
            errors.append(_record_error(record, "episode turn ranges must partition all messages in order"))

    if not isinstance(metrics, dict):
        errors.append(_record_error(record, "timeline_metrics must be an object"))
    else:
        role_counts = {
            "customer": sum(1 for message in messages if message.get("role") == "customer"),
            "boss": sum(1 for message in messages if message.get("role") == "boss"),
        }
        expected_values = {
            "message_count": len(messages),
            "customer_message_count": role_counts["customer"],
            "boss_message_count": role_counts["boss"],
            "episode_count": len(episode_items),
        }
        for field, expected in expected_values.items():
            if metrics.get(field) != expected:
                errors.append(
                    _record_error(
                        record,
                        f"timeline_metrics.{field} must be {expected}, got {metrics.get(field)!r}",
                    )
                )
        if not str(metrics.get("round_definition") or "").strip():
            errors.append(_record_error(record, "timeline_metrics.round_definition is required"))

    if not isinstance(replay_scenarios, list) or not replay_scenarios:
        errors.append(_record_error(record, "replay_scenarios must be a non-empty list"))
    else:
        scenario_ids: set[str] = set()
        for scenario in replay_scenarios:
            if not isinstance(scenario, dict):
                errors.append(_record_error(record, "replay scenario must be an object"))
                continue
            scenario_id = str(scenario.get("id") or "")
            if not scenario_id:
                errors.append(_record_error(record, "replay scenario id is required"))
            elif scenario_id in scenario_ids:
                errors.append(_record_error(record, f"duplicate replay scenario id: {scenario_id}"))
            scenario_ids.add(scenario_id)
            events = scenario.get("events")
            if not isinstance(events, list) or not events:
                errors.append(_record_error(record, f"replay scenario {scenario_id}: events must be non-empty"))
                continue
            for event in events:
                if not isinstance(event, dict):
                    errors.append(_record_error(record, f"replay scenario {scenario_id}: event must be an object"))
                    continue
                source_turn = event.get("source_turn")
                if source_turn is not None and source_turn not in valid_turns:
                    errors.append(
                        _record_error(
                            record,
                            f"replay scenario {scenario_id}: source_turn not found: {source_turn}",
                        )
                    )
                if event.get("actor") not in {"customer", "boss", "system"}:
                    errors.append(
                        _record_error(
                            record,
                            f"replay scenario {scenario_id}: actor must be customer, boss or system",
                        )
                    )
                if not str(event.get("text") or "").strip():
                    errors.append(_record_error(record, f"replay scenario {scenario_id}: event text is required"))
            if not isinstance(scenario.get("expected"), dict) or not scenario.get("expected"):
                errors.append(_record_error(record, f"replay scenario {scenario_id}: expected is required"))
    return errors


def _record_error(record: dict[str, Any], message: str) -> str:
    record_id = record.get("id") or f"line:{record.get('_line_number')}"
    return f"{record_id}: {message}"


def _validate_real_owner_style_examples(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    evidence_texts = _collect_style_evidence_texts(records)
    normalized_evidence = [_normalize_style_text(text) for text in evidence_texts if str(text).strip()]
    combined_evidence = "\n".join(normalized_evidence)

    for index, example in enumerate(REAL_OWNER_WECHAT_STYLE_EXAMPLES, start=1):
        good = str(example.get("good") or "").strip()
        bad = str(example.get("bad") or "").strip()
        scenario = str(example.get("scenario") or "").strip()
        style_note = str(example.get("style_note") or "").strip()
        if not good or not bad or not scenario or not style_note:
            errors.append(f"REAL_OWNER_WECHAT_STYLE_EXAMPLES[{index}]: scenario/good/bad/style_note are required")
            continue
        if _normalize_style_text(good) == _normalize_style_text(bad):
            errors.append(f"REAL_OWNER_WECHAT_STYLE_EXAMPLES[{index}]: good and bad examples must differ")
        if not _style_example_backed_by_evidence(good, normalized_evidence, combined_evidence):
            errors.append(
                f"REAL_OWNER_WECHAT_STYLE_EXAMPLES[{index}]: good phrase is not backed by real owner dataset: {good!r}"
            )
    return errors


def _collect_style_evidence_texts(records: list[dict[str, Any]]) -> list[str]:
    evidence: list[str] = []
    for record in records:
        messages = record.get("messages") if isinstance(record.get("messages"), list) else []
        for message in messages:
            if isinstance(message, dict):
                evidence.append(str(message.get("text") or ""))
        evidence.extend(_adjacent_boss_message_windows(messages))
        for case in record.get("eval_cases") or []:
            if not isinstance(case, dict):
                continue
            expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
            evidence.extend(str(item) for item in expected.get("good_examples") or [])
    return evidence


def _adjacent_boss_message_windows(messages: list[Any]) -> list[str]:
    windows: list[str] = []
    boss_run: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "boss":
            boss_run.append(str(message.get("text") or ""))
            continue
        windows.extend(_sliding_windows(boss_run, max_size=4))
        boss_run = []
    windows.extend(_sliding_windows(boss_run, max_size=4))
    return windows


def _sliding_windows(items: list[str], *, max_size: int) -> list[str]:
    windows: list[str] = []
    for start in range(len(items)):
        for end in range(start + 2, min(len(items), start + max_size) + 1):
            windows.append("，".join(item for item in items[start:end] if item))
    return windows


def _style_example_backed_by_evidence(good: str, normalized_evidence: list[str], combined_evidence: str) -> bool:
    normalized_good = _normalize_style_text(good)
    if not normalized_good:
        return False
    if any(normalized_good in evidence for evidence in normalized_evidence):
        return True
    required_parts = [
        _normalize_style_text(part)
        for part in re.split(r"[\s,，、。.!！?？;；]+", good)
        if len(_normalize_style_text(part)) >= 2
    ]
    return bool(required_parts) and all(part in combined_evidence for part in required_parts)


def _normalize_style_text(text: str) -> str:
    return re.sub(r"[\s,，、。.!！?？;；:：/\\（）()“”\"'`]+", "", str(text)).lower()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate real owner chat golden dataset consistency.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Validate one JSONL dataset. By default validates the base JSONL and the multi-day timeline JSON.",
    )
    args = parser.parse_args(argv)

    records = read_jsonl(args.dataset) if args.dataset else read_default_records()
    errors = validate_dataset(records)
    payload = {
        "dataset": str(args.dataset) if args.dataset else [str(DEFAULT_DATASET_PATH), str(DEFAULT_TIMELINE_PATH)],
        "record_count": len(records),
        "error_count": len(errors),
        "errors": errors,
        "status": "passed" if not errors else "failed",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
