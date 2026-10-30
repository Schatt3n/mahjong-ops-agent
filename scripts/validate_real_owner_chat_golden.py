#!/usr/bin/env python3
"""校验真实老板私聊黄金数据的时间线、事实和评测引用。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from mahjong_agent_runtime.copywriting import REAL_OWNER_WECHAT_STYLE_EXAMPLES  # noqa: E402
from eval_product_contracts import PRODUCT_REVISION_KIND, resolve_product_contract_revisions  # noqa: E402

DEFAULT_DATASET_PATH = ROOT / "eval" / "golden" / "real_owner_chat_golden.jsonl"
LIVE_EVAL_SCRIPT = ROOT / "scripts" / "run_real_owner_chat_live_eval.py"
DEFAULT_TIMELINE_PATH = ROOT / "eval" / "golden" / "real_owner_chat_timeline_20260705_20260718.json"
DEFAULT_TIMELINE_PATHS = tuple(sorted((ROOT / "eval" / "golden").glob("real_owner_chat_timeline_*.json")))
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
    """读取jsonl，不产生业务副作用。"""
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
    """读取jsondocument，不产生业务副作用。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON document must be an object")
    payload["_source_path"] = str(path)
    return payload


def read_default_records() -> list[dict[str, Any]]:
    """读取defaultrecords，不产生业务副作用。"""
    return [
        *read_jsonl(DEFAULT_DATASET_PATH),
        *(read_json_document(path) for path in DEFAULT_TIMELINE_PATHS),
    ]


def validate_dataset(records: list[dict[str, Any]]) -> list[str]:
    """校验dataset；失败时返回或抛出可审计的原因。"""
    errors: list[str] = []
    records_by_id = {str(record.get("id") or ""): record for record in records}
    live_scenario_ids = set(re.findall(r'scenario_id\s*=\s*"([^"]+)"', LIVE_EVAL_SCRIPT.read_text(encoding="utf-8")))
    seen_record_ids: set[str] = set()
    seen_case_ids: set[str] = set()
    for record in records:
        record_id = str(record.get("id") or "")
        if not record_id or record_id in seen_record_ids:
            errors.append(_record_error(record, f"missing or duplicate record id: {record_id!r}"))
        seen_record_ids.add(record_id)
        record_cases = record.get("eval_cases")
        for case in record_cases if isinstance(record_cases, list) else []:
            if isinstance(case, dict) and isinstance(case.get("id"), str):
                case_id = case["id"].strip()
                if case_id in seen_case_ids:
                    errors.append(_record_error(record, f"duplicate dataset eval case id: {case_id}"))
                seen_case_ids.add(case_id)
        kind = str(record.get("kind") or "")
        if kind == "real_owner_chat_golden":
            errors.extend(_validate_transcript_record(record))
        elif kind == "real_owner_chat_eval_supplement":
            errors.extend(_validate_supplement_record(record, records_by_id))
        elif kind == "owner_chat_live_semantic_regression":
            errors.extend(_validate_live_semantic_regression_record(record))
        elif kind == "owner_chat_hard_assertion_revision":
            errors.extend(_validate_hard_assertion_review_record(record))
        elif kind == PRODUCT_REVISION_KIND:
            # 元数据只在严格解析后生效，不把真人产品指令冒充旧微信转写。
            pass
        else:
            errors.append(_record_error(record, f"unsupported kind: {kind!r}"))
        errors.extend(_validate_semantic_eval_expectations(record, live_scenario_ids))
    try:
        snapshots = resolve_hard_assertion_snapshots(records)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        errors.extend(_validate_current_hard_assertion_snapshots(snapshots))
    errors.extend(_validate_real_owner_style_examples(records))
    try:
        from run_inquiry_grounding_eval import inquiry_input_catalog
        from run_real_owner_chat_live_eval import live_eval_scenarios
        catalog = {scene.scenario_id: {"required_reply_any": scene.required_reply_any,
                   "forbidden_reply_contains": scene.forbidden_reply_contains} for scene in live_eval_scenarios()}
        resolve_product_contract_revisions(records, inquiry_inputs=inquiry_input_catalog(), scenario_contracts=catalog)
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def _exact_json_value(value: Any) -> str:
    """以 JSON 类型精确比较快照，避免布尔值和整数被 Python 等号混为一谈。"""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _legacy_live_scenario_mapping() -> dict[str, str]:
    """静态读取运行器既有的人工案例映射，补录元数据不能自行创造另一套来源映射。"""
    for node in ast.parse(LIVE_EVAL_SCRIPT.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "LEGACY_GOLDEN_SCENARIOS" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    return {}


def _register_hard_assertion_baselines(
    record: dict[str, Any],
    prior_cases: dict[str, tuple[str, dict[str, Any]]],
    duplicate_case_ids: set[str],
    snapshots: dict[tuple[str, str], dict[str, Any]],
    baselines_seen: set[str],
    legacy_mapping: dict[str, str],
) -> list[str]:
    """为缺少硬快照的历史人工案例追加来源基线，不能覆盖原记录或跳过修订链。

    来源必须是此前唯一案例，其内容摘要与旧运行器的明确映射都要相符。
    本记录的新案例、未来案例、已有快照与第二次补录均不接受。
    """
    if "hard_assertion_baselines" not in record:
        return []
    baselines = record["hard_assertion_baselines"]
    if record.get("kind") != "owner_chat_hard_assertion_revision":
        return [_record_error(record, "hard assertion baselines require a metadata-only review record")]
    if not isinstance(baselines, list) or not baselines:
        return [_record_error(record, "hard_assertion_baselines must be a non-empty list")]
    errors: list[str] = []
    for baseline in baselines:
        if not isinstance(baseline, dict):
            errors.append(_record_error(record, "hard assertion baseline must be an object"))
            continue
        invalid: list[str] = []
        for field in ("baseline_id", "target_case_id", "source_record_id", "live_scenario_id", "source_case_sha256", "source_run_id", "source_catalog_commit"):
            if not isinstance(baseline.get(field), str) or not baseline[field].strip():
                invalid.append(f"hard assertion baseline {field} must be a non-empty string")
        baseline_id = baseline.get("baseline_id")
        if isinstance(baseline_id, str):
            if baseline_id in baselines_seen:
                invalid.append(f"duplicate hard assertion baseline_id: {baseline_id}")
            baselines_seen.add(baseline_id)
        target, live_id = baseline.get("target_case_id"), baseline.get("live_scenario_id")
        prior = prior_cases.get(target) if isinstance(target, str) else None
        key = (target, live_id) if isinstance(target, str) and isinstance(live_id, str) else None
        if prior is None or target in duplicate_case_ids or baseline.get("source_record_id") != prior[0]:
            invalid.append("hard assertion baseline target must be a unique prior case in the source record")
        else:
            case = prior[1]
            if "existing_hard_assertions_snapshot" in case or "live_scenario_id" in case or any(item[0] == target for item in snapshots):
                invalid.append("hard assertion baseline cannot replace an existing snapshot or live mapping")
            if legacy_mapping.get(target) != live_id:
                invalid.append("hard assertion baseline must match the existing legacy live mapping")
            digest = hashlib.sha256(_exact_json_value(case).encode("utf-8")).hexdigest()
            if baseline.get("source_case_sha256") != digest:
                invalid.append("hard assertion baseline source_case_sha256 does not match the original case")
        if not isinstance(baseline.get("snapshot"), dict) or not baseline["snapshot"]:
            invalid.append("hard assertion baseline snapshot must be a non-empty object")
        if invalid:
            errors.extend(_record_error(record, error) for error in invalid)
        else:
            snapshots[key] = baseline["snapshot"]
    return errors


def resolve_hard_assertion_snapshots(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """解析评测资产的追加硬断言修订链，不改原始记录或运行时配置。

    同一记录先登记原案例，随后按记录顺序应用显式修订；旧快照必须精确等于
    上一步，未知/未来目标、错场景、重复修订和断链均拒绝。修订不是新语义案例。
    """
    snapshots: dict[tuple[str, str], dict[str, Any]] = {}
    case_ids: set[str] = set()
    duplicate_case_ids: set[str] = set()
    revisions_seen: set[str] = set()
    baselines_seen: set[str] = set()
    prior_cases: dict[str, tuple[str, dict[str, Any]]] = {}
    legacy_mapping = _legacy_live_scenario_mapping()
    errors: list[str] = []
    for record in records:
        # 基线只能引用先前记录；同记录新增的合成案例不能伪装成历史来源。
        errors.extend(_register_hard_assertion_baselines(
            record, prior_cases, duplicate_case_ids, snapshots, baselines_seen, legacy_mapping,
        ))
        for case in record.get("eval_cases", []) if isinstance(record.get("eval_cases"), list) else []:
            if not isinstance(case, dict):
                continue
            case_id, live_id = case.get("id"), case.get("live_scenario_id")
            if isinstance(case_id, str):
                if case_id in case_ids:
                    duplicate_case_ids.add(case_id)
                    errors.append(_record_error(record, f"duplicate hard assertion case id: {case_id}"))
                case_ids.add(case_id)
                prior_cases[case_id] = (str(record.get("id") or ""), case)
            snapshot = case.get("existing_hard_assertions_snapshot")
            if isinstance(case_id, str) and isinstance(live_id, str) and isinstance(snapshot, dict) and snapshot:
                snapshots[(case_id, live_id)] = snapshot
        if "hard_assertion_revisions" not in record:
            continue
        revisions = record["hard_assertion_revisions"]
        if not isinstance(revisions, list) or not revisions:
            errors.append(_record_error(record, "hard_assertion_revisions must be a non-empty list"))
            continue
        for revision in revisions:
            if not isinstance(revision, dict):
                errors.append(_record_error(record, "hard assertion revision must be an object"))
                continue
            revision_errors: list[str] = []
            for field in ("revision_id", "target_case_id", "live_scenario_id", "reason", "source_run_id"):
                if not isinstance(revision.get(field), str) or not revision[field].strip():
                    revision_errors.append(f"hard assertion revision {field} must be a non-empty string")
            revision_id = revision.get("revision_id")
            if isinstance(revision_id, str):
                if revision_id.strip() in revisions_seen:
                    revision_errors.append(f"duplicate hard assertion revision_id: {revision_id}")
                revisions_seen.add(revision_id.strip())
            for field in ("previous_snapshot", "current_snapshot"):
                if not isinstance(revision.get(field), dict) or not revision[field]:
                    revision_errors.append(f"hard assertion revision {field} must be a non-empty object")
            target, live_id = revision.get("target_case_id"), revision.get("live_scenario_id")
            key = (target, live_id) if isinstance(target, str) and isinstance(live_id, str) else None
            if key not in snapshots or target in duplicate_case_ids:
                revision_errors.append("hard assertion revision target must be a unique prior case with matching live_scenario_id")
            elif _exact_json_value(revision.get("previous_snapshot")) != _exact_json_value(snapshots[key]):
                revision_errors.append("hard assertion revision previous_snapshot does not match the preceding snapshot")
            if (isinstance(revision.get("previous_snapshot"), dict) and isinstance(revision.get("current_snapshot"), dict)
                    and _exact_json_value(revision["previous_snapshot"]) == _exact_json_value(revision["current_snapshot"])):
                revision_errors.append("hard assertion revision must change the preceding snapshot")
            if revision_errors:
                errors.extend(_record_error(record, error) for error in revision_errors)
            else:
                snapshots[key] = revision["current_snapshot"]
    if errors:
        raise ValueError("; ".join(errors))
    return snapshots


def _validate_current_hard_assertion_snapshots(snapshots: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    """从唯一 live catalog 构建场景声明核对最新快照，不执行 setup、模型或业务工具。"""
    if not snapshots:
        return []
    spec = importlib.util.spec_from_file_location("owner_live_hard_assertion_catalog", LIVE_EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        return ["cannot load live scenario catalog for hard assertions"]
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    scenarios = {scene.scenario_id: scene for scene in module.live_eval_scenarios()}
    errors: list[str] = []
    for (case_id, live_id), snapshot in snapshots.items():
        scene = scenarios.get(live_id)
        for field, expected in snapshot.items():
            if scene is None or not hasattr(scene, field) or _exact_json_value(getattr(scene, field)) != _exact_json_value(expected):
                errors.append(f"eval case {case_id}: latest hard snapshot {field} must match live scenario {live_id}")
    return errors


def _live_scenario_fixture_refs() -> dict[str, str]:
    """静态读取既有场景声明的 setup 引用，不执行夹具或创建业务运行库。"""
    refs: dict[str, str] = {}
    for node in ast.walk(ast.parse(LIVE_EVAL_SCRIPT.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "LiveEvalScenario":
            continue
        fields = {keyword.arg: keyword.value for keyword in node.keywords}
        scenario_id, setup = fields.get("scenario_id"), fields.get("setup")
        if isinstance(scenario_id, ast.Constant) and isinstance(scenario_id.value, str) and isinstance(setup, ast.Name):
            refs[scenario_id.value] = f"scripts/run_real_owner_chat_live_eval.py::{setup.id}"
    return refs


def _validate_live_semantic_regression_record(record: dict[str, Any]) -> list[str]:
    """合成夹具单独验证输入和上下文，不伪造真人消息轮次或来源。"""
    errors: list[str] = []
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        errors.append(_record_error(record, "synthetic schema_version must be integer 1"))
    source = record.get("source")
    if not isinstance(source, dict) or source.get("kind") != "synthetic_regression_fixture" or source.get("not_original_human_transcript") is not True:
        errors.append(_record_error(record, "synthetic source must declare synthetic_regression_fixture and not_original_human_transcript=true"))
    errors.extend(_validate_hypothetical_derivation(record, source))
    cases = record.get("eval_cases")
    if not isinstance(cases, list) or not cases:
        return [*errors, _record_error(record, "synthetic eval_cases must be a non-empty list")]
    fixture_refs = _live_scenario_fixture_refs()
    seen_scenarios: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            errors.append(_record_error(record, "synthetic eval case must be an object"))
            continue
        prefix = f"eval case {case.get('id')}"
        live_id = case.get("live_scenario_id")
        if isinstance(live_id, str) and live_id in seen_scenarios:
            errors.append(_record_error(record, f"{prefix}: duplicate synthetic live_scenario_id: {live_id}"))
        if isinstance(live_id, str):
            seen_scenarios.add(live_id)
        case_source = case.get("source")
        if not isinstance(case_source, dict) or case_source.get("kind") != "synthetic_regression_fixture" or case_source.get("not_original_human_transcript") is not True:
            errors.append(_record_error(record, f"{prefix}: synthetic case source is required"))
        errors.extend(_validate_hypothetical_derivation(record, case_source))
        user_input = case.get("input")
        if not isinstance(user_input, dict):
            errors.append(_record_error(record, f"{prefix}: input must be an object"))
        else:
            for field in ("text", "sender_id", "sender_name", "conversation_id", "message_id"):
                if not isinstance(user_input.get(field), str) or not user_input[field].strip():
                    errors.append(_record_error(record, f"{prefix}: input.{field} must be a non-empty string"))
            if "quoted_message" in user_input:
                quote = user_input["quoted_message"]
                if not isinstance(quote, dict) or not isinstance(quote.get("text"), str) or not quote["text"].strip():
                    errors.append(_record_error(record, f"{prefix}: input.quoted_message.text must be a non-empty string"))
        context = case.get("context")
        if not isinstance(context, dict):
            errors.append(_record_error(record, f"{prefix}: context must be an object"))
            continue
        fixture_ref = context.get("fixture_ref")
        expected_fixture = fixture_refs.get(live_id) if isinstance(live_id, str) else None
        if not isinstance(fixture_ref, str) or fixture_ref != expected_fixture:
            errors.append(_record_error(record, f"{prefix}: context.fixture_ref must match the live scenario setup"))
        if not isinstance(context.get("known"), dict) or not context["known"]:
            errors.append(_record_error(record, f"{prefix}: context.known must be a non-empty object"))
        unknown = context.get("unknown")
        if not isinstance(unknown, list) or not unknown or any(not isinstance(value, str) or not value.strip() for value in unknown):
            errors.append(_record_error(record, f"{prefix}: context.unknown must be a non-empty list of strings"))
        if not isinstance(case.get("existing_hard_assertions_snapshot"), dict) or not case["existing_hard_assertions_snapshot"]:
            errors.append(_record_error(record, f"{prefix}: existing_hard_assertions_snapshot must be a non-empty object"))
        errors.extend(_validate_hard_assertions_history(record, case))
    return errors


def _validate_hard_assertion_review_record(record: dict[str, Any]) -> list[str]:
    """验证仅元数据的产品合同审查来源，实际硬快照仍走唯一修订链与 catalog 核对。"""
    errors: list[str] = []
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        errors.append(_record_error(record, "hard assertion review schema_version must be integer 1"))
    if any(field in record for field in ("eval_cases", "messages", "input", "expected")):
        errors.append(_record_error(record, "hard assertion review must be metadata-only without eval_cases or transcript"))
    source = record.get("source")
    if not isinstance(source, dict) or source.get("kind") != "user_product_contract_review" or source.get("not_original_human_transcript") is not True:
        errors.append(_record_error(record, "hard assertion review source must declare user_product_contract_review and not_original_human_transcript=true"))
    else:
        for field in ("statement", "basis", "source_run_id", "observed_date"):
            if not isinstance(source.get(field), str) or not source[field].strip():
                errors.append(_record_error(record, f"hard assertion review source {field} must be a non-empty string"))
    if not isinstance(record.get("hard_assertion_revisions"), list) or not record["hard_assertion_revisions"]:
        errors.append(_record_error(record, "hard assertion review requires non-empty hard_assertion_revisions"))
    return errors


def _validate_hypothetical_derivation(record: dict[str, Any], source: Any) -> list[str]:
    """派生假设必须标明来源与假设，不得冒充原始真人输入或成功回放证据。"""
    if not isinstance(source, dict) or "derivation_kind" not in source:
        return []
    errors: list[str] = []
    if source["derivation_kind"] != "hypothetical_real_failure_derivation":
        errors.append(_record_error(record, "derivation_kind must be hypothetical_real_failure_derivation"))
    for field in ("source_run_id", "source_scenario_id", "hypothesis"):
        if not isinstance(source.get(field), str) or not source[field].strip():
            errors.append(_record_error(record, f"hypothetical derivation {field} must be a non-empty string"))
    return errors


def _validate_hard_assertions_history(record: dict[str, Any], case: dict[str, Any]) -> list[str]:
    """保留硬验收修订的旧要求及原因，历史快照不冒充当前场景合同。

    未修订的旧资产无需新增字段；一旦声明修订历史，就必须能区分版本并回溯
    对应评测批次。这里不让历史快照匹配当前配置，否则会迫使覆盖原失败要求。
    """
    if "hard_assertions_history" not in case:
        return []
    prefix = f"eval case {case.get('id')}"
    history = case["hard_assertions_history"]
    if not isinstance(history, list) or not history:
        return [_record_error(record, f"{prefix}: hard_assertions_history must be a non-empty list")]
    errors: list[str] = []
    revision_ids: set[str] = set()
    for revision in history:
        if not isinstance(revision, dict):
            errors.append(_record_error(record, f"{prefix}: hard assertion revision must be an object"))
            continue
        for field in ("revision_id", "reason", "source_run_id"):
            if not isinstance(revision.get(field), str) or not revision[field].strip():
                errors.append(_record_error(record, f"{prefix}: hard assertion {field} must be a non-empty string"))
        revision_id = revision.get("revision_id")
        if isinstance(revision_id, str) and revision_id.strip():
            if revision_id.strip() in revision_ids:
                errors.append(_record_error(record, f"{prefix}: duplicate hard assertion revision_id: {revision_id}"))
            revision_ids.add(revision_id.strip())
        if not isinstance(revision.get("previous_snapshot"), dict) or not revision["previous_snapshot"]:
            errors.append(_record_error(record, f"{prefix}: hard assertion previous_snapshot must be a non-empty object"))
    return errors


def _validate_semantic_eval_expectations(record: dict[str, Any], live_scenario_ids: set[str]) -> list[str]:
    """校验显式声明的语义合同与真实场景引用；旧证据未声明时保持原样。"""
    errors: list[str] = []
    seen_case_ids: set[str] = set()
    record_cases = record.get("eval_cases")
    for case in record_cases if isinstance(record_cases, list) else []:
        if not isinstance(case, dict):
            continue
        expected = case.get("expected")
        semantic_fields = ("reply_reference", "required_meaning", "forbidden_meaning")
        if record.get("kind") != "owner_chat_live_semantic_regression" and "live_scenario_id" not in case and not (
            isinstance(expected, dict) and any(field in expected for field in semantic_fields)
        ):
            continue
        case_id = case.get("id")
        prefix = f"eval case {case_id}"
        if not isinstance(case_id, str) or not case_id.strip():
            errors.append(_record_error(record, "semantic eval case id must be a non-empty string"))
        elif case_id.strip() in seen_case_ids:
            errors.append(_record_error(record, f"duplicate semantic eval case id: {case_id}"))
        else:
            seen_case_ids.add(case_id.strip())
        live_id = case.get("live_scenario_id")
        if not isinstance(live_id, str) or not live_id.strip():
            errors.append(_record_error(record, f"{prefix}: live_scenario_id must be a non-empty string"))
        elif live_id not in live_scenario_ids:
            errors.append(_record_error(record, f"{prefix}: live_scenario_id not found: {live_id}"))
        if not isinstance(expected, dict):
            errors.append(_record_error(record, f"{prefix}: semantic expected must be an object"))
            continue
        reply_reference = expected.get("reply_reference")
        if not isinstance(reply_reference, str) or not reply_reference.strip():
            errors.append(_record_error(record, f"{prefix}: reply_reference must be a non-empty string"))
        # 同一案例内跨正反语义保持唯一，避免相同 ID 同时要求和禁止。
        seen_meaning_ids: set[str] = set()
        for field in ("required_meaning", "forbidden_meaning"):
            meanings = expected.get(field)
            if not isinstance(meanings, list) or not meanings:
                errors.append(_record_error(record, f"{prefix}: {field} must be a non-empty list"))
                continue
            for meaning in meanings:
                if not isinstance(meaning, dict):
                    errors.append(_record_error(record, f"{prefix}: {field} item must be an object"))
                    continue
                meaning_id = meaning.get("id")
                description = meaning.get("description")
                if not isinstance(meaning_id, str) or not meaning_id.strip():
                    errors.append(_record_error(record, f"{prefix}: {field} id must be a non-empty string"))
                elif meaning_id.strip() in seen_meaning_ids:
                    errors.append(_record_error(record, f"{prefix}: duplicate meaning id: {meaning_id}"))
                else:
                    seen_meaning_ids.add(meaning_id.strip())
                if not isinstance(description, str) or not description.strip():
                    errors.append(_record_error(record, f"{prefix}: {field} description must be a non-empty string"))
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
    errors.extend(_validate_source_events(record, source_files))
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


def _validate_source_events(record: dict[str, Any], source_files: set[str]) -> list[str]:
    """校验非消息事件和引用，例如撤回的消息。"""

    errors: list[str] = []
    source_events = record.get("source_events")
    event_items = source_events if isinstance(source_events, list) else []
    event_ids: set[str] = set()
    for event in event_items:
        if not isinstance(event, dict):
            errors.append(_record_error(record, "source event must be an object"))
            continue
        event_id = str(event.get("id") or "")
        if not event_id:
            errors.append(_record_error(record, "source event id is required"))
        elif event_id in event_ids:
            errors.append(_record_error(record, f"duplicate source event id: {event_id}"))
        event_ids.add(event_id)
        if not str(event.get("event_type") or "").strip():
            errors.append(_record_error(record, f"source event {event_id}: event_type is required"))
        if event.get("actor") not in {"customer", "boss", "system"}:
            errors.append(_record_error(record, f"source event {event_id}: invalid actor"))
        if not isinstance(event.get("content_available"), bool):
            errors.append(_record_error(record, f"source event {event_id}: content_available must be boolean"))
        source_image = str(event.get("source_image") or "")
        if source_files and source_image not in source_files:
            errors.append(_record_error(record, f"source event {event_id}: source_image not listed"))

    for fact in record.get("business_facts") or []:
        if not isinstance(fact, dict):
            continue
        for event_id in fact.get("source_event_ids") or []:
            if event_id not in event_ids:
                errors.append(
                    _record_error(record, f"business fact {fact.get('id')}: source_event_id not found: {event_id}")
                )
    for case in record.get("eval_cases") or []:
        if not isinstance(case, dict):
            continue
        for event_id in case.get("source_event_refs") or []:
            if event_id not in event_ids:
                errors.append(
                    _record_error(record, f"eval case {case.get('id')}: source_event_ref not found: {event_id}")
                )
    return errors


def _validate_timeline_extensions(
    record: dict[str, Any],
    messages: list[dict[str, Any]],
    expected_turns: list[int],
) -> list[str]:
    """校验多天转录文本的可选剧集和回放元数据。"""

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
        # 合成回归可验证语义，但不能充当真人老板风格的原始话术证据。
        if record.get("kind") == "owner_chat_live_semantic_regression":
            continue
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
        help="Validate one JSONL dataset. By default validates the base JSONL and every multi-day timeline JSON.",
    )
    args = parser.parse_args(argv)

    records = read_jsonl(args.dataset) if args.dataset else read_default_records()
    errors = validate_dataset(records)
    payload = {
        "dataset": (
            str(args.dataset)
            if args.dataset
            else [str(DEFAULT_DATASET_PATH), *(str(path) for path in DEFAULT_TIMELINE_PATHS)]
        ),
        "record_count": len(records),
        "error_count": len(errors),
        "errors": errors,
        "status": "passed" if not errors else "failed",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
