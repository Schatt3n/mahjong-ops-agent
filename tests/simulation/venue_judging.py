"""从真实交付记录构造有明确引用范围的精简营业多轮评测。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any

from .venue_acceptance import _delivered_replies
from .venue_actors import SemanticJudge


Row = Mapping[str, Any]
ROLE_CHECK = "顾客动作符合其画像、意图和实际看到的信息"
AGENT_CHECK = "Agent 依据已送达对话承接需求，不重复追问已知条件且不误报或无依据承诺"
CHECKS = (ROLE_CHECK, AGENT_CHECK)
_PROFILE_FIELDS = ("preferred_game", "stake_min", "stake_max", "smoking", "available_periods", "speaking_style", "temperament")
_TERM_FIELDS = ("game_type", "stake", "base_stake", "start_time", "planned_start_at", "planned_end_at", "duration_hours",
                "smoke", "smoking", "smoking_policy", "party_size", "case_goal", "case_intent")
_GAME_FIELDS = ("game_id", "status", "planned_start_at", "planned_end_at", "remaining_seats", "seats_total")
_STATE_FIELDS = ("status", "game_id", "reservation_id", "demand_id", "start_at", "end_at", "expires_at")
_WRITING_TOOLS = {"customer_visible_content_review", "customer_visible_text_generation"}
_CITATION_RULES = """每个判定只引用下列证据。顾客项引用 customer_message（顾客等待或离开时可引用 customer_action），可补充本人的 customer_action、当时实际看到的 agent_reply。画像是倾向，允许自然变动。店家项必须引用实际送达的 agent_reply，并按需补充 customer_message 与 agent_execution。店家永远不知道 customer_action 中的私有设定。无送达时不能判店家通过，可引用入站、执行错误指出失败或证据不足。最终状态只描述会话结束时，不能倒推前一轮。没有适配牌局时，如实告知或在同意后候补也可通过。"""


@dataclass(slots=True)
class _Evidence:
    """区分可引用的上下文与每项判定必须命中的评价对象。"""

    rows: list[dict[str, Any]]
    allowed: dict[str, set[str]]
    anchors: dict[str, set[str]]

    def add(self, key: str, identifier: str, fields: dict[str, Any], *, context=(), targets=()) -> None:
        """证据 ID 直接关联归档消息或 trace，分别限制两个检查项的引用。"""
        self.rows.append({key: identifier, **fields})
        for check in (*context, *targets):
            self.allowed[check].add(identifier)
        for check in targets:
            self.anchors[check].add(identifier)


def _pick(row: Row, fields: Sequence[str]) -> dict[str, Any]:
    return {field: row[field] for field in fields if field in row and row[field] is not None}


def _byte_count(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _game(row: Row) -> dict[str, Any]:
    game = row.get("game") or row
    terms = _pick(game.get("requirement") or {}, _TERM_FIELDS)
    return _pick(game, _GAME_FIELDS) | ({"terms": terms} if terms else {}) | _pick(row, ("match_kind",))


def _executions(turn: Row) -> list[dict[str, Any]]:
    """保留业务动作与关键返回值，原始客户列表、话术模型和供应商诊断留在本地归档。"""
    summaries = []
    for tool in turn.get("tool_results") or ():
        name = tool.get("name")
        if not tool.get("called") or not name or name in _WRITING_TOOLS:
            continue
        allowed = tool.get("allowed") is True
        error = tool.get("error")
        summary: dict[str, Any] = {"name": name, "execution": "ok" if allowed and not error else "error" if allowed else "blocked"}
        if summary["execution"] != "ok":
            summaries.append(summary)
            continue
        result = tool.get("result") or {}
        returned = _pick(result, ("configured", "start_at", "end_at", "available_count", "cancelled_count"))
        if result.get("requirement"):
            returned["query"] = _pick(result["requirement"], _TERM_FIELDS)
        for label in ("matches", "alternatives"):
            if isinstance(result.get(label), list):
                options = result[label]
                returned[label] = {"count": len(options), "options": [_game(item) for item in options if isinstance(item, Mapping)]}
        if isinstance(result.get("game"), Mapping):
            returned["game"] = _game(result["game"])
        for label in ("reservation", "waiting_demand"):
            if isinstance(result.get(label), Mapping):
                returned[label] = _pick(result[label], _STATE_FIELDS)
        if returned:
            summary["result"] = returned
        summaries.append(summary)
    return summaries


def _role(turn: Row, *, include_profile: bool, visible_text: Sequence[str]) -> dict[str, Any]:
    """只提取顾客本人的本轮动机、动作及与已见人物相关的偏好。"""
    decision = turn.get("decision") or {}
    context = decision.get("context") or {}
    actions = decision.get("action") or {}
    events = context.get("events") or ()
    role = _pick(decision, ("source", "status"))
    if actions.get("action"):
        role["action"] = actions["action"]
    if events and events[-1].get("details"):
        role["situation"] = _pick(events[-1]["details"], _TERM_FIELDS)
    profile = context.get("persona") or {}
    if include_profile and profile:
        role["private_profile"] = _pick(profile, _PROFILE_FIELDS)
    names = "\n".join(visible_text)
    relationships = [
        _pick(item, ("target_name", "kind")) for item in profile.get("relationships", ())
        if item.get("target_name") and item["target_name"] in names
    ]
    if relationships:
        role["private_relationships"] = relationships
    return role


def _project_dialogue(customer_id: str, turns: Sequence[Row], records: Iterable[Row], outcome: Row | None) -> _Evidence:
    """根据同一顾客的实际来话和投递建立最小可审计证据。"""
    evidence = _Evidence([], {check: set() for check in CHECKS}, {check: set() for check in CHECKS})
    ledger = {row["message_id"]: row for row in records}
    positions = {message_id: index for index, message_id in enumerate(ledger)}
    replies = _delivered_replies(ledger, positions)

    profile_added = False
    execution_rows = []
    own_reply_ids = set()
    for turn in turns:
        if turn.get("actor_id") != customer_id:
            continue
        message_id = turn.get("message_id")
        inbound = ledger.get(message_id, {})
        if inbound.get("direction") != "inbound" or inbound.get("sender_id") != customer_id:
            inbound = {}
        cause = turn.get("caused_by")
        safe_cause = cause if cause in own_reply_ids else None
        text = [str(ledger[safe_cause].get("text", ""))] if safe_cause else []
        if inbound:
            text.append(str(inbound.get("text", "")))
        role = _role(turn, include_profile=not profile_added, visible_text=text)
        profile_added = profile_added or "private_profile" in role
        valid_action = role.get("action") if role.get("status") == "valid" else None
        if role:
            decision = turn.get("decision") or {}
            role_trace = decision.get("trace_id") or turn.get("event_id")
            if role_trace:
                evidence.add("trace_id" if decision.get("trace_id") else "event_id", role_trace,
                             {"kind": "customer_action", "input": message_id if inbound else None,
                              "caused_by": safe_cause, **role},
                             targets=(ROLE_CHECK,) if valid_action in {"wait", "leave", "arrive"} else (), context=(ROLE_CHECK,))
        if not inbound:
            continue
        evidence.add("message_id", message_id, {"kind": "customer_message", **_pick(inbound, ("text", "sent_at"))},
                     targets=(ROLE_CHECK,) if valid_action == "send" else (), context=CHECKS)
        for reply in replies.get(message_id, ()):
            evidence.add("message_id", reply["message_id"], {"kind": "agent_reply", "answers": message_id,
                         **_pick(reply, ("text", "sent_at"))}, targets=(AGENT_CHECK,), context=(ROLE_CHECK,))
            own_reply_ids.add(reply["message_id"])
        if trace_id := turn.get("trace_id"):
            execution = {"kind": "agent_execution", "responds_to": message_id, "tools": _executions(turn),
                         **_pick(turn, ("runtime_status", "error_type", "failure_stage"))}
            if trace_id not in evidence.allowed[AGENT_CHECK]:
                evidence.add("trace_id", trace_id, execution, context=(AGENT_CHECK,))
                execution_rows.append((message_id, evidence.rows[-1]))

    if outcome:
        sources = outcome.get("source_message_ids") or ()
        final_execution = next((row for message_id, row in reversed(execution_rows) if message_id in sources), None)
        if final_execution is not None:
            final_execution["final_outcome"] = _pick(outcome, ("kind", "status"))
    return evidence


def review_dialogue(judge: SemanticJudge, *, customer_id: str, turns: Sequence[Row], records: Iterable[Row],
                    outcome: Row | None = None) -> dict[str, Any]:
    """用一次独立请求评审顾客动作和店家回复，校验每个结果的归档证据范围。"""
    projected = _project_dialogue(customer_id, turns, records, outcome)
    result = judge.judge(f"dialogue:{customer_id}", projected.rows, trace_id=f"judge:{customer_id}",
                         checks=CHECKS, instructions=_CITATION_RULES).to_dict()
    result["projection"] = {
        "version": 1, "rows": len(projected.rows), "evidence_bytes": _byte_count(projected.rows),
        "source_turn_bytes": _byte_count(turns),
    }
    if result["status"] != "valid":
        return result
    for finding in result["findings"]:
        check, ids, verdict = finding["check"], set(finding["evidence_ids"]), finding["verdict"]
        # Agent 的通过必须指向实际回复。店家未交付或无回复时可以判失败，引用实际执行或入站证据。
        need_target = verdict == "pass" or verdict == "fail" and (check != AGENT_CHECK or projected.anchors[check])
        if not ids <= projected.allowed[check] or need_target and not ids.intersection(projected.anchors[check]):
            result.update(status="invalid", verdict="inconclusive", findings=(), error="judgment does not cite its review subject")
            break
    return result
