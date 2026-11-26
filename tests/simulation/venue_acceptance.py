"""用归档交付、业务结果和已知事实审阅仿真，不把同群发言或组局率当成对话质量。"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from mahjong_agent_runtime.customer_visible_contract import CUSTOMER_EXECUTION_FAILURE_REPLY


Row = Mapping[str, Any]
AGENT_ID = "simulation-agent"
VALID_NON_FORMATION = {"waiting", "no_match", "declined", "cancelled", "expired_unfilled"}
OPTION_RESULTS = {"options_found", "match_offered"}
GAME_MUTATIONS = {"create_game", "join_game", "record_candidate_reply", "update_game_status"}
PENDING_HANDLING = {"waiting_user": "clarification_requested", "needs_human": "human_review_requested"}


def evaluate_venue_acceptance(
    *, messages: Iterable[Row], turns: Iterable[Row], outcomes: Iterable[Row], constraints: Iterable[Row] = (),
    reviewed_shop_replies: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """按聊天账本核验因果轮次，并从调用、持久状态和逐轮语义判断验收业务结果。"""
    rows = {row["message_id"]: row for row in messages}
    positions = {message_id: index for index, message_id in enumerate(rows)}
    turn_rows = list(turns)
    turn_by_input = {row["message_id"]: row for row in turn_rows if row.get("message_id")}
    event_inputs = {row["event_id"]: row["message_id"] for row in turn_rows if row.get("event_id") and row.get("message_id")}
    replies = _delivered_replies(rows, positions)
    conversations, concurrency = _dialogues(rows, positions, replies, turn_by_input, event_inputs)
    result_rows = [_check_outcome(row, rows, turn_by_input, replies, reviewed_shop_replies or {}) for row in outcomes]
    constraint_rows = [_check_constraints(row, rows, positions, replies) for row in constraints]
    technical = [
        {"customer_id": row.get("actor_id"), "message_id": row.get("message_id"), "trace_id": row.get("trace_id"), "kind": fault}
        for row in turn_rows if (fault := _fault_kind(row)) is not None
    ]
    counts = Counter(row["kind"] for row in result_rows if row["status"] == "passed")
    findings = result_rows + constraint_rows
    if any(row["status"] == "failed" for row in findings):
        status = "failed"
    elif technical or not result_rows or any(row["status"] in {"inconclusive", "review_required"} for row in findings):
        status = "inconclusive"
    else:
        status = "passed"
    return {
        "status": status,
        "conversations": conversations,
        "concurrent_private_conversations": concurrency,
        "multi_round_private_customers": sum(row["longest_private_chain"] >= 2 for row in conversations.values()),
        "outcomes": result_rows,
        "outcome_counts": dict(counts),
        "pending_outcome_counts": dict(Counter(row["kind"] for row in result_rows if row["status"] == "pending")),
        "valid_non_formation_count": sum(counts[kind] for kind in VALID_NON_FORMATION),
        "technical_failures": technical,
        "turn_constraints": constraint_rows,
    }


def _delivered_replies(rows: dict[str, Row], positions: dict[str, int]) -> dict[str, list[Row]]:
    """只配对确实落入正确目标账本的 Agent 回复；群公告和会员互聊不能成为私聊轮次。"""
    replies: dict[str, list[Row]] = defaultdict(list)
    for row in rows.values():
        source_id = row.get("source_message_id")
        source = rows.get(source_id)
        if row.get("direction") != "outbound" or row.get("sender_id") != AGENT_ID or not source:
            continue
        if source.get("direction") != "inbound" or positions[source_id] >= positions[row["message_id"]]:
            continue
        same_conversation = row.get("conversation_id") == source.get("conversation_id")
        if row.get("kind") == "private" and row.get("recipient_id") == source.get("sender_id"):
            if source.get("kind") == "group" or same_conversation:
                replies[source_id].append(row)
        elif row.get("kind") == source.get("kind") == "group" and same_conversation:
            if row.get("recipient_id") == source.get("recipient_id"):
                replies[source_id].append(row)
    return replies


def _prior_private_input(
    row: Row, turn: Row, rows: dict[str, Row], positions: dict[str, int], replies: dict[str, list[Row]],
    event_inputs: dict[str, str], private_inputs: dict[str, tuple[str, int]],
) -> str | None:
    """续轮必须实际引用或由本人的上轮已交付私聊回复触发。"""
    cause = turn.get("caused_by") or row.get("quoted_message_id")
    if cause in event_inputs:
        candidates = replies.get(event_inputs[cause], ())
    else:
        candidates = (rows[cause],) if cause in rows else ()
    for previous in reversed(candidates):
        prior_input = previous.get("source_message_id")
        if (prior_input in private_inputs and previous in replies.get(prior_input, ())
                and previous.get("kind") == "private" and previous.get("recipient_id") == row.get("sender_id")
                and previous.get("conversation_id") == row.get("conversation_id")
                and positions[previous["message_id"]] < positions[row["message_id"]]):
            return prior_input
    return None


def _dialogues(rows, positions, replies, turns, event_inputs) -> tuple[dict[str, dict[str, int]], int]:
    """报告每人实际答复轮数，并量化不同私聊在归档时间线上的重叠。"""
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    private_inputs: dict[str, tuple[str, int]] = {}
    spans: dict[str, tuple[str, int, int]] = {}
    for message_id, row in rows.items():
        if row.get("direction") != "inbound" or not row.get("sender_id"):
            continue
        customer_id = row["sender_id"]
        counts = counters[customer_id]
        outgoing = replies.get(message_id, ())
        if row.get("kind") == "group":
            counts["group_messages"] += 1
            counts["group_rounds"] += bool(outgoing)
            counts["group_to_private_rounds"] += any(reply.get("kind") == "private" for reply in outgoing)
            continue
        if row.get("recipient_id") != AGENT_ID:
            continue
        counts["private_messages"] += 1
        if not outgoing:
            continue
        counts["private_rounds"] += 1
        prior = _prior_private_input(row, turns.get(message_id, {}), rows, positions, replies, event_inputs, private_inputs)
        if prior is None:
            root, depth = message_id, 1
        else:
            root, prior_depth = private_inputs[prior]
            depth = prior_depth + 1
            counts["causal_private_followups"] += 1
            end = max(positions[reply["message_id"]] for reply in outgoing)
            previous_end = spans.get(root, (customer_id, positions[root], end))[2]
            spans[root] = (customer_id, positions[root], max(previous_end, end))
        private_inputs[message_id] = (root, depth)
        counts["longest_private_chain"] = max(counts["longest_private_chain"], depth)

    points = sorted(point for customer_id, start, end in spans.values()
                    for point in ((start, 1, customer_id), (end, -1, customer_id)))
    active: Counter[str] = Counter()
    peak = 0
    for _, direction, customer_id in points:
        active[customer_id] += direction
        if active[customer_id] == 0:
            del active[customer_id]
        peak = max(peak, len(active))
    fields = ("group_messages", "group_rounds", "group_to_private_rounds", "private_messages", "private_rounds",
              "causal_private_followups", "longest_private_chain")
    return {key: {field: count[field] for field in fields} for key, count in counters.items()}, peak


def _fault_kind(turn: Row) -> str | None:
    """保留技术失败类别；未收到回复、等客人、隐私静默均不自动算供应商异常。"""
    status = str(turn.get("runtime_status") or "").casefold()
    error = str(turn.get("error_type") or "").casefold()
    if any(marker in status or marker in error for marker in ("timeout", "timedout", "deadlineexceeded")):
        return "timeout"
    if error or status in {"failed", "model_error", "transport_error"}:
        return "model_fault"
    return None


def assess_pending_handling(delivered: Sequence[Row], reviewed_shop_replies: Mapping[str, str]) -> dict[str, Any]:
    """分别核验待处理回复是否实际送达、内容有实质、处置经过独立判定。"""
    if not delivered or not str(delivered[-1].get("text") or "").strip():
        return {"status": "inconclusive", "substantive_delivery": False, "reason": "缺少该轮顾客实际收到的待处理答复"}
    final = delivered[-1]
    reply_id = final["message_id"]
    if final["text"].strip() == CUSTOMER_EXECUTION_FAILURE_REPLY:
        return {"delivered_reply_id": reply_id, "status": "failed", "substantive_delivery": False,
                "reason": "仅送达执行退避提示，缺少有实质内容的后续处理"}
    evidence = {"delivered_reply_id": reply_id, "substantive_delivery": True}
    verdict = reviewed_shop_replies.get(reply_id)
    if verdict == "pass":
        return {**evidence, "status": "pending", "reason": "独立评审核实本轮处理适当，业务待继续"}
    if verdict == "fail":
        return {**evidence, "status": "failed", "reason": "独立评审判定本轮待处理答复不适当"}
    return {**evidence, "status": "review_required", "reason": "已送达待处理答复，仍需独立核对处理是否适当"}


def _check_outcome(outcome: Row, rows, turns, replies, reviewed_shop_replies: Mapping[str, str]) -> dict[str, Any]:
    """校验单个客户、同一业务请求的终态；合法的不成局有独立的业务依据。"""
    customer_id, kind = outcome.get("customer_id"), outcome.get("kind")
    source_ids = list(dict.fromkeys(outcome.get("source_message_ids") or ()))
    finding = {"customer_id": customer_id, "kind": kind, "source_message_ids": source_ids}
    if not source_ids or not all(rows.get(key, {}).get("direction") == "inbound" and
                                 rows[key].get("sender_id") == customer_id for key in source_ids):
        return {**finding, "status": "inconclusive", "reason": "缺少归属该客户的入站消息"}
    turn_rows = [turns.get(key, {}) for key in source_ids]
    faults = {_fault_kind(row) for row in turn_rows}
    finding["trace_ids"] = list(dict.fromkeys(row["trace_id"] for row in turn_rows if row.get("trace_id")))
    if kind in {"timeout", "model_fault"}:
        return {**finding, "status": "inconclusive", "reason": "有技术失败证据" if kind in faults else "缺少对应故障证据"}
    if faults - {None}:
        return {**finding, "status": "inconclusive", "reason": "该客户本次业务有技术失败"}
    if kind in PENDING_HANDLING.values():
        if PENDING_HANDLING.get(turn_rows[-1].get("runtime_status")) != kind:
            return {**finding, "status": "inconclusive", "reason": "最后一轮运行状态不支持该待处理类型"}
        return finding | assess_pending_handling(replies.get(source_ids[-1], ()), reviewed_shop_replies)
    if kind not in {"formed", "completed", *VALID_NON_FORMATION, *OPTION_RESULTS}:
        return {**finding, "status": "inconclusive", "reason": "未提供可核验的业务结果"}

    tools = [tool for turn in turn_rows for tool in turn.get("tool_results", ())
             if tool.get("called") is True and tool.get("allowed") is True and not tool.get("error")]
    game = outcome.get("game") or {}
    demand = outcome.get("waiting_demand") or {}
    participants = {member["customer_id"]: member["status"] for member in game.get("participants", ())}
    relevant_game = game.get("game_id") and (game.get("organizer_id") == customer_id or customer_id in participants)
    game_updates = [tool for tool in tools if tool.get("name") in GAME_MUTATIONS and
                    (tool.get("result", {}).get("game") or {}).get("game_id") == game.get("game_id")]
    # 原子群认领走正式领域 Store，也能提交转移，未必产生模型工具调用。
    game_linked = bool(game_updates) or any(
        step.get("entity_type") == "game_participant" and step.get("entity_id") == f"{game.get('game_id')}:{customer_id}"
        for turn in turn_rows for step in turn.get("state_transitions", ())
    )
    delivered = any(replies.get(key) for key in source_ids)
    if kind == "completed":
        verified = relevant_game and game_linked and _played_to_completion(
            game, outcome.get("reservation") or {}, customer_id, outcome.get("duration_hours"),
        )
    elif kind == "formed":
        verified = (relevant_game and game_linked and participants.get(customer_id) in {"joined", "confirmed"}
                    and game.get("status") in {"ready", "finished"} and game.get("remaining_seats") == 0)
    elif kind == "waiting":
        verified = (demand.get("sender_id") == customer_id and demand.get("status") == "active" and any(
            tool.get("name") == "register_waiting_demand" and
            (tool.get("result", {}).get("waiting_demand") or {}).get("demand_id") == demand.get("demand_id")
            for tool in tools)) or (relevant_game and game.get("status") in {"forming", "inviting"} and bool(game_updates))
        if (not verified and delivered and outcome.get("clarification_confirmed") is True
                and turn_rows[-1].get("runtime_status") == "waiting_user"):
            verified = True
    elif kind == "no_match" or kind in OPTION_RESULTS:
        searches = [tool["result"] for tool in tools if tool.get("name") == "search_current_games" and tool.get("result")]
        verified = delivered and _matches_claimed_search_result(
            kind, searches[-1] if searches else {}, outcome.get("game_id"),
        ) and not any(tool.get("name") in {"create_game", "join_game"} for tool in tools)
    elif kind == "declined":
        verified = participants.get(customer_id) in {"declined", "rejected"} and bool(game_updates)
        if outcome.get("explicit_decline") is True and participants.get(customer_id) not in {"joined", "confirmed"}:
            verified = True
    elif kind == "expired_unfilled":
        verified = relevant_game and game.get("status") == "cancelled" and game.get("closed_reason") == "expired_without_full_table"
    else:
        verified = relevant_game and game.get("status") == "cancelled" and bool(game_updates)
        if (demand.get("sender_id") == customer_id and demand.get("status") == "cancelled" and any(
                tool.get("name") == "cancel_waiting_demand" and any(
                    item.get("demand_id") == demand.get("demand_id") for item in tool.get("result", {}).get("cancelled_demands", ())
                ) for tool in tools)):
            verified = True
    return {**finding, "status": "passed" if verified else "failed",
            "reason": "已核对归档业务证据" if verified else "声明的业务结果缺少相符的可见或持久证据"}


def _matches_claimed_search_result(kind: str, result: Row, game_id: str | None) -> bool:
    """查到备选、未查到和真实入局各自有证据；某个具体局的提议需在查询结果中。"""
    matches = result.get("matches")
    alternatives = result.get("alternatives")
    if not isinstance(matches, list) or not isinstance(alternatives, list):
        return False
    if kind == "no_match":
        return not (matches or alternatives)
    options = [item.get("game") or item for item in (*matches, *alternatives) if isinstance(item, Mapping)]
    return bool(options) and (game_id is None or any(item.get("game_id") == game_id for item in options))


def _played_to_completion(game: Row, reservation: Row, customer_id: str, duration_hours: float | None) -> bool:
    """用实际到场、开打、散场和释放记录验证完成，预计时间和满座都不替代实际状态。"""
    roster = set(game.get("actual_attendee_ids", ()))
    if (game.get("status") != "finished" or customer_id not in roster or len(roster) != game.get("seats_total", 4)
            or reservation.get("status") != "released" or reservation.get("game_id") != game.get("game_id")
            or not game.get("play_reservation_id") or reservation.get("reservation_id") != game["play_reservation_id"]):
        return False
    try:
        start = datetime.fromisoformat(game.get("actual_started_at", ""))
        end = datetime.fromisoformat(game.get("actual_ended_at", ""))
        requested = duration_hours if duration_hours is not None else (game.get("requirement") or {}).get("duration_hours")
        return end > start and (requested is None or end - start == timedelta(hours=float(requested)))
    except (ValueError, TypeError, OverflowError):
        return False


def _check_constraints(check: Row, rows, positions, replies) -> dict[str, Any]:
    """复核独立判定或场景提供的结构化字段判断，文字措辞不作机械匹配。"""
    message_id = check.get("message_id")
    finding = {"message_id": message_id, "status": "inconclusive", "violations": [], "known_source_message_ids": {}}
    target = rows.get(message_id)
    if not target or not replies.get(message_id):
        return finding
    if not any(name in check for name in ("reply_facts", "asked_fields", "tool_facts")):
        return finding
    known = {}
    for field, assertion in check.get("known_facts", {}).items():
        source_id = assertion.get("source_message_id")
        source = rows.get(source_id)
        if (not source or positions[source_id] > positions[message_id] or source.get("direction") != "inbound"
                or source.get("sender_id") != target.get("sender_id")):
            return finding
        known[field] = assertion["value"]
        finding["known_source_message_ids"][field] = source_id
    violations = finding["violations"]
    reconfirm = set(check.get("allowed_reconfirm_fields", ()))
    for field in set(check.get("asked_fields", ())).intersection(known) - reconfirm:
        violations.append(f"reasked:{field}")
    for check_kind in ("reply_facts", "tool_facts"):
        for field, value in check.get(check_kind, {}).items():
            if field in known and value != known[field]:
                violations.append(f"conflicting_{check_kind}:{field}")
    finding["violations"] = sorted(violations)
    finding["status"] = "failed" if violations else "passed"
    return finding
