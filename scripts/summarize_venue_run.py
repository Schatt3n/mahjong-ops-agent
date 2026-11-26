#!/usr/bin/env python3
"""从沙盒报告生成可提交仓库的最小证据，避免携带个人或请求原文。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable
from contextlib import closing
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from urllib.parse import quote


def summary(report: dict) -> dict:
    """只保留计数、配置、检查和结论，不复制原始消息、完整请求或角色身份。"""
    evidence = {name: report.get(name) for name in (
        "mode", "scenario", "scenario_date", "seed", "status", "people", "targeted_users",
        "workers", "max_turns", "real_customer_actions", "peak_inflight_decisions", "peak_inflight_deliveries", "chat_events",
        "sandbox_consent_prompt_sha256", "elapsed_seconds",
    )}
    if report.get("show_verified_write_scope"):
        evidence["show_verified_write_scope"] = True
    if report.get("sandbox_reply_consent_prompt_sha256"):
        evidence["sandbox_reply_consent_prompt_sha256"] = report["sandbox_reply_consent_prompt_sha256"]
    if runtime_limits := report.get("runtime_limits"):
        evidence["runtime_limits"] = runtime_limits
    acceptance = report.get("acceptance") or {}
    evidence["checks"] = report.get("checks", {})
    if contacts := report.get("initial_contacts"):
        evidence["initial_contacts"] = contacts
    if initiation := report.get("private_initiation"):
        evidence["private_initiation"] = initiation
    if link := report.get("friend_group_link"):
        evidence["friend_group_link"] = link
    if historical := report.get("historical_continuation"):
        evidence["historical_continuation"] = historical
    if rolling := report.get("rolling_roles"):
        evidence["rolling_roles"] = rolling
        evidence["rolling_failure_counts"] = {
            "events": len(report.get("failed_events") or ()),
            "transport": len(report.get("model_transport_errors") or ()),
        }
    if board := report.get("rolling_board"):
        evidence["rolling_board"] = {key: value for key, value in board.items() if key != "links"}
    if history := report.get("rolling_history"):
        evidence["rolling_history"] = history
    if play := report.get("rolling_play"):
        evidence["rolling_play"] = {key: value for key, value in play.items() if key not in {"evidence", "participation_evidence"}}
    if peers := report.get("rolling_peers"):
        evidence["rolling_peers"] = peers
    if table_groups := report.get("rolling_table_groups"):
        evidence["rolling_table_groups"] = {key: value for key, value in table_groups.items()
                                             if key != "roster_reconciliations"}
        if reconciliations := table_groups.get("roster_reconciliations"):
            evidence["rolling_table_groups"]["roster_reconciliation_count"] = len(reconciliations)
    if recovery := report.get("role_transport_recovery"):
        evidence["role_transport_recovery"] = recovery
    if profile_sources := report.get("role_seed_profiles"):
        evidence["role_seed_profiles"] = profile_sources
    if profile_evidence := report.get("profile_evidence"):
        evidence["profile_evidence"] = profile_evidence
    evidence["causal_multi_round_customers"] = acceptance.get("multi_round_private_customers")
    evidence["outcome_counts"] = acceptance.get("outcome_counts", {})
    evidence["technical_failure_counts"] = dict(Counter(row["kind"] for row in acceptance.get("technical_failures", ())))
    judge = report.get("independent_judge")
    evidence["independent_judge"] = {key: judge.get(key) for key in ("status", "verdict")} if judge else None
    if authorization := report.get("write_authorization"):
        evidence["write_authorization"] = authorization
    if coverage := report.get("write_authorization_coverage"):
        evidence["write_authorization_coverage"] = coverage
    evidence["judge_samples"] = [
        {
            "status": row["status"], "verdict": row["verdict"],
            "checks": {finding["check"]: finding["verdict"] for finding in row.get("findings", ())},
            "evidence_bytes": (row.get("projection") or {}).get("evidence_bytes"),
            "elapsed_ms": (row.get("diagnostics") or {}).get("total_elapsed_ms"),
        }
        for row in report.get("semantic_samples", ())
    ]
    evidence["models"] = {
        stage: {key: config.get(key) for key in (
            "model", "provider", "protocol", "max_output_tokens", "max_concurrency", "reasoning_effort_on_wire",
        )}
        for stage, config in report.get("model_execution", {}).items()
    }
    if cohort := report.get("cohort"):
        initial, final = cohort.get("initial_game") or {}, cohort.get("final_game") or {}
        evidence["cohort"] = {
            "expected_outcome": cohort.get("expected_outcome"),
            "expected_counts": dict(Counter(cohort.get("expected_outcomes", {}).values())),
            "authorized_transitions": cohort.get("authorized_transitions", {}),
            "initial_remaining_seats": initial.get("remaining_seats"),
            "final_remaining_seats": final.get("remaining_seats"),
            "final_status": final.get("status"),
        }
    return evidence


def actor_observations(events: Iterable[dict]) -> dict:
    """计数真实角色选择、首句差异和格式修正，摘要不携带原话或身份。"""
    first_messages: dict[str, str] = {}
    actual_days = defaultdict(set)
    actions: Counter[str] = Counter()
    schema_repairs = 0
    for row in events:
        decision = row.get("decision") or {}
        if decision.get("source") != "model":
            continue
        schema_repairs += bool(decision.get("prior_attempts"))
        if decision.get("status") != "valid" or (decision.get("response_source") or {}).get("http_response_complete") is not True:
            continue
        if row.get("actor_id") and row.get("virtual_at"):
            actual_days[row["actor_id"]].add(datetime.fromisoformat(row["virtual_at"]).date())
        action = decision.get("action") or {}
        kind = action.get("action")
        if kind in {"send", "wait", "arrive", "leave", "invite_companion", "accept_companion", "decline_companion"}:
            actions[kind] += 1
        if kind == "send" and row.get("message_id") and row.get("actor_id"):
            first_messages.setdefault(row["actor_id"], " ".join(action["text"].split()))
    return {
        "valid_actions": dict(actions), "schema_repairs_attempted": schema_repairs,
        "customers_with_delivered_input": len(first_messages),
        "distinct_first_message_texts": len(set(first_messages.values())),
        "real_decision_actors_with_two_calendar_days": sum(len(days) >= 2 for days in actual_days.values()),
    }


def audit_owner_friend_routes(report: dict, archive_path: Path, events: Iterable[dict]) -> dict:
    """独立核对两轮好友私聊的由来、路由与实际回执，只输出计数和断言。"""
    if report.get("scenario") != "dialogues" or report.get("max_turns") != 2:
        raise ValueError("好友私聊审计仅用于无群发的两轮 dialogues 场景")
    if not report.get("initial_contacts"):
        raise ValueError("运行报告没有好友启动证据")
    targets = report["targeted_users"]
    contacts = report["initial_contacts"]
    with closing(sqlite3.connect(archive_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        (roster,) = db.execute("SELECT room_id, members_json FROM venue_memberships").fetchall()
        room_id = roster["room_id"]
        members = set(json.loads(roster["members_json"]))
        friendship = db.execute(
            "SELECT friend_ids_json FROM venue_owner_friendships WHERE room_id = ?", (room_id,),
        ).fetchone()
        friends = json.loads(friendship["friend_ids_json"]) if friendship is not None else []
        messages = db.execute("""
            SELECT sequence, room_id, conversation_id, message_id, sender_id, recipient_id,
                   kind, direction, source_message_id, trace_id
            FROM venue_messages ORDER BY sequence
        """).fetchall()
    inbound = {row["message_id"]: row for row in messages if row["direction"] == "inbound"}
    replies = {row["source_message_id"]: row for row in messages if row["direction"] == "outbound"}
    agent_id = "simulation-agent"
    namespace = quote(room_id, safe="")
    matched = set()
    for message_id, row in inbound.items():
        reply = replies.get(message_id)
        owner = row["sender_id"]
        conversation = f"im:dm:{namespace}:{quote(owner, safe='')}"
        if (reply and owner in members and row["recipient_id"] == reply["sender_id"] == agent_id
                and reply["recipient_id"] == owner
                and row["conversation_id"] == reply["conversation_id"] == conversation
                and row["trace_id"] == reply["trace_id"] and row["sequence"] < reply["sequence"]):
            matched.add(message_id)
    turns = defaultdict(list)
    for event in events:
        turns[event.get("actor_id")].append(event)
    causal = 0
    for account, customer_turns in turns.items():
        if len(customer_turns) != 2:
            continue
        first, second = customer_turns
        first_id, second_id = first.get("message_id"), second.get("message_id")
        if (first_id in matched and second_id in matched
                and account == inbound[first_id]["sender_id"] == inbound[second_id]["sender_id"]
                and first.get("caused_by") is None
                and first.get("reply_message_id") == second.get("caused_by") == replies[first_id]["message_id"]
                and second.get("reply_message_id") == replies[second_id]["message_id"]
                and replies[first_id]["sequence"] < inbound[second_id]["sequence"]):
            causal += 1
    routes = {row["conversation_id"] for row in inbound.values()}
    checks = {
        "archive_friend_roster_matches_start": (
            set(friends) == members and len(friends) == len(members)
            == report["people"] == contacts["owner_friends"] == contacts["group_members"]
        ),
        "private_only_archive_matches_event_counts": (
            all(row["kind"] == "private" and row["room_id"] == room_id for row in messages)
            and len(messages) == report["chat_events"] == targets * 4
            and len(inbound) == len(replies) == targets * 2
        ),
        "one_route_per_active_friend": len(routes) == len(turns) == targets,
        "all_replies_returned_to_originating_friend": len(matched) == len(inbound) == targets * 2,
        "all_round_two_events_followed_own_first_reply": (
            causal == targets == report["acceptance"]["multi_round_private_customers"]
        ),
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "audited_private_routes": len(routes), "matched_private_replies": len(matched),
            "causal_two_round_customers": causal}


def audit_rolling_expansion(report: dict, target: int, *, archived_actor_days: int | None = None) -> dict:
    """用实际角色凭据、交付、营业状态和覆盖检查下一档扩容的先决条件。"""
    runtime = report.get("rolling_roles") or {}
    samples = report.get("semantic_samples") or ()
    channel_counts = runtime.get("sent_channels") or {}
    linked = runtime.get("cross_channel_replies") or {}
    wakes = runtime.get("wake_kinds") or {}
    checked = report.get("checks") or {}
    role_profiles = report.get("role_seed_profiles") or {}
    sample_target = min(target, 20)
    checks = {
        "real_history_grounded_rolling_run": report.get("mode") == "real" and report.get("scenario") == "rolling-roles"
            and role_profiles.get("derived_roles") == role_profiles.get("distinct_speech_profiles") == report.get("people") == 500,
        "all_expected_actors_decided_over_two_calendar_days": (
            report.get("targeted_users") == runtime.get("real_decision_actors") ==
            runtime.get("real_decision_actors_with_two_calendar_days", archived_actor_days) == target
        ),
        "group_and_private_channels_with_actual_wakes": all(channel_counts.get(kind, 0) > 0 for kind in ("group", "private"))
            and all(wakes.get(kind, 0) > 0 for kind in ("group_attention", "private_receipt")),
        "actual_cross_channel_response": any(linked.get(kind, 0) > 0 for kind in ("group_to_private", "private_to_group")),
        "across_day_private_context_and_weather": runtime.get("actors_with_prior_day_private_receipts_in_context", 0) > 0
            and all(wakes.get(kind, 0) > 0 for kind in ("rain", "traffic")),
        "delivery_isolated_and_persisted_seat_limits_hold": all(checked.get(key) is True for key in (
            "role_context_matches_visible_channel_history", "private_receipts_wake_only_their_recipient",
            "no_overfilled_table_in_store_or_tool_evidence",
        )),
        "semantic_review_reached_target": len({row.get("case_id") for row in samples if row.get("case_id")}) >= sample_target and all(
            row.get("status") == "valid" and row.get("verdict") == "pass" for row in samples
        ),
        "overall_main_and_actor_acceptance": report.get("status") == "passed" and not report.get("failed_events"),
    }
    return {"target_actual_roles": target, "target_semantic_reviews": sample_target,
            "checks": checks, "status": "passed" if all(checks.values()) else "failed"}


def main() -> int:
    """输出 JSON 摘要，原始报告留在运行目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--audit-owner-friend-routes", action="store_true",
                        help="审计无群发的两轮好友私聊账本与角色事件")
    parser.add_argument("--require-real-active", type=int, choices=(12, 50, 500), help="执行指定档位的真实营业扩容验收")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    evidence = summary(report)
    events = args.report.with_name("events.jsonl")
    if events.is_file():
        with events.open(encoding="utf-8") as handle:
            evidence["actor_observations"] = actor_observations(json.loads(line) for line in handle if line.strip())
    if args.audit_owner_friend_routes:
        try:
            with events.open(encoding="utf-8") as handle:
                audit = audit_owner_friend_routes(report, args.report.with_name("chat.sqlite3"),
                                                  (json.loads(line) for line in handle if line.strip()))
        except (OSError, ValueError, sqlite3.Error) as exc:
            parser.error(str(exc))
        evidence["owner_friend_route_audit"] = audit
    if args.require_real_active:
        evidence["rolling_expansion_acceptance"] = audit_rolling_expansion(report, args.require_real_active,
            archived_actor_days=(evidence.get("actor_observations") or {}).get("real_decision_actors_with_two_calendar_days"))
    print(json.dumps({"run": args.report.parent.name, **evidence}, ensure_ascii=False, indent=2))
    return int(any(evidence[key]["status"] != "passed" for key in (
        "owner_friend_route_audit", "rolling_expansion_acceptance"
    ) if key in evidence))


if __name__ == "__main__":
    raise SystemExit(main())
