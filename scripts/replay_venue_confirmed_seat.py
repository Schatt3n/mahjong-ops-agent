#!/usr/bin/env python3
"""用 OpenCode Go 重放已占座顾客的晚到协商和明确退出，验证真实私聊及持久状态。"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from functools import partial
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mahjong_agent_runtime import SQLiteAgentStore  # noqa: E402
from mahjong_agent_runtime.models import using_clock  # noqa: E402
from scripts.replay_venue_consent import (  # noqa: E402
    ReplaySource, _deliver_replay_message, _delivered_reply, _run_venue_replay, _snapshot, _successful_tools,
)
from tests.simulation.venue_actors import SemanticJudge  # noqa: E402
from tests.simulation.venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel  # noqa: E402
from tests.simulation.venue_scenario import VENUE  # noqa: E402

FIXTURE = ROOT / "eval" / "golden" / "venue_confirmed_seat_eta_replay.json"
CUSTOMER = "replay-guest"
OWNER_RECEIPT = "owner:replay-confirmed-seat"


def _source(fixture: dict, withdrawal: bool) -> ReplaySource:
    """从脱敏样本恢复同桌、同一到店协商时刻和独立退出对照。"""
    start = datetime.fromisoformat(fixture["scheduled_start"])
    sent_at = datetime.fromisoformat(fixture["reply_sent_at"])
    members = ("replay-host", "replay-member-a", "replay-member-b")
    identities = dict(zip((*members, CUSTOMER), ("牌友甲", "牌友乙", "牌友丙", "牌友丁")))
    expected = fixture["expected"]
    initial_game = {"organizer_id": members[0], "remaining_seats": 1,
                    "participants": [{"customer_id": item, "status": "confirmed", "seat_count": 1} for item in members],
                    "requirement": {"game_type": expected["game_type"], "stake": "1", "smoke_preference": "no_smoking",
                                    "start_time_kind": "scheduled", "start_time": expected["start_time"],
                                    "planned_start_at": start.isoformat(), "duration_kind": "fixed",
                                    "duration_hours": expected["duration_hours"]},
                    "created_at": (start - timedelta(hours=1)).isoformat()}
    case = "confirmed_customer_explicit_withdrawal" if withdrawal else fixture["case_id"]
    text_field = "explicit_withdrawal_control" if withdrawal else "customer_eta_message"
    turn = MessageEnvelope(f"customer:{case}", CUSTOMER, fixture[text_field], sent_at, "private", AGENT_ID)
    return ReplaySource(seed=73, identities=identities, venue=VENUE, initial_game=initial_game, turns=(turn,), source_kind=case)


def _judge_reply(channel: SandboxChannel, turn: MessageEnvelope, judge: SemanticJudge, *, case: str,
                 expected_meaning: str, persisted: dict, archive: Path) -> str:
    """判定器只看该顾客实际收到的对话及脱敏席位和时间状态。"""
    evidence = [{"message_id": "seat-and-time", **persisted}]
    reply_ids = set()
    for row in channel.events_for(turn.sender_id):
        message_id = row["message_id"]
        if message_id == OWNER_RECEIPT:
            speaker = "此前送达的店家私聊"
        elif message_id == turn.message_id:
            speaker = "客户"
        elif row.get("source_message_id") == turn.message_id and row.get("sender_id") == AGENT_ID:
            speaker = "实际送达的店家回复"
            reply_ids.add(message_id)
        else:
            continue
        evidence.append({"message_id": message_id, "speaker": speaker, "text": row["text"]})
    verdict = judge.judge(case, evidence, trace_id=f"judge:{case}", checks=("回复准确承接客户诉求和实际席位状态",),
                          instructions=f"以营业库事实核对答复。接受不同口吻。预期含义：{expected_meaning}")
    archive.write_text(json.dumps(verdict.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cited = {identifier for finding in verdict.findings for identifier in finding.evidence_ids}
    if verdict.status != "valid" or (verdict.verdict == "pass" and
            not ({turn.message_id, "seat-and-time"} <= cited and reply_ids & cited)):
        return "inconclusive"
    return verdict.verdict


def _execute(store: SQLiteAgentStore, channel: SandboxChannel, source: ReplaySource, game_id: str,
             events: Path, *, fixture: dict, withdrawal: bool, judge: SemanticJudge) -> dict:
    """通过真实店家私聊触发主 Agent，重开 SQLite 核对状态后打分。"""
    turn, = source.turns
    receipt_at = turn.sent_at - timedelta(minutes=1)
    with using_clock(lambda: receipt_at):
        store.join_game(game_id=game_id, customer_id=CUSTOMER, display_name=source.identities[CUSTOMER],
                        trace_id="replay-initial-confirmed-seat")
        receipt = channel.publish_owner(kind="private", recipient_id=CUSTOMER, text=fixture["delivered_shop_message"],
                                        sent_at=receipt_at, message_id=OWNER_RECEIPT)
        before = store.require_game(game_id)
        initial_games, initial_rooms = _snapshot(store)
    if receipt["status"] != "delivered" or before.remaining_seats() != 0 or before.status.value != "ready":
        raise ValueError("应从已送达的留位和已满四席的原桌开始")
    response = _deliver_replay_message(channel, events, source.source_kind, turn)
    reopened = SQLiteAgentStore(store.path)
    try:
        after = reopened.require_game(game_id)
        games, rooms = _snapshot(reopened)
    finally:
        reopened._connection.close()
    member, = (item for item in after.participants if item.customer_id == CUSTOMER)
    participation = [tool for tool in _successful_tools(response) if tool["name"] in {"join_game", "record_candidate_reply"}]
    expected = fixture["expected"]
    checks = {
        "same_game_and_room_inventory": games.keys() == initial_games.keys() and rooms == initial_rooms,
        "other_seats_kept": [item for item in before.participants if item.customer_id != CUSTOMER] ==
                            [item for item in after.participants if item.customer_id != CUSTOMER],
        "same_hours": before.planned_start_at == after.planned_start_at and before.planned_end_at == after.planned_end_at,
        "expected_seat_state": (
            (member.status, after.remaining_seats()) == ("declined", 1) and
            len(participation) == 1 and participation[0]["result"].get("recorded_status") == "declined" and
            participation[0]["result"].get("game", {}).get("game_id") == game_id
            if withdrawal else
            not participation and after.to_dict() == before.to_dict() and
            (member.status, after.remaining_seats()) == (expected["customer_status"], expected["remaining_seats"])
        ),
    }
    delivered = _delivered_reply(response, turn)
    persisted = {"status": after.status.value, "customer_status": member.status, "remaining_seats": after.remaining_seats(),
                 "planned_start_at": after.planned_start_at.isoformat(), "planned_end_at": after.planned_end_at.isoformat()}
    semantics = _judge_reply(channel, turn, judge, case=source.source_kind, persisted=persisted,
                             expected_meaning=("简短确认本桌退出，已实际释放本人席位。" if withdrawal else expected["reply_meaning"]),
                             archive=events.with_name("confirmed_seat_judgment.json")) if delivered else "inconclusive"
    return {"status": ("failed" if not all(checks.values()) or semantics == "fail" else
                       "passed" if semantics == "pass" else "inconclusive"),
            "checks": checks | {"reply_delivered": delivered, "independent_reply_semantics": semantics == "pass"},
            "reply_verdict": semantics, "runtime_status": response.get("status"), "final_reply": response.get("final_reply"),
            "trace_id": response.get("trace_id"), "persisted": persisted}


def main() -> int:
    """为每条输入打开独立 SQLite，保留旧失败并归档新执行。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runtime_data/venue_simulation/confirmed_seat_replays")
    parser.add_argument("--case", choices=("both", "eta", "withdrawal"), default="both")
    args = parser.parse_args()
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    results = []
    for withdrawal in {"both": (False, True), "eta": (False,), "withdrawal": (True,)}[args.case]:
        source = _source(fixture, withdrawal)
        results.append(_run_venue_replay(lambda: source, partial(_execute, fixture=fixture, withdrawal=withdrawal),
            args.runs_dir / f"{source.source_kind}-{uuid.uuid4().hex[:10]}", kind=source.source_kind,
            planner_low=False, clarify_consent=False, review_clarifications=True))
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    return int(not all(result["status"] == "passed" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
