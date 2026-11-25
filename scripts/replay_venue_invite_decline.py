#!/usr/bin/env python3
"""冻结工作日外地拒邀原话，单独验收已送达邀约的实际关闭与后续通知。"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mahjong_agent_runtime import SQLiteAgentStore  # noqa: E402
from mahjong_agent_runtime.models import InviteStatus, using_clock  # noqa: E402
from scripts.replay_venue_consent import (  # noqa: E402
    ReplaySource, _deliver_replay_message, _delivered_reply, _run_venue_replay, _snapshot, _successful_tools,
)
from tests.simulation.venue_actors import SemanticJudge  # noqa: E402
from tests.simulation.venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel  # noqa: E402
from tests.simulation.venue_scenario import VENUE  # noqa: E402

FIXTURE = ROOT / "eval" / "golden" / "venue_invite_decline_replay.json"
_CUSTOMER = "replay-customer"
_NEIGHBOR = "replay-neighbor"
_INVITE_ID = "owner:weekday-invite-decline"
_NEIGHBOR_INVITE_ID = "owner:weekday-invite-decline:neighbor"
_PARTICIPATION_TOOLS = {"join_game", "record_candidate_reply"}
_DECLINE_AND_NOTICE_CHECK = "店家准确回应本桌拒绝和未来通知，不承诺无实际安排的通知"


def _load_fixture() -> dict:
    """读取便携的脱敏原话及营业时间，不加载原运行的画像和凭证。"""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _source(fixture: dict, *, decline_only: bool = False) -> ReplaySource:
    """恢复原来话的时间、空席和档位，其他牌友只保留合成身份。"""
    start = datetime.fromisoformat(fixture["scheduled_start"])
    sent_at = datetime.fromisoformat(fixture["reply_sent_at"])
    members = ("replay-member0", "replay-member1", "replay-member2")
    identities = dict(zip(members, ("牌友甲", "牌友乙", "牌友丙")))
    identities.update({_CUSTOMER: "牌友丁", _NEIGHBOR: "牌友戊"})
    initial_game = {
        "organizer_id": members[0], "remaining_seats": 1,
        "participants": [
            {"customer_id": customer, "status": "confirmed", "seat_count": 1} for customer in members
        ],
        "requirement": {
            **{field: fixture[field] for field in ("game_type", "stake", "smoke_preference", "duration_hours")},
            "start_time_kind": "scheduled", "start_time": start.strftime("%H:%M"),
            "planned_start_at": start.isoformat(), "duration_kind": "fixed",
        },
        "created_at": (start - timedelta(hours=1)).isoformat(),
    }
    case = "weekday_invite_decline_only" if decline_only else fixture["case_id"]
    text = fixture["adjacent_reply"] if decline_only else fixture["customer_reply"]
    turn = MessageEnvelope(f"customer:{case}", _CUSTOMER, text,
                           sent_at, "private", AGENT_ID)
    return ReplaySource(fixture["seed"], identities, VENUE, initial_game, (turn,),
                        source_kind=case)


def _publish_invite(store: SQLiteAgentStore, channel: SandboxChannel, source: ReplaySource,
                    game_id: str, fixture: dict) -> str:
    """先经沙盒实际交付，再用 Store 接口标记原邀约已发，避免把待审草稿当成来话原因。"""
    turn, = source.turns
    invited_at = datetime.fromisoformat(fixture["invite_sent_at"])
    if invited_at.utcoffset() is None or invited_at >= turn.sent_at:
        raise ValueError("邀约必须早于本人的入站回话")
    invites = {
        _NEIGHBOR: (_NEIGHBOR_INVITE_ID, "杭州麻将还有个位置，你来吗？"),
        turn.sender_id: (_INVITE_ID, fixture["invite"]),
    }
    with using_clock(lambda: invited_at):
        drafts, _ = store.create_invite_drafts(
            game_id=game_id, invitations=[
                {"customer_id": customer, "display_name": source.identities[customer], "message_text": text}
                for customer, (_, text) in invites.items()
            ], trace_id=_INVITE_ID,
        )
        for draft in drafts:
            message_id, text = invites[draft.customer_id]
            delivery = channel.publish_owner(kind="private", recipient_id=draft.customer_id, text=text,
                                              sent_at=invited_at, message_id=message_id)
            if delivery["status"] != "delivered":
                raise ValueError("邀约未实际交付给对应顾客")
            store.update_invite_delivery_status(draft_id=draft.draft_id, status="sent", trace_id=message_id,
                                                reason="沙盒店员私聊已交付")
    return next(draft.draft_id for draft in drafts if draft.customer_id == turn.sender_id)


def _reply_semantics(channel: SandboxChannel, source: ReplaySource, judge: SemanticJudge, *,
                     final_invite_status: str, notices: list[dict], events_path: Path) -> str:
    """只给独立判定器投影该私聊实际收发、本次邀约状态，以及实际保存的后续安排。"""
    turn, = source.turns
    dialogue = [row for row in channel.events_for(turn.sender_id) if row.get("kind") == "private"]
    evidence = [{"message_id": "persisted", "invitation_status": final_invite_status, "future_arrangements": notices}]
    reply_ids = set()
    for row in dialogue:
        message_id = row["message_id"]
        if message_id == _INVITE_ID:
            speaker = "shop_invite"
        elif message_id == turn.message_id:
            speaker = "customer"
        elif row.get("source_message_id") == turn.message_id and row["sender_id"] == AGENT_ID:
            speaker = "shop_reply"
            reply_ids.add(message_id)
        else:
            continue
        evidence.append({"message_id": message_id, "speaker": speaker, "text": row["text"]})
    result = judge.judge(
        source.source_kind, evidence, trace_id=f"judge:{source.source_kind}", checks=(_DECLINE_AND_NOTICE_CHECK,),
        instructions=(
            "顾客若同时要求以后获知同档位牌局，按原话区分该要求与当前不参加。"
            "若顾客只拒绝本桌，无需主动添加后续通知。"
            "店家可简短确认本桌，并说明未来能安排什么或问实际缺失条件。礼貌说下次再约不是已订通知。"
            "已承诺以后符合即通知，须有真实且适用的后续安排；长期画像或候选偏好不算安排。"
            "核对实际送达的 shop_reply 与 persisted，引用顾客来话和相应回复证据。"
        ),
    )
    events_path.with_name("invite_decline_judgment.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    cited = {identifier for finding in result.findings for identifier in finding.evidence_ids}
    if result.status != "valid" or (
        result.verdict == "pass" and not ({turn.message_id, "persisted"} <= cited and reply_ids & cited)
    ):
        return "inconclusive"
    return result.verdict


def _execute_replay(store: SQLiteAgentStore, channel: SandboxChannel, source: ReplaySource, game_id: str,
                    events_path: Path, *, fixture: dict, judge: SemanticJudge | None = None,
                    decline_only: bool = False) -> dict:
    """在同一入站跟踪当前邀约和未来安排，并用重开的业务库校验作用域。"""
    turn, = source.turns
    draft_id = _publish_invite(store, channel, source, game_id, fixture)
    before = store.require_game(game_id)
    before_game_ids, before_reservations = _snapshot(store)
    before_invites = {key: draft.to_dict() for key, draft in store.invite_drafts.items() if key != draft_id}
    if store.invite_drafts[draft_id].status != InviteStatus.SENT:
        raise ValueError("拒邀前未找到已送达的本次邀约")
    response = _deliver_replay_message(channel, events_path, source.source_kind, turn)
    participation = [tool for tool in _successful_tools(response) if tool["name"] in _PARTICIPATION_TOOLS]
    transitions = participation[0].get("state_transitions", ()) if len(participation) == 1 else ()
    invite_transitions = [entry for entry in transitions
                          if entry.get("entity_type") == "invite_draft" and entry.get("entity_id") == draft_id]
    conversation_id = channel.private_conversation_id(turn.sender_id)
    reopened = SQLiteAgentStore(store.path)
    try:
        after = reopened.require_game(game_id)
        games, rooms = _snapshot(reopened)
        final_invite_status = reopened.invite_drafts[draft_id].status.value
        other_invites = {key: draft.to_dict() for key, draft in reopened.invite_drafts.items() if key != draft_id}
        notices = [
            {"kind": "waiting_demand", "terms": demand.demand, "expires_at": demand.expires_at.isoformat(),
             "matched_game_id": demand.matched_game_id, "status": demand.status.value}
            for demand in reopened.waiting_demands.values()
            if demand.sender_id == turn.sender_id and demand.conversation_id == conversation_id
        ] + [
            {"kind": task.task_type, "due_at": task.due_at.isoformat(), "expires_at": task.payload.get("expires_at"),
             "status": task.status.value, "goal": task.payload.get("goal"), "waiting_for": task.payload.get("waiting_for")}
            for task in reopened.scheduled_tasks.values()
            if task.subject_id == turn.sender_id and task.conversation_id == conversation_id
            and task.task_type == "conversation_follow_up"
        ]
    finally:
        reopened._connection.close()
    checks = {
        "decline_tool_recorded_on_original_invite": (
            len(participation) == 1 and participation[0]["name"] == "record_candidate_reply"
            and not participation[0].get("deduplicated")
            and participation[0]["result"].get("recorded_status") == "declined"
            and (participation[0]["result"].get("game") or {}).get("game_id") == game_id
            and len(invite_transitions) == 1 and invite_transitions[0].get("from_status") == "sent"
            and invite_transitions[0].get("to_status") == "declined"
            and invite_transitions[0].get("trace_id") == response["trace_id"]
        ),
        "decline_persisted": final_invite_status == InviteStatus.DECLINED,
        "table_and_seats_preserved": (
            after.participants == before.participants and after.requirement == before.requirement
            and after.remaining_seats() == before.remaining_seats() == 1
            and games.keys() == before_game_ids.keys()
        ),
        "unrelated_invites_and_rooms_preserved": (
            any(draft["status"] == "sent" for draft in before_invites.values())
            and other_invites == before_invites and rooms == before_reservations
            and not any(row["message_id"] == _NEIGHBOR_INVITE_ID for row in channel.events_for(turn.sender_id))
        ),
        "future_arrangements_do_not_reuse_declined_game": all(row.get("matched_game_id") != game_id for row in notices),
        "reply_delivered": _delivered_reply(response, turn),
    }
    if decline_only:
        checks["no_unrequested_future_arrangement"] = not notices
    semantic = None
    if judge is not None and checks["reply_delivered"]:
        semantic = _reply_semantics(channel, source, judge, final_invite_status=final_invite_status,
                                    notices=notices, events_path=events_path)
    hard_ok = all(checks.values())
    return {
        "status": "failed" if not hard_ok or semantic == "fail" else ("passed" if semantic == "pass" else "inconclusive"),
        "checks": checks | {"reply_semantics": semantic == "pass" if judge is not None else None},
        "reply_verdict": semantic, "future_arrangements": notices, "trace_id": response.get("trace_id"),
        "source": "authored adjacent control" if decline_only else fixture["source"],
    }


def main() -> int:
    """使用 OpenCode Go 主 Agent 与独立判定器回放一条固定的历史顾客消息。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runtime_data/venue_simulation/invite_decline_replays")
    parser.add_argument("--guard-writes", action="store_true")
    parser.add_argument("--case", choices=("archived", "decline-only"), default="archived")
    args = parser.parse_args()
    fixture = _load_fixture()
    decline_only = args.case == "decline-only"
    source = _source(fixture, decline_only=decline_only)
    run_dir = args.runs_dir / f"replay-{uuid.uuid4().hex[:10]}"
    result = _run_venue_replay(
        lambda: source,
        lambda *args, **opts: _execute_replay(*args, fixture=fixture, decline_only=decline_only, **opts),
        run_dir, kind=source.source_kind, planner_low=False, clarify_consent=False,
        review_clarifications=True, guard_writes=args.guard_writes,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
