#!/usr/bin/env python3
"""用真实角色和独立语义核验重放四人桌群开场协商。"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
from tempfile import mkdtemp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mahjong_agent_runtime import OpenAICompatibleAgentClient, SQLiteAgentStore
from mahjong_agent_runtime.domains.game_commitments import active_game_participant_ids
from mahjong_agent_runtime.llm import model_response_source
from mahjong_agent_runtime.models import DEFAULT_TZ, using_clock
from scripts.run_venue_simulation import _real_roles, _select_local_model
from tests.simulation.sim_adapter import build_runtime
from tests.simulation.venue_actors import ActorContext, generate_personas, write_personas
from tests.simulation.venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel
from tests.simulation.venue_rolling_companions import TABLE_COMPANION_WAKE
from tests.simulation.venue_rolling_roles import RollingRoleTransportPaused, VenueRollingRoles
from tests.simulation.venue_table_group import VenueTableGroup
from tests.simulation.venue_table_opening import VenueTableOpening
from tests.simulation.venue_table_shop import VenueTableShop
from tests.simulation.venue_world import VenueWorld


_REVIEW = """你独立核对四位牌友对同一份开场提议的真实回复。只根据已经送达的消息判断。
accept 仅限本人明确同意该具体开场，比如照当前清晰提议回复行、可以、我准时到。
decline 用于本人明确无法按该时间、拒绝或另提时间取代它。条件式接受、只报估计、问句、打招呼和不清楚的用 unclear。
保留原回复消息 ID，每个消息恰好一次。没有回复的人不能加票。选择下一个备选只依据当前回复出现的实际时间分歧。
votes 的格式为 {"消息ID":"accept|decline|unclear"}。"""


def _review(client, proposal, replies, *, later: datetime | None, trace_id: str) -> tuple[dict, dict]:
    """校验独立模型从这一版提议实际收到的原话提取的意见。"""
    payload = {
        "proposal": proposal.message.text,
        "later_candidate": later.isoformat() if later else None,
        "replies": {row.message.message_id: row.message.text for row in replies},
    }
    schema = ('只返回 JSON 对象 {"votes":{...}}。' if later is None else
              '只返回 JSON 对象 {"votes":{...},"try_later":false}。只有目前有人拒绝且从原话看较晚备选仍合理，try_later 才为 true。')
    raw = client.complete([{"role": "system", "content": f"{_REVIEW}\n{schema}"},
                           {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                          trace_id=trace_id, timeout_seconds=35)
    diagnostics = {"raw": str(raw), "response_source": model_response_source(raw, trace_id=trace_id, step_index=1)}
    try:
        data = json.loads(str(raw))
        votes = data["votes"]
        if not isinstance(votes, dict) or set(votes) != set(payload["replies"]) or any(
            value not in ("accept", "decline", "unclear") for value in votes.values()
        ):
            raise ValueError("invalid table opening votes")
        if (set(data) != ({"votes"} if later is None else {"votes", "try_later"})
                or type(data.get("try_later", False)) is not bool
                or data.get("try_later") and "decline" not in votes.values()):
            raise ValueError("invalid table opening review")
    except (TypeError, KeyError, ValueError) as exc:
        diagnostics["error"] = str(exc)
        return {}, diagnostics
    return data, diagnostics


def _group_fixture(archive: Path, opened_at: datetime, people) -> VenueTableGroup:
    """由正式业务事务建一张人齐开并由四位角色分别占席的沙盒局。"""
    store = SQLiteAgentStore(archive / "agent.sqlite3")
    with using_clock(lambda: opened_at - timedelta(minutes=5)):
        store.configure_rooms(["东二包间"])
        owner, *others = people
        game, _ = store.create_game(
            conversation_id="opening-organizer-private", organizer_id=owner.sender_id, organizer_name=owner.name,
            known_players=[], requirement={"game_type": "hangzhou_mahjong", "stake": 1,
                "smoke_preference": "no_smoking", "start_time_kind": "asap_when_full", "duration_hours": 4},
            trace_id="table-opening-seed",
        )
        for person in others:
            store.join_game(game_id=game.game_id, customer_id=person.sender_id, display_name=person.name,
                            trace_id=f"participant:{person.sender_id}")
    return VenueTableGroup(store, archive / "table-chat.sqlite3", game.game_id,
                           actor_ids=[person.sender_id for person in people] + ["table-outsider"], opened_at=opened_at)


def _actor_round(actor, group, people, windows, proposal, start, opened_at, *, number: int) -> tuple[list, list]:
    """只有真实收到提议的角色可自主回复。各人只知道自己的时刻及送达的小群历史。"""
    decisions, replies = [], []
    for offset, person in enumerate(people, 1):
        arrival, free_until = windows[person.sender_id]
        receipt = group.received(person.sender_id, proposal.message.message_id)
        if receipt is None:
            continue
        at = proposal.message.sent_at.astimezone(DEFAULT_TZ) + timedelta(seconds=offset * 10)
        details = {"case_intent": (
            "你已报名这张桌但此前没有定开场。刚收到小群的具体时间提议。按本人的可达时间和群里实际对话"
            "决定接受、提出别的时间、回复自己不能等或先不说话。这次你打算打满四小时，若四小时安排不进自己的时间就说明。"
            "如果同意，直接在小群里说明自己可以。没有决定就不用答应。"
        ), "can_arrive": True, "free_from": opened_at.isoformat(),
            "free_until": free_until.isoformat(),
            "earliest_arrival": arrival.isoformat(), "scheduled_start": start.isoformat()}
        context = ActorContext(
            person, at.isoformat(), history=group.events_for(person.sender_id),
            events=[{"kind": "table_opening_proposal", "caused_by": receipt.message.message_id, "details": details}],
            private_targets=(), group_targets=(group.room_id,),
        )
        trace_id = f"table-opening:{number}:{offset}"
        decision = actor.decide(context, trace_id=trace_id)
        decisions.append(decision.to_dict())
        action = decision.action
        if (action is not None and action.action == "send" and action.kind == "group"
                and action.recipient_id == group.room_id):
            replies.append(group.send(MessageEnvelope(trace_id, person.sender_id, action.text, at, "group",
                                                     group.room_id, action.quote_message_id)))
    return decisions, replies


def observed_time_conflict(round_record: dict) -> bool:
    """核对确实没有同意的人是否拒绝或明示了冲突，但不把附条件当作退席。"""
    votes = round_record["review"].get("votes") or {}
    for row in round_record["decisions"]:
        vote = votes.get(row["trace_id"])
        if vote == "decline":
            return True
        if vote != "unclear" or not row.get("availability_reviews"):
            continue
        details = row["context"]["events"][-1]["details"]
        start = datetime.fromisoformat(details["scheduled_start"])
        claims = row["availability_reviews"][-1].get("claims") or ()
        if any(claim["kind"] == "play_until" and datetime.fromisoformat(claim["local_at"]) < start + timedelta(hours=4)
               for claim in claims):
            return True
    return False


def run(archive: Path, *, no_common_time: bool = False) -> dict:
    """保留实际对话和独立核验，明确区分协商完成与合理未定时。"""
    opened_at = datetime(2026, 11, 27, 18, 10, tzinfo=DEFAULT_TZ)
    early, late = opened_at.replace(hour=19, minute=0), opened_at.replace(hour=19, minute=30)
    people = generate_personas(69012, count=4)
    arrivals = (opened_at.replace(hour=18, minute=45), opened_at.replace(hour=18, minute=55), early, late)
    final_window = opened_at.replace(hour=23, minute=59)
    windows = {person.sender_id: (arrival, early + timedelta(hours=4) if no_common_time and index == 1 else final_window)
               for index, (person, arrival) in enumerate(zip(people, arrivals, strict=True))}
    config = _select_local_model("opencode-go", None, workers=1)
    actor, _ = _real_roles(config, force_opencode_role=True)
    group = _group_fixture(archive, opened_at, people)
    opening = VenueTableOpening(group)
    history = []
    for number, start in enumerate((early, late), 1):
        at = opened_at + timedelta(minutes=number - 1)
        proposal = group.send(opening.proposal_message(f"opening-proposal:{number}", at=at, start_at=start))
        opening.record_proposal(proposal, start_at=start)
        decisions, replies = _actor_round(actor, group, people, windows, proposal, start, opened_at, number=number)
        try:
            review, diagnostics = _review(actor.client, proposal, replies,
                later=late if number == 1 else None, trace_id=f"table-opening-review:{number}")
        except Exception as exc:
            review, diagnostics = {}, {"error": type(exc).__name__}
        results = {}
        for reply in replies:
            ident = reply.message.message_id
            vote = (review.get("votes") or {}).get(ident)
            if vote in {"accept", "decline"}:
                earliest, free_until = windows[reply.message.sender_id]
                if vote == "accept" and (start < earliest or start + timedelta(hours=4) > free_until):
                    results[ident] = "acceptance_conflicts_with_private_schedule"
                else:
                    results[ident] = opening.record_vote(reply, proposal_id=proposal.message.message_id,
                                                        accepts=vote == "accept")
        history.append({"round": number, "proposal": proposal.history_row(), "decisions": decisions,
                        "review": review, "review_source": diagnostics, "results": results})
        if opening.current()["committed"] or not review.get("try_later"):
            break
    game = group.store.require_game(group.game_id)
    if game.planned_start_at is not None:
        group.send(MessageEnvelope(
            "opening-confirmed", AGENT_ID, f"那就 {game.planned_start_at:%H:%M} 开，大家按这个时间到哈。",
            opened_at + timedelta(minutes=3), "group", group.room_id,
        ))
    settlement = opening.current()
    checks = {
        "five_distinct_group_members_and_outsider_isolated": len(group.member_ids()) == 5 and not group.events_for("table-outsider"),
        "all_real_roles_completed_valid_actions": all(
            len(row["decisions"]) == 4 and all(item["status"] == "valid" and
            (item.get("response_source") or {}).get("http_response_complete") is True for item in row["decisions"])
            for row in history),
        "all_real_reviews_have_valid_votes": all(row["review"] for row in history),
        "settlement_matches_fixture_expectation": (not settlement["committed"] if no_common_time else
            settlement["committed"] and settlement["accepted_count"] == 4 and game.planned_start_at == late),
        "no_common_time_conflicts_observed_for_both_candidates": not no_common_time or (len(history) == 2 and all(
            observed_time_conflict(row) for row in history)),
        "no_ineligible_time_accepted_or_physical_side_effect": not any(
            value == "acceptance_conflicts_with_private_schedule" for row in history for value in row["results"].values()
        ) and not game.actual_started_at and not group.store.room_reservations,
    }
    records = [item.history_row() for item in group.delivered_receipts()]
    report = {"name": archive.name, "status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "scope": "real_customers_and_independent_table_opening_semantic_review", "scenario": (
                  "no_common_time" if no_common_time else "agree_after_counterproposal"),
              "model": config.model, "settlement": settlement, "rounds": history, "records": records}
    (archive / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    return report


class _CompanionProbe(VenueRollingRoles):
    """专项只领取真实桌群链路产生的事件，省去两天与本桌无关的营业唤醒。"""

    def _seed_timers(self) -> None:
        """现有 READY 牌局由通用调度自动找到，不额外安排角色定时活动。"""


def _prepare_companion(archive: Path, *, unavailable: bool, main_client=None):
    """锁定私人同行来话、匿名席和两种日程。历史素材通过真实私聊账本送达。"""
    at = datetime(2026, 12, 17, 17, 30, tzinfo=DEFAULT_TZ) if unavailable else datetime(2026, 11, 27, 17, 30, tzinfo=DEFAULT_TZ)
    seed = 61
    roster = {person.sender_id: person for person in generate_personas(seed, count=24)}
    people = tuple(roster[f"venue-sender-{index:04d}"] for index in (13, 1, 15, 18, 2))
    host, friend, *others, bystander = people
    store = SQLiteAgentStore(archive / "agent.sqlite3")
    runtime, _ = build_runtime(store, "real" if main_client is not None else "mock", model_client=main_client)
    channel = SandboxChannel(runtime, room_id="companion-probe", archive_path=archive / "chat.sqlite3",
                             identities={person.sender_id: person.name for person in people})
    with using_clock(lambda: at - timedelta(minutes=1)):
        game, _ = store.create_game(conversation_id=channel.private_conversation_id(host.sender_id),
            organizer_id=host.sender_id, organizer_name=host.name, trace_id="companion-probe-captain",
            known_players=[{"customer_id": host.sender_id, "seat_count": 2, "anonymous_seat_count": 1}],
            requirement={"game_type": "hangzhou_mahjong", "stake": 1, "smoke_preference": "no_smoking",
                         "start_time_kind": "asap_when_full", "duration_hours": 4})
        for person in others:
            game, _ = store.join_game(game_id=game.game_id, customer_id=person.sender_id, display_name=person.name,
                                    trace_id=f"companion-probe-seat:{person.sender_id}")
    prior = (
        MessageEnvelope("companion-probe-peer-1", host.sender_id,
            "今晚想一起去打无烟杭麻不？一块底，大概打四个小时，想六点半到七点之间开。你要是想去我先找老板帮咱俩留位置。",
            at - timedelta(hours=8), "private", friend.sender_id),
        MessageEnvelope("companion-probe-peer-2", friend.sender_id,
            "我有兴趣，你先问吧。几点开还得看我排班，有群了叫我一声。",
            at - timedelta(hours=7, minutes=59), "private", host.sender_id),
    )
    for message in prior:
        channel.send(message)
    source_ids = {message.message_id for message in prior}
    world = VenueWorld(seed, at)
    common_time = at.replace(hour=18, minute=30)
    checks = {
        "captain_has_one_anonymous_seat_and_one_known_companion": game.seat_summary()["anonymous_seat_count"] == 1
            and [edge.target_sender_id for edge in host.relationships if edge.kind == "regular_partner"] == [friend.sender_id],
        "peer_terms_actually_delivered_to_both": all(channel.completed_peer_message(message_id) is not None for message_id in source_ids)
            and all(source_ids <= {row["message_id"] for row in channel.events_for(person.sender_id)} for person in (host, friend)),
        "peer_history_private_to_captain_and_friend": all(source_ids.isdisjoint({row["message_id"] for row in channel.events_for(viewer)})
            for viewer in (AGENT_ID, bystander.sender_id, *(person.sender_id for person in others))),
        "fixture_calendars_match_case": all(world.availability(person).available_for(common_time, 4) for person in (host, *others))
            and world.availability(friend).available_for(common_time, 4) is not unavailable,
    }
    write_personas(archive / "personas.json", people)
    return at, seed, people, channel, game.game_id, source_ids, checks


def run_companion(archive: Path, *, unavailable: bool, prepare_only: bool) -> dict:
    """留存真实卡片操作、双方私聊来源和四人最终票据。准备档无需调用模型。"""
    config = None if prepare_only else _select_local_model("opencode-go", None, workers=1)
    main_client = OpenAICompatibleAgentClient(config) if config is not None else None
    at, seed, people, channel, game_id, source_ids, checks = _prepare_companion(archive, unavailable=unavailable, main_client=main_client)
    report = {"checks": checks, "settlement": None, "rounds": [], "status": "failed", "seed": seed,
              "start_at": at.isoformat(), "game_id": game_id, "peer_source_message_ids": sorted(source_ids),
              "scope": "rolling_companion", "case": "invitee_unavailable" if unavailable else "compatible"}
    if config is None or not all(checks.values()):
        if prepare_only and all(checks.values()):
            report["status"] = "prepared"
        (archive / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report

    actor, _ = _real_roles(config, force_opencode_role=True)
    scenario = _CompanionProbe(people=people, actor=actor, judge=None, channel=channel, run_dir=archive,
        start_at=at, seed=seed, active_users=len(people), workers=1, dialogue_turns=2, judge_users=1,
        table_groups=True, table_shop=VenueTableShop(actor.client, archive / "table_shop.jsonl"))
    try:
        rolling = scenario.run()
    except RollingRoleTransportPaused as exc:
        report.update(status="transport_paused", transport_checkpoint=exc.checkpoint)
        (archive / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    group = channel.table_group(VenueTableGroup.room_id_for_game(game_id))
    game = group.store.require_game(game_id)
    settlement = VenueTableOpening(group).current()
    proposals = [row for row in group.delivered_receipts() if row.message.message_id.startswith("table-opening-offer|")]
    roles = scenario.journal.events()
    answer = next((event for event in roles if event.kind == TABLE_COMPANION_WAKE), None)
    answer_decision = answer.decision["decision"] if answer else {}
    role_id = people[1].sender_id
    invitation = group.invitation_for(role_id, answer.origin_id) if answer else None
    live = rolling["checks"].get("customer_decisions_from_model") is True
    checks.update(real_model_actions=live,
        companion_saw_peer_terms=bool(answer and source_ids <= {row["message_id"] for row in answer_decision["context"]["history"]}),
        exact_private_invitation_source=bool(invitation and answer.actor_id == role_id and answer.caused_by == invitation.invitation_id
            and invitation.inviter_id == people[0].sender_id and group.invitation_for(AGENT_ID, invitation.invitation_id) is None),
        bystander_cannot_see_table_chat=not group.events_for(people[-1].sender_id))
    if unavailable:
        outcome = (answer_decision.get("action") or {}).get("action")
        checks["valid_refusal_or_silence"] = bool(outcome in {"decline_companion", "wait"}
            and invitation and invitation.accepted is (False if outcome == "decline_companion" else None)
            and role_id not in group.member_ids() and game.seat_summary()["anonymous_seat_count"] == 1
            and game.planned_start_at is None)
    else:
        expected = {person.sender_id for person in people[:-1]}
        proposal_at = next((row.message.sent_at for row in proposals if settlement
                            and row.message.message_id == settlement["proposal_id"]), None)
        checks["confirmed_once_with_four_real_votes"] = bool(settlement and settlement["committed"]
            and invitation and invitation.accepted is True
            and settlement["accepted_count"] == 4 and group.member_ids() == {AGENT_ID, *expected}
            and active_game_participant_ids(game) == expected
            and sum(row.reason == "venue_table_opening_confirmed" for row in group.store.transitions) == 1)
        calendars = [scenario.world.availability(person) for person in people[:-1]]
        checks["opening_fits_every_actual_calendar"] = bool(proposal_at and game.planned_start_at and all(
            day.available_for(game.planned_start_at, 4)
            and proposal_at + timedelta(minutes=day.travel_minutes) <= game.planned_start_at for day in calendars))
    report.update(status="passed" if all(checks.values()) else "failed", model=config.model, settlement=settlement,
                  game=game.to_dict(), table_groups=rolling["rolling_table_groups"], observed_companion_action=answer_decision.get("action"),
                  rounds=[row.history_row() for row in proposals])
    (archive / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    """显式开启真模型并保留独立沙盒档案。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runtime_data/venue_simulation/table_opening")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--no-common-time", action="store_true", help="此档两项开场提议都有人无法参加")
    modes.add_argument("--rolling-companion", choices=("compatible", "invitee-unavailable"), help="从店家提醒起验证同行邀请和四人协商")
    parser.add_argument("--prepare-only", action="store_true", help="只落地同行专项的模拟盘面及私聊来源")
    args = parser.parse_args()
    if args.prepare_only and not args.rolling_companion:
        parser.error("--prepare-only 需要 --rolling-companion")
    args.runs_dir.mkdir(exist_ok=True, parents=True)
    archive = Path(mkdtemp(prefix="table-opening-", dir=args.runs_dir))
    report = (run_companion(archive, unavailable=args.rolling_companion == "invitee-unavailable", prepare_only=args.prepare_only)
              if args.rolling_companion else run(archive, no_common_time=args.no_common_time))
    print(json.dumps({"path": str(archive), "status": report["status"], "checks": report["checks"],
                      "settlement": report["settlement"], "rounds": len(report["rounds"])}, ensure_ascii=False))
    return int(report["status"] not in {"passed", "prepared"})


if __name__ == "__main__":
    raise SystemExit(main())
