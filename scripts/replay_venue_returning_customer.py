#!/usr/bin/env python3
"""验证同一顾客三次到店散场后，用口语化短句开启新的组局任务。"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
from tempfile import mkdtemp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mahjong_agent_runtime import JsonlTraceRecorder, OpenAICompatibleAgentClient, SQLiteAgentStore
from mahjong_agent_runtime.models import using_clock
from scripts.replay_venue_consent import _deliver_replay_message, _delivered_reply, _successful_tools
from scripts.run_venue_simulation import _model_execution, _real_roles, _select_local_model
from tests.simulation.sim_adapter import build_runtime
from tests.simulation.venue_actors import ActorContext, FamiliarShop, generate_personas
from tests.simulation.venue_channel import AGENT_ID, OWNER_ID, MessageEnvelope, SandboxChannel
from tests.simulation.venue_dialogues import _live_actor_action


FIXTURE = ROOT / "eval/golden/venue_returning_customer_replay.json"
ROOM = "东二包间"
CASES = ("open-code-customer", "exact-existing-table", "exact-new-table")


def _table(store, channel, peers, start, terms, *, trace: str):
    """通过业务接口建立牌桌和真实确认席，历史桌和将来桌共用同一流程。"""
    host, *players = peers
    with using_clock(lambda: start - timedelta(hours=1)):
        game, _ = store.create_game(conversation_id=channel.private_conversation_id(host.sender_id),
            organizer_id=host.sender_id, organizer_name=host.name, known_players=[], trace_id=trace,
            requirement={**terms, "start_time_kind": "scheduled", "start_time": start.strftime("%H:%M"),
                         "planned_start_at": start.isoformat(), "duration_kind": "fixed", "duration_hours": 2})
        for player in players:
            store.join_game(game_id=game.game_id, customer_id=player.sender_id, display_name=player.name, trace_id=trace)
    return game.game_id


def _completed_history(store, channel, customer, peers, start, fixture):
    """建立已实际签到、开打、散场的历史。顾客画像仍从空白身份开始。"""
    terms, receipt, games = fixture["learned_conditions"], None, set()
    participants = (*peers, customer)
    for days in fixture["completed_attended_days_ago"]:
        play = start - timedelta(days=days)
        game_id = _table(store, channel, participants, play, terms, trace=f"historical-game:{days}")
        with using_clock(lambda: play):
            store.start_game_play(game_id, attendee_ids=[person.sender_id for person in participants],
                                  room_id=ROOM, at=play, trace_id=f"historical-start:{days}")
        finish = play + timedelta(hours=2)
        with using_clock(lambda: finish):
            store.finish_game_play(game_id, at=finish, trace_id=f"historical-finish:{days}")
        checkout = f"past-checkout:{days}"
        channel.publish_owner(kind="private", recipient_id=customer.sender_id, sent_at=finish, message_id=checkout,
            text=("你常打的那桌半块底无烟杭麻散场了，下回见。" if not games else "今天这桌散了哈，下回见。"))
        if receipt is None:
            receipt = checkout
        games.add(game_id)
    return receipt, games


def _role_opening(actor, customer, channel, *, at, time, terms, receipt):
    """真实顾客看到本人曾收到的结账消息，开场由自己的模型决定。"""
    history = [{**row, "sender_id": AGENT_ID} if row["sender_id"] == OWNER_ID else row
               for row in channel.events_for(customer.sender_id)]
    context = ActorContext(customer, at.isoformat(), history=history, events=[{
        "event_id": "returning-open", "kind": "session_open", "at": at.isoformat(),
        "details": {"case_goal": "organize_new", "start_time": time, "knows_shopkeeper": True,
            "case_intent": "今晚要请熟悉的店家组局。对方知道你的惯例，你习惯给他发一句简短微信，只讲来意和几点开，其他常打条件如果他问再说。"},
    }], familiar_shop=FamiliarShop(acknowledgement_message_id=receipt, usual_preferences=terms))
    return actor.decide(context, trace_id="role:returning-open")


def _used_conditions(query: dict, terms: dict, start: datetime) -> bool:
    """同时核对真正执行的个人条件与开场时间，不能只在回复里提条件。"""
    same_terms = all(str(query.get(key)) == str(value) for key, value in terms.items())
    exact = query.get("planned_start_at") or query.get("start_time")
    if exact:
        return same_terms and exact in {start.isoformat(), start.strftime("%H:%M")}
    window = query.get("start_time_range") or {}
    return (same_terms and window.get("start_at") == start.isoformat()
            and window.get("end_at") == start.isoformat())


def replay(case: str, fixture: dict, archive: Path, *, config) -> dict:
    """分别验收已有桌、需要新开和 OpenCode 顾客自主开场。"""
    actor, judge = _real_roles(config, force_opencode_role=True)
    people = generate_personas(42)
    customer = next(person for person in people if person.preferred_game == "杭州麻将"
                    and person.stake_min <= 0.5 <= person.stake_max and person.smoking == "non_smoking"
                    and person.temperament.opening_disclosure == "main_need_first")
    peers = tuple(person for person in people if person != customer)[:3]
    start = datetime.fromisoformat(f'{fixture["day"]}T{fixture["opening_time"]}:00+08:00')
    at = start - timedelta(minutes=40)
    terms = fixture["learned_conditions"]
    trace = JsonlTraceRecorder(archive / "agent_trace.log")
    store = SQLiteAgentStore(archive / "agent.sqlite3")
    with using_clock(lambda: start - timedelta(days=10)):
        store.configure_rooms([ROOM])
        runtime, _ = build_runtime(store, "real", seed=42, model_client=OpenAICompatibleAgentClient(config),
                                   trace_recorder=trace, local_coordination=True)
        channel = SandboxChannel(runtime, room_id="venue-group",
                                 identities={person.sender_id: person.name for person in (*peers, customer)},
                                 archive_path=archive / "chat.sqlite3")
    receipt, historical_ids = _completed_history(store, channel, customer, peers, start, fixture)
    prior = SQLiteAgentStore(store.path)
    history_default = prior.completed_game_preference(customer.sender_id)
    prior._connection.close()
    matching = None if case == "exact-new-table" else _table(store, channel, peers, start, terms, trace="next-table")
    matching_before = store.require_game(matching).to_dict() if matching else None
    role = _role_opening(actor, customer, channel, at=at, time=fixture["opening_time"], terms=terms, receipt=receipt
                         ) if case == "open-code-customer" else None
    if role is not None:
        (archive / "role.json").write_text(json.dumps(role.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    text = fixture["exact_customer_opening"]
    if role is not None:
        proposed = role.action
        if proposed is None or proposed.action != "send" or proposed.kind != "private" or proposed.recipient_id != AGENT_ID:
            return {"status": "inconclusive", "case": case, "role_status": role.status, "role_trace": role.trace_id}
        text = proposed.text
    turn = MessageEnvelope("returning-open", customer.sender_id, text, at, "private", AGENT_ID)
    conversation = channel.private_conversation_id(customer.sender_id)
    assert store.current_task_context(conversation, customer.sender_id) is None
    response = _deliver_replay_message(channel, archive / "events.jsonl", case, turn)
    rows = channel.events()
    checks, persisted = _verify(store, case, turn, conversation, start, terms, historical_ids,
        matching, matching_before, history_default, response, runtime.tool_gateway.tools, role,
        expected_new_party=fixture["expected_new_party"], inbound_count=sum(row["direction"] == "inbound" for row in rows))
    (archive / "chat.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    evidence = [{"event_id": "checks", "actual_tool_and_state_checks": checks, "persisted_table": persisted}]
    evidence.extend({"message_id": row["message_id"], "direction": row["direction"], "text": row["text"]}
                    for row in rows if row["direction"] in {"inbound", "outbound"})
    assessment = ["店家沿用顾客常打条件处理这次19:00的组局，没有重复询问已知的玩法、档位和烟况，回复如实承接实际局况"]
    instructions = "按顾客实际发送的来话和实际收到的回复判断。检查实际局况请引用 checks，聊天检查引用对应来话及所有相关店家回复。"
    if case == "exact-new-table":
        assessment += ["本次只收到一次本人组局委托，没有提同行。店家回复未要求确认这次有几个人或是否只来一位",
                       "店家如实说明已登记本人一位的待组局，没有虚称其他人已经答应、人齐或已订包间"]
        instructions += "新局缺三席时，可以说“还差三个”或省略人数。询问‘几位’‘就你一个吗’和要求稍后确认人数均不通过。"
    if role is not None:
        assessment.append("熟客首句只提供必要来意或时间，没有一开口就重报玩法、档位和烟况")
    verdict = judge.judge(case, evidence, trace_id=f"judge:{case}", checks=assessment, instructions=instructions)
    status = ("failed" if not all(checks.values()) or verdict.verdict == "fail" else
              "passed" if verdict.status == "valid" and verdict.verdict == "pass" else "inconclusive")
    return {"status": status, "case": case, "checks": checks, "attendance_default": history_default,
        "persisted_table": persisted, "trace_ids": [response["trace_id"]], "judgment": verdict.to_dict(),
        "model_execution": {"agent": _model_execution(runtime.llm_client, timeout_seconds=runtime.llm_timeout_seconds),
                            "role": _model_execution(actor.client, timeout_seconds=actor.timeout_seconds) if role else None}}


def _verify(store, case, turn, conversation, start, terms, history, matching,
            matching_before, history_default, response, tool_definitions, role, *, expected_new_party, inbound_count):
    """重开业务库核对实际查找及建局，一条明确委托即按本人一席结算。"""
    reopened = SQLiteAgentStore(store.path)
    try:
        games = reopened.games
        bookings = reopened.room_reservations
    finally:
        reopened._connection.close()
    customer_id = turn.sender_id
    tools = _successful_tools(response)
    lookups = [tool["result"] for tool in tools if tool["name"] == "search_current_games"]
    matching_found = any((match.get("game") or {}).get("game_id") == matching
                         for row in lookups for match in row.get("matches", ()))
    additions = set(games) - history - ({matching} if matching else set())
    new = games[next(iter(additions))] if len(additions) == 1 else None
    checks = {
        "default_available_after_reopening": history_default == {**terms, "completed_games": len(history)},
        "first_request_used_history_and_new_time": any(_used_conditions(row.get("requirement", {}), terms, start)
                                                       for row in lookups),
        "single_customer_message": inbound_count == 1,
        "customer_received_reply": _delivered_reply(response, turn),
    }
    persisted = {"created_tables": len(additions), "room_reserved_for_current_table": False}
    if case == "exact-new-table":
        creations = [tool for tool in tools if tool["name"] == "create_game" and not tool.get("deduplicated")]
        checks["one_new_table_from_original_message"] = bool(new and len(creations) == 1
            and creations[0].get("result", {}).get("game", {}).get("game_id") == new.game_id
            and new.conversation_id == conversation and new.organizer_id == customer_id and new.planned_start_at == start
            and all(str(new.requirement.get(key)) == str(value) for key, value in terms.items()))
        claims = [(item.customer_id, item.seat_count) for item in new.participants if item.status in {"joined", "confirmed"}] if new else []
        checks["persisted_organizer_only_and_173"] = bool(new and claims == [(customer_id, expected_new_party["organizer_seats"])]
            and new.status.value in {"forming", "inviting"} and new.remaining_seats() == expected_new_party["vacant_seats"]
            and new.requirement.get("seat_format") == expected_new_party["seat_format"])
        reserved = bool(new and any(booking.game_id == new.game_id for booking in bookings.values()))
        checks["no_unrequested_room_booking"] = not reserved and not any(tool["name"] == "reserve_room" for tool in tools)
        if new:
            persisted.update(status=new.status.value, organizer_seats=sum(seats for actor, seats in claims if actor == customer_id),
                claimed_seats=sum(seats for _, seats in claims), vacant_seats=new.remaining_seats(),
                seat_format=new.requirement.get("seat_format"), room_reserved_for_current_table=reserved)
    else:
        checks["looked_up_existing_table_without_writing"] = bool(matching_found and not additions and matching in games
            and games[matching].to_dict() == matching_before and games[matching].remaining_seats() == 1
            and all(tool["name"] not in tool_definitions or tool_definitions[tool["name"]].execution_mode == "read_only"
                    for tool in tools))
        if matching in games:
            persisted.update(status=games[matching].status.value, vacant_seats=games[matching].remaining_seats())
    if role is not None:
        checks["genuine_customer_model_response"] = _live_actor_action({"decision": role.to_dict()})
    return checks, persisted


def main() -> int:
    """每个已归档用例独立建库，原话和实际结果可重放。"""
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runtime_data/venue_simulation/returning_customer")
    args = parser.parse_args()
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    archive = Path(mkdtemp(prefix=f"{args.case}-", dir=args.runs_dir))
    print(archive, flush=True)
    report = replay(args.case, fixture, archive, config=_select_local_model("opencode-go", None, workers=1))
    (archive / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report["status"], report.get("checks", {}), flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
