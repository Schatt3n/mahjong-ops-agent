#!/usr/bin/env python3
"""在独立门店对照只查现成局与后来明确委托组局的业务边界。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mahjong_agent_runtime import SQLiteAgentStore  # noqa: E402
from scripts.replay_venue_consent import (  # noqa: E402
    ReplaySource, _deliver_replay_message, _delivered_reply, _load_archived_turns, _load_portable_fixture,
    _run_venue_replay, _snapshot, _successful_tools,
)
from tests.simulation.venue_actors import SemanticJudge, generate_personas  # noqa: E402
from tests.simulation.venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel  # noqa: E402
from tests.simulation.venue_cohort import _INCOMPATIBLE_GAME_TYPES  # noqa: E402

_PORTABLE_FIXTURE = ROOT / "eval" / "golden" / "venue_lookup_delegation_replay.json"
_CONTROL_MESSAGE_ID = "customer:replay-lookup-explicit-delegation"
_LOOKUP_WRITES = {
    "create_game", "join_game", "record_candidate_reply", "create_invite_drafts", "reserve_room",
    "register_waiting_demand", "cancel_waiting_demand", "update_game_requirement", "update_game_status",
}
_CLARIFICATION_CHECK = "店家实际询问的是回答现成局查询所必需且客户尚未给过的条件，没有许诺留位或组局"


@dataclass(frozen=True, slots=True)
class LookupReplaySource(ReplaySource):
    """在通用重放材料之外，另存本案应查询的标准玩法。"""

    expected_game_type: str = ""


def _load_lookup_archive(source_dir: Path, customer: str) -> LookupReplaySource:
    """只加载混合仿真中目标为现成局查询的顾客及其完整入站因果链。"""
    report = json.loads((source_dir / "report.json").read_text(encoding="utf-8"))
    venue = json.loads((source_dir / "venue.json").read_text(encoding="utf-8"))
    cohort = report.get("cohort") or {}
    if (report.get("scenario") != "mixed" or not cohort.get("initial_game")
            or cohort.get("expected_outcomes", {}).get(customer) != "no_match"):
        raise ValueError("来源必须是有初始库存的混合仿真，且顾客目标为现成局查询")
    seed = report["seed"]
    people = {person.sender_id: person for person in generate_personas(seed)}
    person = people[customer]
    game_type = _INCOMPATIBLE_GAME_TYPES[person.preferred_game]
    turns = _load_archived_turns(source_dir, customer)
    control = (f"现在我明确请你帮我新建并招人组一桌，今晚八点的一元底{person.preferred_game}，打两个小时。"
               "就我一个人参加，只开待组局，不订包间，也别登记候补。")
    return LookupReplaySource(seed, {key: value.name for key, value in people.items()}, venue, cohort["initial_game"], turns,
                              explicit_control=control, expected_game_type=game_type)


def _load_lookup_fixture() -> LookupReplaySource:
    """采用仓库另写的查询、委托对照文本和一桌无匹配的现有库存。"""
    base = _load_portable_fixture()
    record = json.loads(_PORTABLE_FIXTURE.read_text(encoding="utf-8"))
    first = base.turns[0]
    messages = tuple(replace(first, message_id=f"customer:lookup-fixture-{index + 1}", text=body,
                             sent_at=first.sent_at + timedelta(minutes=index))
                     for index, body in enumerate(record["turns"]))
    return LookupReplaySource(
        base.seed, base.identities, base.venue, base.initial_game, messages, source_kind=base.source_kind,
        explicit_control=record["explicit_control"],
        expected_game_type=record["game_type"],
    )


def _waiting_snapshot(store: SQLiteAgentStore) -> dict:
    """读取可持久回查的候补登记，比较前后身份、条件和状态。"""
    return {key: demand.to_dict() for key, demand in store.waiting_demands.items()}


def _lookup_evidence(response: dict, expected_game_type: str) -> bool:
    """接受实际执行了与客户玩法一致的现成牌局查询。"""
    return any(
        tool.get("name") == "search_current_games"
        and (tool.get("result") or {}).get("requirement", {}).get("game_type") == expected_game_type
        and (tool.get("result") or {}).get("matches") == [] and (tool.get("result") or {}).get("alternatives") == []
        for tool in _successful_tools(response)
    )


def _clarification_evidence(response: dict, message: MessageEnvelope, history: list[dict],
                            judge: SemanticJudge | None, events_path: Path) -> bool:
    """没查局时，由独立模型核对确有新缺项、实际澄清，并保存可引用证据。"""
    if judge is None or response.get("status") != "waiting_user" or not _delivered_reply(response, message):
        return False
    outcome = judge.judge(f"lookup-clarification:{message.message_id}", history,
                          trace_id=f"judge-lookup:{message.message_id}", checks=(_CLARIFICATION_CHECK,),
                          instructions="只判本次最后一条店家消息。此前消息只用来核对用户已经告知的条件。")
    with events_path.with_name("clarifications.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(outcome.to_dict(), ensure_ascii=False) + "\n")
    return outcome.status == "valid" and outcome.verdict == "pass"


def _control_terms_match(requirement: dict, control: MessageEnvelope, expected_game_type: str) -> bool:
    """核对新建局沿用明确委托的玩法、同一天晚八点、一元底及两小时。"""
    start_at = requirement.get("planned_start_at")
    expected_start_at = control.sent_at.replace(hour=20, minute=0, second=0, microsecond=0)
    return bool(
        requirement.get("game_type") == expected_game_type and requirement.get("base_stake") == 1
        and requirement.get("duration_kind") == "fixed" and requirement.get("duration_hours") == 2
        and start_at and datetime.fromisoformat(start_at) == expected_start_at
    )


def _execute_lookup(
    store: SQLiteAgentStore, channel: SandboxChannel, source: LookupReplaySource, game_id: str,
    events_path: Path, *, judge: SemanticJudge | None = None,
) -> dict:
    """逐轮验证只查未写，发生越权即存档停止，安全后才尝试正对照。"""
    checks: dict[str, bool | None] = dict.fromkeys(
        [f"readonly_turn_{index + 1}_{aspect}" for index in range(len(source.turns))
         for aspect in ("reply_delivered", "query_or_clarification", "no_unauthorized_writes")]
        + ["control_reply_delivered", "control_created_forming_game", "control_only_authorized_organizer_seat",
           "control_no_other_game_room_or_waiting_changes"]
    )
    baseline_games, baseline_rooms = _snapshot(store)
    baseline_waiting = _waiting_snapshot(store)
    history = []
    for index, message in enumerate(source.turns, start=1):
        response = _deliver_replay_message(channel, events_path, f"{source.source_kind}_lookup_{index}", message)
        history.append({"message_id": message.message_id, "speaker": "customer", "text": message.text})
        history.extend({"message_id": row["message_id"], "speaker": "agent", "text": row["text"]}
                       for row in response.get("archive_outputs", ()) if row.get("recipient_id") == message.sender_id)
        reply_ok = _delivered_reply(response, message)
        scoped_tools = _successful_tools(response)
        state_unchanged = (
            (_snapshot(store), _waiting_snapshot(store)) == ((baseline_games, baseline_rooms), baseline_waiting)
            and not any(tool.get("name") in _LOOKUP_WRITES for tool in scoped_tools)
        )
        checks[f"readonly_turn_{index}_reply_delivered"] = reply_ok
        checks[f"readonly_turn_{index}_no_unauthorized_writes"] = state_unchanged
        checks[f"readonly_turn_{index}_query_or_clarification"] = _lookup_evidence(response, source.expected_game_type) or (
            state_unchanged and _clarification_evidence(response, message, history, judge, events_path)
        )
        if not all(checks[name] for name in checks if name.startswith(f"readonly_turn_{index}_")):
            return {"status": "failed", "checks": checks}

    last = source.turns[-1]
    control = MessageEnvelope(_CONTROL_MESSAGE_ID, last.sender_id, source.explicit_control,
                              last.sent_at + timedelta(minutes=1), "private", AGENT_ID)
    response = _deliver_replay_message(channel, events_path, "authored_explicit_delegation", control)
    called = _successful_tools(response)
    reopened = SQLiteAgentStore(store.path)
    end_games, end_rooms = _snapshot(reopened)
    written = end_games.keys() - {game_id}
    creates = [tool for tool in called if tool.get("name") == "create_game"]
    created = end_games[next(iter(written))] if len(written) == 1 else None
    checks["control_reply_delivered"] = _delivered_reply(response, control)
    checks["control_created_forming_game"] = bool(
        created and created["status"] == "forming" and created["organizer_id"] == last.sender_id
        and _control_terms_match(created["requirement"], control, source.expected_game_type) and len(creates) == 1
        and (creates[0].get("result", {}).get("game") or {}).get("game_id") == created["game_id"]
    )
    checks["control_only_authorized_organizer_seat"] = bool(
        created and len(created["participants"]) == 1 and created["participants"][0]["customer_id"] == last.sender_id
        and created["participants"][0]["seat_count"] == 1 and created["remaining_seats"] == 3
        and not any(tool.get("name") in {"join_game", "record_candidate_reply"} for tool in called)
    )
    checks["control_no_other_game_room_or_waiting_changes"] = (
        end_games.get(game_id) == baseline_games[game_id] and end_rooms == baseline_rooms
        and _waiting_snapshot(reopened) == baseline_waiting
        and not any(tool.get("name") in {"reserve_room", "register_waiting_demand", "cancel_waiting_demand"} for tool in called)
    )
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks}


def main() -> int:
    """使用原始坏例或便携样例验证相同的权限合同。"""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("source_dir", type=Path, nargs="?", help="含查询误组局的历史混合仿真目录")
    source.add_argument("--fixture", action="store_true", help="使用仓库中另写的权限边界样例")
    parser.add_argument("--customer", help="需重放的模拟顾客编号，历史目录必填")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runtime_data" / "venue_simulation" / "lookup_replays")
    parser.add_argument("--planner-low", action="store_true")
    parser.add_argument("--clarify-consent", action="store_true")
    args = parser.parse_args()
    if bool(args.source_dir) != bool(args.customer):
        parser.error("历史目录须同时指定 --customer，便携样例不使用它")
    run_dir = args.runs_dir / f"replay-{uuid.uuid4().hex[:10]}"
    summary = _run_venue_replay(
        lambda: _load_lookup_archive(args.source_dir, args.customer) if args.source_dir else _load_lookup_fixture(),
        _execute_lookup, run_dir, kind="archived_lookup_replay" if args.source_dir else "portable_lookup_fixture",
        planner_low=args.planner_low, clarify_consent=args.clarify_consent, review_clarifications=True,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
