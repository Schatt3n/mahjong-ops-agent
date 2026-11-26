#!/usr/bin/env python3
"""在 OpenCode 沙盒复验含糊原桌指代与另查红中的同轮回复。"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mahjong_agent_runtime import SQLiteAgentStore  # noqa: E402
from mahjong_agent_runtime.customer_visible_review import CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME  # noqa: E402
from mahjong_agent_runtime.domains.stake_values import stake_options  # noqa: E402
from mahjong_agent_runtime.models import DEFAULT_TZ, using_clock  # noqa: E402
from scripts.replay_venue_consent import (  # noqa: E402
    ReplaySource, _deliver_replay_message, _delivered_reply, _run_venue_replay, _snapshot, _successful_tools,
)
from tests.simulation.venue_actors import SemanticJudge  # noqa: E402
from tests.simulation.venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel  # noqa: E402
from tests.simulation.venue_rolling_board import VenueRollingBoard  # noqa: E402
from tests.simulation.venue_scenario import VENUE  # noqa: E402

_CASE = "synthetic_mixed_board_seq87_sibling"
_REFERENCE_RUN = "2026-11-25-real-34fa8e9813"
_CHECKS = {
    "原桌答复与正式看板及同桌查询事实一致": {"board", "table-read", "reply"},
    "单独回答七点后的一元红中查询且与实际空结果一致": {"personal-read", "reply"},
    "用具体看板特征或追问澄清一缺指代，不直接把歧义绑定到当前桌": {"inbound", "board", "reply"},
}
_REVIEW_INSTRUCTIONS = (
    "只判实际送达的 reply。board、table-read、personal-read 是已核实的独立事实，"
    "禁止因答复没有提到后台查询动作而声称没有查原桌。"
    "用户说的『一缺』可指只差一人，也可能是对一人已经报名的简写。看板公示的桌是 1 号、"
    "人齐开、川麻，现有一人并且尚缺三席。合格回复可以问是不是那张人齐开的 1 号川麻桌，"
    "也可以清楚用『如果说的是……』区别当前可核实的原桌和另一种解释。"
    "直接说『你说的一缺那桌就是这桌』让歧义继续存在，纵使余位说对也不通过指代项。"
    "另查的七点后一元红中暂时没有，不可把结果盖到川麻原桌、声称登记或保证后续通知。"
    "自然口语和语序均可。每项请按下列顺序引用该项所需的证据，可再引其他证据。"
    + " ".join(f"{check}：{', '.join(sorted(ids))}。" for check, ids in _CHECKS.items())
)


def _source() -> ReplaySource:
    """用检查点的条件重建合成邻例，不依赖原聊天或运行归档。"""
    sent_at = datetime(2026, 11, 26, 18, 35, tzinfo=DEFAULT_TZ)
    game = {
        "organizer_id": "mixed-board-host",
        "requirement": {"game_type": "sichuan_mahjong", "stake_range": {"min": 2, "max": 3},
                        "smoke_preference": "no_smoking", "start_time_kind": "asap_when_full"},
    }
    message = MessageEnvelope(
        "customer:synthetic-mixed-board", "mixed-board-guest",
        "刚才群里说一缺的那桌，是川麻吗？另外今晚七点以后有一块底的红中吗？",
        sent_at, "private", AGENT_ID,
    )
    return ReplaySource(42, {"mixed-board-host": "小夏", "mixed-board-guest": "小陆"}, VENUE,
                        game, (message,), source_kind=_CASE)


def _seed_board_game(store: SQLiteAgentStore, channel: SandboxChannel, source: ReplaySource) -> str:
    """在正式门店库预置一人局并由建局事务安排看板。"""
    organizer = source.initial_game["organizer_id"]
    with using_clock(lambda: source.turns[0].sent_at - timedelta(minutes=15)):
        game, _ = store.create_game(
            conversation_id=channel.private_conversation_id(organizer), organizer_id=organizer,
            organizer_name=source.identities[organizer], requirement=source.initial_game["requirement"],
            known_players=[{"customer_id": organizer, "status": "confirmed", "seat_count": 1}],
            trace_id=f"{_CASE}:fixture",
        )
    return game.game_id


def _business_state(store: SQLiteAgentStore) -> tuple:
    """从正式库分别读取牌局、预约、候补以及两类邀约稿。"""
    return (*_snapshot(store),
            {key: value.to_dict() for key, value in store.waiting_demands.items()},
            {key: value.to_dict() for key, value in store.invite_drafts.items()},
            {key: value.to_dict() for key, value in store.outbound_message_drafts.items()})


def _same_table(result: dict, game_id: str) -> bool:
    requirement = result["requirement"]
    if {"start_time_range", "planned_start_at", "start_time", "must_end_by"} & requirement.keys():
        return False
    return any(
        game["game_id"] == game_id and game["remaining_seats"] == 3
        and game["requirement"]["game_type"] == "sichuan_mahjong"
        and game["requirement"]["start_time_kind"] == "asap_when_full"
        for match in result["matches"] if (game := match.get("game"))
    )


def _evening_red_center(result: dict, message: MessageEnvelope) -> bool:
    requirement = result["requirement"]
    window = requirement.get("start_time_range")
    if (requirement.get("game_type") != "red_center_mahjong" or set(stake_options(requirement)) != {(1, 1, None)}
            or not window or any(result.get(field) for field in ("matches", "alternatives", "nearby_options"))):
        return False
    start, end = (datetime.fromisoformat(window[key]) for key in ("start_at", "end_at"))
    seven = message.sent_at.replace(hour=19, minute=0, second=0, microsecond=0)
    return start == seven and start < end <= seven.replace(hour=0) + timedelta(days=1)


def _review(channel, message, board_row, table_read, personal_read, response, judge, events_path) -> dict:
    """给独立评审只读的限定证据，并核对每个有效判决实际引用了相应范围。"""
    visible_replies = {row["message_id"]: row for row in channel.events_for(message.sender_id)
                       if row.get("source_message_id") == message.message_id
                       and row["kind"] == "private" and row["sender_id"] == AGENT_ID
                       and row["recipient_id"] == message.sender_id}
    replied = [row for row in response["archive_outputs"] if row["message_id"] in visible_replies
               and visible_replies[row["message_id"]]["text"] == response["final_reply"]]
    if len(replied) != 1:
        return {"status": "missing_reply", "verdict": "inconclusive"}
    table_result = table_read["result"]
    personal_result = personal_read["result"]
    evidence = [
        {"message_id": "inbound", "speaker": "customer", "text": message.text},
        {"message_id": "board", "speaker": "agent_board", "text": board_row["text"]},
        {"message_id": "table-read", "lookup_scope": "specific_table", "tool_call_id": table_read["call_id"],
         "confirmed_board_item": 1, "game_type": "sichuan_mahjong", "start_time_kind": "asap_when_full",
         "remaining_seats": 3, "match_count": len(table_result["matches"])},
        {"message_id": "personal-read", "lookup_scope": "personal", "tool_call_id": personal_read["call_id"],
         "requirement": personal_result["requirement"], "match_count": len(personal_result["matches"])},
        {"message_id": "reply", "speaker": "agent", "text": replied[0]["text"]},
    ]
    result = judge.judge(_CASE, evidence, trace_id=f"judge:{_CASE}", checks=tuple(_CHECKS),
                         instructions=_REVIEW_INSTRUCTIONS)
    payload = result.to_dict()
    events_path.with_name("mixed_board_judgment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    citations_valid = result.status == "valid" and all(
        finding.verdict == "inconclusive" or _CHECKS[finding.check] <= set(finding.evidence_ids)
        for finding in result.findings
    )
    return {"status": result.status, "verdict": result.verdict if citations_valid else "inconclusive",
            "citations_valid": citations_valid,
            "findings": {finding.check: finding.verdict for finding in result.findings}}


def _execute(store, channel, source, game_id, events_path, *, judge: SemanticJudge | None = None) -> dict:
    """从实际群看板与私聊入口验证两个查询范围、收件及无业务写入。"""
    message, = source.turns
    board = VenueRollingBoard(channel)
    due = board.next_due()
    if due is None or due >= message.sent_at:
        return {"status": "failed", "checks": {"board_delivered": False}}
    publication = board.publish_due(due)
    visible = channel.events_for(message.sender_id)
    published = [(row, snapshot) for row, snapshot in board._verified_boards(visible)
                 if row in publication["archive_outputs"]
                 and any(item.item_no == 1 and item.game_id == game_id for item in snapshot.items)]
    if publication["status"] != "completed" or len(published) != 1:
        return {"status": "failed", "checks": {"board_delivered": False}}

    board_row, _ = published[0]
    before = _business_state(store)
    response = _deliver_replay_message(channel, events_path, _CASE, message)
    tools = _successful_tools(response)
    searches = [tool for tool in tools if tool["name"] == "search_current_games"]
    named = [tool for tool in searches if tool["result"].get("lookup_scope") == "specific_table"]
    personal = [tool for tool in searches if tool["result"].get("lookup_scope") == "personal"]
    named_ok = bool(named) and all(_same_table(tool["result"], game_id) for tool in named)
    personal_ok = bool(personal) and all(_evening_red_center(tool["result"], message) for tool in personal)
    ids = {tool.get("call_id") for tool in named + personal}
    outbound = [row for row in channel.events() if row.get("source_message_id") == message.message_id]
    reopened = SQLiteAgentStore(store.path)
    try:
        registry = channel.runtime.tool_gateway.tools
        checks = {
            "board_delivered": True,
            "private_reply_delivered": _delivered_reply(response, message) and bool(outbound),
            "only_original_customer_received_the_reply": bool(outbound) and all(
                row["kind"] == "private" and row["recipient_id"] == message.sender_id
                and row["conversation_id"] == channel.private_conversation_id(message.sender_id)
                for row in outbound),
            "same_table_found_without_personal_window": named_ok,
            "separate_red_center_evening_search_found_no_match": personal_ok,
            "queries_have_separate_scoped_call_ids": (
                bool(named and personal) and len(searches) == len(named + personal) == len(ids) and None not in ids
            ),
            "no_successful_write_tools": all(
                tool["name"] == CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME
                or tool["name"] in registry and registry[tool["name"]].execution_mode == "read_only"
                for tool in tools
            ),
            "game_room_waiting_and_invite_state_unchanged": _business_state(reopened) == before,
        }
    finally:
        reopened._connection.close()
    hard = all(checks.values())
    result = (_review(channel, message, board_row, named[-1], personal[-1], response, judge, events_path)
              if hard and judge is not None else {"status": "not_run", "verdict": "inconclusive"})
    verdict = result["verdict"]
    return {
        "status": "failed" if not hard or verdict == "fail" else "passed" if verdict == "pass" else "inconclusive",
        "checks": checks, "hard_checks_passed": hard, "semantic_judgment": result,
        "fixture_source": {"kind": "synthetic_sibling", "reference_run": _REFERENCE_RUN, "reference_inbound_sequence": 87},
    }


def main() -> int:
    """使用当前工作树生产运行时和 OpenCode，归档到独立目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runtime_data" / "venue_simulation" / "mixed_board_replays")
    args = parser.parse_args()
    run_dir = args.runs_dir / f"replay-{uuid.uuid4().hex[:10]}"
    result = _run_venue_replay(_source, _execute, run_dir, kind=_CASE, planner_low=False, clarify_consent=False,
                               review_clarifications=True, seed_store=_seed_board_game)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
