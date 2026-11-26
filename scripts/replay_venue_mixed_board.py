#!/usr/bin/env python3
"""在 OpenCode 沙盒回放按群看板编号问桌、另查红中的私聊。"""

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
from mahjong_agent_runtime.domains.temporal import OVERNIGHT_SEARCH_END_HOUR  # noqa: E402
from mahjong_agent_runtime.models import DEFAULT_TZ, using_clock  # noqa: E402
from scripts.replay_venue_consent import (  # noqa: E402
    ReplaySource, _deliver_replay_message, _delivered_reply, _run_venue_replay, _snapshot, _successful_tools,
)
from tests.simulation.venue_actors import SemanticJudge  # noqa: E402
from tests.simulation.venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel  # noqa: E402
from tests.simulation.venue_rolling_board import VenueRollingBoard  # noqa: E402
from tests.simulation.venue_scenario import VENUE  # noqa: E402

_AMBIGUOUS_REFERENCE_CASE = "synthetic_mixed_board_seq87_sibling"
_CLEAR_REFERENCE_CASE = "synthetic_mixed_board_clear_reference_sibling"
_REFERENCE_RUN = "2026-11-25-real-34fa8e9813"
_RED_CENTER_RESULT = "单独如实告知今晚七点后目前没有符合条件的一元红中"
_RED_CENTER_OFFER = "得出完整空结果后，征询是否要帮顾客组一桌红中"
_COMMON_CHECKS = {
    _RED_CENTER_RESULT: {"personal-read", "reply"},
}
_CHECKS = _COMMON_CHECKS | {
    "按已有线索试探原桌，或如实说明还不确定顾客指哪桌": {"inbound", "board", "reply"},
}
_CLEAR_REFERENCE_CHECKS = _COMMON_CHECKS | {
    "直接回答原桌及其真实公开情况，无需让顾客再确认是哪桌": {"inbound", "board", "table-read", "reply"},
    _RED_CENTER_OFFER: {"inbound", "personal-read", "reply"},
}
_RESULT_INSTRUCTIONS = (
    "personal-read 证明今晚七点至次日六点的一元红中目前没有符合条件的现成局。"
    "用店家的口吻给出这项库存结论，不能以等待、技术故障、转人工或笼统问还要什么取代。"
    "『暂时没有合适的』等自然说法可以通过；只说『这轮没查到』没有给出库存结论。"
    "结合来话判断所指，可省略重复的已知条件；不能说成全店都没有红中、川麻桌也没有，或以后一定没有。"
)
_REVIEW_INSTRUCTIONS = (
    "只判实际送达的 reply。board 是正式看板，table-read 若存在则是核实的实时桌况。"
    "用户说的『一缺』可指只差一人，也可能是对一人已经报名的简写。看板公示的桌是 1 号、"
    "人齐开、川麻，现有一人并且尚缺三席。允许用已知看板特征试探，例如『你说的是人齐开的那桌吗？』，"
    "也允许说『我没听明白你指哪桌』。是否再提问由对话决定，无须先挑一个候选或使用固定句式。"
    "即使猜测后来有偏差，只要回复把它说成可能性、候选桌况有依据，就按当时的处理评判。"
    "没有可靠定位却断言『你说的一缺那桌就是这桌』才在指代项判失败。仍须另行回答红中有无。"
    + _RESULT_INSTRUCTIONS
    + "给出空结果后可征询是否帮忙凑桌，不可声称登记或保证后续通知。每项须引用对应证据，可再引其他证据。"
    + " ".join(f"{check}：{', '.join(sorted(ids))}。" for check, ids in _CHECKS.items())
)
_CLEAR_REFERENCE_INSTRUCTIONS = (
    "只判实际送达的 reply。board、table-read、personal-read 是已经核实的事实。"
    "顾客通过群里的桌号和人齐开指向正式看板上的1号桌。直接答是否是川麻即可。"
    "可自然补充还缺三位，不需要复述全部公开特征。如果提及其他桌况，必须与事实一致。"
    "若仍让顾客确认原桌身份或让他等核对，原桌答复项不通过。"
    + _RESULT_INSTRUCTIONS
    + "空结果与后续服务提议分别检查。确认目前没有合适的红中后，像店家一样问要不要帮忙组一桌。"
    "询问是否有其他需求不算该提议；不可声称已经开始凑、已登记、正持续留意或承诺后续通知。"
    "可合在同一条消息，句式和顺序不限。每项须引用对应证据，可再引其他证据。"
    + " ".join(f"{check}：{', '.join(sorted(ids))}。" for check, ids in _CLEAR_REFERENCE_CHECKS.items())
)


def _source(*, ambiguous_reference: bool = False) -> ReplaySource:
    """日常询问使用看板编号，原合成问法保留用于歧义压力测试。"""
    sent_at = datetime(2026, 11, 26, 18, 35, tzinfo=DEFAULT_TZ)
    game = {
        "organizer_id": "mixed-board-host",
        "requirement": {"game_type": "sichuan_mahjong", "stake_range": {"min": 2, "max": 3},
                        "smoke_preference": "no_smoking", "start_time_kind": "asap_when_full"},
    }
    if ambiguous_reference:
        case, message_id = _AMBIGUOUS_REFERENCE_CASE, "customer:synthetic-mixed-board"
        text = "刚才群里说一缺的那桌，是川麻吗？另外今晚七点以后有一块底的红中吗？"
    else:
        case, message_id = _CLEAR_REFERENCE_CASE, "customer:synthetic-mixed-board-clear"
        text = "老板，群里1号那桌人齐开的，是川麻吗？今晚七点以后有一块底的红中吗？"
    message = MessageEnvelope(message_id, "mixed-board-guest", text, sent_at, "private", AGENT_ID)
    venue = {**VENUE, "closes": f"{OVERNIGHT_SEARCH_END_HOUR:02d}:00"}
    return ReplaySource(42, {"mixed-board-host": "小夏", "mixed-board-guest": "小陆"}, venue,
                        game, (message,), source_kind=case)


def _seed_board_game(store: SQLiteAgentStore, channel: SandboxChannel, source: ReplaySource) -> str:
    """在正式门店库预置一人局并由建局事务安排看板。"""
    organizer = source.initial_game["organizer_id"]
    with using_clock(lambda: source.turns[0].sent_at - timedelta(minutes=15)):
        game, _ = store.create_game(
            conversation_id=channel.private_conversation_id(organizer), organizer_id=organizer,
            organizer_name=source.identities[organizer], requirement=source.initial_game["requirement"],
            known_players=[{"customer_id": organizer, "status": "confirmed", "seat_count": 1}],
            trace_id=f"{source.source_kind}:fixture",
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
    closing = seven.replace(hour=OVERNIGHT_SEARCH_END_HOUR) + timedelta(days=1)
    return start == seven and end == closing


def _review(channel, message, board_row, table_read, personal_read, response, judge, events_path, *, case: str) -> dict:
    """给独立评审只读的限定证据，并核对每个有效判决实际引用了相应范围。"""
    visible_replies = {row["message_id"]: row for row in channel.events_for(message.sender_id)
                       if row.get("source_message_id") == message.message_id
                       and row["kind"] == "private" and row["sender_id"] == AGENT_ID
                       and row["recipient_id"] == message.sender_id}
    replied = [row for row in response["archive_outputs"] if row["message_id"] in visible_replies
               and visible_replies[row["message_id"]]["text"] == response["final_reply"]]
    if len(replied) != 1:
        return {"status": "missing_reply", "verdict": "inconclusive"}
    personal_result = personal_read["result"]
    evidence = [
        {"message_id": "inbound", "speaker": "customer", "text": message.text},
        {"message_id": "board", "speaker": "agent_board", "text": board_row["text"]},
        {"message_id": "personal-read", "lookup_scope": "personal", "tool_call_id": personal_read["call_id"],
         "requirement": personal_result["requirement"], "match_count": len(personal_result["matches"])},
        {"message_id": "reply", "speaker": "agent", "text": replied[0]["text"]},
    ]
    if table_read is not None:
        evidence.insert(2, {
            "message_id": "table-read", "lookup_scope": "specific_table", "tool_call_id": table_read["call_id"],
            "confirmed_board_item": 1, "game_type": "sichuan_mahjong", "start_time_kind": "asap_when_full",
            "remaining_seats": 3, "match_count": len(table_read["result"]["matches"]),
        })
    checks, instructions = ((_CLEAR_REFERENCE_CHECKS, _CLEAR_REFERENCE_INSTRUCTIONS)
                            if case == _CLEAR_REFERENCE_CASE else (_CHECKS, _REVIEW_INSTRUCTIONS))
    result = judge.judge(case, evidence, trace_id=f"judge:{case}", checks=tuple(checks), instructions=instructions)
    payload = result.to_dict()
    events_path.with_name("mixed_board_judgment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    citations_valid = result.status == "valid" and all(
        finding.verdict == "inconclusive" or checks[finding.check] <= set(finding.evidence_ids)
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
    response = _deliver_replay_message(channel, events_path, source.source_kind, message)
    tools = _successful_tools(response)
    searches = [tool for tool in tools if tool["name"] == "search_current_games"]
    named = [tool for tool in searches if tool["result"].get("lookup_scope") == "specific_table"]
    personal = [tool for tool in searches if tool["result"].get("lookup_scope") == "personal"]
    ambiguous = source.source_kind == _AMBIGUOUS_REFERENCE_CASE
    personal_ok = bool(personal) and all(_evening_red_center(tool["result"], message) for tool in personal)
    ids = {tool.get("call_id") for tool in named + personal}
    outbound = [row for row in channel.events() if row.get("source_message_id") == message.message_id]
    reopened = SQLiteAgentStore(store.path)
    try:
        registry = channel.runtime.tool_gateway.tools
        checks = {
            "board_delivered": True,
            "private_reply_delivered": _delivered_reply(response, message) and any(
                row["text"] == response.get("final_reply") for row in outbound
            ),
            "only_original_customer_received_the_reply": bool(outbound) and all(
                row["kind"] == "private" and row["recipient_id"] == message.sender_id
                and row["conversation_id"] == channel.private_conversation_id(message.sender_id)
                for row in outbound),
            "separate_red_center_evening_search_found_no_match": personal_ok,
            "queries_have_separate_scoped_call_ids": (
                bool(personal) and len(searches) == len(named + personal) == len(ids) and None not in ids
            ),
            "no_successful_write_tools": all(
                tool["name"] == CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME
                or tool["name"] in registry and registry[tool["name"]].execution_mode == "read_only"
                for tool in tools
            ),
            "game_room_waiting_and_invite_state_unchanged": _business_state(reopened) == before,
        }
        if not ambiguous or named:
            checks["same_table_found_without_personal_window"] = bool(named) and all(
                _same_table(tool["result"], game_id) for tool in named
            )
    finally:
        reopened._connection.close()
    hard = all(checks.values())
    result = (_review(channel, message, board_row, named[-1] if named else None, personal[-1], response, judge, events_path,
                      case=source.source_kind)
              if hard and judge is not None else {"status": "not_run", "verdict": "inconclusive"})
    verdict = result["verdict"]
    return {
        "status": "failed" if not hard or verdict == "fail" else "passed" if verdict == "pass" else "inconclusive",
        "checks": checks, "hard_checks_passed": hard, "semantic_judgment": result,
        "fixture_source": {
            "kind": "synthetic_sibling", "reference_run": _REFERENCE_RUN, "reference_inbound_sequence": 87,
            "scenario": "ambiguous_reference_stress" if ambiguous else "clear_reference",
        },
    }


def main() -> int:
    """使用当前工作树生产运行时和 OpenCode，归档到独立目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runtime_data" / "venue_simulation" / "mixed_board_replays")
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--ambiguous-reference", action="store_true", help="回放原来带『一缺』歧义的合成压力用例")
    variants.add_argument("--clear-reference", dest="ambiguous_reference", action="store_false",
                          help="回放按群里桌号询问的日常用例（默认）")
    args = parser.parse_args()
    run_dir = args.runs_dir / f"replay-{uuid.uuid4().hex[:10]}"
    source = _source(ambiguous_reference=args.ambiguous_reference)
    result = _run_venue_replay(lambda: source, _execute, run_dir, kind=source.source_kind,
                               planner_low=False, clarify_consent=False, review_clarifications=True,
                               seed_store=_seed_board_game)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
