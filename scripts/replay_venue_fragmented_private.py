#!/usr/bin/env python3
"""把历史私聊里的下午、无烟询局拆成独立入站，与一条消息做受控对照。"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from mahjong_agent_runtime.customer_visible_contract import CUSTOMER_EXECUTION_FAILURE_REPLY  # noqa: E402
from mahjong_agent_runtime.domains.game_domain import current_game_query_start_range  # noqa: E402
from mahjong_agent_runtime.domains.value_utils import normalize_smoke_preference  # noqa: E402
from mahjong_agent_runtime.models import DEFAULT_TZ, ConversationRole, using_clock  # noqa: E402
from scripts.replay_venue_consent import (  # noqa: E402
    ReplaySource, _deliver_replay_message, _delivered_reply, _run_venue_replay, _successful_tools,
)
from scripts.replay_venue_mixed_board import _business_state  # noqa: E402
from tests.simulation.venue_channel import AGENT_ID, MessageEnvelope  # noqa: E402
from tests.simulation.venue_private_history_seed import (  # noqa: E402
    DEFAULT_PRIVATE_HISTORY_CASE_ID, DEFAULT_PRIVATE_HISTORY_DATASET, DEFAULT_PRIVATE_HISTORY_EPISODE_ID,
)
from tests.simulation.venue_scenario import VENUE  # noqa: E402


_KIND = "derived_private_availability_bubbles"
_CUSTOMER = "fragmented-guest"
_BUBBLES = ("老板我问下哈", "我下午可以打", "想打无烟的", "三点到五点开的有位置吗？")
_GAP = timedelta(seconds=5)
_CHECKS = ("答复保留下午三至五点开场和无烟这两个条件", "告知真实现成桌且未声称已占位或建局")


def _source(*, single_message: bool = False) -> ReplaySource:
    """以现有真人项目需求衍生短气泡，便携回放不读取期望后续或未确认的私聊。"""
    at = datetime(2026, 11, 26, 12, 29, tzinfo=DEFAULT_TZ)
    identities = {_CUSTOMER: "模拟顾客", "host-match": "牌友甲", "host-smoke": "牌友乙", "host-time": "牌友丙"}
    texts = ("\n".join(_BUBBLES),) if single_message else _BUBBLES
    turns = tuple(MessageEnvelope(f"customer:fragmented:{index + 1}", _CUSTOMER, text,
                                  at + index * _GAP, "private", AGENT_ID)
                  for index, text in enumerate(texts))
    return ReplaySource(42, identities, VENUE, {}, turns, source_kind=_KIND)


def _seed_tables(store, channel, source) -> str:
    """放一桌符合范围的无烟桌，并用同档有烟桌和晚场无烟桌检验条件实际生效。"""
    at = source.turns[0].sent_at
    ids = {}
    with using_clock(lambda: at - timedelta(minutes=15)):
        for host, hour, minute, smoke in (
            ("host-match", 16, 0, "no_smoking"),
            ("host-smoke", 16, 0, "smoking"),
            ("host-time", 19, 0, "no_smoking"),
        ):
            start = at.replace(hour=hour, minute=minute)
            game, _ = store.create_game(
                conversation_id=channel.private_conversation_id(host), organizer_id=host,
                organizer_name=source.identities[host], trace_id=f"{_KIND}:{host}",
                requirement={"game_type": "hangzhou_mahjong", "stake": "1", "start_time_kind": "scheduled",
                             "planned_start_at": start.isoformat(), "smoke_preference": smoke},
                known_players=[{"customer_id": host, "status": "confirmed", "seat_count": 1}],
            )
            ids[host] = game.game_id
    return ids["host-match"]


def _matching_read(result: dict, *, at: datetime, game_id: str) -> bool:
    """有效查局覆盖完整开局区间，且有烟和错时的两桌都没有进入选项。"""
    requirement = result["requirement"]
    bounds = current_game_query_start_range(requirement)
    return (bounds == (at.replace(hour=15, minute=0, second=0), at.replace(hour=17, minute=0, second=0))
            and normalize_smoke_preference(requirement.get("smoke_preference")) == "no_smoking"
            and {item["game"]["game_id"] for item in result["matches"]} == {game_id}
            and not any(result.get(field) for field in ("nearby_options", "alternatives")))


def _judge_reply(judge, source, read, reply, events_path) -> dict:
    """把每条实际来话及可信查局交给独立评审，判决须引用查局和正式答复。"""
    evidence = [{"message_id": f"bubble-{index}", "role": "customer", "text": bubble.text}
                for index, bubble in enumerate(source.turns, 1)]
    evidence.extend(({"message_id": "actual-read", "requirement": read["requirement"],
                      "matched_openings": [option["game"].get("planned_start_at") for option in read["matches"]]},
                     {"message_id": "actual-reply", "text": reply["text"]}))
    result = judge.judge(_KIND, evidence, trace_id=f"{_KIND}:judge", checks=_CHECKS,
        instructions=("依次理解所有顾客气泡。顾客只问现成桌，最后定位下午三到五点开的无烟局。"
                      "实际匹配是下午四点开、无烟、1元杭麻。允许自然省略顾客已经说过的条件。"
                      "每项明确判决均引用 actual-read、actual-reply，并补充支持结论的相关 bubble 编号。"))
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"phase": "independent_judge", "result": result.to_dict()}, ensure_ascii=False) + "\n")
    citations_valid = result.status == "valid" and all(
        item.verdict == "inconclusive" or {"actual-read", "actual-reply"} <= set(item.evidence_ids)
        for item in result.findings
    )
    return {"status": result.status, "verdict": result.verdict if citations_valid else "inconclusive",
            "findings": {item.check: item.verdict for item in result.findings}, "citations_valid": citations_valid}


def _execute(store, channel, source, game_id, events_path, *, judge=None) -> dict:
    """按虚拟时间驱动消息和到期调度，硬验收实际查询、交付和各轮业务状态。"""
    initial = _business_state(store)
    source_ids = [bubble.message_id for bubble in source.turns]
    responses = []
    read_only_states = []

    def dispatch_due(at):
        """使用生产批次的到期时刻触发真实路由，另存回放调度的回执。"""
        deliveries = channel.dispatch_due_private_inputs(at=at)
        with events_path.open("a", encoding="utf-8") as handle:
            for delivery in deliveries:
                handle.write(json.dumps({"phase": "quiet_deadline", "virtual_at": at.isoformat(),
                                         "response": delivery}, ensure_ascii=False) + "\n")
        responses.extend(deliveries)
        read_only_states.append(_business_state(store) == initial)

    for bubble in source.turns:
        dispatch_due(bubble.sent_at)
        responses.append(_deliver_replay_message(channel, events_path, "incoming_bubble", bubble))
        read_only_states.append(_business_state(store) == initial)
    last = source.turns[-1]
    conversation_id = channel.private_conversation_id(last.sender_id)
    pending = store.pending_input_batch(conversation_id, last.sender_id)
    if pending is not None and pending.status.value == "pending":
        dispatch_due(pending.quiet_deadline)

    inbound = [row for row in channel.events() if row["direction"] == "inbound"]
    private_turns = [turn for turn in store.recent_turns(conversation_id, limit=40)
                     if turn.role == ConversationRole.USER]
    windows = [turn.metadata["input_window"]["source_message_ids"] for turn in private_turns
               if isinstance(turn.metadata.get("input_window"), dict)]
    runtime_ids = list(dict.fromkeys(fragment_id for window in windows for fragment_id in window))
    final_responses = [response for response in responses if _delivered_reply(response, last)]
    final = final_responses[-1] if final_responses else {}
    reply = next((row for row in reversed(final.get("archive_outputs", ()))
                  if row.get("source_message_id") == last.message_id and row.get("recipient_id") == last.sender_id
                  and row.get("text") == final.get("final_reply") and row["text"] != CUSTOMER_EXECUTION_FAILURE_REPLY), None)
    reads = [tool["result"] for tool in _successful_tools(final) if tool["name"] == "search_current_games"]
    read = next((result for result in reversed(reads) if _matching_read(result, at=last.sent_at, game_id=game_id)), None)
    checks = {
        "raw_bubbles_kept_separate_and_ordered":
            [(row["message_id"], row["text"], datetime.fromisoformat(row["sent_at"])) for row in inbound]
            == [(bubble.message_id, bubble.text, bubble.sent_at) for bubble in source.turns],
        "all_business_bubbles_reached_runtime_in_order": runtime_ids[-min(3, len(source_ids)):] == source_ids[-3:],
        "final_query_keeps_both_late_constraints": read is not None,
        "actual_final_reply_received_by_only_the_requester": bool(reply and datetime.fromisoformat(reply["sent_at"]) >= last.sent_at)
            and all(row["kind"] == "private" and row["recipient_id"] == last.sender_id
                    for row in channel.events() if row.get("source_message_id") in source_ids and row["direction"] == "outbound"),
        "read_only_at_every_step": all(read_only_states),
    }
    hard = all(checks.values())
    semantic = (_judge_reply(judge, source, read, reply, events_path) if hard and judge is not None
                else {"status": "not_run", "verdict": "inconclusive"})
    verdict = semantic["verdict"]
    return {"status": "failed" if not hard or verdict == "fail" else "passed" if verdict == "pass" else "inconclusive",
            "checks": checks, "hard_checks_passed": hard, "semantic_judgment": semantic,
            "input_bubble_count": len(source.turns), "input_window_message_ids": windows,
            "fixture_source": {"kind": "derived_not_original", "dataset": DEFAULT_PRIVATE_HISTORY_DATASET.name,
                               "case_id": DEFAULT_PRIVATE_HISTORY_CASE_ID, "episode_id": DEFAULT_PRIVATE_HISTORY_EPISODE_ID,
                               "source_turns": [1, 3], "interval_seconds": int(_GAP.total_seconds()) if len(source.turns) > 1 else None}}


def main() -> int:
    """在独立 SQLite 和通道账本中跑一次真实模型，原始源证据保持独立。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-message", action="store_true", help="相同正文合在一个微信气泡，作为送达形态对照")
    parser.add_argument("--runs-dir", type=Path,
                        default=ROOT / "runtime_data" / "venue_simulation" / "fragmented_private_replays")
    args = parser.parse_args()
    run_dir = args.runs_dir / f"replay-{uuid.uuid4().hex[:10]}"
    report = _run_venue_replay(lambda: _source(single_message=args.single_message), _execute, run_dir, kind=_KIND,
                               planner_low=False, clarify_consent=False, review_clarifications=True,
                               seed_store=_seed_tables, production_private_aggregation=True)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
