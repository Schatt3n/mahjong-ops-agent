#!/usr/bin/env python3
"""Run phase-one group-session scenarios against the configured real model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mahjong_agent_runtime import InMemoryAgentStore, OpenAICompatibleAgentClient  # noqa: E402
from mahjong_agent_runtime.env import load_dotenv_defaults  # noqa: E402
from mahjong_agent_runtime.group_chat import (  # noqa: E402
    GroupMessage,
    GroupSessionClassifier,
    GroupSessionPipeline,
    MessageAccumulator,
    OwnerMessageParser,
    QuickFilter,
    SessionCrystallizer,
    SessionRouter,
)


TZ = ZoneInfo("Asia/Shanghai")
OWNER_BOARD = """cq371 人齐开 1块无烟
cq272 18.30 1块无烟

红中173 18.00 无烟3爆
红中272 19.00 无烟 568
红中173 22.30 368"""


def group_message(text: str, *, sender: str, message_id: str, sent_at: datetime) -> GroupMessage:
    return GroupMessage(
        room_id="live-simulation-room",
        conversation_id="wechaty:room:live-simulation-room",
        sender_external_id=sender,
        sender_name=sender,
        text=text,
        message_id=message_id,
        sent_at=sent_at,
    )


def make_pipeline(client, *, on_crystallized=None) -> tuple[InMemoryAgentStore, GroupSessionPipeline]:
    store = InMemoryAgentStore()
    pipeline = GroupSessionPipeline(
        store=store,
        owner_parser=OwnerMessageParser(owner_external_ids={"owner"}),
        quick_filter=QuickFilter(),
        accumulator=MessageAccumulator(quiet_seconds=0.1, continuation_seconds=120),
        session_router=SessionRouter(),
        crystallizer=SessionCrystallizer(on_crystallized=on_crystallized),
        classifier=GroupSessionClassifier(client, timeout_seconds=30),
    )
    return store, pipeline


def submit_and_flush(
    pipeline: GroupSessionPipeline,
    text: str,
    *,
    sender: str,
    message_id: str,
    sent_at: datetime,
    trace_id: str,
):
    pipeline.accept(group_message(text, sender=sender, message_id=message_id, sent_at=sent_at), trace_id=trace_id)
    return pipeline.flush_due(at=sent_at + timedelta(seconds=1))[0]


def run(env_file: Path) -> dict:
    load_dotenv_defaults(env_file)
    client = OpenAICompatibleAgentClient.from_env()
    if client is None:
        raise RuntimeError(f"no model configuration found after loading {env_file}")
    stamp = datetime.now(TZ).replace(microsecond=0)
    store, pipeline = make_pipeline(client)
    owner_result = pipeline.accept(
        group_message(OWNER_BOARD, sender="owner", message_id="board", sent_at=stamp),
        trace_id="trace_group_session_owner",
    )
    cases = []

    pipeline.accept(
        group_message("红中 568 我打", sender="德不孤", message_id="claim-1", sent_at=stamp + timedelta(seconds=1)),
        trace_id="trace_group_session_unique",
    )
    pipeline.accept(
        group_message("7点", sender="德不孤", message_id="claim-2", sent_at=stamp + timedelta(seconds=1.05)),
        trace_id="trace_group_session_unique",
    )
    cases.extend(pipeline.flush_due(at=stamp + timedelta(seconds=2)))

    pipeline.accept(
        group_message("1块无烟的我来", sender="someone", message_id="ambiguous", sent_at=stamp + timedelta(seconds=3)),
        trace_id="trace_group_session_ambiguous",
    )
    cases.extend(pipeline.flush_due(at=stamp + timedelta(seconds=4)))

    pipeline.accept(
        group_message("我以为是群主搞的高科技", sender="ninet", message_id="chat", sent_at=stamp + timedelta(seconds=5)),
        trace_id="trace_group_session_chat",
    )
    cases.extend(pipeline.flush_due(at=stamp + timedelta(seconds=6)))

    filtered = pipeline.accept(
        group_message("哈哈哈哈哈哈", sender="ninet-2", message_id="noise", sent_at=stamp + timedelta(seconds=7)),
        trace_id="trace_group_session_noise",
    )
    board = store.get_group_board_state("live-simulation-room")

    phase_two_store, phase_two = make_pipeline(client)
    demand = submit_and_flush(
        phase_two,
        "川麻换三 6.30 132 三缺一",
        sender="元宝",
        message_id="demand",
        sent_at=stamp + timedelta(minutes=10),
        trace_id="trace_group_session_demand",
    )
    merge_outputs = []
    for index, (sender, text, minute) in enumerate(
        (("A", "有人今晚打吗", 20), ("B", "+1", 21), ("A", "7点1块无烟", 25))
    ):
        merge_outputs.append(
            submit_and_flush(
                phase_two,
                text,
                sender=sender,
                message_id=f"merge-{index}",
                sent_at=stamp + timedelta(minutes=minute),
                trace_id=f"trace_group_session_merge_{index}",
            )
        )
    active_merged = [
        item
        for item in phase_two.session_router.list_sessions("live-simulation-room")
        if item.status == "active" and {message.message_id for message in item.messages} >= {"merge-0", "merge-1", "merge-2"}
    ]

    crystallized_ids: list[str] = []
    _, crystal_pipeline = make_pipeline(client, on_crystallized=lambda session: crystallized_ids.append(session.id))
    for index, (sender, text) in enumerate(
        (("A", "有人打吗"), ("B", "+1"), ("C", "我也来"), ("A", "7点1块无烟"), ("B", "行"), ("C", "ok"))
    ):
        submit_and_flush(
            crystal_pipeline,
            text,
            sender=sender,
            message_id=f"crystal-{index}",
            sent_at=stamp + timedelta(minutes=30 + index),
            trace_id=f"trace_group_session_crystal_{index}",
        )
    serialized = [
        {
            "action": item.action,
            "session_id": item.session_id,
            "classification": item.classification.to_dict() if item.classification else None,
        }
        for item in [*cases, demand, *merge_outputs]
    ]
    checks = {
        "owner_board_has_five_items": owner_result.action == "board_replaced" and board is not None and len(board.items) == 5,
        "unique_claim_matches_four": bool(
            len(cases) > 0
            and cases[0].classification
            and cases[0].classification.intent == "claim"
            and cases[0].classification.matched_board_no == 4
            and cases[0].action == "private_switch"
        ),
        "ambiguous_claim_is_not_forced": bool(
            len(cases) > 1
            and cases[1].classification
            and cases[1].classification.intent == "claim"
            and cases[1].classification.matched_board_no is None
            and cases[1].classification.confidence < 0.7
        ),
        "chitchat_is_ignored": bool(
            len(cases) > 2
            and cases[2].classification
            and cases[2].classification.intent == "chitchat"
            and cases[2].action == "ignore"
        ),
        "laughter_skips_model_path": filtered.action == "filtered",
        "new_demand_routes_private": bool(
            demand.classification
            and demand.classification.intent == "new_demand"
            and demand.action == "private_switch"
        ),
        "related_sessions_merge": len(active_merged) == 1 and len(active_merged[0].messages) == 3,
        "stable_formation_crystallizes_once": len(set(crystallized_ids)) == 1,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "outcomes": serialized,
        "active_merged_sessions": [item.to_context() for item in active_merged],
        "crystallized_session_ids": crystallized_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path.home() / ".local/share/mahjong-ops-agent/.env",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=ROOT / "runtime_data/group_session_live_simulation_report.json",
    )
    args = parser.parse_args()
    result = run(args.env_file.expanduser())
    report_path = args.report_path.expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result["report_path"] = str(report_path)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
