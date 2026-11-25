#!/usr/bin/env python3
"""用冻结顾客原话和正式私聊入口，验收按指定时长查房及无时段默认查询。"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
from tempfile import mkdtemp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mahjong_agent_runtime import CustomerProfile, JsonlTraceRecorder, OpenAICompatibleAgentClient, SQLiteAgentStore
from mahjong_agent_runtime.env import load_dotenv_defaults
from mahjong_agent_runtime.models import using_clock
from scripts.replay_venue_consent import _delivered_reply, _deliver_replay_message, _snapshot, _successful_tools
from scripts.run_venue_simulation import _model_execution, _select_local_model
from tests.simulation.sim_adapter import build_runtime
from tests.simulation.venue_actors import SemanticJudge
from tests.simulation.venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel
from tests.simulation.venue_models import opencode_auxiliary


FIXTURE = ROOT / "eval/golden/venue_room_duration_replay.json"
CUSTOMER = "replay-room-customer"
REPLY_CHECK = "回复准确传达顾客所问时段的房态，且没有承诺或声称已经订房"


def replay_case(case: dict, path: Path, *, config, judge: SemanticJudge) -> dict:
    """起点占用房与半小时后才占用的房同时在库，确保用默认四小时会读到不同结果。"""
    path.mkdir()
    at = datetime.fromisoformat(case["at"])
    message = MessageEnvelope(case["id"], CUSTOMER, case["text"], at, "private", AGENT_ID)
    store = SQLiteAgentStore(path / "agent.sqlite3")
    with using_clock(lambda: at):
        store.upsert_customer(CustomerProfile(CUSTOMER, "顾客"))
        store.configure_rooms(["东二包间", "东三包间"])
        for room, offset in (("东二包间", 0), ("东三包间", 30)):
            start = at + timedelta(minutes=offset)
            store.reserve_room(conversation_id="fixture-reservation", game_id=None, room_id=room, start_at=start,
                               end_at=start + timedelta(hours=3), trace_id="fixture")
        runtime, _ = build_runtime(store, "real", seed=69, model_client=OpenAICompatibleAgentClient(config),
                                   trace_recorder=JsonlTraceRecorder(path / "agent_trace.log"), local_coordination=True)
        channel = SandboxChannel(runtime, room_id="room-duration-replay", identities={CUSTOMER: "顾客"},
                                 archive_path=path / "chat.sqlite3")
        before = _snapshot(store)
        response = _deliver_replay_message(channel, path / "events.jsonl", "room_inquiry", message)
        after = _snapshot(SQLiteAgentStore(store.path))
    tools = _successful_tools(response)
    queries = [item["result"] for item in tools if item["name"] == "check_room_availability"]
    expected_end = (at + timedelta(minutes=case["expected_minutes"])).isoformat()
    checks = {
        "all_room_queries_use_requested_window": bool(queries) and all(
            (item["start_at"], item["end_at"]) == (at.isoformat(), expected_end) for item in queries),
        "correct_inventory": bool(queries) and all(item["available_count"] == case["expected_available"] for item in queries),
        "no_business_write": before == after and not any(
            item["name"] in {"create_game", "reserve_room"} for item in tools),
        "private_reply_delivered": _delivered_reply(response, message),
    }
    chat = channel.events()
    (path / "chat.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chat), encoding="utf-8")
    evidence = [
        {"event_id": "inbound", "text": message.text, "expected_window_end": expected_end},
        {"event_id": "inventory", "queries": [
            {key: row[key] for key in ("start_at", "end_at", "available_count")} for row in queries],
         "reservation_changed": before != after},
        {"event_id": "reply", "text": [row["text"] for row in chat if row["direction"] == "outbound"]},
    ]
    judgment = judge.judge(case["id"], evidence, trace_id=f'judge:{case["id"]}', checks=(REPLY_CHECK,))
    status = ("failed" if not all(checks.values()) or judgment.verdict == "fail"
              else "passed" if judgment.status == "valid" and judgment.verdict == "pass" else "inconclusive")
    return {"id": case["id"], "status": status, "trace_id": response["trace_id"], "checks": checks,
            "model_execution": _model_execution(runtime.llm_client, timeout_seconds=runtime.llm_timeout_seconds),
            "judgment": judgment.to_dict()}


def main() -> int:
    """每个固定输入在独立新库执行，保留真实工具、投递、独立判定及 trace。"""
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case["id"] for case in cases])
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runtime_data/venue_simulation/room_duration_replay")
    args = parser.parse_args()
    if args.env_file:
        load_dotenv_defaults(args.env_file)
    config = _select_local_model("opencode-go", None, workers=1)
    judge = SemanticJudge(opencode_auxiliary(config, max_tokens=1200, workers=1))
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix="room-duration-", dir=args.runs_dir))
    print(output, flush=True)
    reports = []
    for case in cases:
        if args.case and args.case != case["id"]:
            continue
        report = replay_case(case, output / case["id"], config=config, judge=judge)
        reports.append(report)
        print(case["id"], report["status"], report["checks"], flush=True)
        (output / "report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(any(report["status"] != "passed" for report in reports))


if __name__ == "__main__":
    raise SystemExit(main())
