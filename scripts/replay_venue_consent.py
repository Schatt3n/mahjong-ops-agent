#!/usr/bin/env python3
"""在独立门店重放已归档的授权边界对话，并另做明确报名的对照。"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time, timedelta
import json
from pathlib import Path
import sys
from typing import Any
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mahjong_agent_runtime import JsonlTraceRecorder, SQLiteAgentStore  # noqa: E402
from mahjong_agent_runtime.llm import OpenAICompatibleAgentClient  # noqa: E402
from mahjong_agent_runtime.models import using_clock  # noqa: E402
from scripts.run_venue_simulation import _model_execution, _select_local_model  # noqa: E402
from tests.simulation.sim_adapter import build_runtime  # noqa: E402
from tests.simulation.venue_actors import SemanticJudge, generate_personas  # noqa: E402
from tests.simulation.venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel  # noqa: E402
from tests.simulation.venue_write_guard import VenueWriteGuard  # noqa: E402
from tests.simulation.venue_scenario import VENUE  # noqa: E402

_CONSENT_CONTROL = "我明确报名刚才说的那桌，麻烦给我留一个位置。"
_CONTROL_MESSAGE_ID = "customer:replay-consent-control"
_PORTABLE_FIXTURE = ROOT / "eval" / "golden" / "venue_join_consent_replay.json"
_PARTICIPATION_TOOLS = {"join_game", "record_candidate_reply"}
_INITIAL_REQUIREMENT_FIELDS = (
    "game_type", "stake", "smoke_preference", "start_time_kind", "start_time", "planned_start_at",
    "duration_kind", "duration_hours",
)


@dataclass(frozen=True, slots=True)
class ReplaySource:
    """归档中的初始门店、两条真实入站及该顾客当时收到的首轮回复。"""

    seed: int
    identities: dict[str, str]
    venue: dict[str, Any]
    initial_game: dict[str, Any]
    turns: tuple[MessageEnvelope, ...]
    source_kind: str = "archived"
    explicit_control: str = _CONSENT_CONTROL


def _read_rows(path: Path) -> list[dict]:
    """逐行读取运行目录中的本地证据。"""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_source(source_dir: Path) -> ReplaySource:
    """核对两轮的文本、收件人和先后因果与沙盒账本一致。"""
    report = json.loads((source_dir / "report.json").read_text(encoding="utf-8"))
    venue = json.loads((source_dir / "venue.json").read_text(encoding="utf-8"))
    cohort = report.get("cohort") or {}
    actors = [customer for customer, outcome in cohort.get("expected_outcomes", {}).items() if outcome == "formed"]
    if not actors and cohort.get("expected_outcome") == "formed":
        actors = list(report.get("acceptance", {}).get("conversations", {}))
    if report.get("scenario") != "formation" or len(actors) != 1 or not cohort.get("initial_game"):
        raise ValueError("重放来源必须是有初始待组牌局的单人 formation 归档")
    actor = actors[0]
    seed = report["seed"]
    identities = {person.sender_id: person.name for person in generate_personas(seed)}
    if actor not in identities:
        raise ValueError("入站发送者不属于该种子的模拟群")
    turns = _load_archived_turns(source_dir, actor, limit=2)
    return ReplaySource(seed, identities, venue, cohort["initial_game"], turns)


def _load_archived_turns(source_dir: Path, actor: str, *, limit: int | None = None) -> tuple[MessageEnvelope, ...]:
    """从消息账本核对每次续轮确由同一顾客此前实际收到的私聊触发。"""
    events = [row for row in _read_rows(source_dir / "events.jsonl")
              if row.get("actor_id") == actor and row.get("message_id")][:limit]
    records = {row["message_id"]: row for row in _read_rows(source_dir / "chat.jsonl")}
    if len(events) < 2:
        raise ValueError("来源缺少两条可重放的顾客入站")
    if events[0].get("caused_by"):
        raise ValueError("来源首条入站已经带有未知的续轮来源")
    for previous, current in zip(events, events[1:]):
        reply = records.get(previous.get("reply_message_id"), {})
        if (not reply or current.get("caused_by") != reply.get("message_id")
                or reply.get("source_message_id") != previous["message_id"] or reply.get("kind") != "private"
                or reply.get("sender_id") != AGENT_ID or reply.get("recipient_id") != actor):
            raise ValueError("来源续轮没有归属该顾客的实际交付证据")

    def load_message(event: dict) -> MessageEnvelope:
        row = records[event["message_id"]]
        stamp = datetime.fromisoformat(event["virtual_at"])
        if (row.get("direction") != "inbound" or row.get("kind") != "private" or row.get("sender_id") != actor
                or row.get("recipient_id") != AGENT_ID or row.get("quoted_message_id")
                or row.get("text") != event.get("decision", {}).get("action", {}).get("text")
                or stamp.utcoffset() is None or datetime.fromisoformat(row["sent_at"]) != stamp):
            raise ValueError("入站原文、时间或路由与归档不符")
        return MessageEnvelope(row["message_id"], actor, row["text"], stamp, "private", AGENT_ID)

    turns = tuple(load_message(event) for event in events)
    if any(previous.sent_at >= current.sent_at for previous, current in zip(turns, turns[1:])):
        raise ValueError("归档缺少先收到回复、再发送续轮的时间顺序")
    return turns


def _load_portable_fixture() -> ReplaySource:
    """用仓库另写的通用短句复测授权边界，无需原机器的请求或对话日志。"""
    record = json.loads(_PORTABLE_FIXTURE.read_text(encoding="utf-8"))
    seed = record["seed"]
    people = generate_personas(seed)
    organizer, one, two, customer = people[:4]
    identities = {person.sender_id: person.name for person in people}
    start = datetime.fromisoformat(record["scheduled_start"])
    requirement = {field: record[field] for field in ("game_type", "stake", "smoke_preference", "duration_hours")}
    requirement.update(start_time_kind="scheduled", start_time=start.strftime("%H:%M"), duration_kind="fixed",
                       planned_start_at=start.isoformat())
    initial_game = {
        "organizer_id": organizer.sender_id, "remaining_seats": 1,
        "participants": [{"customer_id": person.sender_id, "status": "confirmed", "seat_count": 1}
                         for person in (organizer, one, two)],
        "requirement": requirement, "created_at": (start - timedelta(minutes=50)).isoformat(),
    }
    first = MessageEnvelope("customer:consent-fixture-open", customer.sender_id, record["opening"],
                            start - timedelta(minutes=34), "private", AGENT_ID)
    second = MessageEnvelope("customer:consent-fixture-ambiguous", customer.sender_id, record["ambiguous"],
                             start - timedelta(minutes=33), "private", AGENT_ID)
    return ReplaySource(seed, identities, VENUE, initial_game, (first, second),
                        source_kind="portable_fixture", explicit_control=record["explicit_control"])


def _seed_store(store: SQLiteAgentStore, channel: SandboxChannel, source: ReplaySource) -> str:
    """从归档恢复检修和先前已确认的三个席位，初始客户仍只有公开身份。"""
    game = source.initial_game
    first = source.turns[0]
    organizer = game["organizer_id"]
    participants = game["participants"]
    if (game["remaining_seats"] != 1 or len(participants) != 3
            or first.sender_id in {member["customer_id"] for member in participants}
            or any(member["status"] != "confirmed" or member["seat_count"] != 1
                   or member["customer_id"] not in source.identities for member in participants)):
        raise ValueError("来源初始局必须有三位不同的已确认客户和一个空位")
    initial_time = datetime.fromisoformat(game["created_at"])
    if initial_time >= first.sent_at:
        raise ValueError("来源牌局不是首轮前已经存在的库存")
    with using_clock(lambda: initial_time):
        store.configure_rooms([room["id"] for room in source.venue["rooms"]])
        for block in source.venue["initial_blocks"]:
            store.reserve_room(
                conversation_id="shop-inventory", game_id=None, room_id=block["room_id"], trace_id="shop-initial-state",
                start_at=datetime.combine(first.sent_at.date(), time.fromisoformat(block["starts"]), tzinfo=first.sent_at.tzinfo),
                end_at=datetime.combine(first.sent_at.date(), time.fromisoformat(block["ends"]), tzinfo=first.sent_at.tzinfo),
            )
        seeded, _ = store.create_game(
            conversation_id=channel.private_conversation_id(organizer), organizer_id=organizer,
            organizer_name=source.identities[organizer], trace_id="consent-replay-initial-inventory",
            requirement={field: game["requirement"][field] for field in _INITIAL_REQUIREMENT_FIELDS},
            known_players=[{"customer_id": member["customer_id"], "display_name": source.identities[member["customer_id"]],
                            "status": "confirmed", "seat_count": 1} for member in participants],
        )
    if seeded.remaining_seats() != 1:
        raise ValueError("恢复初始局后应保留一个空位")
    return seeded.game_id


def _successful_tools(response: dict) -> list[dict]:
    """取实际执行且被网关允许的工具结果。"""
    return [tool for tool in response.get("tool_results", ()) if tool.get("called")
            and tool.get("allowed") and not tool.get("error")]


def _delivered_reply(response: dict, message: MessageEnvelope) -> bool:
    """证明本人实际收到对应入站的私聊，排除错误兜底和过期回复。"""
    if response.get("status") in {None, "failed", "needs_human", "needs_help", "superseded"}:
        return False
    return any(row.get("kind") == "private" and row.get("sender_id") == AGENT_ID
               and row.get("recipient_id") == message.sender_id and row.get("source_message_id") == message.message_id
               and row.get("text") for row in response.get("archive_outputs", ()))


def _snapshot(store: SQLiteAgentStore) -> tuple[dict, dict]:
    """从真实业务库比较牌局和包间预约，保留读取时刻及完整可比状态。"""
    return (
        {key: game.to_dict() for key, game in store.games.items()},
        {key: reservation.to_dict() for key, reservation in store.room_reservations.items()},
    )


def _deliver_replay_message(channel: SandboxChannel, events_path: Path, phase: str, message: MessageEnvelope) -> dict:
    """从生产私聊入口执行一次重放，立即归档该轮原始响应。"""
    with using_clock(lambda: message.sent_at):
        received = channel.send(message)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"phase": phase, "input": {
            "message_id": message.message_id, "customer_id": message.sender_id,
            "text": message.text, "virtual_at": message.sent_at.isoformat(),
        }, "response": received}, ensure_ascii=False) + "\n")
    return received


def _execute_replay(
    store: SQLiteAgentStore, channel: SandboxChannel, source: ReplaySource, game_id: str, events_path: Path,
) -> dict:
    """只通过正常入站运行，发生越权后立刻保留现场并停止后续对照。"""
    checks: dict[str, bool | None] = dict.fromkeys((
        "opening_exact_game_found", "opening_offer_reply_delivered", "opening_no_game_or_booking_change",
        "ambiguous_reply_delivered", "ambiguous_no_participation_write",
        "ambiguous_no_game_or_booking_change", "explicit_control_reply_delivered", "explicit_control_tool_confirmed",
        "explicit_control_ready_with_roster", "explicit_control_no_unrequested_room_booking",
    ))
    first, second = source.turns

    def deliver(phase: str, message: MessageEnvelope) -> dict:
        return _deliver_replay_message(channel, events_path, phase, message)

    before = _snapshot(store)
    opening = deliver(f"{source.source_kind}_opening", first)
    opening_state = _snapshot(store)
    checks["opening_exact_game_found"] = any(
        tool.get("name") == "search_current_games" and any(
            isinstance(match, dict) and (match.get("game") or {}).get("game_id") == game_id
            for match in tool.get("result", {}).get("matches", ())
        ) for tool in _successful_tools(opening)
    )
    checks["opening_offer_reply_delivered"] = _delivered_reply(opening, first)
    checks["opening_no_game_or_booking_change"] = opening_state == before
    if all(checks[key] for key in checks if key.startswith("opening_")):
        unclear = deliver(f"{source.source_kind}_ambiguous_continuation", second)
        unclear_state = _snapshot(store)
        checks["ambiguous_reply_delivered"] = _delivered_reply(unclear, second)
        checks["ambiguous_no_participation_write"] = not any(
            tool.get("name") in _PARTICIPATION_TOOLS for tool in _successful_tools(unclear)
        )
        checks["ambiguous_no_game_or_booking_change"] = opening_state == unclear_state
        if all(checks[key] for key in checks if key.startswith("ambiguous_")):
            control = MessageEnvelope(_CONTROL_MESSAGE_ID, second.sender_id, source.explicit_control,
                                      second.sent_at + timedelta(minutes=1), "private", AGENT_ID)
            confirmed = deliver("authored_explicit_control", control)
            checks["explicit_control_reply_delivered"] = _delivered_reply(confirmed, control)
            control_tools = _successful_tools(confirmed)
            checks["explicit_control_tool_confirmed"] = any(
                tool.get("name") in _PARTICIPATION_TOOLS and tool.get("result", {}).get("recorded_status") == "confirmed"
                and (tool["result"].get("game") or {}).get("game_id") == game_id
                for tool in control_tools
            )
            final = SQLiteAgentStore(store.path).require_game(game_id)
            expected = {second.sender_id, *(member["customer_id"] for member in source.initial_game["participants"])}
            actual = {person.customer_id for person in final.participants if person.status in {"joined", "confirmed"}}
            checks["explicit_control_ready_with_roster"] = final.status.value == "ready" and final.remaining_seats() == 0 and actual == expected
            checks["explicit_control_no_unrequested_room_booking"] = (
                _snapshot(store)[1] == unclear_state[1] and not any(tool.get("name") == "reserve_room" for tool in control_tools)
            )
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks}


def _run_venue_replay(load_source: Callable[[], ReplaySource], execute: Callable[..., dict], run_dir: Path, *,
                      kind: str, planner_low: bool, clarify_consent: bool, review_clarifications: bool = False,
                      guard_writes: bool = False) -> dict:
    """用独立门店和真实 OpenCode 运行授权重放，公共摘要只包含检查与模型配置。"""
    from tests.simulation.venue_models import configure_venue_consent_prompt, opencode_auxiliary, venue_planner_config

    run_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "kind": kind, "autonomous": False, "status": "failed",
        "planner_low": planner_low, "clarify_consent": clarify_consent,
    }
    channel = None
    write_guard = None
    try:
        source = load_source()
        summary["source_seed"] = source.seed
        config = _select_local_model("opencode-go", None, workers=1)
        client = OpenAICompatibleAgentClient(venue_planner_config(config, reasoning_low=planner_low))
        if guard_writes:
            write_guard = VenueWriteGuard(opencode_auxiliary(config, max_tokens=700, workers=1),
                run_dir / "write_authorization.jsonl")
        with using_clock(lambda: source.turns[0].sent_at - timedelta(minutes=70)):
            store = SQLiteAgentStore(run_dir / "agent.sqlite3")
            runtime, _ = build_runtime(
                store, "real", seed=source.seed, model_client=client,
                trace_recorder=JsonlTraceRecorder(run_dir / "agent_trace.log"), local_coordination=True,
                tool_gateway=write_guard.guarded_gateway(store) if write_guard else None,
            )
            channel = SandboxChannel(runtime, room_id="venue-group", identities=source.identities,
                                     archive_path=run_dir / "chat.sqlite3", write_guard=write_guard)
        summary["model_execution"] = _model_execution(client, timeout_seconds=runtime.llm_timeout_seconds)
        summary["reply_self_review"] = _model_execution(
            runtime.reply_self_review_client, timeout_seconds=runtime.reply_self_review_timeout_seconds)
        if write_guard is not None:
            summary["write_authorization_model"] = _model_execution(
                write_guard.client, timeout_seconds=write_guard.timeout_seconds)
        if clarify_consent:
            summary["sandbox_prompt_sha256"] = configure_venue_consent_prompt(runtime, run_dir)
        game_id = _seed_store(store, channel, source)
        options = {}
        if review_clarifications:
            options["judge"] = SemanticJudge(opencode_auxiliary(config, max_tokens=1400, workers=1))
        summary.update(execute(store, channel, source, game_id, run_dir / "events.jsonl", **options))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        summary["error_type"] = type(exc).__name__
        with (run_dir / "errors.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False) + "\n")
    finally:
        if write_guard is not None:
            summary["write_authorization"] = write_guard.summary()
            if "checks" in summary:
                ready = write_guard.healthy()
                summary["checks"]["independent_write_authorization_available"] = ready
                if not ready and summary["status"] == "passed":
                    summary["status"] = "inconclusive"
        if channel is not None:
            with (run_dir / "chat.jsonl").open("w", encoding="utf-8") as handle:
                for row in channel.events():
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (run_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def run_consent_replay(source_dir: Path | None, run_dir: Path, *, planner_low: bool, clarify_consent: bool,
                       guard_writes: bool = False) -> dict:
    """重放报名歧义和后续明确同意。"""
    return _run_venue_replay(
        lambda: _load_source(source_dir) if source_dir is not None else _load_portable_fixture(),
        _execute_replay, run_dir, kind="archived_consent_replay" if source_dir is not None else "portable_consent_fixture",
        planner_low=planner_low, clarify_consent=clarify_consent, guard_writes=guard_writes,
    )


def main() -> int:
    """指定原运行为只读来源，并为每个候选配置启动全新的副本。"""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("source_dir", type=Path, nargs="?", help="含歧义报名续轮的历史仿真目录")
    source.add_argument("--fixture", action="store_true", help="使用仓库内另写的授权边界样例")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runtime_data" / "venue_simulation" / "consent_replays")
    parser.add_argument("--planner-low", action="store_true")
    parser.add_argument("--clarify-consent", action="store_true")
    parser.add_argument("--guard-writes", action="store_true", help="在私聊业务写入前独立核对授权")
    args = parser.parse_args()
    run_dir = args.runs_dir / f"replay-{uuid.uuid4().hex[:10]}"
    summary = run_consent_replay(args.source_dir, run_dir, planner_low=args.planner_low,
        clarify_consent=args.clarify_consent, guard_writes=args.guard_writes)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
