"""用真实模型在独立 SQLite 回放脱敏群时间线，外发仅记入沙盒。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, time, timedelta
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mahjong_agent_runtime import AgentRuntime, OpenAICompatibleAgentClient, SQLiteAgentStore, TokenBudget
from mahjong_agent_runtime.env import load_dotenv_defaults
from mahjong_agent_runtime.group_chat import ChannelIdentity, GroupRoomPolicy
from mahjong_agent_runtime.models import CustomerProfile, DEFAULT_TZ, now
from mahjong_agent_runtime.review_config import build_reply_review_client, reply_review_timeout_seconds
from manual_group_simulation import process_manual_group
from run_real_group_chat_flow_eval import _configured_credentials, _EvidenceTraceRecorder, _safe_evidence, _write_evidence
from validate_real_group_chat_dataset import validate_datasets

ENV_FILE = Path.home() / ".local/share/mahjong-ops-agent/.env"
ALIAS = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,79}")
CONTENT_TYPES = {"text", "quote_reply", "emoji"}


def load_cases(path: Path) -> list[dict]:
    """在创建沙盒和调用模型前检查顺序、角色、正文及每条引用的来源。"""
    cases, errors = validate_datasets([path])
    if errors:
        raise ValueError("\n".join(errors))
    case_ids = set()
    for case in cases:
        case_id = case["id"]
        if case["case_type"] != "group_topic_timeline" or not ALIAS.fullmatch(case_id) or case_id in case_ids:
            raise ValueError("群时间线类型或用例编号无效")
        case_ids.add(case_id)
        if time.fromisoformat(case["timeline_start_local"]).tzinfo is not None:
            raise ValueError("时间线锚点应为无时区的本地时刻")
        aliases = set()
        previous_offset = -1
        for message in case["messages"]:
            alias = message["message_alias"]
            offset = message["offset_seconds"]
            content_type = message["content_type"]
            text = message["text"]
            target = message.get("target_message_alias")
            if not ALIAS.fullmatch(alias) or alias in aliases or not ALIAS.fullmatch(message["actor_alias"]):
                raise ValueError("消息或发言人别名无效")
            if type(offset) is not int or offset < max(0, previous_offset):
                raise ValueError("消息偏移须为按顺序排列的非负整数")
            if message["role"] != "customer" or content_type not in CONTENT_TYPES or not isinstance(text, str):
                raise ValueError("消息角色、类型或正文无效")
            if (content_type == "emoji") == bool(text.strip()):
                raise ValueError("文本须有正文，纯表情须为空")
            if (target in aliases) != (content_type == "quote_reply") or (target and content_type != "quote_reply"):
                raise ValueError("引用必须指向同一时间线中较早的消息")
            aliases.add(alias)
            previous_offset = offset
        if not aliases:
            raise ValueError("群聊时间线为空")
    if not cases:
        raise ValueError("群聊数据集为空")
    return cases


def write_report(path: Path, report: dict, credentials: tuple[str, ...]) -> None:
    """原子替换本轮快照，使运行中读取也能得到完整 JSON。"""
    pending = path.with_suffix(".tmp")
    _write_evidence(pending, report, credentials)
    pending.replace(path)


def run_case(case: dict, folder: Path, client, review_client, credentials: tuple[str, ...]) -> dict:
    """顺序回放原时间间隔与引用，只将当前受控轮送入正式群入口。"""
    folder.mkdir()
    store = SQLiteAgentStore(folder / "state.sqlite3")
    report_path = folder / "report.json"
    run_id = case["id"]
    room_id = f"manual-room:{run_id}"
    anchor = datetime.combine(now().date(), time.fromisoformat(case["timeline_start_local"]), DEFAULT_TZ)
    report = {
        "case_id": run_id,
        "status": "running",
        "anchor": anchor.isoformat(),
        "candidate_expectation": case.get("candidate_expectation"),
        "turns": [],
    }
    try:
        store.configure_rooms(["replay_room_1", "replay_room_2", "replay_room_3"])
        store.upsert_group_room_policy(GroupRoomPolicy(
            room_id=room_id, channel="console", managed=True, board_enabled=True, outbound_enabled=True,
        ))
        aliases = dict.fromkeys(message["actor_alias"] for message in case["messages"])
        actors = {}
        for index, alias in enumerate(aliases, start=1):
            profile = CustomerProfile(customer_id=alias, display_name=f"群友{index}")
            store.upsert_customer(profile)
            store.upsert_channel_identity(ChannelIdentity(
                channel="console", external_user_id=alias, customer_id=alias, public_name=profile.display_name,
                private_conversation_id=f"replay:{run_id}:{alias}",
            ))
            actors[alias] = profile.to_dict()
        sandbox = {"run_id": run_id, "manual_actors": list(actors.values())}
        runtime = AgentRuntime(
            llm_client=client,
            store=store,
            trace_recorder=_EvidenceTraceRecorder(folder / "agent_trace.log", credentials),
            token_budget=TokenBudget(
                max_tokens_per_call=int(os.getenv("MAHJONG_AGENT_MAX_TOKENS_PER_CALL", "64000")),
                max_calls_per_turn=int(os.getenv("MAHJONG_AGENT_MAX_CALLS_PER_TURN", "8")),
            ),
            customer_visible_text_generation_enabled=True,
            customer_visible_text_generation_client=client,
            reply_self_review_enabled=True,
            reply_self_review_client=review_client,
            reply_self_review_timeout_seconds=reply_review_timeout_seconds(),
        )
        for item in case["messages"]:
            alias = item["message_alias"]
            stamp = anchor + timedelta(seconds=item["offset_seconds"])
            trace_id = f"{run_id}:{alias}"
            turn = {"input": item, "sent_at": stamp.isoformat(), "trace_id": trace_id, "status": "running"}
            report["turns"].append(turn)
            write_report(report_path, report, credentials)
            print(json.dumps({"case": run_id, "turn": alias, "status": "running"}), flush=True)
            try:
                replay_turn(runtime, sandbox, actors[item["actor_alias"]], turn, room_id, stamp)
            except Exception as exc:
                turn.update(status="error", error=f"{type(exc).__name__}: {exc}")
            events = runtime.trace_recorder.get_trace(trace_id)
            turn["trace_steps"] = dict(Counter(event.step for event in events))
            write_report(report_path, report, credentials)
            print(json.dumps({"case": run_id, "turn": alias, "status": turn["status"],
                              "action": turn.get("routing_action")}), flush=True)
        report["state"] = {
            "topics": store.list_group_topics(room_id=room_id, include_cold=True, limit=100),
            "games": [game.to_dict() for game in store.games.values()],
            "scheduled_tasks": [task.to_dict() for task in store.scheduled_tasks.values()],
        }
        report["status"] = "failed" if any(turn["status"] == "error" for turn in report["turns"]) else "completed"
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        write_report(report_path, report, credentials)
        store._connection.close()
    return report


def replay_turn(runtime, sandbox: dict, actor: dict, turn: dict, room_id: str, stamp: datetime) -> None:
    """保留入口实际产生的结果，纯表情按原生入口的无正文行为单独计数。"""
    item = turn["input"]
    if item["content_type"] == "emoji":
        turn.update(status="skipped_non_text", routing_action="ignore", sandbox_outputs=[], runtime_results=[])
        return
    quote_id = item.get("target_message_alias")
    origin = runtime.store.get_group_topic_for_message_reference("console", room_id, quote_id) if quote_id else None
    _, processing, source = process_manual_group(
        runtime, sandbox, actor, item["text"], message_id=item["message_alias"], stamp=stamp,
        trace_id=turn["trace_id"], quote_id=quote_id,
    )
    receipt = processing["topic_analysis"]["receipt"]
    classification = (receipt.get("message") or {}).get("classification") or {}
    model_input = (classification.get("audit") or {}).get("input") or {}
    turn.update(
        status="completed",
        group_input=source.to_dict(),
        routing_action=processing["routing"]["action"],
        routing_game_id=processing["routing"].get("game_id"),
        topic_receipt=receipt,
        runtime_results=processing["runtime_results"],
        sandbox_outputs=processing["outputs"],
        state_transitions=processing["state_transitions"],
        board_flush_count=processing["routing"].get("board_flush_count", 0),
    )
    if quote_id:
        expected_topic = origin["topic_id"] if origin else None
        candidate_ids = [card["topic_id"] for card in model_input.get("candidates", []) if card.get("quoted_reference")]
        observed_quote = (model_input.get("message") or {}).get("quoted_message_id")
        turn["quote_check"] = {
            "expected_message_id": quote_id,
            "group_message_id": source.quoted_message_id,
            "model_message_id": observed_quote,
            "expected_topic_id": expected_topic,
            "model_quoted_topic_ids": candidate_ids,
        }
        if source.quoted_message_id != quote_id or observed_quote != quote_id or (
            expected_topic and expected_topic not in candidate_ids
        ):
            raise AssertionError("群入口或模型请求漏掉已知引用")
    statuses = [result["status"] for result in turn["runtime_results"]]
    if any(status not in {"completed", "waiting_user"} for status in statuses):
        raise RuntimeError(f"主 Agent 未正常完成：{statuses}")


def main() -> int:
    """校验后才启动真实模型，并为每次回放创建独立证据目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "runtime_data/real_group_chat_topic_replay")
    parser.add_argument("--env-file", type=Path, default=ENV_FILE)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    if args.validate_only:
        print(json.dumps({"validated_cases": len(cases), "messages": sum(len(case["messages"]) for case in cases)}))
        return 0
    load_dotenv_defaults(args.env_file)
    load_dotenv_defaults(ROOT / ".env")
    # 群路由的可选缓存由环境启用，独立回放只使用本次 SQLite。
    os.environ.pop("MAHJONG_TOPIC_REDIS_URL", None)
    client = OpenAICompatibleAgentClient.from_env()
    if client is None:
        parser.error("真模型回放需要已配置的模型及 API 凭证")
    review_client = build_reply_review_client(client.config)
    credentials = tuple(sorted(set(_configured_credentials(client)) | set(_configured_credentials(review_client)),
                               key=len, reverse=True))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="run_", dir=args.work_dir))
    print(json.dumps({"run_dir": str(folder), "model": _safe_evidence(client.config.model, credentials)}), flush=True)
    summary = {"model": client.config.model, "cases": []}
    for case in cases:
        report = run_case(case, folder / case["id"], client, review_client, credentials)
        summary["cases"].append({"id": report["case_id"], "status": report["status"],
                                 "turns": dict(Counter(turn["status"] for turn in report["turns"]))})
        write_report(folder / "summary.json", summary, credentials)
    return 0 if all(case["status"] == "completed" for case in summary["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
