#!/usr/bin/env python3
"""回放老板已提出五小时备选后的私聊，核对询问与等待均不会占座或改条件。"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import run_real_owner_chat_live_eval as owner
from run_inquiry_grounding_eval import assess_reply_semantics
from run_temporary_recruitment_probe import BusinessClock, dump

from mahjong_agent_runtime import CustomerProfile, InMemoryTraceRecorder, OpenAICompatibleAgentClient, SQLiteAgentStore, UserMessage
from mahjong_agent_runtime.env import load_dotenv_defaults
from mahjong_agent_runtime.llm import validate_model_response_source
from mahjong_agent_runtime.models import ConversationRole, ConversationTurn

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval/golden/real_owner_chat_timeline_20261127.json"
CONVERSATION = "fallback_private_chat"
CUSTOMER = "fallback_customer"
PRIVATE_MARKER = "仅店员可见的夹具备注"
EXPECTATIONS = {
    12: "理解客户在问五小时备选的同桌是谁。若提供名单，仅使用获准披露的公开别名牌友甲和牌友乙。",
    14: "理解客户暂时还没决定，可简短承接等待。不能把稍等当成同意、拒绝或约定复查时间。",
}


def customer_message(turn: dict, stamp: datetime) -> UserMessage:
    """以原始客户气泡构造私聊输入，回放时钟与来源记录分离。"""
    return UserMessage(conversation_id=CONVERSATION, sender_id=CUSTOMER, sender_name="客户",
                       text=turn["text"], message_id=f"source-turn-{turn['turn']}", sent_at=stamp)


def seed_history_and_games(store: SQLiteAgentStore, clock: BusinessClock, turns: dict) -> str:
    """把已观察到的老板提议作为历史，合成牌局只补齐评测所需的可见范围和状态。"""
    store.upsert_customer(CustomerProfile(CUSTOMER, "客户"))
    anchor = datetime.fromisoformat(turns[8]["observed_at"])
    task, _ = store.activate_task_context(conversation_id=CONVERSATION, customer_id=CUSTOMER,
        trace_id="fixture", activity_at=anchor, started_at=anchor, reason="保留本次四小时组局目标",
        force_new=False, archive_previous=False)
    for index in range(8, 12):
        turn = turns[index]
        stamp = datetime.fromisoformat(turn["observed_at"])
        trace_id = f"fixture:{index}"
        if turn["role"] == "customer":
            store.append_user_turn(customer_message(turn, stamp), trace_id)
        else:
            store.append_turn(CONVERSATION, ConversationTurn(role=ConversationRole.ASSISTANT,
                content=turn["text"], trace_id=trace_id, occurred_at=stamp,
                metadata={"task_context_id": task.task_context_id}))

    clock.set("16:52")
    requirement = {"start_time_kind": "scheduled", "start_time": "18:30",
                   "planned_start_at": f"{clock.day}T18:30:00+08:00"}
    store.create_game(conversation_id=CONVERSATION, organizer_id=CUSTOMER, organizer_name="客户",
        requirement={**requirement, "duration_hours": 4},
        known_players=[{"customer_id": CUSTOMER, "display_name": "客户"}], trace_id="fixture:original")
    players = []
    for key, name in (("fallback_peer_a", "牌友甲"), ("fallback_peer_b", "牌友乙")):
        store.upsert_customer(CustomerProfile(key, PRIVATE_MARKER, public_name=name, private_remark=PRIVATE_MARKER))
        players.append({"customer_id": key, "display_name": name})
    # 合成同私聊局用于触发现有公开昵称白名单，客户在备选局仍然没有席位。
    store.create_game(conversation_id=CONVERSATION, organizer_id="fallback_peer_a", organizer_name="牌友甲",
        requirement={**requirement, "duration_hours": 5}, known_players=players, trace_id="fixture:option")
    return task.task_context_id


def business_state(store: SQLiteAgentStore) -> dict:
    """比较持久业务状态与时长记忆，允许回放正常写入对话和审计。"""
    state = {name: {key: row.to_dict() for key, row in getattr(store, name).items()} for name in (
        "games", "customers", "room_reservations", "invite_drafts", "outbound_message_drafts",
        "waiting_demands", "scheduled_tasks", "pending_memory_candidates",
    )}
    state["duration_memories"] = {key: row.to_dict() for key, row in store.task_memories.items()
        if row.field in {"duration_hours", "max_duration_hours"}}
    task = store.current_task_context(CONVERSATION, CUSTOMER)
    state["task_context_id"] = task.task_context_id if task else None
    return state


def main() -> int:
    """在独立 SQLite 中连续执行两条客户原话，留下真实工具、语义评审和状态证据。"""
    raw = DATASET.read_bytes()
    source = json.loads(raw)
    turns = {turn["turn"]: turn for turn in source["messages"]}
    load_dotenv_defaults(ROOT / ".env")
    client = OpenAICompatibleAgentClient.from_env()
    if client is None:
        raise SystemExit("真实回放需要本地模型配置和密钥")
    parent = ROOT / "runtime_data" / "owner_duration_fallback"
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="probe-", dir=parent))
    report = {"status": "failed", "source": source["id"], "dataset_sha256": hashlib.sha256(raw).hexdigest(),
              "scope": "customer_replies_after_observed_owner_offer", "provider": client.config.provider,
              "model": client.config.model, "external_messages_sent": False, "turns": []}
    print(f"报告目录 {folder}", flush=True)
    try:
        store = SQLiteAgentStore(folder / "state.sqlite3")
        with closing(store._connection), BusinessClock(datetime.fromisoformat(turns[8]["observed_at"]).date().isoformat()) as clock:
            trace = InMemoryTraceRecorder()
            runtime = owner.build_runtime(client, store, trace, SimpleNamespace(
                max_steps=8, max_tokens_per_call=64000, max_calls_per_turn=12,
                skip_text_generation=True, skip_review=False, timeout_seconds=90))
            original_task_id = seed_history_and_games(store, clock, turns)
            for index, meaning in EXPECTATIONS.items():
                # 后续气泡无逐条时刻，用同一时间块内的合成秒数保留因果顺序。
                clock.current = datetime.fromisoformat(turns[11]["observed_at"]).replace(second=index - 11)
                before = business_state(store)
                message = customer_message(turns[index], clock.current)
                result = runtime.handle_user_message(message, trace_id=f"duration-fallback:turn-{index}")
                events = trace.get_trace(result.trace_id)
                dump(folder / f"turn-{index}-trace.json", [event.to_dict() for event in events])
                after = business_state(store)
                responses = [event for event in events if event.step == "llm_response"]
                has_live_response = bool(responses) and all(validate_model_response_source(
                    event.content.get("response_source"), raw_response=event.content.get("content"),
                    trace_id=event.trace_id, step_index=event.content.get("step_index"),
                    diagnostics=event.content.get("completion_diagnostics", {}), require_http=True) for event in responses)
                checks = {"unchanged_business_state": before == after and after["task_context_id"] == original_task_id,
                    "finished": result.status in {"completed", "waiting_user"},
                    "private_remark_absent": PRIVATE_MARKER not in result.final_reply,
                    "live_model_source": has_live_response}
                row = {"source_turn": index, "before": before, "after": after, "result": result.to_dict(),
                       "checks": checks}
                report["turns"].append(row)
                if not has_live_response:
                    # 供应商拒绝或未返回完整响应时，无从判定回退话术是否理解了客户。
                    row["model_errors"] = [event.content.get("error") for event in events if event.step == "llm_error"]
                    row["semantic"] = {"status": "not_run", "reason": "没有完整的主模型响应"}
                    break
                scene = owner.LiveEvalScenario(scenario_id="duration_fallback_follow_up", name="五小时备选追问",
                    message=message, setup=lambda store: None)
                expectation = {"required_meaning": [{"id": "interpret_customer_reply", "description": meaning}],
                    "forbidden_meaning": owner.common_forbidden_semantic_meanings() + [{"id": "invent_acceptance",
                        "description": "不能声称客户已接受五小时、已加入、已占座、已成局或已订房。"}]}
                assessment = assess_reply_semantics(client,
                    evidence=owner.reply_semantic_evidence(result, scene, events, store), expectation=expectation,
                    trace_id=f"{result.trace_id}:judge")
                checks["semantic_passed"] = assessment["passed"] is True
                recovery = owner.assess_action_contract_recovery(events, runtime_status=result.status,
                    final_reply=result.final_reply, business_checks_passed=all(checks.values()),
                    semantic_passed=checks["semantic_passed"])
                checks["action_contract"] = recovery["quality_status"] == "passed"
                row.update(semantic=assessment, recovery=recovery)
                if not all(checks.values()):
                    break
            if len(report["turns"]) == len(EXPECTATIONS) and all(all(turn["checks"].values()) for turn in report["turns"]):
                report["status"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        dump(folder / "report.json", report)
    for turn in report["turns"]:
        print(f"来源第 {turn['source_turn']} 条消息，校验 {turn['checks']}")
    print(report.get("error") or report["status"])
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
