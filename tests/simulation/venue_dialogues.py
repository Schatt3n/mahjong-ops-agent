"""以已送达回复为因果触发，在沙盒中推进多位顾客的开放式对话。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import random
from threading import Lock
from typing import Any

from mahjong_agent_runtime.models import using_clock

from .venue_acceptance import evaluate_venue_acceptance
from .venue_actors import ActorContext, ModelActor, Persona
from .venue_channel import AGENT_ID, MessageEnvelope
from .venue_judging import review_dialogue


@dataclass(slots=True)
class Dialogue:
    """单人一次营业会话；后续轮次必须引用此前真实送达的私聊。"""

    person: Persona
    goal: str
    cause: str | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)
    active: bool = True


class VenueDialogues:
    """同一波次并行处理不同客户，一个客户下一轮由本人收到的回复推进。"""

    def __init__(self, *, people, actor, judge, channel, store, run_dir, start_at, seed, active_users,
                 workers, dialogue_turns, background_events, fixture_agent=None, judge_users=3):
        people = list(people)
        self.population = len(people)
        random.Random(seed).shuffle(people)
        goals = ("lookup", "try_join", "wait_or_leave")
        self.sessions = [Dialogue(person, goals[i % len(goals)]) for i, person in enumerate(people[:active_users])]
        self.background = people[active_users:active_users + background_events]
        self.actor, self.judge, self.channel, self.store = actor, judge, channel, store
        self.run_dir, self.start_at, self.fixture_agent = run_dir, start_at, fixture_agent
        self.workers, self.dialogue_turns, self.judge_users = workers, dialogue_turns, judge_users
        self.model_errors: list[dict[str, Any]] = []
        self.failed_events: list[str] = []
        self.lock = Lock()
        self.in_flight = self.peak_sends = 0

    def run(self) -> dict[str, Any]:
        """归档实际并行交付，按会话统计多轮并核对服务端终态。"""
        stamp = self.start_at - timedelta(minutes=35)
        with self.run_dir.joinpath("events.jsonl").open("a", encoding="utf-8") as log:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                group_jobs = [pool.submit(self._turn, Dialogue(person, "group"), stamp, 1) for person in self.background]
                for job in group_jobs:
                    row = job.result()
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                    log.flush()
                    self._capture_error(row)
                for index in range(1, self.dialogue_turns + 1):
                    now = stamp + timedelta(minutes=index)
                    active = [session for session in self.sessions if session.active]
                    if not active:
                        break
                    jobs = [pool.submit(self._turn, session, now, index) for session in active]
                    for session, job in zip(active, jobs):
                        row = job.result()
                        log.write(json.dumps(row, ensure_ascii=False) + "\n")
                        log.flush()
                        session.turns.append(row)
                        session.cause = row.get("reply_message_id")
                        session.active = self._should_continue(session, row)
                        self._capture_error(row)
        records = self.channel.events()
        with self.run_dir.joinpath("chat.jsonl").open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        turns = [turn for session in self.sessions for turn in session.turns]
        outcomes = [self._outcome(session) for session in self.sessions]
        acceptance = evaluate_venue_acceptance(messages=records, turns=turns, outcomes=outcomes)
        checks = {
            "all_targeted_users_have_evidence": all(session.turns and session.turns[0]["acted"] for session in self.sessions),
            "overlapping_channel_delivery": self.workers == 1 or len(self.sessions) == 1 or self.peak_sends >= 2,
            "outcomes_grounded": acceptance["status"] == "passed",
        }
        model_actions = sum(_live_actor_action(row) for row in turns)
        if isinstance(self.actor, ModelActor):
            checks["customer_decisions_from_model"] = bool(turns) and model_actions == len(turns)
            # 真实会话可以在首个答案后结束；若有已答复的续轮，仍须承接本人收到的上一条回复。
            conversations = acceptance["conversations"]
            checks["causal_private_dialogues"] = all(
                (counts := conversations.get(session.person.sender_id, {})).get("private_rounds", 0)
                == counts.get("causal_private_followups", 0) + 1
                for session in self.sessions
            )
        else:
            checks["causal_multi_turn_coverage"] = acceptance["multi_round_private_customers"] >= (len(self.sessions) + 1) // 2
        reviewed = self._review_samples(records, acceptance["outcomes"])
        valid_review = reviewed and all(row["status"] == "valid" and row["verdict"] == "pass" for row in reviewed)
        review_status = "not_run"
        if reviewed:
            review_status = "valid" if all(row["status"] == "valid" for row in reviewed) else "partial"
        raw_trace = self.run_dir / "agent_trace.log"
        if raw_trace.exists():
            with raw_trace.open(encoding="utf-8") as handle:
                checks["role_truth_not_in_agent_trace"] = not any("venue-persona-" in line for line in handle)
        return {
            "checks": checks, "people": self.population,
            "targeted_users": len(self.sessions), "peak_inflight_deliveries": self.peak_sends,
            "real_customer_actions": model_actions,
            "workers": self.workers, "max_turns": self.dialogue_turns, "chat_events": len(records),
            "acceptance": acceptance, "model_transport_errors": self.model_errors,
            "failed_events": self.failed_events,
            "semantic_samples": reviewed,
            "independent_judge": None if self.judge is None else {
                "status": review_status,
                "verdict": "pass" if valid_review else "fail" if any(row["verdict"] == "fail" for row in reviewed) else "inconclusive",
            },
        }

    def _capture_error(self, row: dict[str, Any]) -> None:
        """只把传输故障列为模型不可用，动作合同错误另保留在本轮证据。"""
        if row.get("error_type"):
            self.failed_events.append(row["event_id"])
        if row.get("error_type") in {"transport_error", "runtime_transport_error"}:
            self.model_errors.append({"event_id": row["event_id"], "model_stage": row.get("failure_stage", "role"),
                                      "http_statuses": row.get("http_statuses", [])})

    def _should_continue(self, session: Dialogue, row: dict[str, Any]) -> bool:
        """续轮必须有已送达的答复，出现技术故障时保留证据并结束当前会话。"""
        return bool(session.cause) and not row.get("error_type")

    def _dialogue_details(self, session: Dialogue, index: int) -> dict[str, Any]:
        """给角色自己的当期意向和时间，后续业务事实从实际回复取得。"""
        person = session.person
        return {"activity": "open_dialogue", "case_goal": session.goal, "game_type": person.preferred_game,
                "stake": person.stake_min, "start_time": self.start_at.strftime("%H:%M"), "duration_hours": 4}

    def _fixture_turn(self, session: Dialogue, index: int) -> tuple[str, dict[str, Any]]:
        """提供离线合同桩的预期，仅在未启用真实模型时使用。"""
        return ("dialogue_followup" if index > 1 else "dialogue_open", {})

    def _turn(self, session: Dialogue, at: datetime, index: int) -> dict[str, Any]:
        person = session.person
        event_id = f"talk:{person.sender_id}:{index}"
        row = {"event_id": event_id, "actor_id": person.sender_id, "virtual_at": at.isoformat(),
               "caused_by": session.cause, "acted": False}
        with using_clock(lambda: at):
            try:
                history = self.channel.events_for(person.sender_id)[-20:]
                if session.cause:
                    # 群聊热度不能把本人实际收到的触发回复挤出角色窗口。
                    cause = next((item for item in history if item["message_id"] == session.cause), None)
                    if cause is None:
                        cause = next(item for item in self.channel.events_for(person.sender_id) if item["message_id"] == session.cause)
                        history = [cause, *history[-19:]]
                kind = "activity" if session.goal == "group" else "delivered" if session.cause else "session_open"
                details = self._dialogue_details(session, index)
                context = ActorContext(person, at.isoformat(), history=history,
                    events=[{"event_id": event_id, "kind": kind, "at": at.isoformat(),
                             "caused_by": session.cause, "details": details if session.goal != "group" else {"case_goal": "group"}}],
                    private_targets=(AGENT_ID,), group_targets=(self.channel.room_id,) if session.goal == "group" else (),
                    visible_groups=(self.channel.room_id,),
                )
                decision = self.actor.decide(context, trace_id=f"role:{event_id}")
                row.update(decision=decision.to_dict(), acted=True)
                if decision.action is None:
                    row.update(error_type=decision.status, http_statuses=[
                        item["http_status"] for item in decision.diagnostics.get("attempts", []) if item.get("http_status") is not None
                    ])
                    return row
                action = decision.action
                if action.action != "send":
                    return row
                message_id = f"customer:{event_id}"
                kind = action.kind or "private"
                recipient = action.recipient_id or (self.channel.room_id if kind == "group" else AGENT_ID)
                if self.fixture_agent is not None and session.goal != "group":
                    with self.lock:
                        self.fixture_agent.turns[message_id] = self._fixture_turn(session, index)
                row["message_id"] = message_id
                with self.lock:
                    self.in_flight += 1
                    self.peak_sends = max(self.peak_sends, self.in_flight)
                try:
                    response = self.channel.send(MessageEnvelope(
                        message_id, person.sender_id, action.text, at, kind, recipient,
                    ))
                finally:
                    with self.lock:
                        self.in_flight -= 1
                outputs = [item for item in response.get("archive_outputs", [])
                           if item["kind"] == "private" and item["recipient_id"] == person.sender_id]
                row.update(trace_id=response.get("trace_id"), runtime_status=response.get("status"),
                           tool_results=response.get("tool_results", []), outputs=outputs,
                           reply_message_id=outputs[-1]["message_id"] if outputs else None)
                if row["runtime_status"] in {"needs_human", "failed", "needs_help"} and row["trace_id"]:
                    traces = self.channel.runtime.trace_recorder.get_trace(row["trace_id"])
                    failure = _model_failure([item.content for item in traces if item.step == "llm_error"])
                    if failure:
                        row.update(failure, failure_stage="agent")
                return row
            except (OSError, ValueError, RuntimeError, StopIteration) as exc:
                return {**row, "error_type": type(exc).__name__, "error": str(exc), "failure_stage": "delivery"}

    def _outcome(self, session: Dialogue) -> dict[str, Any]:
        """以实际工具和最终业务状态归因，未取得匹配可作为合法结果。"""
        customer_id = session.person.sender_id
        message_ids = [row["message_id"] for row in session.turns if row.get("message_id")]
        result = {"customer_id": customer_id, "source_message_ids": message_ids}
        if any(row.get("error_type") == "timeout" for row in session.turns):
            return {**result, "kind": "timeout"}
        if any(row.get("error_type") for row in session.turns):
            return {**result, "kind": "model_fault"}
        tools = [tool for turn in session.turns for tool in turn.get("tool_results", [])
                 if tool.get("called") and tool.get("allowed") and not tool.get("error")]
        game_ids = dict.fromkeys((item.get("result", {}).get("game") or {}).get("game_id") for item in reversed(tools))
        for game_id in (value for value in game_ids if value):
            game = self.store.require_game(game_id)
            participant = next((person for person in game.participants if person.customer_id == customer_id), None)
            if game.organizer_id != customer_id and participant is None:
                continue
            if participant is not None and participant.status in {"declined", "rejected"}:
                return {**result, "kind": "declined", "game": game.to_dict()}
            if game.closed_reason == "expired_without_full_table" and game.status.value == "cancelled":
                return {**result, "kind": "expired_unfilled", "game": game.to_dict()}
            if game.status.value == "finished" and customer_id in game.actual_attendee_ids:
                reservation = self.store.room_reservations.get(game.play_reservation_id)
                return {**result, "kind": "completed", "game": game.to_dict(),
                        "reservation": reservation.to_dict() if reservation else None}
            kind = {"ready": "formed", "finished": "formed", "forming": "waiting",
                    "inviting": "waiting", "cancelled": "cancelled"}.get(game.status.value)
            if kind:
                return {**result, "kind": kind, "game": game.to_dict()}
        for tool in reversed(tools):
            returned = tool.get("result", {})
            demands = returned.get("cancelled_demands") or [returned.get("waiting_demand") or {}]
            for item in demands:
                demand = self.store.waiting_demands.get(item.get("demand_id"))
                if demand is not None and demand.sender_id == customer_id:
                    kind = {"active": "waiting", "cancelled": "cancelled"}.get(demand.status.value)
                    if kind:
                        return {**result, "kind": kind, "waiting_demand": demand.to_dict()}
            if tool["name"] == "search_current_games" and "matches" in returned and "alternatives" in returned:
                return {**result, "kind": "match_offered" if returned["matches"] or returned["alternatives"] else "no_match"}
        return {**result, "kind": "unverified"}

    def _review_samples(self, records, outcomes) -> list[dict[str, Any]]:
        if self.judge is None:
            return []
        candidates = [session for session in self.sessions if any(row.get("reply_message_id") for row in session.turns)]
        outcomes_by_customer = {outcome["customer_id"]: outcome for outcome in outcomes}
        return [
            review_dialogue(self.judge, customer_id=session.person.sender_id, turns=session.turns,
                records=records, outcome=outcomes_by_customer.get(session.person.sender_id))
            for session in candidates[:self.judge_users]
        ]


def _model_failure(errors: list[dict[str, Any]]) -> dict[str, Any]:
    """从供应商证据区分超时、输出到上限和传输异常，保留本轮技术原因。"""
    if not errors:
        return {}
    attempts = [attempt for error in errors for attempt in (
        error.get("completion_diagnostics") or {}).get("attempts", [])]
    last = attempts[-1] if attempts else {}
    if last.get("incomplete_reason") == "max_output_tokens" or last.get("finish_reason") == "length":
        kind = "model_output_incomplete"
    elif last.get("error_type") in {"TimeoutError", "DeadlineExceeded"}:
        kind = "timeout"
    else:
        kind = "runtime_transport_error"
    return {"error_type": kind, "http_statuses": [item["http_status"] for item in attempts
                                                 if item.get("http_status") is not None]}


def _live_actor_action(row: dict[str, Any]) -> bool:
    """确认顾客动作有实际模型响应凭据，离线桩和未调用模型的动作不计入。"""
    decision = row.get("decision") or {}
    source = decision.get("response_source") or {}
    return (decision.get("source") == "model" and decision.get("status") == "valid"
            and source.get("http_response_complete") is True)
