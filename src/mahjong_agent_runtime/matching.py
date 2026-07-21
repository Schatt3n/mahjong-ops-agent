"""Reverse-trigger matching for demands that arrived before a suitable game."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol

from .hooks import HookEvent
from .models import (
    AgentRuntimeResult,
    Game,
    GameStatus,
    OutboundMessageDraft,
    ScheduledAgentTask,
    SystemTriggerMessage,
    WaitingDemand,
    new_id,
    now,
)
from .stores import AgentStore
from .domains.waiting_domain import (
    WAITING_DEMAND_EXPIRY_TASK_TYPE,
    next_waiting_expiry_due,
)


class SystemTriggerHandler(Protocol):
    def __call__(
        self,
        trigger: SystemTriggerMessage,
        *,
        trace_id: str | None = None,
    ) -> AgentRuntimeResult: ...


class MatchDispatcher(Protocol):
    def dispatch_match(
        self,
        demand: WaitingDemand,
        game: Game,
        *,
        trace_id: str,
    ) -> OutboundMessageDraft | None: ...


@dataclass(slots=True)
class OutboundDispatcher:
    """Route one match event through the normal Agent loop and persist its draft.

    The model writes the customer-facing sentence and the existing reply review
    contract approves it. This component only supplies a sanitized event, binds
    the result to the target channel, and records the business reference needed
    to understand a later short reply such as ``打``.
    """

    store: AgentStore
    system_trigger_handler: SystemTriggerHandler
    trace_recorder: Any

    def dispatch_match(
        self,
        demand: WaitingDemand,
        game: Game,
        *,
        trace_id: str,
    ) -> OutboundMessageDraft | None:
        match_key = waiting_match_key(demand.demand_id, game.game_id)
        existing = self._existing_draft(match_key)
        if existing is not None:
            self.trace_recorder.record(
                trace_id,
                "waiting_match_dispatch_deduplicated",
                {"match_key": match_key, "draft_id": existing.draft_id},
            )
            return existing

        trigger = SystemTriggerMessage(
            trigger_id=match_key,
            trigger_type="waiting_demand_match_found",
            conversation_id=demand.conversation_id,
            sender_id=demand.sender_id,
            sender_name=demand.sender_name,
            payload={
                "waiting_demand": _waiting_demand_context(demand),
                "game": public_match_game_context(game),
                "response_contract": {
                    "must": "展示档位、烟况、时间和当前人数/缺口，并征求客户是否参加",
                    "must_not": "不得直接把客户加入局，不得泄露参与者姓名或后台执行细节",
                    "next_step": "等待客户明确确认后，后续普通用户消息才能调用 join_game",
                },
            },
        )
        trigger_trace_id = f"{trace_id}:waiting:{demand.demand_id}"
        self.trace_recorder.record(
            trace_id,
            "waiting_match_dispatch_started",
            {
                "match_key": match_key,
                "target_conversation_id": demand.conversation_id,
                "target_sender_id": demand.sender_id,
                "trigger_trace_id": trigger_trace_id,
            },
        )
        result = self.system_trigger_handler(trigger, trace_id=trigger_trace_id)
        reply = str(result.final_reply or "").strip()
        if not reply:
            raise RuntimeError("system-trigger Agent returned an empty waiting-match notification")

        drafts, transitions = self.store.create_outbound_message_drafts(
            conversation_id=demand.conversation_id,
            drafts=[
                {
                    "recipient_id": demand.sender_id,
                    "recipient_name": demand.sender_name,
                    "channel": str(demand.demand.get("source_channel") or "internal"),
                    "message_text": reply,
                    "purpose": "waiting_match_notification",
                    "metadata": {
                        "source": "waiting_match_trigger",
                        "purpose": "waiting_match_notification",
                        "game_id": game.game_id,
                        "waiting_demand_id": demand.demand_id,
                        "waiting_match_key": match_key,
                        "system_trigger_id": trigger.trigger_id,
                        "customer_visible_processing": "normal_agent_loop",
                    },
                }
            ],
            trace_id=trigger_trace_id,
        )
        for transition in transitions:
            self.trace_recorder.record(trigger_trace_id, "state_transition", transition.to_dict())
        draft = drafts[0]
        self.trace_recorder.record(
            trace_id,
            "waiting_match_dispatched",
            {
                "match_key": match_key,
                "draft_id": draft.draft_id,
                "trigger_trace_id": trigger_trace_id,
            },
        )
        return draft

    def _existing_draft(self, match_key: str) -> OutboundMessageDraft | None:
        return next(
            (
                draft
                for draft in self.store.outbound_message_drafts.values()
                if str(draft.metadata.get("waiting_match_key") or "") == match_key
            ),
            None,
        )


@dataclass(slots=True)
class MatchTrigger:
    """Claim compatible waiting demands after a game mutation succeeds."""

    store: AgentStore
    dispatcher: MatchDispatcher
    trace_recorder: Any

    def __call__(self, event: HookEvent) -> None:
        """Hook entrypoint for successful create/exit tool results."""

        result = dict(event.payload.get("result") or {})
        name = str(result.get("name") or "")
        if name not in {"create_game", "record_candidate_reply"}:
            return
        if not result.get("called") or not result.get("allowed") or result.get("error"):
            return
        if name == "record_candidate_reply":
            payload = dict(result.get("result") or {})
            if str(payload.get("recorded_status") or "") != "declined":
                return
        game_payload = dict((result.get("result") or {}).get("game") or {})
        game_id = str(game_payload.get("game_id") or "")
        if not game_id:
            return
        self.match_game(game_id, trace_id=event.trace_id, source_tool=name)

    def match_game(self, game_id: str, *, trace_id: str, source_tool: str) -> list[str]:
        """Scan, atomically claim, and dispatch all compatible demands."""

        expired = self.store.expire_stale_demands(at=now(), trace_id=trace_id)
        if expired:
            self.trace_recorder.record(
                trace_id,
                "waiting_demands_expired",
                {"demand_ids": [item.demand_id for item in expired], "source": "match_trigger"},
            )
        game = self.store.require_game(game_id)
        if game.status not in {GameStatus.FORMING, GameStatus.INVITING} or game.remaining_seats() <= 0:
            return []

        active = self.store.list_active_demands(at=now())
        self.trace_recorder.record(
            trace_id,
            "waiting_match_scan_started",
            {
                "game_id": game.game_id,
                "source_tool": source_tool,
                "active_demand_count": len(active),
            },
        )
        matched_ids: list[str] = []
        for demand in active:
            reason = waiting_demand_mismatch_reason(demand, game)
            if reason:
                self.trace_recorder.record(
                    trace_id,
                    "waiting_match_skipped",
                    {"game_id": game.game_id, "demand_id": demand.demand_id, "reason": reason},
                )
                continue
            claimed = self.store.claim_waiting_demand_match(demand.demand_id, game.game_id, at=now())
            if claimed is None:
                self.trace_recorder.record(
                    trace_id,
                    "waiting_match_claim_lost",
                    {"game_id": game.game_id, "demand_id": demand.demand_id},
                    level="WARN",
                )
                continue
            self.trace_recorder.record(
                trace_id,
                "waiting_match_claimed",
                {
                    "game_id": game.game_id,
                    "demand_id": claimed.demand_id,
                    "old_status": "active",
                    "new_status": "matched",
                },
            )
            try:
                self.dispatcher.dispatch_match(claimed, game, trace_id=trace_id)
            except Exception as exc:
                self.store.release_waiting_demand_match(claimed.demand_id, game.game_id)
                self.trace_recorder.record(
                    trace_id,
                    "waiting_match_dispatch_failed",
                    {
                        "game_id": game.game_id,
                        "demand_id": claimed.demand_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "claim_released": True,
                    },
                    level="ERROR",
                )
                continue
            matched_ids.append(claimed.demand_id)
        self.trace_recorder.record(
            trace_id,
            "waiting_match_scan_completed",
            {"game_id": game.game_id, "matched_demand_ids": matched_ids},
        )
        return matched_ids


def waiting_demand_mismatch_reason(demand: WaitingDemand, game: Game) -> str:
    """Return a stable mismatch reason or an empty string when compatible."""

    if any(item.customer_id == demand.sender_id for item in game.participants):
        return "customer_already_in_game"
    expected_stake = _normalize_stake(demand.demand.get("stake"))
    actual_stake = _normalize_stake(game.requirement.get("stake"))
    if expected_stake and expected_stake != actual_stake:
        return "stake_mismatch"
    expected_smoke = _normalize_smoke(demand.demand.get("smoke_preference"))
    actual_smoke = _normalize_smoke(game.requirement.get("smoke_preference"))
    if expected_smoke != "any" and expected_smoke and expected_smoke != actual_smoke:
        return "smoke_mismatch"
    if not _time_preference_matches(str(demand.demand.get("time_preference") or ""), game):
        return "time_mismatch"
    participant_tokens = {
        token
        for item in game.participants
        for token in (str(item.customer_id).strip(), str(item.display_name).strip())
        if token
    }
    for constraint in demand.demand.get("extra_constraints") or []:
        avoided = _avoided_player_token(str(constraint))
        if avoided and any(avoided in token or token in avoided for token in participant_tokens):
            return "relationship_constraint"
    return ""


def public_match_game_context(game: Game) -> dict[str, Any]:
    """Expose only public game facts; participant identities never cross here."""

    remaining = game.remaining_seats()
    claimed = max(0, game.seats_total - remaining)
    return {
        "game_id": game.game_id,
        "stake": str(game.requirement.get("stake") or ""),
        "smoke": _normalize_smoke(game.requirement.get("smoke_preference")),
        "smoke_label": _smoke_label(game.requirement.get("smoke_preference")),
        "time_label": _time_label(game),
        "planned_start_at": game.planned_start_at.isoformat() if game.planned_start_at else None,
        "current_player_count": claimed,
        "remaining_seats": remaining,
        "shortage_label": _shortage_label(claimed, remaining),
    }


def waiting_match_key(demand_id: str, game_id: str) -> str:
    return f"waiting-match:{demand_id}:{game_id}"


def handle_waiting_expiration_task(
    store: AgentStore,
    task: ScheduledAgentTask,
    *,
    trace_id: str,
    trace_recorder: Any,
) -> list[str]:
    """Expire stale demands and durably enqueue the next 60-second sweep."""

    if task.task_type != WAITING_DEMAND_EXPIRY_TASK_TYPE or task.aggregate_type != "waiting_list":
        raise ValueError(f"unsupported waiting-list task: {task.task_type}/{task.aggregate_type}")
    stamp = max(now(), task.due_at)
    expired = store.expire_stale_demands(at=stamp, trace_id=trace_id)
    next_task, transition = store.ensure_waiting_demand_expiration_task(
        due_at=next_waiting_expiry_due(stamp),
        trace_id=trace_id,
    )
    trace_recorder.record(
        trace_id,
        "waiting_demand_expiration_sweep",
        {
            "task_id": task.task_id,
            "expired_demand_ids": [item.demand_id for item in expired],
            "next_task": next_task.to_dict(),
            "next_task_transition": transition.to_dict() if transition else None,
        },
    )
    return [item.demand_id for item in expired]


def _waiting_demand_context(demand: WaitingDemand) -> dict[str, Any]:
    return {
        "demand_id": demand.demand_id,
        "stake": demand.demand.get("stake"),
        "smoke_preference": demand.demand.get("smoke_preference"),
        "time_preference": demand.demand.get("time_preference"),
        "extra_constraints": list(demand.demand.get("extra_constraints") or []),
        "expires_at": demand.expires_at.isoformat(),
    }


def _normalize_stake(value: object) -> str:
    text = str(value or "").strip().replace("，", ".").replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return text.lower()
    return f"{number:g}"


def _normalize_smoke(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "无烟": "no_smoking",
        "no_smoke": "no_smoking",
        "no_smoking": "no_smoking",
        "有烟": "smoking",
        "烟": "smoking",
        "smoking": "smoking",
        "不限": "any",
        "都可": "any",
        "烟都可": "any",
        "any": "any",
        "": "any",
    }
    return aliases.get(text, text)


def _smoke_label(value: object) -> str:
    return {"no_smoking": "无烟", "smoking": "有烟", "any": "烟都可"}.get(
        _normalize_smoke(value),
        str(value or ""),
    )


def _time_label(game: Game) -> str:
    if game.planned_start_at is not None:
        return game.planned_start_at.strftime("%H:%M")
    value = str(game.requirement.get("start_time") or "")
    if value:
        return value
    return "人齐开"


def _time_preference_matches(preference: str, game: Game) -> bool:
    text = str(preference or "").strip()
    if not text or text in {"不限", "都可", "尽快", "人齐开"}:
        return True
    start = game.planned_start_at
    if start is None:
        return True
    current = now()
    if "明天" in text and start.date() != (current + timedelta(days=1)).date():
        return False
    if "今晚" in text and (start.date() != current.date() or start.hour < 17):
        return False
    if "上午" in text and start.hour >= 12:
        return False
    if "下午" in text and not 12 <= start.hour < 18:
        return False
    if "晚上" in text and start.hour < 17:
        return False
    return True


def _avoided_player_token(constraint: str) -> str:
    text = str(constraint or "").strip()
    for pattern in (
        r"不(?:想)?和(.+?)(?:一起)?打",
        r"不要(?:和)?(.+?)(?:一起)?打",
        r"避开(.+)",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ""


def _shortage_label(claimed: int, remaining: int) -> str:
    chinese = {0: "零", 1: "一", 2: "二", 3: "三", 4: "四"}
    return f"{chinese.get(claimed, str(claimed))}缺{chinese.get(remaining, str(remaining))}"


__all__ = [
    "MatchDispatcher",
    "MatchTrigger",
    "OutboundDispatcher",
    "handle_waiting_expiration_task",
    "public_match_game_context",
    "waiting_demand_mismatch_reason",
    "waiting_match_key",
]
