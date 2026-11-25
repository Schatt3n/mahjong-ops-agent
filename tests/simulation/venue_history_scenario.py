"""让角色从已投递的真实群聊前缀自主续演，保留活动和店家处理证据。"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
import json

from mahjong_agent_runtime.models import using_clock

from .venue_actors import ActorAction, ActorContext, MockActor, ModelActor
from .venue_channel import AGENT_ID, MessageEnvelope
from .venue_dialogues import _live_actor_action, _model_failure
from .venue_history_seed import HistoricalGroupSeed
from .venue_judging import review_dialogue


class HistoricalGroupFixtureActor(MockActor):
    """离线验证旧群消息触发查询，真实角色另由 OpenCode 自行决定。"""

    @staticmethod
    def _choose(context, projected, event, rng):
        delivery = projected.get("delivery") or {}
        if str(delivery.get("message_id", "")).startswith("history:"):
            text = "刚才群里说的那桌条件确认了吗？我先问问。"
            if context.persona.temperament.purpose == "socialize":
                return ActorAction("send", text, "group", context.group_targets[0])
            return ActorAction("send", text)
        return MockActor._choose(context, projected, event, rng)


class VenueHistoricalGroupScenario:
    """旧消息只投递一次，切点后的行为才进入真实群聊和私聊入口。"""

    def __init__(self, *, history: HistoricalGroupSeed, people, actor, judge, channel, run_dir, start_at,
                 active_users: int, dialogue_turns: int, judge_users: int):
        """为源群友分配稳定的沙盒账号，其余顾客可作为同群旁观者参与。"""
        self.history, self.people = history, tuple(people)
        self.actor, self.judge, self.channel = actor, judge, channel
        self.run_dir, self.anchor = run_dir, start_at.replace(hour=history.timeline_start_local.hour,
            minute=history.timeline_start_local.minute, second=history.timeline_start_local.second)
        self.active_users, self.dialogue_turns, self.judge_users = active_users, dialogue_turns, judge_users
        self.members = dict(zip(history.actor_aliases, self.people, strict=False))

    def run(self) -> dict:
        """回灌已知前缀，记录同群顾客从切点起的自主动作和因果续轮。"""
        store = self.channel.runtime.store
        initial_games = set(store.games)
        prefix = self._deliver_prefix()
        past_only = set(store.games) == initial_games and not store.waiting_demands
        cutoff = prefix[-1]
        audience = [person for person in self.people if person.sender_id != cutoff.sender_id][:self.active_users]
        paused = {person.sender_id: cutoff.message_id for person in audience}
        turns = {person.sender_id: [] for person in audience}
        at = cutoff.sent_at
        with self.run_dir.joinpath("events.jsonl").open("w", encoding="utf-8") as events:
            for index in range(1, self.dialogue_turns + 1):
                if not paused:
                    break
                next_round = {}
                for person in audience:
                    cause = paused.get(person.sender_id)
                    if cause is None:
                        continue
                    at += timedelta(seconds=1)
                    row = self._decide(person, at=at, cause=cause, index=index)
                    turns[person.sender_id].append(row)
                    events.write(json.dumps(row, ensure_ascii=False) + "\n")
                    events.flush()
                    if row.get("reply_message_id") and not row.get("error_type"):
                        next_round[person.sender_id] = row["reply_message_id"]
                paused = next_round

        records = self.channel.events()
        with self.run_dir.joinpath("chat.jsonl").open("w", encoding="utf-8") as chat:
            for record in records:
                chat.write(json.dumps(record, ensure_ascii=False) + "\n")
        rows = [row for items in turns.values() for row in items]
        reviews = [] if self.judge is None else [
            review_dialogue(self.judge, customer_id=customer, turns=items, records=records, role_only=True,
                public_history=True, additional_instructions=(
                    "这是模拟顾客从已实际投递的真人群聊继续。group_history 是当时已见的公开消息。"
                    "只根据这些来源、本人偏好和之后实际收到的回复判断动作，不要求和真人后续一致。"
                    "群友表达愿意打无烟本身没有改变原帖烟况或占用席位。顾客可以等待、问清条件或私聊店家。"
                )) for customer, items in list(turns.items())[:self.judge_users]
        ]
        sent = [row for row in rows if row.get("message_id")]
        by_id = {row["message_id"]: row for row in records}
        action_counts = Counter(row["decision"]["action"]["action"] for row in rows if row["decision"].get("action"))
        failures = [row for row in rows if row.get("error_type")]
        checks = {
            "historical_prefix_delivery_matches_source": all(
                by_id.get(message.message_id, {}).get("direction") == "history"
                and by_id[message.message_id]["text"] == message.text
                and by_id[message.message_id]["quoted_message_id"] == message.quoted_message_id
                and datetime.fromisoformat(by_id[message.message_id]["sent_at"]) == message.sent_at for message in prefix
            ) and [row["message_id"] for row in records if row["direction"] == "history"] == [m.message_id for m in prefix],
            "historical_messages_did_not_reexecute_business": past_only,
            "role_saw_its_actual_delivered_trigger": bool(rows) and all(row["visible_cause_verified"] for row in rows),
            "generated_messages_delivered_as_decided": bool(sent) and all(
                (message := by_id.get(row["message_id"], {})).get("sender_id") == row["actor_id"]
                and message["text"] == row["decision"]["action"]["text"] and message["kind"] == row["channel"]
                and message["recipient_id"] == row["decision"]["action"].get("recipient_id", AGENT_ID)
                and bool(row.get("channel_trace_id")) for row in sent
            ),
            "requested_guests_evaluated_at_cutoff": len(audience) == self.active_users and all(turns.values()),
        }
        model_actions = sum(_live_actor_action(row) for row in rows)
        if isinstance(self.actor, ModelActor):
            checks["customer_decisions_from_model"] = model_actions == len(rows)
        return {
            "people": len(self.people), "targeted_users": len(audience), "workers": 1, "max_turns": self.dialogue_turns,
            "checks": checks, "chat_events": len(records), "real_customer_actions": model_actions,
            "failed_events": [row["event_id"] for row in failures],
            "model_transport_errors": [{"event_id": row["event_id"], "http_statuses": row.get("http_statuses", [])}
                                       for row in failures if row["error_type"] in {"transport_error", "runtime_transport_error", "timeout"}],
            "historical_continuation": {
                "case": self.history.case_id, "source_date": self.history.capture_date.isoformat(),
                "cutoff": self.history.cutoff, "prefix_messages": len(prefix), "held_out_messages": self.history.held_out_count,
                "actions": dict(action_counts), "sent_channels": dict(Counter(row["channel"] for row in sent)),
                "causal_followups": sum(row["caused_by"] != cutoff.message_id for row in rows),
            },
            "semantic_samples": reviews,
            "independent_judge": None if self.judge is None else {
                "status": "valid" if reviews and all(row["status"] == "valid" for row in reviews) else "partial",
                "verdict": "fail" if any(row["verdict"] == "fail" for row in reviews) else
                    "pass" if reviews and all(row["verdict"] == "pass" for row in reviews) else "inconclusive",
            },
        }

    def _deliver_prefix(self) -> list[MessageEnvelope]:
        """把经验证的群聊原话和引用按相对时刻写入收件账本。"""
        envelopes = []
        for row in self.history.prefix:
            envelope = MessageEnvelope(
                f"history:{row.message_alias}", self.members[row.actor_alias].sender_id, row.text or "[表情]",
                self.anchor + timedelta(seconds=row.offset_seconds), "group", self.channel.room_id,
                quoted_message_id=f"history:{row.quoted_message_alias}" if row.quoted_message_alias else None,
            )
            self.channel.deliver_group_history(envelope)
            envelopes.append(envelope)
        return envelopes

    def _decide(self, person, *, at: datetime, cause: str, index: int) -> dict:
        """一轮先读取真实可见来话，按角色产出的正文和渠道送达，保留后续回执。"""
        actor_id = person.sender_id
        event_id = f"historical:{actor_id}:{index}"
        mentions = {f"history:{row.message_alias}": tuple(self.members[alias].sender_id for alias in row.mentioned_actor_aliases)
                    for row in self.history.prefix if row.mentioned_actor_aliases}
        history = self.channel.events_for(actor_id)
        history = [{**item, "details": {"mentions": ",".join(mentions[item["message_id"]])}}
                   if item["message_id"] in mentions else item for item in history]
        context = ActorContext(person, at.isoformat(), history=history,
            events=[{"event_id": event_id, "kind": "delivered", "caused_by": cause, "at": at.isoformat()}],
            group_targets=(self.channel.room_id,), visible_groups=(self.channel.room_id,))
        row = {"event_id": event_id, "actor_id": actor_id, "virtual_at": at.isoformat(), "caused_by": cause}
        with using_clock(lambda: at):
            decision = self.actor.decide(context, trace_id=f"role:{event_id}")
            delivery = decision.context.get("delivery") or {}
            row.update(decision=decision.to_dict(), visible_cause_verified=delivery.get("message_id") == cause
                       and all(datetime.fromisoformat(item["sent_at"]) <= at for item in history))
            if decision.action is None:
                return dict(row, error_type=decision.status, failure_stage="role")
            if decision.action.action != "send":
                return row
            action = decision.action
            kind = action.kind or "private"
            message_id = f"customer:{event_id}"
            row.update(channel=kind, message_id=message_id)
            try:
                result = self.channel.send(MessageEnvelope(message_id, actor_id, action.text, at, kind,
                    action.recipient_id or AGENT_ID))
            except (RuntimeError, ValueError, OSError) as exc:
                return dict(row, error_type=type(exc).__name__, error=str(exc), failure_stage="delivery")
        outputs = [item for item in result.get("archive_outputs", ()) if item["kind"] == "group"
                   or item["recipient_id"] == actor_id]
        row.update(channel_trace_id=result["trace_id"], runtime_status=result["status"],
                   tool_results=result.get("tool_results", []), outputs=outputs,
                   reply_message_id=outputs[-1]["message_id"] if outputs else None)
        if kind == "private" or result.get("group_processing", {}).get("runtime_results"):
            row["trace_id"] = result["trace_id"]
        if result["status"] in {"needs_help", "needs_human", "failed"}:
            trace = self.channel.runtime.trace_recorder.get_trace(result["trace_id"])
            failure = _model_failure([item.content for item in trace if item.step == "llm_error"])
            row.update(failure or {"error_type": "runtime_needs_help"}, failure_stage="agent")
        return row
