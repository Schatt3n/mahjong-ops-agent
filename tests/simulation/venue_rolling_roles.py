"""用同群收发回执和私有日历唤醒角色，跨营业日保留独立的决策与收件箱。"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock

from mahjong_agent_runtime.models import using_clock

from .venue_actors import ActorContext, ModelActor
from .venue_channel import AGENT_ID, MessageEnvelope
from .venue_dialogues import _live_actor_action, _model_failure
from .venue_judging import review_dialogue
from .venue_role_journal import RoleEvent, VenueRoleJournal
from .venue_world import VenueWorld


_WORLD_WAKES = {"rain", "traffic"}
_RECEIPT_WAKES = {"private_receipt", "group_attention"}
_WAIT_FOR_REPLY = timedelta(minutes=2)
_GROUP_ATTENTION = 2


class VenueRollingRoles:
    """持久化每人回合，角色先决定，再让同一波来话逐条走正式入口。"""

    def __init__(self, *, people, actor, judge, channel, run_dir: Path, start_at: datetime, seed: int,
                 active_users: int, workers: int, dialogue_turns: int, judge_users: int) -> None:
        """从独立沙盒读取群成员、拟启动人数、并发上限及每人的续聊预算。"""
        self.people = tuple(people)
        if (not 1 <= active_users <= len(self.people) or workers < 1 or dialogue_turns < 2
                or judge_users < 1 or len({person.sender_id for person in self.people}) != len(self.people)):
            raise ValueError("滚动角色需要独立的演员、有效的并发人数和续聊预算")
        self.actor, self.judge, self.channel = actor, judge, channel
        self.run_dir, self.start_at, self.seed = Path(run_dir), start_at, seed
        self.workers, self.dialogue_turns, self.judge_users = workers, dialogue_turns, judge_users
        self.actors = {person.sender_id: person for person in self.people[:active_users]}
        self.owner_friends = frozenset(channel.owner_friends)
        self.world = VenueWorld(seed, start_at)
        self.journal = VenueRoleJournal(self.run_dir / "role_events.sqlite3")
        self._private_world: dict[tuple, dict] = {}
        self._receipts: dict[tuple[str, str], str] = {}
        self._receipt_counts: Counter = Counter()
        self._overlap_lock = Lock()
        self._inflight_decisions = self._inflight_deliveries = 0

    def run(self) -> dict:
        """从最早未办结的事件继续，决策屏障之后才开始该波投递。"""
        with self.journal.run_owner(), ThreadPoolExecutor(max_workers=self.workers) as pool:
            self._seed_timers()
            self._restore_wakes()
            while wave := self.journal.claim_wave(self.workers):
                # 同一波决定的可见消息在各线程内读取，直到全部决定落盘都不开始送达。
                planned = tuple(pool.map(self._plan, wave))
                # 不同人的私聊可并行提交。共享群入口与后续唤醒保持稳定顺序。
                independent = [event for event in planned
                               if ((event.decision or {}).get("decision", {}).get("action") or {}).get("kind") != "group"]
                delivered = dict(zip((event.event_id for event in independent), pool.map(self._dispatch, independent), strict=True))
                for event in planned:
                    result, receipts = delivered[event.event_id] if event.event_id in delivered else self._dispatch(event)
                    if receipts:
                        result["scheduled_wakes"] = self._wake_from_deliveries(event, receipts)
                    self.journal.finish(event, result)
                    if event.kind in _WORLD_WAKES:
                        self._private_world[self._day_key(event.actor_id, event.due_at)] = event.details

            events = self.journal.events()
            records = self.channel.events()
            rows = [{**item.decision, **item.result} for item in events if item.status == "finished"]
            for name, replay in (("events", rows), ("chat", records)):
                with self.run_dir.joinpath(f"{name}.jsonl").open("w", encoding="utf-8") as handle:
                    for row in replay:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            return self._report(events, rows, records)

    def _day_key(self, actor_id: str, at: datetime) -> tuple:
        """收件预算和天气状态都按本人及门店时区的营业日计算。"""
        return actor_id, at.astimezone(self.start_at.tzinfo).date()

    def _seed_timers(self) -> None:
        """同一个演练目录能重放计划，只补齐尚未入账的两日私人唤醒。"""
        plan = {"version": 1, "actors": list(self.actors), "seed": self.seed, "start_at": self.start_at.isoformat(),
                "room_id": self.channel.room_id, "workers": self.workers, "dialogue_turns": self.dialogue_turns}
        plan_path = self.run_dir / "rolling_roles_plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
                raise ValueError("滚动角色的恢复参数与原运行计划不同")
        else:
            staged = plan_path.with_suffix(".tmp")
            staged.write_text(json.dumps(plan, ensure_ascii=False) + "\n", encoding="utf-8")
            staged.replace(plan_path)
        previous_world_actor = None
        for index, disturbance in enumerate(("rain", "traffic")):
            day = self.start_at + timedelta(days=index)
            candidates = []
            for person in self.actors.values():
                availability = self.world.availability(person, at=day)
                at = availability.free_from + timedelta(minutes=5)
                if index == 0:
                    at = max(at, self.start_at)
                daily = self.world.role_event(person, "session_open", at)
                self.journal.enqueue(actor_id=person.sender_id, kind="timer", origin_id=day.date().isoformat(),
                                     due_at=at, details=daily["details"])
                weather_at = at + timedelta(minutes=20)
                if availability.available_for(weather_at + timedelta(minutes=10)):
                    candidates.append((person, weather_at))
            if not candidates:
                continue
            person, weather_at = min(candidates, key=lambda pair: (pair[0].sender_id == previous_world_actor,
                self._attention_rank(f"world:{day.date().isoformat()}", pair[0].sender_id)))
            previous_world_actor = person.sender_id
            private = self.world.role_event(person, disturbance, weather_at,
                scheduled_start=weather_at + timedelta(minutes=10) if disturbance == "traffic" else None)
            self.journal.enqueue(actor_id=person.sender_id, kind=disturbance, origin_id=day.date().isoformat(),
                                 due_at=weather_at, details=private["details"])

    def _restore_wakes(self) -> None:
        """从已有事件恢复私人状态与当日预算，进程中断不会重新分配注意力。"""
        self._private_world.clear()
        self._receipts.clear()
        self._receipt_counts.clear()
        for event in self.journal.events():
            key = self._day_key(event.actor_id, event.due_at)
            if event.kind in _WORLD_WAKES and event.status == "finished":
                self._private_world[key] = event.details
            if event.kind in _RECEIPT_WAKES:
                self._receipts[event.actor_id, event.origin_id] = event.event_id
                self._receipt_counts[key, event.kind] += 1

    def _history(self, event: RoleEvent) -> list[dict]:
        """保留各自群聊和私聊的近期消息，以及实际触发及可见引用链。"""
        visible = [row for row in self.channel.events_for(event.actor_id)
                   if datetime.fromisoformat(row["sent_at"]) <= event.due_at]
        by_id = {row["message_id"]: row for row in visible}
        ids = {row["message_id"] for kind in ("group", "private")
               for row in [entry for entry in visible if entry["kind"] == kind][-16:]}
        reference = event.caused_by
        retained = set()
        while reference in by_id and reference not in retained:
            retained.add(reference)
            row = by_id[reference]
            reference = row.get("quoted_message_id") or row.get("source_message_id")
        ids.update(retained)
        return [{**row, "issuer": "self"} if row["sender_id"] == event.actor_id else row
                for row in visible if row["message_id"] in ids]

    def _plan(self, event: RoleEvent) -> RoleEvent:
        """角色只看自己的日程、可见消息和真实来源；已决定的动作必须复用。"""
        if event.decision is not None:
            return event
        person = self.actors[event.actor_id]
        at = event.due_at.astimezone(self.start_at.tzinfo)
        if event.kind in _WORLD_WAKES:
            private = dict(event.details)
        else:
            world_change = self._private_world.get(self._day_key(event.actor_id, at))
            private = dict(world_change) if world_change is not None else self.world.role_event(person, "session_open", at)["details"]
        private.update({key: value for key, value in event.details.items() if key != "depth"})
        if at >= datetime.fromisoformat(private["free_until"]):
            private["can_arrive"] = False
            private.pop("earliest_arrival", None)
        kind = event.kind if event.kind in _WORLD_WAKES else "delivered" if event.caused_by else "activity"
        if kind == "activity":
            # 定时唤醒是自己的空闲安排，不能捏造一条老板来话作为原因。
            private["activity"] = "open_dialogue"
        elif kind in _WORLD_WAKES:
            private["case_intent"] = "仅知道自己的出行变化。按已有聊天决定是否需要联系店家或群友。"
        context = ActorContext(person, at.isoformat(), history=self._history(event), events=[{
            "event_id": event.event_id, "kind": kind, "caused_by": event.caused_by, "at": at.isoformat(),
            "visible_to": (event.actor_id,), "details": private,
        }], private_targets=(AGENT_ID,) if event.actor_id in self.owner_friends else (),
            group_targets=(self.channel.room_id,), visible_groups=(self.channel.room_id,))
        with self._overlap_lock:
            self._inflight_decisions += 1
            inflight = self._inflight_decisions
        try:
            with using_clock(lambda: at):
                decision = self.actor.decide(context, trace_id=f"role:{event.event_id}")
        finally:
            with self._overlap_lock:
                self._inflight_decisions -= 1
        delivery = decision.context.get("delivery") or {}
        return self.journal.save_decision(event, {
            "event_id": event.event_id, "actor_id": event.actor_id, "virtual_at": at.isoformat(),
            "caused_by": event.caused_by, "wake_kind": event.kind, "decision": decision.to_dict(),
            "visible_cause_verified": delivery.get("message_id") == event.caused_by,
            "inflight_decisions_at_start": inflight,
        })

    def _dispatch(self, event: RoleEvent) -> tuple[dict, list[dict]]:
        """按已存动作投递，返回可供调度器有序处理的真实交付。"""
        planned = event.decision or {}
        decision = planned["decision"]
        if not planned["visible_cause_verified"]:
            return {"failure_stage": "role", "error_type": "invisible_wake_source"}, []
        action = decision.get("action")
        if not action or decision.get("status") != "valid":
            return {"failure_stage": "role", "error_type": decision.get("status", "invalid")}, []
        if action["action"] in {"wait", "leave"}:
            return {"runtime_status": action["action"]}, []
        if action["action"] != "send":
            return {"failure_stage": "role", "error_type": "arrival_without_venue_event"}, []
        kind, recipient = action.get("kind", "private"), action.get("recipient_id", AGENT_ID)
        targets = self.owner_friends if kind == "private" and recipient == AGENT_ID else set()
        if not (kind == "group" and recipient == self.channel.room_id or event.actor_id in targets):
            return {"failure_stage": "delivery", "error_type": "unauthorized_action_recipient"}, []
        at = event.due_at.astimezone(self.start_at.tzinfo)
        message_id = f"customer:{event.event_id}"
        row = {"channel": kind, "message_id": message_id}
        with self._overlap_lock:
            self._inflight_deliveries += 1
            row["inflight_deliveries_at_start"] = self._inflight_deliveries
        try:
            with using_clock(lambda: at):
                response = self.channel.send(MessageEnvelope(message_id, event.actor_id, action["text"], at, kind, recipient))
        except (RuntimeError, ValueError, OSError) as exc:
            return dict(row, error_type=type(exc).__name__, error=str(exc), failure_stage="delivery"), []
        finally:
            with self._overlap_lock:
                self._inflight_deliveries -= 1
        outputs = response.get("archive_outputs", [])
        visible = [item for item in outputs if item["kind"] == "group" or item["recipient_id"] == event.actor_id]
        row.update(channel_trace_id=response["trace_id"], runtime_status=response["status"],
                   tool_results=response.get("tool_results", []), outputs=visible,
                   reply_message_id=visible[-1]["message_id"] if visible else None,
                   delivery_deduplicated=response.get("delivery_deduplicated", False))
        if kind == "private" or response.get("group_processing", {}).get("runtime_results"):
            row["trace_id"] = response["trace_id"]
        if row["runtime_status"] in {"needs_help", "needs_human", "failed"}:
            traces = self.channel.runtime.trace_recorder.get_trace(response["trace_id"])
            failure = _model_failure([item.content for item in traces if item.step == "llm_error"])
            row.update(failure or {"error_type": "runtime_needs_help"}, failure_stage="agent")
        broadcasts = [{"kind": "group", "sender_id": event.actor_id, "message_id": message_id}] if kind == "group" else []
        return row, [*outputs, *broadcasts]

    def _attention_rank(self, source: str, actor_id: str) -> bytes:
        """固定来源编号的群内注意力不因进程重开或其他成员回复而漂移。"""
        return sha256(f"{self.seed}|{source}|{actor_id}".encode()).digest()

    def _wake_from_deliveries(self, parent: RoleEvent, messages: list[dict]) -> list[str]:
        """按真正收件范围安排回复和群内关注，控制每日扇出和转发深度。"""
        depth = parent.details.get("depth", 0) + 1
        if depth >= self.dialogue_turns:
            return []
        due = parent.due_at + _WAIT_FOR_REPLY
        if due.astimezone(self.start_at.tzinfo).date() > (self.start_at + timedelta(days=1)).date():
            return []
        scheduled = []
        for item in sorted(messages, key=lambda entry: entry["kind"] != "private"):
            message_id = item["message_id"]
            if item["kind"] == "private":
                audience = (item["recipient_id"],)
                kind = "private_receipt"
            else:
                audience = sorted((key for key in self.actors if key != item["sender_id"]),
                    key=lambda key: (self.actors[key].activity == "quiet", self._attention_rank(message_id, key)))[:_GROUP_ATTENTION]
                kind = "group_attention"
            for actor_id in audience:
                key = actor_id, message_id
                if saved := self._receipts.get(key):
                    scheduled.append(saved)
                    continue
                budget = self._day_key(actor_id, due), kind
                if (actor_id not in self.actors or actor_id == item["sender_id"]
                        or self._receipt_counts[budget] >= self.dialogue_turns - 1):
                    continue
                wake = self.journal.enqueue(actor_id=actor_id, kind=kind, origin_id=message_id,
                    due_at=due, caused_by=message_id, details={"depth": depth})
                self._receipts[key] = wake.event_id
                self._receipt_counts[budget] += 1
                scheduled.append(wake.event_id)
        return scheduled

    def _report(self, events: tuple[RoleEvent, ...], rows: list[dict], records: list[dict]) -> dict:
        """按独立决策、实际消息、因果唤醒和营业日分别验收与采样。"""
        chat = {row["message_id"]: row for row in records}
        by_actor = defaultdict(list)
        for row in rows:
            by_actor[row["actor_id"]].append(row)
        samples = self._review(by_actor, records)
        failures = [row for row in rows if row.get("error_type")]
        decided = Counter((row["decision"].get("action") or {}).get("action", "invalid") for row in rows)
        sent = [row for row in rows if row.get("channel_trace_id") and row.get("message_id") in chat]
        kinds = Counter(item.kind for item in events if item.status == "finished")
        days = Counter(item.actor_id for item in events if item.kind == "timer" and item.status == "finished")
        actors = sum(days[actor_id] == 2 for actor_id in self.actors)
        live = [row for row in rows if _live_actor_action(row)]
        own_prior_day = set()
        prior_day_private_receipt = set()
        mismatched_visibility = 0
        for row in rows:
            actor_id = row["actor_id"]
            at = datetime.fromisoformat(row["virtual_at"])
            for seen in row["decision"]["context"].get("history", ()):
                source = chat.get(seen.get("message_id"), {})
                if not source:
                    mismatched_visibility += 1
                    continue
                sent_at = datetime.fromisoformat(source["sent_at"])
                owns = actor_id in {source["sender_id"], source["recipient_id"]}
                if (sent_at > at or source["text"] != seen.get("text")
                        or (source["kind"] == "private" and not owns)):
                    mismatched_visibility += 1
                if owns and self._day_key(actor_id, sent_at)[1] < self._day_key(actor_id, at)[1]:
                    own_prior_day.add(actor_id)
                    if source["kind"] == "private" and source["recipient_id"] == actor_id:
                        prior_day_private_receipt.add(actor_id)
        checks = {
            "all_target_roles_reached_two_calendar_days": actors == len(self.actors),
            "role_events_all_finished": bool(events) and len(events) == len(rows),
            "role_message_sources_were_visible_at_decision": bool(rows) and all(row["visible_cause_verified"] for row in rows),
            "role_context_matches_visible_channel_history": mismatched_visibility == 0,
            "role_sends_match_delivered_channel_rows": len(sent) == decided["send"] and all(
                chat[row["message_id"]]["text"] == row["decision"]["action"]["text"]
                and chat[row["message_id"]]["sender_id"] == row["actor_id"]
                and chat[row["message_id"]]["recipient_id"] == row["decision"]["action"].get("recipient_id", AGENT_ID)
                and chat[row["message_id"]]["kind"] == row["channel"] for row in sent),
            "private_receipts_wake_only_their_recipient": all(
                chat[item.caused_by]["kind"] == "private" and chat[item.caused_by]["recipient_id"] == item.actor_id
                for item in events if item.kind == "private_receipt" and item.caused_by in chat)
                and all(item.caused_by in chat for item in events if item.kind == "private_receipt"),
        }
        if isinstance(self.actor, ModelActor):
            checks["customer_decisions_from_model"] = len(live) == len(rows)
        valid_samples = bool(samples) and all(sample["status"] == "valid" for sample in samples)
        independent = None if self.judge is None else {"status": "valid" if valid_samples else "partial",
            "verdict": "fail" if any(sample["verdict"] == "fail" for sample in samples) else
                "pass" if valid_samples and all(sample["verdict"] == "pass" for sample in samples) else "inconclusive"}
        return {
            "people": len(self.people), "targeted_users": len(self.actors), "workers": self.workers,
            "max_turns": self.dialogue_turns, "checks": checks, "chat_events": len(records),
            "real_customer_actions": len(live),
            "peak_inflight_deliveries": max((row.get("inflight_deliveries_at_start", 0) for row in rows), default=0),
            "peak_inflight_decisions": max((row.get("inflight_decisions_at_start", 0) for row in rows), default=0),
            "failed_events": [row["event_id"] for row in failures],
            "model_transport_errors": [{"event_id": row["event_id"], "http_statuses": row.get("http_statuses", [])}
                                       for row in failures if row["error_type"] in {"transport_error", "runtime_transport_error", "timeout"}],
            "rolling_roles": {"actors_with_two_calendar_days": actors, "decision_actors": len(by_actor),
                "real_decision_actors": len({row["actor_id"] for row in live}), "scheduled_events": len(events),
                "wake_kinds": dict(kinds), "actions": dict(decided),
                "sent_channels": dict(Counter(row["channel"] for row in sent)),
                "causally_woken_actors": len({item.actor_id for item in events if item.kind in _RECEIPT_WAKES}),
                "actors_with_own_prior_day_messages_in_context": len(own_prior_day),
                "actors_with_prior_day_private_receipts_in_context": len(prior_day_private_receipt),
                "out_of_scope_context_rows": mismatched_visibility,
                "reused_channel_receipts": sum(row.get("delivery_deduplicated", False) for row in sent),
                "semantic_review_customers": len(samples)},
            "semantic_samples": samples, "independent_judge": independent,
        }

    def _review(self, by_actor: dict, records: list[dict]) -> list[dict]:
        """选取可核对真实动态的角色，沉默与私有变动仍是有效评测对象。"""
        if self.judge is None:
            return []
        candidates = sorted(by_actor, key=lambda actor_id: (not any(row.get("caused_by") for row in by_actor[actor_id]),
            self._attention_rank("semantic-sample", actor_id)))
        return [review_dialogue(self.judge, customer_id=actor_id, turns=by_actor[actor_id], records=records,
                    role_only=True, delivered_history=True, additional_instructions=(
                        "这是同一位顾客的两日回合。区分每天的日历变化、实际引用和已送达私聊。"
                        "无参与意愿、天气不合适或话题与本人无关时，等待可以是有效决定。"
                    )) for actor_id in candidates[:self.judge_users]]
