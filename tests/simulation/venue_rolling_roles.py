"""用同群收发回执和私有日历唤醒角色，跨营业日保留独立的决策与收件箱。"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from sqlite3 import Error as SQLiteError
from threading import Lock

from mahjong_agent_runtime.models import using_clock

from .venue_acceptance import _delivered_replies, assess_pending_handling
from .venue_actors import ActorContext, ModelActor
from .venue_channel import AGENT_ID, MessageEnvelope
from .venue_dialogues import _live_actor_action, _model_failure
from .venue_judging import ROLLING_REVIEW_INSTRUCTIONS, review_dialogue, shop_reply_verdicts
from .venue_role_journal import RoleEvent, VenueRoleJournal
from .venue_rolling_board import VenueRollingBoard
from .venue_rolling_history import deliver_rolling_group_history
from .venue_rolling_play_link import PLAY_ROLE_KINDS, PLAY_TICK_KIND, VenueRollingPlayLink
from .venue_rolling_world import VenueRollingWorld
from .venue_world import VenueWorld


_WORLD_WAKES = {"rain", "traffic"}
_OVERTIME_WAKE = "overtime"
_RECEIPT_WAKES = {"private_receipt", "group_attention", "peer_receipt"}
_WAIT_FOR_REPLY = timedelta(minutes=2)
_ROLE_PURPOSES = {"find_table", "bring_friends", "ask_venue", "socialize"}
_ROLE_GOALS = {"lookup", "try_join", "organize_new", "wait_or_leave", "group"}
_BOARD_WAKE = "board_publication"
_LATE_ARRIVAL_DELIVERY_WAKE = "late_arrival_delivery"
_CONTROL_WAKES = {_BOARD_WAKE, PLAY_TICK_KIND, _LATE_ARRIVAL_DELIVERY_WAKE}


class RollingRoleTransportPaused(RuntimeError):
    """当前波在投递前遇到角色模型连接失败，等同源可用后从此处再试。"""

    filename = "provider_pause.json"

    def __init__(self, checkpoint: dict) -> None:
        """携带可展示的局部进度，完整请求仍保存在独立的 SQLite 账本。"""
        super().__init__("rolling role model transport failed before wave delivery")
        self.checkpoint = checkpoint


class VenueRollingRoles:
    """持久化每人回合，角色先决定，再让同一波来话逐条走正式入口。"""

    def __init__(self, *, people, actor, judge, channel, run_dir: Path, start_at: datetime, seed: int,
                 active_users: int, workers: int, dialogue_turns: int, judge_users: int, publish_boards: bool = False,
                 seed_public_history: bool = False, physical_play: bool = False, peer_dialogues: bool = False) -> None:
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
        self.group_members = channel.group_member_ids
        self.peer_dialogues = peer_dialogues
        self._regular_partners = {
            actor_id: tuple(dict.fromkeys(
                edge.target_sender_id for edge in person.relationships
                if edge.kind == "regular_partner" and edge.target_sender_id in self.actors and edge.target_sender_id != actor_id
            ))
            for actor_id, person in self.actors.items()
        } if peer_dialogues else {}
        self.world = VenueWorld(seed, start_at)
        self.board = VenueRollingBoard(channel) if publish_boards else None
        self.seed_public_history = seed_public_history
        self.journal = VenueRoleJournal(self.run_dir / VenueRoleJournal.filename)
        self.play = (VenueRollingPlayLink(VenueRollingWorld(channel.runtime.store, self.run_dir), channel, self.journal,
                    active_actors=self.actors, allow_controlled=not isinstance(actor, ModelActor)) if physical_play else None)
        self._private_world: dict[tuple, dict] = {}
        self._overtime_until: dict[str, datetime] = {}
        self._receipts: dict[tuple[str, str], str] = {}
        self._receipt_counts: Counter = Counter()
        self._overlap_lock = Lock()
        self._inflight_decisions = self._inflight_deliveries = 0

    def schedule_overtime(self, actor_id: str, *, origin_id: str, at: datetime, busy_until: datetime) -> RoleEvent:
        """明确启用仅本人可见的加班事件。重复来源复用事件，不产生入站或改动牌局。"""
        if actor_id not in self.actors or actor_id not in self.owner_friends:
            raise ValueError("加班事件要求已启用且能私聊店家的顾客")
        if at.utcoffset() is None or busy_until.utcoffset() is None or busy_until <= at:
            raise ValueError("加班事件需带时区，且忙完时刻须在事件之后")
        cutoff = (self.start_at + timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
        if not self.start_at <= at < cutoff:
            raise ValueError("加班事件需发生在当前演练内")
        return self.journal.enqueue(actor_id=actor_id, kind=_OVERTIME_WAKE, origin_id=origin_id, due_at=at,
            details={"work_busy_until": busy_until.astimezone(self.start_at.tzinfo).isoformat()})

    def run(self) -> dict:
        """从最早未办结的事件继续，决策屏障之后才开始该波投递。"""
        with self.journal.run_owner(), ThreadPoolExecutor(max_workers=self.workers) as pool:
            self._seed_timers()
            historical_opening = self._seed_history() if self.seed_public_history else None
            self._restore_wakes()
            if self.play is not None:
                attempts = self.journal.transport_attempts()
                # 暂停的波还没投递，恢复时先走原波的决策和投递，之后才推进同一时刻的房态。
                if not attempts or self.journal.get(attempts[-1]["event_id"]).status == "finished":
                    self.play.sync(self.journal.clock() or self.start_at)
            while True:
                self._schedule_next_board()
                self._schedule_next_late_arrival_delivery()
                wave = self.journal.claim_wave(self.workers)
                if not wave:
                    break
                # 同一波决定的可见消息在各线程内读取，直到全部决定落盘都不开始送达。
                planned = tuple(pool.map(self._plan, (event for event in wave if event.kind not in _CONTROL_WAKES)))
                if deferred := [event for event in planned if event.decision is None]:
                    self._pause_before_delivery(wave, planned, deferred)
                # 不同人的私聊可并行提交。共享群入口与后续唤醒保持稳定顺序。
                independent = [event for event in planned
                               if ((event.decision or {}).get("decision", {}).get("action") or {}).get("kind") != "group"]
                delivered = dict(zip((event.event_id for event in independent), pool.map(self._dispatch, independent), strict=True))
                for event in planned:
                    result, receipts = delivered[event.event_id] if event.event_id in delivered else self._dispatch(event)
                    if receipts:
                        result["scheduled_wakes"] = self._wake_from_deliveries(event, receipts)
                    self.journal.finish(event, result)
                    self._remember_private_world(event)
                for event in wave:
                    if event.kind == _BOARD_WAKE:
                        self._publish_board(event)
                    elif event.kind == _LATE_ARRIVAL_DELIVERY_WAKE:
                        self._deliver_late_arrival(event)
                    elif event.kind == PLAY_TICK_KIND:
                        self.play.finish_control(event)
                if self.play is not None:
                    self.play.sync(wave[0].due_at)

            journal = self.journal.events()
            events = tuple(event for event in journal if event.kind not in _CONTROL_WAKES)
            records = self.channel.events()
            rows = [{**item.decision, **item.result} for item in events if item.status == "finished"]
            for name, replay in (("events", rows), ("chat", records)):
                with self.run_dir.joinpath(f"{name}.jsonl").open("w", encoding="utf-8") as handle:
                    for row in replay:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            report = self._report(events, rows, records)
            if attempts := self.journal.transport_attempts():
                report["role_transport_recovery"] = {
                    "failed_requests": len(attempts), "affected_events": len({row["event_id"] for row in attempts}),
                }
                saved = {event.event_id: event.decision for event in events if event.status == "finished"}
                timeouts = Counter(
                    row["event_id"] for row in attempts if row["decision"].get("transport_timeout") is True
                    and row["decision"].get("raw") is None and row["decision"].get("action") is None
                )
                if timeouts:
                    report["model_timeout_retries"] = [
                        {"model_stage": "role", "event_id": event_id, "timeout_attempts": count,
                         "transport_recovered": self._same_input_timeout_recovered(event_id, attempts, saved)}
                        for event_id, count in timeouts.items()
                    ]
            if historical_opening is not None:
                report["rolling_history"] = historical_opening.audit() | {
                    "finished_attention_events": sum(event.caused_by == historical_opening.last_message_id
                                                     and event.status == "finished" for event in events),
                }
            if self.board is not None:
                publications = [item for item in journal if item.kind == _BOARD_WAKE]
                report["rolling_board"] = self.board.audit(rows, records) | {"publication_ticks": len(publications)}
                with self.run_dir.joinpath("board_events.jsonl").open("w", encoding="utf-8") as handle:
                    for item in publications:
                        handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
            if self.play is not None:
                report["rolling_play"] = self.play.report()
            return report

    @staticmethod
    def _same_input_timeout_recovered(event_id: str, attempts: tuple[dict, ...], saved: dict) -> bool:
        """续跑只认可原事件、同一可见输入和正式存档的有效模型决定。"""
        current = saved.get(event_id) or {}
        context = (current.get("decision") or {}).get("context")
        return _live_actor_action(current) and all(
            row["decision"].get("context") == context
            for row in attempts if row["event_id"] == event_id and row["decision"].get("transport_timeout") is True
        )

    def _pause_before_delivery(self, wave: tuple[RoleEvent, ...], planned: tuple[RoleEvent, ...],
                               deferred: list[RoleEvent]) -> None:
        """整波保留在同一个虚拟时刻，已成功的顾客决定和先前交付可安全续用。"""
        attempts = self.journal.transport_attempts()
        deferred_ids = {event.event_id for event in deferred}
        statuses = {item.get("http_status") for row in attempts if row["event_id"] in deferred_ids
                    for item in (row["decision"].get("diagnostics") or {}).get("attempts", ())
                    if type(item.get("http_status")) is int}
        checkpoint = {
            "status": "paused_transport", "scenario": "rolling-roles", "stage": "role_before_delivery",
            "virtual_at": wave[0].due_at.astimezone(self.start_at.tzinfo).isoformat(),
            "pending_event_ids": [event.event_id for event in wave],
            "transport_failure_event_ids": [event.event_id for event in deferred],
            "saved_peer_decisions": len(planned) - len(deferred),
            "archived_messages": len(self.channel.events()),
            "cumulative_failed_requests": len(attempts),
            "failed_http_statuses": sorted(statuses),
        }
        path = self.run_dir / RollingRoleTransportPaused.filename
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        raise RollingRoleTransportPaused(checkpoint)

    def _seed_history(self):
        """用已有群消息而非预设顾客话术轻量启动一批公开关注。"""
        opening = deliver_rolling_group_history(self.channel, roster_ids=(item.sender_id for item in self.people),
                                                active_actor_ids=self.actors, start_at=self.start_at)
        audience = self._group_audience(opening.last_message_id, opening.last_author_id, self.channel.room_id)
        for actor_id in audience:
            self.journal.enqueue(actor_id=actor_id, kind="group_attention", origin_id=opening.last_message_id,
                                 due_at=self.start_at, caused_by=opening.last_message_id, details={"depth": 0})
        return opening

    def _schedule_next_board(self) -> None:
        """把正式发布任务放入同一虚拟时钟，清理完第二个营业日即停。"""
        if self.board is None or (due := self.board.next_due()) is None:
            return
        at = max(due, self.journal.clock() or self.start_at)
        cutoff = (self.start_at + timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
        if at >= cutoff:
            return
        origin = due.astimezone(timezone.utc).isoformat()
        event = self.journal.enqueue(actor_id=AGENT_ID, kind=_BOARD_WAKE, origin_id=origin,
                                     due_at=at, details={"formal_due_at": origin})
        if event.status == "finished":
            raise RuntimeError("正式看板任务在既定时刻仍未推进")

    def _publish_board(self, event: RoleEvent) -> None:
        """续跑可恢复已交付快照并补唤醒，同一发布动作不算顾客决定。"""
        event = self.journal.save_decision(event, {"due_at": event.details["formal_due_at"]})
        at = event.due_at.astimezone(self.start_at.tzinfo)
        publication = self.board.publish_due(at)
        receipts = self.board.receipts_at(at)
        wakes = self._wake_from_deliveries(event, receipts)
        self.journal.finish(event, {"publication": publication, "delivered_board_ids": [row["message_id"] for row in receipts],
                                    "scheduled_wakes": wakes})

    def _schedule_next_late_arrival_delivery(self) -> None:
        """把迟到私聊待发或重试放入虚拟事件队列，包括没有人再发消息的情形。"""
        pending = self.channel.next_late_arrival_delivery()
        if pending is None:
            return
        due, task_id = pending
        at = max(due, self.journal.clock() or self.start_at)
        cutoff = (self.start_at + timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
        if at < cutoff:
            self.journal.enqueue(actor_id=AGENT_ID, kind=_LATE_ARRIVAL_DELIVERY_WAKE,
                origin_id=f"{task_id}:{due.astimezone(timezone.utc).isoformat()}", due_at=at, details={"task_id": task_id})

    def _deliver_late_arrival(self, event: RoleEvent) -> None:
        """只在沙盒真实收件后唤醒对应角色，重开时可复用交付证据。"""
        task_id = event.details["task_id"]
        event = self.journal.save_decision(event, {"task_id": task_id})
        at = event.due_at.astimezone(self.start_at.tzinfo)
        receipts = self.channel.deliver_late_arrival_notice(task_id, source_message_id=event.event_id, sent_at=at)
        wakes = self._wake_from_deliveries(event, receipts)
        self.journal.finish(event, {"delivered_ids": [row["message_id"] for row in receipts], "scheduled_wakes": wakes})

    def _day_key(self, actor_id: str, at: datetime) -> tuple:
        """收件预算和天气状态都按本人及门店时区的营业日计算。"""
        return actor_id, at.astimezone(self.start_at.tzinfo).date()

    def _seed_timers(self) -> None:
        """同一个演练目录能重放计划，只补齐尚未入账的两日私人唤醒。"""
        plan = {"version": 1, "actors": list(self.actors), "seed": self.seed, "start_at": self.start_at.isoformat(),
                "room_id": self.channel.room_id, "workers": self.workers, "dialogue_turns": self.dialogue_turns}
        if self.board is not None:
            plan["publish_boards"] = True
        if self.seed_public_history:
            plan["seed_public_history"] = True
        if self.play is not None:
            plan["physical_play"] = True
        if self.peer_dialogues:
            plan["peer_dialogues"] = True
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
        self._overtime_until.clear()
        self._receipts.clear()
        self._receipt_counts.clear()
        for event in self.journal.events():
            key = self._day_key(event.actor_id, event.due_at)
            if event.status == "finished":
                self._remember_private_world(event)
            if event.kind in _RECEIPT_WAKES:
                self._receipts[event.actor_id, event.origin_id] = event.event_id
                self._receipt_counts[key, event.kind] += 1

    def _remember_private_world(self, event: RoleEvent) -> None:
        """恢复独立的工作与天气来源，同日天气更新不能缩短已经出现的加班。"""
        if event.kind == _OVERTIME_WAKE:
            self._overtime_until[event.actor_id] = datetime.fromisoformat(event.details["work_busy_until"])
        elif event.kind in _WORLD_WAKES:
            self._private_world[self._day_key(event.actor_id, event.due_at)] = event.details

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

    def _private_targets(self, event: RoleEvent) -> tuple[tuple[str, ...], str | None]:
        """可联系本人认识的活跃搭子，当前可信收件还允许按原会话回复发送者。"""
        targets = (AGENT_ID,) if event.actor_id in self.owner_friends else ()
        if not self.peer_dialogues or event.kind == _OVERTIME_WAKE or event.kind in PLAY_ROLE_KINDS:
            return targets, None
        partners = self._regular_partners.get(event.actor_id, ())
        if event.kind == "peer_receipt" and event.caused_by:
            source = self.channel.completed_peer_message(event.caused_by)
            if (source is not None and source["recipient_id"] == event.actor_id and source["sender_id"] in self.actors
                    and source["sender_id"] != event.actor_id and datetime.fromisoformat(source["sent_at"]) <= event.due_at):
                sender = source["sender_id"]
                return tuple(dict.fromkeys((*targets, *partners, sender))), sender
        return (*targets, *partners), partners[0] if partners and event.kind == "timer" else None

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
        overtime = event.kind == _OVERTIME_WAKE
        busy_until = (datetime.fromisoformat(event.details["work_busy_until"]) if overtime
                      else self._overtime_until.get(event.actor_id))
        if busy_until is not None:
            commute = timedelta(minutes=private.get("travel_minutes") or 0)
            if at < busy_until + commute:
                private = self.world.apply_work_overtime(private, busy_until=busy_until)
        if overtime:
            private["event_condition"] = "unexpected_work_overtime"
            private["case_intent"] = (
                "突然加班。按本人实际还能否赴约及已有聊天决定先等、私聊店家说明晚到或退出。"
                "没有与同桌商量之前，别当成开场已经改了。"
            )
        if at >= datetime.fromisoformat(private["free_until"]):
            private["can_arrive"] = False
            private.pop("earliest_arrival", None)
        kind = event.kind if overtime or event.kind in _WORLD_WAKES or event.kind in PLAY_ROLE_KINDS else (
            "delivered" if event.caused_by else "activity")
        private_targets, peer_id = self._private_targets(event)
        if peer_id is not None:
            private["peer_id"] = peer_id
        if kind == "activity":
            # 定时唤醒是自己的空闲安排，不能捏造一条老板来话作为原因。
            private["activity"] = "open_dialogue"
            if peer_id is not None and person.temperament.purpose == "bring_friends":
                private["case_intent"] = "如果想和搭子一起打，可以先私下问本人时间和意愿。"
        elif kind in _WORLD_WAKES:
            private["case_intent"] = "仅知道自己的出行变化。按已有聊天决定是否需要联系店家或群友。"
        elif event.kind == "peer_receipt" and peer_id is not None:
            private["case_intent"] = "这是实际收到的搭子私信，可自行决定是否通过私聊回复本人。"
        private_cue = overtime or event.kind in PLAY_ROLE_KINDS
        groups = (self.channel.room_id,) if event.actor_id in self.group_members else ()
        context = ActorContext(person, at.isoformat(), history=self._history(event), events=[{
            "event_id": event.event_id, "kind": kind, "caused_by": event.caused_by, "at": at.isoformat(),
            "visible_to": (event.actor_id,), "details": private,
        }], private_targets=private_targets,
            group_targets=() if private_cue else groups, visible_groups=groups)
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
        planned = {
            "event_id": event.event_id, "actor_id": event.actor_id, "virtual_at": at.isoformat(),
            "caused_by": event.caused_by, "wake_kind": event.kind, "decision": decision.to_dict(),
            "visible_cause_verified": (delivery.get("message_id") == event.caused_by
                                       and (event.kind != "peer_receipt" or
                                            peer_id is not None and delivery.get("sender_id") == peer_id)),
            "inflight_decisions_at_start": inflight,
        }
        if decision.source == "model" and decision.status == "transport_error":
            self.journal.record_transport_attempt(event, planned)
            return event
        return self.journal.save_decision(event, planned)

    def _dispatch(self, event: RoleEvent) -> tuple[dict, list[dict]]:
        """按已存动作投递，返回可供调度器有序处理的真实交付。"""
        planned = event.decision or {}
        decision = planned["decision"]
        if not planned["visible_cause_verified"]:
            return {"failure_stage": "role", "error_type": "invisible_wake_source"}, []
        action = decision.get("action")
        if not action or decision.get("status") != "valid":
            return {"failure_stage": "role", "error_type": decision.get("status", "invalid")}, []
        if self.play is not None and (physical := self.play.role_result(event)) is not None:
            return physical, []
        if action["action"] in {"wait", "leave"}:
            return {"runtime_status": action["action"]}, []
        if action["action"] != "send":
            return {"failure_stage": "role", "error_type": "arrival_without_venue_event"}, []
        kind, recipient = action.get("kind", "private"), action.get("recipient_id", AGENT_ID)
        private_targets, _ = self._private_targets(event)
        if not (kind == "group" and event.kind != _OVERTIME_WAKE
                and recipient == self.channel.room_id and event.actor_id in self.group_members
                or kind == "private" and recipient in private_targets):
            return {"failure_stage": "delivery", "error_type": "unauthorized_action_recipient"}, []
        is_peer = kind == "private" and recipient != AGENT_ID
        at = event.due_at.astimezone(self.start_at.tzinfo)
        message_id = f"customer:{event.event_id}"
        message = MessageEnvelope(message_id, event.actor_id, action["text"], at, kind, recipient,
                                  action.get("quote_message_id"))
        row = {"channel": kind, "message_id": message_id}
        with self._overlap_lock:
            self._inflight_deliveries += 1
            row["inflight_deliveries_at_start"] = self._inflight_deliveries
        try:
            with using_clock(lambda: at):
                response = self.channel.send(message)
        except (RuntimeError, ValueError, OSError, SQLiteError) as exc:
            failure = dict(row, error_type=type(exc).__name__, error=str(exc), failure_stage="delivery")
            return (failure, []) if is_peer else self._record_failed_shop_dispatch(message, failure)
        finally:
            with self._overlap_lock:
                self._inflight_deliveries -= 1
        outputs = response.get("archive_outputs", [])
        visible = [item for item in outputs if item["kind"] == "group" or item["recipient_id"] == event.actor_id]
        row.update(channel_trace_id=response["trace_id"], runtime_status=response["status"],
                   tool_results=response.get("tool_results", []), outputs=visible,
                   reply_message_id=visible[-1]["message_id"] if visible else None,
                   delivery_deduplicated=response.get("delivery_deduplicated", False))
        if (kind == "private" and not is_peer) or response.get("group_processing", {}).get("runtime_results"):
            row["trace_id"] = response["trace_id"]
        if row["runtime_status"] in {"needs_help", "needs_human", "failed"}:
            traces = self.channel.runtime.trace_recorder.get_trace(response["trace_id"])
            failure = _model_failure([item.content for item in traces if item.step == "llm_error"])
            if failure or row["runtime_status"] != "needs_human":
                row.update(failure or {"error_type": "runtime_needs_help"}, failure_stage="agent")
            else:
                row["human_review_requested"] = True
                handling = assess_pending_handling(visible, {})
                if handling["status"] != "review_required":
                    error_type = ("runtime_human_review_fallback" if handling["status"] == "failed"
                                  else "runtime_human_review_not_delivered")
                    row.update(error_type=error_type, failure_stage="agent")
        if is_peer:
            receipt = self.channel.completed_peer_message(message_id)
            if receipt is None or response["status"] != "delivered":
                return dict(row, error_type="peer_receipt_not_completed", failure_stage="delivery"), []
            row["peer_delivery"] = True
            return row, [receipt]
        broadcasts = [{"kind": "group", "sender_id": event.actor_id, "recipient_id": recipient,
                       "message_id": message_id}] if kind == "group" else []
        return row, [*outputs, *broadcasts]

    def _record_failed_shop_dispatch(self, message: MessageEnvelope, failure: dict) -> tuple[dict, list[dict]]:
        """店家处理失败不撤回已入档原话。只用精确来话以及同源真实输出作为收件证据。"""
        try:
            records = self.channel.events()
        except (OSError, SQLiteError) as exc:
            return dict(failure, receipt_lookup_error=type(exc).__name__), []
        receipt = next((row for row in records if row["message_id"] == message.message_id), None)
        conversation_id = (self.channel.group_conversation_id if message.kind == "group"
                           else self.channel.private_conversation_id(message.sender_id))
        expected = {
            "room_id": self.channel.room_id, "conversation_id": conversation_id, "direction": "inbound",
            "sender_id": message.sender_id, "recipient_id": message.recipient_id, "kind": message.kind,
            "text": message.text, "sent_at": message.sent_at.astimezone(timezone.utc).isoformat(),
            "quoted_message_id": message.quoted_message_id, "source_message_id": None,
        }
        # 渠道在开始处理店家回复前，已在一个事务中保存原话和它的处理收据。
        if not receipt or not receipt.get("trace_id") or any(receipt.get(key) != val for key, val in expected.items()):
            return failure, []
        outputs = [row for row in records if row["direction"] == "outbound"
                   and row["source_message_id"] == message.message_id and row["sequence"] > receipt["sequence"]]
        visible = [row for row in outputs if row["kind"] == "group" or row["recipient_id"] == message.sender_id]
        failure.update(channel_trace_id=receipt["trace_id"], trace_id=receipt["trace_id"],
                       failure_stage="agent", runtime_status="failed", outputs=visible,
                       reply_message_id=visible[-1]["message_id"] if visible else None)
        return failure, [*outputs, *([receipt] if message.kind == "group" else [])]

    def _attention_rank(self, source: str, actor_id: str) -> bytes:
        """固定种子的顺序不随进程重开或其他成员的决定变化。"""
        return sha256(f"{self.seed}|{source}|{actor_id}".encode()).digest()

    def _group_audience(self, message_id: str, sender: str, room_id: str) -> list[str]:
        """房间的活跃成员独立决定是否回应，作者和其他房间不触发回合。"""
        if room_id != self.channel.room_id:
            return []
        return sorted((self.actors.keys() & self.group_members) - {sender},
                      key=lambda actor_id: self._attention_rank(message_id, actor_id))

    def _wake_from_deliveries(self, parent: RoleEvent, messages: list[dict]) -> list[str]:
        """为真实收件人安排独立回合，仍使用每人的每日预算和转发深度。"""
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
                kind = "peer_receipt" if item.get("direction") == "peer" else "private_receipt"
            else:
                audience = self._group_audience(message_id, item["sender_id"], item.get("recipient_id", ""))
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
        failures = [row for row in rows if row.get("error_type")]
        decided = Counter((row["decision"].get("action") or {}).get("action", "invalid") for row in rows)
        sent = [row for row in rows if row.get("channel_trace_id")
                and chat.get(row.get("message_id"), {}).get("trace_id") == row["channel_trace_id"]]
        venue_sends = [row for row in sent if chat[row["message_id"]].get("direction") != "peer"]
        cross_channel: Counter = Counter()
        for row in venue_sends:
            source = chat.get(row.get("caused_by"))
            if not source or not row["visible_cause_verified"] or source["sender_id"] == row["actor_id"]:
                continue
            channel = chat[row["message_id"]]["kind"]
            if (row["wake_kind"], source["kind"], channel) == ("group_attention", "group", "private"):
                cross_channel["group_to_private"] += 1
            if ((row["wake_kind"], source["kind"], channel) == ("private_receipt", "private", "group")
                    and source["recipient_id"] == row["actor_id"] and source.get("direction") != "peer"):
                cross_channel["private_to_group"] += 1
        samples, sampling = self._review(by_actor, records, {row["message_id"] for row in sent})
        human_review_turns = [row for row in rows if row.get("human_review_requested")]
        human_review = []
        if human_review_turns:
            replies = _delivered_replies(chat, {key: index for index, key in enumerate(chat)})
            verdicts = shop_reply_verdicts(samples)
            human_review = [{"event_id": row["event_id"], "message_id": row["message_id"],
                             "trace_id": row.get("trace_id"),
                             **assess_pending_handling(replies.get(row["message_id"], ()), verdicts)}
                            for row in human_review_turns]
        business = self._business_state(rows)
        kinds = Counter(item.kind for item in events if item.status == "finished")
        days = Counter(item.actor_id for item in events if item.kind == "timer" and item.status == "finished")
        actors = sum(days[actor_id] == 2 for actor_id in self.actors)
        live = [row for row in rows if _live_actor_action(row)]
        availability_extractions = [
            item for row in rows for item in row["decision"].get("availability_reviews", ())
            if item.get("stage") == "availability_extraction"
        ]
        live_days = defaultdict(set)
        for row in live:
            actor_id = row["actor_id"]
            live_days[actor_id].add(self._day_key(actor_id, datetime.fromisoformat(row["virtual_at"]))[1])
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
                    if (source["kind"] == "private" and source["recipient_id"] == actor_id
                            and source.get("direction") != "peer"):
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
                and chat[row["message_id"]]["direction"] == ("peer" if row.get("peer_delivery") else "inbound")
                and chat[row["message_id"]]["quoted_message_id"] == row["decision"]["action"].get("quote_message_id")
                and chat[row["message_id"]]["kind"] == row["channel"] for row in sent),
            "group_attention_wakes_only_other_room_members": all(
                item.actor_id in self.group_members and item.caused_by in chat
                and chat[item.caused_by]["kind"] == "group" and chat[item.caused_by]["recipient_id"] == self.channel.room_id
                and chat[item.caused_by]["sender_id"] != item.actor_id
                for item in events if item.kind == "group_attention"),
            "private_receipts_wake_only_their_recipient": all(
                chat[item.caused_by]["kind"] == "private" and chat[item.caused_by]["recipient_id"] == item.actor_id
                and chat[item.caused_by].get("direction") != "peer"
                for item in events if item.kind == "private_receipt" and item.caused_by in chat)
                and all(item.caused_by in chat for item in events if item.kind == "private_receipt"),
            "no_overfilled_table_in_store_or_tool_evidence": not (
                business["seats"]["overfilled_tables"] or business["tool_game_snapshots"]["overfilled"]),
        }
        if isinstance(self.actor, ModelActor):
            checks["customer_decisions_from_model"] = len(live) == len(rows)
        if human_review:
            checks["human_review_requests_have_substantive_delivery"] = all(row["substantive_delivery"] for row in human_review)
        peers = {}
        if self.peer_dialogues:
            deliveries = [chat[row["message_id"]] for row in sent if row.get("peer_delivery")]
            peer_wakes = [event for event in events if event.kind == "peer_receipt"]
            by_message = {message["message_id"]: message for message in deliveries}
            replies = 0
            for event in peer_wakes:
                source = by_message.get(event.caused_by)
                result = event.result or {}
                reply = by_message.get(result.get("message_id"))
                if (source is not None and reply is not None and reply["recipient_id"] == source["sender_id"]
                        and reply["conversation_id"] == source["conversation_id"]):
                    replies += 1
            checks["peer_receipts_wake_only_their_recipient"] = all(
                event.caused_by in by_message and by_message[event.caused_by]["recipient_id"] == event.actor_id
                for event in peer_wakes)
            peers["rolling_peers"] = {
                "status": "observed" if deliveries else "not_observed",
                "delivered_messages": len(deliveries), "receipt_wakes": len(peer_wakes), "causal_replies": replies,
                "private_conversations": len({item["conversation_id"] for item in deliveries}),
            }
        valid_samples = (bool(samples) and all(sample["status"] == "valid" for sample in samples)
                         and all(row["status"] != "review_required" for row in human_review))
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
                "real_decision_actors_with_two_calendar_days": sum(len(day_set) >= 2 for day_set in live_days.values()),
                "wake_kinds": dict(kinds), "actions": dict(decided),
                "sent_channels": dict(Counter(row["channel"] for row in venue_sends)),
                "cross_channel_replies": {"group_to_private": cross_channel["group_to_private"],
                                          "private_to_group": cross_channel["private_to_group"]},
                "causally_woken_actors": len({item.actor_id for item in events if item.kind in _RECEIPT_WAKES}),
                "actors_with_own_prior_day_messages_in_context": len(own_prior_day),
                "actors_with_prior_day_private_receipts_in_context": len(prior_day_private_receipt),
                "out_of_scope_context_rows": mismatched_visibility,
                "reused_channel_receipts": sum(row.get("delivery_deduplicated", False) for row in sent),
                "availability_extractions": {
                    "attempted": len(availability_extractions),
                    "with_complete_provider_response": sum(
                        (item.get("response_source") or {}).get("http_response_complete") is True
                        for item in availability_extractions),
                    "rejected_conflicts": sum(bool(item.get("conflict")) for item in availability_extractions),
                },
                "semantic_review_customers": len(samples), "semantic_sampling": sampling, "business_state": business},
            "semantic_samples": samples, "independent_judge": independent, **peers,
            **({"human_review_handling": human_review} if human_review else {}),
        }

    def _business_state(self, rows: list[dict]) -> dict:
        """只读持久化局况，另按实际非重复工具结果核对曾返回过的完整局况。"""
        store = self.channel.runtime.store
        games = tuple(store.games.values())
        seats = [game.seat_summary() for game in games]
        tools = [tool for row in rows for tool in row.get("tool_results", ())
                 if tool.get("called") is True and tool.get("allowed") is True
                 and not tool.get("error") and not tool.get("deduplicated")]
        observed_seats = [summary for tool in tools for summary in _tool_seat_summaries(tool.get("result") or {})]
        waiting = tuple(store.waiting_demands.values())
        reservations = tuple(store.room_reservations.values())
        return {
            "games_by_status": dict(sorted(Counter(game.status.value for game in games).items())),
            "seats": {
                "tables": len(seats), "capacity": sum(row["seats_total"] for row in seats),
                "claimed": sum(row["claimed_seats"] for row in seats),
                "vacant": sum(row["remaining_seats"] for row in seats),
                "full_tables": sum(row["claimed_seats"] == row["seats_total"] for row in seats),
                "overfilled_tables": sum(row["claimed_seats"] > row["seats_total"] for row in seats),
                "overflow_seats": sum(max(0, row["claimed_seats"] - row["seats_total"]) for row in seats),
            },
            "waiting_demand_count": len(waiting),
            "waiting_demands_by_status": dict(sorted(Counter(item.status.value for item in waiting).items())),
            "room_reservation_count": len(reservations),
            "reservations_by_status": dict(sorted(Counter(item.status for item in reservations).items())),
            "reservations_with_game": sum(item.game_id is not None for item in reservations),
            "successful_tool_executions": len(tools),
            "successful_tools_by_name": dict(sorted(Counter(tool["name"] for tool in tools).items())),
            "tool_game_snapshots": {"checked": len(observed_seats),
                "overfilled": sum(row["claimed_seats"] > row["seats_total"] for row in observed_seats)},
        }

    def _review(self, by_actor: dict, records: list[dict], delivered: set[str]) -> tuple[list[dict], dict]:
        """分层选择实际角色并报告未覆盖人数，汇总不携带角色编号或原话。"""
        strata = {actor_id: _actor_strata(rows, delivered, self.actors[actor_id].temperament.purpose)
                  for actor_id, rows in by_actor.items()}
        observed = Counter(tag for tags in strata.values() for tag in tags)
        selected: list[str] = []
        covered: Counter = Counter()
        if self.judge is not None:
            remaining = set(strata)
            while remaining and len(selected) < self.judge_users:
                def rank(actor_id):
                    """优先覆盖新分层，再兼顾较稀有或较少抽到的分层。"""
                    unrepresented = [tag for tag in strata[actor_id] if not covered[tag]]
                    return (-len(unrepresented), -sum(1_000_000 // observed[tag] for tag in unrepresented),
                            -sum(1_000_000 // (covered[tag] + 1) for tag in strata[actor_id]),
                            self._attention_rank("semantic-strata-v1", actor_id), actor_id)

                actor_id = min(remaining, key=rank)
                remaining.remove(actor_id)
                selected.append(actor_id)
                covered.update(strata[actor_id])
        samples = [review_dialogue(self.judge, customer_id=actor_id, turns=by_actor[actor_id], records=records,
                    role_only=True, delivered_history=True, additional_instructions=ROLLING_REVIEW_INSTRUCTIONS,
                    rolling_check_ids=True)
                   for actor_id in selected]
        valid = [actor_id for actor_id, sample in zip(selected, samples, strict=True) if sample["status"] == "valid"]
        valid_coverage = Counter(tag for actor_id in valid for tag in strata[actor_id])
        coverage: dict = {axis: {} for axis in ("event", "channel", "action", "role_intent")}
        for axis, value in sorted(observed):
            tag = axis, value
            coverage.setdefault(axis, {})[value] = {"observed_actors": observed[tag], "sampled_actors": covered[tag],
                                                    "valid_judgment_actors": valid_coverage[tag]}
        return samples, {
            "method": "deterministic_strata_v1", "judge_enabled": self.judge is not None,
            "eligible_actors": len(strata), "sample_budget": self.judge_users, "sampled_actors": len(samples),
            "missing_sample_actors": len(strata) - len(samples), "valid_judgment_actors": len(valid),
            "actors_without_valid_judgment": len(strata) - len(valid),
            "outcomes": dict(sorted(Counter(sample["verdict"] if sample["status"] == "valid" else "unavailable"
                                            for sample in samples).items())),
            "strata": coverage,
        }


def _tool_seat_summaries(result: dict) -> list[dict]:
    """识别正式写工具与查局工具提供的席位汇总，不从客户人数或自由文本推断。"""
    games = [result.get("game")]
    games.extend(item.get("game") for bucket in ("matches", "alternatives", "nearby_options")
                 for item in (result.get(bucket) or ()) if isinstance(item, dict))
    return [summary for game in games if isinstance(game, dict)
            and isinstance((summary := game.get("seat_summary")), dict)
            and all(type(summary.get(key)) is int for key in ("seats_total", "claimed_seats"))]


def _actor_strata(rows: list[dict], delivered: set[str], purpose: str) -> set[tuple[str, str]]:
    """按实际日志给同一角色打有限类别标签，意图标签来自结构化角色设定。"""
    labels = {("role_intent", purpose if purpose in _ROLE_PURPOSES else "other")}
    for row in rows:
        event = row.get("wake_kind")
        labels.add(("event", event if event in _WORLD_WAKES | _RECEIPT_WAKES | {"timer", _OVERTIME_WAKE} else "other"))
        channel = "peer" if row.get("peer_delivery") else row.get("channel")
        labels.add(("channel", channel if row.get("message_id") in delivered and channel in {"group", "private", "peer"} else "none"))
        decision = row["decision"]
        action = (decision.get("action") or {}).get("action")
        labels.add(("action", action if action in {"send", "wait", "leave", "arrive"} else "invalid"))
        if row.get("human_review_requested") and not row.get("error_type"):
            labels.add(("shop_handling", "human_review_requested"))
        for context_event in decision.get("context", {}).get("events", ()):
            goal = (context_event.get("details") or {}).get("case_goal")
            if goal in _ROLE_GOALS:
                labels.add(("role_intent", goal))
    return labels
