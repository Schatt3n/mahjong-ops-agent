"""以登记好的门店事实和只给本人的突发事件验证真实营业沟通。"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from typing import Any

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.knowledge import default_terminology_repository
from mahjong_agent_runtime.models import GameStatus, using_clock

from .venue_actors import ActorAction, ActorContext, ActorDecision, ModelActor
from .venue_channel import AGENT_ID, OWNER_ID, MessageEnvelope
from .venue_dialogues import _live_actor_action, _model_failure
from .venue_fixture import VenueFixtureAgent, _call, _reply
from .venue_judging import review_dialogue
from .venue_world import VenueWorld


DISTURBANCE_CASES = ("rain", "traffic", "long_commute", "out_of_town", "repeat_visit")
_CASE_RULES = {
    "rain": "本人受暴雨影响已无法赴局。有取消来话时，店家应按本人意愿释放该席，不能改动他人的座位或把整桌取消。",
    "traffic": "本人出发后堵车。如果只是报到达延后或询问能否等，原局排期和座位照旧。回复可确认本人座位仍在，但未有牌友同意的证据就不能说某个迟到时刻到店没问题。只问顾客是否需要转达可通过，声称接手去问须有对应处理证据。",
    "long_commute": "本人距店较远，预计来不及原定开始。顾客问能否保留到晚些时候，应交代现有席位，说明未确认大家能否等。询问顾客是否需要转达可通过，答应自己会去问须有对应处理证据。无须替其他玩家同意或立即改整桌。",
    "out_of_town": "主动邀约实际已送达，本人不在市内且当天回不来。明确拒绝应有本人邀约的已执行记录，不要重复邀约或把本次不在当成长期偏好。如果顾客另请下次通知，要有新的有效安排才能对未来通知作承诺。",
    "repeat_visit": "顾客上午已玩完，下午有空才开启新询局目标。应按新的时间和本人意思响应，不能复用上午的牌局作为下午可报名的牌局。",
}
_HANDLED = {"completed", "waiting_user", "waiting_tool"}
_CONFIRMED = {"confirmed", "joined", "accepted", "arrived"}


def _declined_invite_in_current_turn(turn: dict[str, Any], game_id: str, draft_id: str) -> bool:
    """只接受本轮通过正式网关写入同一份已送达邀约的状态迁移。"""
    return any(
        tool.get("name") == "record_candidate_reply" and tool.get("called") and tool.get("allowed")
        and not tool.get("error") and not tool.get("deduplicated")
        and (tool.get("result") or {}).get("recorded_status") == "declined"
        and ((tool.get("result") or {}).get("game") or {}).get("game_id") == game_id
        and any(change.get("entity_type") == "invite_draft" and change.get("entity_id") == draft_id
                and change.get("from_status") == "sent" and change.get("to_status") == "declined"
                and change.get("trace_id") == turn.get("trace_id")
                for change in tool.get("state_transitions", ()))
        for tool in turn.get("tool_results", ())
    )


class VenueDisturbances:
    """每例先设定独立的牌局与顾客，再经实际私聊工具和数据库验收。"""

    def __init__(self, *, people, actor, judge, channel, store, run_dir, start_at, seed,
                 background_events, fixture_agent=None, selected_case=None) -> None:
        self.world = VenueWorld(seed, start_at)
        self.people = tuple(people)
        self.actor, self.judge, self.channel, self.store = actor, judge, channel, store
        self.fixture_agent, self.run_dir, self.start_at = fixture_agent, run_dir, start_at
        self.cases = (selected_case,) if selected_case else DISTURBANCE_CASES
        if background_events or any(case not in DISTURBANCE_CASES for case in self.cases):
            raise ValueError("突发事件场景只运行显式选定的私聊案例")
        self.errors: list[dict[str, Any]] = []

    def run(self) -> dict[str, Any]:
        """归档模型决定、交付、实际工具和重开 SQLite 后的结果。"""
        customers = self._choose_customers()
        used = {person.sender_id for person in customers.values()}
        fillers = iter(person for person in self.people if person.sender_id not in used)
        episodes = [
            self._episode(case, person, tuple(next(fillers) for _ in range(3)))
            for case in ("repeat_visit", "rain", "out_of_town", "traffic", "long_commute")
            if (person := customers.get(case)) is not None
        ]
        records = self.channel.events()
        events = [turn for episode in episodes for turn in episode["turns"]]
        for name, rows in (("events", events), ("chat", records)):
            (self.run_dir / (name + ".jsonl")).write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
            )
        checks = {f"{episode['case']}.{name}": passed
                  for episode in episodes for name, passed in episode["checks"].items()}
        model_actions = sum(_live_actor_action(turn) for turn in events)
        if isinstance(self.actor, ModelActor):
            checks["customer_decisions_from_model"] = bool(events) and model_actions == len(events)
        trace = self.run_dir / "agent_trace.log"
        checks["private_schedule_not_in_agent_trace"] = not trace.is_file() or all(
            not any(marker in line for marker in ('"event_condition"', '"free_from"', '"free_until"', '"work_status"'))
            for line in trace.open(encoding="utf-8")
        )
        samples = self._judge_episodes(episodes, records)
        independent = None
        if self.judge is not None:
            valid = len(samples) == len(episodes) and all(item["status"] == "valid" for item in samples)
            verdict = ("fail" if any(item["verdict"] == "fail" for item in samples)
                       else "pass" if valid and all(item["verdict"] == "pass" for item in samples)
                       else "inconclusive")
            independent = {"status": "valid" if valid else "partial", "verdict": verdict}
        return {
            "checks": checks, "people": len(self.people), "targeted_users": len(episodes), "workers": 1,
            "max_turns": 1, "chat_events": len(records), "real_customer_actions": model_actions,
            "model_transport_errors": self.errors,
            "failed_events": [turn["event_id"] for turn in events if turn.get("error_type")],
            "semantic_samples": samples, "independent_judge": independent,
            "disturbances": [{key: value for key, value in episode.items() if key != "turns"} for episode in episodes],
        }

    def _choose_customers(self) -> dict:
        """按真实空闲和当地出行条件选择不同顾客，尽量让三种玩法都参与。"""
        chosen = {}
        used_games: set[str] = set()
        morning = self.start_at.replace(hour=10, minute=30)
        afternoon = self.start_at.replace(hour=15)
        for case in sorted(self.cases, key=lambda item: item != "repeat_visit"):
            candidates = []
            for person in self.people:
                if person in chosen.values():
                    continue
                day = self.world.availability(person)
                if case == "repeat_visit":
                    qualifies = day.available_for(morning, 1.5) and day.available_for(afternoon, 2)
                elif case == "out_of_town":
                    qualifies = day.location == "outside_city"
                else:
                    qualifies = day.available_for(self.start_at, 2)
                    if case == "long_commute":
                        qualifies = qualifies and day.location == "across_town"
                if qualifies:
                    candidates.append(person)
            if not candidates:
                raise ValueError(f"种子没有满足突发场景前提的顾客：{case}")
            chosen[case] = min(candidates, key=lambda person: person.preferred_game in used_games)
            used_games.add(chosen[case].preferred_game)
        return chosen

    def _create_game(self, start: datetime, person, others, *, include_customer: bool,
                     duration: float = 2) -> str:
        """通过 Store 布置初始报名，后续变更必须走顾客入站。"""
        term = default_terminology_repository().first_match(person.preferred_game, categories={"game_type"})
        players = (*others, person) if include_customer else others
        with using_clock(lambda: start - timedelta(hours=1)):
            game, _ = self.store.create_game(
                conversation_id=self.channel.private_conversation_id(others[0].sender_id),
                organizer_id=others[0].sender_id, organizer_name=others[0].name,
                requirement={"game_type": term.term.canonical["requested_game"], "start_time_kind": "scheduled",
                             "planned_start_at": start.isoformat(), "start_time": start.strftime("%H:%M"),
                             "duration_kind": "fixed", "duration_hours": duration, "stake": str(person.stake_min),
                             "smoke_preference": "no_smoking"},
                known_players=[], trace_id=f"disturbance-setup:{person.sender_id}:{start.isoformat()}",
            )
            for player in players[1:]:
                self.store.join_game(game_id=game.game_id, customer_id=player.sender_id,
                                     display_name=player.name, seat_count=1,
                                     trace_id=f"disturbance-confirm:{start.isoformat()}:{player.sender_id}")
        return game.game_id

    def _episode(self, case: str, person, others) -> dict[str, Any]:
        """用当前营业事实发出店员起始消息，再交给角色处理临时变动。"""
        start = self.start_at.replace(hour=15) if case == "repeat_visit" else self.start_at
        at = start - timedelta(minutes=2 if case in {"traffic", "long_commute"} else 25)
        already_joined = case in {"rain", "traffic", "long_commute"}
        game_id = self._create_game(start, person, others, include_customer=already_joined)
        person_id = person.sender_id
        old_game = None
        draft_id = None
        invitation_id = None
        if case == "repeat_visit":
            event = self.world.role_event(person, case, at, scheduled_start=start)
            ended = datetime.fromisoformat(event["details"]["previous_session_end"])
            started = ended - timedelta(hours=1)
            old_id = self._create_game(started, person, others, include_customer=True, duration=1)
            room = self.store.search_room_availability(start_at=started, end_at=ended)["available_room_ids"][0]
            roster = [party.sender_id for party in (*others, person)]
            self.store.start_game_play(old_id, attendee_ids=roster, room_id=room, at=started,
                                       trace_id=f"disturbance-actual-start:{case}")
            old_game, _ = self.store.finish_game_play(old_id, at=ended, trace_id=f"disturbance-actual-end:{case}")
            # 让顾客从自己的上午收据知道实际档位，再决定下午怎么开口；下午库存始终对顾客不可见。
            self.channel.publish_owner(
                kind="private", recipient_id=person_id, sent_at=ended + timedelta(seconds=1),
                message_id="shop-receipt:morning-repeat-visit",
                text=f"上午这桌{person.preferred_game}散了哈，打的是{person.stake_min:g}底，你刚那一场已结。",
            )
        else:
            invitation_id = f"shop-setup:{case}"
            hour = start.strftime("%H:%M")
            if already_joined:
                text = f"你今天 {hour} 那桌{person.preferred_game}的位置记好了，到时候见。"
            else:
                text = f"今晚 {hour} {person.preferred_game}缺一，{person.stake_min}底，无烟，打两个钟头，你有空来吗？"
                with using_clock(lambda: at):
                    drafts, _ = self.store.create_invite_drafts(
                        game_id=game_id, invitations=[{"customer_id": person_id, "display_name": person.name,
                                                       "message_text": text}], trace_id=invitation_id,
                    )
                    draft_id = drafts[0].draft_id
            delivery = self.channel.publish_owner(kind="private", recipient_id=person_id, text=text,
                                                  sent_at=at, message_id=invitation_id)
            if draft_id and delivery["status"] == "delivered":
                with using_clock(lambda: at):
                    self.store.update_invite_delivery_status(draft_id=draft_id, status="sent",
                        trace_id=invitation_id, reason="沙盒店员私聊已交付")
            event = self.world.role_event(person, case, at + timedelta(seconds=30), scheduled_start=start,
                                          caused_by=invitation_id)

        before = self.store.require_game(game_id)
        game_ids = set(self.store.games)
        turn = self._turn(person, case, event, game_id)
        with using_clock(lambda: at + timedelta(seconds=45)):
            reopened = SQLiteAgentStore(self.store.path)
            final = reopened.require_game(game_id)
            statuses = {member.customer_id: member.status for member in final.participants}
            had_seat = any(member.customer_id == person_id and member.status in _CONFIRMED
                           for member in before.participants)
            turn["persisted_game_state"] = {
                "game_id": game_id,
                "customer_seat_before": had_seat,
                "customer_seat_after": statuses.get(person_id) in _CONFIRMED,
                "game_start_before": before.planned_start_at.isoformat(),
                "game_start_after": final.planned_start_at.isoformat() if final.planned_start_at is not None else None,
            }
            checks = {
                "customer_turn_delivered": bool(turn.get("message_id")),
                "shop_reply_delivered": bool(turn.get("reply_message_id")) and turn.get("runtime_status") in _HANDLED,
                "other_confirmed_seats_intact": all(statuses.get(member.sender_id) in _CONFIRMED for member in others),
                "original_start_time_intact": final.planned_start_at == before.planned_start_at,
                "original_end_time_intact": final.planned_end_at == before.planned_end_at,
                "no_unrequested_game": set(reopened.games) == game_ids,
            }
            if case == "rain":
                checks["customer_seat_released"] = statuses.get(person_id) == "declined" and final.remaining_seats() == 1
            elif already_joined:
                checks["customer_seat_retained"] = statuses.get(person_id) in _CONFIRMED and final.remaining_seats() == 0
            elif case == "out_of_town":
                checks["customer_did_not_join"] = statuses.get(person_id) not in _CONFIRMED and final.remaining_seats() == 1
                checks["invite_decline_recorded"] = reopened.invite_drafts[draft_id].status.value == "declined"
                checks["invite_decline_tool_evidence"] = _declined_invite_in_current_turn(turn, game_id, draft_id)
            else:
                historical = reopened.require_game(old_game.game_id)
                checks["morning_play_stayed_finished"] = (historical.to_dict() == old_game.to_dict()
                    and historical.status == GameStatus.FINISHED and person_id in historical.actual_attendee_ids)
                checks["afternoon_table_looked_up"] = any(
                    (match.get("game") or {}).get("game_id") == game_id
                    for tool in turn.get("tool_results", ()) if tool["name"] == "search_current_games"
                    and tool.get("called") and tool.get("allowed") and not tool.get("error")
                    for match in (tool.get("result") or {}).get("matches", ())
                )
        checks["event_observed_by_customer"] = event["event_id"] in {
            item.get("event_id") for item in turn.get("decision", {}).get("context", {}).get("events", ())
        }
        return {"case": case, "customer_id": person_id, "game_type": person.preferred_game, "game_id": game_id,
                "morning_game_id": old_game.game_id if old_game else None, "invitation_message_id": invitation_id,
                "checks": checks, "turns": [turn]}

    def _turn(self, person, case, event, game_id: str) -> dict[str, Any]:
        """角色可自己决定是否发言，只有实际消息走真实 Agent 入口。"""
        at = datetime.fromisoformat(event["at"])
        event["details"].update(case_goal="lookup" if case == "repeat_visit" else "wait_or_leave",
                                game_type=person.preferred_game, knows_shopkeeper=True)
        if case == "rain":
            event["details"]["case_intent"] = "暴雨让回店的路暂时封了，今天剩下的时间也赶不到店，原本已经报过名。"
        if case == "repeat_visit":
            event["details"]["earliest_start_time"] = "15:00"
            event["details"]["latest_start_time"] = "16:00"
            event["details"]["case_intent"] = (
                "上午打完，下午又有空，想看看三四点开有没有现成局，仍想打上午那种档位。"
                "自己的上午收据能核对玩法和底注。开口方式、说几项条件由你决定，不要求逐项交代。"
            )
        expression = {
            "event_id": event["event_id"], "actor_id": person.sender_id, "virtual_at": at.isoformat(),
            "caused_by": event.get("caused_by"), "kind": case,
        }
        # 实际账本保留人工店员来源；角色视角用同一已鉴权的店家私聊账号接话。
        history = [{**message, "sender_id": AGENT_ID} if message["sender_id"] == OWNER_ID else message
                   for message in self.channel.events_for(person.sender_id)]
        if event.get("caused_by") and not any(message["message_id"] == event["caused_by"] for message in history):
            raise ValueError("顾客未收到这条私聊邀请或报名确认")
        context = ActorContext(person, at.isoformat(), history=history, events=[event])
        with using_clock(lambda: at):
            decision = self.actor.decide(context, trace_id=f"role:{event['event_id']}")
            expression["decision"] = decision.to_dict()
            if decision.action is None:
                expression["error_type"] = decision.status
                if decision.status == "transport_error":
                    self.errors.append({"event_id": event["event_id"], "model_stage": "role", "http_statuses": [
                        attempt["http_status"] for attempt in decision.diagnostics.get("attempts", ())
                        if attempt.get("http_status") is not None
                    ]})
                return expression
            if decision.action.action != "send":
                return expression
            action = decision.action
            message_id = f"customer:{event['event_id']}"
            if self.fixture_agent is not None:
                self.fixture_agent.turns[message_id] = (case, {"game_id": game_id})
            try:
                received = self.channel.send(MessageEnvelope(message_id, person.sender_id, action.text, at,
                                                              "private", AGENT_ID))
            except (KeyError, OSError, RuntimeError, ValueError) as exc:
                expression.update(error_type=type(exc).__name__, error=str(exc))
                return expression
            outputs = [row for row in received["archive_outputs"]
                       if row["recipient_id"] == person.sender_id and row["kind"] == "private"]
            trace_id = received.get("trace_id")
            expression.update(message_id=message_id, trace_id=trace_id, outputs=outputs,
                              runtime_status=received["status"], tool_results=received.get("tool_results", []),
                              reply_message_id=outputs[-1]["message_id"] if outputs else None)
            if received["status"] not in _HANDLED and trace_id:
                llm_errors = [row.content for row in self.channel.runtime.trace_recorder.get_trace(trace_id)
                              if row.step == "llm_error"]
                failure = _model_failure(llm_errors)
                if failure:
                    expression.update(failure)
                    self.errors.append({"event_id": event["event_id"], "model_stage": "agent",
                                        "http_statuses": failure["http_statuses"]})
        return expression

    def _judge_episodes(self, episodes, records) -> list[dict[str, Any]]:
        """独立判话术是否承接事件；仍以硬断言和原始工具状态核对成功。"""
        if self.judge is None:
            return []
        return [
            review_dialogue(self.judge, customer_id=episode["customer_id"], turns=episode["turns"],
                records=records, role_only=not episode["checks"]["shop_reply_delivered"],
                additional_instructions=_CASE_RULES[episode["case"]])
            for episode in episodes if episode["turns"][0]["decision"]["status"] == "valid"
        ]


class DisturbanceFixtureActor:
    """只用于离线检查布置、消息路由和持久状态，不能证明角色生成质量。"""

    def decide(self, context, *, trace_id: str) -> ActorDecision:
        """用离线案例输入复验持久化，不参与任何真实模型运行。"""
        data = context.to_dict()
        case = data["events"][-1]["kind"]
        details = data["events"][-1]["details"]
        text = {
            "rain": "外面暴雨我过不去了，今晚那桌把我那位让了吧",
            "traffic": "路上堵死了，可能要晚十来分钟",
            "long_commute": "我离你那有点远，能等半个钟头吗？",
            "out_of_town": "我还在外地，今天不回，打不了",
            "repeat_visit": f"下午三四点有{details.get('game_type')}吗，还想来打会儿",
        }[case]
        action = ActorAction("send", text, "private", AGENT_ID)
        return ActorDecision(trace_id, "mock", "valid", action, None, data, {})


class DisturbanceFixtureAgent(VenueFixtureAgent):
    """离线测试使用真实网关操作；只有模型响应在本类替换。"""

    def complete(self, messages, *, trace_id: str, timeout_seconds: float) -> str:
        """按预先登记的离线测试消息返回候选动作，正式代码负责执行与落库。"""
        context = self._context_payload(messages)
        message = context.get("current_message") or {}
        spec = self.turns.get(message.get("message_id"))
        if spec is None:
            return super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)
        case, details = spec
        if not context.get("previous_tool_results"):
            if case in {"rain", "out_of_town"}:
                return _call("record_candidate_reply", {"game_id": details["game_id"],
                    "customer_id": message["sender_id"], "display_name": message["sender_name"], "status": "declined"})
            if case == "repeat_visit":
                return _call("search_current_games", {"requirement": {}, "limit": 8})
        return _reply({"rain": "好，那就先把你这位放出来。", "out_of_town": "好，回来再约。",
            "traffic": "知道了，我先问下另外几个人。", "long_commute": "好，我问下他们能不能等。",
            "repeat_visit": "三点还有个现成的，想打吗？"}[case], status="waiting_user" if case == "repeat_visit" else "completed")
