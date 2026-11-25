"""在独立门店核对多轮私聊中的成局、无匹配及同库存下的并行结果。"""

from __future__ import annotations

from datetime import datetime, timedelta
from itertools import zip_longest
import json
import random
from typing import Any

from mahjong_agent_runtime.domains.game_domain import current_game_query_start_range
from mahjong_agent_runtime.models import using_clock

from .venue_actors import ModelActor, Persona
from .venue_channel import AGENT_ID
from .venue_dialogues import Dialogue, VenueDialogues
from .venue_write_guard import VenueGuardedToolGateway


_FORMATION_DURATION_HOURS = 3
_LOOKUP_WINDOW_LENGTH = timedelta(hours=2)
_INCOMPATIBLE_GAME_TYPES = {"四川麻将": "sichuan_mahjong", "国标麻将": "national_standard_mahjong"}


class VenueCohort(VenueDialogues):
    """场景只设定初始库存与当事人意向，查桌、答复和确认均由真实交互产生。"""

    def __init__(self, *, people, case: str, seed: int, active_users: int, **kwargs) -> None:
        """挑选画像相容的生成角色，保持每个人的原始性格和关系。"""
        people = list(people)
        if case not in {"formation", "no-match", "mixed"} or active_users < 1 or active_users > len(people):
            raise ValueError("小批验收要求有效场景和至少一位候选顾客")
        if case == "formation" and active_users != 1:
            raise ValueError("单局最后一席验收只接受一位活跃顾客")
        if case == "mixed" and active_users < 2:
            raise ValueError("混合验收至少需要一位可匹配和一位玩法不同的顾客")
        super().__init__(people=people, seed=seed, active_users=active_users, **kwargs)
        self.case = case
        ranked = list(people)
        random.Random(seed).shuffle(ranked)
        if case != "no-match":
            seeker = next((person for person in ranked if _formation_seeker(person)), None)
            if seeker is None:
                raise ValueError("当前种子没有适合查找这桌的画像")
            # 小批也让不同玩法都出现，组内保留固定种子的顺序和营业时段优先级。
            by_game = [sorted((person for person in ranked if person.preferred_game == game), key=_lookup_priority)
                       for game in _INCOMPATIBLE_GAME_TYPES]
            incompatible = [person for batch in zip_longest(*by_game) for person in batch if person is not None]
            if len(incompatible) < active_users - 1:
                raise ValueError("当前种子无法为混合验收提供足够多的不同玩法顾客")
            selected = [seeker]
            selected.extend(incompatible[:active_users - 1])
            blocked = {person.sender_id for person in selected}
            blocked.update(edge.target_sender_id for edge in seeker.relationships if edge.kind == "avoid_same_table")
            self.early_players = [person for person in ranked if person.sender_id not in blocked][:3]
            if len(self.early_players) != 3:
                raise ValueError("当前种子不足以安排三位已报名玩家")
        else:
            ranked.sort(key=_lookup_priority)
            selected = ranked[:active_users]
            self.early_players = []
        self.sessions = [Dialogue(person, "try_join" if self.early_players and i == 0 else "lookup")
                         for i, person in enumerate(selected)]
        occupied = {person.sender_id for person in [*selected, *self.early_players]}
        self.background = [person for person in ranked if person.sender_id not in occupied][:len(self.background)]
        self.seeded_game_id: str | None = None

    def run(self) -> dict[str, Any]:
        """核对严格私聊因果链和审批后的真实席位，保留门店初态与终态。"""
        if self.store.games:
            raise ValueError("小批验收必须由一个没有历史牌局的独立门店开始")
        initial = self._seed_game() if self.early_players else {}
        report = super().run()
        expected = {session.person.sender_id: "formed" if session.goal == "try_join" else "no_match"
                    for session in self.sessions}
        findings = report["acceptance"]
        actual = {row["customer_id"]: row["kind"] for row in findings["outcomes"] if row["status"] == "passed"}
        # 固定合同桩专门压测多轮；真实查询允许问清一次后离开，报名仍需下方的本人续轮确认。
        if isinstance(self.actor, ModelActor):
            lookup_sessions = [session for session in self.sessions if session.goal == "lookup"]
            if lookup_sessions:
                report["checks"]["cohort_opening_search_covers_lookup_window"] = all(
                    _opening_query_covers_window(session.turns, self.start_at) for session in lookup_sessions
                )
        else:
            report["checks"]["cohort_each_customer_has_causal_multi_turn"] = (
                findings["multi_round_private_customers"] == len(self.sessions)
            )
        report["cohort"] = {"case": self.case, "expected_outcomes": expected, "initial_game": initial, "final_game": None}
        allowed_waitlists = {}
        if self.case == "mixed":
            waiting_demands = {key: item.to_dict() for key, item in self.store.waiting_demands.items()}
            candidates = {customer for customer, kind in expected.items()
                          if kind == "no_match" and actual.get(customer) == "waiting"}
            allowed_waitlists = self._authorized_waitlists(candidates, waiting_demands)
            report["cohort"]["authorized_transitions"] = {"no_match_to_waiting": len(allowed_waitlists)}
            report["checks"]["cohort_mixed_concurrent_deliveries"] = report["peak_inflight_deliveries"] >= 2
            report["checks"]["cohort_no_unasked_game_created"] = set(self.store.games) == {self.seeded_game_id}
            report["checks"]["cohort_waitlist_matches_authorized_transitions"] = (
                set(waiting_demands) == set(allowed_waitlists.values())
            )
            report["checks"]["cohort_incompatible_lookups_saw_open_inventory"] = all(
                self._looked_up_different_game(session) for session in self.sessions if session.goal == "lookup"
            )
        else:
            report["cohort"]["expected_outcome"] = next(iter(expected.values()))
        report["checks"]["cohort_expected_business_outcome"] = actual == (
            expected | dict.fromkeys(allowed_waitlists, "waiting")
        )
        if self.seeded_game_id:
            final = self.store.require_game(self.seeded_game_id)
            report["cohort"]["final_game"] = final.to_dict()
            actual_players = {member.customer_id for member in final.participants if member.status in {"joined", "confirmed"}}
            seeker = self.sessions[0].person.sender_id
            report["checks"]["cohort_seeded_table_filled_by_customer"] = (
                final.status.value == "ready" and not final.remaining_seats()
                and actual_players == {seeker, *(person.sender_id for person in self.early_players)}
                and self._confirmed_after_reply(self.sessions[0])
            )
        else:
            report["checks"]["cohort_no_unasked_game_created"] = not self.store.games
        return report

    def _authorized_waitlists(self, candidates: set[str], stored: dict[str, dict[str, Any]]) -> dict[str, str]:
        """本次混合场景仅采纳同一沙盒独立网关完成审计的候补变更。"""
        gateway = self.channel.runtime.tool_gateway
        if not candidates or not isinstance(gateway, VenueGuardedToolGateway):
            return {}
        with gateway.guard.log_path.open(encoding="utf-8") as handle:
            audit = [json.loads(line) for line in handle if line.strip()]
        archive = {row["message_id"]: row for row in self.channel.events()}
        return {
            session.person.sender_id: demand_id
            for session in self.sessions if session.person.sender_id in candidates
            if (demand_id := _scoped_waitlist_effect(session.person.sender_id, session.turns, archive, audit, stored))
        }

    def _seed_game(self) -> dict[str, Any]:
        """只预置三位非被测顾客的报名，最后一席保留给正式交互。"""
        organizer = self.early_players[0]
        with using_clock(lambda: self.start_at - timedelta(minutes=50)):
            game, _ = self.store.create_game(
                conversation_id=self.channel.private_conversation_id(organizer.sender_id),
                organizer_id=organizer.sender_id, organizer_name=organizer.name,
                requirement={
                    "game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoking",
                    "start_time_kind": "scheduled", "start_time": self.start_at.strftime("%H:%M"),
                    "planned_start_at": self.start_at.isoformat(),
                    "duration_kind": "fixed", "duration_hours": _FORMATION_DURATION_HOURS,
                },
                known_players=[{"customer_id": player.sender_id, "display_name": player.name,
                                "status": "confirmed", "seat_count": 1} for player in self.early_players],
                trace_id="cohort-initial-inventory",
            )
        if game.remaining_seats() != 1:
            raise ValueError("预置牌局应当恰好空出一个座位")
        self.seeded_game_id = game.game_id
        return game.to_dict()

    def _dialogue_details(self, session: Dialogue, index: int) -> dict[str, Any]:
        """提供本人的当期意图，不透露库存、席位、其他客户或预期验收结果。"""
        details = super()._dialogue_details(session, index)
        details["duration_hours"] = session.person.temperament.preferred_duration_hours
        details["party_size"] = 1
        details["knows_shopkeeper"] = False
        if session.goal == "try_join":
            details.update(stake=1, smoke="non_smoking",
                           case_intent=("你刚加了店家微信，两人还没聊过。今天想一个人找现成局，"
                                        "具体桌况要听店家说。有感兴趣的桌，再聊影响你能不能去的条件。"))
        else:
            details.pop("start_time")
            details.update(
                earliest_start_time=self.start_at.strftime("%H:%M"),
                latest_start_time=(self.start_at + _LOOKUP_WINDOW_LENGTH).strftime("%H:%M"),
                case_intent=("你刚加了店家微信，两人还没聊过。今晚一个人找现成牌局。你已经知道自己可在这段连续时间内任意时间开打。"
                             "没有合适的现成局就先作罢，今晚不打算另开一桌。"),
            )
            if self.case == "mixed":
                details["case_intent"] += f"今天只考虑{session.person.preferred_game}，暂时不换玩法。"
                if session.person.temperament.fallback == "wait_for_update":
                    details["case_intent"] += "如果你仍有兴趣，可以请老板留意后续合适的现成局。"
        return details

    def _fixture_turn(self, session: Dialogue, index: int) -> tuple[str, dict[str, Any]]:
        """离线测试也经过查询取证和明确报名两轮，真实顾客仍由模型自由发言。"""
        if session.goal == "try_join":
            phase = "cohort_offer" if index == 1 else "cohort_join"
            return phase, {"game_id": self.seeded_game_id}
        if self.case == "mixed":
            return "cohort_no_match", {"game_type": _INCOMPATIBLE_GAME_TYPES[session.person.preferred_game],
                                       "followup": index > 1}
        return super()._fixture_turn(session, index)

    def _should_continue(self, session: Dialogue, row: dict[str, Any]) -> bool:
        """收到正式入局确认后结束本次找局，其他情况仍由交付事件驱动。"""
        return super()._should_continue(session, row) and not self._confirmed_after_reply(session)

    def _confirmed_after_reply(self, session: Dialogue) -> bool:
        """要求本人的续轮调用真实入局工具，并已经收到该轮回复。"""
        if not self.seeded_game_id or not session.turns:
            return False
        row = session.turns[-1]
        return bool(row.get("caused_by") and row.get("reply_message_id") and any(
            tool.get("name") in {"join_game", "record_candidate_reply"} and tool.get("called") and tool.get("allowed")
            and not tool.get("error") and (tool.get("result", {}).get("game") or {}).get("game_id") == self.seeded_game_id
            and tool["result"].get("recorded_status") == "confirmed"
            for tool in row.get("tool_results", ())
        ))

    @staticmethod
    def _looked_up_different_game(session: Dialogue) -> bool:
        """核对顾客查的是本人偏好的玩法，且查无匹配时门店仍有其他可报名局。"""
        preferred = session.person.preferred_game
        for row in session.turns:
            for tool in row.get("tool_results", ()):
                if tool.get("name") != "search_current_games" or not tool.get("allowed") or tool.get("error"):
                    continue
                result = tool.get("result", {})
                searched_type = result.get("requirement", {}).get("game_type")
                scope = result.get("customer_reply_contract", {}).get("evidence_scope", {})
                if (searched_type in {_INCOMPATIBLE_GAME_TYPES[preferred], preferred}
                        and result.get("matches") == [] and result.get("alternatives") == []
                        and scope.get("unfiltered_available_pool_empty") is False):
                    return True
        return False


def _opening_query_covers_window(turns: list[dict[str, Any]], start_at: datetime) -> bool:
    """确认工具实际查过连续可开打时段，或者有覆盖它的更宽空结果。"""
    if not turns:
        return False
    for tool in turns[0].get("tool_results", ()):
        if (tool.get("name") != "search_current_games" or not tool.get("called")
                or not tool.get("allowed") or tool.get("error")):
            continue
        result = tool.get("result", {})
        params = result.get("requirement") or {}
        window = current_game_query_start_range(params)
        if window and window == (start_at, start_at + _LOOKUP_WINDOW_LENGTH):
            return True
        # 只有空的更宽查询才可采纳，多个单点不能覆盖分钟级的连续区间。
        if not any(params.get(key) for key in (
            "start_time", "planned_start_at", "start_at", "start_datetime", "start_time_at", "start_time_kind",
        )) and (window is None or window[0] <= start_at
                and window[1] >= start_at + _LOOKUP_WINDOW_LENGTH) and result.get("matches") == [] and result.get("alternatives") == []:
            return True
    return False


def _formation_seeker(person: Persona) -> bool:
    """选择本身有报名意愿且和库存玩法、档位、时长相容的生成画像。"""
    return (person.preferred_game == "杭州麻将" and person.stake_min <= 1 <= person.stake_max
            and "晚上" in person.available_periods and person.temperament.purpose == "find_table"
            and person.temperament.commitment == "ready_to_join"
            and person.temperament.preferred_duration_hours == _FORMATION_DURATION_HOURS)


def _lookup_priority(person: Persona) -> tuple[bool, bool]:
    """按营业时段和来店目标选择角色，保留不同的协商与收尾习惯。"""
    return person.temperament.purpose != "find_table", "晚上" not in person.available_periods


def _scoped_waitlist_effect(customer_id: str, turns, archive, audit, stored) -> str | None:
    """匹配本人本轮的私聊、授权范围、真实工具写入与当前候补状态。"""
    for turn in turns:
        message_id = turn.get("message_id")
        message = archive.get(message_id, {})
        prior_reply = archive.get(turn.get("caused_by"), {})
        reply = archive.get(turn.get("reply_message_id"), {})
        conversation = message.get("conversation_id")
        text = message.get("text")
        if (turn.get("actor_id") != customer_id or not conversation or not isinstance(text, str)
                or message.get("kind") != "private" or message.get("direction") != "inbound"
                or message.get("sender_id") != customer_id or message.get("recipient_id") != AGENT_ID
                or reply.get("source_message_id") != message_id):
            continue
        if not all(record.get("kind") == "private" and record.get("direction") == "outbound"
                   and record.get("sender_id") == AGENT_ID and record.get("recipient_id") == customer_id
                   and record.get("conversation_id") == conversation for record in (prior_reply, reply)):
            continue
        if not any(
            entry.get("event") == "authorization" and entry.get("status") == "authorized"
            and (entry.get("sender_id"), entry.get("conversation_id"), entry.get("source_message_id"))
            == (customer_id, conversation, message_id)
            and any(permission.get("scope") == "waitlist" and isinstance(evidence := permission.get("evidence"), str)
                    and evidence.strip() and evidence in text for permission in entry.get("permissions", ()))
            for entry in audit
        ):
            continue
        for tool in turn.get("tool_results", ()):
            if (tool.get("name") != "register_waiting_demand" or tool.get("called") is not True
                    or tool.get("allowed") is not True or tool.get("error")):
                continue
            created = tool.get("result", {}).get("waiting_demand", {})
            demand_id = created.get("demand_id")
            recorded = stored.get(demand_id, {})
            if (demand_id and all(record.get("sender_id") == customer_id and record.get("conversation_id") == conversation
                                  and record.get("status") == "active" for record in (created, recorded))):
                return demand_id
    return None
