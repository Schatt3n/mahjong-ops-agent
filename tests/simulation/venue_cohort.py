"""以独立门店的两种初始库存核对真实私聊能否办成局或如实查无匹配。"""

from __future__ import annotations

from datetime import timedelta
import random
from typing import Any

from mahjong_agent_runtime.models import using_clock

from .venue_actors import Persona
from .venue_dialogues import Dialogue, VenueDialogues


class VenueCohort(VenueDialogues):
    """场景只设定初始库存与当事人意向，查桌、答复和确认均由真实交互产生。"""

    def __init__(self, *, people, case: str, seed: int, active_users: int, **kwargs) -> None:
        """挑选画像相容的生成角色，保持每个人的原始性格和关系。"""
        people = list(people)
        if case not in {"formation", "no-match"} or active_users < 1 or active_users > len(people):
            raise ValueError("小批验收要求有效场景和至少一位候选顾客")
        if case == "formation" and active_users != 1:
            raise ValueError("单局最后一席验收只接受一位活跃顾客")
        super().__init__(people=people, seed=seed, active_users=active_users, **kwargs)
        self.case = case
        ranked = list(people)
        random.Random(seed).shuffle(ranked)
        if case == "formation":
            seeker = next((person for person in ranked if _formation_seeker(person)), None)
            if seeker is None:
                raise ValueError("当前种子没有适合查找这桌的画像")
            selected = [seeker]
            blocked = {seeker.sender_id, *(edge.target_sender_id for edge in seeker.relationships
                                           if edge.kind == "avoid_same_table")}
            self.early_players = [person for person in ranked if person.sender_id not in blocked][:3]
            if len(self.early_players) != 3:
                raise ValueError("当前种子不足以安排三位已报名玩家")
        else:
            ranked.sort(key=lambda person: (
                person.temperament.purpose != "find_table", "晚上" not in person.available_periods,
                person.temperament.fallback != "follow_up", person.temperament.negotiation != "adjust_time",
            ))
            selected = ranked[:active_users]
            self.early_players = []
        self.sessions = [Dialogue(person, "try_join" if case == "formation" else "lookup") for person in selected]
        occupied = {person.sender_id for person in [*selected, *self.early_players]}
        self.background = [person for person in ranked if person.sender_id not in occupied][:len(self.background)]
        self.seeded_game_id: str | None = None

    def run(self) -> dict[str, Any]:
        """核对严格私聊因果链和审批后的真实席位，保留门店初态与终态。"""
        if self.store.games:
            raise ValueError("小批验收必须由一个没有历史牌局的独立门店开始")
        initial = self._seed_game() if self.case == "formation" else {}
        report = super().run()
        expected = "formed" if self.case == "formation" else "no_match"
        findings = report["acceptance"]
        report["checks"]["cohort_each_customer_has_causal_multi_turn"] = (
            findings["multi_round_private_customers"] == len(self.sessions)
        )
        report["checks"]["cohort_expected_business_outcome"] = findings["outcome_counts"] == {expected: len(self.sessions)}
        report["cohort"] = {"case": self.case, "expected_outcome": expected, "initial_game": initial, "final_game": None}
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

    def _seed_game(self) -> dict[str, Any]:
        """只预置三位非被测顾客的报名，最后一席保留给正式交互。"""
        organizer = self.early_players[0]
        duration = self.sessions[0].person.temperament.preferred_duration_hours
        with using_clock(lambda: self.start_at - timedelta(minutes=50)):
            game, _ = self.store.create_game(
                conversation_id=self.channel.private_conversation_id(organizer.sender_id),
                organizer_id=organizer.sender_id, organizer_name=organizer.name,
                requirement={
                    "game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoking",
                    "start_time_kind": "scheduled", "start_time": self.start_at.strftime("%H:%M"),
                    "planned_start_at": self.start_at.isoformat(),
                    "duration_kind": "fixed", "duration_hours": duration,
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
        if self.case == "formation":
            details.update(stake=1, smoke="non_smoking",
                           case_intent="今天想一个人找现成局。第一次先问清楚有哪些选择；实际收到条件合适的选项后，你自己决定要不要明确报名。")
        else:
            details["case_intent"] = ("今天一个人只想查已有的牌局，暂时不请店家新组局或登记候补。"
                                      "如果想要的时段没有，再考虑问稍晚一点或其他可接受的时段，然后按自己的情况决定。")
        return details

    def _fixture_turn(self, session: Dialogue, index: int) -> tuple[str, dict[str, Any]]:
        """离线测试也经过查询取证和明确报名两轮，真实顾客仍由模型自由发言。"""
        if self.case == "formation":
            phase = "cohort_offer" if index == 1 else "cohort_join"
            return phase, {"game_id": self.seeded_game_id, "duration": session.person.temperament.preferred_duration_hours}
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


def _formation_seeker(person: Persona) -> bool:
    """选择本身有报名意愿且和库存玩法、档位、时长相容的生成画像。"""
    return (person.preferred_game == "杭州麻将" and person.stake_min <= 1 <= person.stake_max
            and "晚上" in person.available_periods and person.temperament.purpose == "find_table"
            and person.temperament.commitment == "ready_to_join" and person.temperament.preferred_duration_hours <= 4)
