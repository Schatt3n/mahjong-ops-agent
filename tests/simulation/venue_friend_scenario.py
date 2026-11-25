"""核验已有好友从本人私聊发起新局，并以实际交付和落库为准。"""

from __future__ import annotations

from datetime import timedelta
import json

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.knowledge import default_terminology_repository
from mahjong_agent_runtime.models import GameStatus

from .venue_acceptance import _delivered_replies
from .venue_channel import AGENT_ID
from .venue_dialogues import Dialogue, VenueDialogues
from .venue_write_guard import VenueGuardedToolGateway


class VenueFriendScenario(VenueDialogues):
    """从空局池开始，由好友角色私聊委托、承接回复并创建自己的待组局。"""

    def __init__(self, *, people, **options) -> None:
        super().__init__(people=people, **options)
        self._starts = {}
        for index, session in enumerate(self.sessions):
            session.goal = "organize_new"
            self._starts[session.person.sender_id] = self.start_at + timedelta(minutes=30 * (index % 3))

    def _dialogue_details(self, session: Dialogue, index: int) -> dict:
        """把当前来意和个人条件交给角色模型，话术与续聊由角色自行判断。"""
        details = super()._dialogue_details(session, index)
        start = self._starts.get(session.person.sender_id, self.start_at)
        duration = min(session.person.temperament.preferred_duration_hours, 23 - start.hour)
        return {
            **details,
            "start_time": start.strftime("%H:%M"),
            "duration_hours": duration,
            "party_size": 1,
            "case_intent": ("你和老板早就是好友。你想在自己的空闲时段发起一桌新局，今天先自己占一位，"
                            "请老板帮忙找其余人。可以先说来意，再补足老板追问的条件。"
                            "若已经收到登记结果，可以询问还缺几位或如何等消息。"
                            "本轮只委托组局找人，暂时不单独预订包间。"),
        }

    def _fixture_turn(self, session: Dialogue, index: int) -> tuple[str, dict]:
        """离线让正式入口先问条件，再按已收到的明确委托调用建局。"""
        if index == 1:
            return "dialogue_open", {}
        details = self._dialogue_details(session, index)
        start = self._starts[session.person.sender_id]
        term = default_terminology_repository().first_match(session.person.preferred_game, categories={"game_type"})
        game_type = term.term.canonical["requested_game"] if term else session.person.preferred_game
        return "need", {"requirement": {
            "game_type": game_type,
            "stake": str(details["stake"]),
            "duration_kind": "fixed",
            "duration_hours": details["duration_hours"],
            "start_time_kind": "scheduled",
            "start_time": details["start_time"],
            "planned_start_at": start.isoformat(),
        }}

    def _should_continue(self, session: Dialogue, row: dict) -> bool:
        """登记后仍保留至少两轮往来；缺少实际回复或角色选择停止时正常结束。"""
        if not super()._should_continue(session, row):
            return False
        created = any(_successful_creations(turn) for turn in session.turns)
        return len(session.turns) < 2 or not created

    def run(self) -> dict:
        """区分开口组局、正式登记和人数凑齐，并逐人核验前两项。"""
        if self.store.games:
            raise ValueError("private initiation scenario requires an empty game inventory")
        friends = self.channel.owner_friends
        if any(session.person.sender_id not in friends for session in self.sessions):
            raise ValueError("each selected organizer must be an existing owner friend")
        report = super().run()
        archived = {row["message_id"]: row for row in self.channel.events()}
        positions = {key: index for index, key in enumerate(archived)}
        replies = _delivered_replies(archived, positions)
        gateway = self.channel.runtime.tool_gateway
        decisions = None
        if isinstance(gateway, VenueGuardedToolGateway):
            decisions = []
            if gateway.guard.log_path.is_file():
                with gateway.guard.log_path.open(encoding="utf-8") as handle:
                    decisions = [json.loads(line) for line in handle if line.strip()]
        persisted = SQLiteAgentStore(self.store.path).games
        by_customer = {}
        private_openers = 0
        for session in self.sessions:
            customer_id = session.person.sender_id
            conversation = self.channel.private_conversation_id(customer_id)
            first = session.turns[0] if session.turns else {}
            if not first.get("caused_by") and _private_input(archived.get(first.get("message_id"), {}), customer_id, conversation):
                private_openers += 1
            by_customer[customer_id] = _private_created_ids(
                session, conversation, archived, replies, persisted, decisions,
            )
        game_ids = set().union(*by_customer.values())
        checks = report["checks"]
        checks["existing_friends_can_open_private_dialogues"] = private_openers == len(self.sessions)
        checks["each_friend_has_causal_private_followup"] = (
            report["acceptance"]["multi_round_private_customers"] == len(self.sessions)
        )
        checks["each_friend_created_own_new_game"] = all(len(ids) == 1 for ids in by_customer.values())
        checks["created_games_have_distinct_private_origins"] = len(game_ids) == len(self.sessions) and game_ids == set(persisted)
        report["private_initiation"] = {
            "owner_friends": len(friends),
            "private_openers": private_openers,
            "verified_organizers": sum(len(ids) == 1 for ids in by_customer.values()),
            "persisted_new_games": len(game_ids),
            "forming_games": sum(persisted[key].status == GameStatus.FORMING for key in game_ids),
            "recruiting_games": sum(persisted[key].status == GameStatus.INVITING for key in game_ids),
            "scoped_creation_evidence_required": decisions is not None,
        }
        return report


def _private_input(row: dict, customer_id: str, conversation: str) -> bool:
    """只接受该好友发给老板且归档在本人私聊中的真实入站。"""
    return (row.get("kind") == "private" and row.get("direction") == "inbound"
            and row.get("sender_id") == customer_id and row.get("recipient_id") == AGENT_ID
            and row.get("conversation_id") == conversation)


def _successful_creations(turn: dict) -> list[dict]:
    """工具必须通过正式网关、实际执行并返回建局结果。"""
    return [tool for tool in turn.get("tool_results", ()) if tool.get("name") == "create_game"
            and tool.get("called") is True and tool.get("allowed") is True and not tool.get("error")]


def _private_created_ids(session: Dialogue, conversation: str, archive: dict, replies: dict, persisted: dict,
                         decisions: list[dict] | None) -> set[str]:
    """同一私聊的授权、正式建局、已送达回复与落库组织者必须一致。"""
    customer_id = session.person.sender_id
    games = set()
    for turn in session.turns:
        message_id = turn.get("message_id")
        message = archive.get(message_id, {})
        if not _private_input(message, customer_id, conversation) or not any(
            item["kind"] == "private" and item["conversation_id"] == conversation for item in replies.get(message_id, ())
        ):
            continue
        if decisions is not None and not any(
            row.get("event") == "authorization" and row.get("status") == "authorized"
            and (row.get("sender_id"), row.get("conversation_id"), row.get("source_message_id"))
            == (customer_id, conversation, message_id)
            and any(item.get("scope") == "organize" and item.get("evidence")
                    and item["evidence"] in message["text"] for item in row.get("permissions", ()))
            for row in decisions
        ):
            continue
        for tool in _successful_creations(turn):
            created = tool.get("result", {}).get("game") or {}
            game_id = created.get("game_id")
            game = persisted.get(game_id)
            if (game and game.organizer_id == created.get("organizer_id") == customer_id
                    and game.conversation_id == created.get("conversation_id") == conversation
                    and game.option_root_game_id is None and game.status in {GameStatus.FORMING, GameStatus.INVITING}):
                games.add(game_id)
    return games
