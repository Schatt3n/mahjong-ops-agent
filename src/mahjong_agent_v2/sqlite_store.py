from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    AgentRuntimeResultV2,
    ConversationRoleV2,
    ConversationTurnV2,
    CustomerProfileV2,
    DEFAULT_TZ_V2,
    GameParticipantV2,
    GameStatusV2,
    GameV2,
    InviteDraftV2,
    InviteStatusV2,
    StateTransitionV2,
    ToolResultV2,
    new_id,
)
from .state_policy import StatePolicyV2
from .store import _score_customer, _score_requirement


@dataclass(slots=True)
class SQLiteAgentStoreV2:
    """SQLite-backed V2 store.

    This is independent from the legacy trial SQLite schema. It stores V2
    customers, games, invite drafts, conversation turns, idempotency ledger and
    state transitions in dedicated tables so the new agent runtime can survive
    process restarts.
    """

    path: str | Path
    state_policy: StatePolicyV2 = field(default_factory=StatePolicyV2.default)
    _connection: sqlite3.Connection = field(init=False, repr=False)
    _lock: threading.RLock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._migrate()

    @property
    def customers(self) -> dict[str, CustomerProfileV2]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM v2_customers").fetchall()
            return {customer.customer_id: customer for customer in (_customer_from_json(row["payload"]) for row in rows)}

    @property
    def games(self) -> dict[str, GameV2]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM v2_games").fetchall()
            return {game.game_id: game for game in (_game_from_json(row["payload"]) for row in rows)}

    @property
    def invite_drafts(self) -> dict[str, InviteDraftV2]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM v2_invite_drafts").fetchall()
            return {draft.draft_id: draft for draft in (_invite_from_json(row["payload"]) for row in rows)}

    @property
    def transitions(self) -> list[StateTransitionV2]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM v2_state_transitions ORDER BY id").fetchall()
            return [_transition_from_json(row["payload"]) for row in rows]

    def upsert_customer(self, profile: CustomerProfileV2) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO v2_customers(customer_id, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(customer_id) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (profile.customer_id, _dump(profile.to_dict()), _now_iso()),
            )

    def append_turn(self, conversation_id: str, turn: ConversationTurnV2) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO v2_conversation_turns(conversation_id, trace_id, role, occurred_at, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    turn.trace_id,
                    turn.role.value,
                    turn.occurred_at.isoformat(),
                    _dump(turn.to_dict()),
                ),
            )

    def append_user_turn(self, message, trace_id: str) -> None:
        self.append_turn(
            message.conversation_id,
            ConversationTurnV2(
                role=ConversationRoleV2.USER,
                content=message.text,
                trace_id=trace_id,
                sender_id=message.sender_id,
                sender_name=message.sender_name,
                occurred_at=message.sent_at,
            ),
        )

    def append_assistant_turn(self, conversation_id: str, content: str, trace_id: str) -> None:
        self.append_turn(
            conversation_id,
            ConversationTurnV2(
                role=ConversationRoleV2.ASSISTANT,
                content=content,
                trace_id=trace_id,
            ),
        )

    def append_tool_turn(self, conversation_id: str, content: str, trace_id: str) -> None:
        self.append_turn(
            conversation_id,
            ConversationTurnV2(
                role=ConversationRoleV2.TOOL,
                content=content,
                trace_id=trace_id,
            ),
        )

    def recent_turns(self, conversation_id: str, limit: int = 12) -> list[ConversationTurnV2]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload
                FROM v2_conversation_turns
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, int(limit)),
            ).fetchall()
            turns = [_turn_from_json(row["payload"]) for row in rows]
            return list(reversed(turns))

    def active_games(self, conversation_id: str | None = None) -> list[GameV2]:
        games = [
            game
            for game in self.games.values()
            if game.status in self.state_policy.active_game_statuses
        ]
        if conversation_id:
            scoped = [game for game in games if game.conversation_id == conversation_id]
            return scoped or games
        return games

    def idempotent_result(self, key: str | None) -> ToolResultV2 | None:
        if not key:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM v2_idempotency_ledger WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            return _tool_result_from_json(row["payload"])

    def remember_result(self, key: str | None, result: ToolResultV2) -> None:
        if not key:
            return
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO v2_idempotency_ledger(idempotency_key, payload, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(idempotency_key) DO NOTHING
                """,
                (key, _dump(result.to_dict()), _now_iso()),
            )

    def idempotent_message_result(self, message_id: str | None) -> AgentRuntimeResultV2 | None:
        if not message_id:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM v2_message_results WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if row is None:
                return None
            return AgentRuntimeResultV2.from_payload(json.loads(row["payload"]))

    def remember_message_result(self, message_id: str | None, result: AgentRuntimeResultV2) -> None:
        if not message_id:
            return
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO v2_message_results(message_id, conversation_id, trace_id, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO NOTHING
                """,
                (
                    message_id,
                    _conversation_id_from_result(result),
                    result.trace_id,
                    _dump(result.to_dict()),
                    _now_iso(),
                ),
            )

    def search_current_games(self, requirement: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for game in self.active_games():
            score, reasons = _score_requirement(requirement, game.requirement)
            if game.remaining_seats() <= 0:
                continue
            if score <= 0 and requirement:
                continue
            scored.append(
                {
                    "game": game.to_dict(),
                    "score": score,
                    "reasons": reasons or ["active_open_game"],
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: int(limit)]

    def search_customers(
        self,
        requirement: dict[str, Any],
        *,
        exclude_customer_ids: list[str] | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        excluded = set(exclude_customer_ids or [])
        scored: list[dict[str, Any]] = []
        for customer in self.customers.values():
            if customer.no_contact or customer.customer_id in excluded:
                continue
            if self.active_game_for_customer(customer.customer_id) is not None:
                continue
            score, reasons = _score_customer(requirement, customer)
            if score <= 0:
                continue
            scored.append({"customer": customer.to_dict(), "score": score, "reasons": reasons})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: int(limit)]

    def create_game(
        self,
        *,
        conversation_id: str,
        organizer_id: str,
        organizer_name: str,
        requirement: dict[str, Any],
        known_players: list[dict[str, Any]] | None,
        trace_id: str,
    ) -> tuple[GameV2, StateTransitionV2]:
        with self._lock, self._connection:
            game = GameV2(
                game_id=new_id("gamev2"),
                conversation_id=conversation_id,
                organizer_id=organizer_id,
                organizer_name=organizer_name,
                requirement=dict(requirement),
                participants=[],
            )
            self.state_policy.ensure_game_transition(None, game.status)
            players = list(known_players or [])
            if not players:
                players = [{"customer_id": organizer_id, "display_name": organizer_name, "source": "organizer"}]
            for player in players:
                customer_id = str(player.get("customer_id") or player.get("id") or "").strip()
                display_name = str(player.get("display_name") or player.get("name") or customer_id or "客户").strip()
                if not customer_id:
                    customer_id = new_id("guest")
                if not any(participant.customer_id == customer_id for participant in game.participants):
                    game.participants.append(
                        GameParticipantV2(
                            customer_id=customer_id,
                            display_name=display_name,
                            status=str(player.get("status") or "joined"),
                            source=str(player.get("source") or "organizer"),
                        )
                    )
            self._upsert_game(game)
            transition = self._transition(
                entity_type="game",
                entity_id=game.game_id,
                from_status=None,
                to_status=game.status.value,
                reason="create_game",
                trace_id=trace_id,
            )
            return game, transition

    def create_invite_drafts(
        self,
        *,
        game_id: str,
        invitations: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[list[InviteDraftV2], list[StateTransitionV2]]:
        with self._lock, self._connection:
            game = self._game_by_id(game_id)
            if game is None:
                raise ValueError(f"game_id not found: {game_id}")
            self.state_policy.ensure_can_create_invite_drafts(game)
            drafts: list[InviteDraftV2] = []
            transitions: list[StateTransitionV2] = []
            customers = self.customers
            for invitation in invitations:
                customer_id = str(invitation.get("customer_id") or "").strip()
                message_text = str(invitation.get("message_text") or "").strip()
                if not customer_id or not message_text:
                    raise ValueError("each invitation requires customer_id and message_text")
                if self.active_game_for_customer(customer_id) is not None:
                    continue
                profile = customers.get(customer_id)
                display_name = str(invitation.get("display_name") or (profile.display_name if profile else customer_id))
                draft = InviteDraftV2(
                    draft_id=new_id("draftv2"),
                    game_id=game_id,
                    customer_id=customer_id,
                    display_name=display_name,
                    message_text=message_text,
                    metadata={"trace_id": trace_id},
                )
                self._upsert_invite(draft)
                drafts.append(draft)
            if drafts and game.status == GameStatusV2.FORMING:
                from_status = game.status.value
                self.state_policy.ensure_game_transition(game.status, GameStatusV2.INVITING)
                game.status = GameStatusV2.INVITING
                game.updated_at = datetime.now(DEFAULT_TZ_V2)
                self._upsert_game(game)
                transitions.append(
                    self._transition(
                        entity_type="game",
                        entity_id=game.game_id,
                        from_status=from_status,
                        to_status=game.status.value,
                        reason="invite_drafts_created",
                        trace_id=trace_id,
                    )
                )
            return drafts, transitions

    def record_candidate_reply(
        self,
        *,
        game_id: str,
        customer_id: str,
        status: str,
        trace_id: str,
    ) -> tuple[GameV2, list[StateTransitionV2]]:
        with self._lock, self._connection:
            game = self._game_by_id(game_id)
            if game is None:
                raise ValueError(f"game_id not found: {game_id}")
            next_status = InviteStatusV2(status)
            transitions: list[StateTransitionV2] = []
            drafts = [
                draft
                for draft in self.invite_drafts.values()
                if draft.game_id == game_id and draft.customer_id == customer_id
            ]
            self.state_policy.ensure_candidate_reply_allowed(game, drafts)
            for draft in drafts:
                self.state_policy.ensure_invite_transition(draft.status, next_status)
                if draft.status == next_status:
                    continue
                from_status = draft.status.value
                draft.status = next_status
                draft.updated_at = datetime.now(DEFAULT_TZ_V2)
                self._upsert_invite(draft)
                transitions.append(
                    self._transition(
                        entity_type="invite_draft",
                        entity_id=draft.draft_id,
                        from_status=from_status,
                        to_status=draft.status.value,
                        reason="candidate_reply",
                        trace_id=trace_id,
                    )
                )
            if next_status == InviteStatusV2.CONFIRMED and not any(
                participant.customer_id == customer_id for participant in game.participants
            ):
                profile = self.customers.get(customer_id)
                game.participants.append(
                    GameParticipantV2(
                        customer_id=customer_id,
                        display_name=profile.display_name if profile else customer_id,
                        status="confirmed",
                        source="candidate_reply",
                    )
                )
            if game.remaining_seats() == 0 and game.status != GameStatusV2.READY:
                from_status = game.status.value
                self.state_policy.ensure_game_transition(game.status, GameStatusV2.READY)
                game.status = GameStatusV2.READY
                game.updated_at = datetime.now(DEFAULT_TZ_V2)
                transitions.append(
                    self._transition(
                        entity_type="game",
                        entity_id=game.game_id,
                        from_status=from_status,
                        to_status=game.status.value,
                        reason="all_seats_confirmed",
                        trace_id=trace_id,
                    )
                )
            self._upsert_game(game)
            return game, transitions

    def update_game_status(
        self,
        *,
        game_id: str,
        status: str,
        reason: str,
        trace_id: str,
    ) -> tuple[GameV2, StateTransitionV2]:
        with self._lock, self._connection:
            game = self._game_by_id(game_id)
            if game is None:
                raise ValueError(f"game_id not found: {game_id}")
            next_status = GameStatusV2(status)
            from_status = game.status.value
            self.state_policy.ensure_game_transition(game.status, next_status)
            game.status = next_status
            game.updated_at = datetime.now(DEFAULT_TZ_V2)
            self._upsert_game(game)
            transition = self._transition(
                entity_type="game",
                entity_id=game.game_id,
                from_status=from_status,
                to_status=game.status.value,
                reason=reason or "update_game_status",
                trace_id=trace_id,
            )
            return game, transition

    def active_game_for_customer(self, customer_id: str) -> GameV2 | None:
        for game in self.games.values():
            if game.status not in self.state_policy.active_game_statuses:
                continue
            if any(
                participant.customer_id == customer_id and participant.status in {"joined", "confirmed"}
                for participant in game.participants
            ):
                return game
        for draft in self.invite_drafts.values():
            if draft.customer_id == customer_id and draft.status in {
                *self.state_policy.occupied_invite_statuses,
            }:
                game = self.games.get(draft.game_id)
                if game and game.status in self.state_policy.active_game_statuses:
                    return game
        return None

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _migrate(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS v2_customers (
                customer_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS v2_games (
                game_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_v2_games_conversation_status
                ON v2_games(conversation_id, status);
            CREATE TABLE IF NOT EXISTS v2_invite_drafts (
                draft_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_v2_invites_customer_status
                ON v2_invite_drafts(customer_id, status);
            CREATE TABLE IF NOT EXISTS v2_conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                role TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_v2_turns_conversation_id
                ON v2_conversation_turns(conversation_id, id);
            CREATE TABLE IF NOT EXISTS v2_idempotency_ledger (
                idempotency_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS v2_message_results (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_v2_message_results_conversation
                ON v2_message_results(conversation_id, created_at);
            CREATE TABLE IF NOT EXISTS v2_state_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_v2_transitions_entity
                ON v2_state_transitions(entity_type, entity_id);
            """
        )
        self._connection.commit()

    def _upsert_game(self, game: GameV2) -> None:
        self._connection.execute(
            """
            INSERT INTO v2_games(game_id, conversation_id, status, payload, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(game_id) DO UPDATE SET
                conversation_id=excluded.conversation_id,
                status=excluded.status,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (game.game_id, game.conversation_id, game.status.value, _dump(game.to_dict()), game.updated_at.isoformat()),
        )

    def _upsert_invite(self, draft: InviteDraftV2) -> None:
        self._connection.execute(
            """
            INSERT INTO v2_invite_drafts(draft_id, game_id, customer_id, status, payload, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(draft_id) DO UPDATE SET
                game_id=excluded.game_id,
                customer_id=excluded.customer_id,
                status=excluded.status,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (
                draft.draft_id,
                draft.game_id,
                draft.customer_id,
                draft.status.value,
                _dump(draft.to_dict()),
                draft.updated_at.isoformat(),
            ),
        )

    def _game_by_id(self, game_id: str) -> GameV2 | None:
        row = self._connection.execute("SELECT payload FROM v2_games WHERE game_id = ?", (game_id,)).fetchone()
        return _game_from_json(row["payload"]) if row else None

    def _transition(
        self,
        *,
        entity_type: str,
        entity_id: str,
        from_status: str | None,
        to_status: str,
        reason: str,
        trace_id: str,
    ) -> StateTransitionV2:
        transition = StateTransitionV2(
            entity_type=entity_type,
            entity_id=entity_id,
            from_status=from_status,
            to_status=to_status,
            reason=reason,
            trace_id=trace_id,
        )
        self._connection.execute(
            """
            INSERT INTO v2_state_transitions(trace_id, entity_type, entity_id, payload)
            VALUES (?, ?, ?, ?)
            """,
            (trace_id, entity_type, entity_id, _dump(transition.to_dict())),
        )
        return transition


def _customer_from_json(raw: str) -> CustomerProfileV2:
    data = json.loads(raw)
    return CustomerProfileV2(
        customer_id=str(data["customer_id"]),
        display_name=str(data["display_name"]),
        gender=data.get("gender"),
        preferred_games=[str(item) for item in data.get("preferred_games") or []],
        preferred_stakes=[str(item) for item in data.get("preferred_stakes") or []],
        preferred_time_tags=[str(item) for item in data.get("preferred_time_tags") or []],
        smoke_preference=data.get("smoke_preference"),
        fatigue_score=float(data.get("fatigue_score") or 0),
        response_score=float(data.get("response_score") or 0),
        no_contact=bool(data.get("no_contact", False)),
        notes=str(data.get("notes") or ""),
    )


def _game_from_json(raw: str) -> GameV2:
    data = json.loads(raw)
    return GameV2(
        game_id=str(data["game_id"]),
        conversation_id=str(data["conversation_id"]),
        organizer_id=str(data["organizer_id"]),
        organizer_name=str(data["organizer_name"]),
        requirement=dict(data.get("requirement") or {}),
        status=GameStatusV2(str(data.get("status") or GameStatusV2.FORMING.value)),
        participants=[
            GameParticipantV2(
                customer_id=str(item.get("customer_id") or ""),
                display_name=str(item.get("display_name") or ""),
                status=str(item.get("status") or "joined"),
                source=str(item.get("source") or "organizer"),
            )
            for item in data.get("participants") or []
        ],
        seats_total=int(data.get("seats_total") or 4),
        created_at=_parse_datetime(data.get("created_at")),
        updated_at=_parse_datetime(data.get("updated_at")),
    )


def _invite_from_json(raw: str) -> InviteDraftV2:
    data = json.loads(raw)
    return InviteDraftV2(
        draft_id=str(data["draft_id"]),
        game_id=str(data["game_id"]),
        customer_id=str(data["customer_id"]),
        display_name=str(data["display_name"]),
        message_text=str(data["message_text"]),
        status=InviteStatusV2(str(data.get("status") or InviteStatusV2.PENDING_APPROVAL.value)),
        created_at=_parse_datetime(data.get("created_at")),
        updated_at=_parse_datetime(data.get("updated_at")),
        metadata=dict(data.get("metadata") or {}),
    )


def _turn_from_json(raw: str) -> ConversationTurnV2:
    data = json.loads(raw)
    return ConversationTurnV2(
        role=ConversationRoleV2(str(data.get("role") or ConversationRoleV2.USER.value)),
        content=str(data.get("content") or ""),
        trace_id=str(data.get("trace_id") or ""),
        sender_id=data.get("sender_id"),
        sender_name=data.get("sender_name"),
        metadata=dict(data.get("metadata") or {}),
        occurred_at=_parse_datetime(data.get("occurred_at")),
    )


def _transition_from_json(raw: str) -> StateTransitionV2:
    data = json.loads(raw)
    return StateTransitionV2(
        entity_type=str(data["entity_type"]),
        entity_id=str(data["entity_id"]),
        from_status=data.get("from_status"),
        to_status=str(data["to_status"]),
        reason=str(data.get("reason") or ""),
        trace_id=str(data.get("trace_id") or ""),
        occurred_at=_parse_datetime(data.get("occurred_at")),
    )


def _tool_result_from_json(raw: str) -> ToolResultV2:
    data = json.loads(raw)
    return ToolResultV2(
        name=str(data["name"]),
        called=bool(data.get("called")),
        allowed=bool(data.get("allowed")),
        result=dict(data.get("result") or {}),
        error=data.get("error"),
        idempotency_key=data.get("idempotency_key"),
        deduplicated=bool(data.get("deduplicated", False)),
        state_transitions=[_transition_from_json(_dump(item)) for item in data.get("state_transitions") or []],
    )


def _conversation_id_from_result(result: AgentRuntimeResultV2) -> str:
    if result.conversation_id:
        return result.conversation_id
    for tool_result in result.tool_results:
        game = tool_result.result.get("game") if isinstance(tool_result.result, dict) else None
        if isinstance(game, dict) and game.get("conversation_id"):
            return str(game["conversation_id"])
        drafts = tool_result.result.get("drafts") if isinstance(tool_result.result, dict) else None
        if isinstance(drafts, list):
            for draft in drafts:
                if isinstance(draft, dict) and draft.get("conversation_id"):
                    return str(draft["conversation_id"])
    return ""


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _now_iso() -> str:
    return datetime.now(DEFAULT_TZ_V2).isoformat()


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if not value:
        return datetime.now(DEFAULT_TZ_V2)
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=DEFAULT_TZ_V2)
