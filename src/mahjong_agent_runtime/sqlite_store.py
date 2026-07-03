from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    AgentAction,
    AgentRuntimeResult,
    ConversationCheckpoint,
    ConversationRole,
    ConversationTurn,
    CustomerProfile,
    DEFAULT_TZ,
    GameParticipant,
    GameStatus,
    Game,
    InviteDraft,
    InviteStatus,
    OutboundDraftStatus,
    OutboundMessageDraft,
    StateTransition,
    ToolCall,
    ToolResult,
)
from .store import (
    ALLOWED_GAME_TRANSITIONS,
    invite_status_from_candidate_status,
    score_customer,
    score_requirement,
)


@dataclass(slots=True)
class SQLiteAgentStore:
    path: str | Path
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
    def customers(self) -> dict[str, CustomerProfile]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM runtime_customers").fetchall()
            return {item.customer_id: item for item in (_customer_from_payload(_loads(row["payload"])) for row in rows)}

    @property
    def games(self) -> dict[str, Game]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM runtime_games").fetchall()
            return {item.game_id: item for item in (_game_from_payload(_loads(row["payload"])) for row in rows)}

    @property
    def invite_drafts(self) -> dict[str, InviteDraft]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM runtime_invite_drafts").fetchall()
            return {item.draft_id: item for item in (_invite_from_payload(_loads(row["payload"])) for row in rows)}

    @property
    def outbound_message_drafts(self) -> dict[str, OutboundMessageDraft]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM runtime_outbound_message_drafts").fetchall()
            return {item.draft_id: item for item in (_outbound_message_draft_from_payload(_loads(row["payload"])) for row in rows)}

    @property
    def conversation_checkpoints(self) -> dict[str, ConversationCheckpoint]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM runtime_conversation_checkpoints").fetchall()
            return {
                item.conversation_id: item
                for item in (_checkpoint_from_payload(_loads(row["payload"])) for row in rows)
            }

    @property
    def transitions(self) -> list[StateTransition]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM runtime_state_transitions ORDER BY id").fetchall()
            return [_transition_from_payload(_loads(row["payload"])) for row in rows]

    @property
    def badcases(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM runtime_badcases ORDER BY id").fetchall()
            return [_loads(row["payload"]) for row in rows]

    def upsert_customer(self, profile: CustomerProfile) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runtime_customers(customer_id, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(customer_id) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (profile.customer_id, _dumps(profile.to_dict()), _now_iso()),
            )

    def append_user_turn(self, message, trace_id: str) -> None:
        self.append_turn(
            message.conversation_id,
            ConversationTurn(
                role=ConversationRole.USER,
                content=message.text,
                trace_id=trace_id,
                sender_id=message.sender_id,
                sender_name=message.sender_name,
                occurred_at=message.sent_at,
            ),
        )

    def append_assistant_turn(self, conversation_id: str, text: str, trace_id: str) -> None:
        self.append_turn(
            conversation_id,
            ConversationTurn(role=ConversationRole.ASSISTANT, content=text, trace_id=trace_id),
        )

    def append_tool_turn(self, conversation_id: str, text: str, trace_id: str) -> None:
        self.append_turn(
            conversation_id,
            ConversationTurn(role=ConversationRole.TOOL, content=text, trace_id=trace_id),
        )

    def append_turn(self, conversation_id: str, turn: ConversationTurn) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runtime_conversation_turns(conversation_id, trace_id, role, occurred_at, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, turn.trace_id, turn.role.value, turn.occurred_at.isoformat(), _dumps(turn.to_dict())),
            )

    def recent_turns(self, conversation_id: str, limit: int = 30) -> list[ConversationTurn]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload
                FROM runtime_conversation_turns
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, int(limit)),
            ).fetchall()
            turns = [_turn_from_payload(_loads(row["payload"])) for row in rows]
            return list(reversed(turns))

    def get_conversation_checkpoint(self, conversation_id: str) -> ConversationCheckpoint | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_conversation_checkpoints WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            return _checkpoint_from_payload(_loads(row["payload"]))

    def upsert_conversation_checkpoint(
        self,
        *,
        conversation_id: str,
        summary: str,
        facts: dict[str, Any],
        open_questions: list[str],
        trace_id: str,
    ) -> tuple[ConversationCheckpoint, StateTransition]:
        with self._lock, self._connection:
            previous = self.get_conversation_checkpoint(conversation_id)
            checkpoint = ConversationCheckpoint(
                conversation_id=conversation_id,
                summary=summary,
                facts=dict(facts),
                open_questions=list(open_questions),
                source_trace_id=trace_id,
            )
            transition = StateTransition(
                "conversation_checkpoint",
                conversation_id,
                "exists" if previous else None,
                "updated",
                "update_context_checkpoint",
                trace_id,
            )
            self._connection.execute(
                """
                INSERT INTO runtime_conversation_checkpoints(conversation_id, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (conversation_id, _dumps(checkpoint.to_dict()), checkpoint.updated_at.isoformat()),
            )
            self._append_transition(transition)
            return checkpoint, transition

    def active_games(self, conversation_id: str | None = None) -> list[Game]:
        games = [
            item
            for item in self.games.values()
            if item.status.value in {GameStatus.FORMING.value, GameStatus.INVITING.value, GameStatus.READY.value}
        ]
        if conversation_id:
            scoped = [item for item in games if item.conversation_id == conversation_id]
            return scoped or games
        return games

    def idempotent_result(self, key: str | None) -> ToolResult | None:
        if not key:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_idempotency_ledger WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            return _tool_result_from_payload(_loads(row["payload"]))

    def claim_idempotent_result(self, key: str | None, claimed_result: ToolResult) -> tuple[bool, ToolResult | None]:
        if not key:
            return True, None
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO runtime_idempotency_ledger(idempotency_key, payload, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(idempotency_key) DO NOTHING
                """,
                (key, _dumps(claimed_result.to_dict()), _now_iso()),
            )
            if cursor.rowcount == 1:
                return True, None
            row = self._connection.execute(
                "SELECT payload FROM runtime_idempotency_ledger WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return False, None
            return False, _tool_result_from_payload(_loads(row["payload"]))

    def remember_result(self, key: str | None, result: ToolResult) -> None:
        if not key:
            return
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runtime_idempotency_ledger(idempotency_key, payload, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(idempotency_key) DO UPDATE SET
                    payload=excluded.payload
                """,
                (key, _dumps(result.to_dict()), _now_iso()),
            )

    def idempotent_message_result(self, message_id: str | None) -> AgentRuntimeResult | None:
        if not message_id:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_message_results WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if row is None:
                return None
            return _runtime_result_from_payload(_loads(row["payload"]))

    def remember_message_result(self, message_id: str | None, result: AgentRuntimeResult) -> None:
        if not message_id:
            return
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runtime_message_results(message_id, conversation_id, trace_id, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO NOTHING
                """,
                (message_id, result.conversation_id, result.trace_id, _dumps(result.to_dict()), _now_iso()),
            )

    def search_current_games(self, requirement: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for game in self.active_games():
            if game.remaining_seats() <= 0:
                continue
            score, reasons = score_requirement(requirement, game.requirement)
            if requirement and score <= 0:
                continue
            scored.append({"game": game.to_dict(), "score": score, "reasons": reasons or ["active_open_game"]})
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
            if self.active_game_for_customer(customer.customer_id):
                continue
            score, reasons = score_customer(requirement, customer)
            if score <= 0:
                continue
            scored.append({"customer": customer.to_dict(), "score": score, "reasons": reasons})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: int(limit)]

    def active_game_for_customer(self, customer_id: str) -> Game | None:
        for game in self.games.values():
            if game.status.value not in {GameStatus.FORMING.value, GameStatus.INVITING.value, GameStatus.READY.value}:
                continue
            if any(item.customer_id == customer_id and item.status in {"joined", "confirmed"} for item in game.participants):
                return game
        return None

    def create_game(
        self,
        *,
        conversation_id: str,
        organizer_id: str,
        organizer_name: str,
        requirement: dict[str, Any],
        known_players: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[Game, StateTransition]:
        with self._lock, self._connection:
            from .models import GameParticipant, new_id

            game = Game(
                game_id=new_id("game"),
                conversation_id=conversation_id,
                organizer_id=organizer_id,
                organizer_name=organizer_name,
                requirement=dict(requirement),
                participants=[
                    GameParticipant(
                        customer_id=str(item.get("customer_id") or ""),
                        display_name=str(item.get("display_name") or item.get("customer_id") or ""),
                        status=str(item.get("status") or "joined"),
                        source=str(item.get("source") or "organizer"),
                    )
                    for item in known_players
                    if isinstance(item, dict)
                ],
            )
            transition = StateTransition("game", game.game_id, None, game.status.value, "create_game", trace_id)
            self._save_game(game)
            self._append_transition(transition)
            return game, transition

    def create_invite_drafts(
        self,
        *,
        game_id: str,
        invitations: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[list[InviteDraft], list[StateTransition]]:
        with self._lock, self._connection:
            from .models import new_id, now

            game = self.require_game(game_id)
            transitions: list[StateTransition] = []
            if game.status == GameStatus.FORMING:
                old = game.status.value
                game.status = GameStatus.INVITING
                game.updated_at = now()
                transitions.append(StateTransition("game", game.game_id, old, game.status.value, "create_invite_drafts", trace_id))
                self._save_game(game)
            drafts: list[InviteDraft] = []
            for raw in invitations:
                if not isinstance(raw, dict):
                    continue
                draft = InviteDraft(
                    draft_id=new_id("draft"),
                    game_id=game_id,
                    customer_id=str(raw.get("customer_id") or ""),
                    display_name=str(raw.get("display_name") or raw.get("customer_id") or ""),
                    message_text=str(raw.get("message_text") or ""),
                    metadata=dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {},
                )
                drafts.append(draft)
                transitions.append(StateTransition("invite_draft", draft.draft_id, None, draft.status.value, "create_invite_drafts", trace_id))
                self._save_invite(draft)
            for transition in transitions:
                self._append_transition(transition)
            return drafts, transitions

    def create_outbound_message_drafts(
        self,
        *,
        conversation_id: str,
        drafts: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[list[OutboundMessageDraft], list[StateTransition]]:
        with self._lock, self._connection:
            from .models import new_id

            created: list[OutboundMessageDraft] = []
            transitions: list[StateTransition] = []
            for raw in drafts:
                if not isinstance(raw, dict):
                    continue
                draft = OutboundMessageDraft(
                    draft_id=new_id("outbound"),
                    conversation_id=conversation_id,
                    recipient_id=str(raw.get("recipient_id") or ""),
                    recipient_name=str(raw.get("recipient_name") or raw.get("recipient_id") or ""),
                    channel=str(raw.get("channel") or ""),
                    message_text=str(raw.get("message_text") or ""),
                    purpose=str(raw.get("purpose") or ""),
                    metadata=dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {},
                )
                created.append(draft)
                transitions.append(
                    StateTransition(
                        "outbound_message_draft",
                        draft.draft_id,
                        None,
                        draft.status.value,
                        "create_outbound_message_drafts",
                        trace_id,
                    )
                )
                self._save_outbound_message_draft(draft)
            for transition in transitions:
                self._append_transition(transition)
            return created, transitions

    def record_candidate_reply(
        self,
        *,
        game_id: str,
        customer_id: str,
        display_name: str,
        status: str,
        trace_id: str,
    ) -> tuple[Game, list[StateTransition]]:
        with self._lock, self._connection:
            game = self.require_game(game_id)
            transitions: list[StateTransition] = []
            normalized_status = status.strip()
            for draft in self.invite_drafts.values():
                if draft.game_id == game_id and draft.customer_id == customer_id:
                    old = draft.status.value
                    draft.status = invite_status_from_candidate_status(normalized_status)
                    draft.updated_at = datetime.now(DEFAULT_TZ)
                    transitions.append(StateTransition("invite_draft", draft.draft_id, old, draft.status.value, "record_candidate_reply", trace_id))
                    self._save_invite(draft)
            if normalized_status in {"accepted", "confirmed", "arrived"} and not any(
                item.customer_id == customer_id and item.status in {"joined", "confirmed"} for item in game.participants
            ):
                game.participants.append(
                    GameParticipant(
                        customer_id=customer_id,
                        display_name=display_name or customer_id,
                        status="confirmed",
                        source="candidate_reply",
                    )
                )
                transitions.append(
                    StateTransition(
                        "game_participant",
                        f"{game.game_id}:{customer_id}",
                        None,
                        "confirmed",
                        "record_candidate_reply",
                        trace_id,
                    )
                )
            if game.remaining_seats() == 0 and game.status != GameStatus.READY:
                old = game.status.value
                game.status = GameStatus.READY
                transitions.append(StateTransition("game", game.game_id, old, game.status.value, "seats_full", trace_id))
            game.updated_at = datetime.now(DEFAULT_TZ)
            self._save_game(game)
            for transition in transitions:
                self._append_transition(transition)
            return game, transitions

    def update_game_status(self, *, game_id: str, status: str, reason: str, trace_id: str) -> tuple[Game, StateTransition]:
        with self._lock, self._connection:
            game = self.require_game(game_id)
            target = GameStatus(status)
            old = game.status.value
            allowed = ALLOWED_GAME_TRANSITIONS.get(old, set())
            if target.value != old and target.value not in allowed:
                raise ValueError(f"illegal game status transition: {old}->{target.value}")
            game.status = target
            game.updated_at = datetime.now(DEFAULT_TZ)
            transition = StateTransition("game", game.game_id, old, target.value, reason or "update_game_status", trace_id)
            self._save_game(game)
            self._append_transition(transition)
            return game, transition

    def record_badcase(self, payload: dict[str, Any], *, trace_id: str, conversation_id: str) -> dict[str, Any]:
        with self._lock, self._connection:
            from .models import new_id

            record = {"badcase_id": new_id("badcase"), "trace_id": trace_id, "conversation_id": conversation_id, **dict(payload)}
            self._connection.execute(
                """
                INSERT INTO runtime_badcases(badcase_id, trace_id, conversation_id, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (record["badcase_id"], trace_id, conversation_id, _dumps(record), _now_iso()),
            )
            return record

    def require_game(self, game_id: str) -> Game:
        with self._lock:
            row = self._connection.execute("SELECT payload FROM runtime_games WHERE game_id = ?", (game_id,)).fetchone()
            if row is None:
                raise ValueError(f"game not found: {game_id}")
            return _game_from_payload(_loads(row["payload"]))

    def _save_game(self, game: Game) -> None:
        self._connection.execute(
            """
            INSERT INTO runtime_games(game_id, conversation_id, status, payload, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(game_id) DO UPDATE SET
                conversation_id=excluded.conversation_id,
                status=excluded.status,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (game.game_id, game.conversation_id, game.status.value, _dumps(game.to_dict()), game.updated_at.isoformat()),
        )

    def _save_invite(self, draft: InviteDraft) -> None:
        self._connection.execute(
            """
            INSERT INTO runtime_invite_drafts(draft_id, game_id, customer_id, status, payload, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(draft_id) DO UPDATE SET
                game_id=excluded.game_id,
                customer_id=excluded.customer_id,
                status=excluded.status,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (draft.draft_id, draft.game_id, draft.customer_id, draft.status.value, _dumps(draft.to_dict()), draft.updated_at.isoformat()),
        )

    def _save_outbound_message_draft(self, draft: OutboundMessageDraft) -> None:
        self._connection.execute(
            """
            INSERT INTO runtime_outbound_message_drafts(draft_id, conversation_id, recipient_id, channel, status, payload, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(draft_id) DO UPDATE SET
                conversation_id=excluded.conversation_id,
                recipient_id=excluded.recipient_id,
                channel=excluded.channel,
                status=excluded.status,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (
                draft.draft_id,
                draft.conversation_id,
                draft.recipient_id,
                draft.channel,
                draft.status.value,
                _dumps(draft.to_dict()),
                draft.updated_at.isoformat(),
            ),
        )

    def _append_transition(self, transition: StateTransition) -> None:
        self._connection.execute(
            """
            INSERT INTO runtime_state_transitions(trace_id, entity_type, entity_id, occurred_at, payload)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                transition.trace_id,
                transition.entity_type,
                transition.entity_id,
                transition.occurred_at.isoformat(),
                _dumps(transition.to_dict()),
            ),
        )

    def _migrate(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runtime_customers(
                customer_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_games(
                game_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_invite_drafts(
                draft_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_outbound_message_drafts(
                draft_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                recipient_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_state_transitions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_conversation_turns(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                role TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_conversation_checkpoints(
                conversation_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_idempotency_ledger(
                idempotency_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_message_results(
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_badcases(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                badcase_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_runtime_turns_conversation_id ON runtime_conversation_turns(conversation_id, id);
            CREATE INDEX IF NOT EXISTS idx_runtime_games_status ON runtime_games(status);
            CREATE INDEX IF NOT EXISTS idx_runtime_invites_game_id ON runtime_invite_drafts(game_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_outbound_conversation_id ON runtime_outbound_message_drafts(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_checkpoints_updated_at ON runtime_conversation_checkpoints(updated_at);
            """
        )
        self._connection.commit()


def _customer_from_payload(payload: dict[str, Any]) -> CustomerProfile:
    return CustomerProfile(
        customer_id=str(payload.get("customer_id") or ""),
        display_name=str(payload.get("display_name") or ""),
        gender=payload.get("gender"),
        preferred_games=[str(item) for item in payload.get("preferred_games") or []],
        preferred_stakes=[str(item) for item in payload.get("preferred_stakes") or []],
        preferred_time_tags=[str(item) for item in payload.get("preferred_time_tags") or []],
        smoke_preference=payload.get("smoke_preference"),
        response_score=float(payload.get("response_score") or 0.5),
        fatigue_score=float(payload.get("fatigue_score") or 0.0),
        no_contact=bool(payload.get("no_contact")),
        notes=str(payload.get("notes") or ""),
    )


def _turn_from_payload(payload: dict[str, Any]) -> ConversationTurn:
    return ConversationTurn(
        role=ConversationRole(str(payload.get("role") or ConversationRole.USER.value)),
        content=str(payload.get("content") or ""),
        trace_id=str(payload.get("trace_id") or ""),
        sender_id=payload.get("sender_id"),
        sender_name=payload.get("sender_name"),
        metadata=dict(payload.get("metadata") or {}),
        occurred_at=_datetime_from_payload(payload.get("occurred_at")),
    )


def _checkpoint_from_payload(payload: dict[str, Any]) -> ConversationCheckpoint:
    return ConversationCheckpoint(
        conversation_id=str(payload.get("conversation_id") or ""),
        summary=str(payload.get("summary") or ""),
        facts=dict(payload.get("facts") or {}) if isinstance(payload.get("facts"), dict) else {},
        open_questions=[str(item) for item in payload.get("open_questions") or []],
        source_trace_id=payload.get("source_trace_id"),
        updated_at=_datetime_from_payload(payload.get("updated_at")),
    )


def _game_from_payload(payload: dict[str, Any]) -> Game:
    return Game(
        game_id=str(payload.get("game_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        organizer_id=str(payload.get("organizer_id") or ""),
        organizer_name=str(payload.get("organizer_name") or ""),
        requirement=dict(payload.get("requirement") or {}),
        status=GameStatus(str(payload.get("status") or GameStatus.FORMING.value)),
        participants=[
            GameParticipant(
                customer_id=str(item.get("customer_id") or ""),
                display_name=str(item.get("display_name") or ""),
                status=str(item.get("status") or "joined"),
                source=str(item.get("source") or "organizer"),
            )
            for item in payload.get("participants") or []
            if isinstance(item, dict)
        ],
        seats_total=int(payload.get("seats_total") or 4),
        created_at=_datetime_from_payload(payload.get("created_at")),
        updated_at=_datetime_from_payload(payload.get("updated_at")),
    )


def _invite_from_payload(payload: dict[str, Any]) -> InviteDraft:
    return InviteDraft(
        draft_id=str(payload.get("draft_id") or ""),
        game_id=str(payload.get("game_id") or ""),
        customer_id=str(payload.get("customer_id") or ""),
        display_name=str(payload.get("display_name") or ""),
        message_text=str(payload.get("message_text") or ""),
        status=InviteStatus(str(payload.get("status") or InviteStatus.PENDING_APPROVAL.value)),
        metadata=dict(payload.get("metadata") or {}),
        created_at=_datetime_from_payload(payload.get("created_at")),
        updated_at=_datetime_from_payload(payload.get("updated_at")),
    )


def _outbound_message_draft_from_payload(payload: dict[str, Any]) -> OutboundMessageDraft:
    return OutboundMessageDraft(
        draft_id=str(payload.get("draft_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        recipient_id=str(payload.get("recipient_id") or ""),
        recipient_name=str(payload.get("recipient_name") or ""),
        channel=str(payload.get("channel") or ""),
        message_text=str(payload.get("message_text") or ""),
        purpose=str(payload.get("purpose") or ""),
        status=OutboundDraftStatus(str(payload.get("status") or OutboundDraftStatus.PENDING_APPROVAL.value)),
        metadata=dict(payload.get("metadata") or {}),
        created_at=_datetime_from_payload(payload.get("created_at")),
        updated_at=_datetime_from_payload(payload.get("updated_at")),
    )


def _transition_from_payload(payload: dict[str, Any]) -> StateTransition:
    return StateTransition(
        entity_type=str(payload.get("entity_type") or ""),
        entity_id=str(payload.get("entity_id") or ""),
        from_status=payload.get("from_status"),
        to_status=str(payload.get("to_status") or ""),
        reason=str(payload.get("reason") or ""),
        trace_id=str(payload.get("trace_id") or ""),
        occurred_at=_datetime_from_payload(payload.get("occurred_at")),
    )


def _tool_call_from_payload(payload: dict[str, Any]) -> ToolCall:
    return ToolCall(
        name=str(payload.get("name") or ""),
        arguments=dict(payload.get("arguments") or {}),
        reason=str(payload.get("reason") or ""),
        idempotency_key=payload.get("idempotency_key"),
    )


def _action_from_payload(payload: dict[str, Any]) -> AgentAction:
    return AgentAction(
        goal=str(payload.get("goal") or ""),
        objective_status=str(payload.get("objective_status") or "unknown"),
        reasoning_summary=str(payload.get("reasoning_summary") or ""),
        reply_to_user=str(payload.get("reply_to_user") or ""),
        tool_calls=[
            _tool_call_from_payload(item)
            for item in payload.get("tool_calls") or []
            if isinstance(item, dict)
        ],
        needs_human=bool(payload.get("needs_human")),
        stop_reason=dict(payload.get("stop_reason") or {}) if isinstance(payload.get("stop_reason"), dict) else {},
        badcase=payload.get("badcase") if isinstance(payload.get("badcase"), dict) else None,
    )


def _tool_result_from_payload(payload: dict[str, Any]) -> ToolResult:
    return ToolResult(
        name=str(payload.get("name") or ""),
        called=bool(payload.get("called")),
        allowed=bool(payload.get("allowed")),
        result=dict(payload.get("result") or {}),
        error=payload.get("error"),
        idempotency_key=payload.get("idempotency_key"),
        deduplicated=bool(payload.get("deduplicated")),
        state_transitions=[
            _transition_from_payload(item)
            for item in payload.get("state_transitions") or []
            if isinstance(item, dict)
        ],
    )


def _runtime_result_from_payload(payload: dict[str, Any]) -> AgentRuntimeResult:
    return AgentRuntimeResult(
        trace_id=str(payload.get("trace_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        final_reply=str(payload.get("final_reply") or ""),
        actions=[
            _action_from_payload(item)
            for item in payload.get("actions") or []
            if isinstance(item, dict)
        ],
        tool_results=[
            _tool_result_from_payload(item)
            for item in payload.get("tool_results") or []
            if isinstance(item, dict)
        ],
        state_transitions=[
            _transition_from_payload(item)
            for item in payload.get("state_transitions") or []
            if isinstance(item, dict)
        ],
    )


def _datetime_from_payload(value: Any) -> datetime:
    if value:
        return datetime.fromisoformat(str(value))
    return datetime.now(DEFAULT_TZ)


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _loads(payload: str) -> dict[str, Any]:
    raw = json.loads(payload)
    return raw if isinstance(raw, dict) else {}


def _now_iso() -> str:
    return datetime.now(DEFAULT_TZ).isoformat()
