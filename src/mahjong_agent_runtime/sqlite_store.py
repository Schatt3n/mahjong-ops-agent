from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    AgentActionV3,
    AgentRuntimeResultV3,
    ConversationCheckpointV3,
    ConversationRoleV3,
    ConversationTurnV3,
    CustomerProfileV3,
    DEFAULT_TZ_V3,
    GameParticipantV3,
    GameStatusV3,
    GameV3,
    InviteDraftV3,
    InviteStatusV3,
    OutboundDraftStatusV3,
    OutboundMessageDraftV3,
    StateTransitionV3,
    ToolCallV3,
    ToolResultV3,
)
from .store import (
    ALLOWED_GAME_TRANSITIONS,
    invite_status_from_candidate_status,
    score_customer,
    score_requirement,
)


@dataclass(slots=True)
class SQLiteAgentStoreV3:
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
    def customers(self) -> dict[str, CustomerProfileV3]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM v3_customers").fetchall()
            return {item.customer_id: item for item in (_customer_from_payload(_loads(row["payload"])) for row in rows)}

    @property
    def games(self) -> dict[str, GameV3]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM v3_games").fetchall()
            return {item.game_id: item for item in (_game_from_payload(_loads(row["payload"])) for row in rows)}

    @property
    def invite_drafts(self) -> dict[str, InviteDraftV3]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM v3_invite_drafts").fetchall()
            return {item.draft_id: item for item in (_invite_from_payload(_loads(row["payload"])) for row in rows)}

    @property
    def outbound_message_drafts(self) -> dict[str, OutboundMessageDraftV3]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM v3_outbound_message_drafts").fetchall()
            return {item.draft_id: item for item in (_outbound_message_draft_from_payload(_loads(row["payload"])) for row in rows)}

    @property
    def conversation_checkpoints(self) -> dict[str, ConversationCheckpointV3]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM v3_conversation_checkpoints").fetchall()
            return {
                item.conversation_id: item
                for item in (_checkpoint_from_payload(_loads(row["payload"])) for row in rows)
            }

    @property
    def transitions(self) -> list[StateTransitionV3]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM v3_state_transitions ORDER BY id").fetchall()
            return [_transition_from_payload(_loads(row["payload"])) for row in rows]

    @property
    def badcases(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM v3_badcases ORDER BY id").fetchall()
            return [_loads(row["payload"]) for row in rows]

    def upsert_customer(self, profile: CustomerProfileV3) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO v3_customers(customer_id, payload, updated_at)
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
            ConversationTurnV3(
                role=ConversationRoleV3.USER,
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
            ConversationTurnV3(role=ConversationRoleV3.ASSISTANT, content=text, trace_id=trace_id),
        )

    def append_tool_turn(self, conversation_id: str, text: str, trace_id: str) -> None:
        self.append_turn(
            conversation_id,
            ConversationTurnV3(role=ConversationRoleV3.TOOL, content=text, trace_id=trace_id),
        )

    def append_turn(self, conversation_id: str, turn: ConversationTurnV3) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO v3_conversation_turns(conversation_id, trace_id, role, occurred_at, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, turn.trace_id, turn.role.value, turn.occurred_at.isoformat(), _dumps(turn.to_dict())),
            )

    def recent_turns(self, conversation_id: str, limit: int = 30) -> list[ConversationTurnV3]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload
                FROM v3_conversation_turns
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, int(limit)),
            ).fetchall()
            turns = [_turn_from_payload(_loads(row["payload"])) for row in rows]
            return list(reversed(turns))

    def get_conversation_checkpoint(self, conversation_id: str) -> ConversationCheckpointV3 | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM v3_conversation_checkpoints WHERE conversation_id = ?",
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
    ) -> tuple[ConversationCheckpointV3, StateTransitionV3]:
        with self._lock, self._connection:
            previous = self.get_conversation_checkpoint(conversation_id)
            checkpoint = ConversationCheckpointV3(
                conversation_id=conversation_id,
                summary=summary,
                facts=dict(facts),
                open_questions=list(open_questions),
                source_trace_id=trace_id,
            )
            transition = StateTransitionV3(
                "conversation_checkpoint",
                conversation_id,
                "exists" if previous else None,
                "updated",
                "update_context_checkpoint",
                trace_id,
            )
            self._connection.execute(
                """
                INSERT INTO v3_conversation_checkpoints(conversation_id, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (conversation_id, _dumps(checkpoint.to_dict()), checkpoint.updated_at.isoformat()),
            )
            self._append_transition(transition)
            return checkpoint, transition

    def active_games(self, conversation_id: str | None = None) -> list[GameV3]:
        games = [
            item
            for item in self.games.values()
            if item.status.value in {GameStatusV3.FORMING.value, GameStatusV3.INVITING.value, GameStatusV3.READY.value}
        ]
        if conversation_id:
            scoped = [item for item in games if item.conversation_id == conversation_id]
            return scoped or games
        return games

    def idempotent_result(self, key: str | None) -> ToolResultV3 | None:
        if not key:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM v3_idempotency_ledger WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            return _tool_result_from_payload(_loads(row["payload"]))

    def claim_idempotent_result(self, key: str | None, claimed_result: ToolResultV3) -> tuple[bool, ToolResultV3 | None]:
        if not key:
            return True, None
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO v3_idempotency_ledger(idempotency_key, payload, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(idempotency_key) DO NOTHING
                """,
                (key, _dumps(claimed_result.to_dict()), _now_iso()),
            )
            if cursor.rowcount == 1:
                return True, None
            row = self._connection.execute(
                "SELECT payload FROM v3_idempotency_ledger WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return False, None
            return False, _tool_result_from_payload(_loads(row["payload"]))

    def remember_result(self, key: str | None, result: ToolResultV3) -> None:
        if not key:
            return
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO v3_idempotency_ledger(idempotency_key, payload, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(idempotency_key) DO UPDATE SET
                    payload=excluded.payload
                """,
                (key, _dumps(result.to_dict()), _now_iso()),
            )

    def idempotent_message_result(self, message_id: str | None) -> AgentRuntimeResultV3 | None:
        if not message_id:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM v3_message_results WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if row is None:
                return None
            return _runtime_result_from_payload(_loads(row["payload"]))

    def remember_message_result(self, message_id: str | None, result: AgentRuntimeResultV3) -> None:
        if not message_id:
            return
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO v3_message_results(message_id, conversation_id, trace_id, payload, created_at)
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

    def active_game_for_customer(self, customer_id: str) -> GameV3 | None:
        for game in self.games.values():
            if game.status.value not in {GameStatusV3.FORMING.value, GameStatusV3.INVITING.value, GameStatusV3.READY.value}:
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
    ) -> tuple[GameV3, StateTransitionV3]:
        with self._lock, self._connection:
            from .models import GameParticipantV3, new_id

            game = GameV3(
                game_id=new_id("game"),
                conversation_id=conversation_id,
                organizer_id=organizer_id,
                organizer_name=organizer_name,
                requirement=dict(requirement),
                participants=[
                    GameParticipantV3(
                        customer_id=str(item.get("customer_id") or ""),
                        display_name=str(item.get("display_name") or item.get("customer_id") or ""),
                        status=str(item.get("status") or "joined"),
                        source=str(item.get("source") or "organizer"),
                    )
                    for item in known_players
                    if isinstance(item, dict)
                ],
            )
            transition = StateTransitionV3("game", game.game_id, None, game.status.value, "create_game", trace_id)
            self._save_game(game)
            self._append_transition(transition)
            return game, transition

    def create_invite_drafts(
        self,
        *,
        game_id: str,
        invitations: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[list[InviteDraftV3], list[StateTransitionV3]]:
        with self._lock, self._connection:
            from .models import new_id, now_v3

            game = self.require_game(game_id)
            transitions: list[StateTransitionV3] = []
            if game.status == GameStatusV3.FORMING:
                old = game.status.value
                game.status = GameStatusV3.INVITING
                game.updated_at = now_v3()
                transitions.append(StateTransitionV3("game", game.game_id, old, game.status.value, "create_invite_drafts", trace_id))
                self._save_game(game)
            drafts: list[InviteDraftV3] = []
            for raw in invitations:
                if not isinstance(raw, dict):
                    continue
                draft = InviteDraftV3(
                    draft_id=new_id("draft"),
                    game_id=game_id,
                    customer_id=str(raw.get("customer_id") or ""),
                    display_name=str(raw.get("display_name") or raw.get("customer_id") or ""),
                    message_text=str(raw.get("message_text") or ""),
                    metadata=dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {},
                )
                drafts.append(draft)
                transitions.append(StateTransitionV3("invite_draft", draft.draft_id, None, draft.status.value, "create_invite_drafts", trace_id))
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
    ) -> tuple[list[OutboundMessageDraftV3], list[StateTransitionV3]]:
        with self._lock, self._connection:
            from .models import new_id

            created: list[OutboundMessageDraftV3] = []
            transitions: list[StateTransitionV3] = []
            for raw in drafts:
                if not isinstance(raw, dict):
                    continue
                draft = OutboundMessageDraftV3(
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
                    StateTransitionV3(
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
    ) -> tuple[GameV3, list[StateTransitionV3]]:
        with self._lock, self._connection:
            game = self.require_game(game_id)
            transitions: list[StateTransitionV3] = []
            normalized_status = status.strip()
            for draft in self.invite_drafts.values():
                if draft.game_id == game_id and draft.customer_id == customer_id:
                    old = draft.status.value
                    draft.status = invite_status_from_candidate_status(normalized_status)
                    draft.updated_at = datetime.now(DEFAULT_TZ_V3)
                    transitions.append(StateTransitionV3("invite_draft", draft.draft_id, old, draft.status.value, "record_candidate_reply", trace_id))
                    self._save_invite(draft)
            if normalized_status in {"accepted", "confirmed", "arrived"} and not any(
                item.customer_id == customer_id and item.status in {"joined", "confirmed"} for item in game.participants
            ):
                game.participants.append(
                    GameParticipantV3(
                        customer_id=customer_id,
                        display_name=display_name or customer_id,
                        status="confirmed",
                        source="candidate_reply",
                    )
                )
                transitions.append(
                    StateTransitionV3(
                        "game_participant",
                        f"{game.game_id}:{customer_id}",
                        None,
                        "confirmed",
                        "record_candidate_reply",
                        trace_id,
                    )
                )
            if game.remaining_seats() == 0 and game.status != GameStatusV3.READY:
                old = game.status.value
                game.status = GameStatusV3.READY
                transitions.append(StateTransitionV3("game", game.game_id, old, game.status.value, "seats_full", trace_id))
            game.updated_at = datetime.now(DEFAULT_TZ_V3)
            self._save_game(game)
            for transition in transitions:
                self._append_transition(transition)
            return game, transitions

    def update_game_status(self, *, game_id: str, status: str, reason: str, trace_id: str) -> tuple[GameV3, StateTransitionV3]:
        with self._lock, self._connection:
            game = self.require_game(game_id)
            target = GameStatusV3(status)
            old = game.status.value
            allowed = ALLOWED_GAME_TRANSITIONS.get(old, set())
            if target.value != old and target.value not in allowed:
                raise ValueError(f"illegal game status transition: {old}->{target.value}")
            game.status = target
            game.updated_at = datetime.now(DEFAULT_TZ_V3)
            transition = StateTransitionV3("game", game.game_id, old, target.value, reason or "update_game_status", trace_id)
            self._save_game(game)
            self._append_transition(transition)
            return game, transition

    def record_badcase(self, payload: dict[str, Any], *, trace_id: str, conversation_id: str) -> dict[str, Any]:
        with self._lock, self._connection:
            from .models import new_id

            record = {"badcase_id": new_id("badcase"), "trace_id": trace_id, "conversation_id": conversation_id, **dict(payload)}
            self._connection.execute(
                """
                INSERT INTO v3_badcases(badcase_id, trace_id, conversation_id, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (record["badcase_id"], trace_id, conversation_id, _dumps(record), _now_iso()),
            )
            return record

    def require_game(self, game_id: str) -> GameV3:
        with self._lock:
            row = self._connection.execute("SELECT payload FROM v3_games WHERE game_id = ?", (game_id,)).fetchone()
            if row is None:
                raise ValueError(f"game not found: {game_id}")
            return _game_from_payload(_loads(row["payload"]))

    def _save_game(self, game: GameV3) -> None:
        self._connection.execute(
            """
            INSERT INTO v3_games(game_id, conversation_id, status, payload, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(game_id) DO UPDATE SET
                conversation_id=excluded.conversation_id,
                status=excluded.status,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (game.game_id, game.conversation_id, game.status.value, _dumps(game.to_dict()), game.updated_at.isoformat()),
        )

    def _save_invite(self, draft: InviteDraftV3) -> None:
        self._connection.execute(
            """
            INSERT INTO v3_invite_drafts(draft_id, game_id, customer_id, status, payload, updated_at)
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

    def _save_outbound_message_draft(self, draft: OutboundMessageDraftV3) -> None:
        self._connection.execute(
            """
            INSERT INTO v3_outbound_message_drafts(draft_id, conversation_id, recipient_id, channel, status, payload, updated_at)
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

    def _append_transition(self, transition: StateTransitionV3) -> None:
        self._connection.execute(
            """
            INSERT INTO v3_state_transitions(trace_id, entity_type, entity_id, occurred_at, payload)
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
            CREATE TABLE IF NOT EXISTS v3_customers(
                customer_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS v3_games(
                game_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS v3_invite_drafts(
                draft_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS v3_outbound_message_drafts(
                draft_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                recipient_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS v3_state_transitions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS v3_conversation_turns(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                role TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS v3_conversation_checkpoints(
                conversation_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS v3_idempotency_ledger(
                idempotency_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS v3_message_results(
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS v3_badcases(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                badcase_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_v3_turns_conversation_id ON v3_conversation_turns(conversation_id, id);
            CREATE INDEX IF NOT EXISTS idx_v3_games_status ON v3_games(status);
            CREATE INDEX IF NOT EXISTS idx_v3_invites_game_id ON v3_invite_drafts(game_id);
            CREATE INDEX IF NOT EXISTS idx_v3_outbound_conversation_id ON v3_outbound_message_drafts(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_v3_checkpoints_updated_at ON v3_conversation_checkpoints(updated_at);
            """
        )
        self._connection.commit()


def _customer_from_payload(payload: dict[str, Any]) -> CustomerProfileV3:
    return CustomerProfileV3(
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


def _turn_from_payload(payload: dict[str, Any]) -> ConversationTurnV3:
    return ConversationTurnV3(
        role=ConversationRoleV3(str(payload.get("role") or ConversationRoleV3.USER.value)),
        content=str(payload.get("content") or ""),
        trace_id=str(payload.get("trace_id") or ""),
        sender_id=payload.get("sender_id"),
        sender_name=payload.get("sender_name"),
        metadata=dict(payload.get("metadata") or {}),
        occurred_at=_datetime_from_payload(payload.get("occurred_at")),
    )


def _checkpoint_from_payload(payload: dict[str, Any]) -> ConversationCheckpointV3:
    return ConversationCheckpointV3(
        conversation_id=str(payload.get("conversation_id") or ""),
        summary=str(payload.get("summary") or ""),
        facts=dict(payload.get("facts") or {}) if isinstance(payload.get("facts"), dict) else {},
        open_questions=[str(item) for item in payload.get("open_questions") or []],
        source_trace_id=payload.get("source_trace_id"),
        updated_at=_datetime_from_payload(payload.get("updated_at")),
    )


def _game_from_payload(payload: dict[str, Any]) -> GameV3:
    return GameV3(
        game_id=str(payload.get("game_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        organizer_id=str(payload.get("organizer_id") or ""),
        organizer_name=str(payload.get("organizer_name") or ""),
        requirement=dict(payload.get("requirement") or {}),
        status=GameStatusV3(str(payload.get("status") or GameStatusV3.FORMING.value)),
        participants=[
            GameParticipantV3(
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


def _invite_from_payload(payload: dict[str, Any]) -> InviteDraftV3:
    return InviteDraftV3(
        draft_id=str(payload.get("draft_id") or ""),
        game_id=str(payload.get("game_id") or ""),
        customer_id=str(payload.get("customer_id") or ""),
        display_name=str(payload.get("display_name") or ""),
        message_text=str(payload.get("message_text") or ""),
        status=InviteStatusV3(str(payload.get("status") or InviteStatusV3.PENDING_APPROVAL.value)),
        metadata=dict(payload.get("metadata") or {}),
        created_at=_datetime_from_payload(payload.get("created_at")),
        updated_at=_datetime_from_payload(payload.get("updated_at")),
    )


def _outbound_message_draft_from_payload(payload: dict[str, Any]) -> OutboundMessageDraftV3:
    return OutboundMessageDraftV3(
        draft_id=str(payload.get("draft_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        recipient_id=str(payload.get("recipient_id") or ""),
        recipient_name=str(payload.get("recipient_name") or ""),
        channel=str(payload.get("channel") or ""),
        message_text=str(payload.get("message_text") or ""),
        purpose=str(payload.get("purpose") or ""),
        status=OutboundDraftStatusV3(str(payload.get("status") or OutboundDraftStatusV3.PENDING_APPROVAL.value)),
        metadata=dict(payload.get("metadata") or {}),
        created_at=_datetime_from_payload(payload.get("created_at")),
        updated_at=_datetime_from_payload(payload.get("updated_at")),
    )


def _transition_from_payload(payload: dict[str, Any]) -> StateTransitionV3:
    return StateTransitionV3(
        entity_type=str(payload.get("entity_type") or ""),
        entity_id=str(payload.get("entity_id") or ""),
        from_status=payload.get("from_status"),
        to_status=str(payload.get("to_status") or ""),
        reason=str(payload.get("reason") or ""),
        trace_id=str(payload.get("trace_id") or ""),
        occurred_at=_datetime_from_payload(payload.get("occurred_at")),
    )


def _tool_call_from_payload(payload: dict[str, Any]) -> ToolCallV3:
    return ToolCallV3(
        name=str(payload.get("name") or ""),
        arguments=dict(payload.get("arguments") or {}),
        reason=str(payload.get("reason") or ""),
        idempotency_key=payload.get("idempotency_key"),
    )


def _action_from_payload(payload: dict[str, Any]) -> AgentActionV3:
    return AgentActionV3(
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


def _tool_result_from_payload(payload: dict[str, Any]) -> ToolResultV3:
    return ToolResultV3(
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


def _runtime_result_from_payload(payload: dict[str, Any]) -> AgentRuntimeResultV3:
    return AgentRuntimeResultV3(
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
    return datetime.now(DEFAULT_TZ_V3)


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _loads(payload: str) -> dict[str, Any]:
    raw = json.loads(payload)
    return raw if isinstance(raw, dict) else {}


def _now_iso() -> str:
    return datetime.now(DEFAULT_TZ_V3).isoformat()
