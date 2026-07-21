"""SQLite game persistence store operations."""

from __future__ import annotations

import json
from ...models import Game
from .serialization import (
    _dumps,
    _game_storage_payload,
)

class SQLiteGamePersistenceStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

    __slots__ = ()

    def _save_game(self, game: Game) -> None:
        """Persist the game base row and normalized participants atomically.

        Callers already hold ``_write_transaction`` or the connection context.
        ``runtime_games.payload`` deliberately excludes participant/party views;
        those views are rebuilt from ``runtime_game_participants`` on reads.
        """

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
            (
                game.game_id,
                game.conversation_id,
                game.status.value,
                _dumps(_game_storage_payload(game)),
                game.updated_at.isoformat(),
            ),
        )
        self._save_game_participants(game)

    def _save_game_participants(self, game: Game) -> None:
        """Synchronize the participant rows without changing their join time."""

        customer_ids = [str(item.customer_id) for item in game.participants]
        if customer_ids:
            placeholders = ",".join("?" for _ in customer_ids)
            self._connection.execute(
                f"DELETE FROM runtime_game_participants WHERE game_id = ? AND customer_id NOT IN ({placeholders})",
                (game.game_id, *customer_ids),
            )
        else:
            self._connection.execute(
                "DELETE FROM runtime_game_participants WHERE game_id = ?",
                (game.game_id,),
            )
        self._connection.executemany(
            """
            INSERT INTO runtime_game_participants(
                game_id, customer_id, display_name, status, source, seat_count,
                party_id, known_member_ids, anonymous_seat_count, joined_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id, customer_id) DO UPDATE SET
                display_name=excluded.display_name,
                status=excluded.status,
                source=excluded.source,
                seat_count=excluded.seat_count,
                party_id=excluded.party_id,
                known_member_ids=excluded.known_member_ids,
                anonymous_seat_count=excluded.anonymous_seat_count,
                updated_at=excluded.updated_at
            """,
            [
                (
                    game.game_id,
                    participant.customer_id,
                    participant.display_name,
                    participant.status,
                    participant.source,
                    max(1, int(participant.seat_count)),
                    participant.party_id,
                    json.dumps(list(participant.known_member_ids), ensure_ascii=False, sort_keys=True),
                    max(0, int(participant.anonymous_seat_count)),
                    participant.joined_at.isoformat(),
                    game.updated_at.isoformat(),
                )
                for participant in game.participants
            ],
        )
