from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import pytest

from mahjong_agent_runtime import SQLiteAgentStore, ToolCall, ToolGateway


def test_create_and_join_use_normalized_participant_table(tmp_path) -> None:
    db_path = tmp_path / "normalized_participants.sqlite3"
    store = SQLiteAgentStore(db_path)
    game, _ = store.create_game(
        conversation_id="conversation_owner",
        organizer_id="owner",
        organizer_name="发起人",
        requirement={"game_type": "hangzhou_mahjong", "stake": "0.5"},
        known_players=[{"customer_id": "owner", "display_name": "发起人", "seat_count": 2}],
        trace_id="trace_create",
    )

    with store._lock:
        payload_row = store._connection.execute(
            "SELECT payload FROM runtime_games WHERE game_id = ?",
            (game.game_id,),
        ).fetchone()
        participant_rows = store._connection.execute(
            """
            SELECT game_id, customer_id, status, seat_count, joined_at
            FROM runtime_game_participants
            WHERE game_id = ?
            """,
            (game.game_id,),
        ).fetchall()

    payload = json.loads(payload_row["payload"])
    assert "participants" not in payload
    assert "parties" not in payload
    assert "seat_claims" not in payload
    assert [(row["customer_id"], row["status"], row["seat_count"]) for row in participant_rows] == [
        ("owner", "joined", 2)
    ]
    assert datetime.fromisoformat(participant_rows[0]["joined_at"])

    gateway = ToolGateway(store)
    result = gateway.execute(
        ToolCall(
            name="join_game",
            arguments={
                "game_id": game.game_id,
                "customer_id": "candidate",
                "display_name": "候选人",
                "seat_count": 1,
            },
            reason="候选人明确确认加入",
        ),
        trace_id="trace_join",
        conversation_id="conversation_owner",
        sender_id="candidate",
        sender_name="候选人",
        step_index=0,
        source_message_id="message_join",
    )

    assert result.called is True
    assert result.allowed is True
    joined_at = store._connection.execute(
        """
        SELECT joined_at FROM runtime_game_participants
        WHERE game_id = ? AND customer_id = ?
        """,
        (game.game_id, "candidate"),
    ).fetchone()["joined_at"]
    store.record_candidate_reply(
        game_id=game.game_id,
        customer_id="candidate",
        display_name="候选人",
        status="negotiating",
        seat_count=1,
        trace_id="trace_status_update",
    )
    updated = store._connection.execute(
        """
        SELECT status, joined_at FROM runtime_game_participants
        WHERE game_id = ? AND customer_id = ?
        """,
        (game.game_id, "candidate"),
    ).fetchone()
    assert updated["status"] == "negotiating"
    assert updated["joined_at"] == joined_at

    reopened = SQLiteAgentStore(db_path)
    persisted = reopened.require_game(game.game_id)
    assert [(item.customer_id, item.status, item.seat_count) for item in persisted.participants] == [
        ("owner", "joined", 2),
        ("candidate", "negotiating", 1),
    ]


def test_fresh_sqlite_store_marks_current_schema_version(tmp_path) -> None:
    db_path = tmp_path / "current_schema.sqlite3"

    store = SQLiteAgentStore(db_path)
    version = int(store._connection.execute("PRAGMA user_version").fetchone()[0])

    assert version == 1


def test_old_embedded_participant_schema_is_rejected_without_mutation(tmp_path) -> None:
    db_path = tmp_path / "legacy_participants.sqlite3"
    created_at = "2026-07-20T10:00:00+08:00"
    legacy_payload = {
        "game_id": "game_legacy",
        "conversation_id": "conversation_legacy",
        "organizer_id": "owner",
        "organizer_name": "发起人",
        "requirement": {"game_type": "hangzhou_mahjong", "stake": "1"},
        "status": "forming",
        "participants": [
            {
                "customer_id": "owner",
                "display_name": "发起人",
                "status": "joined",
                "source": "requester",
                "seat_count": 1,
                "party_id": "party_owner",
                "known_member_ids": ["owner"],
                "anonymous_seat_count": 0,
            },
            {
                "customer_id": "friend",
                "display_name": "朋友",
                "status": "confirmed",
                "source": "candidate_reply",
                "seat_count": 2,
                "party_id": "party_friend",
                "known_member_ids": ["friend"],
                "anonymous_seat_count": 1,
            },
        ],
        "parties": [],
        "seats_total": 4,
        "created_at": created_at,
        "updated_at": created_at,
    }
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE runtime_games(
            game_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            status TEXT NOT NULL,
            payload TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO runtime_games(game_id, conversation_id, status, payload, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("game_legacy", "conversation_legacy", "forming", json.dumps(legacy_payload), created_at),
    )
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="automatic legacy migration is intentionally disabled"):
        SQLiteAgentStore(db_path)

    # 拒绝旧版结构时，绝不能改写原始载荷。
    connection = sqlite3.connect(db_path)
    stored_payload = json.loads(
        connection.execute(
            "SELECT payload FROM runtime_games WHERE game_id = 'game_legacy'"
        ).fetchone()[0]
    )
    connection.close()
    assert stored_payload == legacy_payload
