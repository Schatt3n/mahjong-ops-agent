from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .models import CustomerProfile, DEFAULT_TZ, PlayPreference
from .observability import to_trace_payload


class CustomerProfileRepository(Protocol):
    def load_all(self) -> list[CustomerProfile]:
        ...

    def save(self, profile: CustomerProfile) -> None:
        ...


class SQLiteCustomerProfileRepository:
    """SQLite-backed customer profile repository for local deployments."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def load_all(self) -> list[CustomerProfile]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM controlled_customer_profiles
                ORDER BY display_name, customer_id
                """
            ).fetchall()
        return [_profile_from_row(row) for row in rows]

    def save(self, profile: CustomerProfile) -> None:
        now_text = datetime.now(DEFAULT_TZ).isoformat()
        payload = _profile_payload(profile)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO controlled_customer_profiles(
                    customer_id, display_name, payload_json, updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(customer_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (profile.id, profile.display_name, _dump_json(payload), now_text),
            )

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS controlled_customer_profiles (
                    customer_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_controlled_customer_profiles_display_name
                    ON controlled_customer_profiles(display_name, customer_id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn


def _profile_payload(profile: CustomerProfile) -> dict[str, Any]:
    return {
        "id": profile.id,
        "display_name": profile.display_name,
        "aliases": list(profile.aliases),
        "preferred_levels": list(profile.preferred_levels),
        "play_preferences": [asdict(item) for item in profile.play_preferences],
        "tags": list(profile.tags),
        "smoke_free_preference": profile.smoke_free_preference,
        "usual_party_size": profile.usual_party_size,
        "usual_party_size_confidence": profile.usual_party_size_confidence,
        "usual_start_hours": list(profile.usual_start_hours),
        "usual_weekdays": list(profile.usual_weekdays),
        "no_contact": profile.no_contact,
        "last_invited_at": profile.last_invited_at.isoformat() if profile.last_invited_at else None,
        "decline_count_30d": profile.decline_count_30d,
        "max_games_per_day": profile.max_games_per_day,
        "min_hours_between_games": profile.min_hours_between_games,
        "invite_cooldown_hours": profile.invite_cooldown_hours,
        "daily_invite_limit": profile.daily_invite_limit,
        "fatigue_sensitivity": profile.fatigue_sensitivity,
        "metadata": to_trace_payload(profile.metadata),
    }


def _profile_from_row(row: sqlite3.Row) -> CustomerProfile:
    payload = _loads_dict(str(row["payload_json"] or "{}"))
    return CustomerProfile(
        id=str(payload.get("id") or row["customer_id"]),
        display_name=str(payload.get("display_name") or row["display_name"]),
        aliases=[str(item) for item in payload.get("aliases") or []],
        preferred_levels=[str(item) for item in payload.get("preferred_levels") or []],
        play_preferences=[
            _play_preference_from_payload(item)
            for item in payload.get("play_preferences") or []
            if isinstance(item, dict)
        ],
        tags=[str(item) for item in payload.get("tags") or []],
        smoke_free_preference=payload.get("smoke_free_preference")
        if isinstance(payload.get("smoke_free_preference"), bool)
        else None,
        usual_party_size=_optional_int(payload.get("usual_party_size")),
        usual_party_size_confidence=float(payload.get("usual_party_size_confidence") or 0.0),
        usual_start_hours=[int(item) for item in payload.get("usual_start_hours") or []],
        usual_weekdays=[int(item) for item in payload.get("usual_weekdays") or []],
        no_contact=bool(payload.get("no_contact")),
        last_invited_at=_parse_datetime(str(payload.get("last_invited_at") or "")),
        decline_count_30d=int(payload.get("decline_count_30d") or 0),
        max_games_per_day=int(payload.get("max_games_per_day") or 1),
        min_hours_between_games=float(payload.get("min_hours_between_games") or 6.0),
        invite_cooldown_hours=float(payload.get("invite_cooldown_hours") or 6.0),
        daily_invite_limit=int(payload.get("daily_invite_limit") or 3),
        fatigue_sensitivity=float(payload.get("fatigue_sensitivity") or 1.0),
        metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {},
    )


def _play_preference_from_payload(payload: dict[str, Any]) -> PlayPreference:
    return PlayPreference(
        game_type=str(payload.get("game_type") or ""),
        preferred_levels=[str(item) for item in payload.get("preferred_levels") or []],
        preferred_rulesets=[str(item) for item in payload.get("preferred_rulesets") or []],
        preferred_variants=[str(item) for item in payload.get("preferred_variants") or []],
        preferred_play_options=[str(item) for item in payload.get("preferred_play_options") or []],
        avoid_play_options=[str(item) for item in payload.get("avoid_play_options") or []],
    )


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=DEFAULT_TZ)
    return parsed


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dump_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads_dict(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
