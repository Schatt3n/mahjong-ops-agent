"""Layer 1: deterministic population and group factory for simulations."""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

from faker import Faker


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mahjong_agent_runtime import CustomerProfile, SQLiteAgentStore  # noqa: E402


DEFAULT_DB_PATH = Path(__file__).with_name("test_sim.db")
DEVELOPMENT_DB_PATH = ROOT / "data" / "agent_runtime.sqlite3"
GROUP_ID = "sim_group_001"
GROUP_NAME = "百人麻将测试群"
DEFAULT_USER_COUNT = 100
DEFAULT_SEED = 42

PERSONA_LURKER = "lurker"
PERSONA_ACTIVE_GAMBLER = "active_gambler"
PERSONA_TROUBLEMAKER = "troublemaker"


@dataclass(slots=True, frozen=True)
class VirtualUser:
    """One deterministic synthetic customer plus simulation-only attributes."""

    customer_id: str
    display_name: str
    balance: float
    preferred_game: str
    persona: str
    interleaves_chitchat: bool = False


def ensure_isolated_database(path: Path) -> Path:
    """Refuse every database except an explicitly named simulation database."""

    resolved = path.expanduser().resolve()
    if resolved == DEVELOPMENT_DB_PATH.resolve():
        raise RuntimeError("simulation database must not be the development database")
    if resolved.name != "test_sim.db":
        raise RuntimeError("simulation database filename must be test_sim.db")
    return resolved


def reset_sqlite_database(path: Path) -> None:
    """Remove only the isolated simulation database and its SQLite sidecars."""

    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        candidate.unlink(missing_ok=True)


def _persona_for_index(index: int) -> str:
    if index <= 80:
        return PERSONA_LURKER
    if index <= 95:
        return PERSONA_ACTIVE_GAMBLER
    return PERSONA_TROUBLEMAKER


def build_population(
    db_path: Path,
    *,
    user_count: int = DEFAULT_USER_COUNT,
    seed: int = DEFAULT_SEED,
) -> tuple[SQLiteAgentStore, list[VirtualUser]]:
    """Create exactly 100 Chinese users and put all of them in one group.

    Stable customer fields go through ``CustomerProfile``. Balance and persona
    are simulation-only dimensions, so they are also stored in a dedicated
    flat table instead of expanding the production domain model.
    """

    if user_count != DEFAULT_USER_COUNT:
        raise ValueError("the hundred-user simulator requires exactly 100 users")
    isolated_path = ensure_isolated_database(db_path)
    isolated_path.parent.mkdir(parents=True, exist_ok=True)
    reset_sqlite_database(isolated_path)

    store = SQLiteAgentStore(isolated_path)
    faker = Faker("zh_CN")
    faker.seed_instance(seed)
    rng = random.Random(seed)
    users: list[VirtualUser] = []

    for index in range(1, user_count + 1):
        preferred_game = rng.choice(("sichuan_mahjong", "national_standard_mahjong"))
        balance = round(rng.uniform(100.0, 5000.0), 2)
        user = VirtualUser(
            customer_id=f"sim_user_{index:03d}",
            display_name=faker.name(),
            balance=balance,
            preferred_game=preferred_game,
            persona=_persona_for_index(index),
            # Five of the twenty speaking users have a stable conversational
            # habit of briefly chatting before returning to the Mahjong task.
            interleaves_chitchat=index > 80 and index % 4 == 0,
        )
        users.append(user)
        store.upsert_customer(
            CustomerProfile(
                customer_id=user.customer_id,
                display_name=user.display_name,
                public_name=user.display_name,
                preferred_games=[preferred_game],
                profile_facts=[f"simulation_balance={balance:.2f}"],
            )
        )

    with store._lock, store._connection:
        store._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS simulation_user_profiles(
                customer_id TEXT PRIMARY KEY,
                balance REAL NOT NULL,
                preferred_game TEXT NOT NULL,
                persona TEXT NOT NULL,
                interleaves_chitchat INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(customer_id) REFERENCES runtime_customers(customer_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS simulation_group_chats(
                group_id TEXT PRIMARY KEY,
                group_name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS simulation_group_members(
                group_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                joined_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(group_id, customer_id),
                FOREIGN KEY(group_id) REFERENCES simulation_group_chats(group_id) ON DELETE CASCADE,
                FOREIGN KEY(customer_id) REFERENCES runtime_customers(customer_id) ON DELETE CASCADE
            );
            """
        )
        store._connection.executemany(
            """
            INSERT INTO simulation_user_profiles(
                customer_id,
                balance,
                preferred_game,
                persona,
                interleaves_chitchat
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    user.customer_id,
                    user.balance,
                    user.preferred_game,
                    user.persona,
                    int(user.interleaves_chitchat),
                )
                for user in users
            ],
        )
        store._connection.execute(
            "INSERT INTO simulation_group_chats(group_id, group_name) VALUES (?, ?)",
            (GROUP_ID, GROUP_NAME),
        )
        store._connection.executemany(
            "INSERT INTO simulation_group_members(group_id, customer_id) VALUES (?, ?)",
            [(GROUP_ID, user.customer_id) for user in users],
        )
    return store, users
