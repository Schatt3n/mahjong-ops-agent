"""用独立 SQLite 记录角色唤醒、决定与终态，支持中断后继续执行。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any, Literal
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class RoleEvent:
    """一个角色对稳定触发源的事件，决定和投递结果分别持久化。"""

    event_id: str
    actor_id: str
    kind: str
    origin_id: str
    due_at: datetime
    caused_by: str | None
    details: dict[str, Any]
    status: Literal["pending", "claimed", "finished"]
    claim_id: str | None
    decision: dict[str, Any] | None
    result: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """保留事件来源与结果，时间使用含时区的 ISO 表达。"""
        return {**asdict(self), "due_at": self.due_at.isoformat()}


class VenueRoleJournal:
    """调度器按虚拟时间领取动作，SQLite 保存唯一触发与恢复所需的已定动作。"""

    def __init__(self, path: Path) -> None:
        """一个营业演练使用一个事件账本和独立的调度进程锁。"""
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.path.with_name(f"{self.path.name}.owner.lock")
        self._owns_run = False
        with self._database() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS venue_role_events (
                    sequence INTEGER PRIMARY KEY,
                    event_id TEXT NOT NULL UNIQUE,
                    actor_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    origin_id TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    caused_by TEXT,
                    details_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'claimed', 'finished')),
                    claim_id TEXT,
                    decision_json TEXT,
                    result_json TEXT,
                    UNIQUE (actor_id, kind, origin_id),
                    CHECK ((status = 'claimed') = (claim_id IS NOT NULL)),
                    CHECK ((status = 'finished') = (result_json IS NOT NULL))
                );
                CREATE INDEX IF NOT EXISTS venue_role_events_due ON venue_role_events (status, due_at, sequence);
                CREATE INDEX IF NOT EXISTS venue_role_events_actor ON venue_role_events (actor_id, status, due_at, sequence);
                CREATE TABLE IF NOT EXISTS venue_role_clock (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    due_at TEXT NOT NULL
                );
            """)

    @contextmanager
    def _database(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        """领取和状态转移用短事务完成，模型和传输不持有数据库写锁。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.row_factory = sqlite3.Row
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db

    @contextmanager
    def run_owner(self) -> Iterator[VenueRoleJournal]:
        """只允许一个调度器执行。恢复未完成领取时保留已写的角色决定。"""
        with self._lock_path.open("a+b") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("role journal already has an active scheduler") from exc
            self._release_claims()
            self._owns_run = True
            try:
                yield self
            finally:
                self._owns_run = False
                self._release_claims()
                # 退出后旧领取凭证全部失效，调度锁随文件关闭而释放。

    def enqueue(self, *, actor_id: str, kind: str, origin_id: str, due_at: datetime,
                caused_by: str | None = None, details: dict[str, Any] | None = None) -> RoleEvent:
        """同角色、类型和来源的重投返回原事件，拒绝用重投修改时间或负载。"""
        identity = (actor_id, kind, origin_id)
        if any(not isinstance(part, str) or not part.strip() for part in identity):
            raise ValueError("role event requires a stable actor, kind and origin")
        if not isinstance(due_at, datetime) or due_at.utcoffset() is None:
            raise ValueError("role event due time must include a timezone")
        if caused_by is not None and (not isinstance(caused_by, str) or not caused_by.strip()):
            raise ValueError("role event message reference must be nonempty")
        event_id = "role-event:" + sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
        stamp = due_at.astimezone(timezone.utc).isoformat(timespec="microseconds")
        payload = _encode({} if details is None else details)
        with self._database(write=True) as db:
            current = db.execute("SELECT * FROM venue_role_events WHERE event_id = ?", (event_id,)).fetchone()
            if current is None:
                clock = db.execute("SELECT due_at FROM venue_role_clock WHERE singleton = 1").fetchone()
                if clock is not None and stamp < clock["due_at"]:
                    raise ValueError("role event due time is before the virtual clock")
                db.execute("""
                    INSERT INTO venue_role_events (event_id, actor_id, kind, origin_id, due_at, caused_by, details_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (event_id, *identity, stamp, caused_by, payload))
                return self._event(self._row(db, event_id))
            if (current["actor_id"], current["kind"], current["origin_id"], current["due_at"],
                current["caused_by"], current["details_json"]) != (*identity, stamp, caused_by, payload):
                raise ValueError("role event trigger conflicts with an existing event")
            return self._event(current)

    def claim_wave(self, limit: int) -> tuple[RoleEvent, ...]:
        """在最早未办结时刻按稳定顺序领取，每人一个，领取总数受并发上限约束。"""
        self._require_owner()
        if type(limit) is not int or limit < 1:
            raise ValueError("role event worker limit must be a positive integer")
        with self._database(write=True) as db:
            active = db.execute("SELECT COUNT(*) FROM venue_role_events WHERE status = 'claimed'").fetchone()[0]
            if active >= limit:
                return ()
            rows = db.execute("""
                WITH available AS (
                    SELECT pending.event_id, pending.sequence, pending.due_at,
                           ROW_NUMBER() OVER (PARTITION BY pending.actor_id ORDER BY pending.sequence) AS actor_order
                    FROM venue_role_events AS pending
                    WHERE pending.status = 'pending'
                    AND pending.due_at = (SELECT MIN(due_at) FROM venue_role_events WHERE status != 'finished')
                    AND NOT EXISTS (
                        SELECT 1 FROM venue_role_events AS active
                        WHERE active.actor_id = pending.actor_id AND active.status = 'claimed'
                    )
                )
                SELECT event_id, due_at FROM available WHERE actor_order = 1 ORDER BY sequence LIMIT ?
            """, (limit - active,)).fetchall()
            if rows:
                db.execute("""
                    INSERT INTO venue_role_clock VALUES (1, ?)
                    ON CONFLICT (singleton) DO UPDATE SET due_at = excluded.due_at
                """, (rows[0]["due_at"],))
            claimed = []
            for row in rows:
                event_id = row["event_id"]
                db.execute("UPDATE venue_role_events SET status = 'claimed', claim_id = ? WHERE event_id = ?",
                           (uuid4().hex, event_id))
                claimed.append(self._event(self._row(db, event_id)))
            return tuple(claimed)

    def save_decision(self, claim: RoleEvent, decision: dict[str, Any]) -> RoleEvent:
        """投递前保存唯一决定。同一决定重试可重用，另一个决定不得覆盖。"""
        self._require_owner()
        value = _encode(decision)
        with self._database(write=True) as db:
            row = self._claimed_row(db, claim)
            if row["decision_json"] is not None:
                if row["decision_json"] != value:
                    raise ValueError("role event decision conflicts with its saved decision")
                return self._event(row)
            db.execute("UPDATE venue_role_events SET decision_json = ? WHERE event_id = ?", (value, claim.event_id))
            return self._event(self._row(db, claim.event_id))

    def finish(self, claim: RoleEvent, result: dict[str, Any]) -> RoleEvent:
        """保存投递、等待或失败结果，只有持有当前凭证且已记录决定才可办结。"""
        self._require_owner()
        value = _encode(result)
        with self._database(write=True) as db:
            row = self._claimed_row(db, claim)
            if row["decision_json"] is None:
                raise ValueError("role event needs a saved decision before completion")
            db.execute("""
                UPDATE venue_role_events SET status = 'finished', claim_id = NULL, result_json = ? WHERE event_id = ?
            """, (value, claim.event_id))
            return self._event(self._row(db, claim.event_id))

    def get(self, event_id: str) -> RoleEvent:
        """按稳定事件 ID 读取持久化决定及当前终态。"""
        with self._database() as db:
            return self._event(self._row(db, event_id))

    def events(self) -> tuple[RoleEvent, ...]:
        """按虚拟时间和首次入队顺序导出全轮事件供核对。"""
        with self._database() as db:
            return tuple(self._event(row) for row in db.execute("SELECT * FROM venue_role_events ORDER BY due_at, sequence"))

    def _require_owner(self) -> None:
        """只有处在独占生命周期的调度器才能领取或修改执行状态。"""
        if not self._owns_run:
            raise RuntimeError("role event execution requires the active scheduler")

    def _release_claims(self) -> None:
        """只在持有进程锁时回收未完成领取，已决定的动作留供续跑。"""
        with self._database(write=True) as db:
            db.execute("UPDATE venue_role_events SET status = 'pending', claim_id = NULL WHERE status = 'claimed'")

    @staticmethod
    def _row(db: sqlite3.Connection, event_id: str) -> sqlite3.Row:
        """从事务当前快照读取唯一事件。"""
        row = db.execute("SELECT * FROM venue_role_events WHERE event_id = ?", (event_id,)).fetchone()
        if row is None:
            raise KeyError(event_id)
        return row

    def _claimed_row(self, db: sqlite3.Connection, claim: RoleEvent) -> sqlite3.Row:
        """用 SQLite 的当前凭证拒绝旧角色执行者或已办结的重复写。"""
        row = self._row(db, claim.event_id)
        if row["status"] != "claimed" or row["claim_id"] != claim.claim_id:
            raise ValueError("stale role event claim")
        return row

    @staticmethod
    def _event(row: sqlite3.Row) -> RoleEvent:
        """把账本快照解码成独立事件，调用方修改字典不会改写已存结果。"""
        return RoleEvent(
            event_id=row["event_id"], actor_id=row["actor_id"], kind=row["kind"], origin_id=row["origin_id"],
            due_at=datetime.fromisoformat(row["due_at"]), caused_by=row["caused_by"], details=json.loads(row["details_json"]),
            status=row["status"], claim_id=row["claim_id"],
            decision=json.loads(row["decision_json"]) if row["decision_json"] is not None else None,
            result=json.loads(row["result_json"]) if row["result_json"] is not None else None,
        )


def _encode(value: dict[str, Any]) -> str:
    """对象键顺序不影响幂等核对，禁止不可复现的特殊数值。"""
    if not isinstance(value, dict):
        raise ValueError("role event payload must be an object")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
