"""Durable source-stage frontier for continuous cinematic acquisition."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FRONTIER_FILENAME = "source-frontier.sqlite3"
ACTIVE_STATES = ("discovered", "downloaded", "scanned")
TERMINAL_STATES = ("complete", "rejected")
ALL_STATES = (*ACTIVE_STATES, *TERMINAL_STATES)
STAGE_FOR_STATE = {
    "discovered": "download",
    "downloaded": "scan",
    "scanned": "extract",
}
MUTABLE_COLUMNS = {"downloaded_path", "download_json", "scan_json"}


def _now_iso(timestamp: float | None = None) -> str:
    return datetime.fromtimestamp(
        time.time() if timestamp is None else timestamp, UTC
    ).isoformat()


def source_key(platform: str, video_id: str) -> str:
    return f"{platform}:{video_id}"


def connect_frontier(workspace: Path) -> sqlite3.Connection:
    workspace.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        workspace / FRONTIER_FILENAME,
        timeout=30,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS source_jobs (
            source_key TEXT PRIMARY KEY,
            platform TEXT NOT NULL,
            video_id TEXT NOT NULL,
            candidate_json TEXT NOT NULL,
            priority REAL NOT NULL DEFAULT 0,
            state TEXT NOT NULL CHECK(
                state IN ('discovered','downloaded','scanned','complete','rejected')
            ),
            available_at REAL NOT NULL,
            lease_owner TEXT,
            lease_started_at REAL,
            lease_expires_at REAL,
            stage_attempts INTEGER NOT NULL DEFAULT 0,
            total_attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            terminal_reason TEXT,
            downloaded_path TEXT,
            download_json TEXT,
            scan_json TEXT,
            discovered_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(platform, video_id)
        );
        CREATE INDEX IF NOT EXISTS source_jobs_ready
        ON source_jobs(state, available_at, lease_expires_at, priority DESC);
        CREATE TABLE IF NOT EXISTS source_stage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_key TEXT NOT NULL REFERENCES source_jobs(source_key),
            stage TEXT NOT NULL,
            outcome TEXT NOT NULL,
            worker TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            duration_seconds REAL NOT NULL,
            details_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS source_stage_events_finished
        ON source_stage_events(stage, finished_at);
        CREATE TABLE IF NOT EXISTS source_workers (
            worker TEXT PRIMARY KEY,
            stage TEXT NOT NULL,
            state TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            details_json TEXT NOT NULL
        );
        """
    )
    return connection


def enqueue_source(
    connection: sqlite3.Connection,
    candidates: list[dict[str, Any]],
    *,
    priority: float = 0.0,
    initial_state: str = "discovered",
    now: float | None = None,
) -> bool:
    """Insert one source group without resetting work already in the frontier."""
    if not candidates:
        raise ValueError("A source group must contain at least one candidate")
    if initial_state not in ALL_STATES:
        raise ValueError(f"Unknown source state: {initial_state}")
    base = candidates[0]
    platform = str(base.get("source_platform") or "unknown")
    video_id = str(base["video_id"])
    if any(str(item["video_id"]) != video_id for item in candidates):
        raise ValueError("All candidates in a source group must share a video_id")
    timestamp = time.time() if now is None else now
    encoded = json.dumps(candidates, separators=(",", ":"))
    existed = connection.execute(
        "SELECT 1 FROM source_jobs WHERE source_key=?",
        (source_key(platform, video_id),),
    ).fetchone()
    connection.execute(
        """INSERT INTO source_jobs(
        source_key,platform,video_id,candidate_json,priority,state,available_at,
        discovered_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)
        ON CONFLICT(source_key) DO UPDATE SET
        candidate_json=excluded.candidate_json,
        priority=MAX(source_jobs.priority,excluded.priority),
        updated_at=excluded.updated_at""",
        (
            source_key(platform, video_id),
            platform,
            video_id,
            encoded,
            float(priority),
            initial_state,
            timestamp,
            _now_iso(timestamp),
            _now_iso(timestamp),
        ),
    )
    return existed is None


def enqueue_sources(
    connection: sqlite3.Connection,
    groups: Iterable[list[dict[str, Any]]],
    *,
    priority: Any | None = None,
    initial_state: Any | None = None,
) -> int:
    inserted = 0
    connection.execute("BEGIN IMMEDIATE")
    try:
        for group in groups:
            inserted += int(
                enqueue_source(
                    connection,
                    group,
                    priority=float(priority(group) if priority else 0.0),
                    initial_state=str(
                        initial_state(group) if initial_state else "discovered"
                    ),
                )
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return inserted


def _decode_job(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["candidates"] = json.loads(result.pop("candidate_json"))
    for key in ("download_json", "scan_json"):
        value = result.get(key)
        result[key] = json.loads(value) if value else None
    return result


def claim_source(
    connection: sqlite3.Connection,
    state: str,
    *,
    worker: str,
    lease_seconds: float,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Atomically lease the highest-priority ready source in one stage."""
    if state not in ACTIVE_STATES:
        raise ValueError(f"Cannot claim terminal state: {state}")
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    timestamp = time.time() if now is None else now
    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute(
            """SELECT * FROM source_jobs
            WHERE state=? AND available_at<=?
            AND (lease_expires_at IS NULL OR lease_expires_at<=? OR lease_owner=?)
            ORDER BY CASE WHEN lease_owner=? THEN 0 ELSE 1 END,
            priority DESC, discovered_at, source_key LIMIT 1""",
            (state, timestamp, timestamp, worker, worker),
        ).fetchone()
        if row is None:
            connection.commit()
            return None
        changed = connection.execute(
            """UPDATE source_jobs SET lease_owner=?,lease_started_at=?,
            lease_expires_at=?,stage_attempts=stage_attempts+1,
            total_attempts=total_attempts+1,updated_at=?
            WHERE source_key=? AND state=?
            AND (lease_expires_at IS NULL OR lease_expires_at<=? OR lease_owner=?)""",
            (
                worker,
                timestamp,
                timestamp + lease_seconds,
                _now_iso(timestamp),
                row["source_key"],
                state,
                timestamp,
                worker,
            ),
        ).rowcount
        if changed != 1:
            connection.rollback()
            return None
        claimed = connection.execute(
            "SELECT * FROM source_jobs WHERE source_key=?",
            (row["source_key"],),
        ).fetchone()
        connection.commit()
        return _decode_job(claimed)
    except Exception:
        connection.rollback()
        raise


def release_worker_leases(
    connection: sqlite3.Connection,
    *,
    worker: str,
    state: str,
    now: float | None = None,
) -> int:
    """Expire claims left by a previous process with the same stable worker ID."""
    if state not in ACTIVE_STATES:
        raise ValueError(f"Cannot release terminal state: {state}")
    timestamp = time.time() if now is None else now
    return connection.execute(
        """UPDATE source_jobs SET lease_expires_at=?,updated_at=?
        WHERE state=? AND lease_owner=?""",
        (timestamp, _now_iso(timestamp), state, worker),
    ).rowcount


def _finish_event(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    stage: str,
    outcome: str,
    worker: str,
    now: float,
    details: dict[str, Any] | None,
) -> None:
    started = float(row["lease_started_at"] or now)
    connection.execute(
        """INSERT INTO source_stage_events(
        source_key,stage,outcome,worker,started_at,finished_at,duration_seconds,
        details_json) VALUES(?,?,?,?,?,?,?,?)""",
        (
            row["source_key"],
            stage,
            outcome,
            worker,
            _now_iso(started),
            _now_iso(now),
            max(0.0, now - started),
            json.dumps(details or {}, separators=(",", ":")),
        ),
    )


def finish_source(
    connection: sqlite3.Connection,
    source_key_value: str,
    *,
    worker: str,
    expected_state: str,
    next_state: str,
    outcome: str = "success",
    updates: dict[str, Any] | None = None,
    terminal_reason: str | None = None,
    details: dict[str, Any] | None = None,
    now: float | None = None,
) -> None:
    """Commit a leased transition and its timing event in one transaction."""
    if expected_state not in ACTIVE_STATES or next_state not in ALL_STATES:
        raise ValueError("Invalid frontier state transition")
    invalid = set(updates or {}) - MUTABLE_COLUMNS
    if invalid:
        raise ValueError(f"Unsupported source updates: {sorted(invalid)}")
    timestamp = time.time() if now is None else now
    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute(
            "SELECT * FROM source_jobs WHERE source_key=?",
            (source_key_value,),
        ).fetchone()
        if (
            row is None
            or row["state"] != expected_state
            or row["lease_owner"] != worker
        ):
            raise RuntimeError("Source lease is no longer owned by this worker")
        assignments = [
            "state=?",
            "available_at=?",
            "lease_owner=NULL",
            "lease_started_at=NULL",
            "lease_expires_at=NULL",
            "stage_attempts=0",
            "last_error=NULL",
            "terminal_reason=?",
            "updated_at=?",
        ]
        values: list[Any] = [
            next_state,
            timestamp,
            terminal_reason,
            _now_iso(timestamp),
        ]
        for key, value in (updates or {}).items():
            assignments.append(f"{key}=?")
            values.append(
                json.dumps(value, separators=(",", ":"))
                if key.endswith("_json") and value is not None
                else value
            )
        values.append(source_key_value)
        connection.execute(
            f"UPDATE source_jobs SET {','.join(assignments)} WHERE source_key=?",
            values,
        )
        _finish_event(
            connection,
            row,
            stage=STAGE_FOR_STATE[expected_state],
            outcome=outcome,
            worker=worker,
            now=timestamp,
            details=details,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def retry_source(
    connection: sqlite3.Connection,
    source_key_value: str,
    *,
    worker: str,
    expected_state: str,
    error: str,
    backoff_seconds: float,
    max_attempts: int,
    details: dict[str, Any] | None = None,
    now: float | None = None,
) -> str:
    """Release a transient failure or terminally reject an exhausted source."""
    timestamp = time.time() if now is None else now
    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute(
            "SELECT * FROM source_jobs WHERE source_key=?",
            (source_key_value,),
        ).fetchone()
        if (
            row is None
            or row["state"] != expected_state
            or row["lease_owner"] != worker
        ):
            raise RuntimeError("Source lease is no longer owned by this worker")
        exhausted = int(row["stage_attempts"]) >= max_attempts
        next_state = "rejected" if exhausted else expected_state
        connection.execute(
            """UPDATE source_jobs SET state=?,available_at=?,lease_owner=NULL,
            lease_started_at=NULL,lease_expires_at=NULL,last_error=?,
            terminal_reason=?,updated_at=? WHERE source_key=?""",
            (
                next_state,
                timestamp if exhausted else timestamp + max(0.0, backoff_seconds),
                error,
                "retry_exhausted" if exhausted else None,
                _now_iso(timestamp),
                source_key_value,
            ),
        )
        _finish_event(
            connection,
            row,
            stage=STAGE_FOR_STATE[expected_state],
            outcome="retry_exhausted" if exhausted else "retry",
            worker=worker,
            now=timestamp,
            details={"error": error, **(details or {})},
        )
        connection.commit()
        return next_state
    except Exception:
        connection.rollback()
        raise


def frontier_counts(connection: sqlite3.Connection) -> dict[str, int]:
    counts = Counter(
        {
            row["state"]: int(row["count"])
            for row in connection.execute(
                "SELECT state,COUNT(*) AS count FROM source_jobs GROUP BY state"
            )
        }
    )
    return {state: counts[state] for state in ALL_STATES}


def heartbeat_worker(
    connection: sqlite3.Connection,
    worker: str,
    *,
    stage: str,
    state: str = "running",
    details: dict[str, Any] | None = None,
    now: float | None = None,
) -> None:
    connection.execute(
        """INSERT INTO source_workers(worker,stage,state,updated_at,details_json)
        VALUES(?,?,?,?,?) ON CONFLICT(worker) DO UPDATE SET
        stage=excluded.stage,state=excluded.state,updated_at=excluded.updated_at,
        details_json=excluded.details_json""",
        (
            worker,
            stage,
            state,
            _now_iso(now),
            json.dumps(details or {}, separators=(",", ":")),
        ),
    )


def downloaded_queue_bytes(connection: sqlite3.Connection) -> int:
    total = 0
    for row in connection.execute(
        "SELECT download_json FROM source_jobs WHERE state='downloaded'"
    ):
        try:
            total += int(json.loads(row["download_json"] or "{}").get("bytes") or 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return total


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * quantile) - 1))
    return round(float(ordered[index]), 3)


def frontier_snapshot(
    workspace: Path,
    *,
    window_minutes: float = 15.0,
    now: float | None = None,
) -> dict[str, Any]:
    """Return queue and recent-stage health without mutating frontier state."""
    path = workspace / FRONTIER_FILENAME
    if not path.exists():
        return {"enabled": False, "counts": {state: 0 for state in ALL_STATES}}
    timestamp = time.time() if now is None else now
    connection = connect_frontier(workspace)
    counts = frontier_counts(connection)
    leased = connection.execute(
        """SELECT state,COUNT(*) AS count FROM source_jobs
        WHERE lease_expires_at>? GROUP BY state""",
        (timestamp,),
    ).fetchall()
    oldest: dict[str, float | None] = {}
    for state in ACTIVE_STATES:
        row = connection.execute(
            """SELECT MIN(available_at) AS oldest FROM source_jobs
            WHERE state=? AND available_at<=?
            AND (lease_expires_at IS NULL OR lease_expires_at<=?)""",
            (state, timestamp, timestamp),
        ).fetchone()
        oldest[state] = (
            round(max(0.0, timestamp - float(row["oldest"])) / 60.0, 3)
            if row and row["oldest"] is not None
            else None
        )
    cutoff = _now_iso(timestamp - window_minutes * 60.0)
    events = connection.execute(
        """SELECT stage,outcome,duration_seconds FROM source_stage_events
        WHERE finished_at>=?""",
        (cutoff,),
    ).fetchall()
    stage_metrics: dict[str, dict[str, Any]] = {}
    for stage in STAGE_FOR_STATE.values():
        selected = [row for row in events if row["stage"] == stage]
        active = [
            row
            for row in selected
            if not str(row["outcome"]).startswith("cache_")
        ]
        durations = [float(row["duration_seconds"]) for row in active]
        outcomes = Counter(str(row["outcome"]) for row in selected)
        stage_metrics[stage] = {
            "events": len(selected),
            "per_minute": round(len(selected) / max(window_minutes, 1e-9), 4),
            "active_events": len(active),
            "active_per_minute": round(
                len(active) / max(window_minutes, 1e-9), 4
            ),
            "outcomes": dict(outcomes),
            "duration_p50_seconds": _percentile(durations, 0.5),
            "duration_p95_seconds": _percentile(durations, 0.95),
        }
    queue_bytes = downloaded_queue_bytes(connection)
    workers = []
    for row in connection.execute("SELECT * FROM source_workers ORDER BY worker"):
        updated = datetime.fromisoformat(str(row["updated_at"])).timestamp()
        workers.append(
            {
                "worker": row["worker"],
                "stage": row["stage"],
                "state": (row["state"] if timestamp - updated < 30.0 else "stale"),
                "updated_at": row["updated_at"],
                "details": json.loads(row["details_json"]),
            }
        )
    try:
        autoscaler = json.loads((workspace / "source-stage-control.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        autoscaler = {"enabled": False}
    connection.close()
    return {
        "enabled": True,
        "window_minutes": window_minutes,
        "counts": counts,
        "downloaded_queue_bytes": queue_bytes,
        "leased": {row["state"]: int(row["count"]) for row in leased},
        "oldest_ready_minutes": oldest,
        "stages": stage_metrics,
        "workers": workers,
        "autoscaler": autoscaler,
    }
