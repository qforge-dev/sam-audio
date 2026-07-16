"""Durable source-stage frontier for continuous cinematic acquisition."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from collections import Counter, deque
from collections.abc import Iterable, Mapping
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
_NETWORK_WINDOW_SECONDS = 60.0
_network_samples: deque[tuple[float, int]] = deque()
_network_samples_lock = threading.Lock()
DISCOVERY_PROBE_ACTIVE_SOURCES = 8
DISCOVERY_MIN_SCAN_SOURCES = 12
DISCOVERY_MIN_FINAL_RECORDS = 20
DISCOVERY_MIN_SCAN_PASS_RATE = 0.08
DISCOVERY_MIN_FINAL_ACCEPTANCE_RATE = 0.35


def _now_iso(timestamp: float | None = None) -> str:
    return datetime.fromtimestamp(
        time.time() if timestamp is None else timestamp, UTC
    ).isoformat()


def source_key(platform: str, video_id: str) -> str:
    return f"{platform}:{video_id}"


def _is_quality_gated_discovery(key: str) -> bool:
    return key in {
        "deep_page_v1",
        "accepted_related_v1",
        "accepted_channel_v1",
        "query_family:cinematic_gameplay_context_v2",
    }


def discovery_strategy_admission(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Bound new discovery lanes until downstream evidence proves their yield."""
    key = str(metrics.get("key") or "legacy")
    active = int(metrics.get("active_sources") or 0)
    scanned = int(metrics.get("scan_evaluated_sources") or 0)
    scan_passed = int(metrics.get("scan_passed_sources") or 0)
    records = int(metrics.get("final_records") or 0)
    accepted = int(metrics.get("final_accepted") or 0)
    scan_rate = scan_passed / scanned if scanned else None
    final_rate = accepted / records if records else None
    if not _is_quality_gated_discovery(key):
        return {"state": "unrestricted", "new_source_allowance": None}
    reason = None
    if records >= 10 and accepted == 0:
        reason = "no_final_accepts"
    elif (
        records >= DISCOVERY_MIN_FINAL_RECORDS
        and final_rate is not None
        and final_rate < DISCOVERY_MIN_FINAL_ACCEPTANCE_RATE
    ):
        reason = "low_final_acceptance"
    elif (
        scanned >= DISCOVERY_MIN_SCAN_SOURCES
        and scan_rate is not None
        and scan_rate < DISCOVERY_MIN_SCAN_PASS_RATE
    ):
        reason = "low_scan_pass_rate"
    if reason:
        return {
            "state": "suspended",
            "reason": reason,
            "new_source_allowance": 0,
        }
    if records >= DISCOVERY_MIN_FINAL_RECORDS:
        return {"state": "healthy", "new_source_allowance": None}
    return {
        "state": "probing",
        "reason": "awaiting_downstream_sample",
        "new_source_allowance": max(0, DISCOVERY_PROBE_ACTIVE_SOURCES - active),
    }


def discovery_strategy_snapshot(
    connection: sqlite3.Connection,
    *,
    catalog_path: Path | None = None,
    platform: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Attribute source-stage and final acceptance to each discovery lane."""
    parameters: tuple[Any, ...] = (platform,) if platform else ()
    where = "WHERE platform=?" if platform else ""
    rows = connection.execute(
        f"""SELECT platform,state,scan_json,candidate_json FROM source_jobs {where}""",
        parameters,
    ).fetchall()
    aggregates: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            candidates = json.loads(row["candidate_json"])
            base = candidates[0] if candidates else {}
        except (json.JSONDecodeError, TypeError):
            base = {}
        key = str(base.get("discovery_quality_key") or "legacy")
        item = aggregates.setdefault(
            key,
            {
                "key": key,
                "sources": 0,
                "active_sources": 0,
                "scan_evaluated_sources": 0,
                "scan_passed_sources": 0,
                "final_records": 0,
                "final_accepted": 0,
                "platforms": Counter(),
                "states": Counter(),
            },
        )
        state = str(row["state"])
        item["sources"] += 1
        item["platforms"][str(row["platform"])] += 1
        item["states"][state] += 1
        if state in ACTIVE_STATES:
            item["active_sources"] += 1
        if row["scan_json"]:
            item["scan_evaluated_sources"] += 1
            if state in {"scanned", "complete"}:
                item["scan_passed_sources"] += 1
    if catalog_path and catalog_path.exists():
        catalog = sqlite3.connect(f"file:{catalog_path}?mode=ro", uri=True, timeout=30)
        query = (
            """SELECT COALESCE(json_extract(r.record_json,"""
            """'$.discovery_quality_key'),'legacy') AS quality_key,"""
            """COUNT(*) AS records,SUM(CASE WHEN a.sha256 IS NULL THEN 0 ELSE 1 END) """
            """AS accepted FROM records r LEFT JOIN accepted a USING(sha256)"""
        )
        values: tuple[Any, ...] = ()
        if platform:
            query += " WHERE r.platform=?"
            values = (platform,)
        query += " GROUP BY quality_key"
        for quality_key, records, accepted in catalog.execute(query, values):
            key = str(quality_key or "legacy")
            item = aggregates.setdefault(
                key,
                {
                    "key": key,
                    "sources": 0,
                    "active_sources": 0,
                    "scan_evaluated_sources": 0,
                    "scan_passed_sources": 0,
                    "final_records": 0,
                    "final_accepted": 0,
                    "platforms": Counter(),
                    "states": Counter(),
                },
            )
            item["final_records"] += int(records)
            item["final_accepted"] += int(accepted)
        catalog.close()
    result: dict[str, dict[str, Any]] = {}
    for key, item in sorted(aggregates.items()):
        scanned = int(item["scan_evaluated_sources"])
        records = int(item["final_records"])
        normalized = {
            **item,
            "platforms": dict(item["platforms"]),
            "states": dict(item["states"]),
            "scan_pass_rate_percent": (
                round(100.0 * item["scan_passed_sources"] / scanned, 2)
                if scanned
                else None
            ),
            "final_acceptance_percent": (
                round(100.0 * item["final_accepted"] / records, 2) if records else None
            ),
        }
        normalized["admission"] = discovery_strategy_admission(normalized)
        result[key] = normalized
    return result


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
        CREATE TABLE IF NOT EXISTS source_provider_circuits (
            platform TEXT PRIMARY KEY,
            state TEXT NOT NULL CHECK(state IN ('closed','open','half_open')),
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            open_count INTEGER NOT NULL DEFAULT 0,
            retry_at REAL,
            probe_owner TEXT,
            probe_expires_at REAL,
            last_error TEXT,
            last_success_at TEXT,
            updated_at TEXT NOT NULL
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
    respect_provider_circuits: bool = False,
    provider_weights: Mapping[str, float] | None = None,
) -> dict[str, Any] | None:
    """Atomically lease the highest-priority ready source in one stage."""
    if state not in ACTIVE_STATES:
        raise ValueError(f"Cannot claim terminal state: {state}")
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    timestamp = time.time() if now is None else now
    weights = {
        str(platform): float(weight)
        for platform, weight in (provider_weights or {}).items()
        if float(weight) > 0
    }
    connection.execute("BEGIN IMMEDIATE")
    try:
        circuit_join = (
            "LEFT JOIN source_provider_circuits AS circuit "
            "ON circuit.platform=candidate.platform"
            if respect_provider_circuits
            else ""
        )
        circuit_filter = (
            """AND (circuit.platform IS NULL OR circuit.state='closed'
            OR (circuit.state='open' AND circuit.retry_at<=?)
            OR (circuit.state='half_open' AND
                (circuit.probe_expires_at<=? OR circuit.probe_owner=?)))"""
            if respect_provider_circuits
            else ""
        )
        parameters: list[Any] = [state, timestamp, timestamp, worker]
        if respect_provider_circuits:
            parameters.extend((timestamp, timestamp, worker))
        parameters.extend((worker, timestamp, json.dumps(weights)))
        row = connection.execute(
            f"""SELECT candidate.* FROM source_jobs AS candidate
            {circuit_join}
            WHERE candidate.state=? AND candidate.available_at<=?
            AND (candidate.lease_expires_at IS NULL
                 OR candidate.lease_expires_at<=? OR candidate.lease_owner=?)
            {circuit_filter}
            ORDER BY CASE WHEN candidate.lease_owner=? THEN 0 ELSE 1 END,
            (SELECT COUNT(*) FROM source_jobs AS active
             WHERE active.platform=candidate.platform
               AND active.state=candidate.state
               AND active.lease_expires_at>?) /
            MAX(COALESCE(CAST(json_extract(
                ?, '$.' || candidate.platform
            ) AS REAL), 1.0), 0.01) ASC,
            candidate.priority DESC, candidate.discovered_at,
            candidate.source_key LIMIT 1""",
            parameters,
        ).fetchone()
        if row is None:
            connection.commit()
            return None
        if respect_provider_circuits:
            circuit = connection.execute(
                "SELECT * FROM source_provider_circuits WHERE platform=?",
                (row["platform"],),
            ).fetchone()
            if circuit and circuit["state"] != "closed":
                connection.execute(
                    """UPDATE source_provider_circuits SET state='half_open',
                    probe_owner=?,probe_expires_at=?,updated_at=?
                    WHERE platform=?""",
                    (
                        worker,
                        timestamp + lease_seconds,
                        _now_iso(timestamp),
                        row["platform"],
                    ),
                )
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


def provider_circuit_record_success(
    connection: sqlite3.Connection,
    platform: str,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Close a provider circuit after a transport-level success."""
    timestamp = time.time() if now is None else now
    connection.execute(
        """INSERT INTO source_provider_circuits(
        platform,state,consecutive_failures,open_count,retry_at,probe_owner,
        probe_expires_at,last_error,last_success_at,updated_at)
        VALUES(?,'closed',0,0,NULL,NULL,NULL,NULL,?,?)
        ON CONFLICT(platform) DO UPDATE SET state='closed',
        consecutive_failures=0,open_count=0,retry_at=NULL,probe_owner=NULL,
        probe_expires_at=NULL,last_error=NULL,last_success_at=excluded.last_success_at,
        updated_at=excluded.updated_at""",
        (platform, _now_iso(timestamp), _now_iso(timestamp)),
    )
    return {"state": "closed", "platform": platform}


def provider_circuit_record_failure(
    connection: sqlite3.Connection,
    platform: str,
    error: str,
    *,
    failure_threshold: int = 5,
    cooldown_seconds: float = 300.0,
    max_cooldown_seconds: float = 3600.0,
    now: float | None = None,
) -> dict[str, Any]:
    """Record a transport failure and open or re-open a provider circuit."""
    if failure_threshold < 1:
        raise ValueError("failure_threshold must be positive")
    timestamp = time.time() if now is None else now
    current = connection.execute(
        "SELECT * FROM source_provider_circuits WHERE platform=?", (platform,)
    ).fetchone()
    failures = int(current["consecutive_failures"] if current else 0) + 1
    was_probe = bool(current and current["state"] == "half_open")
    should_open = was_probe or failures >= failure_threshold
    open_count = int(current["open_count"] if current else 0)
    retry_at: float | None = None
    state = "closed"
    if should_open:
        open_count += 1
        delay = min(
            max_cooldown_seconds,
            cooldown_seconds * (2 ** max(0, open_count - 1)),
        )
        retry_at = timestamp + max(0.0, delay)
        state = "open"
    connection.execute(
        """INSERT INTO source_provider_circuits(
        platform,state,consecutive_failures,open_count,retry_at,probe_owner,
        probe_expires_at,last_error,last_success_at,updated_at)
        VALUES(?,?,?,?,?,NULL,NULL,?,NULL,?)
        ON CONFLICT(platform) DO UPDATE SET state=excluded.state,
        consecutive_failures=excluded.consecutive_failures,
        open_count=excluded.open_count,retry_at=excluded.retry_at,
        probe_owner=NULL,probe_expires_at=NULL,last_error=excluded.last_error,
        updated_at=excluded.updated_at""",
        (
            platform,
            state,
            failures,
            open_count,
            retry_at,
            error[-2_000:],
            _now_iso(timestamp),
        ),
    )
    return {
        "platform": platform,
        "state": state,
        "consecutive_failures": failures,
        "retry_at": retry_at,
    }


def provider_circuit_snapshot(
    connection: sqlite3.Connection, *, now: float | None = None
) -> dict[str, dict[str, Any]]:
    timestamp = time.time() if now is None else now
    return {
        str(row["platform"]): {
            "state": str(row["state"]),
            "consecutive_failures": int(row["consecutive_failures"]),
            "open_count": int(row["open_count"]),
            "retry_at": row["retry_at"],
            "retry_in_seconds": (
                round(max(0.0, float(row["retry_at"]) - timestamp), 3)
                if row["retry_at"] is not None and row["state"] == "open"
                else None
            ),
            "last_error": row["last_error"],
            "last_success_at": row["last_success_at"],
            "updated_at": row["updated_at"],
        }
        for row in connection.execute(
            "SELECT * FROM source_provider_circuits ORDER BY platform"
        )
    }


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


def frontier_platform_counts(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, int]]:
    """Return every queue/terminal state grouped by source provider."""
    platforms: dict[str, Counter[str]] = {}
    for row in connection.execute(
        "SELECT platform,state,COUNT(*) AS count FROM source_jobs "
        "GROUP BY platform,state ORDER BY platform,state"
    ):
        platforms.setdefault(str(row["platform"]), Counter())[str(row["state"])] = int(
            row["count"]
        )
    return {
        platform: {state: counts[state] for state in ALL_STATES}
        for platform, counts in platforms.items()
    }


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


def _host_receive_rate(timestamp: float) -> dict[str, float | None]:
    """Sample non-loopback host ingress for the live downloader dashboard."""
    try:
        received_bytes = sum(
            int(payload.split()[0])
            for line in Path("/proc/net/dev").read_text().splitlines()
            if ":" in line
            for interface, payload in [line.split(":", 1)]
            if interface.strip() != "lo"
        )
    except (FileNotFoundError, OSError, ValueError):
        return {"receive_megabytes_per_second": None, "observed_seconds": 0.0}
    with _network_samples_lock:
        if _network_samples and received_bytes < _network_samples[-1][1]:
            _network_samples.clear()
        if not _network_samples or timestamp > _network_samples[-1][0]:
            _network_samples.append((timestamp, received_bytes))
        cutoff = timestamp - _NETWORK_WINDOW_SECONDS
        while len(_network_samples) > 2 and _network_samples[1][0] <= cutoff:
            _network_samples.popleft()
        if len(_network_samples) < 2:
            return {"receive_megabytes_per_second": None, "observed_seconds": 0.0}
        started_at, started_bytes = _network_samples[0]
        finished_at, finished_bytes = _network_samples[-1]
    observed = max(0.0, finished_at - started_at)
    rate = (
        max(0, finished_bytes - started_bytes) / observed / 1_000_000.0
        if observed
        else None
    )
    return {
        "receive_megabytes_per_second": round(rate, 3) if rate is not None else None,
        "observed_seconds": round(observed, 1),
    }


def frontier_snapshot(
    workspace: Path,
    *,
    window_minutes: float = 15.0,
    now: float | None = None,
    catalog_path: Path | None = None,
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
    platform_events = connection.execute(
        """SELECT j.platform,e.stage,e.outcome,e.duration_seconds,e.details_json,
        json_extract(j.candidate_json,'$[0].duration_seconds') AS source_duration
        FROM source_stage_events e JOIN source_jobs j USING(source_key)
        WHERE e.finished_at>=?""",
        (cutoff,),
    ).fetchall()
    stage_metrics: dict[str, dict[str, Any]] = {}
    for stage in STAGE_FOR_STATE.values():
        selected = [row for row in events if row["stage"] == stage]
        active = [
            row for row in selected if not str(row["outcome"]).startswith("cache_")
        ]
        durations = [float(row["duration_seconds"]) for row in active]
        outcomes = Counter(str(row["outcome"]) for row in selected)
        stage_metrics[stage] = {
            "events": len(selected),
            "per_minute": round(len(selected) / max(window_minutes, 1e-9), 4),
            "active_events": len(active),
            "active_per_minute": round(len(active) / max(window_minutes, 1e-9), 4),
            "outcomes": dict(outcomes),
            "duration_p50_seconds": _percentile(durations, 0.5),
            "duration_p95_seconds": _percentile(durations, 0.95),
        }
    platform_metrics: dict[str, dict[str, Any]] = {}
    platform_counts = frontier_platform_counts(connection)
    circuits = provider_circuit_snapshot(connection, now=timestamp)
    discovery_strategies = discovery_strategy_snapshot(
        connection, catalog_path=catalog_path
    )
    for platform in sorted(platform_counts):
        downloads = [
            row
            for row in platform_events
            if row["platform"] == platform
            and row["stage"] == "download"
            and not str(row["outcome"]).startswith("cache_")
        ]
        successes = [
            row for row in downloads if row["outcome"] in {"success", "recovered"}
        ]
        transferred_bytes = 0
        transfer_seconds = 0.0
        source_seconds = 0.0
        for row in successes:
            try:
                details = json.loads(row["details_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                details = {}
            if row["outcome"] == "success" and details.get("download_seconds"):
                transferred_bytes += int(details.get("bytes") or 0)
                transfer_seconds += float(details["download_seconds"])
            source_seconds += float(row["source_duration"] or 0.0)
        attempts = len(downloads)
        platform_metrics[platform] = {
            "states": platform_counts[platform],
            "circuit": circuits.get(
                platform,
                {
                    "state": "closed",
                    "consecutive_failures": 0,
                    "retry_in_seconds": None,
                },
            ),
            "download_attempts": attempts,
            "download_attempts_per_minute": round(
                attempts / max(window_minutes, 1e-9), 4
            ),
            "download_successes": len(successes),
            "download_success_percent": round(100.0 * len(successes) / attempts, 2)
            if attempts
            else 0.0,
            "download_megabytes_per_second": round(
                transferred_bytes / 1_000_000.0 / transfer_seconds, 3
            )
            if transfer_seconds
            else 0.0,
            "source_audio_hours_per_wall_hour": round(
                source_seconds / 3600.0 / max(window_minutes / 60.0, 1e-9), 3
            ),
            "downloaded_bytes": transferred_bytes,
        }
    completed_download_bytes = sum(
        int(metrics["downloaded_bytes"]) for metrics in platform_metrics.values()
    )
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
        "download_throughput": {
            "completed_file_megabytes_per_second": round(
                completed_download_bytes
                / max(window_minutes * 60.0, 1e-9)
                / 1_000_000.0,
                3,
            ),
            "host_network": _host_receive_rate(timestamp)
            if now is None
            else {
                "receive_megabytes_per_second": None,
                "observed_seconds": 0.0,
            },
        },
        "leased": {row["state"]: int(row["count"]) for row in leased},
        "oldest_ready_minutes": oldest,
        "stages": stage_metrics,
        "platforms": platform_metrics,
        "discovery_strategies": discovery_strategies,
        "workers": workers,
        "autoscaler": autoscaler,
    }
