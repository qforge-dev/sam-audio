"""Permanent producer/consumer catalog and immutable dataset snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .m2d_validator import (
    ASR_POLICY_VERSION,
    CINEMATIC_POLICY_VERSION,
    _enforce_current_voice_gate,
)
from .youtube_random import _candidate_allowed, analyze_wav, quality_rejections

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1
CATALOG_FILENAME = "catalog.sqlite3"
MANIFEST_FILENAME = "manifest.json"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    os.replace(temporary, path)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def connect(workspace: Path) -> sqlite3.Connection:
    workspace.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(workspace / CATALOG_FILENAME, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS records (
            sha256 TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL UNIQUE,
            filename TEXT NOT NULL UNIQUE,
            platform TEXT NOT NULL,
            video_id TEXT NOT NULL,
            clip_start REAL NOT NULL,
            record_json TEXT NOT NULL,
            discovered_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS m2d_scores (
            filename TEXT PRIMARY KEY,
            accepted INTEGER NOT NULL,
            result_json TEXT NOT NULL,
            scored_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS asr_scores (
            filename TEXT PRIMARY KEY,
            accepted INTEGER NOT NULL,
            result_json TEXT NOT NULL,
            scored_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS accepted (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            sha256 TEXT NOT NULL UNIQUE REFERENCES records(sha256),
            accepted_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS rejected (
            sha256 TEXT PRIMARY KEY REFERENCES records(sha256),
            reason TEXT NOT NULL,
            rejected_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS offsets (
            path TEXT PRIMARY KEY,
            inode INTEGER NOT NULL,
            byte_offset INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS snapshots (
            record_count INTEGER PRIMARY KEY,
            snapshot_id TEXT NOT NULL UNIQUE,
            published_at TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            s3_prefix TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS workers (
            worker TEXT PRIMARY KEY,
            state TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            details_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS acquisition_manifests (
            path TEXT PRIMARY KEY,
            signature TEXT NOT NULL,
            record_count INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    return connection


def _heartbeat(
    connection: sqlite3.Connection,
    worker: str,
    state: str = "running",
    **details: Any,
) -> None:
    connection.execute(
        """INSERT INTO workers(worker,state,updated_at,details_json)
           VALUES(?,?,?,?) ON CONFLICT(worker) DO UPDATE SET
           state=excluded.state, updated_at=excluded.updated_at,
           details_json=excluded.details_json""",
        (worker, state, _now(), json.dumps(details, separators=(",", ":"))),
    )
    connection.commit()


def _safe_manifest(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def promote_once(runs_dir: Path, workspace: Path) -> int:
    """Publish completed quality-gated WAVs into the stable raw queue."""
    raw_dir = workspace / "raw-audio"
    connection = connect(workspace)
    added = 0
    manifests = sorted(runs_dir.glob("run-*/manifest.json"))
    for manifest_path in manifests:
        try:
            stat = manifest_path.stat()
        except FileNotFoundError:
            continue
        manifest_key = str(manifest_path.resolve())
        signature = f"{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}"
        cached = connection.execute(
            "SELECT signature FROM acquisition_manifests WHERE path=?",
            (manifest_key,),
        ).fetchone()
        if cached and cached["signature"] == signature:
            continue
        manifest = _safe_manifest(manifest_path)
        if not manifest:
            continue
        run_dir = manifest_path.parent
        for source_record in manifest.get("records", []):
            if source_record.get("retrieval_status") != "success":
                continue
            if source_record.get("quality_rejections"):
                continue
            if not _candidate_allowed(source_record, profile="cinematic"):
                continue
            source = run_dir / str(source_record.get("local_path", ""))
            if not source.is_file():
                continue
            start_milliseconds = round(
                float(source_record.get("clip_start_seconds", 0)) * 1000
            )
            candidate_id = str(
                source_record.get("candidate_id")
                or f"{source_record.get('video_id')}:{start_milliseconds}"
            )
            declared_digest = str(source_record.get("sha256") or "")
            known = connection.execute(
                "SELECT 1 FROM records WHERE sha256=? OR candidate_id=? LIMIT 1",
                (declared_digest, candidate_id),
            ).fetchone()
            if known:
                continue
            actual_digest = _sha256(source)
            digest = declared_digest or actual_digest
            if len(digest) != 64 or actual_digest != digest:
                logger.warning("Skipping hash mismatch: %s", source)
                continue
            filename = f"{digest}.wav"
            destination = raw_dir / filename
            record = dict(source_record)
            record.update(
                {
                    "sha256": digest,
                    "bytes": source.stat().st_size,
                    "local_path": f"raw-audio/{filename}",
                    "continuous_filename": filename,
                    "continuous_ingested_at": _now(),
                }
            )
            try:
                with connection:
                    cursor = connection.execute(
                        """INSERT OR IGNORE INTO records(
                        sha256,candidate_id,filename,platform,video_id,clip_start,
                        record_json,discovered_at) VALUES(?,?,?,?,?,?,?,?)""",
                        (
                            digest,
                            candidate_id,
                            filename,
                            str(record.get("source_platform") or "unknown"),
                            str(record.get("video_id") or "unknown"),
                            float(record.get("clip_start_seconds") or 0.0),
                            json.dumps(record, separators=(",", ":")),
                            _now(),
                        ),
                    )
                    if cursor.rowcount:
                        _link_or_copy(source, destination)
                        added += 1
            except sqlite3.IntegrityError:
                continue
        with connection:
            connection.execute(
                """INSERT INTO acquisition_manifests(
                path,signature,record_count,updated_at)
                VALUES(?,?,?,?) ON CONFLICT(path) DO UPDATE SET
                signature=excluded.signature,record_count=excluded.record_count,
                updated_at=excluded.updated_at""",
                (
                    manifest_key,
                    signature,
                    len(manifest.get("records", [])),
                    _now(),
                ),
            )
    total = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    _heartbeat(connection, "promoter", manifests=len(manifests), raw_records=total)
    connection.close()
    return added


def _tail_jsonl(connection: sqlite3.Connection, path: Path, table: str) -> int:
    if not path.exists():
        return 0
    stat = path.stat()
    key = str(path.resolve())
    row = connection.execute(
        "SELECT inode,byte_offset FROM offsets WHERE path=?", (key,)
    ).fetchone()
    offset = int(row["byte_offset"]) if row and row["inode"] == stat.st_ino else 0
    processed = 0
    with path.open("rb") as source:
        source.seek(offset)
        while True:
            line_start = source.tell()
            line = source.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                source.seek(line_start)
                break
            try:
                result = json.loads(line)
                filename = str(result["filename"])
            except (json.JSONDecodeError, KeyError, TypeError, UnicodeError):
                continue
            if table == "m2d_scores":
                result = _enforce_current_voice_gate(result, require_cinematic_mix=True)
            connection.execute(
                f"""INSERT INTO {table}(filename,accepted,result_json,scored_at)
                VALUES(?,?,?,?) ON CONFLICT(filename) DO UPDATE SET
                accepted=excluded.accepted,result_json=excluded.result_json,
                scored_at=excluded.scored_at""",
                (
                    filename,
                    int(bool(result.get("accepted"))),
                    json.dumps(result, separators=(",", ":")),
                    str(result.get("scored_at") or _now()),
                ),
            )
            processed += 1
        new_offset = source.tell()
    connection.execute(
        """INSERT INTO offsets(path,inode,byte_offset) VALUES(?,?,?)
        ON CONFLICT(path) DO UPDATE SET inode=excluded.inode,
        byte_offset=excluded.byte_offset""",
        (key, stat.st_ino, new_offset),
    )
    return processed


def catalog_records(
    connection: sqlite3.Connection,
    *,
    start_sequence: int | None = None,
    end_sequence: int | None = None,
    limit: int | None = None,
    newest_first: bool = False,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    parameters: list[Any] = []
    if start_sequence is not None:
        clauses.append("a.sequence>=?")
        parameters.append(start_sequence)
    if end_sequence is not None:
        clauses.append("a.sequence<=?")
        parameters.append(end_sequence)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    order = "DESC" if newest_first else "ASC"
    limit_sql = " LIMIT ?" if limit is not None else ""
    if limit is not None:
        parameters.append(limit)
    rows = connection.execute(
        """SELECT a.sequence,a.accepted_at,r.record_json,m.result_json AS m2d_json,
        s.result_json AS asr_json FROM accepted a JOIN records r USING(sha256)
        JOIN m2d_scores m USING(filename) JOIN asr_scores s USING(filename)
        """
        + where
        + f" ORDER BY a.sequence {order}"
        + limit_sql,
        parameters,
    ).fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        record = json.loads(row["record_json"])
        filename = str(record["continuous_filename"])
        record.update(
            {
                "record_index": int(row["sequence"]) - 1,
                "catalog_sequence": int(row["sequence"]),
                "local_path": f"audio/{filename}",
                "accepted_at": row["accepted_at"],
                "m2d_validation": json.loads(row["m2d_json"]),
                "asr_validation": json.loads(row["asr_json"]),
            }
        )
        records.append(record)
    return records


def write_live_manifest(
    workspace: Path, connection: sqlite3.Connection
) -> dict[str, Any]:
    accepted_count = connection.execute("SELECT COUNT(*) FROM accepted").fetchone()[0]
    unique_source_count = connection.execute(
        """SELECT COUNT(*) FROM (
        SELECT DISTINCT r.platform,r.video_id FROM accepted a
        JOIN records r USING(sha256))"""
    ).fetchone()[0]
    first_accepted = connection.execute(
        "SELECT MIN(accepted_at) FROM accepted"
    ).fetchone()[0]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": "Continuous cinematic dialogue + instrumental music + SFX (30 s)",
        "created_at": first_accepted or _now(),
        "updated_at": _now(),
        "continuous": True,
        "clip_seconds": 30.0,
        "accepted_record_count": accepted_count,
        "maximum_clips_per_source_video": 24,
        "unique_source_video_count": unique_source_count,
        "m2d_policy": CINEMATIC_POLICY_VERSION,
        "foreground_voice_policy": ASR_POLICY_VERSION,
        "explicit_metadata_policy": "cinematic_source_exclusions_v1",
        "manual_review_is_acceptance_gate": False,
        "audio_directory": "audio",
        "catalog": "../catalog.sqlite3",
        "review_window_records": 5000,
        "records": [],
    }
    _atomic_json(workspace / "accepted" / MANIFEST_FILENAME, manifest)
    return manifest


def assemble_once(workspace: Path, max_clips_per_video: int = 24) -> int:
    connection = connect(workspace)
    with connection:
        m2d_paths = [workspace / "m2d-validation.jsonl"] + sorted(
            (workspace / "m2d-validation").glob("*.jsonl")
        )
        asr_paths = [workspace / "asr-validation.jsonl"] + sorted(
            (workspace / "asr-validation").glob("*.jsonl")
        )
        for path in m2d_paths:
            _tail_jsonl(connection, path, "m2d_scores")
        for path in asr_paths:
            _tail_jsonl(connection, path, "asr_scores")
    candidates = connection.execute(
        """SELECT r.*,m.accepted AS m2d_accepted,s.accepted AS asr_accepted
        FROM records r JOIN m2d_scores m USING(filename)
        JOIN asr_scores s USING(filename) LEFT JOIN accepted a USING(sha256)
        LEFT JOIN rejected x USING(sha256)
        WHERE a.sha256 IS NULL AND x.sha256 IS NULL ORDER BY r.discovered_at"""
    ).fetchall()
    added = 0
    for row in candidates:
        if not row["m2d_accepted"] or not row["asr_accepted"]:
            with connection:
                connection.execute(
                    "INSERT OR IGNORE INTO rejected VALUES(?,?,?)",
                    (row["sha256"], "automated_model_gate", _now()),
                )
            continue
        count = connection.execute(
            """SELECT COUNT(*) FROM accepted a JOIN records r USING(sha256)
            WHERE r.platform=? AND r.video_id=?""",
            (row["platform"], row["video_id"]),
        ).fetchone()[0]
        overlap = connection.execute(
            """SELECT 1 FROM accepted a JOIN records r USING(sha256)
            WHERE r.platform=? AND r.video_id=? AND ABS(r.clip_start-?)<30 LIMIT 1""",
            (row["platform"], row["video_id"], row["clip_start"]),
        ).fetchone()
        if count >= max_clips_per_video or overlap:
            reason = "source_video_cap" if count >= max_clips_per_video else "overlap"
            with connection:
                connection.execute(
                    "INSERT OR IGNORE INTO rejected VALUES(?,?,?)",
                    (row["sha256"], reason, _now()),
                )
            continue
        source = workspace / "raw-audio" / row["filename"]
        destination = workspace / "accepted" / "audio" / row["filename"]
        if not source.is_file() or _sha256(source) != row["sha256"]:
            continue
        _link_or_copy(source, destination)
        with connection:
            connection.execute(
                "INSERT OR IGNORE INTO accepted(sha256,accepted_at) VALUES(?,?)",
                (row["sha256"], _now()),
            )
        added += 1
    if added or not (workspace / "accepted" / MANIFEST_FILENAME).exists():
        write_live_manifest(workspace, connection)
    counts = pipeline_counts(connection)
    _heartbeat(connection, "assembler", **counts)
    connection.close()
    return added


def pipeline_counts(connection: sqlite3.Connection) -> dict[str, int]:
    queries = {
        "downloaded": "SELECT COUNT(*) FROM records",
        "m2d_scored": "SELECT COUNT(*) FROM m2d_scores",
        "m2d_accepted": "SELECT COUNT(*) FROM m2d_scores WHERE accepted=1",
        "asr_scored": "SELECT COUNT(*) FROM asr_scores",
        "asr_accepted": "SELECT COUNT(*) FROM asr_scores WHERE accepted=1",
        "accepted": "SELECT COUNT(*) FROM accepted",
        "rejected": "SELECT COUNT(*) FROM rejected",
        "snapshots": "SELECT COUNT(*) FROM snapshots",
    }
    counts = {
        name: connection.execute(query).fetchone()[0] for name, query in queries.items()
    }
    counts.update(
        {
            "m2d_rejected": counts["m2d_scored"] - counts["m2d_accepted"],
            "asr_rejected": counts["asr_scored"] - counts["asr_accepted"],
            "assembly_rejected": connection.execute(
                "SELECT COUNT(*) FROM rejected WHERE reason!='automated_model_gate'"
            ).fetchone()[0],
        }
    )
    counts["rejected_total"] = (
        counts["m2d_rejected"] + counts["asr_rejected"] + counts["assembly_rejected"]
    )
    return counts


def _throughput(
    connection: sqlite3.Connection,
    *,
    table: str,
    timestamp_column: str,
    clip_seconds: float,
    window_minutes: float,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    cutoff = now - timedelta(minutes=window_minutes)
    row = connection.execute(
        f"SELECT COUNT(*),MIN({timestamp_column}) FROM {table} "
        f"WHERE {timestamp_column}>=?",
        (cutoff.isoformat(),),
    ).fetchone()
    count = int(row[0])
    oldest = datetime.fromisoformat(row[1]) if row[1] else None
    observed_minutes = (
        max(1.0, min(window_minutes, (now - oldest).total_seconds() / 60.0))
        if oldest
        else window_minutes
    )
    clips_per_minute = count / observed_minutes
    audio_minutes_per_minute = clips_per_minute * clip_seconds / 60.0
    return {
        "rolling_window_minutes": window_minutes,
        "observed_minutes": round(observed_minutes, 3),
        "clips": count,
        "clips_per_minute": round(clips_per_minute, 4),
        "audio_minutes_per_minute": round(audio_minutes_per_minute, 4),
        "audio_hours_per_hour": round(audio_minutes_per_minute, 4),
    }


def progress_snapshot(
    workspace: Path,
    snapshot_size: int = 5000,
    *,
    throughput_window_minutes: float = 60.0,
    target_hours: float = 10_000.0,
) -> dict[str, Any]:
    connection = connect(workspace)
    counts = pipeline_counts(connection)
    workers = []
    now = datetime.now(UTC)
    for row in connection.execute("SELECT * FROM workers ORDER BY worker"):
        updated = datetime.fromisoformat(row["updated_at"])
        workers.append(
            {
                "worker": row["worker"],
                "state": row["state"]
                if (now - updated).total_seconds() < 30
                else "stale",
                "updated_at": row["updated_at"],
                "details": json.loads(row["details_json"]),
            }
        )
    published = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM snapshots ORDER BY record_count DESC"
        )
    ]
    accepted = counts["accepted"]
    next_snapshot = (accepted // snapshot_size + 1) * snapshot_size
    clip_seconds = 30.0
    throughput = {
        "download": _throughput(
            connection,
            table="records",
            timestamp_column="discovered_at",
            clip_seconds=clip_seconds,
            window_minutes=throughput_window_minutes,
        ),
        "m2d": _throughput(
            connection,
            table="m2d_scores",
            timestamp_column="scored_at",
            clip_seconds=clip_seconds,
            window_minutes=throughput_window_minutes,
        ),
        "asr": _throughput(
            connection,
            table="asr_scores",
            timestamp_column="scored_at",
            clip_seconds=clip_seconds,
            window_minutes=throughput_window_minutes,
        ),
        "accepted": _throughput(
            connection,
            table="accepted",
            timestamp_column="accepted_at",
            clip_seconds=clip_seconds,
            window_minutes=throughput_window_minutes,
        ),
    }
    current_hours = accepted * clip_seconds / 3600.0
    accepted_rate = throughput["accepted"]["audio_hours_per_hour"]
    remaining_hours = max(0.0, target_hours - current_hours)
    eta_wall_hours = remaining_hours / accepted_rate if accepted_rate > 0 else None
    config_path = workspace / "config.json"
    try:
        worker_config = json.loads(config_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        worker_config = {}
    try:
        autoscaler = json.loads((workspace / "autoscaler.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        autoscaler = {"enabled": False, "state": "not_started"}
    queues = {
        "waiting_for_m2d": max(0, counts["downloaded"] - counts["m2d_scored"]),
        "waiting_for_asr": max(0, counts["m2d_accepted"] - counts["asr_scored"]),
        "waiting_for_assembly": max(0, counts["asr_accepted"] - counts["accepted"]),
    }
    stalled_stages = [
        stage
        for stage, backlog, recent_clips in (
            ("m2d", queues["waiting_for_m2d"], throughput["m2d"]["clips"]),
            ("asr", queues["waiting_for_asr"], throughput["asr"]["clips"]),
            (
                "assembly",
                queues["waiting_for_assembly"],
                throughput["accepted"]["clips"],
            ),
        )
        if backlog > 0 and recent_clips == 0
    ]
    unhealthy_workers = [
        worker["worker"] for worker in workers if worker["state"] != "running"
    ]
    constrained_by = str(autoscaler.get("bottleneck") or "")
    if stalled_stages or unhealthy_workers:
        flow_state = "stalled"
        flow_explanation = (
            "A pending queue has no recent consumer output or a worker is unhealthy."
        )
    elif constrained_by in {"cpu", "m2d", "asr"}:
        flow_state = "constrained"
        flow_explanation = (
            f"Work is still flowing, but the autoscaler is managing "
            f"{constrained_by} pressure. This is constrained, not stalled."
        )
    else:
        flow_state = "balanced"
        flow_explanation = (
            "Workers are healthy and every pending queue has recent consumer output. "
            "The processing/output gap is filtering yield, not stuck work."
        )
    processed_rate = float(throughput["download"]["audio_hours_per_hour"])
    output_rate = float(throughput["accepted"]["audio_hours_per_hour"])
    rolling_yield = output_rate / processed_rate * 100.0 if processed_rate else 0.0
    payload = {
        "mode": "continuous",
        "updated_at": _now(),
        "clip_seconds": clip_seconds,
        "counts": counts,
        "queues": queues,
        "flow": {
            "state": flow_state,
            "explanation": flow_explanation,
            "processed_audio_hours_per_wall_hour": processed_rate,
            "accepted_audio_hours_per_wall_hour": output_rate,
            "rolling_yield_percent": round(rolling_yield, 2),
            "window_minutes": throughput_window_minutes,
            "stalled_stages": stalled_stages,
            "unhealthy_workers": unhealthy_workers,
        },
        "next_snapshot": {
            "record_count": next_snapshot,
            "remaining": next_snapshot - accepted,
            "progress": accepted % snapshot_size,
            "size": snapshot_size,
        },
        "workers": workers,
        "worker_config": worker_config,
        "autoscaler": autoscaler,
        "throughput": throughput,
        "goal": {
            "target_audio_hours": target_hours,
            "current_audio_hours": round(current_hours, 4),
            "remaining_audio_hours": round(remaining_hours, 4),
            "accepted_audio_hours_per_wall_hour": accepted_rate,
            "estimated_wall_hours_remaining": (
                round(eta_wall_hours, 2) if eta_wall_hours is not None else None
            ),
            "estimated_days_remaining": (
                round(eta_wall_hours / 24.0, 2) if eta_wall_hours is not None else None
            ),
            "estimated_completion_at": (
                (datetime.now(UTC) + timedelta(hours=eta_wall_hours)).isoformat()
                if eta_wall_hours is not None
                else None
            ),
            "estimate_basis": (
                f"rolling {throughput_window_minutes:g}-minute accepted throughput"
            ),
        },
        "published_snapshots": published,
    }
    connection.close()
    return payload


def _cpu_percent(sample_seconds: float = 0.5) -> float:
    def counters() -> tuple[int, int]:
        values = [
            int(value)
            for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        ]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle

    total_before, idle_before = counters()
    time.sleep(sample_seconds)
    total_after, idle_after = counters()
    total_delta = max(1, total_after - total_before)
    busy = total_delta - (idle_after - idle_before)
    return round(max(0.0, min(100.0, busy * 100.0 / total_delta)), 2)


def _gpu_status() -> dict[str, float | None]:
    try:
        response = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        utilization, used, free = (
            float(value.strip()) for value in response.stdout.splitlines()[0].split(",")
        )
        return {
            "utilization_percent": utilization,
            "memory_used_mb": used,
            "memory_free_mb": free,
        }
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return {
            "utilization_percent": None,
            "memory_used_mb": None,
            "memory_free_mb": None,
        }


def autoscale_decision(
    *,
    download_concurrency: int,
    asr_concurrency: int,
    m2d_backlog: int,
    asr_backlog: int,
    cpu_percent: float,
    gpu_free_mb: float | None,
    download_min: int,
    download_max: int,
    asr_min: int,
    asr_max: int,
    cpu_low: float,
    cpu_high: float,
    m2d_backlog_high: int,
    asr_backlog_high: int,
    gpu_reserve_mb: float,
) -> dict[str, Any]:
    """Choose one conservative scale step using queues and resource headroom."""
    download = download_concurrency
    asr = asr_concurrency
    actions: list[str] = []
    if asr_backlog >= asr_backlog_high:
        bottleneck = "asr"
    elif m2d_backlog >= m2d_backlog_high:
        bottleneck = "m2d"
    elif cpu_percent >= cpu_high:
        bottleneck = "cpu"
    elif m2d_backlog == 0 and asr_backlog == 0:
        bottleneck = "source_yield"
    else:
        bottleneck = "balanced"

    if cpu_percent >= cpu_high and download > download_min:
        download -= 1
        actions.append("reduce_download_for_cpu")
    elif asr_backlog >= asr_backlog_high:
        gpu_has_room = gpu_free_mb is not None and gpu_free_mb >= gpu_reserve_mb
        if cpu_percent < cpu_high and gpu_has_room and asr < asr_max:
            asr += 1
            actions.append("increase_asr")
        elif download > download_min:
            download -= 1
            actions.append("reduce_download_for_asr")
    elif m2d_backlog >= m2d_backlog_high and download > download_min:
        download -= 1
        actions.append("reduce_download_for_m2d")
    elif asr_backlog == 0 and asr > asr_min:
        asr -= 1
        actions.append("decrease_idle_asr")
    elif (
        m2d_backlog == 0
        and asr_backlog == 0
        and cpu_percent <= cpu_low
        and download < download_max
    ):
        download += 1
        actions.append("increase_download")
    return {
        "download_concurrency": max(download_min, min(download_max, download)),
        "asr_concurrency": max(asr_min, min(asr_max, asr)),
        "bottleneck": bottleneck,
        "actions": actions,
    }


def run_autoscaler_once(args: argparse.Namespace) -> dict[str, Any]:
    workspace = args.workspace
    control_path = workspace / "autoscale-control.json"
    status_path = workspace / "autoscaler.json"
    control_exists = control_path.is_file()
    try:
        control = json.loads(control_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        control = {
            "download_concurrency": args.download_max,
            "asr_concurrency": args.asr_min,
        }
    connection = connect(workspace)
    counts = pipeline_counts(connection)
    cpu = _cpu_percent()
    gpu = _gpu_status()
    m2d_backlog = max(0, counts["downloaded"] - counts["m2d_scored"])
    asr_backlog = max(0, counts["m2d_accepted"] - counts["asr_scored"])
    decision = autoscale_decision(
        download_concurrency=int(
            control.get("download_concurrency", args.download_max)
        ),
        asr_concurrency=int(control.get("asr_concurrency", args.asr_min)),
        m2d_backlog=m2d_backlog,
        asr_backlog=asr_backlog,
        cpu_percent=cpu,
        gpu_free_mb=gpu["memory_free_mb"],
        download_min=args.download_min,
        download_max=args.download_max,
        asr_min=args.asr_min,
        asr_max=args.asr_max,
        cpu_low=args.cpu_low,
        cpu_high=args.cpu_high,
        m2d_backlog_high=args.m2d_backlog_high,
        asr_backlog_high=args.asr_backlog_high,
        gpu_reserve_mb=args.gpu_reserve_mb,
    )
    now = datetime.now(UTC)
    last_changed_at = control.get("changed_at")
    elapsed = (
        (now - datetime.fromisoformat(last_changed_at)).total_seconds()
        if last_changed_at
        else float("inf")
    )
    emergency = cpu >= args.cpu_emergency
    apply_change = bool(decision["actions"]) and (
        elapsed >= args.cooldown_seconds or emergency
    )
    if apply_change:
        previous = {
            "download_concurrency": int(
                control.get("download_concurrency", args.download_max)
            ),
            "asr_concurrency": int(control.get("asr_concurrency", args.asr_min)),
        }
        control = {
            "schema_version": 1,
            "download_concurrency": decision["download_concurrency"],
            "asr_concurrency": decision["asr_concurrency"],
            "changed_at": now.isoformat(),
            "actions": decision["actions"],
        }
        _atomic_json(control_path, control)
        with (workspace / "autoscaler-events.jsonl").open("a") as events:
            events.write(
                json.dumps(
                    {
                        "at": now.isoformat(),
                        "previous": previous,
                        "current": control,
                        "cpu_percent": cpu,
                        "gpu": gpu,
                        "m2d_backlog": m2d_backlog,
                        "asr_backlog": asr_backlog,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    elif not control_exists:
        control = {
            "schema_version": 1,
            "download_concurrency": int(
                control.get("download_concurrency", args.download_max)
            ),
            "asr_concurrency": int(control.get("asr_concurrency", args.asr_min)),
            "changed_at": now.isoformat(),
            "actions": ["initialize"],
        }
        _atomic_json(control_path, control)
    throughputs = {
        "download": _throughput(
            connection,
            table="records",
            timestamp_column="discovered_at",
            clip_seconds=30.0,
            window_minutes=15.0,
        ),
        "asr": _throughput(
            connection,
            table="asr_scores",
            timestamp_column="scored_at",
            clip_seconds=30.0,
            window_minutes=15.0,
        ),
        "accepted": _throughput(
            connection,
            table="accepted",
            timestamp_column="accepted_at",
            clip_seconds=30.0,
            window_minutes=15.0,
        ),
    }
    status = {
        "enabled": True,
        "state": "running",
        "observed_at": now.isoformat(),
        "bottleneck": decision["bottleneck"],
        "cpu_percent": cpu,
        "gpu": gpu,
        "queues": {"m2d": m2d_backlog, "asr": asr_backlog},
        "limits": {
            "download_concurrency": int(
                control.get("download_concurrency", args.download_max)
            ),
            "asr_concurrency": int(control.get("asr_concurrency", args.asr_min)),
        },
        "bounds": {
            "download": [args.download_min, args.download_max],
            "asr": [args.asr_min, args.asr_max],
        },
        "decision": (
            decision["actions"]
            if apply_change
            else (["cooldown"] if decision["actions"] else ["hold"])
        ),
        "cooldown_remaining_seconds": max(
            0.0, round(args.cooldown_seconds - elapsed, 1)
        ),
        "throughput": throughputs,
    }
    _atomic_json(status_path, status)
    _heartbeat(
        connection,
        "autoscaler",
        bottleneck=status["bottleneck"],
        cpu_percent=cpu,
        **status["limits"],
    )
    connection.close()
    return status


def run_autoscaler(args: argparse.Namespace) -> None:
    while True:
        try:
            status = run_autoscaler_once(args)
            logger.info(
                "Autoscaler %s CPU %.1f%% queues=%s limits=%s decision=%s",
                status["bottleneck"],
                status["cpu_percent"],
                status["queues"],
                status["limits"],
                status["decision"],
            )
        except Exception:
            logger.exception("Autoscaler iteration failed")
        if not args.follow:
            return
        time.sleep(args.interval_seconds)


def verify_snapshot(
    dataset_dir: Path,
    *,
    expected_count: int | None = None,
    clip_seconds: float | None = None,
    max_clips_per_video: int = 24,
) -> dict[str, Any]:
    manifest = json.loads((dataset_dir / MANIFEST_FILENAME).read_text())
    records = manifest.get("records", [])
    expected = len(records) if expected_count is None else expected_count
    failures: list[dict[str, Any]] = []
    hashes: set[str] = set()
    candidates: set[str] = set()
    source_counts: Counter[tuple[str, str]] = Counter()
    duration = float(clip_seconds or manifest.get("clip_seconds") or 10.0)
    for index, record in enumerate(records):
        reasons: list[str] = []
        path = dataset_dir / str(record.get("local_path", ""))
        if not path.is_file():
            reasons.append("missing_audio")
        else:
            digest = _sha256(path)
            if digest != record.get("sha256"):
                reasons.append("sha256_mismatch")
            try:
                metrics = analyze_wav(path)
                reasons.extend(
                    quality_rejections(
                        metrics, record.get("source_format", {}), clip_seconds=duration
                    )
                )
            except (OSError, ValueError) as error:
                reasons.append(f"audio_error:{type(error).__name__}")
        digest = str(record.get("sha256") or "")
        candidate = str(record.get("candidate_id") or "")
        if digest in hashes:
            reasons.append("duplicate_sha256")
        if candidate in candidates:
            reasons.append("duplicate_candidate")
        hashes.add(digest)
        candidates.add(candidate)
        key = (
            str(record.get("source_platform") or "unknown"),
            str(record.get("video_id") or "unknown"),
        )
        source_counts[key] += 1
        if source_counts[key] > max_clips_per_video:
            reasons.append("source_video_cap")
        if not _candidate_allowed(record, profile="cinematic"):
            reasons.append("metadata_policy")
        m2d = record.get("m2d_validation", {})
        asr = record.get("asr_validation", {})
        if not m2d.get("accepted"):
            reasons.append("m2d_gate")
        if not asr.get("accepted"):
            reasons.append("asr_gate")
        if asr.get("detected_language") != "en":
            reasons.append("language_gate")
        if reasons:
            failures.append({"record_index": index, "reasons": reasons})
    audio_files = list((dataset_dir / "audio").glob("*.wav"))
    audit = {
        "verified_at": _now(),
        "policy": "fully_automated_cinematic_snapshot_v1",
        "manual_checks_required": False,
        "expected_record_count": expected,
        "record_count": len(records),
        "audio_file_count": len(audio_files),
        "unique_sha256_count": len(hashes),
        "unique_candidate_count": len(candidates),
        "unique_source_video_count": len(source_counts),
        "clip_seconds": duration,
        "failure_count": len(failures),
        "failures": failures,
        "all_requirements_pass": (
            len(records) == expected
            and len(audio_files) == expected
            and len(hashes) == expected
            and len(candidates) == expected
            and not failures
        ),
    }
    _atomic_json(dataset_dir / "audit.json", audit)
    return audit


def _snapshot_manifest(
    workspace: Path, start_sequence: int, end_sequence: int, destination: Path
) -> dict[str, Any]:
    source_manifest = json.loads(
        (workspace / "accepted" / MANIFEST_FILENAME).read_text()
    )
    connection = connect(workspace)
    records = catalog_records(
        connection,
        start_sequence=start_sequence,
        end_sequence=end_sequence,
    )
    connection.close()
    expected = end_sequence - start_sequence + 1
    if len(records) != expected:
        raise RuntimeError(f"Only {len(records)} records available for snapshot shard")
    destination.mkdir(parents=True, exist_ok=True)
    audio = destination / "audio"
    audio.mkdir(exist_ok=True)
    copied: list[dict[str, Any]] = []
    selected_names: set[str] = set()
    for index, source_record in enumerate(records):
        record = dict(source_record)
        filename = Path(str(record["local_path"])).name
        selected_names.add(filename)
        _link_or_copy(
            workspace / "accepted" / str(record["local_path"]), audio / filename
        )
        record["snapshot_record_index"] = index
        record["local_path"] = f"audio/{filename}"
        copied.append(record)
    for stale in audio.glob("*.wav"):
        if stale.name not in selected_names:
            stale.unlink()
    snapshot = dict(source_manifest)
    snapshot.update(
        {
            "continuous": False,
            "snapshot_record_count": expected,
            "accepted_record_count": expected,
            "continuous_total_at_snapshot": end_sequence,
            "snapshot_sequence_start": start_sequence,
            "snapshot_sequence_end": end_sequence,
            "snapshot_created_at": _now(),
            "catalog": None,
            "records": copied,
        }
    )
    _atomic_json(destination / MANIFEST_FILENAME, snapshot)
    return snapshot


def publish_snapshot(
    dataset_dir: Path,
    *,
    bucket: str,
    prefix: str,
    snapshot_id: str,
    expected_count: int,
    clip_seconds: float,
    upload_concurrency: int = 10,
) -> dict[str, Any]:
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.exceptions import ClientError

    audit = verify_snapshot(
        dataset_dir,
        expected_count=expected_count,
        clip_seconds=clip_seconds,
    )
    if not audit["all_requirements_pass"]:
        raise RuntimeError("Refusing to publish a snapshot that failed verification")
    s3 = boto3.client("s3")
    transfer_config = TransferConfig(max_concurrency=max(1, upload_concurrency))
    root = prefix.strip("/")
    snapshot_prefix = f"{root}/snapshots/{snapshot_id}"
    ready_key = f"{snapshot_prefix}/READY.json"
    try:
        existing = s3.get_object(Bucket=bucket, Key=ready_key)
        return json.loads(existing["Body"].read())
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {"NoSuchKey", "404"}:
            raise
    records = json.loads((dataset_dir / MANIFEST_FILENAME).read_text())["records"]
    uploaded = 0
    for record in records:
        source = dataset_dir / str(record["local_path"])
        key = f"{root}/audio/{record['sha256']}.wav"
        try:
            s3.head_object(Bucket=bucket, Key=key)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey"}:
                raise
            s3.upload_file(str(source), bucket, key, Config=transfer_config)
            uploaded += 1
    manifest_path = dataset_dir / MANIFEST_FILENAME
    manifest_digest = _sha256(manifest_path)
    (dataset_dir / "manifest.sha256").write_text(
        f"{manifest_digest}  {MANIFEST_FILENAME}\n"
    )
    for name in (MANIFEST_FILENAME, "manifest.sha256", "audit.json"):
        s3.upload_file(
            str(dataset_dir / name),
            bucket,
            f"{snapshot_prefix}/{name}",
            Config=transfer_config,
        )
    ready = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "record_count": expected_count,
        "clip_seconds": clip_seconds,
        "manifest_sha256": manifest_digest,
        "audio_object_prefix": f"s3://{bucket}/{root}/audio/",
        "snapshot_uri": f"s3://{bucket}/{snapshot_prefix}/",
        "new_audio_objects_uploaded": uploaded,
        "published_at": _now(),
        "immutable": True,
    }
    temporary = dataset_dir / "READY.json"
    _atomic_json(temporary, ready)
    s3.upload_file(str(temporary), bucket, ready_key, Config=transfer_config)
    return ready


def publish_due_once(
    workspace: Path,
    *,
    bucket: str,
    prefix: str,
    snapshot_size: int = 5000,
    upload_concurrency: int = 10,
) -> int:
    """Publish every newly completed immutable snapshot boundary."""
    connection = connect(workspace)
    accepted = connection.execute("SELECT COUNT(*) FROM accepted").fetchone()[0]
    published = {
        row[0] for row in connection.execute("SELECT record_count FROM snapshots")
    }
    due = [
        count
        for count in range(snapshot_size, accepted + 1, snapshot_size)
        if count not in published
    ]
    completed = 0
    for count in due:
        start_sequence = count - snapshot_size + 1
        snapshot_id = f"v2-{start_sequence:08d}-{count:08d}"
        snapshot_dir = workspace / "snapshots" / snapshot_id
        _snapshot_manifest(workspace, start_sequence, count, snapshot_dir)
        ready = publish_snapshot(
            snapshot_dir,
            bucket=bucket,
            prefix=prefix,
            snapshot_id=snapshot_id,
            expected_count=snapshot_size,
            clip_seconds=30.0,
            upload_concurrency=upload_concurrency,
        )
        with connection:
            connection.execute(
                """INSERT OR IGNORE INTO snapshots(
                record_count,snapshot_id,published_at,manifest_sha256,s3_prefix
                ) VALUES(?,?,?,?,?)""",
                (
                    count,
                    snapshot_id,
                    ready["published_at"],
                    ready["manifest_sha256"],
                    ready["snapshot_uri"],
                ),
            )
        completed += 1
    _heartbeat(
        connection,
        "snapshot_publisher",
        accepted=accepted,
        last_published=max(published | set(due), default=0),
        next_snapshot=(accepted // snapshot_size + 1) * snapshot_size,
    )
    connection.close()
    return completed


def _loop(action: Any, poll_seconds: float, worker: str) -> None:
    while True:
        try:
            changed = action()
            if changed:
                logger.info("%s published %d new records", worker, changed)
        except Exception:
            logger.exception("%s iteration failed", worker)
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    promote = commands.add_parser("promote")
    promote.add_argument("--runs-dir", type=Path, required=True)
    promote.add_argument("--workspace", type=Path, required=True)
    promote.add_argument("--follow", action="store_true")
    promote.add_argument("--poll-seconds", type=float, default=2.0)
    assemble = commands.add_parser("assemble")
    assemble.add_argument("--workspace", type=Path, required=True)
    assemble.add_argument("--max-clips-per-video", type=int, default=24)
    assemble.add_argument("--follow", action="store_true")
    assemble.add_argument("--poll-seconds", type=float, default=2.0)
    progress = commands.add_parser("progress")
    progress.add_argument("--workspace", type=Path, required=True)
    heartbeat = commands.add_parser("heartbeat")
    heartbeat.add_argument("--workspace", type=Path, required=True)
    heartbeat.add_argument("--worker", required=True)
    heartbeat.add_argument("--state", default="running")
    heartbeat.add_argument("--follow", action="store_true")
    heartbeat.add_argument("--interval-seconds", type=float, default=10.0)
    configure = commands.add_parser("configure")
    configure.add_argument("--workspace", type=Path, required=True)
    configure.add_argument("--download-workers", type=int, required=True)
    configure.add_argument("--m2d-workers", type=int, required=True)
    configure.add_argument("--asr-workers", type=int, required=True)
    configure.add_argument("--upload-concurrency", type=int, required=True)
    configure.add_argument("--autoscaling-enabled", action="store_true")
    configure.add_argument("--download-min", type=int, default=1)
    configure.add_argument("--asr-concurrency-min", type=int, default=1)
    configure.add_argument("--asr-concurrency-max", type=int, default=1)
    autoscale = commands.add_parser("autoscale")
    autoscale.add_argument("--workspace", type=Path, required=True)
    autoscale.add_argument("--download-min", type=int, default=2)
    autoscale.add_argument("--download-max", type=int, default=8)
    autoscale.add_argument("--asr-min", type=int, default=1)
    autoscale.add_argument("--asr-max", type=int, default=2)
    autoscale.add_argument("--cpu-low", type=float, default=55.0)
    autoscale.add_argument("--cpu-high", type=float, default=85.0)
    autoscale.add_argument("--cpu-emergency", type=float, default=95.0)
    autoscale.add_argument("--m2d-backlog-high", type=int, default=64)
    autoscale.add_argument("--asr-backlog-high", type=int, default=8)
    autoscale.add_argument("--gpu-reserve-mb", type=float, default=12_000.0)
    autoscale.add_argument("--cooldown-seconds", type=float, default=60.0)
    autoscale.add_argument("--interval-seconds", type=float, default=10.0)
    autoscale.add_argument("--follow", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--dataset-dir", type=Path, required=True)
    verify.add_argument("--expected-count", type=int)
    verify.add_argument("--clip-seconds", type=float)
    publish = commands.add_parser("publish")
    publish.add_argument("--dataset-dir", type=Path, required=True)
    publish.add_argument("--bucket", required=True)
    publish.add_argument("--prefix", default="cinematic-continuous")
    publish.add_argument("--snapshot-id", required=True)
    publish.add_argument("--expected-count", type=int, required=True)
    publish.add_argument("--clip-seconds", type=float, required=True)
    publish.add_argument("--upload-concurrency", type=int, default=10)
    publish_due = commands.add_parser("publish-due")
    publish_due.add_argument("--workspace", type=Path, required=True)
    publish_due.add_argument("--bucket", required=True)
    publish_due.add_argument("--prefix", default="cinematic-continuous")
    publish_due.add_argument("--snapshot-size", type=int, default=5000)
    publish_due.add_argument("--upload-concurrency", type=int, default=10)
    publish_due.add_argument("--follow", action="store_true")
    publish_due.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.command == "promote":

        def action() -> int:
            return promote_once(args.runs_dir, args.workspace)

        if args.follow:
            _loop(action, args.poll_seconds, "promoter")
        else:
            print(action())
    elif args.command == "assemble":

        def action() -> int:
            return assemble_once(args.workspace, args.max_clips_per_video)

        if args.follow:
            _loop(action, args.poll_seconds, "assembler")
        else:
            print(action())
    elif args.command == "progress":
        print(json.dumps(progress_snapshot(args.workspace), indent=2))
    elif args.command == "heartbeat":
        while True:
            connection = connect(args.workspace)
            _heartbeat(connection, args.worker, args.state)
            connection.close()
            if not args.follow:
                break
            time.sleep(max(1.0, args.interval_seconds))
    elif args.command == "configure":
        _atomic_json(
            args.workspace / "config.json",
            {
                "download_workers": args.download_workers,
                "m2d_workers": args.m2d_workers,
                "asr_workers": args.asr_workers,
                "upload_concurrency": args.upload_concurrency,
                "autoscaling_enabled": args.autoscaling_enabled,
                "download_concurrency_bounds": [
                    args.download_min,
                    args.download_workers,
                ],
                "asr_inference_concurrency_bounds": [
                    args.asr_concurrency_min,
                    args.asr_concurrency_max,
                ],
                "coordinator_workers": {
                    "promoter": 1,
                    "assembler": 1,
                    "snapshot_publisher": 1,
                },
                "updated_at": _now(),
            },
        )
    elif args.command == "autoscale":
        if not 1 <= args.download_min <= args.download_max:
            parser.error("download bounds must satisfy 1 <= min <= max")
        if not 1 <= args.asr_min <= args.asr_max:
            parser.error("ASR bounds must satisfy 1 <= min <= max")
        if not 0 <= args.cpu_low < args.cpu_high <= args.cpu_emergency <= 100:
            parser.error(
                "CPU thresholds must satisfy 0 <= low < high <= emergency <= 100"
            )
        run_autoscaler(args)
    elif args.command == "verify":
        result = verify_snapshot(
            args.dataset_dir,
            expected_count=args.expected_count,
            clip_seconds=args.clip_seconds,
        )
        print(json.dumps(result, indent=2))
        if not result["all_requirements_pass"]:
            raise SystemExit(1)
    elif args.command == "publish":
        print(
            json.dumps(
                publish_snapshot(
                    args.dataset_dir,
                    bucket=args.bucket,
                    prefix=args.prefix,
                    snapshot_id=args.snapshot_id,
                    expected_count=args.expected_count,
                    clip_seconds=args.clip_seconds,
                    upload_concurrency=args.upload_concurrency,
                ),
                indent=2,
            )
        )
    elif args.command == "publish-due":

        def action() -> int:
            return publish_due_once(
                args.workspace,
                bucket=args.bucket,
                prefix=args.prefix,
                snapshot_size=args.snapshot_size,
                upload_concurrency=args.upload_concurrency,
            )

        if args.follow:
            _loop(action, args.poll_seconds, "snapshot publisher")
        else:
            print(action())


if __name__ == "__main__":
    main()
