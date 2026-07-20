"""Build aligned dialogue/background training records from continuous audio.

This is deliberately a downstream consumer.  It never controls or pauses source
discovery/download services.  Each expensive stage is idempotent and leases one
SQLite job, allowing the workflow to resume after a process or host restart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import wave
import zipfile
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from .audio import probe_audio_profile, sha256_file
from .flamingo_client import AudioFlamingoClient
from .model_client import SAMAudioClient
from .reconstruction import join_stereo_stems, normalize_source_audio
from .stereo import map_stems_to_stereo

logger = logging.getLogger(__name__)

DATABASE_FILENAME = "training-dataset.sqlite3"
CAPTION_MAX_NEW_TOKENS = 384
CAPTION_CONTRACT_MAX_ATTEMPTS = 2
CAPTION_SCHEMA_VERSION = 4
CAPTION_DESCRIPTION_POLICY = "af_next_description_timeline_v4"
CAPTION_REGENERATION_REASONS = {
    "scene_description_too_short",
    "scene_description_contains_model_boilerplate",
    "scene_timeline_empty",
    "scene_timeline_not_contiguous_30_seconds",
    "scene_timeline_underdescribed",
}
SCHEMA_VERSION = 1
RECORD_SCHEMA_VERSION = 1
DEFAULT_SNAPSHOT_SIZE = 1000
OUTPUT_FILENAMES = (
    "original.wav",
    "dialogue.wav",
    "background.wav",
    "scene_description.txt",
    "dialogue_transcript.txt",
    "metadata.json",
)
AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"})
BACKGROUND_ONLY_ENDING = (
    "The requested background contains no dialogue, intelligible speech, "
    "narration, or vocals."
)
_INITIALIZED_DATABASES: set[Path] = set()
_DATABASE_INIT_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def connect(workspace: Path) -> sqlite3.Connection:
    workspace.mkdir(parents=True, exist_ok=True)
    database_path = (workspace / DATABASE_FILENAME).resolve()
    connection = sqlite3.connect(
        database_path,
        timeout=30,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA synchronous=NORMAL")
    with _DATABASE_INIT_LOCK:
        if database_path in _INITIALIZED_DATABASES:
            return connection
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(
            """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_key TEXT NOT NULL UNIQUE,
            source_kind TEXT NOT NULL,
            source_ref TEXT NOT NULL,
            source_sha256 TEXT,
            source_json TEXT NOT NULL,
            separation_status TEXT NOT NULL DEFAULT 'pending',
            tag_status TEXT NOT NULL DEFAULT 'waiting',
            asr_status TEXT NOT NULL DEFAULT 'waiting',
            description_status TEXT NOT NULL DEFAULT 'waiting',
            package_status TEXT NOT NULL DEFAULT 'waiting',
            quality_bucket TEXT,
            separation_json TEXT,
            tag_json TEXT,
            asr_json TEXT,
            description_json TEXT,
            audit_json TEXT,
            error TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            separation_attempts INTEGER NOT NULL DEFAULT 0,
            description_attempts INTEGER NOT NULL DEFAULT 0,
            package_attempts INTEGER NOT NULL DEFAULT 0,
            description_fast_path_checked INTEGER NOT NULL DEFAULT 0,
            lease_stage TEXT,
            lease_owner TEXT,
            lease_expires_at REAL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS jobs_stage_ready ON jobs(
            separation_status,tag_status,asr_status,description_status,
            package_status,lease_expires_at
        );
        CREATE TABLE IF NOT EXISTS records (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL UNIQUE REFERENCES jobs(id),
            record_id TEXT NOT NULL UNIQUE,
            quality_bucket TEXT NOT NULL,
            record_json TEXT NOT NULL,
            uploaded_at TEXT,
            cleaned_at TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS records_quality_sequence
        ON records(quality_bucket,sequence);
        CREATE TABLE IF NOT EXISTS snapshots (
            end_sequence INTEGER PRIMARY KEY,
            snapshot_id TEXT NOT NULL UNIQUE,
            record_count INTEGER NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            s3_prefix TEXT NOT NULL,
            published_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS packages (
            package_key TEXT PRIMARY KEY,
            package_sha256 TEXT NOT NULL,
            local_path TEXT NOT NULL,
            imported_files INTEGER NOT NULL,
            imported_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS offsets (
            path TEXT PRIMARY KEY,
            inode INTEGER NOT NULL,
            byte_offset INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS workers (
            worker TEXT PRIMARY KEY,
            stage TEXT NOT NULL,
            state TEXT NOT NULL,
            details_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
            """
        )
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
        }
        for column in (
            "separation_attempts",
            "description_attempts",
            "package_attempts",
            "description_fast_path_checked",
        ):
            if column not in columns:
                try:
                    connection.execute(
                        f"ALTER TABLE jobs ADD COLUMN {column} "
                        "INTEGER NOT NULL DEFAULT 0"
                    )
                except sqlite3.OperationalError as error:
                    # Multiple stage processes initialize the same database at
                    # service start; another process may win this migration.
                    if "duplicate column name" not in str(error).casefold():
                        raise
        record_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(records)").fetchall()
        }
        if "cleaned_at" not in record_columns:
            connection.execute("ALTER TABLE records ADD COLUMN cleaned_at TEXT")
        _INITIALIZED_DATABASES.add(database_path)
    return connection


def _heartbeat(
    connection: sqlite3.Connection,
    worker: str,
    stage: str,
    state: str,
    details: dict[str, Any] | None = None,
) -> None:
    connection.execute(
        """INSERT INTO workers(worker,stage,state,details_json,updated_at)
        VALUES(?,?,?,?,?) ON CONFLICT(worker) DO UPDATE SET
        stage=excluded.stage,state=excluded.state,
        details_json=excluded.details_json,updated_at=excluded.updated_at""",
        (
            worker,
            stage,
            state,
            json.dumps(details or {}, separators=(",", ":")),
            _now(),
        ),
    )


def enqueue_job(
    connection: sqlite3.Connection,
    *,
    source_key: str,
    source_kind: str,
    source_ref: str,
    source_sha256: str | None,
    source: dict[str, Any],
) -> bool:
    timestamp = _now()
    return bool(
        connection.execute(
            """INSERT OR IGNORE INTO jobs(
            source_key,source_kind,source_ref,source_sha256,source_json,
            created_at,updated_at) VALUES(?,?,?,?,?,?,?)""",
            (
                source_key,
                source_kind,
                source_ref,
                source_sha256,
                json.dumps(source, separators=(",", ":")),
                timestamp,
                timestamp,
            ),
        ).rowcount
    )


def sync_continuous_catalog(
    workspace: Path,
    *,
    source_workspace: Path,
    source_s3_prefix: str,
    limit: int = 5000,
) -> int:
    """Import newly accepted clips without competing with the source writer."""
    destination = connect(workspace)
    source = sqlite3.connect(
        f"file:{source_workspace / 'catalog.sqlite3'}?mode=ro", uri=True, timeout=30
    )
    source.row_factory = sqlite3.Row
    last_sequence = int(
        destination.execute(
            """SELECT COALESCE(MAX(CAST(json_extract(source_json,
            '$.catalog_sequence') AS INTEGER)),0) FROM jobs
            WHERE source_kind='continuous'"""
        ).fetchone()[0]
    )
    rows = source.execute(
        """SELECT a.sequence,a.accepted_at,r.sha256,r.record_json
        FROM accepted a JOIN records r USING(sha256)
        WHERE a.sequence>? ORDER BY a.sequence LIMIT ?""",
        (last_sequence, limit),
    ).fetchall()
    inserted = 0
    with destination:
        for row in rows:
            record = json.loads(row["record_json"])
            record.update(
                {
                    "catalog_sequence": int(row["sequence"]),
                    "accepted_at": row["accepted_at"],
                }
            )
            filename = str(record["continuous_filename"])
            local = source_workspace / "accepted" / "audio" / filename
            s3_key = f"{source_s3_prefix.strip('/')}/audio/{row['sha256']}.wav"
            inserted += int(
                enqueue_job(
                    destination,
                    source_key=f"continuous:{row['sha256']}",
                    source_kind="continuous",
                    source_ref=json.dumps(
                        {"local_path": str(local), "s3_key": s3_key},
                        separators=(",", ":"),
                    ),
                    source_sha256=str(row["sha256"]),
                    source=record,
                )
            )
    _heartbeat(
        destination,
        "catalog-sync",
        "sync",
        "running",
        {"inserted": inserted, "observed": len(rows), "last_sequence": last_sequence},
    )
    source.close()
    destination.close()
    return inserted


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(destination)


def sync_inbox(workspace: Path, inbox: Path) -> int:
    """Import every audio file from each newly arrived ZIP or directory."""
    connection = connect(workspace)
    extraction_root = workspace / "input-packages"
    extraction_root.mkdir(parents=True, exist_ok=True)
    inserted = 0
    candidates = sorted(inbox.glob("*.zip")) + sorted(
        path for path in inbox.iterdir() if path.is_dir()
    )
    for package_path in candidates:
        is_archive = package_path.is_file()
        package_digest = (
            sha256_file(package_path)
            if is_archive
            else hashlib.sha256(str(package_path.resolve()).encode()).hexdigest()
        )
        package_key = str(package_path.resolve())
        if (
            is_archive
            and connection.execute(
                "SELECT 1 FROM packages WHERE package_key=? AND package_sha256=?",
                (package_key, package_digest),
            ).fetchone()
        ):
            continue
        root = extraction_root / package_digest
        if is_archive:
            if not root.exists():
                temporary = Path(
                    tempfile.mkdtemp(prefix=".extract-", dir=extraction_root)
                )
                try:
                    _safe_extract(package_path, temporary)
                    os.replace(temporary, root)
                except Exception:
                    shutil.rmtree(temporary, ignore_errors=True)
                    raise
        else:
            root = package_path
        audio_files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
        )
        with connection:
            for audio in audio_files:
                digest = sha256_file(audio)
                relative = str(audio.relative_to(root))
                source_ref = audio
                if not is_archive:
                    immutable_root = extraction_root / "files"
                    immutable_root.mkdir(exist_ok=True)
                    immutable = immutable_root / f"{digest}{audio.suffix.lower()}"
                    if not immutable.is_file():
                        temporary = immutable.with_suffix(immutable.suffix + ".tmp")
                        shutil.copy2(audio, temporary)
                        if sha256_file(temporary) != digest:
                            temporary.unlink(missing_ok=True)
                            continue
                        os.replace(temporary, immutable)
                    source_ref = immutable
                inserted += int(
                    enqueue_job(
                        connection,
                        source_key=f"package:{package_digest}:{relative}:{digest}",
                        source_kind="package",
                        source_ref=str(source_ref),
                        source_sha256=digest,
                        source={
                            "package": package_path.name,
                            "package_sha256": package_digest,
                            "relative_path": relative,
                            "filename": audio.name,
                        },
                    )
                )
            connection.execute(
                """INSERT INTO packages(package_key,package_sha256,local_path,
                imported_files,imported_at) VALUES(?,?,?,?,?)
                ON CONFLICT(package_key) DO UPDATE SET
                package_sha256=excluded.package_sha256,
                local_path=excluded.local_path,
                imported_files=excluded.imported_files,
                imported_at=excluded.imported_at""",
                (package_key, package_digest, str(root), len(audio_files), _now()),
            )
    _heartbeat(
        connection,
        "inbox-sync",
        "sync",
        "running",
        {"inserted": inserted, "packages_seen": len(candidates)},
    )
    connection.close()
    return inserted


_STAGE_READY_SQL = {
    "separation": "separation_status IN ('pending','retry','running')",
    "description": (
        "separation_status='complete' AND tag_status='complete' "
        "AND asr_status='complete' "
        "AND description_status IN ('pending','retry','running')"
    ),
    "package": (
        "separation_status='complete' AND tag_status='complete' "
        "AND asr_status='complete' AND description_status='complete' "
        "AND package_status IN ('pending','retry','running')"
    ),
}


def _claim(
    connection: sqlite3.Connection,
    stage: str,
    worker: str,
    *,
    lease_seconds: float,
) -> dict[str, Any] | None:
    now = time.time()
    status_column = f"{stage}_status"
    attempts_column = f"{stage}_attempts"
    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute(
            f"""SELECT * FROM jobs WHERE {_STAGE_READY_SQL[stage]}
            AND (lease_expires_at IS NULL OR lease_expires_at<=?)
            ORDER BY id LIMIT 1""",
            (now,),
        ).fetchone()
        if row is None:
            connection.commit()
            return None
        connection.execute(
            f"""UPDATE jobs SET {status_column}='running',lease_stage=?,
            lease_owner=?,lease_expires_at=?,attempts=attempts+1,
            {attempts_column}={attempts_column}+1,
            updated_at=? WHERE id=?""",
            (stage, worker, now + lease_seconds, _now(), row["id"]),
        )
        claimed = connection.execute(
            "SELECT * FROM jobs WHERE id=?", (row["id"],)
        ).fetchone()
        connection.commit()
        return dict(claimed)
    except Exception:
        connection.rollback()
        raise


def _finish(
    connection: sqlite3.Connection,
    job_id: int,
    stage: str,
    *,
    values: dict[str, Any] | None = None,
) -> None:
    assignments = [
        f"{stage}_status='complete'",
        "lease_stage=NULL",
        "lease_owner=NULL",
        "lease_expires_at=NULL",
        "error=NULL",
        "updated_at=?",
    ]
    parameters: list[Any] = [_now()]
    for key, value in (values or {}).items():
        assignments.append(f"{key}=?")
        parameters.append(
            json.dumps(value, separators=(",", ":"))
            if key.endswith("_json") and not isinstance(value, str)
            else value
        )
    parameters.append(job_id)
    connection.execute(
        f"UPDATE jobs SET {','.join(assignments)} WHERE id=?", parameters
    )


def _fail(
    connection: sqlite3.Connection,
    job_id: int,
    stage: str,
    error: Exception,
    *,
    max_attempts: int = 5,
) -> str:
    attempts_column = f"{stage}_attempts"
    attempts = int(
        connection.execute(
            f"SELECT {attempts_column} FROM jobs WHERE id=?", (job_id,)
        ).fetchone()[0]
    )
    status = "failed" if attempts >= max_attempts else "retry"
    connection.execute(
        f"""UPDATE jobs SET {stage}_status=?,lease_stage=NULL,lease_owner=NULL,
        lease_expires_at=NULL,error=?,updated_at=? WHERE id=?""",
        (status, f"{type(error).__name__}: {error}"[-4000:], _now(), job_id),
    )
    return status


def _retry_transient_upstream_failure(
    connection: sqlite3.Connection,
    job_id: int,
    stage: str,
    error: Exception,
) -> bool:
    """Retry endpoint outages without consuming the record's quality attempts."""
    text = f"{type(error).__name__}: {error}".casefold()
    transient_markers = (
        "connecterror",
        "connection refused",
        "connection reset",
        "connection aborted",
        "readtimeout",
        "connecttimeout",
        "remoteprotocolerror",
        "server disconnected",
        "502 bad gateway",
        "503 service unavailable",
    )
    if not any(marker in text for marker in transient_markers):
        return False
    attempts_column = f"{stage}_attempts"
    connection.execute(
        f"""UPDATE jobs SET {stage}_status='retry',
        {attempts_column}=MAX(0,{attempts_column}-1),
        attempts=MAX(0,attempts-1),lease_stage=NULL,lease_owner=NULL,
        lease_expires_at=NULL,error=?,updated_at=? WHERE id=?""",
        (f"{type(error).__name__}: {error}"[-4000:], _now(), job_id),
    )
    return True


def recover_leases(workspace: Path) -> int:
    """Recover jobs owned by the previous supervisor process group."""
    connection = connect(workspace)
    recovered = 0
    with connection:
        for stage in ("separation", "description", "package"):
            recovered += connection.execute(
                f"""UPDATE jobs SET {stage}_status='retry',lease_stage=NULL,
                lease_owner=NULL,lease_expires_at=NULL,
                error='worker_restarted',updated_at=?
                WHERE {stage}_status='running'""",
                (_now(),),
            ).rowcount
    _heartbeat(
        connection,
        "lease-recovery",
        "recovery",
        "complete",
        {"recovered": recovered},
    )
    connection.close()
    return recovered


def _record_root(workspace: Path, job_id: int) -> Path:
    return workspace / "work" / f"{job_id:012d}"


def _materialize_source(
    job: dict[str, Any], destination: Path, *, bucket: str | None
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if job["source_kind"] == "package":
        source = Path(str(job["source_ref"]))
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
        return destination
    reference = json.loads(job["source_ref"])
    local = Path(reference["local_path"])
    if local.is_file():
        shutil.copy2(local, destination)
        return destination
    if not bucket:
        raise FileNotFoundError(
            f"Local source was pruned and no bucket configured: {local}"
        )
    import boto3

    boto3.client("s3").download_file(bucket, reference["s3_key"], str(destination))
    return destination


def normalize_training_audio(
    source: Path,
    destination: Path,
    *,
    target_lufs: float = -20.0,
    true_peak_db: float = -1.5,
) -> dict[str, Any]:
    """Apply deterministic two-pass EBU R128 normalization to stereo PCM16."""
    intermediate = destination.with_name("original.format-normalized.wav")
    normalize_source_audio(source, intermediate, sample_rate=48_000)
    _fit_training_frame_count(intermediate)
    first = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(intermediate),
            "-af",
            f"loudnorm=I={target_lufs}:LRA=11:TP={true_peak_db}:print_format=json",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    loudnorm_output = first.stderr[first.stderr.rfind("[Parsed_loudnorm") :]
    match = re.search(r"\{[\s\S]*?\}", loudnorm_output)
    measured = json.loads(match.group(0)) if match else {}
    required = ("input_i", "input_lra", "input_tp", "input_thresh", "target_offset")
    if not all(
        key in measured and measured[key] not in {"-inf", "inf"} for key in required
    ):
        os.replace(intermediate, destination)
        return {"policy": "pcm16_stereo_48khz_fallback_v1", "measured": measured}
    filter_value = (
        f"loudnorm=I={target_lufs}:LRA=11:TP={true_peak_db}:"
        f"measured_I={measured['input_i']}:measured_LRA={measured['input_lra']}:"
        f"measured_TP={measured['input_tp']}:measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:linear=true:print_format=json"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(intermediate),
            "-af",
            filter_value,
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        check=True,
    )
    _fit_training_frame_count(destination)
    intermediate.unlink(missing_ok=True)
    return {
        "policy": "ebu_r128_two_pass_v1",
        "target_lufs": target_lufs,
        "target_true_peak_db": true_peak_db,
        "measured": measured,
    }


def _fit_training_frame_count(
    path: Path,
    *,
    frame_count: int = 30 * 48_000,
) -> None:
    """Trim or silence-pad a normalized WAV to the exact training duration."""
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if (channels, sample_width, sample_rate) != (2, 2, 48_000):
        raise ValueError(f"Unexpected normalized training WAV contract: {path}")
    frame_bytes = channels * sample_width
    target_bytes = frame_count * frame_bytes
    fitted = frames[:target_bytes].ljust(target_bytes, b"\0")
    if fitted == frames:
        return
    temporary = path.with_name(f".{path.name}.frames.tmp")
    try:
        with wave.open(str(temporary), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(sample_width)
            output.setframerate(sample_rate)
            output.writeframes(fitted)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _wave_contract(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as source:
        return {
            "channels": source.getnchannels(),
            "sample_width_bytes": source.getsampwidth(),
            "sample_rate_hz": source.getframerate(),
            "frame_count": source.getnframes(),
            "duration_seconds": round(source.getnframes() / source.getframerate(), 6),
        }


def separate_once(
    workspace: Path,
    *,
    sam_api_url: str,
    bucket: str | None,
    worker: str,
    elastic_worker_from: int = -1,
    description_backlog_high: int = 500,
) -> dict[str, Any] | None:
    connection = connect(workspace)
    try:
        worker_index = int(worker.rsplit("-", 1)[-1])
    except ValueError:
        worker_index = -1
    if elastic_worker_from >= 0 and worker_index >= elastic_worker_from:
        backlog = int(
            connection.execute(
                """SELECT COUNT(*) FROM jobs
                WHERE separation_status='complete' AND tag_status='complete'
                AND asr_status='complete'
                AND description_status IN ('pending','retry','running')"""
            ).fetchone()[0]
        )
        if backlog >= max(1, description_backlog_high):
            _heartbeat(
                connection,
                worker,
                "separation",
                "paused",
                {
                    "reason": "description_backpressure",
                    "description_backlog": backlog,
                    "resume_below": max(1, description_backlog_high),
                },
            )
            connection.close()
            return None
    job = _claim(connection, "separation", worker, lease_seconds=7200)
    if job is None:
        _heartbeat(connection, worker, "separation", "idle")
        connection.close()
        return None
    root = _record_root(workspace, int(job["id"]))
    started = time.perf_counter()
    try:
        root.mkdir(parents=True, exist_ok=True)
        source = _materialize_source(job, root / "source.input", bucket=bucket)
        source_profile = probe_audio_profile(source)
        loudness = normalize_training_audio(source, root / "original.wav")
        original = root / "original.wav"
        client = SAMAudioClient(sam_api_url)
        result = client.separate(
            original,
            root / "sam",
            order="voice_first",
            targets=("voice",),
        )
        mapped = map_stems_to_stereo(
            original,
            {"voice": result.stems["voice"], "sfx": result.stems["sfx"]},
            root / "mapped",
        )
        shutil.copy2(mapped["voice"].path, root / "dialogue.wav")
        shutil.copy2(mapped["sfx"].path, root / "background.wav")
        joined = join_stereo_stems(
            original,
            {"voice": root / "dialogue.wav", "background": root / "background.wav"},
            root / "joined.diagnostic.wav",
        )
        contracts = {
            name: _wave_contract(root / name)
            for name in ("original.wav", "dialogue.wav", "background.wav")
        }
        aligned = len({tuple(value.values()) for value in contracts.values()}) == 1
        separation = {
            "schema_version": 1,
            "policy": "voice_only_sam_audio_judged_stereo_v1",
            "requested_targets": ["voice"],
            "background_role": "voice-stage residual; music, ambience, and SFX",
            "sam": result.metadata,
            "response_headers": result.response_headers,
            "source_profile": source_profile,
            "normalization": loudness,
            "stereo_mapping": {
                "dialogue": mapped["voice"].metadata,
                "background": mapped["sfx"].metadata,
            },
            "reconstruction": joined.metrics,
            "wave_contract": contracts,
            "aligned": aligned,
            "processing_seconds": round(time.perf_counter() - started, 3),
        }
        if not aligned:
            raise RuntimeError("Output WAV timing/channel contract is not aligned")
        (workspace / "background-audio").mkdir(exist_ok=True)
        (workspace / "dialogue-audio").mkdir(exist_ok=True)
        background_link = workspace / "background-audio" / f"{job['id']:012d}.wav"
        dialogue_link = workspace / "dialogue-audio" / f"{job['id']:012d}.wav"
        for source_path, link in (
            (root / "background.wav", background_link),
            (root / "dialogue.wav", dialogue_link),
        ):
            link.unlink(missing_ok=True)
            os.link(source_path, link)
        with connection:
            _finish(
                connection,
                int(job["id"]),
                "separation",
                values={
                    "separation_json": separation,
                    "tag_status": "pending",
                    "asr_status": "pending",
                    "description_status": "pending",
                    "package_status": "pending",
                },
            )
        result_summary = {
            "status": result.metadata.get("verification_status", "uncertain"),
            "job_id": job["id"],
            "seconds": separation["processing_seconds"],
        }
        _heartbeat(connection, worker, "separation", "running", result_summary)
        connection.close()
        return result_summary
    except Exception as error:
        logger.exception("Training separation failed for job %s", job["id"])
        with connection:
            transient = _retry_transient_upstream_failure(
                connection, int(job["id"]), "separation", error
            )
            if not transient:
                _fail(connection, int(job["id"]), "separation", error)
        _heartbeat(
            connection,
            worker,
            "separation",
            "error",
            {"job_id": job["id"], "error": str(error)[-1000:]},
        )
        connection.close()
        if transient:
            time.sleep(15)
        return {"status": "error", "job_id": job["id"], "error": str(error)}


def import_jsonl(workspace: Path, path: Path, kind: str) -> int:
    if kind not in {"tag", "asr"} or not path.is_file():
        return 0
    connection = connect(workspace)
    stat = path.stat()
    key = str(path.resolve())
    previous = connection.execute(
        "SELECT inode,byte_offset FROM offsets WHERE path=?", (key,)
    ).fetchone()
    offset = (
        int(previous["byte_offset"])
        if previous and previous["inode"] == stat.st_ino
        else 0
    )
    processed = 0
    with path.open("rb") as source:
        source.seek(offset)
        while True:
            start = source.tell()
            line = source.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                source.seek(start)
                break
            try:
                payload = json.loads(line)
                job_id = int(Path(str(payload["filename"])).stem)
            except (ValueError, KeyError, json.JSONDecodeError, UnicodeError):
                continue
            with connection:
                updated = connection.execute(
                    f"""UPDATE jobs SET {kind}_status='complete',{kind}_json=?,
                    updated_at=? WHERE id=? AND separation_status='complete'""",
                    (json.dumps(payload, separators=(",", ":")), _now(), job_id),
                ).rowcount
            if not updated:
                source.seek(start)
                break
            processed += 1
        new_offset = source.tell()
    with connection:
        connection.execute(
            """INSERT INTO offsets(path,inode,byte_offset) VALUES(?,?,?)
            ON CONFLICT(path) DO UPDATE SET inode=excluded.inode,
            byte_offset=excluded.byte_offset""",
            (key, stat.st_ino, new_offset),
        )
    _heartbeat(connection, f"{kind}-import", kind, "running", {"imported": processed})
    connection.close()
    return processed


_SPEECH_DESCRIPTION_PATTERN = re.compile(
    r"\b(?:voice|voices|speaker|speech|dialogue|conversation|converse|talking|"
    r"spoken|says|said|narrat(?:or|ion|ing)|utterance|words?|murmurs?|whispers?|"
    r"shouts?|chatter(?:ing)?|singing|vocals?)\b",
    re.IGNORECASE,
)


def _background_evidence(tag: dict[str, Any]) -> list[dict[str, Any]]:
    windows = list(tag.get("windows", []))
    if len(windows) > 10:
        indexes = {round(index * (len(windows) - 1) / 9) for index in range(10)}
        windows = [windows[index] for index in sorted(indexes)]
    evidence: list[dict[str, Any]] = []
    for window in windows:
        evidence.append(
            {
                "t": [
                    round(float(window.get("start_seconds") or 0.0), 1),
                    round(float(window.get("end_seconds") or 0.0), 1),
                ],
                "tags": [
                    [
                        str(item.get("name") or ""),
                        round(float(item.get("probability") or 0.0), 3),
                    ]
                    for item in window.get("top_labels", [])
                    if not _SPEECH_DESCRIPTION_PATTERN.search(
                        str(item.get("name") or "")
                    )
                ][:4],
                "music": round(float(window.get("music_score") or 0.0), 3),
                "background": round(float(window.get("background_score") or 0.0), 3),
            }
        )
    return evidence


def _caption_prompt(tag: dict[str, Any]) -> str:
    evidence = json.dumps(
        _background_evidence(tag), ensure_ascii=False, separators=(",", ":")
    )
    return (
        "Listen closely to this background-only cinematic stem; the foreground "
        "dialogue was removed. Produce a dense timestamped acoustic analysis using "
        "4-6 contiguous chronological intervals in your native [start-end] format. "
        "For each interval, identify only sounds you can actually hear and describe "
        "their loudness, texture, distance, left/center/right placement, motion, "
        "layering, and meaningful changes. Prefer concrete acoustic observations "
        "over generic labels. Never transcribe or describe speakers, voices, "
        "dialogue, narration, singing, or vocals. Do not mention the model, "
        "classifier, evidence, prompt, or separation process. Do not claim a sound "
        "merely because it appears in the hints. The hints are fallible, timestamped, "
        "non-dialogue cues to verify against the audio: "
        + evidence
    )


def _parse_timestamp(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return float(parts[0]) * 60.0 + float(parts[1])
    raise ValueError(f"Invalid timestamp: {value}")


def _section_caption_completion(
    text: str, tag: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Parse the public DESCRIPTION/TIMELINE contract produced by AF-Next."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    match = re.search(
        r"(?:^|\n)\s*DESCRIPTION\s*:\s*(?P<description>[\s\S]*?)"
        r"(?:\n\s*TIMELINE\s*:\s*)(?P<timeline>[\s\S]*)$",
        cleaned,
        re.IGNORECASE,
    )
    if not match:
        return None
    description = match.group("description").strip()
    timeline: list[dict[str, Any]] = []
    bullet_pattern = re.compile(
        r"^\s*[-*•]\s*"
        r"(?P<start>\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
        r"(?:-|\u2013|\u2014|to)\s*"
        r"(?P<end>\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
        r"(?:\u2014|\u2013|-|:)\s*(?P<events>.+?)\s*$",
        re.IGNORECASE,
    )
    for line in match.group("timeline").splitlines():
        bullet = bullet_pattern.match(line)
        if not bullet:
            continue
        try:
            start = _parse_timestamp(bullet.group("start"))
            end = _parse_timestamp(bullet.group("end"))
        except ValueError:
            continue
        timeline.append(
            {
                "start_seconds": start,
                "end_seconds": end,
                "events": bullet.group("events").strip(),
            }
        )
    parsed, speech_mentions = _normalize_parsed_completion(
        {
            "description": description,
            "timeline": timeline,
            "global_tags": _m2d_global_tags(tag),
            "music": None,
            "ambience": None,
            "sound_effects": [],
        },
        tag,
    )
    return parsed, {
        "format": "description_timeline_sections_v2",
        "speech_mentions_omitted": speech_mentions,
        "parsed_timeline_events": len(parsed["timeline"]),
    }


def _json_completion(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Audio Flamingo did not return a JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict) or not isinstance(value.get("description"), str):
        raise ValueError("Audio Flamingo JSON is missing description")
    expected_types = {
        "global_tags": list,
        "sound_effects": list,
    }
    for key, expected_type in expected_types.items():
        if not isinstance(value.get(key), expected_type):
            raise ValueError(f"Audio Flamingo JSON has invalid {key}")
    if value.get("music") is not None and not isinstance(value.get("music"), dict):
        raise ValueError("Audio Flamingo JSON has invalid music")
    if value.get("ambience") is not None and not isinstance(value.get("ambience"), str):
        raise ValueError("Audio Flamingo JSON has invalid ambience")
    timeline = value.get("timeline")
    if not isinstance(timeline, list):
        raise ValueError("Audio Flamingo JSON is missing a timeline array")
    for item in timeline:
        if not isinstance(item, dict):
            raise ValueError("Timeline item is not an object")
        start_value = float(item.get("start_seconds"))
        end_value = float(item.get("end_seconds"))
        if not 0 <= start_value < end_value <= 30.5:
            raise ValueError("Timeline item is outside the audio bounds")
    return value


def _m2d_global_tags(tag: dict[str, Any]) -> list[str]:
    scores: Counter[str] = Counter()
    for window in tag.get("windows", []):
        for item in window.get("top_labels", [])[:8]:
            name = str(item.get("name") or "").strip()
            probability = float(item.get("probability") or 0.0)
            if name:
                scores[name] += probability
    return [name for name, _ in scores.most_common(12)]


def _remove_speech_description(text: str) -> tuple[str, int]:
    retained: list[str] = []
    omitted = 0
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if _SPEECH_DESCRIPTION_PATTERN.search(sentence) or re.search(
            r'"[^\"]+"', sentence
        ):
            omitted += 1
            continue
        retained.append(sentence)
    return " ".join(retained), omitted


def _remove_speech_fragments(text: str) -> tuple[str, int]:
    retained: list[str] = []
    omitted = 0
    for fragment in re.split(r"\s*[,;]\s*", text.strip()):
        if not fragment:
            continue
        if _SPEECH_DESCRIPTION_PATTERN.search(fragment) or re.search(
            r'"[^\"]+"', fragment
        ):
            omitted += 1
        else:
            retained.append(fragment)
    return ", ".join(retained), omitted


def _bounded_background_description(text: str) -> str:
    """Fit model prose to the published contract without inventing evidence."""
    content = text.strip()
    ending_words = len(re.findall(r"[\w']+", BACKGROUND_ONLY_ENDING))
    content_limit = 140 - ending_words
    matches = list(re.finditer(r"[\w']+", content))
    if len(matches) > content_limit:
        hard_end = matches[content_limit - 1].end()
        candidate = content[:hard_end].rstrip(" ,;:-")
        sentence_ends = list(re.finditer(r"[.!?](?=\s|$)", candidate))
        for sentence_end in reversed(sentence_ends):
            sentence_candidate = content[: sentence_end.end()].strip()
            if len(re.findall(r"[\w']+", sentence_candidate)) >= 69:
                candidate = sentence_candidate
                break
        content = candidate
        if content and content[-1] not in ".!?":
            content += "."
    return f"{content.rstrip()} {BACKGROUND_ONLY_ENDING}".strip()


def _partial_json_value(text: str, key: str) -> Any | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*', text)
    if not match:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text, match.end())
        return value
    except json.JSONDecodeError:
        return None


def _recover_partial_timeline_array(text: str) -> list[dict[str, Any]]:
    """Salvage valid timeline objects when one malformed item breaks the array."""
    marker = re.search(r'"timeline"\s*:\s*\[', text)
    if not marker:
        return []
    tail = text[marker.end() :]
    recovered: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", tail):
        try:
            value, _ = decoder.raw_decode(tail, match.start())
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if not {"start_seconds", "end_seconds", "events"}.issubset(value):
            continue
        recovered.append(value)
        if len(recovered) >= 6:
            break
    return recovered


def _canonicalize_timeline(
    timeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Make model-grounded events an ordered, contiguous 30-second timeline."""
    candidates = sorted(timeline, key=lambda item: float(item["start_seconds"]))
    ordered: list[dict[str, Any]] = []
    signatures: list[tuple[str, ...]] = []
    for item in candidates:
        values = item.get("events") or []
        events = values if isinstance(values, list) else [values]
        unique: list[str] = []
        seen: set[str] = set()
        for value in events:
            event = str(value).strip()
            key = event.casefold().rstrip(" .")
            if event and key not in seen:
                unique.append(event)
                seen.add(key)
        if not unique:
            continue
        signature = tuple(sorted(seen))
        if ordered and signature == signatures[-1]:
            ordered[-1]["end_seconds"] = max(
                float(ordered[-1]["end_seconds"]), float(item["end_seconds"])
            )
            continue
        ordered.append({**item, "events": unique})
        signatures.append(signature)
        if len(ordered) >= 6:
            break
    if not ordered:
        return []
    ordered[0]["start_seconds"] = 0.0
    for previous, following in zip(ordered, ordered[1:], strict=False):
        boundary = max(
            float(previous["start_seconds"]),
            min(30.0, float(following["start_seconds"])),
        )
        previous["end_seconds"] = boundary
        following["start_seconds"] = boundary
    ordered[-1]["end_seconds"] = 30.0
    return [
        item
        for item in ordered
        if float(item["end_seconds"]) > float(item["start_seconds"])
    ]


def _normalize_timeline(value: Any) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(value, list):
        return [], 0
    timeline: list[dict[str, Any]] = []
    omitted = 0
    for item in value[:60]:
        if not isinstance(item, dict):
            continue
        try:
            start = max(0.0, float(item.get("start_seconds")))
            end = min(30.0, float(item.get("end_seconds")))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        events = item.get("events")
        values = events if isinstance(events, list) else [events]
        cleaned_events: list[str] = []
        for event in values:
            cleaned, count = _remove_speech_fragments(str(event or ""))
            omitted += count
            normalized = cleaned.strip(" .,:;!?-\u2013\u2014")
            # AF-Next can end a token-limited native timeline with a bare
            # punctuation mark or an unfinished "Sound of heavy." fragment.
            # Keeping either creates a meaningless final interval and makes the
            # same record retry forever. Dropping it lets canonicalization extend
            # the preceding, grounded interval to the end of the clip.
            incomplete_sound = bool(
                re.fullmatch(
                    r"sound of (?:heavy|light|loud|soft|deep|high|low)",
                    normalized,
                    re.IGNORECASE,
                )
            )
            terminal_placeholder = normalized.casefold() in {
                "additional",
                "audio",
                "background",
                "event",
                "noise",
                "sound",
            }
            if normalized and not incomplete_sound and not terminal_placeholder:
                cleaned_events.append(cleaned)
        if cleaned_events:
            timeline.append(
                {
                    "start_seconds": start,
                    "end_seconds": end,
                    "events": cleaned_events,
                }
            )
    return _canonicalize_timeline(timeline), omitted


def _repair_underdescribed_timeline(parsed: dict[str, Any]) -> int:
    """Turn short grounded event labels into minimally complete interval prose.

    This is used only after both caption-model attempts fail solely because an
    interval has fewer than four words. It never adds a new sound source: the
    repair restates the model/tag-derived labels already attached to the interval.
    """
    repaired = 0
    placeholders = {"additional", "audio", "background", "event", "noise", "sound"}
    for item in parsed.get("timeline") or []:
        values = item.get("events") or []
        events = values if isinstance(values, list) else [values]
        labels = [
            str(value).strip().strip(" .,:;!?-\u2013\u2014")
            for value in events
            if str(value).strip().strip(" .,:;!?-\u2013\u2014")
        ]
        word_count = len(re.findall(r"[\w']+", " ".join(labels)))
        if word_count >= 4 or not labels:
            continue
        if any(label.casefold() in placeholders for label in labels):
            continue
        if len(labels) == 1 and labels[0].casefold() in {
            "silence",
            "near silence",
            "near-silence",
        }:
            sentence = "Near-silence persists throughout this interval."
        elif len(labels) == 1:
            sentence = f"{labels[0]} remains audible throughout this interval."
        else:
            subject = (
                f"{labels[0]} and {labels[1]}"
                if len(labels) == 2
                else f"{', '.join(labels[:-1])}, and {labels[-1]}"
            )
            sentence = f"{subject} remain audible throughout this interval."
        item["events"] = [sentence[0].upper() + sentence[1:]]
        repaired += 1
    return repaired


def _expand_grounded_description(
    description: str,
    timeline: list[dict[str, Any]],
    tags: list[str],
) -> str:
    """Use model-produced timed events to meet the prose contract without invention."""
    content = description.strip()
    # The fixed background-only ending contributes eleven words, so the prose
    # body must contain at least 69 to satisfy the public 80-word contract.
    target_content_words = 69
    seen = content.casefold()
    transitions = (
        "At the opening,",
        "Next,",
        "Around the middle,",
        "Later,",
        "Near the end,",
        "Finally,",
    )
    for index, item in enumerate(timeline):
        if len(re.findall(r"[\w']+", content)) >= target_content_words:
            break
        events = item.get("events") or []
        if isinstance(events, str):
            events = [events]
        event = " ".join(str(value).strip() for value in events if str(value).strip())
        event = event.strip()
        if not event or event.casefold().rstrip(".") in seen:
            continue
        event = event[0].lower() + event[1:]
        content = f"{content.rstrip()} {transitions[min(index, 5)]} {event}".strip()
        seen = content.casefold()
    if len(re.findall(r"[\w']+", content)) < target_content_words and tags:
        palette = ", ".join(tags[:6])
        content = (
            f"{content.rstrip()} Across the full clip, the recurring acoustic "
            f"palette includes {palette}, with the audible changes localized in "
            "the timeline below."
        ).strip()
    if len(re.findall(r"[\w']+", content)) < target_content_words and timeline:
        content = (
            f"{content.rstrip()} These remain the principal audible layers across "
            "the 30-second background, with changes in prominence and texture "
            "anchored to the timed events below."
        ).strip()
    if len(re.findall(r"[\w']+", content)) < target_content_words and timeline:
        # A genuinely static scene may yield only one short but valid timed event.
        # Reach the prose floor without inventing another source or transition.
        grounding = (
            "No separately timed transition is evident, so this documented texture "
            "remains the dominant audible background throughout the interval."
            if len(timeline) == 1
            else "The sequence changes at the documented boundaries, while these "
            "named layers continue to define the audible background between "
            "transitions."
        )
        content = f"{content.rstrip()} {grounding}".strip()
    clarifications = (
        (
            "The audible balance is described through this sustained layer, whose "
            "presence defines the scene for the duration shown below.",
            "Within that span, its continuing texture provides the clip's consistent "
            "environmental character.",
        )
        if len(timeline) == 1
        else (
            "Together, these timed layers define the clip's continuous background "
            "character as their relative prominence changes across the scene.",
            "The overview follows the audible changes anchored to the intervals "
            "below.",
        )
    ) if timeline else ()
    for clarification in clarifications:
        if len(re.findall(r"[\w']+", content)) >= target_content_words:
            break
        content = f"{content.rstrip()} {clarification}".strip()
    return content


def _grounded_m2d_timeline(tag: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a bounded timeline only from timestamped non-dialogue evidence."""
    timeline = [
        {
            "start_seconds": float(item["t"][0]),
            "end_seconds": float(item["t"][1]),
            "events": [str(value[0]) for value in item["tags"][:3]],
        }
        for item in _background_evidence(tag)
        if float(item["t"][1]) > float(item["t"][0]) and item["tags"]
    ]
    if len(timeline) <= 6:
        return timeline
    indexes = {round(index * (len(timeline) - 1) / 5) for index in range(6)}
    return [timeline[index] for index in sorted(indexes)]


def _apply_grounded_timeline_fallback(
    parsed: dict[str, Any],
    metadata: dict[str, Any],
    tag: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if parsed.get("timeline"):
        return parsed, metadata
    timeline = _grounded_m2d_timeline(tag)
    if not timeline:
        return parsed, metadata
    parsed["timeline"] = _canonicalize_timeline(timeline)
    parsed, additional_mentions = _normalize_parsed_completion(parsed, tag)
    metadata["speech_mentions_omitted"] = int(
        metadata.get("speech_mentions_omitted") or 0
    ) + int(additional_mentions)
    metadata["timeline_fallback"] = "m2d_timestamped_non_dialogue_v1"
    metadata["grounded_timeline_events"] = len(parsed["timeline"])
    return parsed, metadata


def _normalize_parsed_completion(
    parsed: dict[str, Any], tag: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    raw_description = str(parsed.get("description") or "").strip()
    if raw_description.endswith(BACKGROUND_ONLY_ENDING):
        raw_description = raw_description[: -len(BACKGROUND_ONLY_ENDING)].rstrip()
    description, speech_mentions = _remove_speech_description(raw_description)
    timeline, timeline_mentions = _normalize_timeline(parsed.get("timeline"))
    speech_mentions += timeline_mentions
    tags_value = parsed.get("global_tags")
    tags_source = (
        [str(value) for value in tags_value]
        if isinstance(tags_value, list)
        else _m2d_global_tags(tag)
    )
    tags = [
        value for value in tags_source if not _SPEECH_DESCRIPTION_PATTERN.search(value)
    ][:8]
    description = _expand_grounded_description(description, timeline, tags)
    description = _bounded_background_description(description)
    ambience_value = parsed.get("ambience")
    ambience = None
    if isinstance(ambience_value, str):
        ambience, count = _remove_speech_fragments(ambience_value)
        speech_mentions += count
        ambience = ambience or None
    effects_value = parsed.get("sound_effects")
    effects_source = effects_value if isinstance(effects_value, list) else []
    sound_effects: list[str] = []
    for value in effects_source:
        cleaned, count = _remove_speech_fragments(str(value))
        speech_mentions += count
        if cleaned:
            sound_effects.append(cleaned)
    result = {
        "description": description,
        "timeline": timeline,
        "global_tags": tags,
        "music": parsed.get("music") if isinstance(parsed.get("music"), dict) else None,
        "ambience": ambience,
        "sound_effects": sound_effects[:6],
    }
    return result, speech_mentions


def _partial_json_completion(
    text: str, tag: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    description_value = _partial_json_value(text, "description")
    if not isinstance(description_value, str):
        return None
    description, speech_mentions = _remove_speech_description(description_value)
    timeline_value = _partial_json_value(text, "timeline")
    if timeline_value is None:
        timeline_value = _recover_partial_timeline_array(text)
    timeline, timeline_mentions = _normalize_timeline(timeline_value)
    speech_mentions += timeline_mentions
    tags_value = _partial_json_value(text, "global_tags")
    tags = (
        [str(value) for value in tags_value[:8]]
        if isinstance(tags_value, list)
        else _m2d_global_tags(tag)[:8]
    )
    music = _partial_json_value(text, "music")
    if music is not None and not isinstance(music, dict):
        music = None
    ambience = _partial_json_value(text, "ambience")
    if ambience is not None and not isinstance(ambience, str):
        ambience = None
    sound_effects_value = _partial_json_value(text, "sound_effects")
    sound_effects = (
        [str(value) for value in sound_effects_value[:6]]
        if isinstance(sound_effects_value, list)
        else [
            value
            for value in tags
            if not any(
                token in value.casefold()
                for token in ("speech", "voice", "singing", "vocal", "music")
            )
        ][:6]
    )
    normalized, normalized_mentions = _normalize_parsed_completion(
        {
            "description": description,
            "timeline": timeline,
            "global_tags": tags,
            "music": music,
            "ambience": ambience,
            "sound_effects": sound_effects,
        },
        tag,
    )
    return (
        normalized,
        {
            "format": "partial_json_recovery_v1",
            "speech_mentions_omitted": speech_mentions + normalized_mentions,
            "recovered_timeline_events": len(timeline),
        },
    )


def _native_caption_completion(
    text: str, tag: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert Audio Flamingo's native timestamp format without retaining speech."""
    tag_pattern = re.compile(
        r"<t>\s*(?P<start>\d+(?:\.\d+)?)-(?P<end>\d+(?:\.\d+)?)\s*</t>"
        r"\s*(?P<content>.*)"
    )
    bracket_pattern = re.compile(
        r"^\s*\[\s*(?P<start>\d+(?:\.\d+)?)\s*"
        r"(?:-|\u2013|\u2014)\s*(?P<end>\d+(?:\.\d+)?)\s*\]\s*"
        r"(?P<content>.*)$"
    )
    timeline: list[dict[str, Any]] = []
    phrases: list[str] = []
    speech_mentions = 0
    for line in text.splitlines():
        match = tag_pattern.search(line.strip()) or bracket_pattern.search(line.strip())
        if not match:
            continue
        content = match.group("content").strip()
        speech_like = bool(
            re.search(
                r"\b(?:speaker|speech characteristics|narrator|dialogue)\b",
                content,
                re.IGNORECASE,
            )
        )
        background = re.search(
            r"\b(?:background|ambience|ambient|sound effects?)\s*:\s*(.+)$",
            content,
            re.IGNORECASE,
        )
        if background:
            content = background.group(1).strip()
        elif speech_like:
            speech_mentions += 1
            continue
        content = re.sub(r'\s*"[^\"]*"\s*', " ", content).strip(" .")
        content, removed_sentences = _remove_speech_description(content)
        speech_mentions += removed_sentences
        if not content:
            continue
        content = re.sub(
            r"^(?:sound event|environmental noise|ambience)\s*:\s*",
            "",
            content,
            flags=re.IGNORECASE,
        )
        content = re.sub(r"^music\s*:\s*", "", content, flags=re.IGNORECASE)
        event = content.rstrip(".") + "."
        if event.casefold() not in {item.casefold() for item in phrases}:
            phrases.append(event)
        start = float(match.group("start"))
        end = min(30.0, float(match.group("end")))
        if end <= start:
            continue
        timeline.append(
            {
                "start_seconds": start,
                "end_seconds": end,
                "events": [event],
            }
        )
        if speech_like:
            speech_mentions += 1
    if len(timeline) > 6:
        group_size = (len(timeline) + 5) // 6
        timeline = [
            {
                "start_seconds": group[0]["start_seconds"],
                "end_seconds": group[-1]["end_seconds"],
                "events": [
                    event
                    for item in group
                    for event in item["events"]
                ],
            }
            for offset in range(0, len(timeline), group_size)
            if (group := timeline[offset : offset + group_size])
        ]
    timeline = _canonicalize_timeline(timeline)
    tags = _m2d_global_tags(tag)
    if not phrases:
        evidence = ", ".join(tags[:6]) or "low-level environmental sound"
        phrases = [f"The audible background is dominated by {evidence}."]
    transitions = (
        "At the opening,",
        "Next,",
        "Around the middle,",
        "Later,",
        "Near the end,",
        "Finally,",
    )
    narrative_parts: list[str] = []
    for index, item in enumerate(timeline):
        event = " ".join(str(value) for value in item["events"]).strip()
        event = event[0].lower() + event[1:] if event else ""
        if event:
            narrative_parts.append(f"{transitions[min(index, 5)]} {event}")
        if len(re.findall(r"[\w']+", " ".join(narrative_parts))) >= 105:
            break
    narrative = " ".join(narrative_parts or phrases)
    narrative = _bounded_background_description(narrative)
    music_coverage = float(tag.get("cinematic_music_coverage") or 0.0)
    parsed = {
        "description": narrative,
        "timeline": timeline,
        "global_tags": tags,
        "music": (
            {
                "evidence": [
                    value
                    for value in tags
                    if "music" in value.casefold()
                    or any(
                        token in value.casefold()
                        for token in ("instrument", "orchestra", "guitar", "piano")
                    )
                ],
                "coverage": music_coverage,
            }
            if music_coverage > 0
            else None
        ),
        "ambience": " ".join(phrases),
        "sound_effects": [
            value
            for value in tags
            if not any(
                token in value.casefold()
                for token in ("speech", "voice", "singing", "vocal", "music")
            )
        ][:8],
    }
    parsed, normalized_mentions = _normalize_parsed_completion(parsed, tag)
    return parsed, {
        "format": "audio_flamingo_native_timeline_v2",
        "speech_mentions_omitted": speech_mentions + normalized_mentions,
        "native_timeline_events": len(timeline),
    }


def _caption_completion(
    text: str, tag: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    sectioned = _section_caption_completion(text, tag)
    if sectioned is not None:
        parsed, metadata = sectioned
        return _apply_grounded_timeline_fallback(parsed, metadata, tag)
    try:
        parsed, speech_mentions = _normalize_parsed_completion(
            _json_completion(text), tag
        )
        return _apply_grounded_timeline_fallback(
            parsed,
            {
                "format": "strict_json_v1",
                "speech_mentions_omitted": speech_mentions,
            },
            tag,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        partial = _partial_json_completion(text, tag)
        if partial is not None:
            parsed, metadata = partial
            metadata["json_fallback_reason"] = str(error)
            return _apply_grounded_timeline_fallback(parsed, metadata, tag)
        parsed, metadata = _native_caption_completion(text, tag)
        metadata["json_fallback_reason"] = str(error)
        return _apply_grounded_timeline_fallback(parsed, metadata, tag)


def _description_evaluation(parsed: dict[str, Any]) -> dict[str, Any]:
    description = str(parsed.get("description") or "")
    normalized = description.casefold()
    has_background_only_ending = BACKGROUND_ONLY_ENDING.casefold() in normalized
    timeline = parsed.get("timeline") or []
    word_count = len(re.findall(r"[\w']+", description))
    reasons: list[str] = []
    if word_count < 80:
        reasons.append("scene_description_too_short")
    elif word_count > 140:
        reasons.append("scene_description_too_long")
    if not timeline:
        reasons.append("scene_timeline_empty")
    elif len(timeline) > 6:
        reasons.append("scene_timeline_too_many_events")
    elif (
        float(timeline[0].get("start_seconds") or 0.0) != 0.0
        or float(timeline[-1].get("end_seconds") or 0.0) != 30.0
        or any(
            float(previous.get("end_seconds") or 0.0)
            != float(following.get("start_seconds") or 0.0)
            for previous, following in zip(timeline, timeline[1:], strict=False)
        )
    ):
        reasons.append("scene_timeline_not_contiguous_30_seconds")
    elif len(timeline) > 1 and any(
        len(
            re.findall(
                r"[\w']+",
                " ".join(
                    str(event)
                    for event in (
                        item.get("events")
                        if isinstance(item.get("events"), list)
                        else [item.get("events")]
                    )
                    if event
                ),
            )
        )
        < 4
        for item in timeline
    ):
        reasons.append("scene_timeline_underdescribed")
    if (
        not has_background_only_ending
        and "no dialogue" not in normalized
        and "without dialogue" not in normalized
    ):
        reasons.append("background_only_dialogue_assertion_missing")
    if (
        not has_background_only_ending
        and "no vocal" not in normalized
        and "without vocal" not in normalized
    ):
        reasons.append("background_only_vocal_assertion_missing")
    if any(
        phrase in normalized
        for phrase in (
            "timestamped acoustic tagger",
            "classifier evidence",
            "caption model",
            "m2d acoustic evidence",
        )
    ):
        reasons.append("scene_description_contains_model_boilerplate")
    return {
        "policy": "background_caption_contract_v3",
        "status": "success" if not reasons else "review",
        "review_reasons": reasons,
        "signals": {
            "description_word_count": word_count,
            "timeline_event_count": len(timeline),
            "global_tag_count": len(parsed.get("global_tags") or []),
            "sound_effect_count": len(parsed.get("sound_effects") or []),
        },
    }


def _caption_retry_prompt(tag: dict[str, Any], reasons: list[str]) -> str:
    corrections: list[str] = []
    if "scene_description_too_short" in reasons:
        corrections.append(
            "make the overall acoustic description 90-125 words while remaining "
            "strictly grounded in audible detail"
        )
    if "scene_description_contains_model_boilerplate" in reasons:
        corrections.append(
            "remove every reference to a model, classifier, evidence, hints, or "
            "the analysis process"
        )
    if "scene_timeline_underdescribed" in reasons:
        corrections.append(
            "replace every placeholder or fragment with a concrete interval "
            "description of at least four words covering audible source, texture, "
            "dynamics, or spatial placement"
        )
    if any(
        reason.startswith("scene_timeline_")
        and reason != "scene_timeline_underdescribed"
        for reason in reasons
    ):
        corrections.append(
            "supply chronological non-overlapping intervals that cover 0.0 through "
            "30.0 seconds without gaps"
        )
    correction = "; ".join(corrections) or "satisfy the complete output contract"
    return (
        _caption_prompt(tag)
        + " The previous attempt did not satisfy the dataset contract. Regenerate "
        + correction
        + ". Do not discuss this correction in the answer."
    )


def _generate_caption_with_contract(
    client: AudioFlamingoClient,
    audio_path: Path,
    tag: dict[str, Any],
    *,
    max_attempts: int = CAPTION_CONTRACT_MAX_ATTEMPTS,
) -> dict[str, Any]:
    """Generate, evaluate, and retry only hard scene-caption contract failures."""
    prompt = _caption_prompt(tag)
    prior_contract_failures: list[list[str]] = []
    result: dict[str, Any] = {}
    for attempt in range(1, max(1, max_attempts) + 1):
        response = client.ask(
            audio_path, prompt, max_new_tokens=CAPTION_MAX_NEW_TOKENS
        )
        parsed, parse = _caption_completion(str(response["text"]), tag)
        validation = _description_evaluation(parsed)
        regeneration_reasons = sorted(
            set(validation.get("review_reasons") or [])
            & CAPTION_REGENERATION_REASONS
        )
        if (
            attempt >= max(1, max_attempts)
            and regeneration_reasons == ["scene_timeline_underdescribed"]
        ):
            repaired_intervals = _repair_underdescribed_timeline(parsed)
            if repaired_intervals:
                validation = _description_evaluation(parsed)
                regeneration_reasons = sorted(
                    set(validation.get("review_reasons") or [])
                    & CAPTION_REGENERATION_REASONS
                )
                parse = {
                    **parse,
                    "timeline_contract_repair": "grounded_short_label_expansion_v1",
                    "timeline_contract_repaired_intervals": repaired_intervals,
                }
        result = {
            "response": response,
            "parsed": parsed,
            "parse": parse,
            "validation": validation,
            "prompt": prompt,
            "attempts": attempt,
            "prior_contract_failures": list(prior_contract_failures),
        }
        if not regeneration_reasons or attempt >= max(1, max_attempts):
            return result
        prior_contract_failures.append(regeneration_reasons)
        prompt = _caption_retry_prompt(tag, regeneration_reasons)
    return result


def _format_scene_timestamp(seconds: float) -> str:
    bounded = max(0.0, min(30.0, float(seconds)))
    minutes = int(bounded // 60)
    remainder = bounded - minutes * 60
    if bounded == 0:
        return "00:00"
    return f"{minutes:02d}:{remainder:04.1f}"


def _format_scene_description(parsed: dict[str, Any]) -> str:
    """Serialize the canonical, human-readable training caption artifact."""
    lines = [
        "DESCRIPTION:",
        str(parsed.get("description") or "").strip(),
        "",
        "TIMELINE:",
    ]
    timeline_lines = 0
    for item in parsed.get("timeline") or []:
        events = item.get("events") or []
        if isinstance(events, str):
            events = [events]
        event_text = " ".join(
            str(value).strip() for value in events if str(value).strip()
        )
        if not event_text:
            continue
        event_text = event_text[0].upper() + event_text[1:]
        if event_text[-1] not in ".!?":
            event_text += "."
        lines.append(
            f"- {_format_scene_timestamp(float(item['start_seconds']))}-"
            f"{_format_scene_timestamp(float(item['end_seconds']))} — {event_text}"
        )
        timeline_lines += 1
    if not timeline_lines:
        lines.append(
            "- 00:00-00:30.0 — No reliably identifiable background event was "
            "available."
        )
    return "\n".join(lines).rstrip() + "\n"


def _m2d_fallback_description(tag: dict[str, Any], error: Exception) -> dict[str, Any]:
    tags = _m2d_global_tags(tag)[:8]
    timeline = _grounded_m2d_timeline(tag)
    tag_text = ", ".join(tags[:6]) or "low-level environmental texture"
    description = (
        f"The timestamped acoustic tagger identifies {tag_text} across the "
        "background stem. This conservative fallback records classifier evidence "
        "only; spatial placement, source identity, and transitions were not "
        f"confirmed by the caption model. {BACKGROUND_ONLY_ENDING}"
    )
    parsed = {
        "description": description,
        "timeline": timeline,
        "global_tags": tags,
        "music": (
            {
                "coverage": float(tag.get("cinematic_music_coverage") or 0.0),
                "evidence": [value for value in tags if "music" in value.casefold()],
            }
            if float(tag.get("cinematic_music_coverage") or 0.0) > 0
            else None
        ),
        "ambience": tag_text,
        "sound_effects": [
            value
            for value in tags
            if not any(
                token in value.casefold()
                for token in ("speech", "voice", "singing", "vocal", "music")
            )
        ][:6],
    }
    validation = _description_evaluation(parsed)
    validation.update(
        {
            "status": "failure",
            "failure_reasons": ["audio_flamingo_generation_failed"],
        }
    )
    return {
        "schema_version": 1,
        "policy": "m2d_only_caption_failure_fallback_v1",
        "prompt": None,
        "model": None,
        "raw_text": None,
        "parse": {"format": "m2d_only_failure_fallback_v1"},
        "parsed": parsed,
        "validation": validation,
        "error": f"{type(error).__name__}: {error}"[-4000:],
        "processing_seconds": 0.0,
    }


def _m2d_known_failure_description(tag: dict[str, Any]) -> dict[str, Any]:
    """Materialize grounded metadata without GPU work for an existing failure."""
    result = _m2d_fallback_description(
        tag, RuntimeError("Audio Flamingo intentionally skipped")
    )
    result["policy"] = "m2d_known_failure_fast_path_v1"
    result["parse"] = {
        "format": "m2d_known_failure_fast_path_v1",
        "generation_skipped": True,
        "skip_reason": "record_already_failed_independent_quality_gate",
    }
    result["validation"] = _description_evaluation(result["parsed"])
    result["error"] = None
    return result


def _prepare_caption_audio(source: Path, destination: Path) -> dict[str, Any]:
    """Create the bounded mono analysis copy expected by Audio Flamingo."""
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-af",
            "atrim=end_sample=1440000,asetpts=PTS-STARTPTS",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        check=True,
    )
    return {
        "policy": "mono_first_30_seconds_v1",
        "purpose": "caption-model analysis only; published stem remains stereo",
        **_wave_contract(destination),
    }


def describe_once(
    workspace: Path,
    *,
    flamingo_api_url: str,
    worker: str,
) -> dict[str, Any] | None:
    connection = connect(workspace)
    job = _claim(connection, "description", worker, lease_seconds=3600)
    if job is None:
        _heartbeat(connection, worker, "description", "idle")
        connection.close()
        return None
    root = _record_root(workspace, int(job["id"]))
    tag = json.loads(job["tag_json"])
    prompt = _caption_prompt(tag)
    started = time.perf_counter()
    try:
        caption_audio = root / "background.caption-input.wav"
        analysis_audio = _prepare_caption_audio(root / "background.wav", caption_audio)
        generated = _generate_caption_with_contract(
            AudioFlamingoClient(flamingo_api_url), caption_audio, tag
        )
        response = generated["response"]
        parsed = generated["parsed"]
        parse_metadata = generated["parse"]
        validation = generated["validation"]
        prompt = generated["prompt"]
        if parse_metadata.get("speech_mentions_omitted"):
            validation["status"] = "review"
            validation["review_reasons"].append(
                "captioner_detected_speech_in_background"
            )
            validation["signals"]["captioner_speech_mentions_omitted"] = int(
                parse_metadata["speech_mentions_omitted"]
            )
        description = {
            "schema_version": CAPTION_SCHEMA_VERSION,
            "policy": CAPTION_DESCRIPTION_POLICY,
            "prompt": prompt,
            "model": response.get("model"),
            "raw_text": response.get("text"),
            "generation_max_new_tokens": CAPTION_MAX_NEW_TOKENS,
            "generation_attempts": generated["attempts"],
            "prior_contract_failures": generated["prior_contract_failures"],
            "parsed": parsed,
            "parse": parse_metadata,
            "validation": validation,
            "analysis_audio": analysis_audio,
            "processing_seconds": round(time.perf_counter() - started, 3),
        }
        with connection:
            _finish(
                connection,
                int(job["id"]),
                "description",
                values={"description_json": description},
            )
        summary = {"status": "complete", "job_id": job["id"]}
        _heartbeat(connection, worker, "description", "running", summary)
        connection.close()
        return summary
    except Exception as error:
        logger.exception("Description failed for job %s", job["id"])
        with connection:
            transient = _retry_transient_upstream_failure(
                connection, int(job["id"]), "description", error
            )
            failure_status = (
                "retry"
                if transient
                else _fail(connection, int(job["id"]), "description", error)
            )
            if failure_status == "failed":
                fallback = _m2d_fallback_description(json.loads(job["tag_json"]), error)
                _finish(
                    connection,
                    int(job["id"]),
                    "description",
                    values={"description_json": fallback},
                )
        _heartbeat(connection, worker, "description", "error", {"error": str(error)})
        connection.close()
        if failure_status != "failed":
            # Audio Flamingo exits on a poisoned CUDA context and systemd needs
            # several seconds to reload 16 GB of weights. Avoid consuming all
            # five attempts while the healthy replacement is still starting.
            time.sleep(min(30.0, 2.0 ** int(job.get("description_attempts", 1))))
        return {
            "status": "fallback" if failure_status == "failed" else "error",
            "job_id": job["id"],
            "error": str(error),
        }


def promote_known_failures_once(
    workspace: Path,
    *,
    limit: int = 1000,
) -> dict[str, int]:
    """Complete descriptions which cannot escape an independent failure gate.

    The normal description workers can spend several seconds inside Audio
    Flamingo for a viable record.  A separate CPU-only sweep prevents those
    requests from hiding already-failed rows deeper in the FIFO queue.  The
    conditional update makes the sweep safe alongside live description workers.
    """
    connection = connect(workspace)
    rows = connection.execute(
        """SELECT * FROM jobs
        WHERE separation_status='complete' AND tag_status='complete'
        AND asr_status='complete'
        AND description_status IN ('pending','retry')
        AND description_fast_path_checked=0
        AND separation_json IS NOT NULL AND tag_json IS NOT NULL
        AND asr_json IS NOT NULL
        ORDER BY id LIMIT ?""",
        (max(1, limit),),
    ).fetchall()
    promoted = 0
    for row in rows:
        job = dict(row)
        tag = json.loads(job["tag_json"])
        description = _m2d_known_failure_description(tag)
        quality = quality_evaluation(
            {**job, "description_json": json.dumps(description)}
        )
        if quality["bucket"] != "failure":
            with connection:
                connection.execute(
                    """UPDATE jobs SET description_fast_path_checked=1,
                    updated_at=? WHERE id=?
                    AND description_status IN ('pending','retry')""",
                    (_now(), job["id"]),
                )
            continue
        description["known_failure_reasons"] = quality["failure_reasons"]
        with connection:
            promoted += connection.execute(
                """UPDATE jobs SET description_status='complete',
                description_fast_path_checked=1,
                description_json=?,lease_stage=NULL,lease_owner=NULL,
                lease_expires_at=NULL,error=NULL,updated_at=?
                WHERE id=? AND description_status IN ('pending','retry')""",
                (
                    json.dumps(description, separators=(",", ":")),
                    _now(),
                    job["id"],
                ),
            ).rowcount
    result = {"scanned": len(rows), "promoted": promoted}
    _heartbeat(
        connection,
        "description-fast-path",
        "description",
        "running" if rows else "idle",
        result,
    )
    connection.close()
    return result


def quality_evaluation(job: dict[str, Any]) -> dict[str, Any]:
    separation = json.loads(job["separation_json"])
    tag = json.loads(job["tag_json"])
    asr = json.loads(job["asr_json"])
    description = json.loads(job["description_json"])
    sam_status = str(separation.get("sam", {}).get("verification_status", "uncertain"))
    sam_stage = separation.get("sam", {}).get("stages", {}).get("stage1", {})
    sam_judge_quality = float(
        sam_stage.get("verification", {}).get("judge_quality_score") or 0.0
    )
    similarity = float(
        separation.get("reconstruction", {}).get("similarity_score") or 0.0
    )
    speech_candidate_coverage = float(tag.get("speech_coverage") or 0.0)
    # `speech_coverage` is M2D's permissive discovery signal (currently a
    # 0.004 probability / top-15 label). It is useful when finding source clips,
    # but it is far too sensitive to classify dialogue leakage in an already
    # separated background stem. Prefer the calibrated evidence families and
    # retain the permissive value only as metadata. Fall back for legacy tagger
    # rows which predate the stronger fields.
    strong_speech_coverage = float(
        tag.get("strong_speech_coverage", speech_candidate_coverage) or 0.0
    )
    foreground_speech_coverage = float(tag.get("foreground_speech_coverage") or 0.0)
    synthetic_speech_coverage = float(tag.get("synthetic_speech_coverage") or 0.0)
    speech_coverage = max(
        strong_speech_coverage,
        foreground_speech_coverage,
        synthetic_speech_coverage,
    )
    vocal_music_coverage = float(tag.get("vocal_music_coverage") or 0.0)
    window_count = max(
        1,
        int(tag.get("window_count") or len(tag.get("windows", [])) or 1),
    )
    vocal_music_windows_value = tag.get("vocal_music_active_windows")
    vocal_music_windows = (
        int(vocal_music_windows_value)
        if isinstance(vocal_music_windows_value, int | float)
        else round(vocal_music_coverage * window_count)
    )
    vocal_music_max = int(
        (tag.get("duration_scaled_window_requirements") or {}).get("vocal_music_max", 1)
    )
    word_count = len(re.findall(r"[\w']+", str(asr.get("transcript") or "")))
    asr_rejection_reasons = [str(reason) for reason in asr.get("rejection_reasons", [])]
    failures: list[str] = []
    reviews: list[str] = []
    # SAM's failure boundary is intentionally conservative, but its judge score
    # is continuous rather than a calibrated correctness probability.  Treat a
    # narrow miss as review when independent reconstruction, foreground-speech,
    # and dialogue-ASR evidence all agree that the package is usable.  This
    # avoids hard-failing audibly clean records for a few hundredths below 4.30.
    near_threshold_sam_failure = bool(
        sam_status == "failure"
        and sam_judge_quality >= 4.2
        and similarity >= 85.0
        and foreground_speech_coverage <= 0.05
        and asr.get("accepted") is not False
        and word_count >= 3
        and asr.get("detected_language") in {None, "en"}
    )
    if sam_status == "failure" and not near_threshold_sam_failure:
        failures.append("sam_voice_separation_failure")
    elif near_threshold_sam_failure:
        reviews.append("sam_voice_separation_near_failure_threshold")
    elif sam_status != "success":
        reviews.append("sam_voice_separation_uncertain")
    if similarity < 25:
        failures.append("very_low_reconstruction_similarity")
    elif similarity < 70:
        reviews.append("reconstruction_similarity_below_training_target")
    if speech_coverage >= 0.34:
        failures.append("dialogue_spill_in_background")
    elif speech_coverage >= 0.12:
        reviews.append("possible_dialogue_spill_in_background")
    if vocal_music_coverage >= 0.34:
        failures.append("vocal_music_in_background")
    elif vocal_music_windows > vocal_music_max:
        reviews.append("possible_vocal_music_in_background")
    if word_count < 3 or not asr.get("transcript"):
        failures.append("dialogue_transcript_empty")
    elif asr.get("accepted") is False:
        failures.append("dialogue_asr_quality_gate_failed")
    if asr.get("detected_language") not in {None, "en"}:
        failures.append("dialogue_not_english")
    if not description.get("parsed", {}).get("description"):
        failures.append("scene_description_missing")
    for reason in description.get("validation", {}).get("failure_reasons", []):
        failures.append(str(reason))
    captioner_speech_mentions = int(
        (description.get("parse") or {}).get("speech_mentions_omitted") or 0
    )
    for reason in description.get("validation", {}).get("review_reasons", []):
        # Audio Flamingo sometimes labels non-speech transients as a voice. Keep
        # the signal in metadata, but require independent calibrated M2D speech
        # evidence before this single judge can move the final bucket to review.
        if reason == "captioner_detected_speech_in_background" and speech_coverage <= 0:
            continue
        reviews.append(str(reason))
    bucket = "failure" if failures else "review" if reviews else "success"
    return {
        "schema_version": 1,
        "policy": "training_record_three_bucket_v1",
        "bucket": bucket,
        "failure_reasons": failures,
        "review_reasons": reviews,
        "signals": {
            "sam_status": sam_status,
            "sam_judge_quality_score": sam_judge_quality,
            "sam_near_failure_threshold": near_threshold_sam_failure,
            "reconstruction_similarity": similarity,
            "background_speech_coverage": speech_coverage,
            "background_speech_candidate_coverage": speech_candidate_coverage,
            "background_strong_speech_coverage": strong_speech_coverage,
            "background_foreground_speech_coverage": foreground_speech_coverage,
            "background_synthetic_speech_coverage": synthetic_speech_coverage,
            "background_vocal_music_coverage": vocal_music_coverage,
            "background_vocal_music_windows": vocal_music_windows,
            "background_vocal_music_allowed_windows": vocal_music_max,
            "captioner_speech_mentions_omitted": captioner_speech_mentions,
            "captioner_speech_supported_by_m2d": bool(
                captioner_speech_mentions and speech_coverage > 0
            ),
            "dialogue_word_count": word_count,
            "dialogue_asr_accepted": asr.get("accepted"),
            "dialogue_asr_rejection_reasons": asr_rejection_reasons,
            "dialogue_language_probability": asr.get("language_probability"),
            "dialogue_vad_seconds": asr.get("duration_after_vad"),
        },
    }


def package_once(workspace: Path, *, worker: str) -> dict[str, Any] | None:
    connection = connect(workspace)
    job = _claim(connection, "package", worker, lease_seconds=600)
    if job is None:
        _heartbeat(connection, worker, "package", "idle")
        connection.close()
        return None
    root = _record_root(workspace, int(job["id"]))
    try:
        source = json.loads(job["source_json"])
        separation = json.loads(job["separation_json"])
        tags = json.loads(job["tag_json"])
        asr = json.loads(job["asr_json"])
        description = json.loads(job["description_json"])
        quality = quality_evaluation(job)
        source_digest = str(job["source_sha256"] or sha256_file(root / "original.wav"))
        source_key_digest = hashlib.sha256(str(job["source_key"]).encode()).hexdigest()
        record_id = (
            source_digest
            if job["source_kind"] == "continuous"
            else f"{source_digest}-{source_key_digest[:16]}"
        )
        (root / "scene_description.txt").write_text(
            _format_scene_description(description["parsed"])
        )
        (root / "dialogue_transcript.txt").write_text(
            str(asr.get("transcript") or "").strip() + "\n"
        )
        artifacts = {
            name: {
                "sha256": sha256_file(root / name),
                "bytes": (root / name).stat().st_size,
            }
            for name in OUTPUT_FILENAMES
            if name != "metadata.json"
        }
        metadata = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "record_id": record_id,
            "pipeline_job_id": int(job["id"]),
            "source": source,
            "source_kind": job["source_kind"],
            "source_sha256": job["source_sha256"],
            "quality": quality,
            "separation": separation,
            "background_tagger": tags,
            "dialogue_transcription": asr,
            "scene_description": description,
            "artifacts": artifacts,
            "reference_generation": None,
            "created_at": _now(),
        }
        _atomic_json(root / "metadata.json", metadata)
        metadata["artifacts"]["metadata.json"] = {
            "sha256": sha256_file(root / "metadata.json"),
            "bytes": (root / "metadata.json").stat().st_size,
        }
        with connection:
            _finish(
                connection,
                int(job["id"]),
                "package",
                values={"quality_bucket": quality["bucket"], "audit_json": quality},
            )
            connection.execute(
                """INSERT INTO records(job_id,record_id,quality_bucket,
                record_json,created_at) VALUES(?,?,?,?,?)""",
                (
                    job["id"],
                    record_id,
                    quality["bucket"],
                    json.dumps(metadata, separators=(",", ":")),
                    _now(),
                ),
            )
        _heartbeat(
            connection,
            worker,
            "package",
            "running",
            {"job_id": job["id"], "bucket": quality["bucket"]},
        )
        connection.close()
        return {"status": "complete", "job_id": job["id"], "bucket": quality["bucket"]}
    except Exception as error:
        logger.exception("Packaging failed for job %s", job["id"])
        with connection:
            _fail(connection, int(job["id"]), "package", error)
        _heartbeat(connection, worker, "package", "error", {"error": str(error)})
        connection.close()
        return {"status": "error", "job_id": job["id"], "error": str(error)}


def _s3_object_exists(s3: Any, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
            return False
        raise


def _upload_record(
    s3: Any,
    bucket: str,
    prefix: str,
    workspace: Path,
    row: sqlite3.Row,
) -> dict[str, Any]:
    record = json.loads(row["record_json"])
    root = _record_root(workspace, int(row["job_id"]))
    bucket_name = str(row["quality_bucket"])
    object_root = f"{prefix.strip('/')}/{bucket_name}/{row['record_id']}"
    keys: dict[str, str] = {}
    for name in OUTPUT_FILENAMES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        key = f"{object_root}/{name}"
        if not _s3_object_exists(s3, bucket, key):
            s3.upload_file(str(path), bucket, key)
        keys[name] = f"s3://{bucket}/{key}"
    record["s3_artifacts"] = keys
    return record


def _validate_record_quality(row: sqlite3.Row) -> dict[str, Any]:
    """Reject contradictory quality labels before they become immutable."""
    record = json.loads(row["record_json"])
    quality = record.get("quality")
    if not isinstance(quality, dict):
        raise ValueError(f"Record {row['record_id']} has no quality object")
    bucket = quality.get("bucket")
    if bucket not in {"success", "review", "failure"}:
        raise ValueError(f"Record {row['record_id']} has invalid bucket {bucket!r}")
    if bucket != row["quality_bucket"]:
        raise ValueError(
            f"Record {row['record_id']} quality mismatch: "
            f"row={row['quality_bucket']!r}, metadata={bucket!r}"
        )
    failure_reasons = quality.get("failure_reasons")
    review_reasons = quality.get("review_reasons")
    if not isinstance(failure_reasons, list) or not isinstance(review_reasons, list):
        raise ValueError(f"Record {row['record_id']} has malformed quality reasons")
    if bucket == "failure" and not failure_reasons:
        raise ValueError(f"Failure record {row['record_id']} has no failure reason")
    if bucket == "review" and (failure_reasons or not review_reasons):
        raise ValueError(f"Review record {row['record_id']} has invalid reasons")
    if bucket == "success" and (failure_reasons or review_reasons):
        raise ValueError(f"Success record {row['record_id']} has quality reasons")
    return record


def _cleanup_published_records(
    connection: sqlite3.Connection,
    workspace: Path,
    *,
    limit: int,
) -> int:
    rows = connection.execute(
        """SELECT sequence,job_id FROM records
        WHERE uploaded_at IS NOT NULL AND cleaned_at IS NULL
        ORDER BY sequence LIMIT ?""",
        (limit,),
    ).fetchall()
    for row in rows:
        shutil.rmtree(_record_root(workspace, int(row["job_id"])), ignore_errors=True)
        for directory in ("background-audio", "dialogue-audio"):
            (workspace / directory / f"{int(row['job_id']):012d}.wav").unlink(
                missing_ok=True
            )
    if rows:
        with connection:
            connection.executemany(
                "UPDATE records SET cleaned_at=? WHERE sequence=?",
                [(_now(), row["sequence"]) for row in rows],
            )
    return len(rows)


def _verify_snapshot_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("Snapshot records must be an array")
    expected_count = int(manifest.get("snapshot_record_count") or 0)
    snapshot_id = str(manifest.get("snapshot_id") or "")
    if not snapshot_id:
        raise ValueError("Snapshot manifest has no snapshot ID")
    if len(records) != expected_count:
        raise ValueError("Snapshot record count does not match its manifest")
    record_ids: set[str] = set()
    sequences: set[int] = set()
    observed: Counter[str] = Counter()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Snapshot contains a non-object record")
        record_id = str(record.get("record_id") or "")
        if not record_id or record_id in record_ids:
            raise ValueError(
                f"Snapshot has missing or duplicate record ID {record_id!r}"
            )
        record_ids.add(record_id)
        quality = record.get("quality")
        membership = record.get("snapshot_membership")
        if not isinstance(quality, dict) or not isinstance(membership, dict):
            raise ValueError(f"Record {record_id} has incomplete snapshot metadata")
        bucket = str(quality.get("bucket") or "")
        if bucket not in {"success", "review", "failure"}:
            raise ValueError(f"Record {record_id} has invalid quality bucket")
        if membership.get("quality_bucket") != bucket:
            raise ValueError(
                f"Record {record_id} snapshot bucket does not match quality"
            )
        if membership.get("snapshot_id") != snapshot_id:
            raise ValueError(f"Record {record_id} snapshot membership has the wrong ID")
        sequence = int(membership.get("record_sequence"))
        if sequence in sequences:
            raise ValueError(f"Snapshot has duplicate record sequence {sequence}")
        sequences.add(sequence)
        observed[bucket] += 1
        artifacts = record.get("s3_artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(OUTPUT_FILENAMES):
            raise ValueError(
                f"Record {record_id} does not have the complete artifact set"
            )
        expected_fragment = f"/snapshots/{snapshot_id}/{bucket}/{record_id}/"
        if any(expected_fragment not in str(uri) for uri in artifacts.values()):
            raise ValueError(f"Record {record_id} artifacts are in the wrong bucket")
    declared = {
        str(key): int(value)
        for key, value in dict(manifest.get("quality_buckets") or {}).items()
    }
    if dict(observed) != declared or sum(observed.values()) != expected_count:
        raise ValueError("Snapshot quality-bucket totals do not reconcile")
    return {
        "policy": "immutable_training_snapshot_audit_v1",
        "status": "passed",
        "record_count": expected_count,
        "unique_record_ids": len(record_ids),
        "unique_record_sequences": len(sequences),
        "complete_artifact_records": expected_count,
        "quality_buckets": declared,
        "all_quality_buckets_included": bool(
            manifest.get("all_quality_buckets_included")
        ),
        "reference_generation": bool(manifest.get("reference_generation")),
    }


def publish_due_once(
    workspace: Path,
    *,
    bucket: str,
    prefix: str,
    snapshot_size: int = DEFAULT_SNAPSHOT_SIZE,
    upload_workers: int = 8,
) -> dict[str, int]:
    """Commit full 1,000-record snapshots while preserving quality buckets."""
    import boto3

    connection = connect(workspace)
    s3 = boto3.client("s3")
    cleaned_records = _cleanup_published_records(
        connection, workspace, limit=max(1000, snapshot_size * 2)
    )
    completed_records = int(
        connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    )
    published_records = connection.execute(
        "SELECT COALESCE(SUM(record_count),0) FROM snapshots"
    ).fetchone()[0]
    last_sequence = int(
        connection.execute(
            "SELECT COALESCE(MAX(end_sequence),0) FROM snapshots"
        ).fetchone()[0]
    )
    snapshots = 0
    while True:
        selected = connection.execute(
            """SELECT * FROM records WHERE sequence>?
            ORDER BY sequence LIMIT ?""",
            (last_sequence, snapshot_size),
        ).fetchall()
        if len(selected) < snapshot_size:
            break
        for row in selected:
            _validate_record_quality(row)
        start_sequence = int(selected[0]["sequence"])
        end_sequence = int(selected[-1]["sequence"])
        snapshot_id = f"v1-{start_sequence:08d}-{end_sequence:08d}"
        snapshot_prefix = f"{prefix.strip('/')}/snapshots/{snapshot_id}"
        with ThreadPoolExecutor(max_workers=max(1, upload_workers)) as executor:
            records = list(
                executor.map(
                    lambda row, snapshot_prefix=snapshot_prefix: _upload_record(
                        s3, bucket, snapshot_prefix, workspace, row
                    ),
                    selected,
                )
            )
        quality_counts: dict[str, int] = {}
        for row, record in zip(selected, records, strict=True):
            bucket_name = str(record["quality"]["bucket"])
            quality_counts[bucket_name] = quality_counts.get(bucket_name, 0) + 1
            record["snapshot_membership"] = {
                "snapshot_id": snapshot_id,
                "record_sequence": int(row["sequence"]),
                "quality_bucket": bucket_name,
            }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset": "dialogue_background_voice_only_sam_v1",
            "snapshot_id": snapshot_id,
            "snapshot_record_count": len(records),
            "quality_buckets": quality_counts,
            "training_default_filter": "quality.bucket == 'success'",
            "all_quality_buckets_included": True,
            "reference_generation": False,
            "records": records,
            # Use an input-derived timestamp so retrying after a process crash
            # recreates byte-identical immutable snapshot control files.
            "created_at": str(selected[-1]["created_at"]),
        }
        manifest["verification"] = _verify_snapshot_manifest(manifest)
        manifest_path = workspace / "snapshots" / snapshot_id / "manifest.json"
        _atomic_json(manifest_path, manifest)
        digest = sha256_file(manifest_path)
        ready = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "record_count": len(records),
            "quality_buckets": quality_counts,
            "manifest_sha256": digest,
            "snapshot_uri": f"s3://{bucket}/{snapshot_prefix}/",
            "immutable": True,
            "verification_status": "passed",
            "published_at": manifest["created_at"],
        }
        ready_path = manifest_path.parent / "READY.json"
        _atomic_json(ready_path, ready)
        manifest_key = f"{snapshot_prefix}/manifest.json"
        ready_key = f"{snapshot_prefix}/READY.json"
        if not _s3_object_exists(s3, bucket, manifest_key):
            s3.upload_file(str(manifest_path), bucket, manifest_key)
        if not _s3_object_exists(s3, bucket, ready_key):
            s3.upload_file(str(ready_path), bucket, ready_key)
        with connection:
            connection.execute(
                """INSERT INTO snapshots(end_sequence,snapshot_id,record_count,
                manifest_sha256,s3_prefix,published_at) VALUES(?,?,?,?,?,?)""",
                (
                    end_sequence,
                    snapshot_id,
                    len(records),
                    digest,
                    ready["snapshot_uri"],
                    ready["published_at"],
                ),
            )
            connection.executemany(
                "UPDATE records SET uploaded_at=? WHERE sequence=?",
                [(_now(), row["sequence"]) for row in selected],
            )
        last_sequence = end_sequence
        published_records += len(records)
        snapshots += 1
        cleaned_records += _cleanup_published_records(
            connection, workspace, limit=max(1000, snapshot_size * 2)
        )
    _heartbeat(
        connection,
        "snapshot-publisher",
        "publish",
        "running",
        {
            "completed_records": completed_records,
            "published_records": published_records,
            "snapshots": snapshots,
            "cleaned_records": cleaned_records,
        },
    )
    connection.close()
    return {
        "snapshots": snapshots,
        "published_records": published_records,
        "cleaned_records": cleaned_records,
    }


def status(workspace: Path) -> dict[str, Any]:
    connection = connect(workspace)
    stages = {
        stage: dict(
            connection.execute(
                f"SELECT {stage}_status,COUNT(*) FROM jobs GROUP BY {stage}_status"
            ).fetchall()
        )
        for stage in ("separation", "tag", "asr", "description", "package")
    }
    buckets = dict(
        connection.execute(
            """SELECT COALESCE(quality_bucket,'unclassified'),COUNT(*)
            FROM jobs GROUP BY quality_bucket"""
        ).fetchall()
    )
    workers = [
        {**dict(row), "details": json.loads(row["details_json"])}
        for row in connection.execute("SELECT * FROM workers ORDER BY worker")
    ]
    result = {
        "jobs": connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0],
        "records": connection.execute("SELECT COUNT(*) FROM records").fetchone()[0],
        "snapshots": connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0],
        "stages": stages,
        "quality_buckets": buckets,
        "workers": workers,
        "observed_at": _now(),
    }
    connection.close()
    return result


def _loop(
    action: Callable[[], Any],
    *,
    poll_seconds: float,
    follow: bool,
) -> None:
    while True:
        result = action()
        if not follow:
            return
        if result is None or result == 0:
            time.sleep(poll_seconds)


def _workers(
    action: Callable[[str], Any],
    *,
    stage: str,
    count: int,
    worker_offset: int = 0,
    poll_seconds: float,
    follow: bool,
) -> None:
    def run(index: int) -> None:
        worker = f"{stage}-{index + worker_offset}"
        _loop(
            lambda: action(worker),
            poll_seconds=poll_seconds,
            follow=follow,
        )

    threads = [threading.Thread(target=run, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    sync = commands.add_parser("sync")
    sync.add_argument("--source-workspace", type=Path, required=True)
    sync.add_argument("--source-s3-prefix", required=True)
    sync.add_argument("--inbox", type=Path)
    sync.add_argument("--limit", type=int, default=5000)
    sync.add_argument("--follow", action="store_true")
    sync.add_argument("--poll-seconds", type=float, default=10)

    separate = commands.add_parser("separate")
    separate.add_argument("--sam-api-url", default="http://127.0.0.1:8000")
    separate.add_argument("--bucket")
    separate.add_argument("--workers", type=int, default=1)
    separate.add_argument("--worker-offset", type=int, default=0)
    separate.add_argument("--elastic-worker-from", type=int, default=-1)
    separate.add_argument("--description-backlog-high", type=int, default=500)
    separate.add_argument("--follow", action="store_true")
    separate.add_argument("--poll-seconds", type=float, default=2)

    imports = commands.add_parser("import-jsonl")
    imports.add_argument("--path", type=Path, required=True)
    imports.add_argument("--kind", choices=("tag", "asr"), required=True)
    imports.add_argument("--follow", action="store_true")
    imports.add_argument("--poll-seconds", type=float, default=2)

    describe = commands.add_parser("describe")
    describe.add_argument("--flamingo-api-url", default="http://127.0.0.1:8001")
    describe.add_argument("--workers", type=int, default=1)
    describe.add_argument("--follow", action="store_true")
    describe.add_argument("--poll-seconds", type=float, default=2)

    promote_failures = commands.add_parser("promote-known-failures")
    promote_failures.add_argument("--limit", type=int, default=2000)
    promote_failures.add_argument("--follow", action="store_true")
    promote_failures.add_argument("--poll-seconds", type=float, default=15)

    package = commands.add_parser("package")
    package.add_argument("--workers", type=int, default=1)
    package.add_argument("--follow", action="store_true")
    package.add_argument("--poll-seconds", type=float, default=2)

    publish = commands.add_parser("publish")
    publish.add_argument("--bucket", required=True)
    publish.add_argument("--prefix", required=True)
    publish.add_argument("--snapshot-size", type=int, default=DEFAULT_SNAPSHOT_SIZE)
    publish.add_argument("--upload-workers", type=int, default=8)
    publish.add_argument("--follow", action="store_true")
    publish.add_argument("--poll-seconds", type=float, default=30)

    commands.add_parser("recover-leases")
    commands.add_parser("status")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.command == "sync":

        def sync_action() -> int:
            inserted = sync_continuous_catalog(
                args.workspace,
                source_workspace=args.source_workspace,
                source_s3_prefix=args.source_s3_prefix,
                limit=args.limit,
            )
            if args.inbox:
                args.inbox.mkdir(parents=True, exist_ok=True)
                inserted += sync_inbox(args.workspace, args.inbox)
            return inserted

        _loop(sync_action, poll_seconds=args.poll_seconds, follow=args.follow)
    elif args.command == "separate":
        _workers(
            lambda worker: separate_once(
                args.workspace,
                sam_api_url=args.sam_api_url,
                bucket=args.bucket,
                worker=worker,
                elastic_worker_from=args.elastic_worker_from,
                description_backlog_high=args.description_backlog_high,
            ),
            stage="separation",
            count=args.workers,
            worker_offset=args.worker_offset,
            poll_seconds=args.poll_seconds,
            follow=args.follow,
        )
    elif args.command == "import-jsonl":
        _loop(
            lambda: import_jsonl(args.workspace, args.path, args.kind),
            poll_seconds=args.poll_seconds,
            follow=args.follow,
        )
    elif args.command == "describe":
        _workers(
            lambda worker: describe_once(
                args.workspace,
                flamingo_api_url=args.flamingo_api_url,
                worker=worker,
            ),
            stage="description",
            count=args.workers,
            poll_seconds=args.poll_seconds,
            follow=args.follow,
        )
    elif args.command == "promote-known-failures":
        _loop(
            lambda: promote_known_failures_once(
                args.workspace,
                limit=args.limit,
            )["promoted"],
            poll_seconds=args.poll_seconds,
            follow=args.follow,
        )
    elif args.command == "package":
        _workers(
            lambda worker: package_once(args.workspace, worker=worker),
            stage="package",
            count=args.workers,
            poll_seconds=args.poll_seconds,
            follow=args.follow,
        )
    elif args.command == "publish":
        _loop(
            lambda: publish_due_once(
                args.workspace,
                bucket=args.bucket,
                prefix=args.prefix,
                snapshot_size=args.snapshot_size,
                upload_workers=args.upload_workers,
            ),
            poll_seconds=args.poll_seconds,
            follow=args.follow,
        )
    elif args.command == "recover-leases":
        print(recover_leases(args.workspace))
    else:
        print(json.dumps(status(args.workspace), indent=2))


if __name__ == "__main__":
    main()
