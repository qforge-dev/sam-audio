"""Recaption immutable records into the current scene-caption snapshot revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

from .flamingo_client import AudioFlamingoClient
from .training_dataset import (
    CAPTION_DESCRIPTION_POLICY,
    CAPTION_MAX_NEW_TOKENS,
    CAPTION_REGENERATION_REASONS,
    CAPTION_SCHEMA_VERSION,
    _atomic_json,
    _caption_completion,
    _caption_prompt,
    _description_evaluation,
    _format_scene_description,
    _generate_caption_with_contract,
    _prepare_caption_audio,
    _repair_underdescribed_timeline,
    _s3_object_exists,
    _verify_snapshot_manifest,
    connect,
    quality_evaluation,
    sha256_file,
)

logger = logging.getLogger(__name__)
CAPTION_REVISION = CAPTION_SCHEMA_VERSION
DESCRIPTION_POLICY = CAPTION_DESCRIPTION_POLICY
SNAPSHOT_VERSION = f"v{CAPTION_REVISION}"
SNAPSHOT_TABLE = f"caption_v{CAPTION_REVISION}_snapshots"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _initialize(workspace: Path) -> sqlite3.Connection:
    connection = connect(workspace)
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS caption_v2_records (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL UNIQUE,
            source_sequence INTEGER NOT NULL UNIQUE,
            record_id TEXT NOT NULL UNIQUE,
            source_quality_bucket TEXT NOT NULL,
            quality_bucket TEXT,
            source_s3_prefix TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            lease_owner TEXT,
            lease_expires_at REAL,
            description_json TEXT,
            metadata_json TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            uploaded_at TEXT,
            snapshot_id TEXT
        );
        CREATE INDEX IF NOT EXISTS caption_v2_ready
        ON caption_v2_records(status,uploaded_at,source_sequence,lease_expires_at);
        CREATE TABLE IF NOT EXISTS caption_v2_snapshots (
            end_sequence INTEGER PRIMARY KEY,
            snapshot_id TEXT NOT NULL UNIQUE,
            record_count INTEGER NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            s3_prefix TEXT NOT NULL,
            published_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS caption_v3_snapshots (
            end_sequence INTEGER PRIMARY KEY,
            snapshot_id TEXT NOT NULL UNIQUE,
            record_count INTEGER NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            s3_prefix TEXT NOT NULL,
            published_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS caption_v4_snapshots (
            end_sequence INTEGER PRIMARY KEY,
            snapshot_id TEXT NOT NULL UNIQUE,
            record_count INTEGER NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            s3_prefix TEXT NOT NULL,
            published_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS caption_snapshot_membership (
            snapshot_id TEXT NOT NULL,
            record_id TEXT NOT NULL,
            source_sequence INTEGER NOT NULL,
            job_id INTEGER NOT NULL,
            quality_bucket TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            PRIMARY KEY(snapshot_id,record_id)
        );
        CREATE INDEX IF NOT EXISTS caption_snapshot_membership_order
        ON caption_snapshot_membership(snapshot_id,source_sequence);
        CREATE TABLE IF NOT EXISTS caption_revision_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(caption_v2_records)")
    }
    if "snapshot_id" not in columns:
        connection.execute("ALTER TABLE caption_v2_records ADD COLUMN snapshot_id TEXT")
    return connection


def seed_once(workspace: Path, *, limit: int = 5000) -> int:
    """Register source records; current captions need no additional inference."""
    connection = _initialize(workspace)
    target_row = connection.execute(
        """SELECT value FROM caption_revision_state
        WHERE key='target_source_sequence'"""
    ).fetchone()
    target_sequence = int(target_row[0]) if target_row is not None else None
    rows = connection.execute(
        """SELECT r.*,j.description_json FROM records r
        JOIN jobs j ON j.id=r.job_id
        LEFT JOIN caption_v2_records v ON v.job_id=r.job_id
        WHERE r.uploaded_at IS NOT NULL AND v.job_id IS NULL
        AND (? IS NULL OR r.sequence<=?)
        ORDER BY r.sequence LIMIT ?""",
        (target_sequence, target_sequence, max(1, limit)),
    ).fetchall()
    inserted = 0
    with connection:
        for row in rows:
            snapshot = connection.execute(
                """SELECT s3_prefix FROM snapshots WHERE end_sequence>=?
                ORDER BY end_sequence LIMIT 1""",
                (row["sequence"],),
            ).fetchone()
            if snapshot is None:
                continue
            description = json.loads(row["description_json"] or "{}")
            validation_reasons = set(
                _publication_contract_reasons(row["description_json"])
            )
            is_current = (
                int(description.get("schema_version") or 0) == CAPTION_REVISION
                and description.get("policy") == DESCRIPTION_POLICY
                and not validation_reasons
            )
            metadata = json.loads(row["record_json"])
            has_saved_response = description.get("raw_text") is not None
            inserted += connection.execute(
                """INSERT OR IGNORE INTO caption_v2_records(
                job_id,source_sequence,record_id,source_quality_bucket,
                quality_bucket,source_s3_prefix,status,description_json,
                metadata_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    row["job_id"],
                    row["sequence"],
                    row["record_id"],
                    row["quality_bucket"],
                    row["quality_bucket"] if is_current else None,
                    snapshot["s3_prefix"],
                    (
                        "complete"
                        if is_current
                        else "reparse" if has_saved_response else "pending"
                    ),
                    row["description_json"] if is_current else None,
                    (
                        json.dumps(metadata, separators=(",", ":"))
                        if is_current
                        else None
                    ),
                    row["created_at"],
                    _now(),
                ),
            ).rowcount
    connection.close()
    return inserted + reuse_existing_once(workspace, limit=limit)


def reuse_existing_once(workspace: Path, *, limit: int = 5000) -> int:
    """Reparse saved AF output without spending another GPU inference."""
    connection = _initialize(workspace)
    rows = connection.execute(
        """SELECT v.sequence,v.source_s3_prefix,j.tag_json,j.separation_json,
        j.asr_json,COALESCE(v.description_json,j.description_json)
        AS saved_description_json,r.record_json
        FROM caption_v2_records v JOIN jobs j ON j.id=v.job_id
        JOIN records r ON r.job_id=v.job_id
        WHERE v.status='reparse' AND v.attempts=0
        AND (json_extract(v.description_json,'$.raw_text') IS NOT NULL
        OR json_extract(j.description_json,'$.raw_text') IS NOT NULL)
        ORDER BY v.source_sequence LIMIT ?""",
        (max(1, limit),),
    ).fetchall()
    upgraded = 0
    with connection:
        for row in rows:
            previous = json.loads(row["saved_description_json"])
            tag = json.loads(row["tag_json"])
            parsed, parse = _caption_completion(str(previous["raw_text"]), tag)
            validation = _description_evaluation(parsed)
            regeneration_reasons = sorted(
                set(validation.get("review_reasons") or [])
                & CAPTION_REGENERATION_REASONS
            )
            if regeneration_reasons == ["scene_timeline_underdescribed"]:
                repaired_intervals = _repair_underdescribed_timeline(parsed)
                if repaired_intervals:
                    validation = _description_evaluation(parsed)
                    regeneration_reasons = sorted(
                        set(validation.get("review_reasons") or [])
                        & CAPTION_REGENERATION_REASONS
                    )
                    parse = {
                        **parse,
                        "timeline_contract_repair": (
                            "grounded_saved_short_label_expansion_v1"
                        ),
                        "timeline_contract_repaired_intervals": repaired_intervals,
                    }
            if regeneration_reasons:
                connection.execute(
                    """UPDATE caption_v2_records SET status='pending',attempts=1,
                    error=?,updated_at=?
                    WHERE sequence=? AND status='reparse' AND attempts=0""",
                    (
                        f"saved_caption_failed_v{CAPTION_REVISION}_contract:"
                        + ",".join(regeneration_reasons),
                        _now(),
                        row["sequence"],
                    ),
                )
                continue
            if parse.get("speech_mentions_omitted"):
                validation["status"] = "review"
                validation["review_reasons"].append(
                    "captioner_detected_speech_in_background"
                )
                validation["signals"]["captioner_speech_mentions_omitted"] = int(
                    parse["speech_mentions_omitted"]
                )
            description = {
                **previous,
                "schema_version": CAPTION_REVISION,
                "policy": DESCRIPTION_POLICY,
                "parsed": parsed,
                "parse": parse,
                "validation": validation,
                "revision": {
                    "policy": "reuse_saved_audio_flamingo_response_v1",
                    "original_policy": previous.get("policy"),
                    "reparsed_at": _now(),
                },
            }
            quality = quality_evaluation(
                {
                    "separation_json": row["separation_json"],
                    "tag_json": row["tag_json"],
                    "asr_json": row["asr_json"],
                    "description_json": json.dumps(description),
                }
            )
            metadata = json.loads(row["record_json"])
            metadata["schema_version"] = 2
            metadata["quality"] = quality
            metadata["scene_description"] = description
            metadata["caption_revision"] = {
                "version": CAPTION_REVISION,
                "source_snapshot": row["source_s3_prefix"],
                "inference_reused": True,
                "revised_at": _now(),
            }
            upgraded += connection.execute(
                """UPDATE caption_v2_records SET status='complete',
                quality_bucket=?,description_json=?,metadata_json=?,updated_at=?
                WHERE sequence=? AND status='reparse' AND attempts=0""",
                (
                    quality["bucket"],
                    json.dumps(description, separators=(",", ":")),
                    json.dumps(metadata, separators=(",", ":")),
                    _now(),
                    row["sequence"],
                ),
            ).rowcount
    connection.close()
    return upgraded


def _claim(workspace: Path, worker: str) -> dict[str, Any] | None:
    connection = _initialize(workspace)
    now = time.time()
    connection.execute("BEGIN IMMEDIATE")
    row = connection.execute(
        """SELECT v.*,j.tag_json,j.separation_json,j.asr_json,j.source_json,
        j.source_kind,j.source_sha256,j.source_key
        FROM caption_v2_records v JOIN jobs j ON j.id=v.job_id
        WHERE v.status='pending'
        AND (v.lease_expires_at IS NULL OR v.lease_expires_at<?)
        -- Fresh work must not be starved by a handful of difficult early
        -- records which repeatedly return to pending after contract retries.
        ORDER BY v.attempts,v.source_sequence LIMIT 1""",
        (now,),
    ).fetchone()
    if row is None:
        connection.rollback()
        connection.close()
        return None
    connection.execute(
        """UPDATE caption_v2_records SET status='running',attempts=attempts+1,
        lease_owner=?,lease_expires_at=?,updated_at=? WHERE sequence=?""",
        (worker, now + 1800, _now(), row["sequence"]),
    )
    connection.commit()
    result = dict(row)
    connection.close()
    return result


def _split_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {uri}")
    bucket, _, key = uri[5:].partition("/")
    return bucket, key.rstrip("/")


def describe_once(
    workspace: Path,
    *,
    api_url: str,
    worker: str,
) -> dict[str, Any] | None:
    row = _claim(workspace, worker)
    if row is None:
        return None
    connection = _initialize(workspace)
    try:
        source_bucket, source_prefix = _split_s3_uri(row["source_s3_prefix"])
        source_key = (
            f"{source_prefix}/{row['source_quality_bucket']}/"
            f"{row['record_id']}/background.wav"
        )
        tag = json.loads(row["tag_json"])
        prompt = _caption_prompt(tag)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(
            prefix=f"caption-v{CAPTION_REVISION}-"
        ) as temporary:
            source = Path(temporary) / "background.source.wav"
            prepared = Path(temporary) / "background.caption.wav"
            boto3.client("s3").download_file(source_bucket, source_key, str(source))
            analysis_audio = _prepare_caption_audio(source, prepared)
            generated = _generate_caption_with_contract(
                AudioFlamingoClient(api_url), prepared, tag
            )
        response = generated["response"]
        parsed = generated["parsed"]
        parse = generated["parse"]
        validation = generated["validation"]
        prompt = generated["prompt"]
        if parse.get("speech_mentions_omitted"):
            validation["status"] = "review"
            validation["review_reasons"].append(
                "captioner_detected_speech_in_background"
            )
            validation["signals"]["captioner_speech_mentions_omitted"] = int(
                parse["speech_mentions_omitted"]
            )
        description = {
            "schema_version": CAPTION_REVISION,
            "policy": DESCRIPTION_POLICY,
            "prompt": prompt,
            "model": response.get("model"),
            "raw_text": response.get("text"),
            "generation_max_new_tokens": CAPTION_MAX_NEW_TOKENS,
            "generation_attempts": generated["attempts"],
            "prior_contract_failures": generated["prior_contract_failures"],
            "parsed": parsed,
            "parse": parse,
            "validation": validation,
            "analysis_audio": analysis_audio,
            "processing_seconds": round(time.perf_counter() - started, 3),
        }
        unresolved_contract_reasons = sorted(
            set(validation.get("review_reasons") or [])
            & CAPTION_REGENERATION_REASONS
        )
        if unresolved_contract_reasons:
            with connection:
                connection.execute(
                    """UPDATE caption_v2_records SET status='pending',
                    description_json=?,error=?,lease_owner=NULL,
                    lease_expires_at=?,updated_at=? WHERE sequence=?""",
                    (
                        json.dumps(description, separators=(",", ":")),
                        "caption_contract_retry_exhausted:"
                        + ",".join(unresolved_contract_reasons),
                        time.time() + 15,
                        _now(),
                        row["sequence"],
                    ),
                )
            connection.close()
            return {
                "status": "retry",
                "job_id": row["job_id"],
                "contract_reasons": unresolved_contract_reasons,
            }
        job = {
            "separation_json": row["separation_json"],
            "tag_json": row["tag_json"],
            "asr_json": row["asr_json"],
            "description_json": json.dumps(description, separators=(",", ":")),
        }
        quality = quality_evaluation(job)
        metadata = json.loads(
            connection.execute(
                "SELECT record_json FROM records WHERE job_id=?", (row["job_id"],)
            ).fetchone()[0]
        )
        metadata["schema_version"] = 2
        metadata["quality"] = quality
        metadata["scene_description"] = description
        metadata["caption_revision"] = {
            "version": CAPTION_REVISION,
            "source_snapshot": row["source_s3_prefix"],
            "revised_at": _now(),
        }
        with connection:
            connection.execute(
                """UPDATE caption_v2_records SET status='complete',
                quality_bucket=?,description_json=?,metadata_json=?,error=NULL,
                lease_owner=NULL,lease_expires_at=NULL,updated_at=? WHERE sequence=?""",
                (
                    quality["bucket"],
                    json.dumps(description, separators=(",", ":")),
                    json.dumps(metadata, separators=(",", ":")),
                    _now(),
                    row["sequence"],
                ),
            )
        result = {"status": "complete", "job_id": row["job_id"]}
    except Exception as error:
        logger.exception(
            "Caption v%s failed for job %s", CAPTION_REVISION, row["job_id"]
        )
        with connection:
            connection.execute(
                """UPDATE caption_v2_records SET status='pending',error=?,
                lease_owner=NULL,lease_expires_at=NULL,updated_at=? WHERE sequence=?""",
                (f"{type(error).__name__}: {error}"[-4000:], _now(), row["sequence"]),
            )
        result = {"status": "error", "job_id": row["job_id"], "error": str(error)}
    connection.close()
    return result


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _publish_record(
    s3: Any,
    *,
    bucket: str,
    output_root: str,
    row: sqlite3.Row,
) -> dict[str, Any]:
    metadata = json.loads(row["metadata_json"])
    description = json.loads(row["description_json"])
    source_bucket, source_prefix = _split_s3_uri(row["source_s3_prefix"])
    destination = f"{output_root}/{row['quality_bucket']}/{row['record_id']}"
    source = f"{source_prefix}/{row['source_quality_bucket']}/{row['record_id']}"
    artifacts: dict[str, str] = {}
    for name in ("original.wav", "dialogue.wav", "background.wav"):
        key = f"{destination}/{name}"
        if not _s3_object_exists(s3, bucket, key):
            s3.copy_object(
                Bucket=bucket,
                CopySource={"Bucket": source_bucket, "Key": f"{source}/{name}"},
                Key=key,
            )
        artifacts[name] = f"s3://{bucket}/{key}"
    scene_bytes = _format_scene_description(description["parsed"]).encode()
    transcript_bytes = (
        str(metadata.get("dialogue_transcription", {}).get("transcript") or "").strip()
        + "\n"
    ).encode()
    text_payloads = {
        "scene_description.txt": scene_bytes,
        "dialogue_transcript.txt": transcript_bytes,
    }
    for name, value in text_payloads.items():
        key = f"{destination}/{name}"
        if not _s3_object_exists(s3, bucket, key):
            s3.put_object(Bucket=bucket, Key=key, Body=value)
        artifacts[name] = f"s3://{bucket}/{key}"
        metadata.setdefault("artifacts", {})[name] = {
            "sha256": _sha256_bytes(value),
            "bytes": len(value),
        }
    metadata.get("artifacts", {}).pop("metadata.json", None)
    metadata_bytes = (
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    ).encode()
    metadata["artifacts"]["metadata.json"] = {
        "sha256": _sha256_bytes(metadata_bytes),
        "bytes": len(metadata_bytes),
    }
    metadata_key = f"{destination}/metadata.json"
    if not _s3_object_exists(s3, bucket, metadata_key):
        s3.put_object(Bucket=bucket, Key=metadata_key, Body=metadata_bytes)
    artifacts["metadata.json"] = f"s3://{bucket}/{metadata_key}"
    metadata["s3_artifacts"] = artifacts
    return metadata


def _publication_contract_reasons(description_json: str | None) -> list[str]:
    try:
        description = json.loads(description_json or "")
    except (TypeError, json.JSONDecodeError):
        return ["scene_description_invalid_json"]
    if not isinstance(description, dict):
        return ["scene_description_invalid_json"]
    reasons: list[str] = []
    if int(description.get("schema_version") or 0) != CAPTION_REVISION:
        reasons.append("scene_description_wrong_schema_version")
    if description.get("policy") != DESCRIPTION_POLICY:
        reasons.append("scene_description_wrong_policy")
    parsed = description.get("parsed")
    if not isinstance(parsed, dict):
        return reasons + ["scene_description_missing_parsed_content"]
    reasons.extend(
        reason
        for reason in _description_evaluation(parsed)["review_reasons"]
        if reason in CAPTION_REGENERATION_REASONS
    )
    rendered = _format_scene_description(parsed)
    if not (
        rendered.startswith("DESCRIPTION:\n")
        and "\n\nTIMELINE:\n- 00:00-" in rendered
    ):
        reasons.append("scene_description_render_contract_invalid")
    return sorted(set(reasons))


def requeue_invalid_complete_once(
    workspace: Path, *, limit: int = 1000
) -> int:
    """Return unpublished rows that fail the active contract to GPU inference."""
    connection = _initialize(workspace)
    rows = connection.execute(
        """SELECT sequence,description_json FROM caption_v2_records
        WHERE status='complete' AND uploaded_at IS NULL
        ORDER BY source_sequence LIMIT ?""",
        (max(1, limit),),
    ).fetchall()
    invalid = [
        (int(row["sequence"]), _publication_contract_reasons(row["description_json"]))
        for row in rows
    ]
    invalid = [(sequence, reasons) for sequence, reasons in invalid if reasons]
    with connection:
        connection.executemany(
            """UPDATE caption_v2_records SET status='pending',error=?,
            lease_owner=NULL,lease_expires_at=?,updated_at=?
            WHERE sequence=? AND status='complete' AND uploaded_at IS NULL""",
            [
                (
                    "contract_revalidation:" + ",".join(reasons),
                    time.time() + 5,
                    _now(),
                    sequence,
                )
                for sequence, reasons in invalid
            ],
        )
    connection.close()
    return len(invalid)


def publish_once(
    workspace: Path,
    *,
    bucket: str,
    prefix: str,
    snapshot_size: int = 1000,
    upload_workers: int = 16,
) -> dict[str, Any] | None:
    if snapshot_size != 1000:
        raise ValueError("Caption snapshots must contain exactly 1,000 records")
    connection = _initialize(workspace)
    rows = connection.execute(
        """SELECT * FROM caption_v2_records
        WHERE status='complete' AND uploaded_at IS NULL
        ORDER BY source_sequence LIMIT ?""",
        (snapshot_size,),
    ).fetchall()
    if len(rows) < snapshot_size:
        connection.close()
        return None
    invalid = [
        (int(row["sequence"]), _publication_contract_reasons(row["description_json"]))
        for row in rows
    ]
    invalid = [(sequence, reasons) for sequence, reasons in invalid if reasons]
    if invalid:
        with connection:
            connection.executemany(
                """UPDATE caption_v2_records SET status='pending',error=?,
                lease_owner=NULL,lease_expires_at=NULL,updated_at=?
                WHERE sequence=? AND status='complete' AND uploaded_at IS NULL""",
                [
                    (
                        "prepublication_contract_failed:" + ",".join(reasons),
                        _now(),
                        sequence,
                    )
                    for sequence, reasons in invalid
                ],
            )
        connection.close()
        logger.warning(
            "Deferred %s caption records which failed the publication contract",
            len(invalid),
        )
        return {"records": 0, "deferred_records": len(invalid)}
    start = int(rows[0]["source_sequence"])
    end = int(rows[-1]["source_sequence"])
    snapshot_id = f"{SNAPSHOT_VERSION}-{start:08d}-{end:08d}"
    output_root = f"{prefix.strip('/')}/snapshots/{snapshot_id}"
    s3 = boto3.client(
        "s3",
        config=Config(max_pool_connections=max(20, upload_workers * 2)),
    )
    with ThreadPoolExecutor(max_workers=max(1, upload_workers)) as executor:
        records = list(
            executor.map(
                lambda row: _publish_record(
                    s3, bucket=bucket, output_root=output_root, row=row
                ),
                rows,
            )
        )
    quality_counts: dict[str, int] = {}
    for row, record in zip(rows, records, strict=True):
        quality = str(row["quality_bucket"])
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
        record["snapshot_membership"] = {
            "snapshot_id": snapshot_id,
            "record_sequence": int(row["source_sequence"]),
            "quality_bucket": quality,
        }
    manifest = {
        "schema_version": CAPTION_REVISION,
        "dataset": f"dialogue_background_voice_only_sam_{SNAPSHOT_VERSION}",
        "snapshot_id": snapshot_id,
        "snapshot_record_count": 1000,
        "quality_buckets": quality_counts,
        "training_default_filter": "quality.bucket == 'success'",
        "all_quality_buckets_included": True,
        "reference_generation": False,
        "records": records,
        "created_at": str(rows[-1]["created_at"]),
    }
    manifest["verification"] = _verify_snapshot_manifest(manifest)
    root = workspace / f"caption-{SNAPSHOT_VERSION}-snapshots" / snapshot_id
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    digest = sha256_file(manifest_path)
    published_at = _now()
    ready = {
        "schema_version": CAPTION_REVISION,
        "snapshot_id": snapshot_id,
        "record_count": 1000,
        "quality_buckets": quality_counts,
        "manifest_sha256": digest,
        "snapshot_uri": f"s3://{bucket}/{output_root}/",
        "immutable": True,
        "verification_status": "passed",
        "published_at": published_at,
    }
    ready_path = root / "READY.json"
    _atomic_json(ready_path, ready)
    s3.upload_file(str(manifest_path), bucket, f"{output_root}/manifest.json")
    s3.upload_file(str(ready_path), bucket, f"{output_root}/READY.json")
    with connection:
        connection.execute(
            f"""INSERT INTO {SNAPSHOT_TABLE}(end_sequence,snapshot_id,
            record_count,manifest_sha256,s3_prefix,published_at)
            VALUES(?,?,?,?,?,?)""",  # noqa: S608
            (end, snapshot_id, 1000, digest, ready["snapshot_uri"], published_at),
        )
        connection.executemany(
            """UPDATE caption_v2_records SET uploaded_at=?,snapshot_id=?
            WHERE sequence=?""",
            [(_now(), snapshot_id, row["sequence"]) for row in rows],
        )
        connection.executemany(
            """INSERT OR REPLACE INTO caption_snapshot_membership(
            snapshot_id,record_id,source_sequence,job_id,quality_bucket,
            metadata_json) VALUES(?,?,?,?,?,?)""",
            [
                (
                    snapshot_id,
                    row["record_id"],
                    row["source_sequence"],
                    row["job_id"],
                    row["quality_bucket"],
                    json.dumps(record, separators=(",", ":")),
                )
                for row, record in zip(rows, records, strict=True)
            ],
        )
    connection.close()
    return {"snapshot_id": snapshot_id, "records": 1000}


def status(workspace: Path) -> dict[str, Any]:
    connection = _initialize(workspace)
    result = {
        "records": dict(
            connection.execute(
                "SELECT status,COUNT(*) FROM caption_v2_records GROUP BY status"
            ).fetchall()
        ),
        "published": connection.execute(
            "SELECT COUNT(*) FROM caption_v2_records WHERE uploaded_at IS NOT NULL"
        ).fetchone()[0],
        "snapshots": connection.execute(  # noqa: S608
            f"SELECT COUNT(*) FROM {SNAPSHOT_TABLE}"
        ).fetchone()[0],
    }
    connection.close()
    return result


def recover_leases(workspace: Path) -> int:
    """Requeue work owned by the previous systemd control group."""
    connection = _initialize(workspace)
    with connection:
        recovered = connection.execute(
            """UPDATE caption_v2_records SET status='pending',lease_owner=NULL,
            lease_expires_at=NULL,updated_at=? WHERE status='running'""",
            (_now(),),
        ).rowcount
    connection.close()
    return recovered


def repair_snapshot_membership(workspace: Path) -> int:
    """Preserve immutable snapshot membership independently of active revision."""
    connection = _initialize(workspace)
    with connection:
        repaired = connection.execute(
            """INSERT OR IGNORE INTO caption_snapshot_membership(
            snapshot_id,record_id,source_sequence,job_id,quality_bucket,
            metadata_json)
            SELECT snapshot_id,record_id,source_sequence,job_id,quality_bucket,
            metadata_json FROM caption_v2_records
            WHERE uploaded_at IS NOT NULL AND snapshot_id IS NOT NULL"""
        ).rowcount
    connection.close()
    return repaired


def prepare_revision(workspace: Path) -> bool:
    """Atomically supersede an older immutable stream without mutating its S3 data."""
    connection = _initialize(workspace)
    row = connection.execute(
        "SELECT value FROM caption_revision_state WHERE key='active_revision'"
    ).fetchone()
    current = int(row[0]) if row else 2
    if current >= CAPTION_REVISION:
        connection.close()
        return False
    with connection:
        target = int(
            connection.execute(
                "SELECT COALESCE(MAX(source_sequence),0) FROM caption_v2_records"
            ).fetchone()[0]
        )
        connection.execute(
            """INSERT OR IGNORE INTO caption_snapshot_membership(
            snapshot_id,record_id,source_sequence,job_id,quality_bucket,
            metadata_json)
            SELECT snapshot_id,record_id,source_sequence,job_id,quality_bucket,
            metadata_json FROM caption_v2_records
            WHERE uploaded_at IS NOT NULL AND snapshot_id IS NOT NULL"""
        )
        connection.execute(
            """UPDATE caption_v2_records SET status=CASE
            WHEN json_extract(description_json,'$.raw_text') IS NOT NULL
            OR EXISTS(SELECT 1 FROM jobs j WHERE j.id=caption_v2_records.job_id
                AND json_extract(j.description_json,'$.raw_text') IS NOT NULL)
            THEN 'reparse' ELSE 'pending' END,attempts=0,
            lease_owner=NULL,lease_expires_at=NULL,
            description_json=CASE
            WHEN json_extract(description_json,'$.raw_text') IS NOT NULL
            THEN description_json ELSE NULL END,
            metadata_json=NULL,quality_bucket=NULL,error=NULL,uploaded_at=NULL,
            snapshot_id=NULL,updated_at=?""",
            (_now(),),
        )
        connection.execute(
            """INSERT INTO caption_revision_state(key,value,updated_at)
            VALUES('active_revision',?,?) ON CONFLICT(key) DO UPDATE SET
            value=excluded.value,updated_at=excluded.updated_at""",
            (str(CAPTION_REVISION), _now()),
        )
        connection.execute(
            """INSERT INTO caption_revision_state(key,value,updated_at)
            VALUES('target_source_sequence',?,?) ON CONFLICT(key) DO UPDATE SET
            value=excluded.value,updated_at=excluded.updated_at""",
            (str(target), _now()),
        )
    connection.close()
    return True


def run(
    workspace: Path,
    *,
    api_url: str,
    extra_api_url: str | None,
    bucket: str,
    prefix: str,
    workers: int,
    extra_workers: int,
    upload_workers: int,
) -> None:
    if prepare_revision(workspace):
        logger.info("Prepared caption revision %s", CAPTION_REVISION)
    recovered = recover_leases(workspace)
    if recovered:
        logger.info("Recovered %s caption leases", recovered)
    repaired = repair_snapshot_membership(workspace)
    if repaired:
        logger.info("Preserved %s caption snapshot memberships", repaired)

    def caption_worker(index: int, worker_api_url: str) -> None:
        while True:
            result = describe_once(
                workspace,
                api_url=worker_api_url,
                worker=f"caption-v{CAPTION_REVISION}-{index}",
            )
            if result is None:
                time.sleep(3)
            elif result["status"] == "error":
                time.sleep(10)

    def seed_controller() -> None:
        while True:
            seed_once(workspace)
            time.sleep(3)

    def snapshot_publisher() -> None:
        while True:
            deferred = requeue_invalid_complete_once(workspace)
            if deferred:
                logger.warning(
                    "Returned %s invalid complete captions to inference", deferred
                )
            publish_once(
                workspace,
                bucket=bucket,
                prefix=prefix,
                upload_workers=upload_workers,
            )
            time.sleep(15)

    threads = [
        threading.Thread(target=caption_worker, args=(index, api_url))
        for index in range(max(1, workers))
    ]
    if extra_api_url and extra_workers > 0:
        threads.extend(
            threading.Thread(
                target=caption_worker,
                args=(workers + index, extra_api_url),
            )
            for index in range(extra_workers)
        )
    threads.extend(
        (
            threading.Thread(target=seed_controller),
            threading.Thread(target=snapshot_publisher),
        )
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--api-url", default="http://127.0.0.1:8003")
    run_parser.add_argument("--extra-api-url")
    run_parser.add_argument("--bucket", required=True)
    run_parser.add_argument(
        "--prefix", default=f"dialogue-background-training-v{CAPTION_REVISION}"
    )
    run_parser.add_argument("--workers", type=int, default=4)
    run_parser.add_argument("--extra-workers", type=int, default=0)
    run_parser.add_argument("--upload-workers", type=int, default=16)
    commands.add_parser("seed")
    commands.add_parser("status")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.command == "run":
        run(
            args.workspace,
            api_url=args.api_url,
            extra_api_url=args.extra_api_url,
            bucket=args.bucket,
            prefix=args.prefix,
            workers=args.workers,
            extra_workers=max(0, args.extra_workers),
            upload_workers=args.upload_workers,
        )
    elif args.command == "seed":
        print(seed_once(args.workspace))
    else:
        print(json.dumps(status(args.workspace), indent=2))


if __name__ == "__main__":
    main()
