"""Local browser app for manually reviewing an acquired audio dataset."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

Decision = Literal["good", "perfect", "not_ok"]
TrainingQualityDecision = Literal["success", "failure"]
Reason = Literal[
    "lacking_voice",
    "lacking_music",
    "lacking_background_audio",
    "vocal_music",
    "speech_not_dialogue",
    "too_low_quality",
    "too_quiet",
    "distorted_or_clipped",
    "wrong_balance",
    "other",
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ReviewerIdentity(BaseModel):
    reviewer_id: str = Field(min_length=8, max_length=80, pattern=r"^[\w-]+$")
    reviewer_name: str = Field(min_length=1, max_length=80)

    @field_validator("reviewer_name")
    @classmethod
    def clean_reviewer_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Reviewer name cannot be empty")
        return value


class ClaimNextRequest(ReviewerIdentity):
    release_filename: str | None = None


class ReviewUpdate(ReviewerIdentity):
    decision: Decision
    reasons: list[Reason] = Field(default_factory=list)
    note: str = Field(default="", max_length=1000)


class TrainingQualityOverrideUpdate(BaseModel):
    """A manual annotation layered over an immutable training record."""

    decision: TrainingQualityDecision | None = None
    issues: dict[str, bool] = Field(default_factory=dict)

    @field_validator("issues")
    @classmethod
    def validate_issue_names(cls, value: dict[str, bool]) -> dict[str, bool]:
        cleaned: dict[str, bool] = {}
        for issue, occurs in value.items():
            issue = issue.strip()
            if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", issue):
                raise ValueError(f"Invalid quality issue name: {issue!r}")
            cleaned[issue] = occurs
        return cleaned


class ClaimConflict(RuntimeError):
    """Raised when another reviewer owns a live clip lease."""


class _AppendOnlyLineCounter:
    """Count lines without rereading a growing acquisition log on every poll."""

    def __init__(self, path: Path):
        self.path = path
        self.identity: tuple[int, int] | None = None
        self.offset = 0
        self.count = 0

    def refresh(self) -> int:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return 0
        identity = (stat.st_dev, stat.st_ino)
        if self.identity != identity or stat.st_size < self.offset:
            self.identity = identity
            self.offset = 0
            self.count = 0
        with self.path.open("rb") as source:
            source.seek(self.offset)
            while chunk := source.read(1024 * 1024):
                self.count += chunk.count(b"\n")
            self.offset = source.tell()
        return self.count


class _AppendOnlyJsonlSummary:
    """Maintain latest per-filename acceptance state for an appended JSONL file."""

    def __init__(self, path: Path):
        self.path = path
        self.identity: tuple[int, int] | None = None
        self.offset = 0
        self.records: dict[str, bool] = {}

    def refresh(self) -> tuple[int, set[str], set[str]]:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return 0, set(), set()
        identity = (stat.st_dev, stat.st_ino)
        if self.identity != identity or stat.st_size < self.offset:
            self.identity = identity
            self.offset = 0
            self.records = {}
        with self.path.open("rb") as source:
            source.seek(self.offset)
            while True:
                line_start = source.tell()
                line = source.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    source.seek(line_start)
                    break
                try:
                    item = json.loads(line)
                    filename = str(item["filename"])
                except (json.JSONDecodeError, KeyError, TypeError, UnicodeError):
                    continue
                self.records[filename] = bool(item.get("accepted"))
            self.offset = source.tell()
        accepted = {name for name, passed in self.records.items() if passed}
        return len(self.records), accepted, set(self.records)


class _ManifestSummary:
    """Cache the current-policy filenames from an atomically rewritten manifest."""

    def __init__(self, path: Path):
        self.path = path
        self.signature: tuple[int, int, int] | None = None
        self.target = 0
        self.filenames: set[str] = set()

    def refresh(self) -> tuple[int, int, set[str]]:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return 0, 0, set()
        signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if self.signature != signature:
            try:
                manifest = json.loads(self.path.read_text())
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                return self.target, len(self.filenames), set(self.filenames)
            self.target = int(manifest.get("target_records") or 0)
            self.filenames = {
                Path(str(record["local_path"])).name
                for record in manifest.get("records", [])
                if record.get("local_path")
            }
            self.signature = signature
        return self.target, len(self.filenames), set(self.filenames)


class PipelineProgressStore:
    """Read live acquisition/validation counters without touching worker state."""

    def __init__(
        self,
        batch_dirs: list[Path],
        *,
        final_dir: Path | None = None,
        target: int = 1000,
    ):
        self.batch_dirs = [path.resolve() for path in batch_dirs]
        self.final_dir = final_dir.resolve() if final_dir else None
        self.target = target
        self.lock = threading.Lock()
        self.attempt_trackers = {
            path: _AppendOnlyLineCounter(path / "attempts.jsonl")
            for path in self.batch_dirs
        }
        self.manifest_trackers = {
            path: _ManifestSummary(path / "manifest.json") for path in self.batch_dirs
        }
        self.m2d_trackers = {
            path: _AppendOnlyJsonlSummary(path / "m2d-validation.jsonl")
            for path in self.batch_dirs
        }
        self.asr_trackers = {
            path: _AppendOnlyJsonlSummary(path / "asr-validation.jsonl")
            for path in self.batch_dirs
        }

    def _final_counts(self) -> tuple[int, bool, str | None]:
        if not self.final_dir:
            return 0, False, None
        audit_path = self.final_dir / "audit.json"
        try:
            audit = json.loads(audit_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return 0, False, str(self.final_dir)
        count = int(audit.get("record_count") or 0)
        return count, bool(audit.get("all_requirements_pass")), str(self.final_dir)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            batches: list[dict[str, Any]] = []
            for index, path in enumerate(self.batch_dirs, start=1):
                raw_target, downloaded, current_filenames = self.manifest_trackers[
                    path
                ].refresh()
                attempts = self.attempt_trackers[path].refresh()
                _, m2d_accepted, m2d_filenames = self.m2d_trackers[path].refresh()
                m2d_scored = len(m2d_filenames & current_filenames)
                m2d_accepted &= current_filenames
                _, asr_accepted, asr_filenames = self.asr_trackers[path].refresh()
                asr_scored = len(asr_filenames & m2d_accepted)
                asr_accepted &= m2d_accepted
                combined = len(m2d_accepted & asr_accepted)
                exists = path.exists()
                markers = {
                    "acquisition": path / f".acquisition-complete-{raw_target}",
                    "m2d": path / f".m2d-complete-{raw_target}",
                    "speech": path / f".asr-complete-{raw_target}",
                }
                workers = {
                    worker: (
                        "complete"
                        if raw_target and marker.is_file()
                        else "running"
                        if raw_target
                        else "waiting"
                    )
                    for worker, marker in markers.items()
                }
                if not any((attempts, raw_target, downloaded, m2d_scored, asr_scored)):
                    status = "waiting"
                elif raw_target and downloaded < raw_target:
                    status = "downloading"
                elif m2d_scored < downloaded:
                    status = "m2d_scoring"
                elif asr_scored < len(m2d_accepted):
                    status = "speech_validation"
                else:
                    status = "validated"
                batches.append(
                    {
                        "batch": index,
                        "name": path.name,
                        "path": str(path),
                        "exists": exists,
                        "status": status,
                        "attempts": attempts,
                        "raw_target": raw_target,
                        "downloaded": downloaded,
                        "m2d_scored": m2d_scored,
                        "m2d_accepted": len(m2d_accepted),
                        "asr_scored": asr_scored,
                        "asr_accepted": len(asr_accepted),
                        "combined_eligible": combined,
                        "workers": workers,
                    }
                )
            final_count, final_verified, final_path = self._final_counts()
            totals = {
                key: sum(int(batch[key]) for batch in batches)
                for key in (
                    "attempts",
                    "downloaded",
                    "m2d_scored",
                    "m2d_accepted",
                    "asr_scored",
                    "asr_accepted",
                    "combined_eligible",
                )
            }
            active = next(
                (batch for batch in batches if batch["status"] != "waiting"),
                None,
            )
            non_waiting = [batch for batch in batches if batch["status"] != "waiting"]
            if non_waiting:
                active = non_waiting[-1]
            active_stages = (
                [
                    worker
                    for worker, status in active["workers"].items()
                    if status == "running"
                ]
                if active
                else []
            )
            stage = (
                "complete"
                if final_verified and final_count == self.target
                else (active["status"] if active else "waiting")
            )
            if stage == "complete":
                active_stages = []
            return {
                "updated_at": _now(),
                "target": self.target,
                "stage": stage,
                "active_stages": active_stages,
                "final": {
                    "materialized": final_count,
                    "verified": final_verified,
                    "path": final_path,
                },
                "totals": totals,
                "batches": batches,
                "notes": {
                    "combined_eligible": (
                        "M2D and speech-validation intersection before final "
                        "deduplication and the per-source cap"
                    ),
                    "selection": (
                        "Stereo, source-quality, cinematic metadata, M2D "
                        "voice+music+SFX, English speech, deduplication, and "
                        "a duration-scaled per-source diversity budget"
                    ),
                },
            }


class ContinuousProgressStore:
    """Expose the permanent SQLite-backed pipeline without batch semantics."""

    CACHE_FILENAME = "dashboard-progress-cache.json"

    def __init__(
        self,
        workspace: Path,
        *,
        snapshot_size: int = 2500,
        refresh_seconds: float = 10.0,
    ):
        self.workspace = workspace.resolve()
        self.snapshot_size = snapshot_size
        self.refresh_seconds = refresh_seconds
        self._cache_lock = threading.Lock()
        self._refreshing = False
        self._worker_detail_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._worker_detail_cache_seconds = 3.0
        self._cache_path = self.workspace / self.CACHE_FILENAME
        persisted = self._load_persisted_snapshot()
        self._cached_snapshot = persisted or self._read_snapshot()
        # A persisted snapshot makes startup instant. Refresh its live portions
        # on the first request while reusing the expensive historical strategy
        # aggregation from disk.
        self._expires_at = (
            time.monotonic()
            if persisted is not None
            else time.monotonic() + refresh_seconds
        )

    def _load_persisted_snapshot(self) -> dict[str, Any] | None:
        try:
            snapshot = json.loads(self._cache_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        return snapshot if isinstance(snapshot, dict) else None

    def _persist_snapshot(self, snapshot: dict[str, Any]) -> None:
        temporary = self._cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(snapshot, separators=(",", ":")))
        temporary.replace(self._cache_path)

    def _read_snapshot(self) -> dict[str, Any]:
        from .continuous_dataset import progress_snapshot

        cached_frontier = getattr(self, "_cached_snapshot", {}).get(
            "source_frontier", {}
        )
        cached_strategies = cached_frontier.get("discovery_strategies")
        if isinstance(cached_strategies, dict):
            return progress_snapshot(
                self.workspace,
                self.snapshot_size,
                cached_discovery_strategies=cached_strategies,
            )
        return progress_snapshot(self.workspace, self.snapshot_size)

    def _refresh_cache(self) -> None:
        try:
            snapshot = self._read_snapshot()
        except Exception:
            logger.exception("Background progress refresh failed")
        else:
            with self._cache_lock:
                self._cached_snapshot = snapshot
                self._expires_at = time.monotonic() + self.refresh_seconds
            try:
                self._persist_snapshot(snapshot)
            except OSError:
                logger.exception("Could not persist dashboard progress cache")
        finally:
            with self._cache_lock:
                self._refreshing = False

    def snapshot(self) -> dict[str, Any]:
        start_refresh = False
        with self._cache_lock:
            if time.monotonic() >= self._expires_at and not self._refreshing:
                self._refreshing = True
                start_refresh = True
            snapshot = copy.deepcopy(self._cached_snapshot)
        if start_refresh:
            threading.Thread(
                target=self._refresh_cache,
                name="progress-snapshot-refresh",
                daemon=True,
            ).start()
        return snapshot

    def download_worker_detail(self, worker: str) -> dict[str, Any] | None:
        now = time.monotonic()
        with self._cache_lock:
            cached = self._worker_detail_cache.get(worker)
            if (
                cached is not None
                and now - cached[0] < self._worker_detail_cache_seconds
            ):
                return copy.deepcopy(cached[1])
        from .source_frontier import download_worker_detail

        detail = download_worker_detail(self.workspace, worker)
        if detail is None:
            return None
        with self._cache_lock:
            self._worker_detail_cache[worker] = (now, detail)
        return copy.deepcopy(detail)


class TrainingSnapshotStore:
    """Serve immutable training snapshots without loading their large manifests."""

    AUDIO_FILES = {
        "original": "original.wav",
        "dialogue": "dialogue.wav",
        "background": "background.wav",
    }
    SNAPSHOT_PATTERN = re.compile(r"^v(?P<version>\d+)-(\d{8})-(\d{8})$")
    BALANCE_CACHE_VERSION = 2
    BALANCE_FAMILIES: dict[str, tuple[str, ...]] = {
        "Music and score": (
            "music",
            "musical",
            "instrument",
            "orchestra",
            "guitar",
            "piano",
            "synthesizer",
            "drum",
        ),
        "Transport and traffic": (
            "vehicle",
            "car",
            "truck",
            "bus",
            "train",
            "aircraft",
            "airplane",
            "helicopter",
            "motorcycle",
            "traffic",
            "engine",
            "boat",
        ),
        "Weather and nature": (
            "wind",
            "rain",
            "thunder",
            "storm",
            "fire",
            "forest",
            "nature",
            "weather",
            "eruption",
        ),
        "Water": (
            "water",
            "ocean",
            "sea",
            "river",
            "stream",
            "splash",
            "wave",
        ),
        "Animals": (
            "animal",
            "bird",
            "dog",
            "cat",
            "insect",
            "whale",
            "horse",
            "roar",
            "growl",
        ),
        "Footsteps and movement": (
            "footstep",
            "walk",
            "running",
            "movement",
            "rustle",
            "swish",
            "run",
            "shuffle",
            "clip-clop",
            "gallop",
        ),
        "Impacts and action": (
            "explosion",
            "gun",
            "weapon",
            "impact",
            "thud",
            "slam",
            "crash",
            "bang",
            "fight",
            "punch",
            "whoosh",
            "whip",
            "rumble",
            "boom",
            "tap",
            "fusillade",
            "crush",
        ),
        "Machinery and tools": (
            "machine",
            "mechanism",
            "gear",
            "tool",
            "drill",
            "saw",
            "industrial",
            "motor",
            "printer",
            "ratchet",
        ),
        "Household and objects": (
            "door",
            "glass",
            "dish",
            "cutlery",
            "furniture",
            "drawer",
            "kitchen",
            "clock",
            "tick",
            "camera",
        ),
        "Electronic and UI": (
            "alarm",
            "beep",
            "electronic",
            "ringtone",
            "telephone",
            "computer",
            "sonar",
            "chirp tone",
            "television",
        ),
        "Crowds and public spaces": (
            "crowd",
            "cheering",
            "applause",
            "chatter",
            "public space",
        ),
        "Indoor ambience": (
            "inside",
            "room",
            "hall",
            "indoor",
        ),
        "Outdoor ambience": (
            "outside",
            "outdoor",
            "street",
            "urban",
            "rural",
            "field recording",
        ),
        "Human non-speech": (
            "breathing",
            "cough",
            "laugh",
            "cry",
            "scream",
            "snore",
            "heartbeat",
            "snort",
            "sneeze",
            "snicker",
            "gasp",
            "wail",
            "moan",
            "beatbox",
            "hum",
        ),
    }

    def __init__(
        self,
        workspace: Path,
        *,
        summary_refresh_seconds: float = 15.0,
        overrides_path: Path | None = None,
    ):
        self.workspace = workspace.resolve()
        self.database_path = self.workspace / "training-dataset.sqlite3"
        if not self.database_path.is_file():
            raise FileNotFoundError(
                f"Training dataset database not found: {self.database_path}"
            )
        self._s3_client: Any | None = None
        self._s3_lock = threading.Lock()
        self.overrides_path = (
            overrides_path.resolve()
            if overrides_path
            else self.workspace / "training-review-overrides.sqlite3"
        )
        self._override_lock = threading.Lock()
        self._initialize_overrides()
        self._summary_refresh_seconds = summary_refresh_seconds
        self._summary_lock = threading.Lock()
        self._summary_refreshing = False
        self._cached_transformation_summary = self._read_transformation_summary()
        self._summary_expires_at = time.monotonic() + summary_refresh_seconds
        self._balance_refresh_seconds = 900.0
        self._balance_lock = threading.Lock()
        self._balance_refreshing = False
        self._balance_cache_path = self.workspace / "training-balance-cache-v2.json"
        self._cached_balance_summary = self._load_balance_cache()
        self._balance_expires_at = (
            time.monotonic() + 30.0 if self._cached_balance_summary else 0.0
        )

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.database_path}?mode=ro", uri=True, timeout=10
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _override_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.overrides_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize_overrides(self) -> None:
        """Create sidecar storage without modifying the training dataset DB."""
        self.overrides_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._override_connection()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS training_quality_overrides(
                snapshot_id TEXT NOT NULL,
                record_id TEXT NOT NULL,
                decision TEXT CHECK(decision IN ('success','failure')),
                issues_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL,
                PRIMARY KEY(snapshot_id,record_id)
                )"""
            )
            connection.commit()

    def quality_override(
        self, snapshot_id: str, record_id: str
    ) -> dict[str, Any] | None:
        with closing(self._override_connection()) as connection:
            row = connection.execute(
                """SELECT decision,issues_json,updated_at
                FROM training_quality_overrides
                WHERE snapshot_id=? AND record_id=?""",
                (snapshot_id, record_id),
            ).fetchone()
        if row is None:
            return None
        try:
            issues = json.loads(row["issues_json"])
        except (TypeError, json.JSONDecodeError):
            issues = {}
        return {
            "decision": row["decision"],
            "issues": {
                str(issue): occurs
                for issue, occurs in issues.items()
                if isinstance(occurs, bool)
            },
            "updated_at": row["updated_at"],
        }

    def set_quality_override(
        self,
        snapshot_id: str,
        record_id: str,
        update: TrainingQualityOverrideUpdate,
    ) -> dict[str, Any] | None:
        # Validate membership against the immutable source before writing only
        # to the sidecar annotation database.
        self.record(snapshot_id, record_id=record_id)
        if update.decision is None and not update.issues:
            self.clear_quality_override(snapshot_id, record_id)
            return None
        updated_at = _now()
        with self._override_lock, closing(self._override_connection()) as connection:
            connection.execute(
                """INSERT INTO training_quality_overrides(
                snapshot_id,record_id,decision,issues_json,updated_at
                ) VALUES(?,?,?,?,?)
                ON CONFLICT(snapshot_id,record_id) DO UPDATE SET
                decision=excluded.decision,
                issues_json=excluded.issues_json,
                updated_at=excluded.updated_at""",
                (
                    snapshot_id,
                    record_id,
                    update.decision,
                    json.dumps(update.issues, sort_keys=True, separators=(",", ":")),
                    updated_at,
                ),
            )
            connection.commit()
        return self.quality_override(snapshot_id, record_id)

    def clear_quality_override(self, snapshot_id: str, record_id: str) -> None:
        self.record(snapshot_id, record_id=record_id)
        with self._override_lock, closing(self._override_connection()) as connection:
            connection.execute(
                """DELETE FROM training_quality_overrides
                WHERE snapshot_id=? AND record_id=?""",
                (snapshot_id, record_id),
            )
            connection.commit()

    @staticmethod
    def _clip_measure(count: int) -> dict[str, int | float]:
        """Represent 30-second training clips as a count and equivalent hours."""
        return {"clips": count, "audio_hours": round(count / 120, 3)}

    def _read_transformation_summary(self) -> dict[str, Any]:
        with closing(self._connection()) as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            registered = 0
            stemmed = 0
            if "jobs" in tables:
                row = connection.execute(
                    """SELECT COUNT(*),COALESCE(SUM(
                    CASE WHEN separation_status='complete' THEN 1 ELSE 0 END
                    ),0) FROM jobs"""
                ).fetchone()
                registered, stemmed = int(row[0]), int(row[1])

            review_clips = 0
            source_buckets: dict[str, int] = {}
            if "records" in tables:
                review_clips = int(
                    connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
                )
                source_buckets = {
                    str(row[0]): int(row[1])
                    for row in connection.execute(
                        """SELECT quality_bucket,COUNT(*) FROM records
                        GROUP BY quality_bucket"""
                    )
                }

            revision = 1
            target_sequence: int | None = None
            if "caption_revision_state" in tables:
                revision_state = {
                    str(row[0]): str(row[1])
                    for row in connection.execute(
                        """SELECT key,value FROM caption_revision_state
                        WHERE key IN ('active_revision','target_source_sequence')"""
                    )
                }
                revision = int(revision_state.get("active_revision", revision))
                if revision_state.get("target_source_sequence"):
                    target_sequence = int(revision_state["target_source_sequence"])

            transformed = review_clips
            published = sum(source_buckets.values())
            latest_buckets = source_buckets
            states: dict[str, int] = {"complete": transformed}
            if "caption_v2_records" in tables:
                states = {
                    str(row[0]): int(row[1])
                    for row in connection.execute(
                        "SELECT status,COUNT(*) FROM caption_v2_records GROUP BY status"
                    )
                }
                transformed = states.get("complete", 0)
                published = int(
                    connection.execute(
                        """SELECT COUNT(*) FROM caption_v2_records
                        WHERE uploaded_at IS NOT NULL"""
                    ).fetchone()[0]
                )
                latest_buckets = {
                    str(row[0]): int(row[1])
                    for row in connection.execute(
                        """SELECT quality_bucket,COUNT(*) FROM caption_v2_records
                        WHERE status='complete' AND quality_bucket IS NOT NULL
                        GROUP BY quality_bucket"""
                    )
                }
                # The corrective caption stream is intentionally bounded at the
                # revision's source cursor. Records created after that boundary
                # are already generated with the active schema by the live
                # producer, so include them in the newest-version dashboard
                # totals without appending them to this finite migration.
                record_columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(records)")
                }
                job_columns = {
                    str(row[1]) for row in connection.execute("PRAGMA table_info(jobs)")
                }
                if (
                    target_sequence is not None
                    and {"sequence", "job_id", "quality_bucket"} <= record_columns
                    and {"id", "description_json"} <= job_columns
                ):
                    published_expression = (
                        "COALESCE(SUM(CASE WHEN r.uploaded_at IS NOT NULL "
                        "THEN 1 ELSE 0 END),0)"
                        if "uploaded_at" in record_columns
                        else "COUNT(*)"
                    )
                    post_target_rows = connection.execute(
                        f"""SELECT r.quality_bucket,COUNT(*),
                            {published_expression} FROM records r
                            JOIN jobs j ON j.id=r.job_id
                            WHERE r.sequence>?
                            AND json_extract(j.description_json,'$.schema_version')=?
                            AND json_extract(j.description_json,'$.policy')=?
                            GROUP BY r.quality_bucket""",  # noqa: S608
                        (
                            target_sequence,
                            revision,
                            f"af_next_description_timeline_v{revision}",
                        ),
                    ).fetchall()
                    post_target_buckets = {
                        str(row[0]): int(row[1]) for row in post_target_rows
                    }
                    current_live_records = sum(post_target_buckets.values())
                    published_live_records = sum(
                        int(row[2]) for row in post_target_rows
                    )
                    transformed += current_live_records
                    published += published_live_records
                    states["complete"] = (
                        states.get("complete", 0) + current_live_records
                    )
                    for bucket, count in post_target_buckets.items():
                        latest_buckets[bucket] = latest_buckets.get(bucket, 0) + count

        expected_buckets = ("success", "review", "failure")
        return {
            "revision": revision,
            "registered": self._clip_measure(registered),
            "stemmed": self._clip_measure(stemmed),
            "review_clips": self._clip_measure(review_clips),
            "latest_transformed": self._clip_measure(transformed),
            "latest_published": self._clip_measure(published),
            "remaining": self._clip_measure(max(0, review_clips - transformed)),
            "states": states,
            "buckets": {
                bucket: self._clip_measure(latest_buckets.get(bucket, 0))
                for bucket in expected_buckets
            },
            "measured_at": _now(),
        }

    def _refresh_transformation_summary(self) -> None:
        try:
            summary = self._read_transformation_summary()
        except Exception:
            logger.exception("Background training transformation refresh failed")
        else:
            with self._summary_lock:
                self._cached_transformation_summary = summary
                self._summary_expires_at = (
                    time.monotonic() + self._summary_refresh_seconds
                )
        finally:
            with self._summary_lock:
                self._summary_refreshing = False

    def transformation_summary(self, *, source_clips: int = 0) -> dict[str, Any]:
        """Return a cheap cached view of acquisition-to-training conversion."""
        start_refresh = False
        with self._summary_lock:
            if (
                time.monotonic() >= self._summary_expires_at
                and not self._summary_refreshing
            ):
                self._summary_refreshing = True
                start_refresh = True
            summary = copy.deepcopy(self._cached_transformation_summary)
        if start_refresh:
            threading.Thread(
                target=self._refresh_transformation_summary,
                name="training-transformation-refresh",
                daemon=True,
            ).start()
        summary["source"] = self._clip_measure(source_clips)
        return summary

    def _load_balance_cache(self) -> dict[str, Any] | None:
        try:
            payload = json.loads(self._balance_cache_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if payload.get("cache_version") != self.BALANCE_CACHE_VERSION:
            return None
        return payload

    def _save_balance_cache(self, payload: dict[str, Any]) -> None:
        temporary = self._balance_cache_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")))
        os.replace(temporary, self._balance_cache_path)

    @staticmethod
    def _new_balance_scope() -> dict[str, Any]:
        return {
            "records": 0,
            "platforms": Counter(),
            "background_buckets": Counter(),
            "semantic_tags": Counter(),
            "global_tags": Counter(),
            "sfx_tags": Counter(),
            "tag_labels": {},
            "families": Counter(),
            "quality_reasons": Counter(),
            "dialogue_words": Counter(),
            "dialogue_coverage": Counter(),
            "music_coverage": Counter(),
            "sfx_coverage": Counter(),
            "overlap_coverage": Counter(),
            "sources": Counter(),
            "source_labels": {},
        }

    @staticmethod
    def _coverage_band(value: Any) -> str:
        try:
            coverage = float(value)
        except (TypeError, ValueError):
            return "Unknown"
        if coverage < 0.05:
            return "None (<5%)"
        if coverage < 0.25:
            return "Low (5–25%)"
        if coverage < 0.60:
            return "Medium (25–60%)"
        return "High (60–100%)"

    @staticmethod
    def _dialogue_word_band(value: Any) -> str:
        try:
            words = int(value)
        except (TypeError, ValueError):
            return "Unknown"
        if words < 20:
            return "Sparse (0–19 words)"
        if words < 40:
            return "Light (20–39 words)"
        if words < 60:
            return "Medium (40–59 words)"
        return "Dense (60+ words)"

    @classmethod
    def _family_for_tag(cls, tag: str) -> str | None:
        normalized = tag.casefold()
        for family, keywords in cls.BALANCE_FAMILIES.items():
            if any(keyword in normalized for keyword in keywords):
                return family
        return None

    @staticmethod
    def _counter_series(
        counter: Counter[str],
        total: int,
        *,
        order: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        labels = list(order) if order else [key for key, _ in counter.most_common()]
        return [
            {
                "label": label,
                "records": int(counter.get(label, 0)),
                "audio_hours": round(counter.get(label, 0) / 120, 3),
                "percent": round(counter.get(label, 0) / total * 100, 2)
                if total
                else 0.0,
            }
            for label in labels
            if order or counter.get(label, 0)
        ]

    def _serialize_balance_scope(self, scope: dict[str, Any]) -> dict[str, Any]:
        total = int(scope["records"])
        family_counts = [
            int(scope["families"].get(family, 0)) for family in self.BALANCE_FAMILIES
        ]
        family_median = float(median(family_counts)) if family_counts else 0.0
        families: list[dict[str, Any]] = []
        for family in self.BALANCE_FAMILIES:
            count = int(scope["families"].get(family, 0))
            ratio = count / family_median if family_median else 0.0
            if not count or ratio < 0.5:
                balance = "underrepresented"
            elif ratio >= 1.75:
                balance = "overrepresented"
            else:
                balance = "middle"
            families.append(
                {
                    "label": family,
                    "records": count,
                    "audio_hours": round(count / 120, 3),
                    "percent": round(count / total * 100, 2) if total else 0.0,
                    "relative_to_family_median": round(ratio, 2),
                    "balance": balance,
                }
            )
        families.sort(key=lambda item: (-item["records"], item["label"]))

        tags: list[dict[str, Any]] = []
        for normalized, count in scope["semantic_tags"].most_common():
            variants: Counter[str] = scope["tag_labels"].get(normalized, Counter())
            label = variants.most_common(1)[0][0] if variants else normalized
            percent = count / total * 100 if total else 0.0
            if percent >= 25:
                representation = "dominant"
            elif percent < 2:
                representation = "sparse"
            else:
                representation = "covered"
            in_global = int(scope["global_tags"].get(normalized, 0))
            in_sfx = int(scope["sfx_tags"].get(normalized, 0))
            role = "both" if in_global and in_sfx else "SFX" if in_sfx else "global"
            tags.append(
                {
                    "label": label,
                    "family": self._family_for_tag(label) or "Other / uncategorized",
                    "records": int(count),
                    "audio_hours": round(count / 120, 3),
                    "percent": round(percent, 2),
                    "representation": representation,
                    "role": role,
                    "global_records": in_global,
                    "sfx_records": in_sfx,
                }
            )

        source_items = scope["sources"].most_common(20)
        top_source_records = sum(count for _, count in scope["sources"].most_common(10))
        top_family = families[0] if families else None
        sparse_families = [
            item["label"] for item in families if item["balance"] == "underrepresented"
        ]
        platform_top = scope["platforms"].most_common(1)
        insights: list[dict[str, str]] = []
        if top_family:
            insights.append(
                {
                    "level": "warning"
                    if top_family["balance"] == "overrepresented"
                    else "info",
                    "title": f"Largest family: {top_family['label']}",
                    "detail": (
                        f"Appears in {top_family['percent']:.1f}% of records. "
                        "Multi-label family percentages can overlap."
                    ),
                }
            )
        if sparse_families:
            insights.append(
                {
                    "level": "gap",
                    "title": f"{len(sparse_families)} sparse acoustic families",
                    "detail": ", ".join(sparse_families[:6]),
                }
            )
        if platform_top and total:
            platform, count = platform_top[0]
            share = count / total * 100
            insights.append(
                {
                    "level": "warning" if share >= 70 else "info",
                    "title": f"Largest source platform: {platform}",
                    "detail": f"{share:.1f}% of records come from this platform.",
                }
            )
        if total:
            insights.append(
                {
                    "level": "warning" if top_source_records / total >= 0.1 else "info",
                    "title": "Top 10 source videos",
                    "detail": (
                        f"Supply {top_source_records / total * 100:.1f}% of records; "
                        "lower concentration improves scene diversity."
                    ),
                }
            )
        music_led = int(scope["background_buckets"].get("music_led", 0))
        if total and music_led:
            music_led_share = music_led / total * 100
            insights.append(
                {
                    "level": "warning" if music_led_share >= 60 else "info",
                    "title": "Music-led background share",
                    "detail": (
                        f"{music_led_share:.1f}% of records are music-led; "
                        "effects/ambience-led acquisition would reduce this skew."
                    ),
                }
            )

        return {
            "records": total,
            "audio_hours": round(total / 120, 3),
            "unique_source_videos": len(scope["sources"]),
            "unique_tags": len(tags),
            "family_median_records": round(family_median, 1),
            "insights": insights,
            "families": families,
            "tags": tags,
            "platforms": self._counter_series(scope["platforms"], total),
            "background_buckets": self._counter_series(
                scope["background_buckets"], total
            ),
            "quality_reasons": self._counter_series(scope["quality_reasons"], total),
            "dialogue_words": self._counter_series(
                scope["dialogue_words"],
                total,
                order=(
                    "Sparse (0–19 words)",
                    "Light (20–39 words)",
                    "Medium (40–59 words)",
                    "Dense (60+ words)",
                    "Unknown",
                ),
            ),
            "dialogue_coverage": self._counter_series(
                scope["dialogue_coverage"],
                total,
                order=(
                    "None (<5%)",
                    "Low (5–25%)",
                    "Medium (25–60%)",
                    "High (60–100%)",
                    "Unknown",
                ),
            ),
            "music_coverage": self._counter_series(
                scope["music_coverage"],
                total,
                order=(
                    "None (<5%)",
                    "Low (5–25%)",
                    "Medium (25–60%)",
                    "High (60–100%)",
                    "Unknown",
                ),
            ),
            "sfx_coverage": self._counter_series(
                scope["sfx_coverage"],
                total,
                order=(
                    "None (<5%)",
                    "Low (5–25%)",
                    "Medium (25–60%)",
                    "High (60–100%)",
                    "Unknown",
                ),
            ),
            "overlap_coverage": self._counter_series(
                scope["overlap_coverage"],
                total,
                order=(
                    "None (<5%)",
                    "Low (5–25%)",
                    "Medium (25–60%)",
                    "High (60–100%)",
                    "Unknown",
                ),
            ),
            "top_sources": [
                {
                    "source_id": source_id,
                    "label": scope["source_labels"].get(source_id, source_id),
                    "records": int(count),
                    "audio_hours": round(count / 120, 3),
                    "percent": round(count / total * 100, 2) if total else 0.0,
                }
                for source_id, count in source_items
            ],
        }

    def _read_balance_summary(self) -> dict[str, Any]:
        with closing(self._override_connection()) as override_connection:
            manual_decisions = {
                str(row[0]): str(row[1])
                for row in override_connection.execute(
                    """SELECT record_id,decision FROM training_quality_overrides
                    WHERE decision IS NOT NULL ORDER BY updated_at"""
                )
            }

        scopes = {
            name: self._new_balance_scope()
            for name in ("all", "success", "review", "failure")
        }
        automated_buckets: Counter[str] = Counter()
        effective_buckets: Counter[str] = Counter()
        applied_manual_decisions: set[str] = set()
        with closing(self._connection()) as connection:
            revision_state = {
                str(row[0]): str(row[1])
                for row in connection.execute(
                    """SELECT key,value FROM caption_revision_state
                    WHERE key IN ('active_revision','target_source_sequence')"""
                )
            }
            revision = int(revision_state.get("active_revision", 1))
            target_sequence = int(revision_state.get("target_source_sequence", 0))
            rows = connection.execute(
                """SELECT record_id,quality_bucket,metadata_json
                FROM caption_v2_records
                WHERE status='complete' AND metadata_json IS NOT NULL
                UNION ALL
                SELECT r.record_id,r.quality_bucket,r.record_json
                FROM records r JOIN jobs j ON j.id=r.job_id
                WHERE r.sequence>?
                AND json_extract(j.description_json,'$.schema_version')=?
                AND json_extract(j.description_json,'$.policy')=?""",
                (
                    target_sequence,
                    revision,
                    f"af_next_description_timeline_v{revision}",
                ),
            )
            for row in rows:
                try:
                    metadata = json.loads(row["metadata_json"])
                except (TypeError, json.JSONDecodeError):
                    continue
                record_id = str(row["record_id"])
                automated_bucket = str(row["quality_bucket"])
                effective_bucket = manual_decisions.get(record_id, automated_bucket)
                if record_id in manual_decisions:
                    applied_manual_decisions.add(record_id)
                automated_buckets[automated_bucket] += 1
                effective_buckets[effective_bucket] += 1
                target_scopes = [scopes["all"]]
                if effective_bucket in scopes:
                    target_scopes.append(scopes[effective_bucket])

                source = metadata.get("source") or {}
                tagger = metadata.get("background_tagger") or {}
                transcription = metadata.get("dialogue_transcription") or {}
                quality = metadata.get("quality") or {}
                parsed = (metadata.get("scene_description") or {}).get("parsed") or {}
                platform = str(
                    source.get("source_platform") or source.get("platform") or "unknown"
                )
                background_bucket = str(tagger.get("background_bucket") or "unknown")
                video_id = str(
                    source.get("video_id")
                    or source.get("source_sha256")
                    or metadata.get("source_sha256")
                    or record_id
                )
                source_id = f"{platform}:{video_id}"
                source_label = str(source.get("title") or source_id)
                word_band = self._dialogue_word_band(
                    transcription.get("word_count")
                    if transcription.get("word_count") is not None
                    else (quality.get("signals") or {}).get("dialogue_word_count")
                )
                duration_after_vad = transcription.get("duration_after_vad_seconds")
                dialogue_coverage = self._coverage_band(
                    float(duration_after_vad) / 30
                    if duration_after_vad is not None
                    else None
                )
                music_coverage = self._coverage_band(
                    tagger.get("cinematic_music_coverage")
                )
                sfx_coverage = self._coverage_band(tagger.get("cinematic_sfx_coverage"))
                overlap_coverage = self._coverage_band(tagger.get("overlap_coverage"))
                global_tags = {
                    " ".join(str(tag).split())
                    for tag in parsed.get("global_tags") or []
                    if str(tag).strip()
                }
                sfx_tags = {
                    " ".join(str(tag).split())
                    for tag in parsed.get("sound_effects") or []
                    if str(tag).strip()
                }
                semantic_tags = global_tags | sfx_tags
                reasons = {
                    str(reason)
                    for reason in (
                        list(quality.get("failure_reasons") or [])
                        + list(quality.get("review_reasons") or [])
                    )
                }

                for scope in target_scopes:
                    scope["records"] += 1
                    scope["platforms"][platform] += 1
                    scope["background_buckets"][background_bucket] += 1
                    scope["dialogue_words"][word_band] += 1
                    scope["dialogue_coverage"][dialogue_coverage] += 1
                    scope["music_coverage"][music_coverage] += 1
                    scope["sfx_coverage"][sfx_coverage] += 1
                    scope["overlap_coverage"][overlap_coverage] += 1
                    scope["sources"][source_id] += 1
                    scope["source_labels"].setdefault(source_id, source_label)
                    scope["quality_reasons"].update(reasons)
                    record_families: set[str] = set()
                    for tag in semantic_tags:
                        normalized = tag.casefold()
                        scope["semantic_tags"][normalized] += 1
                        variants = scope["tag_labels"].setdefault(normalized, Counter())
                        variants[tag] += 1
                        family = self._family_for_tag(tag)
                        if family:
                            record_families.add(family)
                    for tag in global_tags:
                        scope["global_tags"][tag.casefold()] += 1
                    for tag in sfx_tags:
                        scope["sfx_tags"][tag.casefold()] += 1
                    scope["families"].update(record_families)

        total = int(scopes["all"]["records"])
        return {
            "status": "ready",
            "cache_version": self.BALANCE_CACHE_VERSION,
            "revision": revision,
            "scope_note": (
                "Newest caption revision only: the fixed migration cohort plus "
                "new live records already generated with that revision. Each "
                "record is counted once; historical snapshot versions are excluded."
            ),
            "tag_note": (
                "Tags are multi-label outputs from each record's final scene "
                "description. Percentages can sum above 100%. Family balance is "
                "relative to the median family coverage, not a fixed product target."
            ),
            "records": total,
            "audio_hours": round(total / 120, 3),
            "manual_decisions": len(applied_manual_decisions),
            "automated_buckets": self._counter_series(automated_buckets, total),
            "effective_buckets": self._counter_series(effective_buckets, total),
            "scopes": {
                name: self._serialize_balance_scope(scope)
                for name, scope in scopes.items()
            },
            "measured_at": _now(),
        }

    def _refresh_balance_summary(self) -> None:
        try:
            summary = self._read_balance_summary()
            self._save_balance_cache(summary)
        except Exception:
            logger.exception("Background training balance refresh failed")
        else:
            with self._balance_lock:
                self._cached_balance_summary = summary
                self._balance_expires_at = (
                    time.monotonic() + self._balance_refresh_seconds
                )
        finally:
            with self._balance_lock:
                self._balance_refreshing = False

    def balance_summary(self) -> dict[str, Any]:
        """Return cached distribution data and refresh it off the request path."""
        start_refresh = False
        with self._balance_lock:
            if (
                time.monotonic() >= self._balance_expires_at
                and not self._balance_refreshing
            ):
                self._balance_refreshing = True
                start_refresh = True
            summary = self._cached_balance_summary
        if start_refresh:
            threading.Thread(
                target=self._refresh_balance_summary,
                name="training-balance-refresh",
                daemon=True,
            ).start()
        if summary is None:
            return {
                "status": "building",
                "message": "Building the first whole-dataset balance summary…",
            }
        public = copy.deepcopy(
            {key: value for key, value in summary.items() if key != "scopes"}
        )
        public["scopes"] = {}
        for name, scope in summary["scopes"].items():
            public_scope = copy.deepcopy(
                {key: value for key, value in scope.items() if key != "tags"}
            )
            public_scope["top_tags"] = copy.deepcopy(scope["tags"][:25])
            public["scopes"][name] = public_scope
        public["refreshing"] = start_refresh
        return public

    def balance_tags(
        self,
        *,
        scope: str = "all",
        query: str = "",
        role: str = "all",
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        if scope not in {"all", "success", "review", "failure"}:
            raise KeyError(scope)
        if role not in {"all", "SFX", "global", "both"}:
            raise KeyError(role)
        with self._balance_lock:
            summary = self._cached_balance_summary
        if summary is None:
            self.balance_summary()
            return {"status": "building", "items": [], "total": 0}
        normalized_query = query.strip().casefold()[:120]
        tags = summary["scopes"][scope]["tags"]
        filtered = [
            tag
            for tag in tags
            if (role == "all" or tag["role"] == role)
            and (
                not normalized_query
                or normalized_query in tag["label"].casefold()
                or normalized_query in tag["family"].casefold()
            )
        ]
        offset = max(0, offset)
        limit = min(250, max(1, limit))
        return {
            "status": "ready",
            "scope": scope,
            "query": query.strip()[:120],
            "role": role,
            "offset": offset,
            "limit": limit,
            "total": len(filtered),
            "items": copy.deepcopy(filtered[offset : offset + limit]),
        }

    @classmethod
    def _sequence_bounds(cls, snapshot_id: str) -> tuple[int, int]:
        match = cls.SNAPSHOT_PATTERN.fullmatch(snapshot_id)
        if not match:
            raise KeyError(snapshot_id)
        return int(match.group(2)), int(match.group(3))

    def _snapshot_row(self, snapshot_id: str) -> sqlite3.Row:
        match = self.SNAPSHOT_PATTERN.fullmatch(snapshot_id)
        if not match:
            raise KeyError(snapshot_id)
        version = int(match.group("version"))
        table = "snapshots" if version == 1 else f"caption_v{version}_snapshots"
        with closing(self._connection()) as connection:
            row = connection.execute(
                f"SELECT * FROM {table} WHERE snapshot_id=?",  # noqa: S608
                (snapshot_id,),
            ).fetchone()
        if row is None:
            raise KeyError(snapshot_id)
        return row

    def snapshots(self) -> dict[str, Any]:
        with closing(self._connection()) as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            rows: list[sqlite3.Row] = list(
                connection.execute(
                    """SELECT snapshot_id,record_count,manifest_sha256,s3_prefix,
                    published_at,1 AS dataset_version,end_sequence FROM snapshots"""
                )
            )
            for version in range(2, 10):
                table = f"caption_v{version}_snapshots"
                if table not in tables:
                    continue
                rows.extend(
                    connection.execute(
                        f"""SELECT snapshot_id,record_count,manifest_sha256,
                        s3_prefix,published_at,{version} AS dataset_version,
                        end_sequence FROM {table}"""  # noqa: S608
                    )
                )
            rows.sort(
                key=lambda row: (row["dataset_version"], row["end_sequence"]),
                reverse=True,
            )
        snapshots: list[dict[str, Any]] = []
        for row in rows:
            version = int(row["dataset_version"])
            directory = "snapshots" if version == 1 else f"caption-v{version}-snapshots"
            ready_path = self.workspace / directory / row["snapshot_id"] / "READY.json"
            try:
                ready = json.loads(ready_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                ready = {}
            snapshots.append(
                {
                    "snapshot_id": row["snapshot_id"],
                    "dataset_version": int(row["dataset_version"]),
                    "record_count": int(row["record_count"]),
                    "quality_buckets": ready.get("quality_buckets", {}),
                    "manifest_sha256": row["manifest_sha256"],
                    "s3_prefix": row["s3_prefix"],
                    "published_at": row["published_at"],
                    "verification_status": ready.get("verification_status"),
                    "immutable": bool(ready.get("immutable", True)),
                }
            )
        return {
            "dataset": "dialogue_background_voice_only_sam",
            "snapshots": snapshots,
            "total_records": sum(item["record_count"] for item in snapshots),
        }

    def record(
        self,
        snapshot_id: str,
        *,
        position: int = 0,
        record_id: str | None = None,
    ) -> dict[str, Any]:
        snapshot = self._snapshot_row(snapshot_id)
        start, end = self._sequence_bounds(snapshot_id)
        is_revision = not snapshot_id.startswith("v1-")
        with closing(self._connection()) as connection:
            if is_revision and record_id:
                row = connection.execute(
                    """SELECT source_sequence AS sequence,job_id,record_id,
                    quality_bucket,metadata_json AS record_json
                    FROM caption_snapshot_membership
                    WHERE snapshot_id=? AND record_id=?""",
                    (snapshot_id, record_id),
                ).fetchone()
            elif is_revision:
                if position < 0 or position >= int(snapshot["record_count"]):
                    raise KeyError(position)
                row = connection.execute(
                    """SELECT source_sequence AS sequence,job_id,record_id,
                    quality_bucket,metadata_json AS record_json
                    FROM caption_snapshot_membership WHERE snapshot_id=?
                    ORDER BY source_sequence LIMIT 1 OFFSET ?""",
                    (snapshot_id, position),
                ).fetchone()
            elif record_id:
                row = connection.execute(
                    """SELECT sequence,job_id,record_id,quality_bucket,record_json
                    FROM records WHERE sequence BETWEEN ? AND ? AND record_id=?""",
                    (start, end, record_id),
                ).fetchone()
            else:
                if position < 0 or position >= int(snapshot["record_count"]):
                    raise KeyError(position)
                row = connection.execute(
                    """SELECT sequence,job_id,record_id,quality_bucket,record_json
                    FROM records WHERE sequence BETWEEN ? AND ?
                    ORDER BY sequence LIMIT 1 OFFSET ?""",
                    (start, end, position),
                ).fetchone()
            if row is None:
                raise KeyError(record_id if record_id else position)
            if is_revision:
                actual_position = int(
                    connection.execute(
                        """SELECT COUNT(*) FROM caption_snapshot_membership
                        WHERE snapshot_id=? AND source_sequence<?""",
                        (snapshot_id, row["sequence"]),
                    ).fetchone()[0]
                )
            else:
                actual_position = int(
                    connection.execute(
                        """SELECT COUNT(*) FROM records
                    WHERE sequence BETWEEN ? AND ? AND sequence<?""",
                        (start, end, row["sequence"]),
                    ).fetchone()[0]
                )
        metadata = json.loads(row["record_json"])
        manual_override = self.quality_override(snapshot_id, row["record_id"])
        metadata.update(
            {
                "snapshot_id": snapshot_id,
                "snapshot_position": actual_position,
                "snapshot_record_count": int(snapshot["record_count"]),
                "record_id": row["record_id"],
                "quality_bucket": row["quality_bucket"],
                "automated_quality_bucket": row["quality_bucket"],
                "manual_quality_override": manual_override,
                "effective_quality_bucket": (
                    manual_override["decision"]
                    if manual_override and manual_override.get("decision")
                    else row["quality_bucket"]
                ),
                "audio": {
                    stem: (
                        f"/api/training/audio/{snapshot_id}/{row['record_id']}/{stem}"
                    )
                    for stem in self.AUDIO_FILES
                },
            }
        )
        return metadata

    def audio_location(
        self, snapshot_id: str, record_id: str, stem: str
    ) -> tuple[Path | None, str, str]:
        filename = self.AUDIO_FILES.get(stem)
        if filename is None:
            raise KeyError(stem)
        snapshot = self._snapshot_row(snapshot_id)
        start, end = self._sequence_bounds(snapshot_id)
        is_revision = not snapshot_id.startswith("v1-")
        with closing(self._connection()) as connection:
            if is_revision:
                row = connection.execute(
                    """SELECT job_id,quality_bucket
                    FROM caption_snapshot_membership
                    WHERE snapshot_id=? AND record_id=?""",
                    (snapshot_id, record_id),
                ).fetchone()
            else:
                row = connection.execute(
                    """SELECT job_id,quality_bucket FROM records
                    WHERE sequence BETWEEN ? AND ? AND record_id=?""",
                    (start, end, record_id),
                ).fetchone()
        if row is None:
            raise KeyError(record_id)
        local = self.workspace / "work" / f"{int(row['job_id']):012d}" / filename
        prefix = str(snapshot["s3_prefix"])
        match = re.fullmatch(r"s3://([^/]+)/(.+?)/?", prefix)
        if match is None:
            raise ValueError(f"Invalid snapshot S3 prefix: {prefix}")
        key = (
            f"{match.group(2).rstrip('/')}/{row['quality_bucket']}/"
            f"{record_id}/{filename}"
        )
        return (local if local.is_file() else None), match.group(1), key

    def s3_client(self) -> Any:
        with self._s3_lock:
            if self._s3_client is None:
                import boto3

                self._s3_client = boto3.client("s3")
            return self._s3_client


class ReviewStore:
    def __init__(
        self,
        dataset_dir: Path,
        *,
        audio_directory: str,
        annotations_path: Path | None = None,
        claim_seconds: int = 600,
    ):
        self.dataset_dir = dataset_dir.resolve()
        self.audio_directory = audio_directory
        self.audio_dir = (self.dataset_dir / audio_directory).resolve()
        self.manifest_path = self.dataset_dir / "manifest.json"
        self.catalog_path = self.dataset_dir.parent / "catalog.sqlite3"
        self.catalog_record_count: int | None = None
        self.catalog_sequence_bounds: tuple[int, int] | None = None
        self.annotations_path = (
            annotations_path.resolve()
            if annotations_path
            else self.dataset_dir / "manual-review.json"
        )
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        if not self.audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")
        self.manifest: dict[str, Any] = {}
        self.records_by_name: dict[str, dict[str, Any]] = {}
        self.filenames: list[str] = []
        self.filename_set: set[str] = set()
        self.manifest_signature: tuple[int, int, int] | None = None
        self._refresh_manifest()
        self.lock = threading.Lock()
        self.claim_seconds = max(30, claim_seconds)
        self.reviews, self.claims = self._load_annotations()

    def _selected_filenames(self) -> list[str]:
        subset = self.manifest.get("balanced_listening_subset", {})
        if self.audio_directory == subset.get("local_directory"):
            candidates = [str(name) for name in subset.get("filenames", [])]
        else:
            candidates = [path.name for path in sorted(self.audio_dir.glob("*.wav"))]
        filenames = [
            name
            for name in candidates
            if Path(name).name == name and (self.audio_dir / name).is_file()
        ]
        return filenames

    def _refresh_manifest_file(self) -> bool:
        """Atomically discover newly accepted clips without restarting the app."""
        try:
            stat = self.manifest_path.stat()
        except FileNotFoundError:
            return False
        signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if signature == self.manifest_signature:
            return False
        if self.manifest and self.catalog_path.is_file():
            # Continuous snapshots rewrite a multi-thousand-record manifest very
            # frequently. The catalog is authoritative for review, while the
            # dataset name and paths loaded at startup are stable.
            self.manifest_signature = signature
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False
        self.manifest = manifest
        if self.catalog_path.is_file():
            self.records_by_name = {}
            self.filenames = []
            self.filename_set = set()
        else:
            self.records_by_name = {
                Path(record["local_path"]).name: record
                for record in manifest.get("records", [])
                if record.get("local_path")
            }
            self.filenames = self._selected_filenames()
            self.filename_set = set(self.filenames)
        self.manifest_signature = signature
        return True

    def _refresh_catalog(self) -> bool:
        if not self.catalog_path.is_file():
            return False
        try:
            with closing(self._catalog_connection()) as connection:
                row = connection.execute(
                    "SELECT COUNT(*),MIN(sequence),MAX(sequence) FROM accepted"
                ).fetchone()
        except (OSError, sqlite3.Error):
            return False
        count = int(row[0])
        bounds = (int(row[1]), int(row[2])) if row[1] is not None else None
        changed = (
            count != self.catalog_record_count or bounds != self.catalog_sequence_bounds
        )
        self.catalog_record_count = count
        self.catalog_sequence_bounds = bounds
        return changed

    def _catalog_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.catalog_path}?mode=ro", uri=True, timeout=5
        )
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _catalog_row_record(row: sqlite3.Row) -> dict[str, Any]:
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
        return record

    def _catalog_record(self, filename: str) -> dict[str, Any] | None:
        cached = self.records_by_name.get(filename)
        if cached:
            return cached
        if not self.catalog_path.is_file():
            return None
        try:
            with closing(self._catalog_connection()) as connection:
                row = connection.execute(
                    """SELECT a.sequence,a.accepted_at,r.record_json,
                    m.result_json AS m2d_json,s.result_json AS asr_json
                    FROM accepted a JOIN records r USING(sha256)
                    JOIN m2d_scores m USING(filename)
                    JOIN asr_scores s USING(filename)
                    WHERE r.filename=? LIMIT 1""",
                    (filename,),
                ).fetchone()
        except (OSError, sqlite3.Error):
            return None
        if not row:
            return None
        record = self._catalog_row_record(row)
        self.records_by_name[filename] = record
        return record

    def _known_filename(self, filename: str) -> bool:
        if Path(filename).name != filename:
            return False
        if self.catalog_path.is_file():
            return self._catalog_record(filename) is not None
        return filename in self.filename_set

    def _random_catalog_filename(self) -> str | None:
        """Find one free record with bounded indexed probes over the full catalog."""
        if not self.catalog_sequence_bounds:
            return None
        minimum, maximum = self.catalog_sequence_bounds
        try:
            with closing(self._catalog_connection()) as connection:
                for _ in range(32):
                    start = secrets.randbelow(maximum - minimum + 1) + minimum
                    rows = connection.execute(
                        """SELECT a.sequence,r.filename FROM accepted a
                        JOIN records r USING(sha256) WHERE a.sequence>=?
                        ORDER BY a.sequence LIMIT 64""",
                        (start,),
                    ).fetchall()
                    if len(rows) < 64:
                        rows += connection.execute(
                            """SELECT a.sequence,r.filename FROM accepted a
                            JOIN records r USING(sha256) WHERE a.sequence<?
                            ORDER BY a.sequence LIMIT ?""",
                            (start, 64 - len(rows)),
                        ).fetchall()
                    candidates = [str(row["filename"]) for row in rows]
                    secrets.SystemRandom().shuffle(candidates)
                    for filename in candidates:
                        if filename in self.reviews or filename in self.claims:
                            continue
                        if (self.audio_dir / filename).is_file():
                            return filename
        except (OSError, sqlite3.Error):
            logger.exception("Could not select a random review clip")
        return None

    def _refresh_manifest(self) -> bool:
        manifest_changed = self._refresh_manifest_file()
        catalog_changed = self._refresh_catalog()
        return manifest_changed or catalog_changed

    def _load_annotations(
        self,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        if not self.annotations_path.exists():
            return {}, {}
        payload = json.loads(self.annotations_path.read_text())
        reviews = payload.get("reviews", {})
        if not isinstance(reviews, dict):
            raise ValueError("manual-review.json has an invalid reviews object")
        claims = payload.get("claims", {})
        if not isinstance(claims, dict):
            raise ValueError("manual-review.json has an invalid claims object")
        selected_reviews = {
            filename: review
            for filename, review in reviews.items()
            if (self.catalog_path.is_file() or filename in self.filename_set)
            and isinstance(review, dict)
        }
        selected_claims = {
            filename: claim
            for filename, claim in claims.items()
            if (self.catalog_path.is_file() or filename in self.filename_set)
            and isinstance(claim, dict)
        }
        return selected_reviews, selected_claims

    def _save(self) -> None:
        payload = {
            "schema_version": 2,
            "dataset_dir": str(self.dataset_dir),
            "audio_directory": self.audio_directory,
            "updated_at": _now(),
            "reviews": self.reviews,
            "claims": self.claims,
        }
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.annotations_path.with_suffix(
            self.annotations_path.suffix + ".tmp"
        )
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, self.annotations_path)

    def _claim_expired(self, claim: dict[str, Any]) -> bool:
        try:
            return datetime.fromisoformat(str(claim["expires_at"])) <= datetime.now(UTC)
        except (KeyError, TypeError, ValueError):
            return True

    def _prune_claims(self) -> bool:
        expired = [
            filename
            for filename, claim in self.claims.items()
            if self._claim_expired(claim) or filename in self.reviews
        ]
        for filename in expired:
            self.claims.pop(filename, None)
        return bool(expired)

    def _new_claim(self, identity: ReviewerIdentity) -> dict[str, Any]:
        now = datetime.now(UTC)
        return {
            "reviewer_id": identity.reviewer_id,
            "reviewer_name": identity.reviewer_name.strip(),
            "claimed_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=self.claim_seconds)).isoformat(),
        }

    def _record_summary(self, filename: str) -> dict[str, Any]:
        record = self._catalog_record(filename) or self.records_by_name.get(
            filename, {}
        )
        validation = record.get("m2d_validation", {})
        labels = Counter(
            label["name"]
            for window in validation.get("windows", [])
            for label in window.get("top_labels", [])[:3]
        )
        return {
            "filename": filename,
            "title": record.get("title"),
            "uploader": record.get("uploader"),
            "source_url": record.get("source_url"),
            "background_bucket": validation.get("background_bucket"),
            "speech_coverage": validation.get("speech_coverage"),
            "strong_speech_coverage": validation.get("strong_speech_coverage"),
            "background_coverage": validation.get("background_coverage"),
            "overlap_coverage": validation.get("overlap_coverage"),
            "vocal_music_coverage": validation.get("vocal_music_coverage"),
            "top_labels": [name for name, _ in labels.most_common(6)],
            "review": self.reviews.get(filename),
            "claim": self.claims.get(filename),
        }

    def state(self) -> dict[str, Any]:
        with self.lock:
            self._refresh_manifest()
            if self._prune_claims():
                self._save()
            decisions = Counter(
                review.get("decision") for review in self.reviews.values()
            )
            reviewed = sum(decisions.values())
            active_reviewers = sorted(
                {claim["reviewer_name"] for claim in self.claims.values()}
            )
            total = self.catalog_record_count or len(self.filenames)
            return {
                "dataset": {
                    "name": self.manifest.get("name"),
                    "dataset_dir": str(self.dataset_dir),
                    "audio_directory": self.audio_directory,
                    "annotations_path": str(self.annotations_path),
                },
                "summary": {
                    "total": total,
                    "reviewed": reviewed,
                    "unreviewed": max(0, total - reviewed),
                    "good": decisions["good"],
                    "perfect": decisions["perfect"],
                    "not_ok": decisions["not_ok"],
                    "active_claims": len(self.claims),
                    "available": max(0, total - reviewed - len(self.claims)),
                    "active_reviewers": active_reviewers,
                },
                "reason_labels": {
                    "lacking_voice": "Lacking voice / dialogue",
                    "lacking_music": "Lacking music",
                    "lacking_background_audio": "Lacking background audio / SFX",
                    "vocal_music": "Singing or vocal music",
                    "speech_not_dialogue": "Speech is not dialogue",
                    "too_low_quality": "Too low quality",
                    "too_quiet": "Too quiet",
                    "distorted_or_clipped": "Distorted or clipped",
                    "wrong_balance": "Wrong voice/background balance",
                    "other": "Other",
                },
            }

    def clip(self, filename: str) -> dict[str, Any]:
        with self.lock:
            self._refresh_manifest()
            if not self._known_filename(filename):
                raise KeyError(filename)
            return self._record_summary(filename)

    def audio_path(self, filename: str) -> Path:
        with self.lock:
            self._refresh_manifest()
        if not self._known_filename(filename):
            raise KeyError(filename)
        path = self.audio_dir / filename
        if not path.is_file():
            raise KeyError(filename)
        return path

    def claim_next(self, request: ClaimNextRequest) -> str | None:
        with self.lock:
            self._refresh_manifest()
            changed = self._prune_claims()
            if request.release_filename:
                current = self.claims.get(request.release_filename)
                if current and current.get("reviewer_id") == request.reviewer_id:
                    self.claims.pop(request.release_filename, None)
                    changed = True
            owned = next(
                (
                    filename
                    for filename, claim in self.claims.items()
                    if claim.get("reviewer_id") == request.reviewer_id
                ),
                None,
            )
            if owned:
                self.claims[owned] = self._new_claim(request)
                self._save()
                return owned
            if self.catalog_path.is_file():
                filename = self._random_catalog_filename()
            else:
                candidates = [
                    filename
                    for filename in self.filenames
                    if filename not in self.reviews and filename not in self.claims
                ]
                filename = secrets.choice(candidates) if candidates else None
            if not filename:
                if changed:
                    self._save()
                return None
            self.claims[filename] = self._new_claim(request)
            self._save()
            return filename

    def claim(self, filename: str, identity: ReviewerIdentity) -> bool:
        with self.lock:
            self._refresh_manifest()
            if not self._known_filename(filename):
                raise KeyError(filename)
            self._prune_claims()
            if filename in self.reviews:
                return False
            existing = self.claims.get(filename)
            if existing and existing.get("reviewer_id") != identity.reviewer_id:
                raise ClaimConflict(
                    f"This clip is being reviewed by {existing['reviewer_name']}"
                )
            for other_filename, other_claim in list(self.claims.items()):
                if (
                    other_filename != filename
                    and other_claim.get("reviewer_id") == identity.reviewer_id
                ):
                    self.claims.pop(other_filename, None)
            self.claims[filename] = self._new_claim(identity)
            self._save()
            return True

    def heartbeat(self, filename: str, identity: ReviewerIdentity) -> dict[str, Any]:
        with self.lock:
            self._refresh_manifest()
            if not self._known_filename(filename):
                raise KeyError(filename)
            self._prune_claims()
            claim = self.claims.get(filename)
            if not claim or claim.get("reviewer_id") != identity.reviewer_id:
                raise ClaimConflict("This clip is no longer assigned to you")
            claim = self._new_claim(identity)
            self.claims[filename] = claim
            self._save()
            return claim

    def release(self, filename: str, identity: ReviewerIdentity) -> None:
        with self.lock:
            self._refresh_manifest()
            if not self._known_filename(filename):
                raise KeyError(filename)
            self._prune_claims()
            claim = self.claims.get(filename)
            if claim and claim.get("reviewer_id") == identity.reviewer_id:
                self.claims.pop(filename, None)
                self._save()

    def update(self, filename: str, update: ReviewUpdate) -> dict[str, Any]:
        if update.decision == "not_ok" and not (update.reasons or update.note.strip()):
            raise ValueError("Not OK requires at least one reason or a note")
        if "other" in update.reasons and not update.note.strip():
            raise ValueError("The Other reason requires a note")
        reasons = (
            list(dict.fromkeys(update.reasons)) if update.decision == "not_ok" else []
        )
        note = update.note.strip() if update.decision == "not_ok" else ""
        review = {
            "decision": update.decision,
            "reasons": reasons,
            "note": note,
            "updated_at": _now(),
            "reviewer_id": update.reviewer_id,
            "reviewer_name": update.reviewer_name.strip(),
        }
        with self.lock:
            self._refresh_manifest()
            if not self._known_filename(filename):
                raise KeyError(filename)
            self._prune_claims()
            claim = self.claims.get(filename)
            if not claim or claim.get("reviewer_id") != update.reviewer_id:
                raise ClaimConflict("This clip is not assigned to you")
            if filename in self.reviews:
                raise ClaimConflict("This clip has already been reviewed")
            self.reviews[filename] = review
            self.claims.pop(filename, None)
            self._save()
        return review

    def clear(self, filename: str, identity: ReviewerIdentity) -> None:
        with self.lock:
            self._refresh_manifest()
            if not self._known_filename(filename):
                raise KeyError(filename)
            existing = self.reviews.get(filename)
            if existing and existing.get("reviewer_id") not in {
                None,
                identity.reviewer_id,
            }:
                raise ClaimConflict(
                    f"Only {existing.get('reviewer_name', 'the original reviewer')} "
                    "can clear this mark"
                )
            self.reviews.pop(filename, None)
            self._save()

    def export_csv(self) -> str:
        with self.lock:
            self._refresh_manifest()
        destination = io.StringIO()
        fields = [
            "filename",
            "decision",
            "reasons",
            "note",
            "updated_at",
            "reviewer_id",
            "reviewer_name",
            "background_bucket",
            "title",
            "source_url",
        ]
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        filenames = (
            sorted(self.reviews) if self.catalog_path.is_file() else self.filenames
        )
        for filename in filenames:
            summary = self._record_summary(filename)
            review = self.reviews.get(filename, {})
            writer.writerow(
                {
                    "filename": filename,
                    "decision": review.get("decision", ""),
                    "reasons": "|".join(review.get("reasons", [])),
                    "note": review.get("note", ""),
                    "updated_at": review.get("updated_at", ""),
                    "reviewer_id": review.get("reviewer_id", ""),
                    "reviewer_name": review.get("reviewer_name", ""),
                    "background_bucket": summary["background_bucket"],
                    "title": summary["title"],
                    "source_url": summary["source_url"],
                }
            )
        return destination.getvalue()


def create_review_app(
    store: ReviewStore,
    progress_store: PipelineProgressStore | ContinuousProgressStore | None = None,
    training_store: TrainingSnapshotStore | None = None,
) -> FastAPI:
    app = FastAPI(title="SAM Audio Manual Review", version="2.0.0")
    html_path = Path(__file__).parent / "web" / "manual_review.html"
    progress_html_path = Path(__file__).parent / "web" / "pipeline_progress.html"
    training_html_path = Path(__file__).parent / "web" / "training_review.html"
    training_balance_html_path = Path(__file__).parent / "web" / "training_balance.html"

    def page() -> HTMLResponse:
        return HTMLResponse(html_path.read_text())

    @app.get("/")
    def index() -> HTMLResponse:
        return page()

    @app.get("/clip/{filename}")
    def clip_page(filename: str) -> HTMLResponse:
        try:
            store.audio_path(filename)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Clip not found") from error
        return page()

    @app.get("/progress")
    def progress_page() -> HTMLResponse:
        if not progress_store:
            raise HTTPException(status_code=404, detail="Progress dashboard disabled")
        return HTMLResponse(progress_html_path.read_text())

    @app.get("/training")
    def training_index_page() -> HTMLResponse:
        if not training_store:
            raise HTTPException(status_code=404, detail="Training review disabled")
        return HTMLResponse(training_html_path.read_text())

    @app.get("/training/balance")
    def training_balance_page() -> HTMLResponse:
        if not training_store:
            raise HTTPException(status_code=404, detail="Training review disabled")
        return HTMLResponse(training_balance_html_path.read_text())

    @app.get("/training/{snapshot_id}")
    @app.get("/training/{snapshot_id}/{record_id}")
    def training_page(
        snapshot_id: str | None = None, record_id: str | None = None
    ) -> HTMLResponse:
        if not training_store:
            raise HTTPException(status_code=404, detail="Training review disabled")
        if snapshot_id:
            try:
                training_store._snapshot_row(snapshot_id)
            except KeyError as error:
                raise HTTPException(
                    status_code=404, detail="Snapshot not found"
                ) from error
        return HTMLResponse(training_html_path.read_text())

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return store.state()

    @app.get("/api/progress")
    def progress() -> dict[str, Any]:
        if not progress_store:
            raise HTTPException(status_code=404, detail="Progress dashboard disabled")
        payload = progress_store.snapshot()
        if isinstance(progress_store, ContinuousProgressStore):
            materialized = int(payload.get("counts", {}).get("accepted") or 0)
        else:
            store.state()
            materialized = store.catalog_record_count or len(store.filenames)
        payload["review_snapshot"] = {
            "materialized": materialized,
            "path": str(store.dataset_dir),
        }
        if training_store:
            payload["training_transformation"] = training_store.transformation_summary(
                source_clips=materialized
            )
        return payload

    @app.get("/api/progress/download-workers/{worker}")
    def download_worker_progress(worker: str) -> dict[str, Any]:
        if not isinstance(progress_store, ContinuousProgressStore):
            raise HTTPException(status_code=404, detail="Worker telemetry unavailable")
        if re.fullmatch(r"download-\d+", worker) is None:
            raise HTTPException(status_code=404, detail="Download worker not found")
        detail = progress_store.download_worker_detail(worker)
        if detail is None:
            raise HTTPException(status_code=404, detail="Download worker not found")
        return detail

    @app.get("/api/training/snapshots")
    def training_snapshots() -> dict[str, Any]:
        if not training_store:
            raise HTTPException(status_code=404, detail="Training review disabled")
        return training_store.snapshots()

    @app.get("/api/training/balance")
    def training_balance() -> dict[str, Any]:
        if not training_store:
            raise HTTPException(status_code=404, detail="Training review disabled")
        return training_store.balance_summary()

    @app.get("/api/training/balance/tags")
    def training_balance_tags(
        scope: str = "all",
        query: str = "",
        role: str = "all",
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        if not training_store:
            raise HTTPException(status_code=404, detail="Training review disabled")
        try:
            return training_store.balance_tags(
                scope=scope,
                query=query,
                role=role,
                offset=offset,
                limit=limit,
            )
        except KeyError as error:
            raise HTTPException(
                status_code=400, detail="Invalid balance tag filter"
            ) from error

    @app.get("/api/training/snapshots/{snapshot_id}/records")
    def training_record(
        snapshot_id: str,
        position: int = 0,
        record_id: str | None = None,
    ) -> dict[str, Any]:
        if not training_store:
            raise HTTPException(status_code=404, detail="Training review disabled")
        try:
            return training_store.record(
                snapshot_id, position=position, record_id=record_id
            )
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="Training record not found"
            ) from error

    @app.put(
        "/api/training/snapshots/{snapshot_id}/records/{record_id}/quality-override"
    )
    def update_training_quality_override(
        snapshot_id: str,
        record_id: str,
        update: TrainingQualityOverrideUpdate,
    ) -> dict[str, Any]:
        if not training_store:
            raise HTTPException(status_code=404, detail="Training review disabled")
        try:
            manual_override = training_store.set_quality_override(
                snapshot_id, record_id, update
            )
            record = training_store.record(snapshot_id, record_id=record_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="Training record not found"
            ) from error
        return {
            "record_id": record_id,
            "automated_quality_bucket": record["automated_quality_bucket"],
            "manual_quality_override": manual_override,
            "effective_quality_bucket": record["effective_quality_bucket"],
        }

    @app.delete(
        "/api/training/snapshots/{snapshot_id}/records/{record_id}/quality-override"
    )
    def delete_training_quality_override(
        snapshot_id: str, record_id: str
    ) -> dict[str, Any]:
        if not training_store:
            raise HTTPException(status_code=404, detail="Training review disabled")
        try:
            training_store.clear_quality_override(snapshot_id, record_id)
            record = training_store.record(snapshot_id, record_id=record_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="Training record not found"
            ) from error
        return {
            "record_id": record_id,
            "automated_quality_bucket": record["automated_quality_bucket"],
            "manual_quality_override": None,
            "effective_quality_bucket": record["automated_quality_bucket"],
        }

    @app.get("/api/training/audio/{snapshot_id}/{record_id}/{stem}")
    def training_audio(
        snapshot_id: str,
        record_id: str,
        stem: str,
        range_header: str | None = Header(default=None, alias="Range"),
    ) -> Any:
        if not training_store:
            raise HTTPException(status_code=404, detail="Training review disabled")
        try:
            local, bucket, key = training_store.audio_location(
                snapshot_id, record_id, stem
            )
        except (KeyError, ValueError) as error:
            raise HTTPException(
                status_code=404, detail="Training audio not found"
            ) from error
        if local is not None:
            return FileResponse(local, media_type="audio/wav")
        arguments: dict[str, Any] = {"Bucket": bucket, "Key": key}
        if range_header:
            arguments["Range"] = range_header
        try:
            response = training_store.s3_client().get_object(**arguments)
        except Exception as error:
            logger.warning("Could not stream training audio %s: %s", key, error)
            raise HTTPException(
                status_code=404, detail="Training audio not found"
            ) from error
        body = response["Body"]
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(response.get("ContentLength", "")),
        }
        if response.get("ContentRange"):
            headers["Content-Range"] = str(response["ContentRange"])

        def chunks() -> Any:
            try:
                yield from body.iter_chunks(chunk_size=256 * 1024)
            finally:
                body.close()

        return StreamingResponse(
            chunks(),
            status_code=206 if response.get("ContentRange") else 200,
            media_type="audio/wav",
            headers=headers,
        )

    @app.get("/healthz")
    def health() -> dict[str, Any]:
        return {"status": "ready", "summary": store.state()["summary"]}

    @app.get("/api/audio/{filename}")
    def audio(filename: str) -> FileResponse:
        try:
            path = store.audio_path(filename)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Clip not found") from error
        return FileResponse(path, media_type="audio/wav")

    @app.get("/api/clips/{filename}")
    def clip(filename: str) -> dict[str, Any]:
        try:
            return store.clip(filename)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Clip not found") from error

    @app.put("/api/reviews/{filename}")
    def update_review(filename: str, update: ReviewUpdate) -> dict[str, Any]:
        try:
            review = store.update(filename, update)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Clip not found") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except ClaimConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {
            "filename": filename,
            "review": review,
            "summary": store.state()["summary"],
        }

    @app.delete("/api/reviews/{filename}")
    def clear_review(filename: str, identity: ReviewerIdentity) -> dict[str, Any]:
        try:
            store.clear(filename, identity)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Clip not found") from error
        except ClaimConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {
            "filename": filename,
            "review": None,
            "summary": store.state()["summary"],
        }

    @app.post("/api/claims/next")
    def claim_next(request: ClaimNextRequest) -> dict[str, Any]:
        filename = store.claim_next(request)
        return {
            "filename": filename,
            "clip": store.clip(filename) if filename else None,
            "summary": store.state()["summary"],
        }

    @app.post("/api/claims/{filename}")
    def claim(filename: str, identity: ReviewerIdentity) -> dict[str, Any]:
        try:
            claimed = store.claim(filename, identity)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Clip not found") from error
        except ClaimConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"filename": filename, "claimed": claimed}

    @app.put("/api/claims/{filename}")
    def heartbeat(filename: str, identity: ReviewerIdentity) -> dict[str, Any]:
        try:
            claim_record = store.heartbeat(filename, identity)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Clip not found") from error
        except ClaimConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"filename": filename, "claim": claim_record}

    @app.delete("/api/claims/{filename}")
    def release(filename: str, identity: ReviewerIdentity) -> dict[str, Any]:
        try:
            store.release(filename, identity)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Clip not found") from error
        return {"filename": filename, "released": True}

    @app.get("/api/export.csv")
    def export_csv() -> StreamingResponse:
        return StreamingResponse(
            iter([store.export_csv()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=manual-review.csv"},
        )

    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--audio-directory", default="balanced-audio")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--claim-seconds", type=int, default=600)
    parser.add_argument("--progress-batch-dir", type=Path, action="append", default=[])
    parser.add_argument("--progress-final-dir", type=Path)
    parser.add_argument("--progress-target", type=int, default=1000)
    parser.add_argument("--continuous-workspace", type=Path)
    parser.add_argument("--snapshot-size", type=int, default=2500)
    parser.add_argument("--training-workspace", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    store = ReviewStore(
        args.dataset_dir,
        audio_directory=args.audio_directory,
        annotations_path=args.annotations,
        claim_seconds=args.claim_seconds,
    )
    progress_store = None
    if args.continuous_workspace:
        progress_store = ContinuousProgressStore(
            args.continuous_workspace, snapshot_size=args.snapshot_size
        )
    elif args.progress_batch_dir:
        progress_store = PipelineProgressStore(
            args.progress_batch_dir,
            final_dir=args.progress_final_dir,
            target=args.progress_target,
        )
    training_store = (
        TrainingSnapshotStore(args.training_workspace)
        if args.training_workspace
        else None
    )
    uvicorn.run(
        create_review_app(store, progress_store, training_store),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
