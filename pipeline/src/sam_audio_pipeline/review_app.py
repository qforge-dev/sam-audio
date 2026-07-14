"""Local browser app for manually reviewing an acquired audio dataset."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import secrets
import sqlite3
import threading
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

Decision = Literal["good", "perfect", "not_ok"]
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
                        "maximum 24 clips per source video"
                    ),
                },
            }


class ContinuousProgressStore:
    """Expose the permanent SQLite-backed pipeline without batch semantics."""

    def __init__(self, workspace: Path, *, snapshot_size: int = 5000):
        self.workspace = workspace.resolve()
        self.snapshot_size = snapshot_size

    def snapshot(self) -> dict[str, Any]:
        from .continuous_dataset import progress_snapshot

        return progress_snapshot(self.workspace, self.snapshot_size)


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
        self.review_window_records = 5000
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
        try:
            manifest = json.loads(self.manifest_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False
        self.manifest = manifest
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
            connection = sqlite3.connect(
                f"file:{self.catalog_path}?mode=ro", uri=True, timeout=5
            )
            connection.row_factory = sqlite3.Row
            count = int(
                connection.execute("SELECT COUNT(*) FROM accepted").fetchone()[0]
            )
            if count == self.catalog_record_count:
                connection.close()
                return False
            from .continuous_dataset import catalog_records

            records = catalog_records(
                connection,
                limit=self.review_window_records,
                newest_first=True,
            )
            connection.close()
        except (OSError, sqlite3.Error):
            return False
        self.records_by_name = {
            Path(record["local_path"]).name: record for record in records
        }
        self.filenames = [
            Path(record["local_path"]).name
            for record in records
            if (self.audio_dir / Path(record["local_path"]).name).is_file()
        ]
        self.filename_set = set(self.filenames)
        self.catalog_record_count = count
        return True

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
            if filename in self.filename_set and isinstance(claim, dict)
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
        record = self.records_by_name.get(filename, {})
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
            clips = [self._record_summary(filename) for filename in self.filenames]
            decisions = Counter(
                review.get("decision") for review in self.reviews.values()
            )
            reviewed = sum(decisions.values())
            active_reviewers = sorted(
                {claim["reviewer_name"] for claim in self.claims.values()}
            )
            total = self.catalog_record_count or len(self.filenames)
            reviewed_in_window = sum(
                filename in self.reviews for filename in self.filenames
            )
            return {
                "dataset": {
                    "name": self.manifest.get("name"),
                    "dataset_dir": str(self.dataset_dir),
                    "audio_directory": self.audio_directory,
                    "annotations_path": str(self.annotations_path),
                    "loaded_review_window": len(self.filenames),
                },
                "summary": {
                    "total": total,
                    "reviewed": reviewed,
                    "unreviewed": max(0, total - reviewed),
                    "good": decisions["good"],
                    "perfect": decisions["perfect"],
                    "not_ok": decisions["not_ok"],
                    "active_claims": len(self.claims),
                    "available": max(
                        0, len(self.filenames) - reviewed_in_window - len(self.claims)
                    ),
                    "loaded_for_review": len(self.filenames),
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
                "clips": clips,
            }

    def audio_path(self, filename: str) -> Path:
        with self.lock:
            self._refresh_manifest()
        if Path(filename).name != filename or filename not in self.filename_set:
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
            candidates = [
                filename
                for filename in self.filenames
                if filename not in self.reviews and filename not in self.claims
            ]
            if not candidates:
                if changed:
                    self._save()
                return None
            filename = secrets.choice(candidates)
            self.claims[filename] = self._new_claim(request)
            self._save()
            return filename

    def claim(self, filename: str, identity: ReviewerIdentity) -> bool:
        with self.lock:
            self._refresh_manifest()
            if filename not in self.filename_set:
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
            if filename not in self.filename_set:
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
            if filename not in self.filename_set:
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
            if filename not in self.filename_set:
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
            if filename not in self.filename_set:
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
        for filename in self.filenames:
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
) -> FastAPI:
    app = FastAPI(title="SAM Audio Manual Review", version="2.0.0")
    html_path = Path(__file__).parent / "web" / "manual_review.html"
    progress_html_path = Path(__file__).parent / "web" / "pipeline_progress.html"

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

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return store.state()

    @app.get("/api/progress")
    def progress() -> dict[str, Any]:
        if not progress_store:
            raise HTTPException(status_code=404, detail="Progress dashboard disabled")
        payload = progress_store.snapshot()
        store.state()
        payload["review_snapshot"] = {
            "materialized": store.catalog_record_count or len(store.filenames),
            "path": str(store.dataset_dir),
        }
        return payload

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
        return {"filename": filename, "summary": store.state()["summary"]}

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
    parser.add_argument("--snapshot-size", type=int, default=5000)
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
    uvicorn.run(
        create_review_app(store, progress_store), host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
