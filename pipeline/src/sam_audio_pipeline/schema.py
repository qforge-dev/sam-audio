"""Stable job, task, stem, and review contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class VerificationStatus(StrEnum):
    SUCCESS = "success"
    UNCERTAIN = "uncertain"
    FAILURE = "failure"


class ProcessingStatus(StrEnum):
    UPLOADING = "uploading"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class ReviewDecision(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    PENDING = "pending"


class SourceUpload(BaseModel):
    source_id: str
    filename: str
    content_type: str = "application/octet-stream"
    s3_key: str
    upload_url: str


class JobCreateRequest(BaseModel):
    filenames: list[str] = Field(min_length=1)
    dataset_id: str = "default"
    source_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)


class DatasetCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class DatasetResponse(BaseModel):
    dataset_id: str
    name: str
    created_at: str


class JobCreateResponse(BaseModel):
    job_id: str
    dataset_id: str
    status: ProcessingStatus
    uploads: list[SourceUpload]


class UploadCompleteRequest(BaseModel):
    source_ids: list[str] | None = None


class QueueTask(BaseModel):
    task_id: str
    task_type: Literal[
        "ingest_source",
        "separate_chunk",
        "describe_scene",
        "describe_music",
        "transcribe_voice",
    ]
    job_id: str
    source_id: str
    chunk_id: str | None = None
    attempt: int = 0
    created_at: str = Field(default_factory=utc_now)


class GateMetrics(BaseModel):
    audible: bool
    peak_dbfs: float
    rms_dbfs: float
    duration_seconds: float
    sample_rate: int
    channels: int


class StemRecord(BaseModel):
    job_id: str
    source_id: str
    chunk_id: str
    stem_type: Literal["music", "voice", "sfx"]
    prompt: str
    assertion: str
    s3_key: str
    sha256: str
    bytes: int
    automatic_status: VerificationStatus
    effective_status: VerificationStatus
    model: str
    settings: dict[str, Any]
    scores: dict[str, Any]
    timings_ms: dict[str, Any]
    created_at: str = Field(default_factory=utc_now)


class ReviewRequest(BaseModel):
    decision: ReviewDecision
    note: str = Field(default="", max_length=1000)


class ReviewItem(BaseModel):
    review_id: str
    job_id: str
    source_id: str
    chunk_id: str
    stem_type: str
    prompt: str
    assertion: str
    automatic_status: VerificationStatus
    effective_status: VerificationStatus
    audio_url: str | None = None
    scores: dict[str, Any] = Field(default_factory=dict)
    timings_ms: dict[str, Any] = Field(default_factory=dict)
