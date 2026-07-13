"""Environment-backed pipeline configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


@dataclass(frozen=True)
class Settings:
    aws_region: str
    bucket: str
    table_name: str
    ingest_queue_url: str
    sam_queue_url: str
    flamingo_queue_url: str
    sam_api_url: str
    chunk_seconds: float
    overlap_seconds: float
    gate_peak_dbfs: float
    gate_rms_dbfs: float
    presign_seconds: int

    @classmethod
    def from_env(cls) -> Settings:
        chunk_seconds = _float("SAM_PIPELINE_CHUNK_SECONDS", 30.0)
        overlap_seconds = _float("SAM_PIPELINE_OVERLAP_SECONDS", 5.0)
        if chunk_seconds <= 0:
            raise ValueError("SAM_PIPELINE_CHUNK_SECONDS must be positive")
        if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
            raise ValueError(
                "SAM_PIPELINE_OVERLAP_SECONDS must be non-negative and shorter "
                "than a chunk"
            )
        return cls(
            aws_region=os.environ.get("AWS_REGION", "us-east-1"),
            bucket=_required("SAM_PIPELINE_BUCKET"),
            table_name=_required("SAM_PIPELINE_TABLE"),
            ingest_queue_url=_required("SAM_PIPELINE_INGEST_QUEUE_URL"),
            sam_queue_url=_required("SAM_PIPELINE_SAM_QUEUE_URL"),
            flamingo_queue_url=_required("SAM_PIPELINE_FLAMINGO_QUEUE_URL"),
            sam_api_url=os.environ.get(
                "SAM_PIPELINE_SAM_API_URL", "http://127.0.0.1:8000"
            ).rstrip("/"),
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            gate_peak_dbfs=_float("SAM_PIPELINE_GATE_PEAK_DBFS", -52.0),
            gate_rms_dbfs=_float("SAM_PIPELINE_GATE_RMS_DBFS", -60.0),
            presign_seconds=int(os.environ.get("SAM_PIPELINE_PRESIGN_SECONDS", "3600")),
        )
