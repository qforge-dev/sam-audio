from __future__ import annotations

from typing import Any

from sam_audio_pipeline.config import Settings
from sam_audio_pipeline.handlers import FlamingoHandler


class FakeAWS:
    def __init__(self):
        self.items: list[dict[str, Any]] = [
            {"entity": "source", "audible_chunk_count": 1},
            {"entity": "chunk", "status": "complete"},
            {"entity": "stem", "stem_type": "music"},
            {"entity": "stem", "stem_type": "voice"},
            {"entity": "model_task", "status": "complete"},
        ]
        self.job_updates: list[dict[str, Any]] = []

    def query_partition(self, _: str):
        return self.items

    def update(self, _: str, sk: str, values: dict[str, Any]) -> None:
        if sk == "META":
            self.job_updates.append(values)


def settings() -> Settings:
    return Settings(
        aws_region="us-east-1",
        bucket="bucket",
        table_name="table",
        ingest_queue_url="ingest",
        sam_queue_url="sam",
        flamingo_queue_url="flamingo",
        sam_api_url="http://127.0.0.1:8000",
        flamingo_api_url="http://127.0.0.1:8001",
        chunk_seconds=30,
        overlap_seconds=5,
        gate_peak_dbfs=-52,
        gate_rms_dbfs=-60,
        presign_seconds=3600,
    )


def test_job_waits_for_every_expected_flamingo_task() -> None:
    aws = FakeAWS()
    handler = FlamingoHandler(settings(), aws)

    handler._refresh_job("job-1")
    assert aws.job_updates == []

    aws.items.extend(
        [
            {"entity": "model_task", "status": "complete"},
            {"entity": "model_task", "status": "complete"},
        ]
    )
    handler._refresh_job("job-1")
    assert aws.job_updates[-1]["status"] == "complete"
