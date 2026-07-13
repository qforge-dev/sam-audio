from __future__ import annotations

from typing import Any

from sam_audio_pipeline.config import Settings
from sam_audio_pipeline.handlers import Reconciler


class FakeAWS:
    def __init__(self):
        self.existing_objects = {"source.wav"}
        self.items: list[dict[str, Any]] = [
            {"entity": "job", "job_id": "job-1", "status": "running"},
            {
                "PK": "JOB#job-1",
                "SK": "SOURCE#source-1",
                "entity": "source",
                "source_id": "source-1",
                "status": "uploading",
                "s3_key": "source.wav",
            },
            {
                "PK": "JOB#job-1",
                "SK": "CHUNK#source-2#000001",
                "entity": "chunk",
                "source_id": "source-2",
                "chunk_id": "000001",
                "status": "queued",
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
        ]
        self.sent: list[tuple[str, Any]] = []

    def query_index(self, *_: Any, **__: Any):
        return self.items[:1]

    def query_partition(self, *_: Any, **__: Any):
        return self.items[1:]

    def object_exists(self, key: str) -> bool:
        return key in self.existing_objects

    def update(self, _: str, sk: str, values: dict[str, Any]) -> None:
        next(item for item in self.items if item.get("SK") == sk).update(values)

    def send_task(self, queue_url: str, task: Any) -> None:
        self.sent.append((queue_url, task))


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


def test_reconciler_recovers_uploaded_source_and_stale_chunk() -> None:
    aws = FakeAWS()

    recovered = Reconciler(settings(), aws).run_once()

    assert recovered == {"ingest": 1, "sam": 1}
    assert [queue for queue, _ in aws.sent] == ["ingest", "sam"]
    assert all(item["status"] == "queued" for item in aws.items[1:])


def test_reconciler_restores_completed_chunk_from_durable_stems() -> None:
    aws = FakeAWS()
    aws.items = [
        {"entity": "job", "job_id": "job-1", "status": "running"},
        {
            "PK": "JOB#job-1",
            "SK": "CHUNK#source-1#000000",
            "entity": "chunk",
            "source_id": "source-1",
            "chunk_id": "000000",
            "status": "queued",
            "updated_at": "2026-07-13T00:00:00+00:00",
        },
        {
            "PK": "JOB#job-1",
            "SK": "STEM#source-1#000000#sfx",
            "entity": "stem",
            "source_id": "source-1",
            "chunk_id": "000000",
            "stem_type": "sfx",
            "automatic_status": "failure",
        },
    ]
    aws.existing_objects.add("jobs/job-1/metadata/source-1/000000.json")

    recovered = Reconciler(settings(), aws).run_once()

    assert recovered == {"ingest": 0, "sam": 0, "sam_completed": 1}
    assert aws.items[1]["status"] == "complete"
    assert aws.items[1]["verification_status"] == "failure"
    assert aws.sent == []


def test_reconciler_restores_chunk_when_every_output_was_gated() -> None:
    aws = FakeAWS()
    aws.items = [
        {"entity": "job", "job_id": "job-1", "status": "running"},
        {
            "PK": "JOB#job-1",
            "SK": "CHUNK#source-1#000000",
            "entity": "chunk",
            "source_id": "source-1",
            "chunk_id": "000000",
            "status": "queued",
            "updated_at": "2026-07-13T00:00:00+00:00",
        },
        {
            "PK": "JOB#job-1",
            "SK": "STEM_RESULT#source-1#000000#music",
            "entity": "stem_result",
            "source_id": "source-1",
            "chunk_id": "000000",
            "stem_type": "music",
            "status": "skipped",
        },
    ]
    aws.existing_objects.add("jobs/job-1/metadata/source-1/000000.json")

    recovered = Reconciler(settings(), aws).run_once()

    assert recovered == {"ingest": 0, "sam": 0, "sam_completed": 1}
    assert aws.items[1]["status"] == "complete"
    assert aws.items[1]["stored_stems"] == []
    assert aws.sent == []


def test_reconciler_repairs_stale_completed_source_counter() -> None:
    aws = FakeAWS()
    job = {
        "PK": "JOB#job-1",
        "SK": "META",
        "entity": "job",
        "job_id": "job-1",
        "status": "complete",
        "source_count": 1,
        "completed_sources": 0,
    }
    aws.items = [
        job,
        {
            "PK": "JOB#job-1",
            "SK": "SOURCE#source-1",
            "entity": "source",
            "source_id": "source-1",
            "status": "chunked",
            "audible_chunk_count": 1,
        },
        {
            "PK": "JOB#job-1",
            "SK": "CHUNK#source-1#000000",
            "entity": "chunk",
            "source_id": "source-1",
            "chunk_id": "000000",
            "status": "complete",
        },
        {
            "entity": "stem",
            "source_id": "source-1",
            "chunk_id": "000000",
            "stem_type": "music",
        },
        {"entity": "model_task", "status": "complete"},
        {"entity": "model_task", "status": "complete"},
    ]
    aws.query_partition = lambda *_: aws.items

    recovered = Reconciler(settings(), aws).run_once()

    assert recovered == {"ingest": 0, "sam": 0}
    assert job["status"] == "complete"
    assert job["completed_sources"] == 1
    assert job["completed_chunks"] == 1
