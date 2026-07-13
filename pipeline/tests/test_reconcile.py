from __future__ import annotations

from typing import Any

from sam_audio_pipeline.config import Settings
from sam_audio_pipeline.handlers import Reconciler


class FakeAWS:
    def __init__(self):
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
        return key == "source.wav"

    def update(self, _: str, sk: str, values: dict[str, Any]) -> None:
        next(item for item in self.items if item.get("SK") == sk).update(values)

    def send_task(self, queue_url: str, task: Any) -> None:
        self.sent.append((queue_url, task))


def test_reconciler_recovers_uploaded_source_and_stale_chunk() -> None:
    settings = Settings(
        aws_region="us-east-1",
        bucket="bucket",
        table_name="table",
        ingest_queue_url="ingest",
        sam_queue_url="sam",
        flamingo_queue_url="flamingo",
        sam_api_url="http://127.0.0.1:8000",
        chunk_seconds=30,
        overlap_seconds=5,
        gate_peak_dbfs=-52,
        gate_rms_dbfs=-60,
        presign_seconds=3600,
    )
    aws = FakeAWS()

    recovered = Reconciler(settings, aws).run_once()

    assert recovered == {"ingest": 1, "sam": 1}
    assert [queue for queue, _ in aws.sent] == ["ingest", "sam"]
    assert all(item["status"] == "queued" for item in aws.items[1:])
