from __future__ import annotations

from collections import defaultdict
from typing import Any

from fastapi.testclient import TestClient

from sam_audio_pipeline.api import create_app
from sam_audio_pipeline.config import Settings


class FakeAWS:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.items: dict[tuple[str, str], dict[str, Any]] = {}
        self.existing_objects: set[str] = set()
        self.sent: list[tuple[str, Any]] = []

    def put(self, item: dict[str, Any]) -> None:
        self.items[(item["PK"], item["SK"])] = item

    def get(self, pk: str, sk: str):
        return self.items.get((pk, sk))

    def update(self, pk: str, sk: str, values: dict[str, Any], **_: Any) -> None:
        self.items[(pk, sk)].update(values)

    def query_partition(self, pk: str):
        return [item for (item_pk, _), item in self.items.items() if item_pk == pk]

    def query_index(self, index_pk: str, **_: Any):
        return [item for item in self.items.values() if item.get("GSI1PK") == index_pk]

    def ensure_dataset(self, dataset_id: str, name: str):
        existing = self.get(f"DATASET#{dataset_id}", "META")
        if existing:
            return existing
        item = {
            "PK": f"DATASET#{dataset_id}",
            "SK": "META",
            "entity": "dataset",
            "dataset_id": dataset_id,
            "name": name,
            "created_at": "2026-07-13T12:00:00Z",
            "GSI1PK": "DATASETS",
            "GSI1SK": f"2026#{dataset_id}",
        }
        self.put(item)
        return item

    def create_job(self, job_id: str, dataset_id: str, source_count: int) -> None:
        self.put(
            {
                "PK": f"JOB#{job_id}",
                "SK": "META",
                "entity": "job",
                "job_id": job_id,
                "dataset_id": dataset_id,
                "source_count": source_count,
                "status": "uploading",
                "created_at": "2026-07-13T12:00:00Z",
                "updated_at": "2026-07-13T12:00:00Z",
                "GSI1PK": "JOBS",
                "GSI1SK": f"2026#{job_id}",
            }
        )

    def create_source(
        self, job_id: str, source_id: str, filename: str, s3_key: str
    ) -> None:
        self.put(
            {
                "PK": f"JOB#{job_id}",
                "SK": f"SOURCE#{source_id}",
                "entity": "source",
                "source_id": source_id,
                "filename": filename,
                "s3_key": s3_key,
                "status": "uploading",
            }
        )

    def presign_upload(self, key: str, content_type: str) -> str:
        return f"https://uploads.invalid/{key}?type={content_type}"

    def object_exists(self, key: str) -> bool:
        return key in self.existing_objects

    def send_task(self, queue_url: str, task: Any) -> None:
        self.sent.append((queue_url, task))

    def queue_metrics(self, _: str):
        return {"queued": 0, "in_flight": 0, "delayed": 0}


def settings() -> Settings:
    return Settings(
        aws_region="us-east-1",
        bucket="test-bucket",
        table_name="test-table",
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


def test_persistent_dataset_accepts_successive_unbounded_jobs() -> None:
    config = settings()
    aws = FakeAWS(config)
    with TestClient(create_app(config, aws)) as client:
        dataset = client.post("/v1/datasets", json={"name": "Library A"})
        assert dataset.status_code == 201
        dataset_id = dataset.json()["dataset_id"]

        filenames = [f"track-{index}.wav" for index in range(250)]
        first = client.post(
            "/v1/jobs",
            json={"dataset_id": dataset_id, "filenames": filenames},
        )
        second = client.post(
            "/v1/jobs",
            json={"dataset_id": dataset_id, "filenames": ["another.wav"]},
        )

    assert first.status_code == 201
    assert len(first.json()["uploads"]) == 250
    assert second.status_code == 201
    assert first.json()["job_id"] != second.json()["job_id"]


def test_upload_confirmation_is_durable_before_enqueue() -> None:
    config = settings()
    aws = FakeAWS(config)
    with TestClient(create_app(config, aws)) as client:
        response = client.post("/v1/jobs", json={"filenames": ["one.wav", "two.mp3"]})
        job = response.json()
        for upload in job["uploads"]:
            aws.existing_objects.add(upload["s3_key"])
        complete = client.post(f"/v1/jobs/{job['job_id']}/uploads-complete", json={})

    assert complete.status_code == 202
    assert len(aws.sent) == 2
    assert all(queue == "ingest" for queue, _ in aws.sent)
    statuses = defaultdict(int)
    for item in aws.query_partition(f"JOB#{job['job_id']}"):
        statuses[item.get("status")] += 1
    assert statuses["queued"] == 3  # job plus two source records
