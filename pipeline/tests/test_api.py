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
        self,
        job_id: str,
        source_id: str,
        filename: str,
        s3_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.put(
            {
                "PK": f"JOB#{job_id}",
                "SK": f"SOURCE#{source_id}",
                "entity": "source",
                "job_id": job_id,
                "source_id": source_id,
                "filename": filename,
                "s3_key": s3_key,
                "source_metadata": metadata or {},
                "status": "uploading",
            }
        )

    def presign_upload(self, key: str, content_type: str) -> str:
        return f"https://uploads.invalid/{key}?type={content_type}"

    def object_exists(self, key: str) -> bool:
        return key in self.existing_objects

    def object_size(self, key: str) -> int:
        item = next(
            (value for value in self.items.values() if value.get("s3_key") == key),
            {},
        )
        return int(item.get("bytes") or 0)

    def send_task(self, queue_url: str, task: Any) -> None:
        self.sent.append((queue_url, task))

    def queue_metrics(self, _: str):
        return {"queued": 0, "in_flight": 0, "delayed": 0}

    def count_index(self, index_pk: str) -> int:
        return sum(item.get("GSI1PK") == index_pk for item in self.items.values())

    def presign_download(self, key: str) -> str:
        return f"https://downloads.invalid/{key}"


def settings() -> Settings:
    return Settings(
        aws_region="us-east-1",
        bucket="test-bucket",
        table_name="test-table",
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


def test_dataset_overview_and_source_explorer_report_real_artifacts() -> None:
    config = settings()
    aws = FakeAWS(config)
    with TestClient(create_app(config, aws)) as client:
        dataset = client.post("/v1/datasets", json={"name": "AudioSet 100"}).json()
        created = client.post(
            "/v1/jobs",
            json={
                "dataset_id": dataset["dataset_id"],
                "filenames": ["dog.wav"],
                "source_metadata": {
                    "dog.wav": {
                        "video_id": "abc123",
                        "start_seconds": 30,
                        "end_seconds": 40,
                    }
                },
            },
        ).json()
        job_id = created["job_id"]
        source_id = created["uploads"][0]["source_id"]
        pk = f"JOB#{job_id}"
        aws.update(
            pk,
            f"SOURCE#{source_id}",
            {
                "status": "complete",
                "bytes": 1000,
                "duration_seconds": 10.0,
            },
        )
        aws.put(
            {
                "PK": pk,
                "SK": f"CHUNK#{source_id}#000000",
                "entity": "chunk",
                "source_id": source_id,
                "chunk_id": "000000",
                "status": "complete",
                "start_seconds": 0,
                "end_seconds": 10,
                "s3_key": "chunk.wav",
                "bytes": 50,
                "gate": {"audible": True},
            }
        )
        for index, stem_type in enumerate(("music", "sfx"), start=1):
            aws.put(
                {
                    "PK": pk,
                    "SK": f"STEM#{source_id}#000000#{stem_type}",
                    "entity": "stem",
                    "job_id": job_id,
                    "source_id": source_id,
                    "chunk_id": "000000",
                    "stem_type": stem_type,
                    "s3_key": f"{stem_type}.wav",
                    "bytes": index * 100,
                    "automatic_status": "success",
                    "effective_status": "success",
                    "settings": {},
                    **(
                        {
                            "stereo_s3_key": "music.stereo.wav",
                            "stereo_bytes": 220,
                            "stereo_mapping": {"frequency_bands": 32},
                        }
                        if stem_type == "music"
                        else {}
                    ),
                }
            )
        aws.put(
            {
                "PK": pk,
                "SK": f"STEM_RESULT#{source_id}#000000#voice",
                "entity": "stem_result",
                "source_id": source_id,
                "chunk_id": "000000",
                "stem_type": "voice",
                "status": "skipped",
            }
        )

        overview = client.get(f"/v1/datasets/{dataset['dataset_id']}/overview").json()
        detail = client.get(f"/v1/jobs/{job_id}/sources/{source_id}").json()

    assert overview["summary"] == {
        "jobs": 1,
        "sources": 1,
        "non_stereo_sources": 0,
        "duration_seconds": 10.0,
        "input_bytes": 1000,
        "chunk_bytes": 50,
        "stem_bytes": 300,
        "stereo_bytes": 220,
        "total_bytes": 1570,
        "chunks": 1,
        "audible_chunks": 1,
        "skipped_chunks": 0,
        "stems": 2,
        "stereo_stems": 1,
        "stems_by_type": {"music": 1, "sfx": 1},
        "stems_by_status": {"success": 2},
        "selected_routes": {},
        "job_statuses": {"uploading": 1},
        "review_remaining": 0,
    }
    assert detail["source"]["source_metadata"]["video_id"] == "abc123"
    assert [stem["stem_type"] for stem in detail["chunks"][0]["stems"]] == [
        "music",
        "sfx",
    ]
    assert detail["chunks"][0]["stems"][0]["stereo_audio_url"].endswith(
        "music.stereo.wav"
    )
    assert detail["chunks"][0]["stems"][1]["stereo_audio_url"] is None
    assert detail["chunks"][0]["omitted_stems"] == ["voice"]
