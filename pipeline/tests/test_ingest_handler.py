from __future__ import annotations

import math
import shutil
import struct
import wave
from pathlib import Path
from typing import Any

from sam_audio_pipeline.config import Settings
from sam_audio_pipeline.handlers import IngestHandler
from sam_audio_pipeline.schema import QueueTask


class FakeAWS:
    def __init__(self, source_path: Path):
        self.source_path = source_path
        self.events: list[str] = []
        self.model_tasks: set[str] = set()
        self.items: dict[str, dict[str, Any]] = {
            "META": {"SK": "META", "entity": "job", "status": "queued"},
            "SOURCE#source-1": {
                "SK": "SOURCE#source-1",
                "entity": "source",
                "source_id": "source-1",
                "filename": "tone.wav",
                "s3_key": "source.wav",
                "status": "queued",
            },
        }

    def get(self, _: str, sk: str) -> dict[str, Any] | None:
        return self.items.get(sk)

    def update(self, _: str, sk: str, values: dict[str, Any]) -> None:
        self.items[sk].update(values)

    def download_file(self, _: str, destination: Path) -> None:
        shutil.copyfile(self.source_path, destination)

    def upload_file(self, *_: Any) -> None:
        pass

    def put(self, item: dict[str, Any]) -> None:
        self.items[item["SK"]] = item

    def put_model_task(self, task: QueueTask, *_: Any) -> bool:
        if task.task_id in self.model_tasks:
            return False
        self.model_tasks.add(task.task_id)
        return True

    def object_exists(self, _: str) -> bool:
        return True

    def send_task(self, _: str, task: QueueTask) -> None:
        self.events.append(task.task_type)

    def query_partition(self, _: str) -> list[dict[str, Any]]:
        return list(self.items.values())


def write_tone(path: Path, *, channels: int = 2) -> None:
    rate = 8000
    frames = bytearray()
    for index in range(rate):
        value = round(0.2 * 32767 * math.sin(2 * math.pi * 440 * index / rate))
        frames.extend(struct.pack("<" + "h" * channels, *([value] * channels)))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(frames)


def test_ingest_persists_source_size_before_temp_directory_cleanup(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tone.wav"
    write_tone(source)
    aws = FakeAWS(source)
    settings = Settings(
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

    IngestHandler(settings, aws).handle(
        QueueTask(
            task_id="ingest-1",
            task_type="ingest_source",
            job_id="job-1",
            source_id="source-1",
        )
    )

    record = aws.items["SOURCE#source-1"]
    assert record["bytes"] == source.stat().st_size
    assert record["duration_seconds"] == 1.0
    assert record["status"] == "analyzing"
    assert record["input_channels"] == 2
    assert aws.items["CHUNK#source-1#000000"]["status"] == "waiting_scene"
    assert aws.events == ["describe_scene"]


def test_ingest_skips_non_stereo_input_before_chunking(tmp_path: Path) -> None:
    source = tmp_path / "mono.wav"
    write_tone(source, channels=1)
    aws = FakeAWS(source)
    settings = Settings(
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

    IngestHandler(settings, aws).handle(
        QueueTask(
            task_id="ingest-mono",
            task_type="ingest_source",
            job_id="job-1",
            source_id="source-1",
        )
    )

    record = aws.items["SOURCE#source-1"]
    assert record["status"] == "complete"
    assert record["skip_reason"] == "non_stereo_input"
    assert record["input_channels"] == 1
    assert record["chunk_count"] == 0
    assert not any(key.startswith("CHUNK#") for key in aws.items)
    assert aws.events == []


def test_ingest_retry_does_not_overwrite_a_completed_chunk(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    write_tone(source)
    aws = FakeAWS(source)
    settings = Settings(
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
    handler = IngestHandler(settings, aws)
    task = QueueTask(
        task_id="ingest-1",
        task_type="ingest_source",
        job_id="job-1",
        source_id="source-1",
    )
    handler.handle(task)
    aws.items["CHUNK#source-1#000000"]["status"] = "complete"
    aws.items["SOURCE#source-1"]["status"] = "running"

    handler.handle(task)

    assert aws.items["CHUNK#source-1#000000"]["status"] == "complete"
    assert aws.events == ["describe_scene"]
