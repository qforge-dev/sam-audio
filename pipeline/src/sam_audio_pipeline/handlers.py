"""Idempotent task handlers for CPU ingestion and SAM inference."""

from __future__ import annotations

import logging
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .audio import chunk_audio, gate_wav, sha256_file
from .aws import PipelineAWS
from .config import Settings
from .model_client import SAMAudioClient
from .schema import QueueTask, StemRecord, VerificationStatus, utc_now

logger = logging.getLogger(__name__)

TERMINAL_CHUNK_STATES = {"complete", "failed", "skipped"}
ASSERTIONS = {
    "music": "This should contain only music.",
    "voice": "This should contain only human voices.",
    "sfx": "This should contain no music and no voices.",
}
PROMPTS = {
    "music": "music soundtrack",
    "voice": "human voices",
    "sfx": "residual after music and voice separation",
}


class IngestHandler:
    def __init__(self, settings: Settings, aws: PipelineAWS):
        self.settings = settings
        self.aws = aws

    def handle(self, task: QueueTask) -> None:
        source_key = f"SOURCE#{task.source_id}"
        source = self.aws.get(f"JOB#{task.job_id}", source_key)
        if not source:
            raise KeyError(f"Source does not exist: {task.source_id}")
        if source.get("status") in {"chunked", "complete"}:
            return
        self.aws.update(
            f"JOB#{task.job_id}",
            source_key,
            {"status": "running", "updated_at": utc_now()},
        )
        with tempfile.TemporaryDirectory(prefix="sam-ingest-") as temporary:
            root = Path(temporary)
            suffix = Path(source["filename"]).suffix or ".audio"
            local_source = root / f"source{suffix}"
            self.aws.download_file(source["s3_key"], local_source)
            chunks = chunk_audio(
                local_source,
                root / "chunks",
                chunk_seconds=self.settings.chunk_seconds,
                overlap_seconds=self.settings.overlap_seconds,
                peak_threshold_dbfs=self.settings.gate_peak_dbfs,
                rms_threshold_dbfs=self.settings.gate_rms_dbfs,
            )
            source_sha256 = sha256_file(local_source)
            audible_count = 0
            for chunk in chunks:
                chunk_key = (
                    f"jobs/{task.job_id}/chunks/{task.source_id}/{chunk.chunk_id}.wav"
                )
                item = {
                    "PK": f"JOB#{task.job_id}",
                    "SK": f"CHUNK#{task.source_id}#{chunk.chunk_id}",
                    "entity": "chunk",
                    "job_id": task.job_id,
                    "source_id": task.source_id,
                    "chunk_id": chunk.chunk_id,
                    "start_seconds": chunk.start_seconds,
                    "end_seconds": chunk.end_seconds,
                    "sha256": chunk.sha256,
                    "gate": chunk.gate.model_dump(mode="json"),
                    "status": "queued" if chunk.gate.audible else "skipped",
                    "s3_key": chunk_key if chunk.gate.audible else None,
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                }
                self.aws.put(item)
                if not chunk.gate.audible:
                    continue
                audible_count += 1
                self.aws.upload_file(chunk.path, chunk_key, "audio/wav")
                separation_task = QueueTask(
                    task_id=f"sam:{task.job_id}:{task.source_id}:{chunk.chunk_id}",
                    task_type="separate_chunk",
                    job_id=task.job_id,
                    source_id=task.source_id,
                    chunk_id=chunk.chunk_id,
                )
                self.aws.send_task(self.settings.sam_queue_url, separation_task)

        if audible_count:
            scene_task = QueueTask(
                task_id=f"scene:{task.job_id}:{task.source_id}",
                task_type="describe_scene",
                job_id=task.job_id,
                source_id=task.source_id,
            )
            self.aws.send_task(self.settings.flamingo_queue_url, scene_task)
        status = "chunked" if audible_count else "complete"
        self.aws.update(
            f"JOB#{task.job_id}",
            source_key,
            {
                "status": status,
                "chunk_count": len(chunks),
                "audible_chunk_count": audible_count,
                "source_sha256": source_sha256,
                "updated_at": utc_now(),
            },
        )
        self.aws.update(
            f"JOB#{task.job_id}",
            "META",
            {"status": "running", "updated_at": utc_now()},
        )


class SeparationHandler:
    def __init__(self, settings: Settings, aws: PipelineAWS):
        self.settings = settings
        self.aws = aws
        self.client = SAMAudioClient(settings.sam_api_url)

    def handle(self, task: QueueTask) -> None:
        if not task.chunk_id:
            raise ValueError("separate_chunk task is missing chunk_id")
        chunk_sk = f"CHUNK#{task.source_id}#{task.chunk_id}"
        chunk = self.aws.get(f"JOB#{task.job_id}", chunk_sk)
        if not chunk:
            raise KeyError(f"Chunk does not exist: {task.chunk_id}")
        if chunk.get("status") == "complete":
            return
        self.aws.update(
            f"JOB#{task.job_id}",
            chunk_sk,
            {"status": "running", "updated_at": utc_now()},
        )
        with tempfile.TemporaryDirectory(prefix="sam-separate-") as temporary:
            root = Path(temporary)
            input_path = root / "chunk.wav"
            self.aws.download_file(chunk["s3_key"], input_path)
            result = self.client.separate(input_path, root / "result")
            statuses = self._statuses(result.metadata)
            stage_by_kind = {
                kind: f"stage{index}"
                for index, kind in enumerate(
                    result.metadata.get("cascade_order", []), start=1
                )
            }
            stored_stems: list[str] = []
            for stem_type, stem_path in result.stems.items():
                gate = gate_wav(
                    stem_path,
                    peak_threshold_dbfs=self.settings.gate_peak_dbfs,
                    rms_threshold_dbfs=self.settings.gate_rms_dbfs,
                )
                stem_key = (
                    f"jobs/{task.job_id}/stems/{task.source_id}/"
                    f"{task.chunk_id}/{stem_type}.wav"
                )
                if not gate.audible:
                    self.aws.put(
                        {
                            "PK": f"JOB#{task.job_id}",
                            "SK": (
                                f"STEM_RESULT#{task.source_id}#{task.chunk_id}#"
                                f"{stem_type}"
                            ),
                            "entity": "stem_result",
                            "job_id": task.job_id,
                            "source_id": task.source_id,
                            "chunk_id": task.chunk_id,
                            "stem_type": stem_type,
                            "status": "skipped",
                            "reason": "sound_gate",
                            "gate": gate.model_dump(mode="json"),
                            "created_at": utc_now(),
                        }
                    )
                    continue
                self.aws.upload_file(stem_path, stem_key, "audio/wav")
                stage_name = stage_by_kind.get(stem_type)
                stage = (
                    result.metadata.get("stages", {}).get(stage_name, {})
                    if stage_name
                    else {}
                )
                scores = stage.get("verification", {}) if stage else {}
                timings = (
                    result.metadata.get("inference_timings_ms", {}).get(stage_name, {})
                    if stage_name
                    else result.metadata.get("inference_timings_ms", {})
                )
                record = StemRecord(
                    job_id=task.job_id,
                    source_id=task.source_id,
                    chunk_id=task.chunk_id,
                    stem_type=stem_type,
                    prompt=PROMPTS[stem_type],
                    assertion=ASSERTIONS[stem_type],
                    s3_key=stem_key,
                    sha256=sha256_file(stem_path),
                    automatic_status=statuses[stem_type],
                    effective_status=statuses[stem_type],
                    model=result.metadata.get("model", "unknown"),
                    settings={
                        "dtype_policy": result.metadata.get("dtype_policy"),
                        "predict_spans": result.metadata.get("predict_spans"),
                        "cascade_order": result.metadata.get("cascade_order"),
                        "stage": stage_name,
                    },
                    scores=scores,
                    timings_ms=timings,
                )
                self.aws.put_stem(record)
                stored_stems.append(stem_type)
                self._enqueue_flamingo(task, stem_type)
            metadata_key = (
                f"jobs/{task.job_id}/metadata/{task.source_id}/{task.chunk_id}.json"
            )
            self.aws.upload_json(result.metadata, metadata_key)
        self.aws.update(
            f"JOB#{task.job_id}",
            chunk_sk,
            {
                "status": "complete",
                "stored_stems": stored_stems,
                "verification_status": result.metadata.get(
                    "verification_status", "uncertain"
                ),
                "metadata_s3_key": metadata_key,
                "updated_at": utc_now(),
            },
        )
        self._refresh_job(task.job_id)

    @staticmethod
    def _statuses(metadata: dict[str, Any]) -> dict[str, VerificationStatus]:
        verification = metadata.get("verification", {})
        stages = verification.get("stage_statuses", {})
        cascade_order = metadata.get("cascade_order", ["music", "voice"])
        by_kind = {
            kind: stages.get(f"stage{index}", "uncertain")
            for index, kind in enumerate(cascade_order, start=1)
        }
        return {
            "music": VerificationStatus(by_kind.get("music", "uncertain")),
            "voice": VerificationStatus(by_kind.get("voice", "uncertain")),
            "sfx": VerificationStatus(metadata.get("verification_status", "uncertain")),
        }

    def _enqueue_flamingo(self, task: QueueTask, stem_type: str) -> None:
        task_type = {
            "music": "describe_music",
            "voice": "transcribe_voice",
        }.get(stem_type)
        if not task_type:
            return
        self.aws.send_task(
            self.settings.flamingo_queue_url,
            QueueTask(
                task_id=(
                    f"flamingo:{task_type}:{task.job_id}:{task.source_id}:"
                    f"{task.chunk_id}"
                ),
                task_type=task_type,
                job_id=task.job_id,
                source_id=task.source_id,
                chunk_id=task.chunk_id,
            ),
        )

    def _refresh_job(self, job_id: str) -> None:
        items = self.aws.query_partition(f"JOB#{job_id}")
        chunks = [item for item in items if item.get("entity") == "chunk"]
        sources = [item for item in items if item.get("entity") == "source"]
        if (
            not chunks
            or not sources
            or not all(
                source.get("status") in {"chunked", "complete"} for source in sources
            )
            or not all(chunk.get("status") in TERMINAL_CHUNK_STATES for chunk in chunks)
        ):
            return
        failures = [chunk for chunk in chunks if chunk.get("status") == "failed"]
        status = "partial" if failures else "stems_ready"
        self.aws.update(
            f"JOB#{job_id}",
            "META",
            {
                "status": status,
                "completed_chunks": len(chunks),
                "failed_chunks": len(failures),
                "updated_at": utc_now(),
            },
        )


class Reconciler:
    def __init__(self, settings: Settings, aws: PipelineAWS):
        self.settings = settings
        self.aws = aws

    @staticmethod
    def _stale(item: dict[str, Any], now: datetime) -> bool:
        value = item.get("updated_at") or item.get("created_at")
        if not value:
            return True
        updated_at = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        threshold = (
            timedelta(hours=2)
            if item.get("status") == "running"
            else timedelta(minutes=15)
        )
        return now - updated_at > threshold

    def run_once(self) -> dict[str, int]:
        now = datetime.now(UTC)
        recovered = {"ingest": 0, "sam": 0}
        jobs = self.aws.query_index("JOBS", limit=10_000, newest_first=False)
        for job in jobs:
            if job.get("status") in {"complete", "failed"}:
                continue
            job_id = job["job_id"]
            items = self.aws.query_partition(f"JOB#{job_id}")
            for item in items:
                entity = item.get("entity")
                status = item.get("status")
                if entity == "source":
                    if status == "uploading" and self.aws.object_exists(item["s3_key"]):
                        self._enqueue_ingest(job_id, item)
                        recovered["ingest"] += 1
                    elif status in {"queued", "running"} and self._stale(item, now):
                        self._enqueue_ingest(job_id, item)
                        recovered["ingest"] += 1
                elif (
                    entity == "chunk"
                    and status in {"queued", "running"}
                    and self._stale(item, now)
                ):
                    self._enqueue_sam(job_id, item)
                    recovered["sam"] += 1
        return recovered

    def _enqueue_ingest(self, job_id: str, source: dict[str, Any]) -> None:
        task = QueueTask(
            task_id=uuid.uuid4().hex,
            task_type="ingest_source",
            job_id=job_id,
            source_id=source["source_id"],
        )
        self.aws.update(
            f"JOB#{job_id}",
            source["SK"],
            {"status": "queued", "updated_at": task.created_at},
        )
        self.aws.send_task(self.settings.ingest_queue_url, task)

    def _enqueue_sam(self, job_id: str, chunk: dict[str, Any]) -> None:
        task = QueueTask(
            task_id=uuid.uuid4().hex,
            task_type="separate_chunk",
            job_id=job_id,
            source_id=chunk["source_id"],
            chunk_id=chunk["chunk_id"],
        )
        self.aws.update(
            f"JOB#{job_id}",
            chunk["SK"],
            {"status": "queued", "updated_at": task.created_at},
        )
        self.aws.send_task(self.settings.sam_queue_url, task)
