"""Idempotent task handlers for CPU ingestion and SAM inference."""

from __future__ import annotations

import ast
import json
import logging
import tempfile
import uuid
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .audio import chunk_audio, gate_wav, probe_channels, probe_duration, sha256_file
from .aws import PipelineAWS
from .config import Settings
from .flamingo_client import AudioFlamingoClient
from .model_client import SAMAudioClient, SeparationResult
from .schema import QueueTask, StemRecord, VerificationStatus, utc_now
from .stereo import StereoMappedStem, map_stems_to_stereo

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


def _scene_presence(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    value: Any = None
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(cleaned)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            break
    parsed = isinstance(value, dict) and isinstance(
        value.get("has_music"), bool
    ) and isinstance(value.get("has_voices"), bool)
    return {
        "has_music": bool(value.get("has_music")) if parsed else True,
        "has_voices": bool(value.get("has_voices")) if parsed else True,
        "parsed": parsed,
        "raw_text": text,
    }


def _terminal_job_summary(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the durable terminal summary once every source has been split."""
    sources = [item for item in items if item.get("entity") == "source"]
    chunks = [item for item in items if item.get("entity") == "chunk"]
    if (
        not sources
        or not chunks
        or not all(
            source.get("status") in {"chunked", "complete"} for source in sources
        )
        or not all(chunk.get("status") in TERMINAL_CHUNK_STATES for chunk in chunks)
    ):
        return None
    stems = [item for item in items if item.get("entity") == "stem"]
    tasks = [item for item in items if item.get("entity") == "model_task"]
    expected_tasks = sum(
        1 for source in sources if source.get("audible_chunk_count", 0) > 0
    ) + sum(1 for stem in stems if stem.get("stem_type") in {"music", "voice"})
    annotations_complete = (
        expected_tasks > 0
        and len(tasks) >= expected_tasks
        and all(task.get("status") == "complete" for task in tasks)
    )
    failed_chunks = sum(chunk.get("status") == "failed" for chunk in chunks)
    if failed_chunks:
        status = "partial" if annotations_complete else "stems_ready"
    else:
        status = "complete" if annotations_complete else "stems_ready"
    return {
        "status": status,
        "completed_sources": len(sources),
        "completed_chunks": len(chunks),
        "failed_chunks": failed_chunks,
        "updated_at": utc_now(),
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
            source_bytes = local_source.stat().st_size
            source_channels = probe_channels(local_source)
            source_sha256 = sha256_file(local_source)
            if source_channels != 2:
                self.aws.update(
                    f"JOB#{task.job_id}",
                    source_key,
                    {
                        "status": "complete",
                        "skip_reason": "non_stereo_input",
                        "input_channels": source_channels,
                        "chunk_count": 0,
                        "audible_chunk_count": 0,
                        "bytes": source_bytes,
                        "duration_seconds": probe_duration(local_source),
                        "source_sha256": source_sha256,
                        "updated_at": utc_now(),
                    },
                )
                self._refresh_job(task.job_id)
                return
            chunks = chunk_audio(
                local_source,
                root / "chunks",
                chunk_seconds=self.settings.chunk_seconds,
                overlap_seconds=self.settings.overlap_seconds,
                peak_threshold_dbfs=self.settings.gate_peak_dbfs,
                rms_threshold_dbfs=self.settings.gate_rms_dbfs,
            )
            audible_count = 0
            for chunk in chunks:
                chunk_key = (
                    f"jobs/{task.job_id}/chunks/{task.source_id}/{chunk.chunk_id}.wav"
                )
                chunk_sk = f"CHUNK#{task.source_id}#{chunk.chunk_id}"
                existing = self.aws.get(f"JOB#{task.job_id}", chunk_sk)
                if existing:
                    if existing.get("gate", {}).get("audible"):
                        audible_count += 1
                        if not self.aws.object_exists(str(existing["s3_key"])):
                            self.aws.upload_file(chunk.path, chunk_key, "audio/wav")
                        if existing.get("status") == "failed":
                            self.aws.update(
                                f"JOB#{task.job_id}",
                                chunk_sk,
                                {"status": "waiting_scene", "updated_at": utc_now()},
                            )
                    continue
                item = {
                    "PK": f"JOB#{task.job_id}",
                    "SK": chunk_sk,
                    "entity": "chunk",
                    "job_id": task.job_id,
                    "source_id": task.source_id,
                    "chunk_id": chunk.chunk_id,
                    "start_seconds": chunk.start_seconds,
                    "end_seconds": chunk.end_seconds,
                    "sha256": chunk.sha256,
                    "gate": chunk.gate.model_dump(mode="json"),
                    "status": "waiting_scene" if chunk.gate.audible else "skipped",
                    "s3_key": chunk_key if chunk.gate.audible else None,
                    "bytes": chunk.path.stat().st_size if chunk.gate.audible else 0,
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                }
                self.aws.put(item)
                if not chunk.gate.audible:
                    continue
                audible_count += 1
                self.aws.upload_file(chunk.path, chunk_key, "audio/wav")
        status = "analyzing" if audible_count else "complete"
        self.aws.update(
            f"JOB#{task.job_id}",
            source_key,
            {
                "status": status,
                "chunk_count": len(chunks),
                "audible_chunk_count": audible_count,
                "bytes": source_bytes,
                "duration_seconds": max(
                    (chunk.end_seconds for chunk in chunks), default=0.0
                ),
                "source_sha256": source_sha256,
                "input_channels": source_channels,
                "updated_at": utc_now(),
            },
        )
        if audible_count:
            scene_task = QueueTask(
                task_id=f"scene:{task.job_id}:{task.source_id}",
                task_type="describe_scene",
                job_id=task.job_id,
                source_id=task.source_id,
            )
            if self.aws.put_model_task(scene_task, "flamingo"):
                self.aws.send_task(self.settings.flamingo_queue_url, scene_task)
        self._refresh_job(task.job_id)

    def _refresh_job(self, job_id: str) -> None:
        items = self.aws.query_partition(f"JOB#{job_id}")
        sources = [item for item in items if item.get("entity") == "source"]
        chunks = [item for item in items if item.get("entity") == "chunk"]
        all_sound_gated = (
            bool(sources)
            and all(source.get("status") == "complete" for source in sources)
            and all(chunk.get("status") == "skipped" for chunk in chunks)
        )
        values: dict[str, Any] = {"status": "running", "updated_at": utc_now()}
        if all_sound_gated:
            values.update(
                {
                    "status": "complete",
                    "completed_sources": len(sources),
                    "completed_chunks": len(chunks),
                    "failed_chunks": 0,
                }
            )
        self.aws.update(f"JOB#{job_id}", "META", values)


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
            targets = tuple(
                task.targets
                if task.targets is not None
                else chunk.get("presence_gate", {}).get(
                    "targets", ["music", "voice"]
                )
            )
            result = self._separate_with_policy(input_path, root, targets=targets)
            result.metadata["presence_gate"] = chunk.get("presence_gate", {})
            statuses = self._statuses(result.metadata, set(result.stems))
            stage_by_kind = {
                kind: f"stage{index}"
                for index, kind in enumerate(
                    result.metadata.get("cascade_order", []), start=1
                )
            }
            gates = {
                stem_type: gate_wav(
                    stem_path,
                    peak_threshold_dbfs=self.settings.gate_peak_dbfs,
                    rms_threshold_dbfs=self.settings.gate_rms_dbfs,
                )
                for stem_type, stem_path in result.stems.items()
            }
            stereo_stems: dict[str, StereoMappedStem] = {}
            try:
                stereo_stems = map_stems_to_stereo(
                    input_path, result.stems, root / "stereo"
                )
                result.metadata["stereo_mapping"] = {
                    "algorithm": "frequency_masked_pan_v2",
                    "stems": {
                        stem_type: mapped.metadata
                        for stem_type, mapped in stereo_stems.items()
                    },
                }
            except Exception:
                logger.exception(
                    "Stereo mapping failed; raw stems remain durable for backfill"
                )
            stored_stems: list[str] = []
            for stem_type, stem_path in result.stems.items():
                gate = gates[stem_type]
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
                stereo = stereo_stems.get(stem_type)
                stereo_key = None
                if stereo:
                    stereo_key = (
                        f"jobs/{task.job_id}/stems/{task.source_id}/"
                        f"{task.chunk_id}/{stem_type}.stereo.wav"
                    )
                    self.aws.upload_file(stereo.path, stereo_key, "audio/wav")
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
                timings = dict(timings)
                if stereo:
                    timings["stereo_mapping"] = stereo.metadata["processing_ms"]
                record = StemRecord(
                    job_id=task.job_id,
                    source_id=task.source_id,
                    chunk_id=task.chunk_id,
                    stem_type=stem_type,
                    prompt=PROMPTS[stem_type],
                    assertion=ASSERTIONS[stem_type],
                    s3_key=stem_key,
                    sha256=sha256_file(stem_path),
                    bytes=stem_path.stat().st_size,
                    stereo_s3_key=stereo_key,
                    stereo_sha256=stereo.sha256 if stereo else None,
                    stereo_bytes=stereo.bytes if stereo else 0,
                    stereo_mapping=stereo.metadata if stereo else {},
                    automatic_status=statuses[stem_type],
                    effective_status=statuses[stem_type],
                    model=result.metadata.get("model", "unknown"),
                    settings={
                        "dtype_policy": result.metadata.get("dtype_policy"),
                        "predict_spans": result.metadata.get("predict_spans"),
                        "cascade_order": result.metadata.get("cascade_order"),
                        "adaptive_routing": result.metadata.get("adaptive_routing"),
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

    def _separate_with_policy(
        self,
        input_path: Path,
        root: Path,
        *,
        targets: tuple[str, ...] = ("music", "voice"),
    ) -> SeparationResult:
        if not targets:
            metadata: dict[str, Any] = {
                "schema_version": 4,
                "model": "semantic_presence_passthrough",
                "verification_status": "success",
                "verification": {
                    "status": "success",
                    "stage_statuses": {},
                    "processing_policy": (
                        "Source-scene preflight found no music or voices; SAM "
                        "inference was skipped and the stereo input was retained "
                        "as the SFX stem."
                    ),
                },
                "requested_order": "none",
                "requested_targets": [],
                "cascade_order": [],
                "inference_timings_ms": {"service_total": 0.0},
                "stages": {},
                "adaptive_routing": {
                    "policy_version": 2,
                    "trigger": "semantic_presence_no_targets",
                    "attempts": [],
                    "selected_order": "sfx_only",
                    "selection_rule": "No separation target was present.",
                },
            }
            return SeparationResult(
                directory=root,
                metadata=metadata,
                stems={"sfx": input_path},
                response_headers={},
            )
        if len(targets) == 1:
            target = targets[0]
            order = "music_first" if target == "music" else "voice_first"
            selected = self.client.separate(
                input_path,
                root / f"{target}-only",
                order=order,
                targets=targets,
            )
            selected.metadata["adaptive_routing"] = {
                "policy_version": 2,
                "default_order": order,
                "trigger": "semantic_presence_single_target",
                "attempts": [self._attempt_summary(selected.metadata)],
                "selected_order": f"{target}_only",
                "selection_rule": (
                    f"Source-scene preflight requested only the {target} stage."
                ),
            }
            return selected
        primary = self.client.separate(
            input_path,
            root / "music-first",
            order="music_first",
            targets=targets,
        )
        attempts = [self._attempt_summary(primary.metadata)]
        selected = primary
        trigger = "primary_accepted"
        if primary.metadata.get("verification_status") == "failure":
            fallback = self.client.separate(
                input_path,
                root / "voice-first",
                order="voice_first",
                targets=targets,
            )
            attempts.append(self._attempt_summary(fallback.metadata))
            trigger = "primary_failure_retry"
            if self._route_score(fallback.metadata) > self._route_score(
                primary.metadata
            ):
                selected = fallback
        selected.metadata["adaptive_routing"] = {
            "policy_version": 2,
            "default_order": "music_first",
            "trigger": trigger,
            "attempts": attempts,
            "selected_order": selected.metadata.get("requested_order"),
            "selection_rule": (
                "Prefer success over uncertain over failure for final and stage "
                "statuses, then prefer higher stage Judge quality; ties retain "
                "music_first."
            ),
        }
        return selected

    @staticmethod
    def _route_score(metadata: dict[str, Any]) -> float:
        status_points = {"failure": 0.0, "uncertain": 1.0, "success": 2.0}
        final = status_points.get(str(metadata.get("verification_status")), 0.0)
        verification = metadata.get("verification", {})
        stages = verification.get("stage_statuses", {})
        stage_score = sum(
            status_points.get(str(stages.get(f"stage{index}")), 0.0) for index in (1, 2)
        )
        judge_score = 0.0
        for stage in metadata.get("stages", {}).values():
            value = stage.get("verification", {}).get("judge_quality_score")
            if isinstance(value, (int, float)):
                judge_score += float(value)
        return final * 100.0 + stage_score * 10.0 + judge_score

    @classmethod
    def _attempt_summary(cls, metadata: dict[str, Any]) -> dict[str, Any]:
        verification = metadata.get("verification", {})
        stages = verification.get("stage_statuses", {})
        quality_by_kind: dict[str, Any] = {}
        status_by_kind: dict[str, Any] = {}
        for index, kind in enumerate(metadata.get("cascade_order", []), start=1):
            stage = metadata.get("stages", {}).get(f"stage{index}", {})
            status_by_kind[kind] = stages.get(f"stage{index}")
            quality_by_kind[kind] = stage.get("verification", {}).get(
                "judge_quality_score"
            )
        return {
            "order": metadata.get("requested_order"),
            "final_status": metadata.get("verification_status"),
            "status_by_kind": status_by_kind,
            "judge_quality_by_kind": quality_by_kind,
            "route_score": cls._route_score(metadata),
            "service_total_ms": metadata.get("inference_timings_ms", {}).get(
                "service_total"
            ),
        }

    @staticmethod
    def _statuses(
        metadata: dict[str, Any], stem_types: set[str]
    ) -> dict[str, VerificationStatus]:
        verification = metadata.get("verification", {})
        stages = verification.get("stage_statuses", {})
        cascade_order = metadata.get("cascade_order", ["music", "voice"])
        by_kind = {
            kind: stages.get(f"stage{index}", "uncertain")
            for index, kind in enumerate(cascade_order, start=1)
        }
        statuses = {
            kind: VerificationStatus(by_kind.get(kind, "uncertain"))
            for kind in stem_types & {"music", "voice"}
        }
        if "sfx" in stem_types:
            statuses["sfx"] = VerificationStatus(
                metadata.get("verification_status", "uncertain")
            )
        return statuses

    def _enqueue_flamingo(self, task: QueueTask, stem_type: str) -> None:
        task_type = {
            "music": "describe_music",
            "voice": "transcribe_voice",
        }.get(stem_type)
        if not task_type:
            return
        flamingo_task = QueueTask(
            task_id=(
                f"flamingo:{task_type}:{task.job_id}:{task.source_id}:{task.chunk_id}"
            ),
            task_type=task_type,
            job_id=task.job_id,
            source_id=task.source_id,
            chunk_id=task.chunk_id,
        )
        if self.aws.put_model_task(flamingo_task, "flamingo"):
            self.aws.send_task(self.settings.flamingo_queue_url, flamingo_task)

    def _refresh_job(self, job_id: str) -> None:
        items = self.aws.query_partition(f"JOB#{job_id}")
        summary = _terminal_job_summary(items)
        if not summary:
            return
        self.aws.update(f"JOB#{job_id}", "META", summary)


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
            if job.get("status") == "failed" or (
                job.get("status") == "complete"
                and int(job.get("completed_sources") or 0)
                >= int(job.get("source_count") or 0)
            ):
                continue
            job_id = job["job_id"]
            items = self.aws.query_partition(f"JOB#{job_id}")
            stems_by_chunk: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
                list
            )
            results_by_chunk: dict[
                tuple[str, str], list[dict[str, Any]]
            ] = defaultdict(list)
            for stem in items:
                if stem.get("entity") == "stem":
                    stems_by_chunk[
                        (str(stem["source_id"]), str(stem["chunk_id"]))
                    ].append(stem)
                elif stem.get("entity") == "stem_result":
                    results_by_chunk[
                        (str(stem["source_id"]), str(stem["chunk_id"]))
                    ].append(stem)
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
                elif entity == "chunk" and status in {"queued", "running"}:
                    chunk_stems = stems_by_chunk.get(
                        (str(item["source_id"]), str(item["chunk_id"])), []
                    )
                    chunk_results = results_by_chunk.get(
                        (str(item["source_id"]), str(item["chunk_id"])), []
                    )
                    metadata_key = (
                        f"jobs/{job_id}/metadata/{item['source_id']}/"
                        f"{item['chunk_id']}.json"
                    )
                    if (chunk_stems or chunk_results) and self.aws.object_exists(
                        metadata_key
                    ):
                        sfx = next(
                            (
                                stem
                                for stem in chunk_stems
                                if stem.get("stem_type") == "sfx"
                            ),
                            None,
                        )
                        self.aws.update(
                            f"JOB#{job_id}",
                            item["SK"],
                            {
                                "status": "complete",
                                "stored_stems": sorted(
                                    str(stem["stem_type"]) for stem in chunk_stems
                                ),
                                "verification_status": (
                                    sfx.get("automatic_status", "uncertain")
                                    if sfx
                                    else "uncertain"
                                ),
                                "metadata_s3_key": metadata_key,
                                "updated_at": utc_now(),
                            },
                        )
                        recovered.setdefault("sam_completed", 0)
                        recovered["sam_completed"] += 1
                    elif self._stale(item, now):
                        self._enqueue_sam(job_id, item)
                        recovered["sam"] += 1
                elif (
                    entity == "model_task"
                    and item.get("queue") == "flamingo"
                    and status in {"queued", "running"}
                    and self._stale(item, now)
                ):
                    task = QueueTask.model_validate(item["task"])
                    self.aws.update(
                        f"JOB#{job_id}",
                        item["SK"],
                        {"status": "queued", "updated_at": utc_now()},
                    )
                    self.aws.send_task(self.settings.flamingo_queue_url, task)
                    recovered.setdefault("flamingo", 0)
                    recovered["flamingo"] += 1
            refreshed_items = self.aws.query_partition(f"JOB#{job_id}")
            summary = _terminal_job_summary(refreshed_items)
            if summary:
                self.aws.update(f"JOB#{job_id}", "META", summary)
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
        presence_targets = chunk.get("presence_gate", {}).get("targets")
        task = QueueTask(
            task_id=uuid.uuid4().hex,
            task_type="separate_chunk",
            job_id=job_id,
            source_id=chunk["source_id"],
            chunk_id=chunk["chunk_id"],
            targets=presence_targets,
        )
        self.aws.update(
            f"JOB#{job_id}",
            chunk["SK"],
            {"status": "queued", "updated_at": task.created_at},
        )
        self.aws.send_task(self.settings.sam_queue_url, task)


class FlamingoHandler:
    PROMPTS = {
        "describe_scene": (
            "Analyze the complete audio. Return strict JSON with keys "
            "has_music (boolean), has_voices (boolean), music_description "
            "(string or null), and scene_description (one sentence). Do not "
            "use Markdown or Python literals."
        ),
        "describe_music": (
            "Describe only the music in this audio in one concise sentence, "
            "including genre, instruments, tempo, mood, and whether it is "
            "foreground or background."
        ),
        "transcribe_voice": (
            "Transcribe all audible speech. Use SPEAKER_1, SPEAKER_2, and so on "
            "for diarization when multiple speakers are distinguishable. Return "
            "only the speaker-labelled transcript, or NO_SPEECH if empty."
        ),
    }

    def __init__(self, settings: Settings, aws: PipelineAWS):
        self.settings = settings
        self.aws = aws
        self.client = AudioFlamingoClient(settings.flamingo_api_url)

    def handle(self, task: QueueTask) -> None:
        if task.task_type not in self.PROMPTS:
            raise ValueError(f"Unsupported Audio Flamingo task: {task.task_type}")
        task_sk = f"TASK#{task.task_id}"
        task_item = self.aws.get(f"JOB#{task.job_id}", task_sk)
        if task_item and task_item.get("status") == "complete":
            return
        if not task_item:
            self.aws.put_model_task(task, "flamingo")
        self.aws.update(
            f"JOB#{task.job_id}",
            task_sk,
            {"status": "running", "updated_at": utc_now()},
        )
        self.aws.update(
            f"JOB#{task.job_id}",
            "META",
            {"status": "annotating", "updated_at": utc_now()},
        )
        audio_key, target_sk = self._input(task)
        with tempfile.TemporaryDirectory(prefix="audio-flamingo-") as temporary:
            audio_path = Path(temporary) / "input.wav"
            self.aws.download_file(audio_key, audio_path)
            result = self.client.ask(audio_path, self.PROMPTS[task.task_type])
        output_key = (
            f"jobs/{task.job_id}/annotations/{task.source_id}/"
            f"{task.chunk_id or 'source'}/{task.task_type}.json"
        )
        output = {
            **result,
            "job_id": task.job_id,
            "source_id": task.source_id,
            "chunk_id": task.chunk_id,
            "task_type": task.task_type,
            "created_at": utc_now(),
        }
        self.aws.upload_json(output, output_key)
        self.aws.put(
            {
                "PK": f"JOB#{task.job_id}",
                "SK": (
                    f"ANNOTATION#{task.source_id}#"
                    f"{task.chunk_id or 'source'}#{task.task_type}"
                ),
                "entity": "annotation",
                **output,
                "s3_key": output_key,
            }
        )
        field = {
            "describe_scene": "scene_description",
            "describe_music": "music_description",
            "transcribe_voice": "voice_transcription",
        }[task.task_type]
        self.aws.update(
            f"JOB#{task.job_id}",
            target_sk,
            {
                field: result["text"],
                f"{field}_s3_key": output_key,
                "updated_at": utc_now(),
            },
        )
        if task.task_type == "describe_scene":
            self._schedule_separation(task, result)
        self.aws.update(
            f"JOB#{task.job_id}",
            task_sk,
            {
                "status": "complete",
                "output_s3_key": output_key,
                "updated_at": utc_now(),
            },
        )
        self._refresh_job(task.job_id)

    def _input(self, task: QueueTask) -> tuple[str, str]:
        if task.task_type == "describe_scene":
            target_sk = f"SOURCE#{task.source_id}"
        else:
            if not task.chunk_id:
                raise ValueError(f"{task.task_type} requires chunk_id")
            stem_type = "music" if task.task_type == "describe_music" else "voice"
            target_sk = f"STEM#{task.source_id}#{task.chunk_id}#{stem_type}"
        item = self.aws.get(f"JOB#{task.job_id}", target_sk)
        if not item or not item.get("s3_key"):
            raise KeyError(f"Audio input record is missing: {target_sk}")
        return item["s3_key"], target_sk

    def _schedule_separation(
        self, task: QueueTask, result: dict[str, Any]
    ) -> None:
        presence = _scene_presence(str(result.get("text") or ""))
        targets = [
            kind
            for kind, present in (
                ("music", presence["has_music"]),
                ("voice", presence["has_voices"]),
            )
            if present
        ]
        presence_gate = {
            **presence,
            "targets": targets,
            "model": result.get("model"),
            "policy": "source_scene_preflight_v1",
        }
        self.aws.update(
            f"JOB#{task.job_id}",
            f"SOURCE#{task.source_id}",
            {
                "status": "chunked",
                "semantic_presence": presence_gate,
                "updated_at": utc_now(),
            },
        )
        for chunk in self.aws.query_partition(f"JOB#{task.job_id}"):
            if (
                chunk.get("entity") != "chunk"
                or chunk.get("source_id") != task.source_id
                or chunk.get("status") not in {"waiting_scene", "failed"}
            ):
                continue
            separation_task = QueueTask(
                task_id=(
                    f"sam:{task.job_id}:{task.source_id}:{chunk['chunk_id']}"
                ),
                task_type="separate_chunk",
                job_id=task.job_id,
                source_id=task.source_id,
                chunk_id=str(chunk["chunk_id"]),
                targets=targets,
            )
            self.aws.update(
                f"JOB#{task.job_id}",
                str(chunk["SK"]),
                {
                    "status": "queued",
                    "presence_gate": presence_gate,
                    "updated_at": utc_now(),
                },
            )
            self.aws.send_task(self.settings.sam_queue_url, separation_task)

    def _refresh_job(self, job_id: str) -> None:
        items = self.aws.query_partition(f"JOB#{job_id}")
        summary = _terminal_job_summary(items)
        if summary and summary["status"] in {"complete", "partial"}:
            self.aws.update(f"JOB#{job_id}", "META", summary)
