"""HTTP control plane and reviewer API."""

from __future__ import annotations

import mimetypes
import re
import statistics
import uuid
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse

from .aws import PipelineAWS
from .config import Settings
from .schema import (
    DatasetCreateRequest,
    DatasetResponse,
    JobCreateRequest,
    JobCreateResponse,
    ProcessingStatus,
    QueueTask,
    ReviewItem,
    ReviewRequest,
    SourceUpload,
    UploadCompleteRequest,
)


def _safe_filename(value: str) -> str:
    basename = Path(value).name.strip()
    safe = re.sub(r"[^A-Za-z0-9._ -]+", "_", basename).strip(" .")
    if not safe:
        raise HTTPException(status_code=422, detail="Filename cannot be empty")
    return safe[:180]


def _review_item(aws: PipelineAWS, item: dict[str, Any]) -> ReviewItem:
    return ReviewItem(
        review_id=item["review_id"],
        job_id=item["job_id"],
        source_id=item["source_id"],
        chunk_id=item["chunk_id"],
        stem_type=item["stem_type"],
        prompt=item["prompt"],
        assertion=item["assertion"],
        automatic_status=item["automatic_status"],
        effective_status=item["effective_status"],
        audio_url=aws.presign_download(item["s3_key"]),
        scores=item.get("scores", {}),
        timings_ms=item.get("timings_ms", {}),
    )


def _review_counts(aws: PipelineAWS) -> dict[str, int]:
    failure = aws.count_index("REVIEW#failure")
    uncertain = aws.count_index("REVIEW#uncertain")
    return {
        "failure": failure,
        "uncertain": uncertain,
        "total": failure + uncertain,
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _similarity_summary(scores: list[float]) -> dict[str, Any]:
    return {
        "count": len(scores),
        "mean": round(statistics.fmean(scores), 4) if scores else None,
        "median": round(statistics.median(scores), 4) if scores else None,
        "minimum": round(min(scores), 4) if scores else None,
        "maximum": round(max(scores), 4) if scores else None,
        "p10": round(value, 4)
        if (value := _percentile(scores, 0.1)) is not None
        else None,
        "p90": round(value, 4)
        if (value := _percentile(scores, 0.9)) is not None
        else None,
    }


def _dataset_snapshot(aws: PipelineAWS, dataset_id: str) -> dict[str, Any]:
    dataset = aws.get(f"DATASET#{dataset_id}", "META")
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    jobs = [
        job
        for job in aws.query_index("JOBS", limit=None, newest_first=True)
        if job.get("dataset_id") == dataset_id
    ]
    all_sources: list[dict[str, Any]] = []
    all_chunks: list[dict[str, Any]] = []
    all_stems: list[dict[str, Any]] = []
    job_summaries: list[dict[str, Any]] = []
    for job in jobs:
        items = aws.query_partition(f"JOB#{job['job_id']}")
        sources = [item for item in items if item.get("entity") == "source"]
        chunks = [item for item in items if item.get("entity") == "chunk"]
        stems = [item for item in items if item.get("entity") == "stem"]
        for chunk in chunks:
            if not chunk.get("bytes") and chunk.get("s3_key"):
                chunk["bytes"] = aws.object_size(str(chunk["s3_key"]))
        for stem in stems:
            if not stem.get("bytes") and stem.get("s3_key"):
                stem["bytes"] = aws.object_size(str(stem["s3_key"]))
            if stem.get("stereo_s3_key") and not stem.get("stereo_bytes"):
                stem["stereo_bytes"] = aws.object_size(
                    str(stem["stereo_s3_key"])
                )
        chunks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        stems_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in chunks:
            chunks_by_source[str(chunk["source_id"])].append(chunk)
        for stem in stems:
            stems_by_source[str(stem["source_id"])].append(stem)
        for source in sources:
            source_id = str(source["source_id"])
            source_chunks = chunks_by_source[source_id]
            source_stems = stems_by_source[source_id]
            source_bytes = int(source.get("bytes") or 0)
            if not source_bytes and source.get("s3_key"):
                source_bytes = aws.object_size(str(source["s3_key"]))
            duration = float(source.get("duration_seconds") or 0.0)
            if not duration and source_chunks:
                duration = max(
                    float(chunk.get("end_seconds") or 0.0) for chunk in source_chunks
                )
            route = next(
                (
                    stem.get("settings", {})
                    .get("adaptive_routing", {})
                    .get("selected_order")
                    for stem in source_stems
                    if stem.get("settings", {}).get("adaptive_routing")
                ),
                None,
            )
            chunk_reconstruction_scores = [
                float(score)
                for chunk in source_chunks
                if (
                    score := chunk.get("reconstruction", {})
                    .get("metrics", {})
                    .get("similarity_score")
                )
                is not None
            ]
            source_similarity = (
                source.get("reconstruction", {})
                .get("metrics", {})
                .get("similarity_score")
            )
            all_sources.append(
                {
                    **source,
                    "bytes": source_bytes,
                    "duration_seconds": duration,
                    "pipeline_status": (
                        "complete"
                        if source_chunks
                        and all(
                            chunk.get("status") in {"complete", "failed", "skipped"}
                            for chunk in source_chunks
                        )
                        else source.get("status")
                    ),
                    "chunk_count": len(source_chunks),
                    "audible_chunk_count": sum(
                        chunk.get("status") != "skipped" for chunk in source_chunks
                    ),
                    "stem_count": len(source_stems),
                    "stem_statuses": dict(
                        Counter(
                            str(stem.get("effective_status")) for stem in source_stems
                        )
                    ),
                    "selected_order": route,
                    "reconstruction_count": (
                        1 if source_similarity is not None else 0
                    ),
                    "reconstructed_chunk_count": len(
                        chunk_reconstruction_scores
                    ),
                    "similarity_score": (
                        round(float(source_similarity), 4)
                        if source_similarity is not None
                        else None
                    ),
                    "similarity_minimum": (
                        round(min(chunk_reconstruction_scores), 4)
                        if chunk_reconstruction_scores
                        else None
                    ),
                }
            )
        job_summaries.append(
            {
                **job,
                "chunk_count": len(chunks),
                "stem_count": len(stems),
                "review_remaining": sum(
                    stem.get("effective_status") in {"failure", "uncertain"}
                    for stem in stems
                ),
            }
        )
        all_chunks.extend(chunks)
        all_stems.extend(stems)
    input_bytes = sum(int(source.get("bytes") or 0) for source in all_sources)
    chunk_bytes = sum(int(chunk.get("bytes") or 0) for chunk in all_chunks)
    stem_bytes = sum(int(stem.get("bytes") or 0) for stem in all_stems)
    stereo_bytes = sum(int(stem.get("stereo_bytes") or 0) for stem in all_stems)
    chunk_reconstruction_bytes = sum(
        int(chunk.get("reconstruction", {}).get("bytes") or 0)
        for chunk in all_chunks
    )
    source_reconstruction_bytes = sum(
        int(source.get("reconstruction", {}).get("bytes") or 0)
        for source in all_sources
    )
    reconstruction_bytes = (
        chunk_reconstruction_bytes + source_reconstruction_bytes
    )
    reconstructions = []
    for source in all_sources:
        reconstruction = source.get("reconstruction", {})
        metrics = reconstruction.get("metrics", {})
        score = metrics.get("similarity_score")
        if score is None:
            continue
        reconstructions.append(
            {
                "job_id": source.get("job_id"),
                "source_id": source.get("source_id"),
                "filename": source.get("filename"),
                "chunk_count": source.get("chunk_count"),
                "duration_seconds": source.get("duration_seconds"),
                "similarity_score": float(score),
                "waveform_correlation": metrics.get("waveform_correlation"),
                "level_delta_db": metrics.get("level_delta_db"),
                "snr_db": metrics.get("snr_db"),
                "selected_order": source.get("selected_order"),
            }
        )
    similarity_scores = [
        float(item["similarity_score"]) for item in reconstructions
    ]
    review_remaining = sum(
        stem.get("effective_status") in {"failure", "uncertain"} for stem in all_stems
    )
    return {
        "dataset": dataset,
        "summary": {
            "jobs": len(jobs),
            "sources": len(all_sources),
            "non_stereo_sources": sum(
                source.get("skip_reason") == "non_stereo_input"
                for source in all_sources
            ),
            "duration_seconds": sum(
                float(source.get("duration_seconds") or 0.0) for source in all_sources
            ),
            "input_bytes": input_bytes,
            "chunk_bytes": chunk_bytes,
            "stem_bytes": stem_bytes,
            "stereo_bytes": stereo_bytes,
            "reconstruction_bytes": reconstruction_bytes,
            "chunk_reconstruction_bytes": chunk_reconstruction_bytes,
            "source_reconstruction_bytes": source_reconstruction_bytes,
            "total_bytes": (
                input_bytes
                + chunk_bytes
                + stem_bytes
                + stereo_bytes
                + reconstruction_bytes
            ),
            "chunks": len(all_chunks),
            "audible_chunks": sum(
                chunk.get("status") != "skipped" for chunk in all_chunks
            ),
            "skipped_chunks": sum(
                chunk.get("status") == "skipped" for chunk in all_chunks
            ),
            "stems": len(all_stems),
            "stereo_stems": sum(bool(stem.get("stereo_s3_key")) for stem in all_stems),
            "reconstructed_chunks": sum(
                bool(chunk.get("reconstruction")) for chunk in all_chunks
            ),
            "reconstructed_sources": len(reconstructions),
            "similarity": _similarity_summary(similarity_scores),
            "stems_by_type": dict(
                Counter(str(stem.get("stem_type")) for stem in all_stems)
            ),
            "stems_by_status": dict(
                Counter(str(stem.get("effective_status")) for stem in all_stems)
            ),
            "selected_routes": dict(
                Counter(
                    str(source.get("selected_order"))
                    for source in all_sources
                    if source.get("selected_order")
                )
            ),
            "job_statuses": dict(Counter(str(job.get("status")) for job in jobs)),
            "review_remaining": review_remaining,
        },
        "jobs": job_summaries,
        "sources": sorted(
            all_sources,
            key=lambda source: str(source.get("created_at", "")),
            reverse=True,
        ),
        "reconstructions": sorted(
            reconstructions,
            key=lambda item: float(item["similarity_score"]),
        ),
    }


def create_app(
    settings: Settings | None = None, aws: PipelineAWS | None = None
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_settings = settings or Settings.from_env()
        app.state.settings = resolved_settings
        app.state.aws = aws or PipelineAWS(resolved_settings)
        yield

    app = FastAPI(
        title="SAM Audio Pipeline",
        version="0.1.0",
        lifespan=lifespan,
    )

    def backend(request: Request) -> PipelineAWS:
        return request.app.state.aws

    @app.get("/healthz")
    def health(request: Request) -> dict[str, Any]:
        config: Settings = request.app.state.settings
        return {
            "status": "ready",
            "region": config.aws_region,
            "bucket": config.bucket,
            "table": config.table_name,
            "chunk_seconds": config.chunk_seconds,
            "overlap_seconds": config.overlap_seconds,
            "cascade_order": ["music", "voice"],
        }

    @app.get("/", response_class=HTMLResponse)
    def panel() -> str:
        return (Path(__file__).parent / "web" / "index.html").read_text()

    @app.get("/review", response_class=HTMLResponse)
    @app.get("/data", response_class=HTMLResponse)
    @app.get("/data/{dataset_id}", response_class=HTMLResponse)
    @app.get("/data/{dataset_id}/jobs/{job_id}", response_class=HTMLResponse)
    @app.get(
        "/data/{dataset_id}/jobs/{job_id}/sources/{source_id}",
        response_class=HTMLResponse,
    )
    def panel_route(
        dataset_id: str | None = None,
        job_id: str | None = None,
        source_id: str | None = None,
    ) -> str:
        return (Path(__file__).parent / "web" / "index.html").read_text()

    @app.get("/v1/overview")
    def overview(request: Request) -> dict[str, Any]:
        store = backend(request)
        return {
            "jobs": store.query_index("JOBS", limit=100, newest_first=True),
            "review_queue": _review_counts(store),
            "queues": {
                "ingest": store.queue_metrics(store.settings.ingest_queue_url),
                "sam": store.queue_metrics(store.settings.sam_queue_url),
                "audio_flamingo": store.queue_metrics(
                    store.settings.flamingo_queue_url
                ),
            },
        }

    @app.post("/v1/jobs", response_model=JobCreateResponse, status_code=201)
    def create_job(payload: JobCreateRequest, request: Request) -> JobCreateResponse:
        store = backend(request)
        dataset_id = re.sub(r"[^A-Za-z0-9_-]+", "-", payload.dataset_id).strip("-")
        if not dataset_id:
            raise HTTPException(status_code=422, detail="Invalid dataset id")
        dataset = store.get(f"DATASET#{dataset_id}", "META")
        if not dataset:
            if dataset_id != "default":
                raise HTTPException(status_code=404, detail="Dataset not found")
            store.ensure_dataset("default", "Default dataset")
        job_id = uuid.uuid4().hex
        store.create_job(job_id, dataset_id, len(payload.filenames))
        uploads: list[SourceUpload] = []
        for submitted_filename in payload.filenames:
            filename = _safe_filename(submitted_filename)
            source_id = uuid.uuid4().hex
            key = f"jobs/{job_id}/sources/{source_id}/{filename}"
            content_type = (
                mimetypes.guess_type(filename)[0] or "application/octet-stream"
            )
            store.create_source(
                job_id,
                source_id,
                filename,
                key,
                payload.source_metadata.get(submitted_filename, {}),
            )
            uploads.append(
                SourceUpload(
                    source_id=source_id,
                    filename=filename,
                    content_type=content_type,
                    s3_key=key,
                    upload_url=store.presign_upload(key, content_type),
                )
            )
        return JobCreateResponse(
            job_id=job_id,
            dataset_id=dataset_id,
            status=ProcessingStatus.UPLOADING,
            uploads=uploads,
        )

    @app.post("/v1/datasets", response_model=DatasetResponse, status_code=201)
    def create_dataset(
        payload: DatasetCreateRequest, request: Request
    ) -> DatasetResponse:
        dataset_id = uuid.uuid4().hex
        item = backend(request).ensure_dataset(dataset_id, payload.name.strip())
        return DatasetResponse.model_validate(item)

    @app.get("/v1/datasets", response_model=list[DatasetResponse])
    def list_datasets(request: Request) -> list[DatasetResponse]:
        store = backend(request)
        store.ensure_dataset("default", "Default dataset")
        return [
            DatasetResponse.model_validate(item)
            for item in store.query_index("DATASETS", limit=1000)
        ]

    @app.get("/v1/datasets/{dataset_id}/overview")
    def dataset_overview(dataset_id: str, request: Request) -> dict[str, Any]:
        return _dataset_snapshot(backend(request), dataset_id)

    @app.post("/v1/jobs/{job_id}/uploads-complete", status_code=202)
    def uploads_complete(
        job_id: str, payload: UploadCompleteRequest, request: Request
    ) -> dict[str, Any]:
        store = backend(request)
        job = store.get(f"JOB#{job_id}", "META")
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        sources = [
            item
            for item in store.query_partition(f"JOB#{job_id}")
            if item.get("entity") == "source"
        ]
        requested = set(payload.source_ids or [item["source_id"] for item in sources])
        known = {item["source_id"] for item in sources}
        if unknown := requested - known:
            raise HTTPException(
                status_code=422,
                detail={"unknown_source_ids": sorted(unknown)},
            )
        queued: list[str] = []
        for source in sources:
            source_id = source["source_id"]
            if source_id not in requested or source.get("status") != "uploading":
                continue
            if not store.object_exists(source["s3_key"]):
                raise HTTPException(
                    status_code=409,
                    detail=f"Upload is missing for source {source_id}",
                )
            task = QueueTask(
                task_id=uuid.uuid4().hex,
                task_type="ingest_source",
                job_id=job_id,
                source_id=source_id,
            )
            store.update(
                f"JOB#{job_id}",
                f"SOURCE#{source_id}",
                {"status": "queued", "updated_at": task.created_at},
            )
            store.send_task(store.settings.ingest_queue_url, task)
            queued.append(source_id)
        store.update(
            f"JOB#{job_id}",
            "META",
            {
                "status": "queued",
                "updated_at": task.created_at if queued else job["updated_at"],
            },
        )
        return {"job_id": job_id, "queued_source_ids": queued}

    @app.get("/v1/jobs")
    def list_jobs(request: Request, limit: int = 50) -> list[dict[str, Any]]:
        if not 1 <= limit <= 200:
            raise HTTPException(
                status_code=422, detail="limit must be between 1 and 200"
            )
        return backend(request).query_index("JOBS", limit=limit, newest_first=True)

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str, request: Request) -> dict[str, Any]:
        items = backend(request).query_partition(f"JOB#{job_id}")
        if not items:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "job": next(item for item in items if item["SK"] == "META"),
            "sources": [item for item in items if item.get("entity") == "source"],
            "chunks": [item for item in items if item.get("entity") == "chunk"],
            "stems": [item for item in items if item.get("entity") == "stem"],
            "stem_results": [
                item for item in items if item.get("entity") == "stem_result"
            ],
            "annotations": [
                item for item in items if item.get("entity") == "annotation"
            ],
            "model_tasks": [
                item for item in items if item.get("entity") == "model_task"
            ],
        }

    @app.get("/v1/jobs/{job_id}/sources/{source_id}")
    def get_source(job_id: str, source_id: str, request: Request) -> dict[str, Any]:
        store = backend(request)
        items = store.query_partition(f"JOB#{job_id}")
        source = next(
            (
                item
                for item in items
                if item.get("entity") == "source" and item.get("source_id") == source_id
            ),
            None,
        )
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        job = next(item for item in items if item["SK"] == "META")
        source_reconstruction = source.get("reconstruction", {})
        source = {
            **source,
            "audio_url": (
                store.presign_download(source["s3_key"])
                if source.get("s3_key") and source.get("status") != "failed"
                else None
            ),
            "joined_audio_url": (
                store.presign_download(source_reconstruction["s3_key"])
                if source_reconstruction.get("s3_key")
                else None
            ),
        }
        chunks = sorted(
            [
                item
                for item in items
                if item.get("entity") == "chunk" and item.get("source_id") == source_id
            ],
            key=lambda chunk: float(chunk.get("start_seconds") or 0.0),
        )
        source_duration = float(source.get("duration_seconds") or 0.0)
        if not source_duration and chunks:
            source_duration = max(
                float(chunk.get("end_seconds") or 0.0) for chunk in chunks
            )
        source["bytes"] = int(source.get("bytes") or 0) or store.object_size(
            str(source["s3_key"])
        )
        source["duration_seconds"] = source_duration
        source["pipeline_status"] = (
            "complete"
            if chunks
            and all(
                chunk.get("status") in {"complete", "failed", "skipped"}
                for chunk in chunks
            )
            else source.get("status")
        )
        stems = [
            item
            for item in items
            if item.get("entity") == "stem" and item.get("source_id") == source_id
        ]
        stem_results = [
            item
            for item in items
            if item.get("entity") == "stem_result"
            and item.get("source_id") == source_id
        ]
        stems_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
        omitted_by_chunk: dict[str, list[str]] = defaultdict(list)
        for stem in stems:
            stems_by_chunk[str(stem["chunk_id"])].append(
                {
                    **stem,
                    "audio_url": store.presign_download(stem["s3_key"]),
                    "stereo_audio_url": (
                        store.presign_download(stem["stereo_s3_key"])
                        if stem.get("stereo_s3_key")
                        else None
                    ),
                }
            )
        for result in stem_results:
            if result.get("status") == "skipped":
                omitted_by_chunk[str(result["chunk_id"])].append(
                    str(result["stem_type"])
                )
        expanded_chunks = []
        for chunk in chunks:
            chunk_id = str(chunk["chunk_id"])
            reconstruction = chunk.get("reconstruction", {})
            expanded_chunks.append(
                {
                    **chunk,
                    "audio_url": (
                        store.presign_download(chunk["s3_key"])
                        if chunk.get("s3_key")
                        else None
                    ),
                    "stems": sorted(
                        stems_by_chunk[chunk_id],
                        key=lambda stem: {"music": 0, "voice": 1, "sfx": 2}.get(
                            str(stem.get("stem_type")), 9
                        ),
                    ),
                    "omitted_stems": sorted(omitted_by_chunk[chunk_id]),
                    "joined_audio_url": (
                        store.presign_download(reconstruction["s3_key"])
                        if reconstruction.get("s3_key")
                        else None
                    ),
                }
            )
        annotations = [
            item
            for item in items
            if item.get("entity") == "annotation" and item.get("source_id") == source_id
        ]
        model_tasks = [
            item
            for item in items
            if item.get("entity") == "model_task" and item.get("source_id") == source_id
        ]
        return {
            "job": job,
            "source": source,
            "chunks": expanded_chunks,
            "annotations": annotations,
            "model_tasks": model_tasks,
        }

    @app.get("/v1/review/stats")
    def review_stats(request: Request) -> dict[str, int]:
        return _review_counts(backend(request))

    @app.get("/v1/review/next", response_model=ReviewItem | None)
    def next_review(request: Request) -> ReviewItem | None:
        store = backend(request)
        candidates = [
            *store.query_index("REVIEW#failure", limit=10),
            *store.query_index("REVIEW#uncertain", limit=10),
        ]
        if not candidates:
            return None
        item = min(candidates, key=lambda candidate: candidate["GSI1SK"])
        return _review_item(store, item)

    @app.post("/v1/review/{review_id}")
    def review(
        review_id: str,
        payload: ReviewRequest,
        request: Request,
        reviewer: Annotated[str, Header(alias="X-Reviewer")] = "local-reviewer",
    ) -> dict[str, Any]:
        parts = review_id.split(":")
        if len(parts) != 4:
            raise HTTPException(status_code=422, detail="Invalid review id")
        job_id, source_id, chunk_id, stem_type = parts
        store = backend(request)
        stem = store.get(
            f"JOB#{job_id}",
            f"STEM#{source_id}#{chunk_id}#{stem_type}",
        )
        if not stem:
            raise HTTPException(status_code=404, detail="Review item not found")
        store.record_review(stem, payload.decision, payload.note, reviewer)
        return {
            "review_id": review_id,
            "decision": payload.decision,
            "reviewer": reviewer,
        }

    return app


app = create_app()


def main() -> None:
    uvicorn.run(
        "sam_audio_pipeline.api:app",
        host="127.0.0.1",
        port=8080,
        workers=1,
    )


if __name__ == "__main__":
    main()
