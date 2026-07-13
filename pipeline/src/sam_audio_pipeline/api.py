"""HTTP control plane and reviewer API."""

from __future__ import annotations

import mimetypes
import re
import uuid
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

    @app.get("/v1/overview")
    def overview(request: Request) -> dict[str, Any]:
        store = backend(request)
        return {
            "jobs": store.query_index("JOBS", limit=100, newest_first=True),
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
            store.create_source(job_id, source_id, filename, key)
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
            "annotations": [
                item for item in items if item.get("entity") == "annotation"
            ],
            "model_tasks": [
                item for item in items if item.get("entity") == "model_task"
            ],
        }

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
