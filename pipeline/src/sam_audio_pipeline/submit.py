"""Submit an acquired AudioSet manifest to a durable pipeline dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def successful_sources(manifest_path: Path) -> list[tuple[Path, dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    sources = []
    for item in manifest:
        if item.get("retrieval_status") != "success":
            continue
        path = manifest_path.parent / item["local_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append((path, item))
    return sources


def submit(
    api_url: str,
    manifest_path: Path,
    *,
    dataset_name: str,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    sources = successful_sources(manifest_path)
    if not sources:
        raise RuntimeError("Manifest contains no successfully acquired audio")
    base_url = api_url.rstrip("/")
    with httpx.Client(timeout=None) as client:
        if dataset_id is None:
            response = client.post(
                f"{base_url}/v1/datasets", json={"name": dataset_name}
            )
            response.raise_for_status()
            dataset_id = response.json()["dataset_id"]
        filenames = [path.name for path, _ in sources]
        source_metadata = {
            path.name: {
                key: value
                for key, value in item.items()
                if key
                in {
                    "video_id",
                    "start_seconds",
                    "end_seconds",
                    "labels",
                    "label_names",
                    "dataset_csv",
                    "ontology_url",
                    "ontology_sha256",
                    "source_url",
                    "selection",
                    "selection_seed",
                    "selection_index",
                    "random_sound_index",
                    "sha256",
                    "bytes",
                    "retrieved_duration_seconds",
                    "audioset_metadata_license",
                    "audioset_ontology_license",
                    "source_audio_rights",
                }
            }
            for path, item in sources
        }
        response = client.post(
            f"{base_url}/v1/jobs",
            json={
                "dataset_id": dataset_id,
                "filenames": filenames,
                "source_metadata": source_metadata,
            },
        )
        response.raise_for_status()
        job = response.json()
        by_filename = {path.name: path for path, _ in sources}
        for index, upload in enumerate(job["uploads"], start=1):
            path = by_filename[upload["filename"]]
            logger.info("Uploading %d/%d %s", index, len(sources), path.name)
            with path.open("rb") as audio:
                uploaded = client.put(
                    upload["upload_url"],
                    headers={"Content-Type": upload["content_type"]},
                    content=audio,
                )
            uploaded.raise_for_status()
        response = client.post(
            f"{base_url}/v1/jobs/{job['job_id']}/uploads-complete", json={}
        )
        response.raise_for_status()
        return {
            "dataset_id": dataset_id,
            "job_id": job["job_id"],
            "source_count": len(sources),
            "queued_source_ids": response.json()["queued_source_ids"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:18080")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-name", default="AudioSet random validation")
    parser.add_argument("--dataset-id")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    result = submit(
        args.api,
        args.manifest,
        dataset_name=args.dataset_name,
        dataset_id=args.dataset_id,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
