"""Acquire a private, reproducible AudioSet calibration/reference set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import sys
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import boto3

from .audio import sha256_file

logger = logging.getLogger(__name__)

CSV_URLS = (
    "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
    "eval_segments.csv",
    "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
    "balanced_train_segments.csv",
)
ONTOLOGY_URL = (
    "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"
)

# Parent categories plus representative non-music/non-speech events.
REFERENCE_CLASSES = {
    "music": "/m/04rlf",
    "speech": "/m/09x0r",
    "silence": "/m/028v0c",
    "dog": "/m/0bt9lr",
    "engine": "/m/02mk9",
    "door": "/m/02dgv",
    "explosion": "/m/014zdl",
    "water": "/m/0838f",
}


def _download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "sam-audio-pipeline"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        path.write_bytes(response.read())


def _segments(path: Path):
    with path.open(newline="", encoding="utf-8") as source:
        rows = (line for line in source if not line.startswith("#"))
        for video_id, start, end, labels in csv.reader(rows, skipinitialspace=True):
            yield {
                "video_id": video_id,
                "start_seconds": float(start),
                "end_seconds": float(end),
                "labels": [label.strip('"') for label in labels.split(",")],
            }


def select_manifest(cache_dir: Path, per_class: int) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ontology_path = cache_dir / "ontology.json"
    if not ontology_path.exists():
        _download(ONTOLOGY_URL, ontology_path)
    ontology_sha256 = sha256_file(ontology_path)
    ontology = json.loads(ontology_path.read_text())
    names = {item["id"]: item["name"] for item in ontology}
    children = {item["id"]: item.get("child_ids", []) for item in ontology}

    def descendants(mid: str) -> set[str]:
        found = {mid}
        pending = [mid]
        while pending:
            child_ids = children.get(pending.pop(), [])
            pending.extend(child for child in child_ids if child not in found)
            found.update(child_ids)
        return found

    families = {name: descendants(mid) for name, mid in REFERENCE_CLASSES.items()}
    music_or_speech = families["music"] | families["speech"]
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_video_ids: set[str] = set()
    for url in CSV_URLS:
        csv_path = cache_dir / url.rsplit("/", 1)[-1]
        if not csv_path.exists():
            _download(url, csv_path)
        for segment in _segments(csv_path):
            for reference_name, mid in REFERENCE_CLASSES.items():
                if len(selected[reference_name]) >= per_class:
                    continue
                if segment["video_id"] in selected_video_ids:
                    continue
                matched_labels = sorted(
                    families[reference_name] & set(segment["labels"])
                )
                if not matched_labels:
                    continue
                excluded = music_or_speech - families[reference_name]
                if excluded & set(segment["labels"]):
                    continue
                selected[reference_name].append(
                    {
                        **segment,
                        "reference_class": reference_name,
                        "reference_mid": mid,
                        "reference_label": names.get(mid, reference_name),
                        "matched_labels": matched_labels,
                        "dataset_csv": url,
                        "ontology_url": ONTOLOGY_URL,
                        "ontology_sha256": ontology_sha256,
                        "source_url": (
                            f"https://www.youtube.com/watch?v={segment['video_id']}"
                        ),
                        "audioset_metadata_license": "CC BY 4.0",
                        "audioset_ontology_license": "CC BY-SA 4.0",
                        "source_audio_rights": (
                            "Underlying media remains subject to its source terms."
                        ),
                    }
                )
                selected_video_ids.add(segment["video_id"])
        if all(len(items) >= per_class for items in selected.values()):
            break
    manifest = [
        item
        for reference_name in REFERENCE_CLASSES
        for item in selected[reference_name][:per_class]
    ]
    missing = [name for name in REFERENCE_CLASSES if len(selected[name]) < per_class]
    if missing:
        raise RuntimeError(f"AudioSet did not yield enough references for: {missing}")
    return manifest


def _acquire(item: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    section = f"*{item['start_seconds']}-{item['end_seconds']}"
    with tempfile.TemporaryDirectory(prefix="audioset-") as temporary:
        template = str(Path(temporary) / "source.%(ext)s")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "--quiet",
                "--no-warnings",
                "--no-playlist",
                "--download-sections",
                section,
                "--force-keyframes-at-cuts",
                "-x",
                "--audio-format",
                "wav",
                "-o",
                template,
                item["source_url"],
            ],
            check=True,
        )
        downloaded = Path(temporary) / "source.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(downloaded),
                "-acodec",
                "pcm_s16le",
                "-ar",
                "48000",
                str(output),
            ],
            check=True,
        )


def acquire(output_dir: Path, per_class: int) -> Path:
    cache = output_dir / "metadata"
    candidates = select_manifest(cache, per_class * 5)
    manifest: list[dict[str, Any]] = []
    successes: dict[str, int] = defaultdict(int)
    for item in candidates:
        name = item["reference_class"]
        if successes[name] >= per_class:
            continue
        video_id = item["video_id"]
        filename = f"{name}/{video_id}_{int(item['start_seconds'])}.wav"
        destination = output_dir / "audio" / filename
        item["local_path"] = str(destination.relative_to(output_dir))
        try:
            _acquire(item, destination)
            item["retrieval_status"] = "success"
            item["sha256"] = sha256_file(destination)
            item["bytes"] = destination.stat().st_size
            successes[name] += 1
        except (OSError, subprocess.CalledProcessError) as error:
            logger.warning("Could not retrieve %s: %s", item["source_url"], error)
            item["retrieval_status"] = "unavailable"
            item["error"] = type(error).__name__
        manifest.append(item)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (output_dir / "manifest.sha256").write_text(f"{digest}  manifest.json\n")
    missing = [name for name in REFERENCE_CLASSES if successes[name] < per_class]
    if missing:
        raise RuntimeError(f"Could not acquire AudioSet references for: {missing}")
    return manifest_path


def upload_reference_set(
    output_dir: Path,
    manifest_path: Path,
    *,
    bucket: str,
    prefix: str,
    region: str,
) -> None:
    s3 = boto3.client("s3", region_name=region)
    manifest = json.loads(manifest_path.read_text())
    normalized_prefix = prefix.strip("/")
    for item in manifest:
        if item.get("retrieval_status") != "success":
            continue
        local_path = output_dir / item["local_path"]
        key = f"{normalized_prefix}/{item['local_path']}"
        s3.upload_file(
            str(local_path),
            bucket,
            key,
            ExtraArgs={"ContentType": "audio/wav"},
        )
        item["s3_bucket"] = bucket
        item["s3_key"] = key
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    digest_path = output_dir / "manifest.sha256"
    digest_path.write_text(f"{digest}  manifest.json\n")
    s3.upload_file(
        str(manifest_path),
        bucket,
        f"{normalized_prefix}/manifest.json",
        ExtraArgs={"ContentType": "application/json"},
    )
    s3.upload_file(
        str(digest_path),
        bucket,
        f"{normalized_prefix}/manifest.sha256",
        ExtraArgs={"ContentType": "text/plain"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-prefix", default="references/audioset")
    parser.add_argument("--aws-region", default="us-east-1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.per_class < 1:
        parser.error("--per-class must be at least one")
    if args.metadata_only:
        manifest = select_manifest(args.output / "metadata", args.per_class)
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    else:
        path = acquire(args.output, args.per_class)
    if args.s3_bucket:
        if args.metadata_only:
            parser.error("--s3-bucket requires audio acquisition")
        upload_reference_set(
            args.output,
            path,
            bucket=args.s3_bucket,
            prefix=args.s3_prefix,
            region=args.aws_region,
        )
    print(path)


if __name__ == "__main__":
    main()
