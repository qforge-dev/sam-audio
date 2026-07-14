"""Build a resumable, quality-gated dataset from general YouTube search."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .audio import sha256_file

logger = logging.getLogger(__name__)

CLIP_SECONDS = 10.0
OUTPUT_SAMPLE_RATE = 48_000
MIN_SOURCE_SAMPLE_RATE = 44_100
MIN_SOURCE_BITRATE_KBPS = 120.0
SILENCE_THRESHOLD_DBFS = -55.0
MIN_RMS_DBFS = -35.0
MIN_PEAK_DBFS = -20.0
MAX_SILENT_FRACTION = 0.10
MAX_SILENT_RUN_SECONDS = 0.75
MIN_SIDE_TO_TOTAL_DB = -45.0
MAX_CLIPPED_FRACTION = 0.01

SCENES = (
    "street market",
    "busy cafe",
    "city festival",
    "family kitchen",
    "travel day",
    "train station",
    "airport terminal",
    "sports stadium",
    "outdoor fair",
    "restaurant service",
    "workshop project",
    "school event",
    "wedding reception",
    "museum visit",
    "city park",
    "shopping mall",
    "harbor tour",
    "road trip",
    "concert venue",
    "gaming session",
    "dance rehearsal",
    "community event",
    "food festival",
    "behind the scenes",
    "live performance",
    "theme park",
    "street food",
    "public transport",
    "home renovation",
    "sports practice",
)

FORMATS = (
    "vlog",
    "documentary",
    "walkthrough",
    "day in the life",
    "highlights",
    "interview",
    "event coverage",
    "travel diary",
    "gameplay commentary",
    "behind the scenes",
    "live report",
    "experience",
    "tour",
    "review",
    "making of",
)

ACTIVITIES = (
    "people talking",
    "crowd reactions",
    "host commentary",
    "friends conversation",
    "live demonstration",
    "public announcement",
    "audience interaction",
    "team discussion",
    "guided tour",
    "narrated experience",
)

AUDIO_HINTS = (
    "background music natural sounds",
    "music dialogue ambient sound",
    "talking soundtrack sound effects",
    "conversation music crowd noise",
    "commentary soundtrack environmental audio",
    "voices music real world sounds",
    "dialogue background score live sound",
    "speech music activity sounds",
)

EXCLUDED_TITLE_TERMS = (
    "ambient sound",
    "asmr",
    "background music",
    "concert live",
    "full album",
    "karaoke",
    "live at ",
    "live from ",
    "lyric video",
    "lyrics",
    "meditation",
    "music mix",
    "nature sounds",
    "no talking",
    "official audio",
    "official music video",
    "official video",
    "playlist",
    "relaxing music",
    "sleep music",
    "soundscape",
    "study music",
    "urban noise",
    "white noise",
)

SEARCH_EXCLUSIONS = (
    "-official -lyrics -karaoke -playlist -album -mix -ambience -soundscape"
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _run(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def build_queries(seed: int, count: int) -> list[str]:
    """Create reproducible, mix-biased general YouTube search queries."""
    generator = random.Random(seed)
    queries: list[str] = []
    seen: set[str] = set()
    while len(queries) < count:
        query = " ".join(
            (
                generator.choice(SCENES),
                generator.choice(FORMATS),
                generator.choice(ACTIVITIES),
                generator.choice(AUDIO_HINTS),
                str(generator.randint(2018, 2026)),
                SEARCH_EXCLUSIONS,
            )
        )
        if query in seen:
            continue
        seen.add(query)
        queries.append(query)
    return queries


def _candidate_allowed(item: dict[str, Any]) -> bool:
    duration = float(item.get("duration") or 0.0)
    title = str(item.get("title") or "").lower()
    return (
        bool(item.get("id"))
        and 30.0 <= duration <= 3600.0
        and item.get("live_status") not in {"is_live", "is_upcoming"}
        and not any(term in title for term in EXCLUDED_TITLE_TERMS)
    )


def _search(query: str, results: int) -> list[dict[str, Any]]:
    response = _run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-update",
            "--quiet",
            "--no-warnings",
            "--flat-playlist",
            "--dump-single-json",
            f"ytsearch{results}:{query}",
        ],
        timeout=75,
    )
    payload = json.loads(response.stdout)
    return [item for item in payload.get("entries", []) if _candidate_allowed(item)]


def discover_candidates(
    output_dir: Path,
    *,
    seed: int,
    query_count: int,
    results_per_query: int,
    workers: int,
    minimum_candidates: int,
) -> list[dict[str, Any]]:
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = metadata_dir / "candidates.json"
    if candidates_path.exists():
        existing = json.loads(candidates_path.read_text())
        if len(existing) >= minimum_candidates:
            logger.info("Reusing %d discovered candidates", len(existing))
            return existing

    queries = build_queries(seed, query_count)
    found: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(_search, query, results_per_query): query
            for query in queries
        }
        for index, future in enumerate(as_completed(pending), start=1):
            query = pending[future]
            try:
                items = future.result()
            except Exception as error:
                failures.append(
                    {"query": query, "error": f"{type(error).__name__}: {error}"}
                )
                continue
            for item in items:
                video_id = str(item["id"])
                if video_id in found:
                    continue
                duration = float(item["duration"])
                cut_generator = random.Random(f"{seed}:{video_id}")
                start = cut_generator.uniform(5.0, duration - CLIP_SECONDS - 5.0)
                found[video_id] = {
                    "video_id": video_id,
                    "source_url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": item.get("title"),
                    "duration_seconds": duration,
                    "uploader": item.get("uploader") or item.get("channel"),
                    "channel_id": item.get("channel_id"),
                    "view_count": item.get("view_count"),
                    "search_query": query,
                    "clip_start_seconds": round(start, 3),
                    "clip_end_seconds": round(start + CLIP_SECONDS, 3),
                    "selection": "seeded_general_youtube_search",
                    "selection_seed": seed,
                    "mixture_bias": ["voice", "music", "environmental_sfx"],
                    "source_audio_rights": (
                        "Underlying media remains subject to its source terms."
                    ),
                }
            if index % 25 == 0:
                logger.info(
                    "Searches %d/%d; %d unique candidates",
                    index,
                    len(queries),
                    len(found),
                )
    candidates = list(found.values())
    random.Random(seed).shuffle(candidates)
    candidates_path.write_text(json.dumps(candidates, indent=2) + "\n")
    (metadata_dir / "search.json").write_text(
        json.dumps(
            {
                "selection": "seeded_general_youtube_search",
                "seed": seed,
                "queries": queries,
                "results_per_query": results_per_query,
                "unique_candidates": len(candidates),
                "failures": failures,
                "created_at": _now(),
            },
            indent=2,
        )
        + "\n"
    )
    if len(candidates) < minimum_candidates:
        raise RuntimeError(
            f"Only discovered {len(candidates)} candidates; "
            f"need at least {minimum_candidates}"
        )
    return candidates


def _dbfs(amplitude: float) -> float:
    return max(-120.0, 20.0 * math.log10(max(amplitude, 1e-12)))


def analyze_wav(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        sample_width = source.getsampwidth()
        frame_count = source.getnframes()
        encoded = source.readframes(frame_count)
    if channels != 2 or sample_width != 2:
        raise ValueError("Quality analysis requires stereo PCM16 WAV")
    samples = np.frombuffer(encoded, dtype="<i2").astype(np.float64)
    samples = samples.reshape(-1, channels) / 32768.0
    rms = float(np.sqrt(np.mean(np.square(samples))))
    peak = float(np.max(np.abs(samples), initial=0.0))
    window = max(1, round(sample_rate * 0.05))
    usable = len(samples) // window * window
    framed = samples[:usable].reshape(-1, window, channels)
    frame_rms = np.sqrt(np.mean(np.square(framed), axis=(1, 2)))
    silent = frame_rms < 10 ** (SILENCE_THRESHOLD_DBFS / 20.0)
    longest = 0
    current = 0
    for value in silent:
        current = current + 1 if value else 0
        longest = max(longest, current)
    side = (samples[:, 0] - samples[:, 1]) / 2.0
    side_energy = float(np.mean(np.square(side)))
    total_energy = float(np.mean(np.square(samples)))
    left_std = float(np.std(samples[:, 0]))
    right_std = float(np.std(samples[:, 1]))
    if left_std <= 1e-12 or right_std <= 1e-12:
        channel_correlation = (
            1.0 if np.array_equal(samples[:, 0], samples[:, 1]) else 0.0
        )
    else:
        channel_correlation = float(
            np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
        )
    return {
        "channels": channels,
        "is_stereo": True,
        "sample_rate_hz": sample_rate,
        "bit_depth": sample_width * 8,
        "duration_seconds": round(frame_count / sample_rate, 6),
        "rms_dbfs": round(_dbfs(rms), 4),
        "peak_dbfs": round(_dbfs(peak), 4),
        "silent_fraction": round(float(np.mean(silent)), 6),
        "longest_silent_run_seconds": round(longest * window / sample_rate, 4),
        "side_to_total_db": round(
            10.0 * math.log10(max(side_energy, 1e-12) / max(total_energy, 1e-12)),
            4,
        ),
        "channel_correlation": round(channel_correlation, 6),
        "clipped_fraction": round(float(np.mean(np.abs(samples) >= 0.999)), 8),
    }


def quality_rejections(
    metrics: dict[str, Any], source_format: dict[str, Any]
) -> list[str]:
    reasons: list[str] = []
    if int(source_format.get("sample_rate_hz") or 0) < MIN_SOURCE_SAMPLE_RATE:
        reasons.append("source_sample_rate")
    if int(source_format.get("channels") or 0) != 2:
        reasons.append("source_not_stereo")
    if float(source_format.get("bitrate_kbps") or 0.0) < MIN_SOURCE_BITRATE_KBPS:
        reasons.append("source_bitrate")
    if abs(float(metrics["duration_seconds"]) - CLIP_SECONDS) > 0.02:
        reasons.append("duration")
    if float(metrics["rms_dbfs"]) < MIN_RMS_DBFS:
        reasons.append("low_rms")
    if float(metrics["peak_dbfs"]) < MIN_PEAK_DBFS:
        reasons.append("low_peak")
    if float(metrics["silent_fraction"]) > MAX_SILENT_FRACTION:
        reasons.append("silent_fraction")
    if float(metrics["longest_silent_run_seconds"]) > MAX_SILENT_RUN_SECONDS:
        reasons.append("silent_run")
    if float(metrics["side_to_total_db"]) < MIN_SIDE_TO_TOTAL_DB:
        reasons.append("dual_mono")
    if float(metrics["clipped_fraction"]) > MAX_CLIPPED_FRACTION:
        reasons.append("clipping")
    return reasons


def _download_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("id"):
            return value
    raise ValueError("yt-dlp did not emit download metadata")


def _source_file(root: Path) -> Path:
    matches = [
        path
        for path in root.glob("source.*")
        if path.suffix not in {".json", ".part", ".ytdl"}
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one downloaded source section, found {matches}")
    return matches[0]


def acquire_candidate(candidate: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    video_id = str(candidate["video_id"])
    start = float(candidate["clip_start_seconds"])
    section_start = max(0.0, start - 1.0)
    section_end = start + CLIP_SECONDS + 1.0
    destination = output_dir / "audio" / f"{video_id}_{round(start * 1000):010d}.wav"
    result = {**candidate, "attempted_at": _now()}
    try:
        with tempfile.TemporaryDirectory(prefix="sam-youtube-random-") as temporary:
            root = Path(temporary)
            download = _run(
                [
                    sys.executable,
                    "-m",
                    "yt_dlp",
                    "--no-update",
                    "--quiet",
                    "--no-warnings",
                    "--no-playlist",
                    "--socket-timeout",
                    "20",
                    "--retries",
                    "2",
                    "--extractor-retries",
                    "2",
                    "--download-sections",
                    f"*{section_start:.3f}-{section_end:.3f}",
                    "--force-keyframes-at-cuts",
                    "-f",
                    (
                        "bestaudio[asr>=44100][audio_channels=2][abr>=120]/"
                        "bestaudio[acodec!=none]"
                    ),
                    "--print-json",
                    "-o",
                    str(root / "source.%(ext)s"),
                    str(candidate["source_url"]),
                ],
                timeout=150,
            )
            info = _download_json(download.stdout)
            source_format = {
                "format_id": info.get("format_id"),
                "codec": info.get("acodec"),
                "container": info.get("ext"),
                "sample_rate_hz": int(info.get("asr") or 0),
                "bitrate_kbps": float(info.get("abr") or 0.0),
                "channels": int(info.get("audio_channels") or 0),
            }
            source = _source_file(root)
            normalized = root / "clip.wav"
            _run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-y",
                    "-ss",
                    "1.0",
                    "-i",
                    str(source),
                    "-t",
                    str(CLIP_SECONDS),
                    "-vn",
                    "-ac",
                    "2",
                    "-ar",
                    str(OUTPUT_SAMPLE_RATE),
                    "-acodec",
                    "pcm_s16le",
                    str(normalized),
                ],
                timeout=45,
            )
            metrics = analyze_wav(normalized)
            rejections = quality_rejections(metrics, source_format)
            result.update(
                {
                    "source_format": source_format,
                    "quality_metrics": metrics,
                    "quality_rejections": rejections,
                }
            )
            if rejections:
                result["retrieval_status"] = "rejected"
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(normalized, destination)
                result.update(
                    {
                        "retrieval_status": "success",
                        "local_path": str(destination.relative_to(output_dir)),
                        "sha256": sha256_file(destination),
                        "bytes": destination.stat().st_size,
                    }
                )
    except Exception as error:
        destination.unlink(missing_ok=True)
        result.update(
            {
                "retrieval_status": "unavailable",
                "error": f"{type(error).__name__}: {error}",
            }
        )
    result["processing_seconds"] = round(time.perf_counter() - started, 3)
    return result


def _load_attempts(
    path: Path, output_dir: Path
) -> tuple[list[dict[str, Any]], set[str]]:
    attempts: list[dict[str, Any]] = []
    attempted: set[str] = set()
    if not path.exists():
        return attempts, attempted
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("retrieval_status") == "success" and not (
            output_dir / str(item.get("local_path"))
        ).exists():
            continue
        attempts.append(item)
        attempted.add(str(item["video_id"]))
    return attempts, attempted


def _accepted(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in attempts if item.get("retrieval_status") == "success"]


def _criteria() -> dict[str, Any]:
    return {
        "clip_seconds": CLIP_SECONDS,
        "output": "stereo PCM16 WAV at 48 kHz",
        "minimum_source_sample_rate_hz": MIN_SOURCE_SAMPLE_RATE,
        "minimum_source_bitrate_kbps": MIN_SOURCE_BITRATE_KBPS,
        "minimum_rms_dbfs": MIN_RMS_DBFS,
        "minimum_peak_dbfs": MIN_PEAK_DBFS,
        "silence_threshold_dbfs": SILENCE_THRESHOLD_DBFS,
        "maximum_silent_fraction": MAX_SILENT_FRACTION,
        "maximum_silent_run_seconds": MAX_SILENT_RUN_SECONDS,
        "minimum_side_to_total_db": MIN_SIDE_TO_TOTAL_DB,
        "maximum_clipped_fraction": MAX_CLIPPED_FRACTION,
        "one_clip_per_unique_video": True,
        "candidate_source": "general YouTube search; no AudioSet metadata",
    }


def write_manifest(
    output_dir: Path,
    attempts: list[dict[str, Any]],
    *,
    target: int,
    seed: int,
) -> Path:
    records = _accepted(attempts)[:target]
    for index, record in enumerate(records):
        record["record_index"] = index
    status_counts: dict[str, int] = {}
    for item in attempts:
        status = str(item.get("retrieval_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    manifest = {
        "schema_version": 1,
        "name": f"General YouTube random {target} · seed {seed}",
        "created_at": _now(),
        "target_records": target,
        "accepted_records": len(records),
        "selection_seed": seed,
        "selection": "seeded_general_youtube_search",
        "mixture_preference": ["voice", "music", "environmental_sfx"],
        "acceptance_criteria": _criteria(),
        "attempt_statuses": status_counts,
        "records": records,
    }
    path = output_dir / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "manifest.sha256").write_text(f"{digest}  manifest.json\n")
    return path


def verify_dataset(output_dir: Path, *, target: int) -> dict[str, Any]:
    manifest = json.loads((output_dir / "manifest.json").read_text())
    records = manifest.get("records", [])
    failures: list[dict[str, Any]] = []
    video_ids: set[str] = set()
    total_bytes = 0
    for record in records:
        path = output_dir / str(record["local_path"])
        reasons: list[str] = []
        if not path.exists():
            reasons.append("missing_file")
        else:
            metrics = analyze_wav(path)
            reasons.extend(quality_rejections(metrics, record["source_format"]))
            if sha256_file(path) != record.get("sha256"):
                reasons.append("sha256")
            total_bytes += path.stat().st_size
        video_id = str(record["video_id"])
        if video_id in video_ids:
            reasons.append("duplicate_video")
        video_ids.add(video_id)
        if reasons:
            failures.append({"video_id": video_id, "reasons": reasons})
    audit = {
        "verified_at": _now(),
        "target_records": target,
        "record_count": len(records),
        "unique_video_count": len(video_ids),
        "total_duration_seconds": len(records) * CLIP_SECONDS,
        "total_bytes": total_bytes,
        "all_requirements_pass": (
            len(records) == target and len(video_ids) == target and not failures
        ),
        "failures": failures,
        "acceptance_criteria": _criteria(),
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def acquire_dataset(
    output_dir: Path,
    *,
    total: int,
    seed: int,
    query_count: int,
    results_per_query: int,
    search_workers: int,
    download_workers: int,
    candidate_multiplier: float,
    max_attempts: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = discover_candidates(
        output_dir,
        seed=seed,
        query_count=query_count,
        results_per_query=results_per_query,
        workers=search_workers,
        minimum_candidates=math.ceil(total * candidate_multiplier),
    )
    attempts_path = output_dir / "attempts.jsonl"
    attempts, attempted = _load_attempts(attempts_path, output_dir)
    accepted_count = len(_accepted(attempts))
    pending = [item for item in candidates if item["video_id"] not in attempted]
    logger.info(
        "Starting with %d/%d accepted; %d candidates remain",
        accepted_count,
        total,
        len(pending),
    )
    attempts_made = 0
    with attempts_path.open("a", encoding="utf-8") as attempt_log:
        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            for offset in range(0, len(pending), download_workers):
                if accepted_count >= total or attempts_made >= max_attempts:
                    break
                batch = pending[offset : offset + download_workers]
                results = list(
                    executor.map(
                        lambda item: acquire_candidate(item, output_dir), batch
                    )
                )
                for result in results:
                    attempts_made += 1
                    if (
                        result.get("retrieval_status") == "success"
                        and accepted_count >= total
                    ):
                        (output_dir / str(result["local_path"])).unlink(missing_ok=True)
                        result["retrieval_status"] = "surplus"
                    if result.get("retrieval_status") == "success":
                        accepted_count += 1
                    attempts.append(result)
                    attempt_log.write(json.dumps(result) + "\n")
                    attempt_log.flush()
                write_manifest(output_dir, attempts, target=total, seed=seed)
                logger.info(
                    "Accepted %d/%d after %d new attempts",
                    accepted_count,
                    total,
                    attempts_made,
                )
    manifest = write_manifest(output_dir, attempts, target=total, seed=seed)
    if accepted_count < total:
        raise RuntimeError(
            f"Only acquired {accepted_count}/{total} clips after "
            f"{attempts_made} new attempts"
        )
    audit = verify_dataset(output_dir, target=total)
    if not audit["all_requirements_pass"]:
        raise RuntimeError("Dataset verification failed; inspect audit.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--query-count", type=int)
    parser.add_argument("--results-per-query", type=int, default=12)
    parser.add_argument("--search-workers", type=int, default=8)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--candidate-multiplier", type=float, default=3.0)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.total < 1:
        parser.error("--total must be positive")
    if args.verify_only:
        result = verify_dataset(args.output, target=args.total)
        print(json.dumps(result, indent=2))
        if not result["all_requirements_pass"]:
            raise SystemExit(1)
        return
    query_count = args.query_count or max(40, math.ceil(args.total * 0.4))
    max_attempts = args.max_attempts or math.ceil(args.total * 3.0)
    path = acquire_dataset(
        args.output,
        total=args.total,
        seed=args.seed,
        query_count=query_count,
        results_per_query=args.results_per_query,
        search_workers=args.search_workers,
        download_workers=args.download_workers,
        candidate_multiplier=args.candidate_multiplier,
        max_attempts=max_attempts,
    )
    print(path)


if __name__ == "__main__":
    main()
