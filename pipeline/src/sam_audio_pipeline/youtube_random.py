"""Build a resumable, quality-gated dataset from general YouTube search."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
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

CINEMATIC_SOURCES = (
    "movie clip",
    "movie scene",
    "film clip",
    "TV show scene",
    "TV series clip",
    "animated movie scene",
    "animated series clip",
    "video game cutscene",
    "game cinematic scene",
    "short film scene",
    "news package",
)

CINEMATIC_SCENES = (
    "action dialogue",
    "dramatic conversation",
    "street dialogue",
    "restaurant conversation",
    "car dialogue",
    "police scene dialogue",
    "hospital scene dialogue",
    "battle dialogue",
    "chase dialogue",
    "argument scene",
    "suspense dialogue",
    "comedy dialogue",
    "crowd scene dialogue",
    "workplace dialogue",
)

CINEMATIC_AUDIO_HINTS = (
    "English HD",
    "English 4K",
    "dialogue HD",
    "soundtrack scene",
    "cinematic sound",
)

CINEMATIC_TITLE_TERMS = (
    "scene",
    "movie clip",
    "film clip",
    "short film",
    "cutscene",
    "cinematic",
    "tv series",
    "episode",
    "news package",
    "field report",
)

CINEMATIC_EXCLUDED_TERMS = (
    *EXCLUDED_TITLE_TERMS,
    "reaction",
    "reacts",
    "review",
    "compilation",
    "top 10",
    "top 20",
    "analysis",
    "breakdown",
    "explained",
    "recap",
    "interview",
    "podcast",
    "vlog",
    "walking tour",
    "walk tour",
    "tutorial",
    "how to",
    "motivational",
    "speech",
    "audiobook",
    "voice over",
    "voiceover",
    "text to speech",
    "ai voice",
    "fan edit",
    "re-edit",
    "dialogue edit",
    "amv",
    "full movie",
    "youtube shorts",
    "#shorts",
    "cinematic sound effects",
    "action camera",
    "whatsapp status",
    "backsound",
    "behind the scenes",
    "bollywood",
    "hindi",
    "tamil",
    "telugu",
    "malayalam",
    "kannada",
    "bengali",
    "punjabi",
    "marathi",
    "gujarati",
    "bhojpuri",
    "mollywood",
    "tollywood",
    "kollywood",
    " india ",
    " indian ",
    "mirzapur",
    "taarak mehta",
    "mammootty",
    "vijayakanth",
    "mohanlal",
    "ajay devgan",
    "sunny deol",
    "sanjay dutt",
    "cine curry",
    "tamilbiscoot",
    "ary digital",
    "hum tv",
    "geo entertainment",
    "zee tv",
    "colors tv",
    "starplus",
    "sony sab",
    "learn english",
    "english lesson",
    "unit 1",
    "unit 2",
    "unit 3",
    "unit 4",
    "unit 5",
)

CINEMATIC_SEARCH_EXCLUSIONS = (
    "-reaction -review -explained -recap -interview -vlog -tutorial "
    "-Bollywood -Hindi -Tamil -Telugu -India -Indian -lyrics -AMV"
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


def build_queries(seed: int, count: int, *, profile: str = "general") -> list[str]:
    """Create reproducible YouTube queries for a general or cinematic mix."""
    generator = random.Random(seed)
    queries: list[str] = []
    seen: set[str] = set()
    while len(queries) < count:
        if profile == "cinematic":
            query = " ".join(
                (
                    generator.choice(CINEMATIC_SOURCES),
                    generator.choice(CINEMATIC_SCENES),
                    generator.choice(CINEMATIC_AUDIO_HINTS),
                    "English",
                    CINEMATIC_SEARCH_EXCLUSIONS,
                )
            )
        else:
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


def _query_for_source(query: str, source: str) -> str:
    """Remove YouTube-only negative tokens for APIs that treat them literally."""
    if source == "youtube":
        return query
    return " ".join(part for part in query.split() if not part.startswith("-"))


def _candidate_allowed(item: dict[str, Any], *, profile: str = "general") -> bool:
    duration = float(item.get("duration") or item.get("duration_seconds") or 0.0)
    title = f" {str(item.get('title') or '').lower()} "
    uploader = f" {str(item.get('uploader') or item.get('channel') or '').lower()} "
    description = f" {str(item.get('description') or '').lower()} "
    tags = " " + " ".join(str(tag).lower() for tag in item.get("tags") or []) + " "
    text = title + uploader + description + tags
    normalized_text = f" {re.sub(r'[^a-z0-9]+', ' ', text).strip()} "
    excluded = (
        CINEMATIC_EXCLUDED_TERMS if profile == "cinematic" else EXCLUDED_TITLE_TERMS
    )
    return (
        bool(item.get("id") or item.get("video_id"))
        and 30.0 <= duration <= 3600.0
        and item.get("live_status") not in {"is_live", "is_upcoming"}
        and not any(
            term in text
            or f" {re.sub(r'[^a-z0-9]+', ' ', term).strip()} " in normalized_text
            for term in excluded
        )
        and (
            profile != "cinematic"
            or any(term in title for term in CINEMATIC_TITLE_TERMS)
        )
    )


def _search_youtube(
    query: str, results: int, profile: str
) -> list[dict[str, Any]]:
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
    return [
        item
        for item in payload.get("entries", [])
        if _candidate_allowed(item, profile=profile)
    ]


def _search_dailymotion(
    query: str, results: int, profile: str
) -> list[dict[str, Any]]:
    fields = (
        "id,title,description,duration,owner.screenname,url,language,tags,"
        "created_time"
    )
    parameters = urllib.parse.urlencode(
        {
            "search": _query_for_source(query, "dailymotion"),
            "fields": fields,
            "limit": min(results, 100),
            "sort": "relevance",
            "language": "en",
        }
    )
    request = urllib.request.Request(
        f"https://api.dailymotion.com/videos?{parameters}",
        headers={"User-Agent": "sam-audio-dataset-builder/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    items: list[dict[str, Any]] = []
    for value in payload.get("list", []):
        item = {
            **value,
            "uploader": value.get("owner.screenname"),
            "source_url": value.get("url"),
        }
        if str(item.get("language") or "").lower() not in {"", "en"}:
            continue
        if _candidate_allowed(item, profile=profile):
            items.append(item)
    return items


def _search(
    query: str, results: int, profile: str, source: str = "youtube"
) -> list[dict[str, Any]]:
    if source == "dailymotion":
        return _search_dailymotion(query, results, profile)
    return _search_youtube(query, results, profile)


def _sample_clip_starts(
    *, seed: int, video_id: str, duration: float, clips_per_video: int
) -> list[float]:
    """Pick deterministic, non-overlapping ten-second excerpts from one source."""
    lower = 5.0
    upper = duration - CLIP_SECONDS - 5.0
    if upper <= lower:
        return []
    maximum = max(1, math.floor((upper - lower) / (CLIP_SECONDS + 2.0)) + 1)
    wanted = min(clips_per_video, maximum)
    generator = random.Random(f"{seed}:{video_id}")
    starts: list[float] = []
    for _ in range(200):
        if len(starts) >= wanted:
            break
        candidate = generator.uniform(lower, upper)
        if all(abs(candidate - existing) >= CLIP_SECONDS + 2.0 for existing in starts):
            starts.append(candidate)
    if not starts:
        starts.append(generator.uniform(lower, upper))
    return sorted(starts)


def discover_candidates(
    output_dir: Path,
    *,
    seed: int,
    query_count: int,
    results_per_query: int,
    workers: int,
    minimum_candidates: int,
    profile: str = "general",
    clips_per_video: int = 1,
    source: str = "youtube",
) -> list[dict[str, Any]]:
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = metadata_dir / "candidates.json"
    search_path = metadata_dir / "search.json"
    if candidates_path.exists() and search_path.exists():
        existing = json.loads(candidates_path.read_text())
        search_metadata = json.loads(search_path.read_text())
        compatible = (
            search_metadata.get("profile", "general") == profile
            and int(search_metadata.get("clips_per_video", 1)) == clips_per_video
            and search_metadata.get("source", "youtube") == source
        )
        if compatible and len(existing) >= minimum_candidates:
            logger.info("Reusing %d discovered candidates", len(existing))
            return existing

    queries = build_queries(seed, query_count, profile=profile)
    found: dict[str, dict[str, Any]] = {}
    found_videos: set[str] = set()
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(_search, query, results_per_query, profile, source): query
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
                if video_id in found_videos:
                    continue
                found_videos.add(video_id)
                duration = float(item["duration"])
                starts = _sample_clip_starts(
                    seed=seed,
                    video_id=video_id,
                    duration=duration,
                    clips_per_video=clips_per_video,
                )
                for segment_index, start in enumerate(starts):
                    candidate_id = f"{video_id}:{round(start * 1000)}"
                    found[candidate_id] = {
                        "candidate_id": candidate_id,
                        "video_id": video_id,
                        "source_url": (
                            item.get("source_url")
                            or (
                                f"https://www.dailymotion.com/video/{video_id}"
                                if source == "dailymotion"
                                else f"https://www.youtube.com/watch?v={video_id}"
                            )
                        ),
                        "source_platform": source,
                        "title": item.get("title"),
                        "duration_seconds": duration,
                        "uploader": item.get("uploader") or item.get("channel"),
                        "channel_id": item.get("channel_id"),
                        "view_count": item.get("view_count"),
                        "search_query": query,
                        "clip_start_seconds": round(start, 3),
                        "clip_end_seconds": round(start + CLIP_SECONDS, 3),
                        "segment_index": segment_index,
                        "selection": f"seeded_{profile}_{source}_search",
                        "selection_seed": seed,
                        "mixture_bias": [
                            "dialogue",
                            "music",
                            "environmental_sfx",
                        ],
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
    search_path.write_text(
        json.dumps(
            {
                "selection": f"seeded_{profile}_{source}_search",
                "source": source,
                "profile": profile,
                "clips_per_video": clips_per_video,
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


def _download_jsons(stdout: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("id"):
            values.append(value)
    if not values:
        raise ValueError("yt-dlp did not emit download metadata")
    return values


def _section_bounds(candidate: dict[str, Any]) -> tuple[float, float, float]:
    start = float(candidate["clip_start_seconds"])
    section_start = max(0.0, start - 5.0)
    section_end = start + CLIP_SECONDS + 5.0
    return section_start, section_end, start - section_start


def _use_full_source_for_group(candidates: list[dict[str, Any]]) -> bool:
    """Prefer one source transfer when section requests cover much of the video."""
    if not candidates:
        return False
    duration = float(candidates[0].get("duration_seconds") or 0.0)
    if duration <= 0:
        return False
    requested_with_padding = len(candidates) * (CLIP_SECONDS + 10.0)
    return requested_with_padding / duration >= 0.4


def _source_file(root: Path) -> Path:
    matches = [
        path
        for path in root.glob("source.*")
        if path.suffix not in {".json", ".part", ".ytdl"}
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one downloaded source section, found {matches}")
    return matches[0]


def _download_section(
    candidate: dict[str, Any],
    root: Path,
    section: str,
    *,
    youtube_client: str,
) -> tuple[subprocess.CompletedProcess[str], Path, str]:
    source_platform = candidate.get("source_platform", "youtube")
    clients = (
        ("dailymotion",)
        if source_platform == "dailymotion"
        else (("mweb", "default") if youtube_client == "auto" else (youtube_client,))
    )
    last_error: Exception | None = None
    for client in clients:
        download_root = root / client
        download_root.mkdir(parents=True, exist_ok=True)
        command = [
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
        ]
        deno = shutil.which("deno") or str(Path.home() / ".deno" / "bin" / "deno")
        if Path(deno).is_file():
            command.extend(["--js-runtimes", f"deno:{deno}"])
        if client == "mweb":
            command.extend(
                ["--extractor-args", "youtube:player_client=mweb"]
            )
        elif client == "android":
            command.extend(
                [
                    "--extractor-args",
                    "youtube:player_client=android;player_skip=webpage,configs",
                ]
            )
        command.extend(
            [
                "--download-sections",
                section,
                "--force-keyframes-at-cuts",
                "-f",
                (
                    "bestaudio[asr>=44100][audio_channels=2][abr>=120]/"
                    "bestaudio[acodec!=none]/best[acodec!=none]"
                ),
                "--print-json",
                "-o",
                str(download_root / "source.%(ext)s"),
                str(candidate["source_url"]),
            ]
        )
        try:
            response = _run(command, timeout=75)
            return response, _source_file(download_root), client
        except Exception as error:
            last_error = error
    assert last_error is not None
    raise last_error


def _source_format(path: Path, info: dict[str, Any], client: str) -> dict[str, Any]:
    response = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,bit_rate",
            "-of",
            "json",
            str(path),
        ],
        timeout=30,
    )
    streams = json.loads(response.stdout).get("streams") or []
    stream = streams[0] if streams else {}
    stream_bitrate = float(stream.get("bit_rate") or 0.0) / 1000.0
    return {
        "format_id": info.get("format_id"),
        "codec": info.get("acodec") or stream.get("codec_name"),
        "container": info.get("ext") or path.suffix.lstrip("."),
        "sample_rate_hz": int(info.get("asr") or stream.get("sample_rate") or 0),
        "bitrate_kbps": float(info.get("abr") or stream_bitrate),
        "channels": int(info.get("audio_channels") or stream.get("channels") or 0),
        "retrieval_client": client,
    }


def _normalize_candidate_from_source(
    candidate: dict[str, Any],
    source: Path,
    info: dict[str, Any],
    retrieval_client: str,
    output_dir: Path,
    temporary_root: Path,
    *,
    started: float,
    source_start_seconds: float | None = None,
) -> dict[str, Any]:
    video_id = str(candidate["video_id"])
    start = float(candidate["clip_start_seconds"])
    if source_start_seconds is None:
        _, _, trim_offset = _section_bounds(candidate)
    else:
        trim_offset = start - source_start_seconds
        if trim_offset < 0:
            raise ValueError("Candidate starts before the downloaded source")
    destination = output_dir / "audio" / f"{video_id}_{round(start * 1000):010d}.wav"
    normalized = temporary_root / f"clip-{video_id}-{round(start * 1000)}.wav"
    result = {**candidate, "attempted_at": _now()}
    source_format = _source_format(source, info, retrieval_client)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            str(trim_offset),
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
        shutil.move(normalized, destination)
        result.update(
            {
                "retrieval_status": "success",
                "local_path": str(destination.relative_to(output_dir)),
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
            }
        )
    result["processing_seconds"] = round(time.perf_counter() - started, 3)
    return result


def acquire_candidate(
    candidate: dict[str, Any],
    output_dir: Path,
    *,
    youtube_client: str = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    section_start, section_end, _ = _section_bounds(candidate)
    result = {**candidate, "attempted_at": _now()}
    try:
        with tempfile.TemporaryDirectory(prefix="sam-source-random-") as temporary:
            root = Path(temporary)
            download, source, retrieval_client = _download_section(
                candidate,
                root,
                f"*{section_start:.3f}-{section_end:.3f}",
                youtube_client=youtube_client,
            )
            info = _download_json(download.stdout)
            return _normalize_candidate_from_source(
                candidate,
                source,
                info,
                retrieval_client,
                output_dir,
                root,
                started=started,
            )
    except Exception as error:
        result.update(
            {
                "retrieval_status": "unavailable",
                "error": f"{type(error).__name__}: {error}",
            }
        )
    result["processing_seconds"] = round(time.perf_counter() - started, 3)
    return result


def acquire_candidate_group(
    candidates: list[dict[str, Any]],
    output_dir: Path,
    *,
    youtube_client: str = "auto",
) -> list[dict[str, Any]]:
    """Retrieve multiple Dailymotion sections with one metadata session."""
    if (
        len(candidates) <= 1
        or candidates[0].get("source_platform") != "dailymotion"
    ):
        return [
            acquire_candidate(item, output_dir, youtube_client=youtube_client)
            for item in candidates
        ]
    video_ids = {str(item["video_id"]) for item in candidates}
    if len(video_ids) != 1:
        raise ValueError("Grouped candidates must come from one video")
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="sam-source-group-") as temporary:
            root = Path(temporary)
            command = [
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
            ]
            use_full_source = _use_full_source_for_group(candidates)
            if use_full_source:
                command.extend(
                    [
                        "-f",
                        (
                            "bestaudio[asr>=44100][audio_channels=2][abr>=120]/"
                            "bestaudio[acodec!=none]/best[acodec!=none]"
                        ),
                        "--print-json",
                        "-o",
                        str(root / "source.%(ext)s"),
                        str(candidates[0]["source_url"]),
                    ]
                )
                response = _run(command, timeout=600)
                full_source = _source_file(root)
                full_info = _download_json(response.stdout)
                info_by_start: dict[float, dict[str, Any]] = {}
            else:
                for item in candidates:
                    section_start, section_end, _ = _section_bounds(item)
                    command.extend(
                        [
                            "--download-sections",
                            f"*{section_start:.3f}-{section_end:.3f}",
                        ]
                    )
                command.extend(
                    [
                        "--force-keyframes-at-cuts",
                        "-f",
                        (
                            "bestaudio[asr>=44100][audio_channels=2][abr>=120]/"
                            "bestaudio[acodec!=none]/best[acodec!=none]"
                        ),
                        "--print-json",
                        "-o",
                        str(root / "source-%(section_start)012.3f.%(ext)s"),
                        str(candidates[0]["source_url"]),
                    ]
                )
                response = _run(command, timeout=60 + 20 * len(candidates))
                info_by_start = {
                    round(float(info["section_start"]), 3): info
                    for info in _download_jsons(response.stdout)
                }
            for item in candidates:
                section_start, _, _ = _section_bounds(item)
                try:
                    if use_full_source:
                        source = full_source
                        info = full_info
                        retrieval_client = "dailymotion-grouped-full"
                        source_start_seconds = 0.0
                    else:
                        matches = list(root.glob(f"source-{section_start:012.3f}.*"))
                        info = info_by_start[round(section_start, 3)]
                        if len(matches) != 1:
                            raise ValueError(
                                "Expected one section for "
                                f"{section_start}, found {matches}"
                            )
                        source = matches[0]
                        retrieval_client = "dailymotion-grouped"
                        source_start_seconds = None
                    results.append(
                        _normalize_candidate_from_source(
                            item,
                            source,
                            info,
                            retrieval_client,
                            output_dir,
                            root,
                            started=started,
                            source_start_seconds=source_start_seconds,
                        )
                    )
                except Exception as error:
                    results.append(
                        {
                            **item,
                            "attempted_at": _now(),
                            "retrieval_status": "unavailable",
                            "error": f"{type(error).__name__}: {error}",
                            "processing_seconds": round(
                                time.perf_counter() - started, 3
                            ),
                        }
                    )
    except Exception as error:
        return [
            {
                **item,
                "attempted_at": _now(),
                "retrieval_status": "unavailable",
                "error": f"{type(error).__name__}: {error}",
                "processing_seconds": round(time.perf_counter() - started, 3),
            }
            for item in candidates
        ]
    return results


def _group_candidates_by_video(
    candidates: list[dict[str, Any]], *, grouped: bool
) -> list[list[dict[str, Any]]]:
    if not grouped:
        return [[item] for item in candidates]
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        groups.setdefault(str(item["video_id"]), []).append(item)
    return list(groups.values())


def _candidate_key(item: dict[str, Any]) -> str:
    return str(
        item.get("candidate_id")
        or f"{item['video_id']}:{round(float(item['clip_start_seconds']) * 1000)}"
    )


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
        attempted.add(_candidate_key(item))
    return attempts, attempted


def _accepted(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in attempts if item.get("retrieval_status") == "success"]


def _criteria(
    *, profile: str = "general", clips_per_video: int = 1, source: str = "youtube"
) -> dict[str, Any]:
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
        "maximum_clips_per_video": clips_per_video,
        "candidate_source": f"{profile} {source.title()} search; no AudioSet metadata",
        "source_platform": source,
        "source_profile": profile,
        "explicit_metadata_exclusions": (
            list(CINEMATIC_EXCLUDED_TERMS) if profile == "cinematic" else []
        ),
    }


def write_manifest(
    output_dir: Path,
    attempts: list[dict[str, Any]],
    *,
    target: int,
    seed: int,
    profile: str = "general",
    clips_per_video: int = 1,
    source: str = "youtube",
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
        "name": f"{profile.title()} {source.title()} {target} · seed {seed}",
        "created_at": _now(),
        "target_records": target,
        "accepted_records": len(records),
        "selection_seed": seed,
        "selection": f"seeded_{profile}_{source}_search",
        "mixture_preference": ["dialogue", "music", "environmental_sfx"],
        "acceptance_criteria": _criteria(
            profile=profile, clips_per_video=clips_per_video, source=source
        ),
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
    video_counts: dict[str, int] = {}
    clip_ids: set[str] = set()
    criteria = manifest.get("acceptance_criteria", {})
    clips_per_video = int(criteria.get("maximum_clips_per_video", 1))
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
        clip_id = _candidate_key(record)
        if clip_id in clip_ids:
            reasons.append("duplicate_clip")
        clip_ids.add(clip_id)
        video_counts[video_id] = video_counts.get(video_id, 0) + 1
        if video_counts[video_id] > clips_per_video:
            reasons.append("too_many_clips_from_video")
        if reasons:
            failures.append({"video_id": video_id, "reasons": reasons})
    audit = {
        "verified_at": _now(),
        "target_records": target,
        "record_count": len(records),
        "unique_video_count": len(video_counts),
        "unique_clip_count": len(clip_ids),
        "total_duration_seconds": len(records) * CLIP_SECONDS,
        "total_bytes": total_bytes,
        "all_requirements_pass": (
            len(records) == target and len(clip_ids) == target and not failures
        ),
        "failures": failures,
        "acceptance_criteria": criteria,
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
    youtube_client: str,
    profile: str = "general",
    clips_per_video: int = 1,
    source: str = "youtube",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = discover_candidates(
        output_dir,
        seed=seed,
        query_count=query_count,
        results_per_query=results_per_query,
        workers=search_workers,
        minimum_candidates=math.ceil(total * candidate_multiplier),
        profile=profile,
        clips_per_video=clips_per_video,
        source=source,
    )
    attempts_path = output_dir / "attempts.jsonl"
    attempts, attempted = _load_attempts(attempts_path, output_dir)
    accepted_count = len(_accepted(attempts))
    pending = [
        item
        for item in candidates
        if _candidate_key(item) not in attempted
    ]
    logger.info(
        "Starting with %d/%d accepted; %d candidates remain",
        accepted_count,
        total,
        len(pending),
    )
    work_groups = _group_candidates_by_video(
        pending,
        grouped=(source == "dailymotion" and clips_per_video > 1),
    )
    attempts_made = 0
    with attempts_path.open("a", encoding="utf-8") as attempt_log:
        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            for offset in range(0, len(work_groups), download_workers):
                if accepted_count >= total or attempts_made >= max_attempts:
                    break
                capacity = max_attempts - attempts_made
                batch: list[list[dict[str, Any]]] = []
                for group in work_groups[offset : offset + download_workers]:
                    if capacity <= 0:
                        break
                    selected = group[:capacity]
                    batch.append(selected)
                    capacity -= len(selected)
                grouped_results = list(
                    executor.map(
                        lambda group: acquire_candidate_group(
                            group, output_dir, youtube_client=youtube_client
                        ),
                        batch,
                    )
                )
                results = [
                    result
                    for group_results in grouped_results
                    for result in group_results
                ]
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
                write_manifest(
                    output_dir,
                    attempts,
                    target=total,
                    seed=seed,
                    profile=profile,
                    clips_per_video=clips_per_video,
                    source=source,
                )
                logger.info(
                    "Accepted %d/%d after %d new attempts",
                    accepted_count,
                    total,
                    attempts_made,
                )
    manifest = write_manifest(
        output_dir,
        attempts,
        target=total,
        seed=seed,
        profile=profile,
        clips_per_video=clips_per_video,
        source=source,
    )
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
    parser.add_argument(
        "--profile", choices=("general", "cinematic"), default="general"
    )
    parser.add_argument(
        "--source", choices=("youtube", "dailymotion"), default="youtube"
    )
    parser.add_argument(
        "--clips-per-video",
        type=int,
        help="Defaults to 3 for cinematic and 1 for general acquisition",
    )
    parser.add_argument(
        "--youtube-client",
        choices=("auto", "default", "mweb", "android"),
        default="auto",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.total < 1:
        parser.error("--total must be positive")
    clips_per_video = args.clips_per_video or (
        3 if args.profile == "cinematic" else 1
    )
    if clips_per_video < 1:
        parser.error("--clips-per-video must be positive")
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
        youtube_client=args.youtube_client,
        profile=args.profile,
        clips_per_video=clips_per_video,
        source=args.source,
    )
    print(path)


if __name__ == "__main__":
    main()
