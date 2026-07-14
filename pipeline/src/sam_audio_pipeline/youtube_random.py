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
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import wave
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .audio import sha256_file
from .source_diversity import (
    DEFAULT_MAX_CLIPS_PER_SOURCE,
    record_source_clip_budget,
    source_clip_budget,
    source_diversity_policy,
)

logger = logging.getLogger(__name__)

CLIP_SECONDS = 10.0
OUTPUT_SAMPLE_RATE = 48_000
MIN_SOURCE_SAMPLE_RATE = 44_100
MIN_SOURCE_BITRATE_KBPS = 120.0
MIN_SOURCE_DURATION_SECONDS = 30.0
MAX_SOURCE_DURATION_SECONDS = 12 * 3600.0
CANDIDATE_DURATION_POLICY = "source_duration_up_to_12h_v1"
DAILYMOTION_SEARCH_POLICY = "seeded_relevance_pages_1_to_6_v1"
YTDLP_PYTHON = sys.executable
SILENCE_THRESHOLD_DBFS = -55.0
MIN_RMS_DBFS = -35.0
MIN_PEAK_DBFS = -20.0
MAX_SILENT_FRACTION = 0.10
MAX_SILENT_RUN_SECONDS = 0.75
MIN_SIDE_TO_TOTAL_DB = -45.0
MAX_CLIPPED_FRACTION = 0.01
MAX_SCAN_REGIONS_PER_ACQUISITION = 8
HIGH_QUALITY_AUDIO_SELECTOR = (
    "bestaudio[asr>=44100][audio_channels=2][abr>=120]/"
    "bestaudio[acodec!=none]/best[acodec!=none]"
)
# Audio-only downloads avoid transferring and decoding an unused 720p video stream.
DAILYMOTION_EFFICIENT_SELECTOR = HIGH_QUALITY_AUDIO_SELECTOR
# Dailymotion exposes combined HLS variants rather than a separate audio stream.
# The lowest >=480p variant carries ~128 kbps audio; 380p carries 32-64 kbps
# and always failed our source-quality gate. Avoid both 1080p transfer waste and
# downloading a 380p source that will be rejected after the fact.
DAILYMOTION_SOURCE_SCAN_SELECTOR = "worst[height>=480][acodec!=none]"
DAILYMOTION_SCAN_PROXY_SELECTOR = "worst[acodec!=none]"

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
    "relaxation",
    "relaxing",
    "scenic film",
    "wild animals",
    "nature sounds",
    "ocean waves",
    "sleep sounds",
    "meditation",
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
    "english sub",
    "englishsub",
    "multi sub",
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
    "prabhas",
    "mass entry",
    "dhanush",
    "vadivelu",
    "adithya tv",
    "hera pheri",
    "paresh raval",
    "manipuri",
    "nepali movie",
    "auzaar",
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

CINEMATIC_PRIORITY_WEIGHTS = (
    ("cutscene", 8),
    ("movie clip", 8),
    ("movie scene", 7),
    ("cinematic", 6),
    ("full episode", 5),
    ("short film", 5),
    ("animated", 4),
    ("battle scene", 4),
    ("dialogue", 4),
    ("fight scene", 4),
    ("episode", 3),
    ("chase", 3),
    ("english", 3),
    ("4k", 1),
    (" hd", 1),
    ("amazing", -2),
    ("best scene", -3),
    ("funny clips", -3),
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
        and MIN_SOURCE_DURATION_SECONDS <= duration <= MAX_SOURCE_DURATION_SECONDS
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


def _cinematic_candidate_priority(item: dict[str, Any]) -> int:
    """Rank explicit cinematic source markers without inspecting the speaker."""
    title = f" {str(item.get('title') or '').lower()} "
    return sum(weight for term, weight in CINEMATIC_PRIORITY_WEIGHTS if term in title)


def _search_youtube(query: str, results: int, profile: str) -> list[dict[str, Any]]:
    response = _run(
        [
            YTDLP_PYTHON,
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


def _dailymotion_search_page(seed: int, query: str, *, pages: int = 6) -> int:
    """Spread repeated query combinations across Dailymotion result pages."""
    return random.Random(f"{seed}:{query}:dailymotion-page").randrange(1, pages + 1)


def _search_dailymotion(
    query: str, results: int, profile: str, *, page: int = 1
) -> list[dict[str, Any]]:
    fields = (
        "id,title,description,duration,owner.screenname,url,language,tags,created_time"
    )
    parameters = urllib.parse.urlencode(
        {
            "search": _query_for_source(query, "dailymotion"),
            "fields": fields,
            "limit": min(results, 100),
            "sort": "relevance",
            "language": "en",
            "page": page,
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
    query: str,
    results: int,
    profile: str,
    source: str = "youtube",
    *,
    search_page: int = 1,
) -> list[dict[str, Any]]:
    if source == "dailymotion":
        return _search_dailymotion(query, results, profile, page=search_page)
    return _search_youtube(query, results, profile)


def _sample_clip_starts(
    *,
    seed: int,
    video_id: str,
    duration: float,
    clips_per_video: int,
    source_content_minutes_per_hour: float | None = None,
    max_clips_per_video: int = DEFAULT_MAX_CLIPS_PER_SOURCE,
) -> list[float]:
    """Pick deterministic, non-overlapping excerpts from one source."""
    lower = 5.0
    upper = duration - CLIP_SECONDS - 5.0
    if upper <= lower:
        return []
    maximum = max(1, math.floor((upper - lower) / (CLIP_SECONDS + 2.0)) + 1)
    clip_budget = clips_per_video
    if source_content_minutes_per_hour is not None:
        clip_budget = source_clip_budget(
            duration,
            clip_seconds=CLIP_SECONDS,
            base_clips=clips_per_video,
            content_minutes_per_hour=source_content_minutes_per_hour,
            max_clips=max_clips_per_video,
        )
    wanted = min(clip_budget, maximum)
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
    source_content_minutes_per_hour: float | None = None,
    max_clips_per_video: int = DEFAULT_MAX_CLIPS_PER_SOURCE,
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
            and search_metadata.get("source_content_minutes_per_hour")
            == source_content_minutes_per_hour
            and int(
                search_metadata.get("max_clips_per_video", DEFAULT_MAX_CLIPS_PER_SOURCE)
            )
            == max_clips_per_video
            and search_metadata.get("source", "youtube") == source
            and search_metadata.get("candidate_duration_policy")
            == CANDIDATE_DURATION_POLICY
            and (
                source != "dailymotion"
                or search_metadata.get("search_page_policy")
                == DAILYMOTION_SEARCH_POLICY
            )
        )
        if compatible:
            filtered = [
                item for item in existing if _candidate_allowed(item, profile=profile)
            ]
            if len(filtered) >= minimum_candidates:
                if len(filtered) != len(existing):
                    logger.info(
                        "Removed %d cached candidates under the current "
                        "metadata policy",
                        len(existing) - len(filtered),
                    )
                    candidates_path.write_text(json.dumps(filtered, indent=2) + "\n")
                    search_metadata["unique_candidates"] = len(filtered)
                    search_metadata["metadata_policy_refiltered_at"] = _now()
                    search_path.write_text(json.dumps(search_metadata, indent=2) + "\n")
                logger.info("Reusing %d discovered candidates", len(filtered))
                return filtered

    queries = build_queries(seed, query_count, profile=profile)
    found: dict[str, dict[str, Any]] = {}
    found_videos: set[str] = set()
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(
                _search,
                query,
                results_per_query,
                profile,
                source,
                search_page=(
                    _dailymotion_search_page(seed, query)
                    if source == "dailymotion"
                    else 1
                ),
            ): query
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
                    source_content_minutes_per_hour=(source_content_minutes_per_hour),
                    max_clips_per_video=max_clips_per_video,
                )
                source_budget = (
                    source_clip_budget(
                        duration,
                        clip_seconds=CLIP_SECONDS,
                        base_clips=clips_per_video,
                        content_minutes_per_hour=source_content_minutes_per_hour,
                        max_clips=max_clips_per_video,
                    )
                    if source_content_minutes_per_hour is not None
                    else clips_per_video
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
                        "source_clip_budget": source_budget,
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
                "source_content_minutes_per_hour": (source_content_minutes_per_hour),
                "max_clips_per_video": max_clips_per_video,
                "candidate_duration_policy": CANDIDATE_DURATION_POLICY,
                "search_page_policy": (
                    DAILYMOTION_SEARCH_POLICY if source == "dailymotion" else None
                ),
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
        channel_correlation = float(np.corrcoef(samples[:, 0], samples[:, 1])[0, 1])
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
    metrics: dict[str, Any],
    source_format: dict[str, Any],
    *,
    clip_seconds: float | None = None,
) -> list[str]:
    reasons: list[str] = []
    if int(source_format.get("sample_rate_hz") or 0) < MIN_SOURCE_SAMPLE_RATE:
        reasons.append("source_sample_rate")
    if int(source_format.get("channels") or 0) != 2:
        reasons.append("source_not_stereo")
    if float(source_format.get("bitrate_kbps") or 0.0) < MIN_SOURCE_BITRATE_KBPS:
        reasons.append("source_bitrate")
    expected_duration = CLIP_SECONDS if clip_seconds is None else clip_seconds
    if abs(float(metrics["duration_seconds"]) - expected_duration) > 0.02:
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
    """Dailymotion grouped clips are faster and more reliable from one transfer."""
    if not candidates:
        return False
    duration = float(candidates[0].get("duration_seconds") or 0.0)
    return len(candidates) > 1 and 0 < duration <= 3600


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
            YTDLP_PYTHON,
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
            command.extend(["--extractor-args", "youtube:player_client=mweb"])
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
                    DAILYMOTION_EFFICIENT_SELECTOR
                    if source_platform == "dailymotion"
                    else HIGH_QUALITY_AUDIO_SELECTOR
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
            "-threads",
            "1",
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
    if len(candidates) <= 1 or candidates[0].get("source_platform") != "dailymotion":
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
                YTDLP_PYTHON,
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
                        DAILYMOTION_EFFICIENT_SELECTOR,
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
                        DAILYMOTION_EFFICIENT_SELECTOR,
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


def load_catalog_source_guidance(
    catalog_path: Path | None, *, platform: str
) -> dict[str, dict[str, Any]]:
    """Load downstream source yield and used intervals for acquisition feedback."""
    if not catalog_path or not catalog_path.exists():
        return {}
    connection = sqlite3.connect(f"file:{catalog_path}?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    guidance: dict[str, dict[str, Any]] = {}
    rows = connection.execute(
        """SELECT r.video_id,r.clip_start,r.record_json,
        COALESCE(m.accepted,0) AS m2d_accepted,
        CASE WHEN a.sha256 IS NULL THEN 0 ELSE 1 END AS final_accepted
        FROM records r LEFT JOIN m2d_scores m USING(filename)
        LEFT JOIN accepted a USING(sha256) WHERE r.platform=?""",
        (platform,),
    )
    for row in rows:
        video_id = str(row["video_id"])
        item = guidance.setdefault(
            video_id,
            {
                "record": json.loads(row["record_json"]),
                "scored": 0,
                "m2d_accepted": 0,
                "accepted": 0,
                "attempted_starts": [],
                "accepted_starts": [],
            },
        )
        item["scored"] += 1
        item["m2d_accepted"] += int(row["m2d_accepted"])
        item["accepted"] += int(row["final_accepted"])
        item["attempted_starts"].append(float(row["clip_start"]))
        if row["final_accepted"]:
            item["accepted_starts"].append(float(row["clip_start"]))
    connection.close()
    return guidance


def _load_catalog_content_priors(
    catalog_path: Path | None, *, platform: str
) -> dict[str, dict[str, float]]:
    """Learn end-to-end acceptance by uploader/query, including ASR failures."""
    empty = {"uploader": {}, "query": {}, "global": {"acceptance": 0.5}}
    if not catalog_path or not catalog_path.exists():
        return empty
    connection = sqlite3.connect(f"file:{catalog_path}?mode=ro", uri=True, timeout=30)
    rows = connection.execute(
        """SELECT r.record_json,CASE WHEN a.sha256 IS NULL THEN 0 ELSE 1 END
        FROM records r LEFT JOIN accepted a USING(sha256)
        WHERE r.platform=? AND json_extract(r.record_json,'$.selection')=
        'whole_source_proxy_scan'""",
        (platform,),
    )
    samples: list[tuple[dict[str, Any], int]] = []
    for record_json, accepted in rows:
        samples.append((json.loads(record_json), int(accepted)))
    connection.close()
    if not samples:
        return empty
    global_rate = sum(accepted for _, accepted in samples) / len(samples)
    aggregates: dict[str, dict[str, list[float]]] = {
        "uploader": {},
        "query": {},
    }
    for item, accepted in samples:
        for family, value in (
            ("uploader", item.get("uploader")),
            ("query", item.get("search_query")),
        ):
            key = str(value or "").strip().lower()
            if not key:
                continue
            aggregate = aggregates[family].setdefault(key, [0.0, 0.0])
            aggregate[0] += 1.0
            aggregate[1] += accepted
    prior_strength = 10.0
    result = {
        family: {
            key: (accepted + prior_strength * global_rate)
            / (count + prior_strength)
            for key, (count, accepted) in values.items()
        }
        for family, values in aggregates.items()
    }
    result["global"] = {"acceptance": global_rate}
    return result


def _inject_productive_catalog_sources(
    candidates: list[dict[str, Any]],
    guidance: dict[str, dict[str, Any]],
    *,
    clips_per_video: int,
    source_content_minutes_per_hour: float | None,
    max_clips_per_video: int,
) -> list[dict[str, Any]]:
    """Keep proven sources available even when a new search page omits them."""
    known = {str(item["video_id"]) for item in candidates}
    augmented = list(candidates)
    for video_id, stats in guidance.items():
        if video_id in known or int(stats["accepted"]) < 1:
            continue
        record = dict(stats["record"])
        duration = float(record.get("duration_seconds") or 0.0)
        budget = (
            source_clip_budget(
                duration,
                clip_seconds=CLIP_SECONDS,
                base_clips=clips_per_video,
                content_minutes_per_hour=source_content_minutes_per_hour,
                max_clips=max_clips_per_video,
            )
            if source_content_minutes_per_hour is not None
            else clips_per_video
        )
        if int(stats["accepted"]) >= budget or not _candidate_allowed(
            record, profile="cinematic"
        ):
            continue
        augmented.append(
            {
                **record,
                "candidate_id": f"{video_id}:catalog-guided",
                "video_id": video_id,
                "clip_start_seconds": 0.0,
                "clip_end_seconds": CLIP_SECONDS,
                "source_clip_budget": budget,
                "selection": "catalog_guided_source_reuse",
            }
        )
    return augmented


def _order_scanned_source_groups(
    groups: list[list[dict[str, Any]]],
    guidance: dict[str, dict[str, Any]],
    scan_priors: dict[str, dict[str, float]] | None = None,
    content_priors: dict[str, dict[str, float]] | None = None,
) -> list[list[dict[str, Any]]]:
    """Use 70% proven-source exploitation while retaining 30% exploration."""

    scan_priors = scan_priors or {"uploader": {}, "query": {}}
    content_priors = content_priors or {
        "uploader": {},
        "query": {},
        "global": {"acceptance": 0.5},
    }

    def score(group: list[dict[str, Any]]) -> tuple[float, float, float]:
        item = group[0]
        stats = guidance.get(str(item["video_id"]), {})
        scored = int(stats.get("scored", 0))
        accepted = int(stats.get("accepted", 0))
        posterior = (accepted + 1.0) / (scored + 10.0)
        duration = float(item.get("duration_seconds") or 0.0)
        return (
            posterior,
            float(_cinematic_candidate_priority(item)),
            min(duration, MAX_SOURCE_DURATION_SECONDS),
        )

    def exploration_score(
        group: list[dict[str, Any]],
    ) -> tuple[float, float, float]:
        item = group[0]
        predictions = [
            scan_priors.get("uploader", {}).get(
                str(item.get("uploader") or "").strip().lower()
            ),
            scan_priors.get("query", {}).get(
                str(item.get("search_query") or "").strip().lower()
            ),
        ]
        known = [float(value) for value in predictions if value is not None]
        predicted_regions = sum(known) / len(known) if known else 0.0
        acceptance_predictions = [
            content_priors.get("uploader", {}).get(
                str(item.get("uploader") or "").strip().lower()
            ),
            content_priors.get("query", {}).get(
                str(item.get("search_query") or "").strip().lower()
            ),
        ]
        known_acceptance = [
            float(value) for value in acceptance_predictions if value is not None
        ]
        predicted_acceptance = (
            sum(known_acceptance) / len(known_acceptance)
            if known_acceptance
            else float(content_priors["global"]["acceptance"])
        )
        return (
            predicted_regions * predicted_acceptance,
            float(_cinematic_candidate_priority(item)),
            min(
                float(item.get("duration_seconds") or 0.0),
                MAX_SOURCE_DURATION_SECONDS,
            ),
        )

    proven: list[list[dict[str, Any]]] = []
    exploration: list[list[dict[str, Any]]] = []
    for group in groups:
        stats = guidance.get(str(group[0]["video_id"]), {})
        scored = int(stats.get("scored", 0))
        accepted = int(stats.get("accepted", 0))
        destination = (
            proven
            if accepted > 0 and accepted / max(1, scored) >= 0.25
            else exploration
        )
        destination.append(group)
    proven.sort(key=score, reverse=True)
    exploration.sort(key=exploration_score, reverse=True)
    ordered: list[list[dict[str, Any]]] = []
    while proven or exploration:
        for _ in range(7):
            if proven:
                ordered.append(proven.pop(0))
        for _ in range(3):
            if exploration:
                ordered.append(exploration.pop(0))
        if not proven and exploration and len(exploration) < 3:
            ordered.extend(exploration)
            exploration.clear()
        if not exploration and proven and len(proven) < 7:
            ordered.extend(proven)
            proven.clear()
    return ordered


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _scan_cache_path(cache_dir: Path, candidate: dict[str, Any]) -> Path:
    platform = re.sub(r"[^a-z0-9_-]+", "-", str(candidate.get("source_platform")))
    video_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(candidate["video_id"]))
    return cache_dir / f"{platform}-{video_id}.json"


def _load_source_scan_priors(
    cache_dir: Path, candidates: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    """Learn uploader/query productivity from completed whole-source scans."""
    metadata = {str(item["video_id"]): item for item in candidates}
    aggregates: dict[str, dict[str, list[float]]] = {
        "uploader": {},
        "query": {},
    }
    global_sources = 0
    global_regions = 0.0
    samples: list[tuple[dict[str, Any], float]] = []
    for path in cache_dir.glob("*.json"):
        try:
            scan = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        item = scan.get("source_metadata") or metadata.get(str(scan.get("video_id")))
        if not item:
            continue
        # Cap one unusually dense source so it cannot dominate a whole uploader.
        reward = float(min(10, len(scan.get("regions") or [])))
        global_sources += 1
        global_regions += reward
        samples.append((item, reward))
    global_mean = global_regions / global_sources if global_sources else 0.0
    prior_strength = 5.0
    for item, reward in samples:
        for family, value in (
            ("uploader", item.get("uploader")),
            ("query", item.get("search_query")),
        ):
            key = str(value or "").strip().lower()
            if not key:
                continue
            aggregate = aggregates[family].setdefault(key, [0.0, 0.0])
            aggregate[0] += 1.0
            aggregate[1] += reward
    return {
        family: {
            key: (total_reward + prior_strength * global_mean)
            / (source_count + prior_strength)
            for key, (source_count, total_reward) in values.items()
        }
        for family, values in aggregates.items()
    }


def _download_full_source_for_scan(
    candidate: dict[str, Any], root: Path
) -> tuple[Path, dict[str, Any]]:
    duration = float(candidate.get("duration_seconds") or 0.0)
    response = _run(
        [
            YTDLP_PYTHON,
            "-m",
            "yt_dlp",
            "--no-update",
            "--quiet",
            "--no-warnings",
            "--no-playlist",
            "--socket-timeout",
            "30",
            "--retries",
            "3",
            "--extractor-retries",
            "3",
            "-f",
            DAILYMOTION_SCAN_PROXY_SELECTOR,
            "--print-json",
            "-o",
            str(root / "source.%(ext)s"),
            str(candidate["source_url"]),
        ],
        timeout=max(900.0, min(3600.0, 600.0 + duration / 10.0)),
    )
    return _source_file(root), _download_json(response.stdout)


def _preflight_source_for_scan(candidate: dict[str, Any]) -> dict[str, Any]:
    """Require a high-quality extraction variant before downloading a proxy."""
    response = _run(
        [
            YTDLP_PYTHON,
            "-m",
            "yt_dlp",
            "--no-update",
            "--quiet",
            "--no-warnings",
            "--no-playlist",
            "--skip-download",
            "-f",
            DAILYMOTION_SOURCE_SCAN_SELECTOR,
            "--print-json",
            str(candidate["source_url"]),
        ],
        timeout=90,
    )
    return _download_json(response.stdout)


def _download_scanned_sections(
    candidate: dict[str, Any],
    regions: list[dict[str, Any]],
    root: Path,
) -> dict[float, tuple[Path, dict[str, Any]]]:
    """Fetch only selected high-quality excerpts after proxy scanning."""
    command = [
        YTDLP_PYTHON,
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
    section_starts: list[float] = []
    for region in regions:
        start = float(region["start_seconds"])
        section_start = max(0.0, start - 5.0)
        section_end = start + CLIP_SECONDS + 5.0
        section_starts.append(round(section_start, 3))
        command.extend(
            ["--download-sections", f"*{section_start:.3f}-{section_end:.3f}"]
        )
    command.extend(
        [
            "-f",
            DAILYMOTION_SOURCE_SCAN_SELECTOR,
            "--print-json",
            "-o",
            str(root / "selected-%(section_start)012.3f.%(ext)s"),
            str(candidate["source_url"]),
        ]
    )
    response = _run(command, timeout=90 + 30 * len(regions))
    info_by_start = {
        round(float(info["section_start"]), 3): info
        for info in _download_jsons(response.stdout)
    }
    selected: dict[float, tuple[Path, dict[str, Any]]] = {}
    for region, section_start in zip(regions, section_starts, strict=True):
        matches = list(root.glob(f"selected-{section_start:012.3f}.*"))
        if len(matches) != 1 or section_start not in info_by_start:
            raise ValueError(
                f"Expected one selected section at {section_start}, found {matches}"
            )
        selected[float(region["start_seconds"])] = (
            matches[0],
            info_by_start[section_start],
        )
    return selected


def _scan_region_available(
    start: float,
    *,
    attempted_starts: list[float],
    accepted_starts: list[float],
    claimed_starts: list[float],
) -> bool:
    return (
        not any(abs(start - value) < 0.5 for value in attempted_starts)
        and not any(abs(start - value) < CLIP_SECONDS for value in accepted_starts)
        and not any(abs(start - value) < 0.5 for value in claimed_starts)
    )


def _remaining_scan_source_budget(
    source_budget: int,
    *,
    accepted_count: int,
    accepted_starts: list[float],
    claimed_starts: list[float],
) -> int:
    """Count accepted and in-flight claims once when enforcing source diversity."""
    unmatched_claims = sum(
        not any(abs(claimed - accepted) < 0.5 for accepted in accepted_starts)
        for claimed in claimed_starts
    )
    return max(0, source_budget - accepted_count - unmatched_claims)


def _scan_group_has_remaining_work(
    group: list[dict[str, Any]],
    *,
    cache_dir: Path,
    guidance: dict[str, dict[str, Any]],
) -> bool:
    """Drop globally exhausted scan groups before they consume worker slots."""
    from .source_scanner import load_cached_scan

    base = group[0]
    video_id = str(base["video_id"])
    stats = guidance.get(video_id, {})
    attempted_starts = [float(value) for value in stats.get("attempted_starts", [])]
    accepted_starts = [float(value) for value in stats.get("accepted_starts", [])]
    accepted_count = int(stats.get("accepted", 0))
    source_budget = int(base.get("source_clip_budget") or len(group))
    cached = load_cached_scan(
        _scan_cache_path(cache_dir, base), clip_seconds=CLIP_SECONDS
    )
    if cached is None:
        return accepted_count < source_budget
    claimed_starts = [
        float(value) for value in cached.get("claimed_starts", [])
    ]
    if (
        _remaining_scan_source_budget(
            source_budget,
            accepted_count=accepted_count,
            accepted_starts=accepted_starts,
            claimed_starts=claimed_starts,
        )
        <= 0
    ):
        return False
    return any(
        _scan_region_available(
            float(region["start_seconds"]),
            attempted_starts=attempted_starts,
            accepted_starts=accepted_starts,
            claimed_starts=claimed_starts,
        )
        for region in cached.get("regions", [])
    )


def acquire_scanned_source_group(
    candidates: list[dict[str, Any]],
    output_dir: Path,
    *,
    scanner: Any,
    cache_dir: Path,
    guidance: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Scan one full source first, then extract only passing stereo regions."""
    from .source_scanner import SCAN_POLICY_VERSION, load_cached_scan

    base = candidates[0]
    video_id = str(base["video_id"])
    stats = guidance.get(video_id, {})
    attempted_starts = [float(value) for value in stats.get("attempted_starts", [])]
    accepted_starts = [float(value) for value in stats.get("accepted_starts", [])]
    accepted_count = int(stats.get("accepted", 0))
    source_budget = int(base.get("source_clip_budget") or len(candidates))
    cache_path = _scan_cache_path(cache_dir, base)
    cached = load_cached_scan(cache_path, clip_seconds=CLIP_SECONDS)
    reused_cache = cached is not None
    cached_claims = [
        float(value) for value in (cached or {}).get("claimed_starts", [])
    ]
    remaining_budget = _remaining_scan_source_budget(
        source_budget,
        accepted_count=accepted_count,
        accepted_starts=accepted_starts,
        claimed_starts=cached_claims,
    )
    started = time.perf_counter()

    def status_result(status: str, **details: Any) -> list[dict[str, Any]]:
        return [
            {
                **candidate,
                "attempted_at": _now(),
                "retrieval_status": status,
                "source_scan": {"policy": SCAN_POLICY_VERSION, **details},
                "processing_seconds": round(time.perf_counter() - started, 3),
            }
            for candidate in candidates
        ]

    if remaining_budget <= 0:
        return status_result("source_budget_exhausted")
    try:
        with tempfile.TemporaryDirectory(prefix="sam-source-scan-") as temporary:
            root = Path(temporary)
            source: Path | None = None
            info: dict[str, Any] | None = None
            source_format: dict[str, Any] | None = None
            if cached is None:
                target_info = _preflight_source_for_scan(base)
                download_started = time.perf_counter()
                source, info = _download_full_source_for_scan(base, root)
                download_seconds = time.perf_counter() - download_started
                source_format = _source_format(source, info, "dailymotion-source-scan")
                source_rejections = []
                if int(source_format.get("channels") or 0) != 2:
                    source_rejections.append("source_not_stereo")
                if (
                    int(source_format.get("sample_rate_hz") or 0)
                    < MIN_SOURCE_SAMPLE_RATE
                ):
                    source_rejections.append("source_sample_rate")
                if source_rejections:
                    cached = {
                        "policy": SCAN_POLICY_VERSION,
                        "clip_seconds": CLIP_SECONDS,
                        "video_id": video_id,
                        "source_format": source_format,
                        "extraction_format_id": target_info.get("format_id"),
                        "source_metadata": {
                            "uploader": base.get("uploader"),
                            "search_query": base.get("search_query"),
                            "title": base.get("title"),
                        },
                        "rejection_reasons": source_rejections,
                        "download_seconds": round(download_seconds, 3),
                        "scanned_at": _now(),
                        "claimed_starts": [],
                        "regions": [],
                    }
                else:
                    proxy = root / "proxy.flac"
                    proxy_started = time.perf_counter()
                    scanner.create_proxy(source, proxy)
                    proxy_seconds = time.perf_counter() - proxy_started
                    stereo_metrics = scanner.stereo_metrics(proxy)
                    common = {
                        "policy": SCAN_POLICY_VERSION,
                        "clip_seconds": CLIP_SECONDS,
                        "video_id": video_id,
                        "source_format": source_format,
                        "extraction_format_id": target_info.get("format_id"),
                        "source_metadata": {
                            "uploader": base.get("uploader"),
                            "search_query": base.get("search_query"),
                            "title": base.get("title"),
                        },
                        "source_stereo_metrics": stereo_metrics,
                        "download_seconds": round(download_seconds, 3),
                        "proxy_seconds": round(proxy_seconds, 3),
                        "scanned_at": _now(),
                        "claimed_starts": [],
                    }
                    if stereo_metrics["side_to_total_db"] < MIN_SIDE_TO_TOTAL_DB:
                        cached = {
                            **common,
                            "rejection_reasons": ["source_dual_mono"],
                            "regions": [],
                        }
                    else:
                        cached = {
                            **scanner.scan(
                                proxy,
                                clip_seconds=CLIP_SECONDS,
                                max_regions=max(60, source_budget * 3),
                            ),
                            **common,
                        }
                _atomic_json(cache_path, cached)
            claimed_starts = [
                float(value) for value in cached.get("claimed_starts", [])
            ]
            available = [
                region
                for region in cached.get("regions", [])
                if _scan_region_available(
                    float(region["start_seconds"]),
                    attempted_starts=attempted_starts,
                    accepted_starts=accepted_starts,
                    claimed_starts=claimed_starts,
                )
            ][: min(remaining_budget, MAX_SCAN_REGIONS_PER_ACQUISITION)]
            if not available:
                return status_result(
                    "source_scan_exhausted",
                    cached=cached is not None,
                    passing_regions=len(cached.get("regions", [])),
                    rejection_reasons=cached.get("rejection_reasons", []),
                )
            extraction_download_started = time.perf_counter()
            selected_sections = _download_scanned_sections(base, available, root)
            extraction_download_seconds = (
                time.perf_counter() - extraction_download_started
            )
            selected_starts = [float(region["start_seconds"]) for region in available]
            cached["claimed_starts"] = sorted(set(claimed_starts + selected_starts))
            _atomic_json(cache_path, cached)
            results: list[dict[str, Any]] = []
            for index, region in enumerate(available):
                start = float(region["start_seconds"])
                candidate = {
                    **base,
                    "candidate_id": f"{video_id}:{round(start * 1000)}",
                    "clip_start_seconds": start,
                    "clip_end_seconds": round(start + CLIP_SECONDS, 3),
                    "segment_index": index,
                    "selection": "whole_source_proxy_scan",
                    "source_scan": {
                        "policy": SCAN_POLICY_VERSION,
                        "cached": reused_cache,
                        "score": region["score"],
                        "evidence": region["evidence"],
                        "m2d_windows": cached.get("m2d_windows"),
                        "active_proxy_windows": cached.get("active_proxy_windows"),
                        "scan_seconds": cached.get("scan_seconds"),
                        "download_seconds": cached.get("download_seconds"),
                        "proxy_seconds": cached.get("proxy_seconds"),
                        "extraction_download_seconds": round(
                            extraction_download_seconds, 3
                        ),
                    },
                }
                try:
                    selected_source, selected_info = selected_sections[start]
                    results.append(
                        _normalize_candidate_from_source(
                            candidate,
                            selected_source,
                            selected_info,
                            "dailymotion-source-scan-section",
                            output_dir,
                            root,
                            started=started,
                            source_start_seconds=None,
                        )
                    )
                except Exception as error:
                    results.append(
                        {
                            **candidate,
                            "attempted_at": _now(),
                            "retrieval_status": "unavailable",
                            "error": f"{type(error).__name__}: {error}",
                            "processing_seconds": round(
                                time.perf_counter() - started, 3
                            ),
                        }
                    )
            return results
    except Exception as error:
        return status_result(
            "source_scan_unavailable", error=f"{type(error).__name__}: {error}"
        )


def _runtime_worker_limit(path: Path | None, *, maximum: int, default: int) -> int:
    """Read an autoscaler limit without making acquisition depend on it."""
    if not path:
        return max(1, min(maximum, default))
    try:
        payload = json.loads(path.read_text())
        value = payload.get("download_concurrency")
        if value is None:
            value = payload.get("limits", {}).get("download_concurrency")
        return max(1, min(maximum, int(value)))
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        return max(1, min(maximum, default))


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
        if (
            item.get("retrieval_status") == "success"
            and not (output_dir / str(item.get("local_path"))).exists()
        ):
            continue
        attempts.append(item)
        attempted.add(_candidate_key(item))
    return attempts, attempted


def _accepted(
    attempts: list[dict[str, Any]], *, profile: str = "general"
) -> list[dict[str, Any]]:
    return [
        item
        for item in attempts
        if item.get("retrieval_status") == "success"
        and _candidate_allowed(item, profile=profile)
    ]


def _criteria(
    *,
    profile: str = "general",
    clips_per_video: int = 1,
    source_content_minutes_per_hour: float | None = None,
    max_clips_per_video: int = DEFAULT_MAX_CLIPS_PER_SOURCE,
    source: str = "youtube",
) -> dict[str, Any]:
    criteria = {
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
    if source_content_minutes_per_hour is not None:
        criteria["source_diversity"] = source_diversity_policy(
            clip_seconds=CLIP_SECONDS,
            base_clips=clips_per_video,
            content_minutes_per_hour=source_content_minutes_per_hour,
            max_clips=max_clips_per_video,
        )
    return criteria


def write_manifest(
    output_dir: Path,
    attempts: list[dict[str, Any]],
    *,
    target: int,
    seed: int,
    profile: str = "general",
    clips_per_video: int = 1,
    source_content_minutes_per_hour: float | None = None,
    max_clips_per_video: int = DEFAULT_MAX_CLIPS_PER_SOURCE,
    source: str = "youtube",
) -> Path:
    records = _accepted(attempts, profile=profile)[:target]
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
            profile=profile,
            clips_per_video=clips_per_video,
            source_content_minutes_per_hour=source_content_minutes_per_hour,
            max_clips_per_video=max_clips_per_video,
            source=source,
        ),
        "attempt_statuses": status_counts,
        "metadata_excluded_success_count": sum(
            item.get("retrieval_status") == "success"
            and not _candidate_allowed(item, profile=profile)
            for item in attempts
        ),
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
    diversity = criteria.get("source_diversity") or {}
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
        source_budget = clips_per_video
        if diversity.get("policy") == "duration_scaled_source_budget_v1":
            source_budget = record_source_clip_budget(
                record,
                clip_seconds=float(diversity.get("clip_seconds") or CLIP_SECONDS),
                base_clips=int(diversity["base_clips_per_source"]),
                content_minutes_per_hour=float(
                    diversity["content_minutes_per_source_hour"]
                ),
                max_clips=int(diversity["maximum_clips_per_source"]),
            )
        if video_counts[video_id] > source_budget:
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
    source_content_minutes_per_hour: float | None = None,
    max_clips_per_video: int = DEFAULT_MAX_CLIPS_PER_SOURCE,
    source: str = "youtube",
    worker_limit_file: Path | None = None,
    catalog_path: Path | None = None,
    source_scan_config: dict[str, Any] | None = None,
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
        source_content_minutes_per_hour=source_content_minutes_per_hour,
        max_clips_per_video=max_clips_per_video,
        source=source,
    )
    guidance = load_catalog_source_guidance(catalog_path, platform=source)
    content_priors = _load_catalog_content_priors(catalog_path, platform=source)
    scanner = None
    scan_cache_dir: Path | None = None
    if source_scan_config is not None:
        from .source_scanner import M2DSourceScanner

        candidates = _inject_productive_catalog_sources(
            candidates,
            guidance,
            clips_per_video=clips_per_video,
            source_content_minutes_per_hour=source_content_minutes_per_hour,
            max_clips_per_video=max_clips_per_video,
        )
        scan_cache_dir = Path(source_scan_config["cache_dir"])
        scanner = M2DSourceScanner(
            m2d_repo=Path(source_scan_config["m2d_repo"]),
            checkpoint=Path(source_scan_config["checkpoint"]),
            class_labels=Path(source_scan_config["class_labels"]),
            ontology=Path(source_scan_config["ontology"]),
            device=str(source_scan_config.get("device", "cuda")),
            batch_size=int(source_scan_config.get("batch_size", 128)),
        )
    attempts_path = output_dir / "attempts.jsonl"
    attempts, attempted = _load_attempts(attempts_path, output_dir)
    accepted_count = len(_accepted(attempts, profile=profile))
    pending = [item for item in candidates if _candidate_key(item) not in attempted]
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
    if scanner is not None:
        scan_priors = _load_source_scan_priors(scan_cache_dir, candidates)
        work_groups = [
            group
            for group in work_groups
            if _scan_group_has_remaining_work(
                group,
                cache_dir=scan_cache_dir,
                guidance=guidance,
            )
        ]
        work_groups = _order_scanned_source_groups(
            work_groups,
            guidance,
            scan_priors=scan_priors,
            content_priors=content_priors,
        )
    elif profile == "cinematic":
        work_groups.sort(
            key=lambda group: _cinematic_candidate_priority(group[0]), reverse=True
        )
    limited_groups: list[list[dict[str, Any]]] = []
    remaining_attempt_capacity = max_attempts
    for group in work_groups:
        if remaining_attempt_capacity <= 0:
            break
        selected = group[:remaining_attempt_capacity]
        if selected:
            limited_groups.append(selected)
            remaining_attempt_capacity -= len(selected)
    attempts_made = 0
    last_manifest_write = 0.0
    with attempts_path.open("a", encoding="utf-8") as attempt_log:
        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            group_iterator = iter(limited_groups)
            in_flight: dict[
                Future[list[dict[str, Any]]], list[dict[str, Any]]
            ] = {}

            def submit_group(group: list[dict[str, Any]]) -> None:
                if scanner is not None and scan_cache_dir is not None:
                    future = executor.submit(
                        acquire_scanned_source_group,
                        group,
                        output_dir,
                        scanner=scanner,
                        cache_dir=scan_cache_dir,
                        guidance=guidance,
                    )
                else:
                    future = executor.submit(
                        acquire_candidate_group,
                        group,
                        output_dir,
                        youtube_client=youtube_client,
                    )
                in_flight[future] = group

            def submit_next() -> bool:
                try:
                    group = next(group_iterator)
                except StopIteration:
                    return False
                submit_group(group)
                return True

            def fill_available_slots() -> None:
                limit = _runtime_worker_limit(
                    worker_limit_file,
                    maximum=download_workers,
                    default=download_workers,
                )
                while accepted_count < total and len(in_flight) < limit:
                    if not submit_next():
                        break

            fill_available_slots()

            while in_flight:
                completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in completed:
                    group = in_flight.pop(future)
                    future_results = future.result()
                    successful_results = 0
                    for result in future_results:
                        attempts_made += 1
                        if (
                            result.get("retrieval_status") == "success"
                            and accepted_count >= total
                        ):
                            (output_dir / str(result["local_path"])).unlink(
                                missing_ok=True
                            )
                            result["retrieval_status"] = "surplus"
                        if result.get("retrieval_status") == "success":
                            accepted_count += 1
                            successful_results += 1
                        attempts.append(result)
                        attempt_log.write(json.dumps(result) + "\n")
                    attempt_log.flush()
                    checkpoint_time = time.monotonic()
                    if checkpoint_time - last_manifest_write >= 1.0:
                        write_manifest(
                            output_dir,
                            attempts,
                            target=total,
                            seed=seed,
                            profile=profile,
                            clips_per_video=clips_per_video,
                            source_content_minutes_per_hour=(
                                source_content_minutes_per_hour
                            ),
                            max_clips_per_video=max_clips_per_video,
                            source=source,
                        )
                        last_manifest_write = checkpoint_time
                    logger.info(
                        "Accepted %d/%d after %d new attempts",
                        accepted_count,
                        total,
                        attempts_made,
                    )
                    if (
                        scanner is not None
                        and successful_results > 0
                        and accepted_count < total
                        and attempts_made < max_attempts
                        and len(in_flight)
                        < _runtime_worker_limit(
                            worker_limit_file,
                            maximum=download_workers,
                            default=download_workers,
                        )
                    ):
                        submit_group(group)
                fill_available_slots()
    manifest = write_manifest(
        output_dir,
        attempts,
        target=total,
        seed=seed,
        profile=profile,
        clips_per_video=clips_per_video,
        source_content_minutes_per_hour=source_content_minutes_per_hour,
        max_clips_per_video=max_clips_per_video,
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
    global CLIP_SECONDS, YTDLP_PYTHON
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--query-count", type=int)
    parser.add_argument("--results-per-query", type=int, default=12)
    parser.add_argument("--search-workers", type=int, default=8)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--worker-limit-file", type=Path)
    parser.add_argument("--catalog", type=Path)
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
    parser.add_argument("--source-content-minutes-per-hour", type=float)
    parser.add_argument(
        "--max-clips-per-video",
        type=int,
        default=DEFAULT_MAX_CLIPS_PER_SOURCE,
        help="Absolute guardrail for duration-scaled source budgets",
    )
    parser.add_argument(
        "--youtube-client",
        choices=("auto", "default", "mweb", "android"),
        default="auto",
    )
    parser.add_argument("--scan-before-extract", action="store_true")
    parser.add_argument("--source-scan-cache", type=Path)
    parser.add_argument("--m2d-repo", type=Path)
    parser.add_argument("--m2d-checkpoint", type=Path)
    parser.add_argument("--m2d-class-labels", type=Path)
    parser.add_argument("--m2d-ontology", type=Path)
    parser.add_argument("--m2d-device", default="cuda")
    parser.add_argument("--m2d-batch-size", type=int, default=128)
    parser.add_argument("--yt-dlp-python", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.total < 1:
        parser.error("--total must be positive")
    if args.clip_seconds <= 0:
        parser.error("--clip-seconds must be positive")
    CLIP_SECONDS = float(args.clip_seconds)
    if args.yt_dlp_python:
        YTDLP_PYTHON = str(args.yt_dlp_python)
    clips_per_video = args.clips_per_video or (3 if args.profile == "cinematic" else 1)
    if clips_per_video < 1:
        parser.error("--clips-per-video must be positive")
    if (
        args.source_content_minutes_per_hour is not None
        and args.source_content_minutes_per_hour <= 0
    ):
        parser.error("--source-content-minutes-per-hour must be positive")
    if args.max_clips_per_video < clips_per_video:
        parser.error("--max-clips-per-video must be at least --clips-per-video")
    scan_paths = (
        args.source_scan_cache,
        args.m2d_repo,
        args.m2d_checkpoint,
        args.m2d_class_labels,
        args.m2d_ontology,
    )
    if args.scan_before_extract and any(path is None for path in scan_paths):
        parser.error(
            "--scan-before-extract requires --source-scan-cache, --m2d-repo, "
            "--m2d-checkpoint, --m2d-class-labels, and --m2d-ontology"
        )
    if args.m2d_batch_size < 1:
        parser.error("--m2d-batch-size must be positive")
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
        source_content_minutes_per_hour=args.source_content_minutes_per_hour,
        max_clips_per_video=args.max_clips_per_video,
        source=args.source,
        worker_limit_file=args.worker_limit_file,
        catalog_path=args.catalog,
        source_scan_config=(
            {
                "cache_dir": args.source_scan_cache,
                "m2d_repo": args.m2d_repo,
                "checkpoint": args.m2d_checkpoint,
                "class_labels": args.m2d_class_labels,
                "ontology": args.m2d_ontology,
                "device": args.m2d_device,
                "batch_size": args.m2d_batch_size,
            }
            if args.scan_before_extract
            else None
        ),
    )
    print(path)


if __name__ == "__main__":
    main()
