"""Build a resumable, quality-gated dataset from general YouTube search."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import math
import os
import random
import re
import secrets
import shutil
import signal
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
from .remote_media import command_for_media_worker
from .source_diversity import (
    DEFAULT_MAX_CLIPS_PER_SOURCE,
    record_source_clip_budget,
    source_clip_budget,
    source_diversity_policy,
)

logger = logging.getLogger(__name__)

CLIP_SECONDS = 10.0
YTDLP_CONCURRENT_FRAGMENTS = max(
    1, int(os.environ.get("SAM_YTDLP_CONCURRENT_FRAGMENTS", "4"))
)
YTDLP_DIRECT_PLATFORMS = frozenset(
    value.strip().lower()
    for value in os.environ.get("SAM_YTDLP_DIRECT_PLATFORMS", "").split(",")
    if value.strip()
)
OUTPUT_SAMPLE_RATE = 48_000
MIN_SOURCE_SAMPLE_RATE = 44_100
MIN_SOURCE_BITRATE_KBPS = 120.0
MIN_SOURCE_DURATION_SECONDS = 30.0
MIN_DISCOVERY_SOURCE_DURATION_SECONDS = 2 * 60.0
MAX_SOURCE_DURATION_SECONDS = 12 * 3600.0
CANDIDATE_DURATION_POLICY = "source_duration_2m_to_12h_v3"
DAILYMOTION_SEARCH_POLICY = "seeded_relevance_pages_1_to_50_gameplay_v5"
MULTI_SOURCE_SEARCH_POLICY = "yt_dlp_native_related_and_channel_v2"
DISCOVERY_EXPANSION_POLICY = "accepted_parent_graph_and_deep_search_v1"
DISCOVERY_EXPANSION_SEEDS_PER_BATCH = 8
SUPPORTED_DISCOVERY_SOURCES = (
    "youtube",
    "dailymotion",
    "vimeo",
    "tiktok",
    "soundcloud",
    "bilibili",
    "internet_archive",
)


def _yt_dlp_transfer_args() -> list[str]:
    """Bound parallel HLS/DASH fragments while leaving direct files unchanged."""
    return ["--concurrent-fragments", str(YTDLP_CONCURRENT_FRAGMENTS)]


YTDLP_SEARCH_PROVIDERS: dict[str, dict[str, Any]] = {
    "soundcloud": {"search_key": "scsearch", "max_results": 20},
    "bilibili": {
        "search_key": "bilisearch",
        "max_results": 8,
        "hydrate": True,
    },
    "vimeo": {
        "search_key": "yvsearch",
        "site": "vimeo.com",
        "max_results": 8,
        "hydrate": True,
    },
    "tiktok": {
        "search_key": "yvsearch",
        "site": "tiktok.com",
        "max_results": 8,
        "hydrate": True,
    },
    "internet_archive": {
        "search_key": "yvsearch",
        "site": "archive.org/details",
        # Archive collection pages can take much longer to inspect than a
        # single-video page. A small sample keeps the provider from delaying
        # the complete round-robin while still continuously adding sources.
        "max_results": 2,
        "hydrate": True,
        "hydrate_timeout": 30,
    },
}
YTDLP_PYTHON = sys.executable
YOUTUBE_PROXY_CONFIG: Path | None = None
SILENCE_THRESHOLD_DBFS = -55.0
MIN_RMS_DBFS = -35.0
MIN_PEAK_DBFS = -20.0
MAX_SILENT_FRACTION = 0.10
MAX_SILENT_RUN_SECONDS = 0.75
MIN_SIDE_TO_TOTAL_DB = -45.0
MAX_CLIPPED_FRACTION = 0.01
MAX_SCAN_REGIONS_PER_ACQUISITION = 8
SOURCE_ASR_PROBE_POLICY = "source_proxy_asr_top3_beam1_v1"
MAX_SOURCE_ASR_PROBES = 3
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
    "gameplay story mode",
    "gameplay walkthrough",
    "RPG gameplay",
    "open world gameplay",
    "adventure game gameplay",
    "horror game gameplay",
    "crime series episode",
    "science fiction series episode",
    "drama series episode",
    "animated episode",
    "web series episode",
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
    "NPC conversation",
    "mission dialogue",
    "combat dialogue",
    "party banter",
    "quest dialogue",
    "radio dialogue",
    "in game dialogue",
    "exploration dialogue",
    "environmental dialogue",
)

CINEMATIC_AUDIO_HINTS = (
    "English HD",
    "English 4K",
    "dialogue HD",
    "soundtrack scene",
    "cinematic sound",
    "English gameplay",
    "English story",
    "English episode",
)

CINEMATIC_BROAD_QUERIES = (
    "gameplay",
    "game walkthrough",
    "story mode",
    "NPC dialogue",
    "mission dialogue",
    "party banter",
    "game cutscene",
    "game movie",
    "full episode",
    "TV series episode",
    "animated episode",
    "web series",
    "movie scene",
    "short film",
)

# These target the exact audiobook-background use case: diegetic conversation
# while gameplay, ambience, effects, and non-vocal score continue underneath.
# They deliberately avoid generic "music" searches, which tend to return songs.
CINEMATIC_GAMEPLAY_SITUATIONS = (
    "open world NPC encounter gameplay",
    "RPG companion banter gameplay",
    "story mission gameplay dialogue",
    "combat mission radio dialogue gameplay",
    "stealth mission radio chatter gameplay",
    "adventure game exploration dialogue",
    "horror game exploration dialogue",
    "gameplay quest conversation",
    "cinematic gameplay full chapter",
    "gameplay longplay story dialogue",
    "in game vehicle dialogue mission",
    "party conversation during gameplay",
    "gameplay environmental dialogue scene",
    "walkthrough cutscene to gameplay transition",
    "gameplay dialogue during combat",
    "gameplay dialogue during exploration",
)

CINEMATIC_GAMEPLAY_AUDIO_CONTEXTS = (
    "ambient soundtrack sound effects",
    "background score environmental sound",
    "cinematic ambience game audio",
    "music underscore combat sounds",
    "environmental ambience soundtrack",
    "diegetic sound background score",
    "game ambience sound effects",
    "cinematic soundscape dialogue",
)

CINEMATIC_TITLE_TERMS = (
    "scene",
    "movie clip",
    "film clip",
    "short film",
    "cutscene",
    "cinematic",
    "gameplay",
    "walkthrough",
    "story mode",
    "game movie",
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
    ("story mode", 7),
    ("game movie", 6),
    ("gameplay", 5),
    ("walkthrough", 4),
    ("npc", 4),
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
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # yt-dlp may have an ffmpeg child which keeps a temporary download open.
        # Killing only yt-dlp lets TemporaryDirectory unlink the path while
        # ffmpeg continues writing to the deleted inode, silently consuming disk.
        stdout, stderr = _terminate_process_group(process)
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=stdout,
            stderr=stderr,
        ) from None
    except BaseException:
        _terminate_process_group(process)
        raise
    completed = subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout=stdout,
        stderr=stderr,
    )
    if completed.returncode:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=stdout,
            stderr=stderr,
        )
    return completed


def _run_search_command(
    command: list[str], *, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run yt-dlp discovery work on the media host when one is configured."""
    wrapped, wrapped_timeout = command_for_media_worker(
        command,
        task="search",
        timeout=timeout,
    )
    return _run(wrapped, timeout=wrapped_timeout)


def _terminate_process_group(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Stop and reap a downloader plus every ffmpeg/ffprobe descendant."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.communicate()


def _protected_proxy_configs(path: Path) -> list[Path]:
    configs = sorted(path.glob("*.conf")) if path.is_dir() else [path]
    configs = [config for config in configs if config.is_file()]
    if not configs:
        raise FileNotFoundError(f"yt-dlp proxy config is empty: {path}")
    for config in configs:
        if config.stat().st_mode & 0o077:
            raise PermissionError(
                f"yt-dlp proxy config must not be group/world accessible: {config}"
            )
    return configs


def _yt_dlp_proxy_args(source_platform: str, affinity_key: str = "") -> list[str]:
    """Select a protected all-provider proxy without credentials in argv."""
    if source_platform.strip().lower() in YTDLP_DIRECT_PLATFORMS:
        return []
    if YOUTUBE_PROXY_CONFIG is None:
        return []
    path = Path(YOUTUBE_PROXY_CONFIG)
    if not path.exists():
        raise FileNotFoundError(f"yt-dlp proxy config is missing: {path}")
    configs = _protected_proxy_configs(path)
    digest = hashlib.sha256(affinity_key.encode()).digest()
    selected = configs[int.from_bytes(digest[:8], "big") % len(configs)]
    return ["--config-locations", str(selected)]


def _candidate_proxy_affinity(candidate: dict[str, Any]) -> str:
    return ":".join(
        (
            str(candidate.get("video_id") or candidate.get("source_url") or "source"),
            str(candidate.get("_youtube_proxy_attempt") or 0),
        )
    )


def _yt_dlp_javascript_args(source_platform: str) -> list[str]:
    if source_platform != "youtube":
        return []
    deno = shutil.which("deno") or str(Path.home() / ".deno" / "bin" / "deno")
    return ["--js-runtimes", f"deno:{deno}"] if Path(deno).is_file() else []


def _yt_dlp_youtube_client_args(source_platform: str, client: str = "tv") -> list[str]:
    """Use the client that succeeds most often on authenticated proxy exits."""
    if source_platform != "youtube":
        return []
    return ["--extractor-args", f"youtube:player_client={client}"]


def _redact_proxy_credentials(value: str) -> str:
    return re.sub(
        r"(https?://)[^\s/:@]+:[^\s/@]+@",
        r"\1***:***@",
        value,
    )


def _exception_text(error: Exception) -> str:
    summary = f"{type(error).__name__}: {error}"
    if not isinstance(error, subprocess.CalledProcessError):
        return summary
    output = _redact_proxy_credentials(
        "\n".join(
            part.strip()
            for part in (error.stdout, error.stderr)
            if part and part.strip()
        )
    )
    if len(output) > 2_000:
        output = output[-2_000:]
    return f"{summary}: {output}" if output else summary


def _permanent_media_error(error: subprocess.CalledProcessError) -> bool:
    message = f"{error.stdout or ''}\n{error.stderr or ''}".lower()
    return any(
        marker in message
        for marker in (
            "not found",
            "video has been deleted",
            "private video",
            "private content",
            "no video formats found",
            "no longer available",
            "this video is unavailable",
        )
    )


def build_query_specs(
    seed: int, count: int, *, profile: str = "general"
) -> list[dict[str, str]]:
    """Create reproducible queries with an attributable query family."""
    generator = random.Random(seed)
    specs: list[dict[str, str]] = []
    seen: set[str] = set()
    while len(specs) < count:
        family = "general_v1"
        if profile == "cinematic":
            lane = generator.random()
            if lane < 0.30:
                query = " ".join(
                    (
                        generator.choice(CINEMATIC_BROAD_QUERIES),
                        generator.choice(("English", "HD", "dialogue", "story")),
                        CINEMATIC_SEARCH_EXCLUSIONS,
                    )
                )
                family = "cinematic_broad_v1"
            elif lane < 0.65:
                query = " ".join(
                    (
                        generator.choice(CINEMATIC_GAMEPLAY_SITUATIONS),
                        generator.choice(CINEMATIC_GAMEPLAY_AUDIO_CONTEXTS),
                        "English HD",
                        CINEMATIC_SEARCH_EXCLUSIONS,
                    )
                )
                family = "cinematic_gameplay_context_v2"
            else:
                query = " ".join(
                    (
                        generator.choice(CINEMATIC_SOURCES),
                        generator.choice(CINEMATIC_SCENES),
                        generator.choice(CINEMATIC_AUDIO_HINTS),
                        "English",
                        CINEMATIC_SEARCH_EXCLUSIONS,
                    )
                )
                family = "cinematic_composed_v1"
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
        specs.append({"query": query, "family": family})
    return specs


def build_queries(seed: int, count: int, *, profile: str = "general") -> list[str]:
    """Create reproducible YouTube queries for a general or cinematic mix."""
    return [spec["query"] for spec in build_query_specs(seed, count, profile=profile)]


def _query_for_source(query: str, source: str) -> str:
    """Remove YouTube-only negative tokens for other provider searches."""
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
            or item.get("source_platform") in YTDLP_SEARCH_PROVIDERS
            or any(term in title for term in CINEMATIC_TITLE_TERMS)
        )
    )


def _discovery_candidate_allowed(
    item: dict[str, Any], *, profile: str, source: str
) -> bool:
    duration = float(item.get("duration") or item.get("duration_seconds") or 0.0)
    return _candidate_allowed(item, profile=profile) and (
        source != "dailymotion" or duration >= MIN_DISCOVERY_SOURCE_DURATION_SECONDS
    )


def _cinematic_candidate_priority(item: dict[str, Any]) -> int:
    """Rank explicit cinematic source markers without inspecting the speaker."""
    title = f" {str(item.get('title') or '').lower()} "
    return sum(weight for term, weight in CINEMATIC_PRIORITY_WEIGHTS if term in title)


def _search_youtube(query: str, results: int, profile: str) -> list[dict[str, Any]]:
    response = _run_search_command(
        [
            YTDLP_PYTHON,
            "-m",
            "yt_dlp",
            "--no-update",
            "--quiet",
            "--no-warnings",
            *_yt_dlp_proxy_args("youtube", query),
            *_yt_dlp_javascript_args("youtube"),
            *_yt_dlp_youtube_client_args("youtube"),
            "--flat-playlist",
            "--dump-single-json",
            f"ytsearch{results}:{query}",
        ],
        timeout=120,
    )
    payload = json.loads(response.stdout)
    return [
        item
        for item in payload.get("entries", [])
        if _discovery_candidate_allowed(item, profile=profile, source="youtube")
    ]


def _search_result_url(item: dict[str, Any]) -> str | None:
    for key in ("webpage_url", "original_url", "source_url", "url"):
        value = str(item.get(key) or "").strip()
        if value.startswith(("http://", "https://")):
            return value
    return None


def _hydrate_search_result(
    url: str,
    *,
    source_platform: str,
    affinity_key: str,
    timeout: float = 90,
) -> dict[str, Any]:
    response = _run_search_command(
        [
            YTDLP_PYTHON,
            "-m",
            "yt_dlp",
            "--no-update",
            "--quiet",
            "--no-warnings",
            "--no-playlist",
            "--skip-download",
            *_yt_dlp_proxy_args(source_platform, affinity_key),
            "--dump-single-json",
            url,
        ],
        timeout=timeout,
    )
    payload = json.loads(response.stdout)
    if not isinstance(payload, dict):
        raise ValueError("yt-dlp metadata response was not an object")
    return payload


def _search_ytdlp_provider(
    query: str,
    results: int,
    profile: str,
    source: str,
) -> list[dict[str, Any]]:
    try:
        provider = YTDLP_SEARCH_PROVIDERS[source]
    except KeyError as error:
        raise ValueError(f"Unsupported discovery source: {source}") from error
    limited_results = min(results, int(provider["max_results"]))
    provider_query = _query_for_source(query, source)
    if provider.get("site"):
        provider_query = f"site:{provider['site']} {provider_query}"
    target = f"{provider['search_key']}{limited_results}:{provider_query}"
    response = _run_search_command(
        [
            YTDLP_PYTHON,
            "-m",
            "yt_dlp",
            "--no-update",
            "--quiet",
            "--no-warnings",
            *_yt_dlp_proxy_args(source, provider_query),
            "--flat-playlist",
            "--dump-single-json",
            target,
        ],
        timeout=90,
    )
    payload = json.loads(response.stdout)
    items: list[dict[str, Any]] = []
    for flat_item in (payload.get("entries") or [])[:limited_results]:
        url = _search_result_url(flat_item)
        if not url:
            continue
        try:
            item = (
                _hydrate_search_result(
                    url,
                    source_platform=source,
                    affinity_key=url,
                    timeout=float(provider.get("hydrate_timeout", 90)),
                )
                if provider.get("hydrate")
                else flat_item
            )
        except Exception:
            logger.debug("Could not hydrate %s search result %s", source, url)
            continue
        item = {
            **item,
            "source_url": _search_result_url(item) or url,
            "source_platform": source,
        }
        if not item.get("id"):
            item["id"] = hashlib.sha256(url.encode()).hexdigest()[:24]
        if _discovery_candidate_allowed(item, profile=profile, source=source):
            items.append(item)
    return items


def _dailymotion_search_page(seed: int, query: str, *, pages: int = 10) -> int:
    """Spread repeated query combinations across Dailymotion result pages."""
    return random.Random(f"{seed}:{query}:dailymotion-page").randrange(1, pages + 1)


def _dailymotion_deep_search_page(seed: int, query: str) -> int:
    """Explore beyond the ten pages covered by the normal search lane."""
    return random.Random(f"{seed}:{query}:dailymotion-deep-page").randrange(11, 51)


def _dailymotion_has_high_quality_format(item: dict[str, Any]) -> bool:
    return any(
        str(value).lower().startswith(("hd", "uhd", "4k"))
        for value in (item.get("available_formats") or [])
    )


def _search_dailymotion(
    query: str, results: int, profile: str, *, page: int = 1
) -> list[dict[str, Any]]:
    fields = (
        "id,title,description,duration,owner,owner.screenname,url,language,tags,"
        "created_time,available_formats"
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
            "uploader_id": value.get("owner"),
            "source_url": value.get("url"),
        }
        if str(item.get("language") or "").lower() not in {"", "en"}:
            continue
        if not _dailymotion_has_high_quality_format(item):
            continue
        if _discovery_candidate_allowed(item, profile=profile, source="dailymotion"):
            items.append(item)
    return items


def _dailymotion_connection(
    path: str,
    *,
    profile: str,
    results: int,
    page: int = 1,
) -> list[dict[str, Any]]:
    """Read a Dailymotion related/channel connection with search-equivalent gates."""
    fields = (
        "id,title,description,duration,owner,owner.screenname,url,language,tags,"
        "created_time,available_formats"
    )
    parameters = urllib.parse.urlencode(
        {"fields": fields, "limit": min(results, 100), "page": max(1, page)}
    )
    request = urllib.request.Request(
        f"https://api.dailymotion.com/{path}?{parameters}",
        headers={"User-Agent": "sam-audio-dataset-builder/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    items: list[dict[str, Any]] = []
    for value in payload.get("list", []):
        item = {
            **value,
            "uploader": value.get("owner.screenname"),
            "uploader_id": value.get("owner"),
            "source_url": value.get("url"),
            "source_platform": "dailymotion",
        }
        if str(item.get("language") or "").lower() not in {"", "en"}:
            continue
        if not _dailymotion_has_high_quality_format(item):
            continue
        if _discovery_candidate_allowed(item, profile=profile, source="dailymotion"):
            items.append(item)
    return items


def _expand_dailymotion_seed(
    seed: int,
    parent: dict[str, Any],
    *,
    profile: str,
    results: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    video_id = str(parent.get("video_id") or "").strip()
    uploader_id = str(
        parent.get("uploader_id") or parent.get("channel_id") or ""
    ).strip()
    if video_id and not uploader_id:
        try:
            parameters = urllib.parse.urlencode({"fields": "owner"})
            request = urllib.request.Request(
                "https://api.dailymotion.com/video/"
                f"{urllib.parse.quote(video_id)}?{parameters}",
                headers={"User-Agent": "sam-audio-dataset-builder/1.0"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                uploader_id = str(json.load(response).get("owner") or "").strip()
        except Exception:
            logger.debug("Could not resolve Dailymotion owner for %s", video_id)
    expanded: list[tuple[str, list[dict[str, Any]]]] = []
    if video_id:
        try:
            expanded.append(
                (
                    "accepted_related_v1",
                    _dailymotion_connection(
                        f"video/{urllib.parse.quote(video_id)}/related",
                        profile=profile,
                        results=max(results, 50),
                    ),
                )
            )
        except Exception:
            logger.debug("Could not expand Dailymotion related for %s", video_id)
    if uploader_id:
        page = random.Random(
            f"{seed}:{video_id}:{uploader_id}:dailymotion-channel"
        ).randrange(1, 6)
        try:
            expanded.append(
                (
                    "accepted_channel_v1",
                    _dailymotion_connection(
                        f"user/{urllib.parse.quote(uploader_id)}/videos",
                        profile=profile,
                        results=max(results, 50),
                        page=page,
                    ),
                )
            )
        except Exception:
            logger.debug(
                "Could not expand Dailymotion channel %s page %d",
                uploader_id,
                page,
            )
    return expanded


def _expand_bilibili_seed(
    parent: dict[str, Any],
    *,
    profile: str,
    results: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Use Bilibili's related graph; same-owner results form the channel lane."""
    video_id = str(parent.get("video_id") or "").strip()
    if not video_id:
        return []
    parameters = urllib.parse.urlencode({"bvid": video_id})
    request = urllib.request.Request(
        f"https://api.bilibili.com/x/web-interface/archive/related?{parameters}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    parent_uploader = str(
        parent.get("uploader_id") or parent.get("channel_id") or ""
    ).strip()
    buckets: dict[str, list[dict[str, Any]]] = {
        "accepted_related_v1": [],
        "accepted_channel_v1": [],
    }
    for value in (payload.get("data") or [])[: max(results * 2, results)]:
        owner = value.get("owner") or {}
        bvid = str(value.get("bvid") or "").strip()
        if not bvid:
            continue
        uploader_id = str(owner.get("mid") or "").strip()
        item = {
            **value,
            "id": bvid,
            "duration": value.get("duration"),
            "uploader": owner.get("name"),
            "uploader_id": uploader_id,
            "channel_id": uploader_id,
            "source_url": f"https://www.bilibili.com/video/{bvid}",
            "source_platform": "bilibili",
        }
        if not _discovery_candidate_allowed(item, profile=profile, source="bilibili"):
            continue
        strategy = (
            "accepted_channel_v1"
            if parent_uploader and uploader_id == parent_uploader
            else "accepted_related_v1"
        )
        if len(buckets[strategy]) < results:
            buckets[strategy].append(item)
    return [(strategy, items) for strategy, items in buckets.items() if items]


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
    if source == "youtube":
        return _search_youtube(query, results, profile)
    return _search_ytdlp_provider(query, results, profile, source)


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
    expansion_seeds: list[dict[str, Any]] | None = None,
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
            and search_metadata.get("discovery_expansion_policy")
            == DISCOVERY_EXPANSION_POLICY
            and (
                source != "dailymotion"
                or search_metadata.get("search_page_policy")
                == DAILYMOTION_SEARCH_POLICY
            )
            and (
                source in {"youtube", "dailymotion"}
                or search_metadata.get("search_provider_policy")
                == MULTI_SOURCE_SEARCH_POLICY
            )
        )
        if compatible:
            filtered = [
                item
                for item in existing
                if _discovery_candidate_allowed(item, profile=profile, source=source)
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

    query_specs = build_query_specs(seed, query_count, profile=profile)
    queries = [spec["query"] for spec in query_specs]
    found: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    discoveries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: dict[Future[Any], dict[str, Any]] = {}
        for spec in query_specs:
            query = spec["query"]
            page = (
                _dailymotion_search_page(seed, query) if source == "dailymotion" else 1
            )
            future = executor.submit(
                _search,
                query,
                results_per_query,
                profile,
                source,
                search_page=page,
            )
            pending[future] = {
                "kind": "search",
                "query": query,
                "family": spec["family"],
                "strategy": (
                    "query_expanded_v2"
                    if spec["family"] == "cinematic_gameplay_context_v2"
                    else "query_v1"
                ),
                "page": page,
            }
            if source == "dailymotion":
                deep_page = _dailymotion_deep_search_page(seed, query)
                future = executor.submit(
                    _search,
                    query,
                    results_per_query,
                    profile,
                    source,
                    search_page=deep_page,
                )
                pending[future] = {
                    "kind": "search",
                    "query": query,
                    "family": spec["family"],
                    "strategy": "deep_page_v1",
                    "page": deep_page,
                }
        productive_seeds = (expansion_seeds or [])[:DISCOVERY_EXPANSION_SEEDS_PER_BATCH]
        for parent in productive_seeds:
            if source == "dailymotion":
                future = executor.submit(
                    _expand_dailymotion_seed,
                    seed,
                    parent,
                    profile=profile,
                    results=results_per_query,
                )
            elif source == "bilibili":
                future = executor.submit(
                    _expand_bilibili_seed,
                    parent,
                    profile=profile,
                    results=results_per_query,
                )
            else:
                continue
            pending[future] = {
                "kind": "expansion",
                "parent": parent,
                "query": str(parent.get("search_query") or ""),
            }
        for index, future in enumerate(as_completed(pending), start=1):
            task = pending[future]
            try:
                result = future.result()
            except Exception as error:
                failures.append(
                    {
                        "query": str(task.get("query") or ""),
                        "strategy": str(task.get("strategy") or task["kind"]),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            if task["kind"] == "expansion":
                parent = task["parent"]
                for strategy, items in result:
                    for item in items:
                        discoveries.append(
                            {
                                "item": item,
                                "query": task["query"],
                                "family": "accepted_parent_graph_v1",
                                "strategy": strategy,
                                "page": None,
                                "parent_video_id": parent.get("video_id"),
                                "parent_uploader": parent.get("uploader"),
                            }
                        )
            else:
                for item in result:
                    discoveries.append({"item": item, **task})
            if index % 25 == 0:
                logger.info(
                    "Discovery tasks %d/%d; %d raw sources",
                    index,
                    len(pending),
                    len(discoveries),
                )
    strategy_order = {
        "accepted_channel_v1": 0,
        "accepted_related_v1": 1,
        "query_expanded_v2": 2,
        "deep_page_v1": 3,
        "query_v1": 4,
    }
    discoveries.sort(
        key=lambda value: (
            strategy_order.get(str(value.get("strategy")), 99),
            str(value["item"].get("id") or ""),
        )
    )
    found_videos: set[str] = set()
    strategy_counts: dict[str, int] = {}
    for discovery in discoveries:
        item = discovery["item"]
        video_id = str(item["id"])
        if video_id in found_videos:
            continue
        found_videos.add(video_id)
        strategy = str(discovery.get("strategy") or "query_v1")
        family = str(discovery.get("family") or "general_v1")
        quality_lane = (
            f"query_family:{family}" if strategy.startswith("query_") else strategy
        )
        quality_key = f"{source}:{quality_lane}"
        strategy_counts[quality_key] = strategy_counts.get(quality_key, 0) + 1
        duration = float(item["duration"])
        starts = _sample_clip_starts(
            seed=seed,
            video_id=video_id,
            duration=duration,
            clips_per_video=clips_per_video,
            source_content_minutes_per_hour=source_content_minutes_per_hour,
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
                    or item.get("webpage_url")
                    or item.get("original_url")
                    or (
                        f"https://www.dailymotion.com/video/{video_id}"
                        if source == "dailymotion"
                        else (
                            f"https://www.youtube.com/watch?v={video_id}"
                            if source == "youtube"
                            else item.get("url")
                        )
                    )
                ),
                "source_platform": source,
                "title": item.get("title"),
                "duration_seconds": duration,
                "uploader": item.get("uploader") or item.get("channel"),
                "uploader_id": item.get("uploader_id"),
                "description": item.get("description"),
                "tags": item.get("tags"),
                "channel_id": item.get("channel_id"),
                "view_count": item.get("view_count"),
                "search_query": discovery.get("query"),
                "discovery_strategy": strategy,
                "discovery_quality_key": quality_key,
                "discovery_query_family": family,
                "discovery_page": discovery.get("page"),
                "discovery_parent_video_id": discovery.get("parent_video_id"),
                "discovery_parent_uploader": discovery.get("parent_uploader"),
                "clip_start_seconds": round(start, 3),
                "clip_end_seconds": round(start + CLIP_SECONDS, 3),
                "segment_index": segment_index,
                "source_clip_budget": source_budget,
                "selection": f"seeded_{profile}_{source}_search",
                "selection_seed": seed,
                "mixture_bias": ["dialogue", "music", "environmental_sfx"],
                "source_audio_rights": (
                    "Underlying media remains subject to its source terms."
                ),
            }
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
                "discovery_expansion_policy": DISCOVERY_EXPANSION_POLICY,
                "search_page_policy": (
                    DAILYMOTION_SEARCH_POLICY if source == "dailymotion" else None
                ),
                "search_provider_policy": (
                    MULTI_SOURCE_SEARCH_POLICY
                    if source not in {"youtube", "dailymotion"}
                    else None
                ),
                "seed": seed,
                "queries": queries,
                "query_specs": query_specs,
                "expansion_seed_count": len(expansion_seeds or []),
                "discovery_strategy_counts": strategy_counts,
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
    if source_platform == "youtube":
        clients = (
            ("tv", "mweb", "default") if youtube_client == "auto" else (youtube_client,)
        )
    else:
        clients = (str(source_platform),)
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
            *_yt_dlp_transfer_args(),
            "--socket-timeout",
            "20",
            "--retries",
            "2",
            "--extractor-retries",
            "2",
        ]
        command.extend(
            _yt_dlp_proxy_args(source_platform, _candidate_proxy_affinity(candidate))
        )
        command.extend(_yt_dlp_javascript_args(source_platform))
        if source_platform == "youtube" and client == "tv":
            command.extend(_yt_dlp_youtube_client_args(source_platform, "tv"))
        elif source_platform == "youtube" and client == "mweb":
            command.extend(["--extractor-args", "youtube:player_client=mweb"])
        elif source_platform == "youtube" and client == "android":
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
                *_yt_dlp_transfer_args(),
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
        CASE WHEN s.filename IS NULL THEN 0 ELSE 1 END AS asr_scored,
        COALESCE(s.accepted,0) AS asr_accepted,
        CASE WHEN a.sha256 IS NULL THEN 0 ELSE 1 END AS final_accepted
        FROM records r LEFT JOIN m2d_scores m USING(filename)
        LEFT JOIN asr_scores s USING(filename)
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
                "asr_scored": 0,
                "asr_accepted": 0,
                "accepted": 0,
                "attempted_starts": [],
                "accepted_starts": [],
            },
        )
        item["scored"] += 1
        item["m2d_accepted"] += int(row["m2d_accepted"])
        item["asr_scored"] += int(row["asr_scored"])
        item["asr_accepted"] += int(row["asr_accepted"])
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
            key: (accepted + prior_strength * global_rate) / (count + prior_strength)
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


def _inject_cached_scan_sources(
    candidates: list[dict[str, Any]],
    cache_dir: Path,
    guidance: dict[str, dict[str, Any]],
    *,
    clips_per_video: int,
    source_content_minutes_per_hour: float | None,
    max_clips_per_video: int,
) -> list[dict[str, Any]]:
    """Make unclaimed passing regions reusable even before a source is accepted."""
    from .source_scanner import region_passes_confidence_gate

    known = {str(item["video_id"]) for item in candidates}
    augmented = list(candidates)
    for path in cache_dir.glob("*.json"):
        try:
            scan = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        video_id = str(scan.get("video_id") or "")
        if (
            not video_id
            or video_id in known
            or _proxy_asr_blocks_extraction(scan)
            or not any(
                region_passes_confidence_gate(region)
                for region in (scan.get("regions") or [])
            )
        ):
            continue
        metadata = scan.get("source_metadata") or {}
        duration = max(
            MIN_SOURCE_DURATION_SECONDS,
            float(scan.get("m2d_windows") or 0.0) + 1.0,
        )
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
        if int(guidance.get(video_id, {}).get("accepted", 0)) >= budget:
            continue
        candidate = {
            "candidate_id": f"{video_id}:scan-cache",
            "video_id": video_id,
            "source_url": f"https://www.dailymotion.com/video/{video_id}",
            "source_platform": "dailymotion",
            "title": metadata.get("title"),
            "duration_seconds": duration,
            "uploader": metadata.get("uploader"),
            "search_query": metadata.get("search_query"),
            "clip_start_seconds": 0.0,
            "clip_end_seconds": CLIP_SECONDS,
            "segment_index": 0,
            "source_clip_budget": budget,
            "selection": "cached_proxy_scan_reuse",
            "mixture_bias": ["dialogue", "music", "environmental_sfx"],
            "source_audio_rights": (
                "Underlying media remains subject to its source terms."
            ),
        }
        if not _candidate_allowed(candidate, profile="cinematic"):
            continue
        augmented.append(candidate)
        known.add(video_id)
    return augmented


def _order_scanned_source_groups(
    groups: list[list[dict[str, Any]]],
    guidance: dict[str, dict[str, Any]],
    scan_priors: dict[str, dict[str, float]] | None = None,
    content_priors: dict[str, dict[str, float]] | None = None,
) -> list[list[dict[str, Any]]]:
    """Use 70% proven-source exploitation while retaining 30% exploration."""

    scan_priors = scan_priors or {"video": {}, "uploader": {}, "query": {}}
    content_priors = content_priors or {
        "uploader": {},
        "query": {},
        "global": {"acceptance": 0.5},
    }

    def score(group: list[dict[str, Any]]) -> tuple[float, float, float, float]:
        item = group[0]
        stats = guidance.get(str(item["video_id"]), {})
        scored = int(stats.get("scored", 0))
        accepted = int(stats.get("accepted", 0))
        posterior = (accepted + 1.0) / (scored + 10.0)
        duration = float(item.get("duration_seconds") or 0.0)
        cached_regions = float(
            scan_priors.get("video", {}).get(str(item["video_id"]), 0.0)
        )
        return (
            posterior,
            cached_regions,
            float(_cinematic_candidate_priority(item)),
            min(duration, MAX_SOURCE_DURATION_SECONDS),
        )

    def exploration_score(
        group: list[dict[str, Any]],
    ) -> tuple[float, float, float]:
        item = group[0]
        cached_regions = scan_priors.get("video", {}).get(str(item["video_id"]))
        if cached_regions is not None:
            predicted_regions = min(
                float(MAX_SCAN_REGIONS_PER_ACQUISITION), float(cached_regions)
            )
        else:
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
        stats = guidance.get(str(item["video_id"]), {})
        scored = int(stats.get("scored", 0))
        accepted = int(stats.get("accepted", 0))
        if scored:
            predicted_acceptance = (accepted + 1.0) / (scored + 10.0)
        else:
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


def _current_proxy_asr(scan: dict[str, Any]) -> dict[str, Any] | None:
    probe = scan.get("proxy_asr") or {}
    if probe.get("policy") != SOURCE_ASR_PROBE_POLICY:
        return None
    return probe


def _proxy_asr_blocks_extraction(scan: dict[str, Any]) -> bool:
    probe = _current_proxy_asr(scan)
    return bool(probe and probe.get("enforced") and probe.get("accepted") is not True)


def _proxy_asr_from_guidance(stats: dict[str, Any]) -> dict[str, Any] | None:
    """Reuse downstream evidence when a source already has final ASR outcomes."""
    asr_scored = int(stats.get("asr_scored", 0))
    asr_accepted = int(stats.get("asr_accepted", 0))
    if asr_accepted > 0:
        return {
            "policy": SOURCE_ASR_PROBE_POLICY,
            "status": "derived_from_catalog",
            "accepted": True,
            "catalog_asr_scored": asr_scored,
            "catalog_asr_accepted": asr_accepted,
            "checked_regions": [],
        }
    if asr_scored >= 2:
        return {
            "policy": SOURCE_ASR_PROBE_POLICY,
            "status": "derived_from_catalog",
            "accepted": False,
            "catalog_asr_scored": asr_scored,
            "catalog_asr_accepted": 0,
            "checked_regions": [],
        }
    return None


def _request_proxy_asr(
    audio_path: Path,
    *,
    video_id: str,
    start_seconds: float,
    request_dir: Path,
    result_dir: Path,
    deadline: float,
) -> dict[str, Any]:
    request_id = secrets.token_hex(16)
    request_path = request_dir / f"{request_id}.json"
    result_path = result_dir / f"{request_id}.json"
    _atomic_json(
        request_path,
        {
            "schema_version": 1,
            "request_id": request_id,
            "policy": SOURCE_ASR_PROBE_POLICY,
            "audio_path": str(audio_path),
            "video_id": video_id,
            "start_seconds": start_seconds,
            "requested_at": _now(),
        },
    )
    try:
        while time.monotonic() < deadline:
            try:
                result = json.loads(result_path.read_text())
                result_path.unlink(missing_ok=True)
                return result
            except FileNotFoundError:
                time.sleep(0.1)
            except json.JSONDecodeError:
                time.sleep(0.05)
        raise TimeoutError(f"Proxy ASR request {request_id} timed out")
    finally:
        request_path.unlink(missing_ok=True)


def _probe_source_proxy_asr(
    proxy: Path,
    regions: list[dict[str, Any]],
    root: Path,
    *,
    video_id: str,
    request_dir: Path,
    result_dir: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Probe top M2D regions until confident English dialogue is confirmed."""
    started = time.perf_counter()
    request_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    checked: list[dict[str, Any]] = []
    for index, region in enumerate(regions[:MAX_SOURCE_ASR_PROBES]):
        region_started = time.perf_counter()
        start = float(region["start_seconds"])
        probe_audio = root / f"asr-probe-{index}.wav"
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{CLIP_SECONDS:.3f}",
                "-i",
                str(proxy),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(probe_audio),
            ],
            timeout=30,
        )
        result = _request_proxy_asr(
            probe_audio,
            video_id=video_id,
            start_seconds=start,
            request_dir=request_dir,
            result_dir=result_dir,
            deadline=deadline,
        )
        checked.append(
            {
                "start_seconds": start,
                "accepted": bool(result.get("accepted")),
                "detected_language": result.get("detected_language"),
                "language_probability": result.get("language_probability"),
                "duration_after_vad_seconds": result.get("duration_after_vad_seconds"),
                "word_count": result.get("word_count"),
                "rejection_reasons": result.get("rejection_reasons", []),
                "error": result.get("error"),
                "processing_seconds": round(time.perf_counter() - region_started, 3),
            }
        )
        if result.get("accepted"):
            return {
                "policy": SOURCE_ASR_PROBE_POLICY,
                "status": "completed",
                "accepted": True,
                "checked_regions": checked,
                "processing_seconds": round(time.perf_counter() - started, 3),
                "completed_at": _now(),
            }
    return {
        "policy": SOURCE_ASR_PROBE_POLICY,
        "status": "completed",
        "accepted": False,
        "checked_regions": checked,
        "processing_seconds": round(time.perf_counter() - started, 3),
        "completed_at": _now(),
    }


def _scan_cache_path(cache_dir: Path, candidate: dict[str, Any]) -> Path:
    platform = re.sub(r"[^a-z0-9_-]+", "-", str(candidate.get("source_platform")))
    video_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(candidate["video_id"]))
    return cache_dir / f"{platform}-{video_id}.json"


def _load_source_scan_priors(
    cache_dir: Path, candidates: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    """Learn uploader/query productivity from completed whole-source scans."""
    from .source_scanner import region_passes_confidence_gate

    metadata = {str(item["video_id"]): item for item in candidates}
    aggregates: dict[str, dict[str, list[float]]] = {
        "uploader": {},
        "query": {},
    }
    global_sources = 0
    global_regions = 0.0
    samples: list[tuple[dict[str, Any], float]] = []
    video_regions: dict[str, float] = {}
    for path in cache_dir.glob("*.json"):
        try:
            scan = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        video_id = str(scan.get("video_id") or "")
        passing_regions = (
            0
            if _proxy_asr_blocks_extraction(scan)
            else sum(
                region_passes_confidence_gate(region)
                for region in (scan.get("regions") or [])
            )
        )
        if video_id:
            video_regions[video_id] = float(passing_regions)
        item = scan.get("source_metadata") or metadata.get(video_id)
        if not item:
            continue
        # Cap one unusually dense source so it cannot dominate a whole uploader.
        reward = float(min(10, passing_regions))
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
        "video": video_regions,
        **{
            family: {
                key: (total_reward + prior_strength * global_mean)
                / (source_count + prior_strength)
                for key, (source_count, total_reward) in values.items()
            }
            for family, values in aggregates.items()
        },
    }


def _download_full_source_for_scan(
    candidate: dict[str, Any], root: Path
) -> tuple[Path, dict[str, Any]]:
    duration = float(candidate.get("duration_seconds") or 0.0)
    source_platform = str(candidate.get("source_platform") or "youtube")
    selector = (
        DAILYMOTION_SCAN_PROXY_SELECTOR
        if source_platform == "dailymotion"
        else HIGH_QUALITY_AUDIO_SELECTOR
    )
    timeout = max(900.0, min(3600.0, 600.0 + duration / 10.0))
    command, command_timeout = command_for_media_worker(
        [
            YTDLP_PYTHON,
            "-m",
            "yt_dlp",
            "--no-update",
            "--quiet",
            "--no-warnings",
            "--no-playlist",
            *_yt_dlp_transfer_args(),
            "--socket-timeout",
            "30",
            "--retries",
            "3",
            "--extractor-retries",
            "3",
            *_yt_dlp_proxy_args(source_platform, _candidate_proxy_affinity(candidate)),
            *_yt_dlp_javascript_args(source_platform),
            *_yt_dlp_youtube_client_args(source_platform),
            "-f",
            selector,
            "--print-json",
            "-o",
            str(root / "source.%(ext)s"),
            str(candidate["source_url"]),
        ],
        task="download",
        timeout=timeout,
    )
    response = _run(command, timeout=command_timeout)
    return _source_file(root), _download_json(response.stdout)


def _preflight_source_for_scan(candidate: dict[str, Any]) -> dict[str, Any]:
    """Require a high-quality extraction variant before downloading a proxy."""
    source_platform = str(candidate.get("source_platform") or "youtube")
    selector = (
        DAILYMOTION_SOURCE_SCAN_SELECTOR
        if source_platform == "dailymotion"
        else HIGH_QUALITY_AUDIO_SELECTOR
    )
    base_command = [
        YTDLP_PYTHON,
        "-m",
        "yt_dlp",
        "--no-update",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "--skip-download",
        *_yt_dlp_proxy_args(source_platform, _candidate_proxy_affinity(candidate)),
        *_yt_dlp_javascript_args(source_platform),
        *_yt_dlp_youtube_client_args(source_platform),
        "-f",
    ]

    def run_preflight(selector: str) -> subprocess.CompletedProcess[str]:
        command, timeout = command_for_media_worker(
            [
                *base_command,
                selector,
                "--print-json",
                str(candidate["source_url"]),
            ],
            task="download",
            timeout=90,
        )
        return _run(command, timeout=timeout)

    try:
        response = run_preflight(selector)
        return {**_download_json(response.stdout), "quality_format_available": True}
    except subprocess.CalledProcessError:
        if source_platform != "dailymotion":
            raise
        # Distinguish a live source that only exposes the low-bitrate 380p
        # variant from a transient extraction/network failure. The former can
        # be cached permanently and skipped before proxy transfer on every run.
        response = run_preflight(DAILYMOTION_SCAN_PROXY_SELECTOR)
        return {**_download_json(response.stdout), "quality_format_available": False}


def _download_scanned_sections(
    candidate: dict[str, Any],
    regions: list[dict[str, Any]],
    root: Path,
) -> dict[float, tuple[Path, dict[str, Any]]]:
    """Fetch only selected high-quality excerpts after proxy scanning."""
    source_platform = str(candidate.get("source_platform") or "youtube")
    selector = (
        DAILYMOTION_SOURCE_SCAN_SELECTOR
        if source_platform == "dailymotion"
        else HIGH_QUALITY_AUDIO_SELECTOR
    )
    command = [
        YTDLP_PYTHON,
        "-m",
        "yt_dlp",
        "--no-update",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        *_yt_dlp_transfer_args(),
        "--socket-timeout",
        "20",
        "--retries",
        "2",
        "--extractor-retries",
        "2",
        *_yt_dlp_proxy_args(source_platform, _candidate_proxy_affinity(candidate)),
        *_yt_dlp_javascript_args(source_platform),
        *_yt_dlp_youtube_client_args(source_platform),
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
            selector,
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
    clip_seconds: float | None = None,
) -> bool:
    separation_seconds = CLIP_SECONDS if clip_seconds is None else clip_seconds
    return (
        not any(abs(start - value) < 0.5 for value in attempted_starts)
        and not any(
            abs(start - value) < separation_seconds for value in accepted_starts
        )
        and not any(abs(start - value) < 0.5 for value in claimed_starts)
    )


def _remaining_scan_source_budget(
    source_budget: int,
    *,
    accepted_count: int,
    attempted_starts: list[float],
    claimed_starts: list[float],
) -> int:
    """Count accepted clips and unresolved in-flight claims toward the budget."""
    unresolved_claims = sum(
        not any(abs(claimed - attempted) < 0.5 for attempted in attempted_starts)
        for claimed in claimed_starts
    )
    return max(0, source_budget - accepted_count - unresolved_claims)


def _scan_group_has_remaining_work(
    group: list[dict[str, Any]],
    *,
    cache_dir: Path,
    guidance: dict[str, dict[str, Any]],
    clip_seconds: float | None = None,
) -> bool:
    """Drop globally exhausted scan groups before they consume worker slots."""
    from .source_scanner import load_cached_scan, region_passes_confidence_gate

    base = group[0]
    video_id = str(base["video_id"])
    stats = guidance.get(video_id, {})
    attempted_starts = [float(value) for value in stats.get("attempted_starts", [])]
    accepted_starts = [float(value) for value in stats.get("accepted_starts", [])]
    accepted_count = int(stats.get("accepted", 0))
    source_budget = int(base.get("source_clip_budget") or len(group))
    effective_clip_seconds = CLIP_SECONDS if clip_seconds is None else clip_seconds
    cached = load_cached_scan(
        _scan_cache_path(cache_dir, base), clip_seconds=effective_clip_seconds
    )
    if cached is None:
        return accepted_count < source_budget
    if _proxy_asr_blocks_extraction(cached):
        return False
    claimed_starts = [float(value) for value in cached.get("claimed_starts", [])]
    if (
        _remaining_scan_source_budget(
            source_budget,
            accepted_count=accepted_count,
            attempted_starts=attempted_starts,
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
            clip_seconds=effective_clip_seconds,
        )
        for region in cached.get("regions", [])
        if region_passes_confidence_gate(region)
    )


def acquire_scanned_source_group(
    candidates: list[dict[str, Any]],
    output_dir: Path,
    *,
    scanner: Any,
    cache_dir: Path,
    guidance: dict[str, dict[str, Any]],
    proxy_asr_mode: str = "off",
    proxy_asr_request_dir: Path | None = None,
    proxy_asr_result_dir: Path | None = None,
    proxy_asr_timeout_seconds: float = 120.0,
    defer_claim_commit: bool = False,
    youtube_proxy_attempt: int = 0,
) -> list[dict[str, Any]]:
    """Serialize scan/claim mutations for one source across producer processes."""
    lock_path = _scan_cache_path(cache_dir, candidates[0]).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return []
        return _acquire_scanned_source_group_locked(
            candidates,
            output_dir,
            scanner=scanner,
            cache_dir=cache_dir,
            guidance=guidance,
            proxy_asr_mode=proxy_asr_mode,
            proxy_asr_request_dir=proxy_asr_request_dir,
            proxy_asr_result_dir=proxy_asr_result_dir,
            proxy_asr_timeout_seconds=proxy_asr_timeout_seconds,
            defer_claim_commit=defer_claim_commit,
            youtube_proxy_attempt=youtube_proxy_attempt,
        )


def _acquire_scanned_source_group_locked(
    candidates: list[dict[str, Any]],
    output_dir: Path,
    *,
    scanner: Any,
    cache_dir: Path,
    guidance: dict[str, dict[str, Any]],
    proxy_asr_mode: str = "off",
    proxy_asr_request_dir: Path | None = None,
    proxy_asr_result_dir: Path | None = None,
    proxy_asr_timeout_seconds: float = 120.0,
    defer_claim_commit: bool = False,
    youtube_proxy_attempt: int = 0,
) -> list[dict[str, Any]]:
    """Scan one full source first, then extract only passing stereo regions."""
    from .source_scanner import (
        SCAN_POLICY_VERSION,
        load_cached_scan,
        region_passes_confidence_gate,
    )

    base = candidates[0]
    retrieval_base = {
        **base,
        "_youtube_proxy_attempt": max(0, youtube_proxy_attempt),
    }
    video_id = str(base["video_id"])
    source_platform = str(base.get("source_platform") or "youtube")
    stats = guidance.get(video_id, {})
    attempted_starts = [float(value) for value in stats.get("attempted_starts", [])]
    accepted_starts = [float(value) for value in stats.get("accepted_starts", [])]
    accepted_count = int(stats.get("accepted", 0))
    source_budget = int(base.get("source_clip_budget") or len(candidates))
    cache_path = _scan_cache_path(cache_dir, base)
    cached = load_cached_scan(cache_path, clip_seconds=CLIP_SECONDS)
    reused_cache = cached is not None
    cached_claims = [float(value) for value in (cached or {}).get("claimed_starts", [])]
    remaining_budget = _remaining_scan_source_budget(
        source_budget,
        accepted_count=accepted_count,
        attempted_starts=attempted_starts,
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
            proxy: Path | None = None
            info: dict[str, Any] | None = None
            source_format: dict[str, Any] | None = None
            if cached is None:
                try:
                    target_info = _preflight_source_for_scan(retrieval_base)
                except subprocess.CalledProcessError as error:
                    if not _permanent_media_error(error):
                        raise
                    cached = {
                        "policy": SCAN_POLICY_VERSION,
                        "clip_seconds": CLIP_SECONDS,
                        "video_id": video_id,
                        "source_metadata": {
                            "uploader": base.get("uploader"),
                            "search_query": base.get("search_query"),
                            "title": base.get("title"),
                        },
                        "rejection_reasons": ["source_permanently_unavailable"],
                        "scanned_at": _now(),
                        "claimed_starts": [],
                        "regions": [],
                    }
                    _atomic_json(cache_path, cached)
                    return status_result(
                        "source_unavailable_cached",
                        cached=False,
                        rejection_reasons=cached["rejection_reasons"],
                    )
                if not target_info.get("quality_format_available", True):
                    cached = {
                        "policy": SCAN_POLICY_VERSION,
                        "clip_seconds": CLIP_SECONDS,
                        "video_id": video_id,
                        "extraction_format_id": None,
                        "available_proxy_format_id": target_info.get("format_id"),
                        "source_metadata": {
                            "uploader": base.get("uploader"),
                            "search_query": base.get("search_query"),
                            "title": base.get("title"),
                        },
                        "rejection_reasons": ["source_high_quality_format_unavailable"],
                        "scanned_at": _now(),
                        "claimed_starts": [],
                        "regions": [],
                    }
                    _atomic_json(cache_path, cached)
                    return status_result(
                        "source_quality_rejected",
                        cached=False,
                        rejection_reasons=cached["rejection_reasons"],
                    )
                download_started = time.perf_counter()
                source, info = _download_full_source_for_scan(retrieval_base, root)
                download_seconds = time.perf_counter() - download_started
                source_format = _source_format(
                    source, info, f"{source_platform}-source-scan"
                )
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
            regions = [
                region
                for region in cached.get("regions", [])
                if region_passes_confidence_gate(region)
            ]
            if regions and proxy_asr_mode != "off":
                if proxy_asr_request_dir is None or proxy_asr_result_dir is None:
                    raise ValueError(
                        "Proxy ASR request and result directories are required"
                    )
                probe = _current_proxy_asr(cached)
                if probe is None:
                    probe = _proxy_asr_from_guidance(stats)
                if probe is None:
                    if proxy is None:
                        probe_source, _ = _download_full_source_for_scan(
                            retrieval_base, root
                        )
                        proxy = root / "probe-proxy.flac"
                        scanner.create_proxy(probe_source, proxy)
                    try:
                        probe = _probe_source_proxy_asr(
                            proxy,
                            regions,
                            root,
                            video_id=video_id,
                            request_dir=proxy_asr_request_dir,
                            result_dir=proxy_asr_result_dir,
                            timeout_seconds=proxy_asr_timeout_seconds,
                        )
                    except TimeoutError as error:
                        probe = {
                            "policy": SOURCE_ASR_PROBE_POLICY,
                            "status": "timeout",
                            "accepted": None,
                            "checked_regions": [],
                            "error": str(error),
                            "completed_at": _now(),
                        }
                probe["enforced"] = proxy_asr_mode == "enforce"
                cached["proxy_asr"] = probe
                _atomic_json(cache_path, cached)
                if proxy_asr_mode == "enforce" and probe.get("accepted") is not True:
                    return status_result(
                        "source_proxy_asr_rejected",
                        cached=reused_cache,
                        proxy_asr=probe,
                    )
            claimed_starts = [
                float(value) for value in cached.get("claimed_starts", [])
            ]
            available = [
                region
                for region in cached.get("regions", [])
                if region_passes_confidence_gate(region)
                and _scan_region_available(
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
            selected_sections = _download_scanned_sections(
                retrieval_base, available, root
            )
            extraction_download_seconds = (
                time.perf_counter() - extraction_download_started
            )
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
                        "proxy_asr": cached.get("proxy_asr"),
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
                            f"{source_platform}-source-scan-section",
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
            attempted_region_starts = [
                float(result["clip_start_seconds"])
                for result in results
                if result.get("selection") == "whole_source_proxy_scan"
                and result.get("retrieval_status")
                in {"success", "rejected", "unavailable"}
            ]
            if not defer_claim_commit:
                cached["claimed_starts"] = sorted(
                    set(claimed_starts + attempted_region_starts)
                )
                _atomic_json(cache_path, cached)
            return results
    except Exception as error:
        return status_result("source_scan_unavailable", error=_exception_text(error))


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
    proxy_asr_mode = "off"
    proxy_asr_request_dir: Path | None = None
    proxy_asr_result_dir: Path | None = None
    proxy_asr_timeout_seconds = 120.0
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
        proxy_asr_mode = str(source_scan_config.get("proxy_asr_mode", "off"))
        if source_scan_config.get("proxy_asr_request_dir") is not None:
            proxy_asr_request_dir = Path(source_scan_config["proxy_asr_request_dir"])
        if source_scan_config.get("proxy_asr_result_dir") is not None:
            proxy_asr_result_dir = Path(source_scan_config["proxy_asr_result_dir"])
        proxy_asr_timeout_seconds = float(
            source_scan_config.get("proxy_asr_timeout_seconds", 120.0)
        )
        scanner = M2DSourceScanner(
            m2d_repo=Path(source_scan_config["m2d_repo"]),
            checkpoint=Path(source_scan_config["checkpoint"]),
            class_labels=Path(source_scan_config["class_labels"]),
            ontology=Path(source_scan_config["ontology"]),
            device=str(source_scan_config.get("device", "cuda")),
            batch_size=int(source_scan_config.get("batch_size", 128)),
        )
        candidates = _inject_cached_scan_sources(
            candidates,
            scan_cache_dir,
            guidance,
            clips_per_video=clips_per_video,
            source_content_minutes_per_hour=source_content_minutes_per_hour,
            max_clips_per_video=max_clips_per_video,
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
    last_logged_accepted = accepted_count
    next_attempt_log = 100
    with attempts_path.open("a", encoding="utf-8") as attempt_log:
        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            group_iterator = iter(limited_groups)
            in_flight: dict[Future[list[dict[str, Any]]], list[dict[str, Any]]] = {}

            def submit_group(group: list[dict[str, Any]]) -> None:
                if scanner is not None and scan_cache_dir is not None:
                    future = executor.submit(
                        acquire_scanned_source_group,
                        group,
                        output_dir,
                        scanner=scanner,
                        cache_dir=scan_cache_dir,
                        guidance=guidance,
                        proxy_asr_mode=proxy_asr_mode,
                        proxy_asr_request_dir=proxy_asr_request_dir,
                        proxy_asr_result_dir=proxy_asr_result_dir,
                        proxy_asr_timeout_seconds=proxy_asr_timeout_seconds,
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
                    if (
                        accepted_count != last_logged_accepted
                        or attempts_made >= next_attempt_log
                    ):
                        logger.info(
                            "Accepted %d/%d after %d new attempts",
                            accepted_count,
                            total,
                            attempts_made,
                        )
                        last_logged_accepted = accepted_count
                        next_attempt_log = attempts_made + 100
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
    global CLIP_SECONDS, YTDLP_PYTHON, YOUTUBE_PROXY_CONFIG
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
    parser.add_argument(
        "--source-asr-probe-mode",
        choices=("off", "shadow", "enforce"),
        default="off",
    )
    parser.add_argument("--source-asr-probe-requests", type=Path)
    parser.add_argument("--source-asr-probe-results", type=Path)
    parser.add_argument("--source-asr-probe-timeout", type=float, default=120.0)
    parser.add_argument("--yt-dlp-python", type=Path)
    parser.add_argument(
        "--youtube-proxy-config",
        type=Path,
        help="Protected yt-dlp config containing the proxy for YouTube only",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Populate the reusable search candidate cache without downloading audio",
    )
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
    YOUTUBE_PROXY_CONFIG = args.youtube_proxy_config
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
    if args.source_asr_probe_timeout <= 0:
        parser.error("--source-asr-probe-timeout must be positive")
    if args.source_asr_probe_mode != "off" and (
        args.source_asr_probe_requests is None or args.source_asr_probe_results is None
    ):
        parser.error(
            "--source-asr-probe-mode requires --source-asr-probe-requests "
            "and --source-asr-probe-results"
        )
    if args.verify_only:
        result = verify_dataset(args.output, target=args.total)
        print(json.dumps(result, indent=2))
        if not result["all_requirements_pass"]:
            raise SystemExit(1)
        return
    query_count = args.query_count or max(40, math.ceil(args.total * 0.4))
    max_attempts = args.max_attempts or math.ceil(args.total * 3.0)
    if args.discover_only:
        candidates = discover_candidates(
            args.output,
            seed=args.seed,
            query_count=query_count,
            results_per_query=args.results_per_query,
            workers=args.search_workers,
            minimum_candidates=math.ceil(args.total * args.candidate_multiplier),
            profile=args.profile,
            clips_per_video=clips_per_video,
            source_content_minutes_per_hour=args.source_content_minutes_per_hour,
            max_clips_per_video=args.max_clips_per_video,
            source=args.source,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "seed": args.seed,
                    "candidate_count": len(candidates),
                },
                indent=2,
            )
        )
        return
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
                "proxy_asr_mode": args.source_asr_probe_mode,
                "proxy_asr_request_dir": args.source_asr_probe_requests,
                "proxy_asr_result_dir": args.source_asr_probe_results,
                "proxy_asr_timeout_seconds": args.source_asr_probe_timeout,
            }
            if args.scan_before_extract
            else None
        ),
    )
    print(path)


if __name__ == "__main__":
    main()
