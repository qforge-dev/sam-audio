from __future__ import annotations

import fcntl
import json
import math
import signal
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

import sam_audio_pipeline.youtube_random as youtube_random
from sam_audio_pipeline.youtube_random import (
    CANDIDATE_DURATION_POLICY,
    DAILYMOTION_SEARCH_POLICY,
    MAX_SOURCE_DURATION_SECONDS,
    _accepted,
    _candidate_allowed,
    _cinematic_candidate_priority,
    _dailymotion_has_high_quality_format,
    _dailymotion_search_page,
    _discovery_candidate_allowed,
    _group_candidates_by_video,
    _inject_cached_scan_sources,
    _load_source_scan_priors,
    _order_scanned_source_groups,
    _permanent_media_error,
    _proxy_asr_blocks_extraction,
    _proxy_asr_from_guidance,
    _query_for_source,
    _remaining_scan_source_budget,
    _runtime_worker_limit,
    _sample_clip_starts,
    _scan_group_has_remaining_work,
    _search_ytdlp_provider,
    _use_full_source_for_group,
    acquire_scanned_source_group,
    analyze_wav,
    build_queries,
    discover_candidates,
    quality_rejections,
)


def write_stereo(path: Path, *, dual_mono: bool = False) -> None:
    sample_rate = 48_000
    timeline = np.arange(sample_rate * 10) / sample_rate
    left = 0.3 * np.sin(2 * math.pi * 440 * timeline)
    right = left if dual_mono else 0.3 * np.sin(2 * math.pi * 554 * timeline)
    encoded = np.rint(np.column_stack((left, right)) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(encoded.tobytes())


def source_format() -> dict[str, object]:
    return {"sample_rate_hz": 48_000, "channels": 2, "bitrate_kbps": 130.0}


def test_run_terminates_the_entire_process_group_on_timeout(monkeypatch) -> None:
    class FakeProcess:
        pid = 1234
        returncode = None
        stdout = None
        stderr = None

        def __init__(self) -> None:
            self.communicate_calls = 0

        def communicate(self, timeout=None):
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise subprocess.TimeoutExpired(["downloader"], timeout)
            self.returncode = -signal.SIGTERM
            return "partial stdout", "partial stderr"

    process = FakeProcess()
    popen_kwargs: dict[str, object] = {}
    signals: list[tuple[int, signal.Signals]] = []

    def fake_popen(command, **kwargs):
        popen_kwargs.update(kwargs)
        return process

    monkeypatch.setattr(youtube_random.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        youtube_random.os, "killpg", lambda pid, sent: signals.append((pid, sent))
    )

    with pytest.raises(subprocess.TimeoutExpired):
        youtube_random._run(["downloader"], timeout=0.1)

    assert popen_kwargs["start_new_session"] is True
    assert signals == [(1234, signal.SIGTERM)]
    assert process.communicate_calls == 2


def test_proxy_config_is_secret_backed_and_applies_to_all_providers(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "youtube-proxy.conf"
    config.write_text("--proxy http://user:secret@proxy.example:80\n")
    config.chmod(0o600)
    monkeypatch.setattr(youtube_random, "YOUTUBE_PROXY_CONFIG", config)

    args = youtube_random._yt_dlp_proxy_args("youtube")

    assert args == ["--config-locations", str(config)]
    assert "http://user:secret@" not in " ".join(args)
    assert youtube_random._yt_dlp_proxy_args("dailymotion") == [
        "--config-locations",
        str(config),
    ]


def test_yt_dlp_transfer_args_use_bounded_fragment_concurrency(monkeypatch) -> None:
    monkeypatch.setattr(youtube_random, "YTDLP_CONCURRENT_FRAGMENTS", 4)

    assert youtube_random._yt_dlp_transfer_args() == ["--concurrent-fragments", "4"]


def test_missing_youtube_proxy_config_fails_before_download(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        youtube_random, "YOUTUBE_PROXY_CONFIG", tmp_path / "missing.conf"
    )

    try:
        youtube_random._yt_dlp_proxy_args("youtube")
    except FileNotFoundError as error:
        assert "missing.conf" in str(error)
    else:
        raise AssertionError("missing proxy config should fail closed")


def test_site_search_hydrates_vimeo_results(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command, *, timeout):
        commands.append(command)
        if "--flat-playlist" in command:
            stdout = json.dumps(
                {"entries": [{"url": "https://vimeo.com/123456"}]}
            )
        else:
            stdout = json.dumps(
                {
                    "id": "123456",
                    "title": "English cinematic movie scene dialogue",
                    "duration": 180,
                    "webpage_url": "https://vimeo.com/123456",
                    "uploader": "short-film-studio",
                }
            )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(youtube_random, "_run", fake_run)

    items = _search_ytdlp_provider(
        "cinematic dialogue -reaction", 100, "cinematic", "vimeo"
    )

    assert len(items) == 1
    assert items[0]["source_url"] == "https://vimeo.com/123456"
    search_target = commands[0][-1]
    assert search_target.startswith("yvsearch8:site:vimeo.com ")
    assert "-reaction" not in search_target
    assert "--skip-download" in commands[1]


def test_youtube_preflight_uses_proxy_config_and_audio_only_selector(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "youtube-proxy.conf"
    config.write_text("--proxy http://user:secret@proxy.example:80\n")
    config.chmod(0o600)
    monkeypatch.setattr(youtube_random, "YOUTUBE_PROXY_CONFIG", config)
    commands: list[list[str]] = []

    def run(command: list[str], *, timeout: float):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"id": "video", "format_id": "251"}),
            stderr="",
        )

    monkeypatch.setattr(youtube_random, "_run", run)

    result = youtube_random._preflight_source_for_scan(
        {
            "source_platform": "youtube",
            "source_url": "https://www.youtube.com/watch?v=video",
        }
    )

    assert result["quality_format_available"] is True
    assert ["--config-locations", str(config)] == commands[0][
        commands[0].index("--config-locations") : commands[0].index(
            "--config-locations"
        )
        + 2
    ]
    assert youtube_random.HIGH_QUALITY_AUDIO_SELECTOR in commands[0]
    assert "youtube:player_client=tv" in commands[0]
    assert "secret" not in " ".join(commands[0])


def test_proxy_pool_pins_a_source_and_changes_across_retries(
    tmp_path: Path, monkeypatch
) -> None:
    pool = tmp_path / "proxy-pool"
    pool.mkdir(mode=0o700)
    for index in range(10):
        config = pool / f"proxy-{index:02d}.conf"
        config.write_text(f"--proxy http://user:secret-{index}@proxy-{index}:80\n")
        config.chmod(0o600)
    monkeypatch.setattr(youtube_random, "YOUTUBE_PROXY_CONFIG", pool)

    first = youtube_random._yt_dlp_proxy_args("youtube", "video:0")
    repeated = youtube_random._yt_dlp_proxy_args("youtube", "video:0")
    retries = {
        tuple(youtube_random._yt_dlp_proxy_args("youtube", f"video:{attempt}"))
        for attempt in range(10)
    }

    assert first == repeated
    assert len(retries) > 1
    assert all("secret" not in " ".join(args) for args in retries)


def test_proxy_config_rejects_readable_secret_files(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "youtube-proxy.conf"
    config.write_text("--proxy http://user:secret@proxy.example:80\n")
    config.chmod(0o644)
    monkeypatch.setattr(youtube_random, "YOUTUBE_PROXY_CONFIG", config)

    try:
        youtube_random._yt_dlp_proxy_args("youtube", "video:0")
    except PermissionError as error:
        assert "group/world accessible" in str(error)
    else:
        raise AssertionError("readable proxy credentials should fail closed")


def test_proxy_credentials_are_redacted_from_failures() -> None:
    error = subprocess.CalledProcessError(
        1,
        ["yt-dlp"],
        stderr="proxy http://user:secret@proxy.example:80 failed",
    )

    message = youtube_random._exception_text(error)

    assert "secret" not in message
    assert "http://***:***@proxy.example:80" in message


def test_runtime_download_limit_is_bounded_and_failure_safe(tmp_path: Path) -> None:
    control = tmp_path / "autoscale-control.json"

    assert _runtime_worker_limit(control, maximum=8, default=6) == 6
    control.write_text(json.dumps({"download_concurrency": 3}))
    assert _runtime_worker_limit(control, maximum=8, default=6) == 3
    control.write_text(json.dumps({"download_concurrency": 99}))
    assert _runtime_worker_limit(control, maximum=8, default=6) == 8
    control.write_text("not-json")
    assert _runtime_worker_limit(control, maximum=8, default=6) == 6


def test_source_level_failure_marks_every_group_candidate_attempted(
    tmp_path: Path,
) -> None:
    candidates = [
        {
            "candidate_id": f"video:{index}",
            "video_id": "video",
            "source_clip_budget": 2,
        }
        for index in range(2)
    ]

    results = acquire_scanned_source_group(
        candidates,
        tmp_path,
        scanner=None,
        cache_dir=tmp_path / "cache",
        guidance={"video": {"accepted": 2}},
    )

    assert [item["candidate_id"] for item in results] == ["video:0", "video:1"]
    assert all(
        item["retrieval_status"] == "source_budget_exhausted" for item in results
    )


def test_scan_budget_does_not_double_count_accepted_claims() -> None:
    remaining = _remaining_scan_source_budget(
        16,
        accepted_count=5,
        attempted_starts=[0.0, 30.0, 60.0, 90.0, 120.0],
        claimed_starts=[0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0],
    )

    assert remaining == 9


def test_completed_rejected_claims_do_not_consume_acceptance_budget() -> None:
    remaining = _remaining_scan_source_budget(
        16,
        accepted_count=1,
        attempted_starts=[0.0, 30.0, 60.0, 90.0, 120.0],
        claimed_starts=[0.0, 30.0, 60.0, 90.0, 120.0],
    )

    assert remaining == 15


def test_exhausted_cached_scan_group_is_removed_before_scheduling(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    group = [
        {
            "video_id": "scene",
            "source_platform": "dailymotion",
            "source_clip_budget": 16,
        }
    ]
    (cache / "dailymotion-scene.json").write_text(
        json.dumps(
            {
                "policy": "whole_source_proxy_m2d_v1",
                "clip_seconds": 10.0,
                "claimed_starts": [30.0],
                "regions": [{"start_seconds": 30.0}],
            }
        )
    )
    assert not _scan_group_has_remaining_work(group, cache_dir=cache, guidance={})
    assert (
        _scan_group_has_remaining_work(
            group,
            cache_dir=cache,
            guidance={"scene": {"attempted_starts": [], "accepted_starts": []}},
        )
        is False
    )


def test_remaining_work_uses_explicit_clip_duration_for_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    group = [
        {
            "video_id": "thirty-second-scene",
            "source_platform": "dailymotion",
            "source_clip_budget": 16,
        }
    ]
    (cache / "dailymotion-thirty-second-scene.json").write_text(
        json.dumps(
            {
                "policy": "whole_source_proxy_m2d_v1",
                "clip_seconds": 30.0,
                "claimed_starts": [30.0],
                "regions": [{"start_seconds": 30.0}],
            }
        )
    )

    assert not _scan_group_has_remaining_work(
        group,
        cache_dir=cache,
        guidance={},
        clip_seconds=30.0,
    )


def test_unseen_scan_group_is_kept_for_exploration(tmp_path: Path) -> None:
    group = [
        {
            "video_id": "new-scene",
            "source_platform": "dailymotion",
            "source_clip_budget": 16,
        }
    ]

    assert _scan_group_has_remaining_work(group, cache_dir=tmp_path, guidance={})


def test_low_quality_source_is_negatively_cached(tmp_path: Path, monkeypatch) -> None:
    candidate = {
        "candidate_id": "low-quality:0",
        "video_id": "low-quality",
        "source_platform": "dailymotion",
        "source_url": "https://example.invalid/low-quality",
        "source_clip_budget": 1,
    }
    cache = tmp_path / "cache"
    monkeypatch.setattr(
        youtube_random,
        "_preflight_source_for_scan",
        lambda _: {"quality_format_available": False, "format_id": "hls-380"},
    )

    results = acquire_scanned_source_group(
        [candidate],
        tmp_path / "output",
        scanner=None,
        cache_dir=cache,
        guidance={},
    )

    assert results[0]["retrieval_status"] == "source_quality_rejected"
    assert not _scan_group_has_remaining_work([candidate], cache_dir=cache, guidance={})
    cached = json.loads((cache / "dailymotion-low-quality.json").read_text())
    assert cached["rejection_reasons"] == ["source_high_quality_format_unavailable"]


def test_busy_cross_process_source_lock_skips_without_waiting(tmp_path: Path) -> None:
    candidate = {
        "candidate_id": "busy:0",
        "video_id": "busy",
        "source_platform": "dailymotion",
        "source_clip_budget": 1,
    }
    cache = tmp_path / "cache"
    cache.mkdir()
    lock_path = cache / "dailymotion-busy.lock"

    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (
            acquire_scanned_source_group(
                [candidate],
                tmp_path / "output",
                scanner=None,
                cache_dir=cache,
                guidance={},
            )
            == []
        )


def test_only_permanent_media_failures_are_negatively_cached() -> None:
    missing = subprocess.CalledProcessError(
        1, ["yt-dlp"], stderr="ERROR: [dailymotion] x: Not found."
    )
    transient = subprocess.CalledProcessError(
        1, ["yt-dlp"], stderr="HTTP Error 429: Too Many Requests"
    )

    assert _permanent_media_error(missing)
    assert _permanent_media_error(
        subprocess.CalledProcessError(1, ["yt-dlp"], stderr="Private content")
    )
    assert _permanent_media_error(
        subprocess.CalledProcessError(1, ["yt-dlp"], stderr="No video formats found")
    )
    assert not _permanent_media_error(transient)


def test_proxy_asr_catalog_evidence_requires_two_failures_or_one_success() -> None:
    assert _proxy_asr_from_guidance({"asr_scored": 1, "asr_accepted": 0}) is None
    rejected = _proxy_asr_from_guidance({"asr_scored": 2, "asr_accepted": 0})
    accepted = _proxy_asr_from_guidance({"asr_scored": 3, "asr_accepted": 1})

    assert rejected is not None and rejected["accepted"] is False
    assert accepted is not None and accepted["accepted"] is True


def test_only_enforced_proxy_asr_rejection_blocks_extraction() -> None:
    shadow = {
        "proxy_asr": {
            "policy": "source_proxy_asr_top3_beam1_v1",
            "accepted": False,
            "enforced": False,
        }
    }
    enforced = {
        "proxy_asr": {
            "policy": "source_proxy_asr_top3_beam1_v1",
            "accepted": False,
            "enforced": True,
        }
    }
    inconclusive = {
        "proxy_asr": {
            "policy": "source_proxy_asr_top3_beam1_v1",
            "accepted": None,
            "enforced": True,
        }
    }

    assert not _proxy_asr_blocks_extraction(shadow)
    assert _proxy_asr_blocks_extraction(enforced)
    assert _proxy_asr_blocks_extraction(inconclusive)


def test_scanned_source_order_learns_productive_uploaders(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    candidates = [
        {
            "video_id": "known-good",
            "uploader": "productive",
            "search_query": "good query",
            "duration_seconds": 120,
        },
        {
            "video_id": "known-bad",
            "uploader": "unproductive",
            "search_query": "bad query",
            "duration_seconds": 120,
        },
    ]
    (cache / "good.json").write_text(
        json.dumps(
            {
                "video_id": "known-good",
                "regions": [
                    {
                        "evidence": {
                            "foreground_speech_coverage": 0.7,
                            "vocal_music_coverage": 0.0,
                        }
                    }
                ]
                * 8,
            }
        )
    )
    (cache / "bad.json").write_text(
        json.dumps({"video_id": "known-bad", "regions": []})
    )
    priors = _load_source_scan_priors(cache, candidates)
    groups = [
        [
            {
                "video_id": "new-bad",
                "uploader": "unproductive",
                "search_query": "bad query",
                "duration_seconds": 120,
            }
        ],
        [
            {
                "video_id": "new-good",
                "uploader": "productive",
                "search_query": "good query",
                "duration_seconds": 120,
            }
        ],
    ]

    ordered = _order_scanned_source_groups(groups, {}, scan_priors=priors)

    assert ordered[0][0]["video_id"] == "new-good"


def test_source_order_prefers_exact_cached_region_inventory() -> None:
    groups = [
        [{"video_id": "unscanned", "duration_seconds": 120}],
        [{"video_id": "cached", "duration_seconds": 120}],
    ]

    ordered = _order_scanned_source_groups(
        groups,
        {},
        scan_priors={
            "video": {"cached": 8.0},
            "uploader": {},
            "query": {},
        },
    )

    assert ordered[0][0]["video_id"] == "cached"


def test_source_order_discount_cached_regions_that_failed_final_validation() -> None:
    groups = [
        [
            {
                "video_id": "dense-but-failing",
                "uploader": "failed",
                "duration_seconds": 120,
            }
        ],
        [
            {
                "video_id": "unscanned",
                "uploader": "promising",
                "duration_seconds": 120,
            }
        ],
    ]

    ordered = _order_scanned_source_groups(
        groups,
        {"dense-but-failing": {"scored": 20, "accepted": 0}},
        scan_priors={
            "video": {"dense-but-failing": 40.0},
            "uploader": {"promising": 4.0},
            "query": {},
        },
        content_priors={
            "uploader": {"promising": 0.5},
            "query": {},
            "global": {"acceptance": 0.5},
        },
    )

    assert ordered[0][0]["video_id"] == "unscanned"


def test_cached_passing_source_is_reintroduced_before_it_has_an_acceptance(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "dailymotion-cached.json").write_text(
        json.dumps(
            {
                "video_id": "cached",
                "m2d_windows": 600,
                "source_metadata": {
                    "title": "English Movie Scene HD",
                    "uploader": "Movie Source",
                    "search_query": "movie scene dialogue",
                },
                "regions": [
                    {
                        "start_seconds": 30.0,
                        "evidence": {
                            "foreground_speech_coverage": 0.7,
                            "vocal_music_coverage": 0.0,
                        },
                    }
                ],
            }
        )
    )

    candidates = _inject_cached_scan_sources(
        [],
        cache,
        {},
        clips_per_video=16,
        source_content_minutes_per_hour=10.0,
        max_clips_per_video=60,
    )

    assert [item["video_id"] for item in candidates] == ["cached"]
    assert candidates[0]["selection"] == "cached_proxy_scan_reuse"


def test_source_order_optimizes_final_yield_not_only_m2d_regions() -> None:
    groups = [
        [{"video_id": "foreign", "uploader": "foreign", "duration_seconds": 120}],
        [{"video_id": "english", "uploader": "english", "duration_seconds": 120}],
    ]
    scan_priors = {
        "uploader": {"foreign": 10.0, "english": 4.0},
        "query": {},
    }
    content_priors = {
        "uploader": {"foreign": 0.1, "english": 0.9},
        "query": {},
        "global": {"acceptance": 0.5},
    }

    ordered = _order_scanned_source_groups(
        groups,
        {},
        scan_priors=scan_priors,
        content_priors=content_priors,
    )

    assert ordered[0][0]["video_id"] == "english"


def test_queries_are_reproducible_mix_biased_and_not_audioset() -> None:
    first = build_queries(17, 20)
    second = build_queries(17, 20)

    assert first == second
    assert len(set(first)) == 20
    assert all("audioset" not in query.lower() for query in first)
    assert all("-official" in query and "-playlist" in query for query in first)
    assert all(
        any(word in query for word in ("music", "soundtrack", "score"))
        for query in first
    )


def test_dailymotion_search_pages_are_reproducible_and_seeded() -> None:
    queries = build_queries(17, 20, profile="cinematic")
    first = [_dailymotion_search_page(17, query) for query in queries]
    second = [_dailymotion_search_page(17, query) for query in queries]
    another_seed = [_dailymotion_search_page(18, query) for query in queries]

    assert first == second
    assert all(1 <= page <= 10 for page in first)
    assert len(set(first)) >= 6
    assert first != another_seed


def test_dailymotion_search_requires_an_hd_source_variant() -> None:
    assert not _dailymotion_has_high_quality_format({"available_formats": ["sd"]})
    assert _dailymotion_has_high_quality_format(
        {"available_formats": ["sd", "hd720@60"]}
    )


def test_cinematic_queries_and_metadata_filter_target_raw_scenes() -> None:
    queries = build_queries(31, 20, profile="cinematic")

    assert all("-reaction" in query and "-Bollywood" in query for query in queries)
    assert all(
        any(
            hint in query
            for hint in (
                "English",
                "HD",
                "dialogue",
                "soundtrack",
                "cinematic",
                "story",
            )
        )
        for query in queries
    )
    assert all(
        any(
            kind in query
            for kind in (
                "scene",
                "clip",
                "cutscene",
                "short film",
                "package",
                "gameplay",
                "episode",
                "dialogue",
                "banter",
                "story mode",
                "walkthrough",
                "game movie",
                "web series",
            )
        )
        for query in queries
    )
    assert _candidate_allowed(
        {
            "id": "scene",
            "title": "Captain America Elevator Scene 4K",
            "duration": 600,
            "channel": "Movie Clips",
        },
        profile="cinematic",
    )
    assert not _candidate_allowed(
        {
            "id": "reaction",
            "title": "Captain America Elevator Scene Reaction",
            "duration": 180,
        },
        profile="cinematic",
    )
    assert not _candidate_allowed(
        {
            "id": "punctuated-country",
            "title": "Dramatic Movie Scene",
            "description": "Produced in India.",
            "duration": 180,
        },
        profile="cinematic",
    )
    assert not _candidate_allowed(
        {
            "id": "hidden-language",
            "title": "Dramatic Movie Scene",
            "description": "A popular Hindi-language movie clip from India",
            "duration": 180,
        },
        profile="cinematic",
    )
    for title in (
        "Battle Through The Heavens Episode 152 English Sub",
        "Prabhas Mass Entry Scene",
        "Raghuvaran Scene - Dhanush Dialogue",
        "Hera Pheri Paresh Raval Comedy Scene",
    ):
        assert not _candidate_allowed(
            {"id": title, "title": title, "duration": 180},
            profile="cinematic",
        )


def test_non_youtube_search_removes_negative_query_tokens() -> None:
    query = "movie scene dialogue -reaction -India English"

    assert _query_for_source(query, "youtube") == query
    assert _query_for_source(query, "dailymotion") == "movie scene dialogue English"
    assert not _candidate_allowed(
        {
            "id": "language",
            "title": "Hindi Movie Scene",
            "duration": 180,
        },
        profile="cinematic",
    )


def test_cinematic_segment_sampling_is_reproducible_and_non_overlapping() -> None:
    first = _sample_clip_starts(
        seed=42, video_id="movie", duration=180, clips_per_video=3
    )
    second = _sample_clip_starts(
        seed=42, video_id="movie", duration=180, clips_per_video=3
    )

    assert first == second
    assert len(first) == 3
    assert all(
        right - left >= 12 for left, right in zip(first, first[1:], strict=False)
    )


def test_explicit_cinematic_titles_are_prioritized_over_generic_clips() -> None:
    cinematic = {
        "title": "English Animated Movie Clip Battle Scene 4K HD",
    }
    generic = {"title": "Amazing Best Scene Funny Clips"}

    assert _cinematic_candidate_priority(cinematic) > 0
    assert _cinematic_candidate_priority(generic) < 0


def test_long_cinematic_sources_can_supply_many_distinct_excerpts() -> None:
    starts = _sample_clip_starts(
        seed=42, video_id="feature-length-source", duration=1800, clips_per_video=48
    )

    assert len(starts) == 48
    assert all(
        right - left >= 12 for left, right in zip(starts, starts[1:], strict=False)
    )


def test_cinematic_sampling_grows_with_source_duration_and_stops_at_guardrail() -> None:
    three_hours = _sample_clip_starts(
        seed=42,
        video_id="three-hour-game",
        duration=3 * 3600,
        clips_per_video=16,
        source_content_minutes_per_hour=10,
        max_clips_per_video=60,
    )
    long_stream = _sample_clip_starts(
        seed=42,
        video_id="long-stream",
        duration=48 * 3600,
        clips_per_video=16,
        source_content_minutes_per_hour=10,
        max_clips_per_video=60,
    )

    # This module defaults to ten-second excerpts, so the absolute guardrail
    # applies. The continuous service sets the excerpt length to 30 seconds.
    assert len(three_hours) == 60
    assert len(long_stream) == 60
    assert all(
        right - left >= 12
        for left, right in zip(three_hours, three_hours[1:], strict=False)
    )


def test_group_download_uses_one_full_transfer_for_all_supported_sources() -> None:
    dense = [{"duration_seconds": 220}] * 11
    sparse = [{"duration_seconds": 3600}] * 48

    assert _use_full_source_for_group(dense) is True
    assert _use_full_source_for_group(sparse) is True
    assert _use_full_source_for_group([{"duration_seconds": 3601}] * 48) is False


def test_dailymotion_work_is_grouped_by_video_without_reordering_clips() -> None:
    candidates = [
        {"video_id": "a", "candidate_id": "a:1"},
        {"video_id": "b", "candidate_id": "b:1"},
        {"video_id": "a", "candidate_id": "a:2"},
    ]

    assert _group_candidates_by_video(candidates, grouped=True) == [
        [candidates[0], candidates[2]],
        [candidates[1]],
    ]
    assert _group_candidates_by_video(candidates, grouped=False) == [
        [candidates[0]],
        [candidates[1]],
        [candidates[2]],
    ]


def test_cached_candidates_are_refiltered_under_current_metadata_policy(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    candidates = [
        {
            "candidate_id": "good:1",
            "video_id": "good",
            "title": "English Movie Scene",
            "duration_seconds": 600,
        },
        {
            "candidate_id": "subbed:1",
            "video_id": "subbed",
            "title": "Movie Scene English Sub",
            "duration_seconds": 600,
        },
    ]
    (metadata / "candidates.json").write_text(json.dumps(candidates))
    (metadata / "search.json").write_text(
        json.dumps(
            {
                "profile": "cinematic",
                "clips_per_video": 48,
                "source": "dailymotion",
                "candidate_duration_policy": CANDIDATE_DURATION_POLICY,
                "search_page_policy": DAILYMOTION_SEARCH_POLICY,
            }
        )
    )

    filtered = discover_candidates(
        tmp_path,
        seed=1,
        query_count=1,
        results_per_query=1,
        workers=1,
        minimum_candidates=1,
        profile="cinematic",
        clips_per_video=48,
        source="dailymotion",
    )

    assert [item["video_id"] for item in filtered] == ["good"]
    assert json.loads((metadata / "search.json").read_text())["unique_candidates"] == 1


def test_accepted_records_follow_current_metadata_policy() -> None:
    allowed = {
        "video_id": "allowed",
        "title": "English Movie Scene HD",
        "duration_seconds": 600,
        "retrieval_status": "success",
    }
    excluded = {
        "video_id": "excluded",
        "title": "Prabhas Mass Entry Scene",
        "duration_seconds": 600,
        "retrieval_status": "success",
    }

    assert _accepted([allowed, excluded], profile="cinematic") == [allowed]


def test_candidate_filter_rejects_short_live_and_pure_audio_results() -> None:
    valid = {"id": "video", "title": "City festival vlog", "duration": 600}

    assert _candidate_allowed(valid)
    assert not _candidate_allowed({**valid, "duration": 20})
    assert not _candidate_allowed({**valid, "live_status": "is_live"})
    assert not _candidate_allowed({**valid, "title": "Official Audio"})
    assert not _candidate_allowed({**valid, "title": "Song (Official Video)"})


def test_discovery_duration_floor_does_not_invalidate_existing_records() -> None:
    short_source = {
        "id": "scene",
        "title": "English Movie Scene HD",
        "duration": 90,
    }

    assert _candidate_allowed(short_source, profile="cinematic")
    assert not _discovery_candidate_allowed(
        short_source, profile="cinematic", source="dailymotion"
    )


def test_candidate_filter_accepts_long_form_sources_with_a_safety_ceiling() -> None:
    long_game = {
        "id": "long-game",
        "title": "Gameplay with dialogue and cinematic soundtrack",
        "duration": 8 * 3600,
    }

    assert _candidate_allowed(long_game)
    assert _candidate_allowed(long_game, profile="cinematic")
    assert _candidate_allowed(
        {**long_game, "title": "RPG Story Mode Walkthrough English"},
        profile="cinematic",
    )
    assert not _candidate_allowed(
        {**long_game, "title": "Gameplay Review and Analysis"},
        profile="cinematic",
    )
    assert not _candidate_allowed(
        {**long_game, "duration": MAX_SOURCE_DURATION_SECONDS + 1}
    )


def test_quality_gate_accepts_active_true_stereo(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    write_stereo(path)

    metrics = analyze_wav(path)

    assert metrics["duration_seconds"] == 10.0
    assert metrics["silent_fraction"] == 0.0
    assert metrics["side_to_total_db"] > -10
    assert quality_rejections(metrics, source_format()) == []


def test_quality_gate_rejects_channel_duplicated_mono(tmp_path: Path) -> None:
    path = tmp_path / "dual-mono.wav"
    write_stereo(path, dual_mono=True)

    metrics = analyze_wav(path)

    assert metrics["channel_correlation"] == 1.0
    assert "dual_mono" in quality_rejections(metrics, source_format())


def test_quality_metrics_remain_finite_for_constant_channel(tmp_path: Path) -> None:
    path = tmp_path / "constant.wav"
    samples = np.zeros((48_000 * 10, 2), dtype="<i2")
    samples[:, 1] = 1000
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes(samples.tobytes())

    metrics = analyze_wav(path)

    assert metrics["channel_correlation"] == 0.0
