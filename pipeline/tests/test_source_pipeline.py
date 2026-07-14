from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import sam_audio_pipeline.source_pipeline as source_pipeline
from sam_audio_pipeline.source_frontier import (
    claim_source,
    connect_frontier,
    enqueue_source,
    finish_source,
    frontier_counts,
)
from sam_audio_pipeline.source_pipeline import (
    DiscoverySettings,
    DownloadSettings,
    ExtractRunWriter,
    ExtractSettings,
    ScanSettings,
    _commit_extraction_claims,
    adopt_cached_scans,
    discover_into_frontier_once,
    download_source_once,
    extract_source_once,
    run_discovery,
    scan_source_once,
    source_autoscale_decision,
)


def _candidate(video: str, segment: int, priority_word: str = "movie") -> dict:
    return {
        "candidate_id": f"{video}:{segment}",
        "video_id": video,
        "source_platform": "dailymotion",
        "source_url": f"https://example.test/{video}",
        "title": f"English {priority_word} scene",
        "duration_seconds": 600,
        "clip_start_seconds": float(segment * 30),
        "clip_end_seconds": float((segment + 1) * 30),
    }


def test_discovery_enqueues_unique_sources_to_high_water(
    tmp_path: Path, monkeypatch
) -> None:
    candidates = [
        _candidate("one", 0),
        _candidate("one", 1),
        _candidate("two", 0),
        _candidate("three", 0),
    ]
    monkeypatch.setattr(
        source_pipeline, "discover_candidates", lambda *args, **kwargs: candidates
    )
    settings = DiscoverySettings(
        workspace=tmp_path / "workspace",
        discovery_dir=tmp_path / "discovery",
        minimum_candidates=1,
        discovered_high_water=2,
    )

    result = discover_into_frontier_once(settings, seed=7)

    assert result["candidate_count"] == 4
    assert result["unique_source_count"] == 3
    assert result["selected_source_count"] == 2
    assert result["inserted_sources"] == 2
    connection = connect_frontier(settings.workspace)
    assert frontier_counts(connection)["discovered"] == 2
    rows = connection.execute("SELECT candidate_json FROM source_jobs").fetchall()
    # One source group keeps both of its candidate records.
    assert any('"one:1"' in row["candidate_json"] for row in rows)


def test_discovery_does_not_search_when_frontier_is_full(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    connection = connect_frontier(workspace)
    enqueue_source(connection, [_candidate("one", 0)])
    connection.close()
    searched = False

    def discover(*args, **kwargs):
        nonlocal searched
        searched = True
        return []

    monkeypatch.setattr(source_pipeline, "discover_candidates", discover)
    settings = DiscoverySettings(
        workspace=workspace,
        discovery_dir=tmp_path / "discovery",
        minimum_candidates=1,
        discovered_high_water=1,
    )

    result = discover_into_frontier_once(settings, seed=8)

    assert result["status"] == "high_water"
    assert not searched


def test_discovery_refresh_does_not_reset_existing_sources(
    tmp_path: Path, monkeypatch
) -> None:
    candidates = [_candidate("one", 0)]
    monkeypatch.setattr(
        source_pipeline, "discover_candidates", lambda *args, **kwargs: candidates
    )
    settings = DiscoverySettings(
        workspace=tmp_path / "workspace",
        discovery_dir=tmp_path / "discovery",
        minimum_candidates=1,
        discovered_high_water=2,
    )
    first = discover_into_frontier_once(settings, seed=9)
    second = discover_into_frontier_once(settings, seed=10)

    assert first["inserted_sources"] == 1
    assert second["inserted_sources"] == 0
    connection = connect_frontier(settings.workspace)
    assert frontier_counts(connection)["discovered"] == 1


def test_one_shot_discovery_advances_past_a_poison_seed(
    tmp_path: Path, monkeypatch
) -> None:
    seed_file = tmp_path / "next-seed"
    monkeypatch.setattr(
        source_pipeline,
        "discover_into_frontier_once",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("empty page")),
    )
    args = SimpleNamespace(
        workspace=tmp_path / "workspace",
        discovery_dir=tmp_path / "discovery",
        source="dailymotion",
        profile="cinematic",
        clip_seconds=30.0,
        query_count=1,
        results_per_query=1,
        search_workers=1,
        minimum_candidates=1,
        clips_per_video=16,
        source_content_minutes_per_hour=10.0,
        max_clips_per_video=60,
        discovered_high_water=10,
        scan_cache=None,
        cached_scan_high_water=4,
        seed_file=seed_file,
        seed=42,
        once=True,
        retry_seconds=1.0,
    )

    with pytest.raises(RuntimeError, match="empty page"):
        run_discovery(args)

    assert seed_file.read_text() == "43\n"


def test_discovery_adopts_positive_and_negative_scan_caches(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    cache = workspace / "source-scans"
    cache.mkdir(parents=True)
    connection = connect_frontier(workspace)
    enqueue_source(connection, [_candidate("positive", 0)])
    enqueue_source(connection, [_candidate("negative", 0)])
    (cache / "dailymotion-positive.json").write_text(
        json.dumps(
            {
                "policy": "whole_source_proxy_m2d_v1",
                "clip_seconds": 30.0,
                "regions": [
                    {
                        "start_seconds": 10.0,
                        "evidence": {
                            "foreground_speech_coverage": 0.8,
                            "vocal_music_coverage": 0.0,
                        },
                    }
                ],
                "proxy_asr": {
                    "policy": "source_proxy_asr_top3_beam1_v1",
                    "accepted": True,
                    "enforced": True,
                },
            }
        )
    )
    (cache / "dailymotion-negative.json").write_text(
        json.dumps(
            {
                "policy": "whole_source_proxy_m2d_v1",
                "clip_seconds": 30.0,
                "regions": [],
            }
        )
    )

    result = adopt_cached_scans(
        connection, cache, clip_seconds=30.0, scanned_high_water=4
    )

    assert result == {"promoted": 1, "rejected": 1, "completed": 0}
    states = dict(connection.execute("SELECT video_id,state FROM source_jobs"))
    assert states == {"positive": "scanned", "negative": "rejected"}
    outcomes = {
        row[0] for row in connection.execute("SELECT outcome FROM source_stage_events")
    }
    assert outcomes == {"cache_reused", "cache_rejected"}


def test_cache_adoption_rejects_inconclusive_enforced_asr(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    cache = workspace / "source-scans"
    cache.mkdir(parents=True)
    connection = connect_frontier(workspace)
    enqueue_source(connection, [_candidate("timeout", 0)])
    claim = claim_source(
        connection, "discovered", worker="setup-download", lease_seconds=30
    )
    assert claim is not None
    finish_source(
        connection,
        claim["source_key"],
        worker="setup-download",
        expected_state="discovered",
        next_state="downloaded",
    )
    claim = claim_source(
        connection, "downloaded", worker="setup-scan", lease_seconds=30
    )
    assert claim is not None
    finish_source(
        connection,
        claim["source_key"],
        worker="setup-scan",
        expected_state="downloaded",
        next_state="scanned",
    )
    (cache / "dailymotion-timeout.json").write_text(
        json.dumps(
            {
                "policy": "whole_source_proxy_m2d_v1",
                "clip_seconds": 30.0,
                "regions": [
                    {
                        "start_seconds": 10.0,
                        "evidence": {
                            "foreground_speech_coverage": 0.8,
                            "vocal_music_coverage": 0.0,
                        },
                    }
                ],
                "proxy_asr": {
                    "policy": "source_proxy_asr_top3_beam1_v1",
                    "accepted": None,
                    "enforced": True,
                },
            }
        )
    )

    result = adopt_cached_scans(
        connection, cache, clip_seconds=30.0, scanned_high_water=4
    )

    assert result == {"promoted": 0, "rejected": 1, "completed": 0}
    row = connection.execute("SELECT state,terminal_reason FROM source_jobs").fetchone()
    assert tuple(row) == ("rejected", "source_proxy_asr_rejected")


def test_cache_adoption_completes_sources_exhausted_in_catalog(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    cache = workspace / "source-scans"
    cache.mkdir(parents=True)
    connection = connect_frontier(workspace)
    enqueue_source(connection, [_candidate("exhausted", 0)])
    (cache / "dailymotion-exhausted.json").write_text(
        json.dumps(
            {
                "policy": "whole_source_proxy_m2d_v1",
                "clip_seconds": 30.0,
                "claimed_starts": [],
                "regions": [
                    {
                        "start_seconds": 10.0,
                        "evidence": {
                            "foreground_speech_coverage": 0.8,
                            "vocal_music_coverage": 0.0,
                        },
                    }
                ],
                "proxy_asr": {
                    "policy": "source_proxy_asr_top3_beam1_v1",
                    "accepted": True,
                    "enforced": True,
                },
            }
        )
    )

    result = adopt_cached_scans(
        connection,
        cache,
        clip_seconds=30.0,
        scanned_high_water=4,
        guidance={"exhausted": {"accepted": 1}},
    )

    assert result == {"promoted": 0, "rejected": 0, "completed": 1}
    assert connection.execute("SELECT state FROM source_jobs").fetchone()[0] == (
        "complete"
    )


def test_extraction_claims_commit_success_and_rejection_after_manifest(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "source-scans"
    cache.mkdir()
    group = [_candidate("claim-after-manifest", 0)]
    path = cache / "dailymotion-claim-after-manifest.json"
    path.write_text(json.dumps({"claimed_starts": [10.0]}))

    count = _commit_extraction_claims(
        cache,
        group,
        [
            {
                "selection": "whole_source_proxy_scan",
                "retrieval_status": "success",
                "clip_start_seconds": 20.0,
            },
            {
                "selection": "whole_source_proxy_scan",
                "retrieval_status": "rejected",
                "clip_start_seconds": 30.0,
            },
            {
                "selection": "seeded_cinematic_search",
                "retrieval_status": "rejected",
                "clip_start_seconds": 40.0,
            },
        ],
    )

    assert count == 2
    assert json.loads(path.read_text())["claimed_starts"] == [10.0, 20.0, 30.0]


def test_download_stage_atomically_publishes_a_valid_source(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    connection = connect_frontier(workspace)
    enqueue_source(connection, [_candidate("one", 0)])
    connection.close()
    monkeypatch.setattr(
        source_pipeline,
        "_preflight_source_for_scan",
        lambda candidate: {"quality_format_available": True, "format_id": "hls-720"},
    )

    def download(candidate, root):
        path = root / "source.mp4"
        path.write_bytes(b"media")
        return path, {"format_id": "hls-380"}

    monkeypatch.setattr(source_pipeline, "_download_full_source_for_scan", download)
    monkeypatch.setattr(
        source_pipeline,
        "_source_format",
        lambda path, info, client: {
            "channels": 2,
            "sample_rate_hz": 48_000,
            "format_id": info["format_id"],
            "client": client,
        },
    )
    settings = DownloadSettings(
        workspace=workspace,
        source_cache=tmp_path / "source-cache",
        downloaded_high_water=4,
    )

    result = download_source_once(settings, worker="download-0")

    assert result is not None
    assert result["status"] == "downloaded"
    assert Path(result["downloaded_path"]).read_bytes() == b"media"
    connection = connect_frontier(workspace)
    row = connection.execute("SELECT * FROM source_jobs").fetchone()
    assert row["state"] == "downloaded"
    assert row["lease_owner"] is None
    event = connection.execute("SELECT * FROM source_stage_events").fetchone()
    assert event["stage"] == "download"
    assert event["outcome"] == "success"


def test_download_stage_caches_permanent_quality_rejection(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    connection = connect_frontier(workspace)
    enqueue_source(connection, [_candidate("low-quality", 0)])
    connection.close()
    monkeypatch.setattr(
        source_pipeline,
        "_preflight_source_for_scan",
        lambda candidate: {
            "quality_format_available": False,
            "format_id": "hls-380",
        },
    )
    settings = DownloadSettings(
        workspace=workspace,
        source_cache=tmp_path / "source-cache",
    )

    result = download_source_once(settings, worker="download-0")

    assert result == {
        "status": "rejected",
        "reason": "source_high_quality_format_unavailable",
        "source_key": "dailymotion:low-quality",
    }
    connection = connect_frontier(workspace)
    row = connection.execute("SELECT * FROM source_jobs").fetchone()
    assert row["state"] == "rejected"
    assert row["terminal_reason"] == "source_high_quality_format_unavailable"


def test_download_stage_does_not_retry_unavailable_formats(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    connection = connect_frontier(workspace)
    enqueue_source(connection, [_candidate("missing-format", 0)])
    connection.close()

    def preflight(candidate):
        raise source_pipeline.subprocess.CalledProcessError(
            1,
            ["yt-dlp"],
            stderr="ERROR: Requested format is not available",
        )

    monkeypatch.setattr(source_pipeline, "_preflight_source_for_scan", preflight)
    settings = DownloadSettings(
        workspace=workspace,
        source_cache=tmp_path / "source-cache",
    )

    result = download_source_once(settings, worker="download-0")

    assert result == {
        "status": "rejected",
        "reason": "source_high_quality_format_unavailable",
        "source_key": "dailymotion:missing-format",
    }
    connection = connect_frontier(workspace)
    row = connection.execute(
        "SELECT state,stage_attempts,last_error FROM source_jobs"
    ).fetchone()
    assert tuple(row) == ("rejected", 0, None)
    details = json.loads(
        connection.execute(
            "SELECT details_json FROM source_stage_events"
        ).fetchone()[0]
    )
    assert "Requested format is not available" in details["error"]


def _downloaded_job(workspace: Path, work_dir: Path, video_id: str = "one") -> None:
    source = work_dir / "source.mp4"
    work_dir.mkdir(parents=True)
    source.write_bytes(b"source")
    connection = connect_frontier(workspace)
    enqueue_source(connection, [_candidate(video_id, 0)])
    claim = claim_source(connection, "discovered", worker="setup", lease_seconds=30)
    assert claim is not None
    finish_source(
        connection,
        claim["source_key"],
        worker="setup",
        expected_state="discovered",
        next_state="downloaded",
        updates={
            "downloaded_path": str(source),
            "download_json": {
                "download_seconds": 2.5,
                "source_format": {"channels": 2, "sample_rate_hz": 48_000},
                "extraction_format_id": "hls-720",
            },
        },
    )
    connection.close()


class FakeScanner:
    def __init__(self, *, matched: bool = True) -> None:
        self.matched = matched

    def create_proxy(self, source: Path, destination: Path) -> None:
        destination.write_bytes(b"proxy")

    def stereo_metrics(self, proxy: Path) -> dict[str, float]:
        return {"side_to_total_db": -12.0}

    def scan(self, proxy: Path, *, clip_seconds: float, max_regions: int) -> dict:
        regions = []
        if self.matched:
            regions = [
                {
                    "start_seconds": 10.0,
                    "end_seconds": 40.0,
                    "score": 8.0,
                    "evidence": {
                        "foreground_speech_coverage": 0.8,
                        "vocal_music_coverage": 0.0,
                    },
                }
            ]
        return {
            "m2d_windows": 100,
            "scan_seconds": 1.25,
            "regions": regions,
        }


def _scan_settings(tmp_path: Path) -> ScanSettings:
    return ScanSettings(
        workspace=tmp_path / "workspace",
        scan_cache=tmp_path / "scan-cache",
        proxy_asr_request_dir=tmp_path / "requests",
        proxy_asr_result_dir=tmp_path / "results",
        scanned_high_water=4,
    )


def test_scan_stage_persists_cache_and_releases_source_storage(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _scan_settings(tmp_path)
    work_dir = tmp_path / "source-work" / "one"
    _downloaded_job(settings.workspace, work_dir)
    monkeypatch.setattr(
        source_pipeline,
        "_probe_source_proxy_asr",
        lambda *args, **kwargs: {
            "status": "completed",
            "accepted": True,
            "checked_regions": [{"start_seconds": 10.0}],
        },
    )

    result = scan_source_once(settings, FakeScanner(), worker="scan-0")

    assert result is not None
    assert result["status"] == "scanned"
    assert result["passing_regions"] == 1
    assert not work_dir.exists()
    cache_files = list(settings.scan_cache.glob("*.json"))
    assert len(cache_files) == 1
    connection = connect_frontier(settings.workspace)
    row = connection.execute("SELECT * FROM source_jobs").fetchone()
    assert row["state"] == "scanned"
    event = connection.execute(
        "SELECT * FROM source_stage_events WHERE stage='scan'"
    ).fetchone()
    assert event["outcome"] == "success"


def test_scan_stage_terminally_rejects_no_match_source(tmp_path: Path) -> None:
    settings = _scan_settings(tmp_path)
    work_dir = tmp_path / "source-work" / "empty"
    _downloaded_job(settings.workspace, work_dir, video_id="empty")

    result = scan_source_once(settings, FakeScanner(matched=False), worker="scan-0")

    assert result == {
        "status": "rejected",
        "reason": "source_m2d_no_match",
        "source_key": "dailymotion:empty",
    }
    assert not work_dir.exists()
    connection = connect_frontier(settings.workspace)
    row = connection.execute("SELECT * FROM source_jobs").fetchone()
    assert row["state"] == "rejected"
    assert row["terminal_reason"] == "source_m2d_no_match"


def _scanned_job(workspace: Path, video_id: str = "one") -> None:
    connection = connect_frontier(workspace)
    enqueue_source(connection, [_candidate(video_id, 0)])
    download_claim = claim_source(
        connection, "discovered", worker="setup-download", lease_seconds=30
    )
    assert download_claim is not None
    finish_source(
        connection,
        download_claim["source_key"],
        worker="setup-download",
        expected_state="discovered",
        next_state="downloaded",
    )
    scan_claim = claim_source(
        connection, "downloaded", worker="setup-scan", lease_seconds=30
    )
    assert scan_claim is not None
    finish_source(
        connection,
        scan_claim["source_key"],
        worker="setup-scan",
        expected_state="downloaded",
        next_state="scanned",
    )
    connection.close()


def _extract_settings(tmp_path: Path) -> ExtractSettings:
    workspace = tmp_path / "workspace"
    return ExtractSettings(
        workspace=workspace,
        runs_dir=workspace / "acquisition-runs",
        scan_cache=workspace / "source-scans",
        catalog=workspace / "catalog.sqlite3",
        proxy_asr_request_dir=workspace / "requests",
        proxy_asr_result_dir=workspace / "results",
        run_target=20,
    )


def test_extract_stage_writes_rotating_manifest_and_completes_source(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _extract_settings(tmp_path)
    _scanned_job(settings.workspace)
    writer = ExtractRunWriter(settings, 0)

    def acquire(group, output_dir, **kwargs):
        audio = output_dir / "audio"
        audio.mkdir(exist_ok=True)
        (audio / "one.wav").write_bytes(b"wave")
        return [
            {
                **group[0],
                "retrieval_status": "success",
                "quality_rejections": [],
                "local_path": "audio/one.wav",
                "sha256": "a" * 64,
                "source_format": {
                    "channels": 2,
                    "sample_rate_hz": 48_000,
                    "bitrate_kbps": 128,
                },
            }
        ]

    monkeypatch.setattr(source_pipeline, "acquire_scanned_source_group", acquire)
    monkeypatch.setattr(
        source_pipeline, "_scan_group_has_remaining_work", lambda *a, **k: False
    )

    result = extract_source_once(settings, writer, worker="extract-0")

    assert result is not None
    assert result["status"] == "complete"
    assert result["clips_published"] == 1
    manifest = json.loads((writer.run_dir / "manifest.json").read_text())
    assert len(manifest["records"]) == 1
    connection = connect_frontier(settings.workspace)
    row = connection.execute("SELECT * FROM source_jobs").fetchone()
    assert row["state"] == "complete"
    event = connection.execute(
        "SELECT * FROM source_stage_events WHERE stage='extract'"
    ).fetchone()
    assert event["outcome"] == "success"


def test_extract_stage_requeues_source_with_remaining_regions(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _extract_settings(tmp_path)
    _scanned_job(settings.workspace, video_id="more")
    writer = ExtractRunWriter(settings, 0)
    monkeypatch.setattr(
        source_pipeline,
        "acquire_scanned_source_group",
        lambda *args, **kwargs: [
            {"retrieval_status": "source_scan_exhausted", "video_id": "more"}
        ],
    )
    monkeypatch.setattr(
        source_pipeline, "_scan_group_has_remaining_work", lambda *a, **k: True
    )

    result = extract_source_once(settings, writer, worker="extract-0")

    assert result is not None
    assert result["status"] == "scanned"
    connection = connect_frontier(settings.workspace)
    row = connection.execute("SELECT * FROM source_jobs").fetchone()
    assert row["state"] == "scanned"
    assert row["lease_owner"] is None


def test_extract_stage_preserves_nested_acquisition_error(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _extract_settings(tmp_path)
    _scanned_job(settings.workspace, video_id="cdn-failure")
    writer = ExtractRunWriter(settings, 0)
    monkeypatch.setattr(
        source_pipeline,
        "acquire_scanned_source_group",
        lambda *args, **kwargs: [
            {
                "retrieval_status": "source_scan_unavailable",
                "source_scan": {"error": "HTTP Error 503"},
            }
        ],
    )

    result = extract_source_once(settings, writer, worker="extract-0")

    assert result is not None
    assert result["status"] == "scanned"
    assert result["error"] == "RuntimeError: HTTP Error 503"


def _source_scale(**overrides):
    values = {
        "limits": {"download": 8, "scan": 2, "extract": 4},
        "counts": {
            "discovered": 100,
            "downloaded": 8,
            "scanned": 8,
            "complete": 0,
            "rejected": 0,
        },
        "cpu_percent": 40.0,
        "bounds": {"download": (2, 16), "scan": (1, 4), "extract": (1, 8)},
        "cpu_low": 55.0,
        "cpu_high": 85.0,
        "scan_backlog_high": 16,
        "extract_backlog_high": 16,
        "download_backlog_low": 4,
    }
    values.update(overrides)
    return source_autoscale_decision(**values)


def test_source_autoscaler_prioritizes_downstream_backlog() -> None:
    extract = _source_scale(
        counts={
            "discovered": 100,
            "downloaded": 20,
            "scanned": 16,
            "complete": 0,
            "rejected": 0,
        }
    )
    scan = _source_scale(
        counts={
            "discovered": 100,
            "downloaded": 16,
            "scanned": 0,
            "complete": 0,
            "rejected": 0,
        }
    )

    assert extract["limits"]["extract"] == 5
    assert extract["actions"] == ["increase_extract"]
    assert scan["limits"]["scan"] == 3
    assert scan["actions"] == ["increase_scan"]


def test_source_autoscaler_adds_supply_only_with_cpu_headroom() -> None:
    result = _source_scale(
        counts={
            "discovered": 100,
            "downloaded": 4,
            "scanned": 0,
            "complete": 0,
            "rejected": 0,
        }
    )

    assert result["limits"]["download"] == 9
    assert result["actions"] == ["increase_download"]


def test_source_autoscaler_reduces_upstream_first_under_cpu_pressure() -> None:
    result = _source_scale(cpu_percent=92.0)

    assert result["limits"] == {"download": 7, "scan": 2, "extract": 4}
    assert result["actions"] == ["reduce_download_for_cpu"]
