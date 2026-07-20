from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sam_audio_pipeline.source_frontier import (
    claim_source,
    connect_frontier,
    discovery_strategy_admission,
    discovery_strategy_snapshot,
    enqueue_source,
    finish_source,
    frontier_counts,
    frontier_snapshot,
    heartbeat_worker,
    provider_circuit_record_failure,
    provider_circuit_record_success,
    provider_circuit_snapshot,
    release_worker_leases,
    retry_source,
)


def test_new_discovery_lane_is_probe_limited_then_suspended_on_zero_yield() -> None:
    probing = discovery_strategy_admission(
        {
            "key": "deep_page_v1",
            "active_sources": 3,
            "scan_evaluated_sources": 4,
            "scan_passed_sources": 1,
            "final_records": 0,
            "final_accepted": 0,
        }
    )
    suspended = discovery_strategy_admission(
        {
            "key": "deep_page_v1",
            "active_sources": 0,
            "scan_evaluated_sources": 12,
            "scan_passed_sources": 0,
            "final_records": 0,
            "final_accepted": 0,
        }
    )

    assert probing == {
        "state": "probing",
        "reason": "awaiting_downstream_sample",
        "new_source_allowance": 5,
    }
    assert suspended["state"] == "suspended"
    assert suspended["reason"] == "low_scan_pass_rate"
    weak_query = discovery_strategy_admission(
        {
            "key": "bilibili:query_family:cinematic_broad_v1",
            "scan_evaluated_sources": 12,
            "scan_passed_sources": 0,
        }
    )
    assert weak_query["state"] == "suspended"


def test_discovery_strategy_snapshot_attributes_frontier_sources(
    tmp_path: Path,
) -> None:
    connection = connect_frontier(tmp_path)
    item = candidate("deep")
    item[0]["discovery_quality_key"] = "dailymotion:deep_page_v1"
    enqueue_source(connection, item)
    connection.execute(
        """UPDATE source_jobs SET state='rejected',scan_json='{}' """
        """WHERE source_key='dailymotion:deep'"""
    )

    snapshot = discovery_strategy_snapshot(connection, platform="dailymotion")

    metrics = snapshot["dailymotion:deep_page_v1"]
    assert metrics["scan_evaluated_sources"] == 1
    assert metrics["scan_passed_sources"] == 0
    assert metrics["scan_pass_rate_percent"] == 0.0
    connection.close()


def test_discovery_strategy_snapshot_includes_final_acceptance(tmp_path: Path) -> None:
    connection = connect_frontier(tmp_path)
    catalog_path = tmp_path / "catalog.sqlite3"
    catalog = sqlite3.connect(catalog_path)
    catalog.executescript(
        """
        CREATE TABLE records(sha256 TEXT,platform TEXT,record_json TEXT);
        CREATE TABLE accepted(sha256 TEXT);
        """
    )
    record = '{"discovery_quality_key":"accepted_related_v1"}'
    catalog.executemany(
        "INSERT INTO records VALUES(?,?,?)",
        (("one", "dailymotion", record), ("two", "dailymotion", record)),
    )
    catalog.execute("INSERT INTO accepted VALUES('one')")
    catalog.commit()
    catalog.close()

    snapshot = discovery_strategy_snapshot(
        connection, catalog_path=catalog_path, platform="dailymotion"
    )

    assert snapshot["accepted_related_v1"]["final_records"] == 2
    assert snapshot["accepted_related_v1"]["final_accepted"] == 1
    assert snapshot["accepted_related_v1"]["final_acceptance_percent"] == 50.0
    connection.close()


def candidate(video_id: str) -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"{video_id}:0",
            "video_id": video_id,
            "source_platform": "dailymotion",
            "source_url": f"https://example.test/{video_id}",
            "duration_seconds": 600.0,
            "clip_start_seconds": 0.0,
        }
    ]


def test_stage_events_have_time_first_dashboard_index(tmp_path: Path) -> None:
    connection = connect_frontier(tmp_path)

    indexes = {
        str(row[1]): row
        for row in connection.execute("PRAGMA index_list(source_stage_events)")
    }
    columns = [
        str(row[2])
        for row in connection.execute(
            "PRAGMA index_info(source_stage_events_finished_at)"
        )
    ]

    assert "source_stage_events_finished_at" in indexes
    assert columns == ["finished_at"]
    connection.close()


def test_enqueue_is_idempotent_without_resetting_state(tmp_path: Path) -> None:
    connection = connect_frontier(tmp_path)
    assert enqueue_source(connection, candidate("movie"), priority=1, now=10)
    claim = claim_source(
        connection, "discovered", worker="downloader-0", lease_seconds=30, now=20
    )
    assert claim is not None
    finish_source(
        connection,
        claim["source_key"],
        worker="downloader-0",
        expected_state="discovered",
        next_state="downloaded",
        updates={"downloaded_path": "/cache/movie.mp4"},
        now=22,
    )

    enqueue_source(connection, candidate("movie"), priority=9, now=30)
    row = connection.execute("SELECT * FROM source_jobs").fetchone()
    assert row["state"] == "downloaded"
    assert row["priority"] == 9
    assert row["downloaded_path"] == "/cache/movie.mp4"
    assert frontier_counts(connection)["downloaded"] == 1


def test_claims_are_exclusive_and_expired_leases_recover(tmp_path: Path) -> None:
    first = connect_frontier(tmp_path)
    second = connect_frontier(tmp_path)
    enqueue_source(first, candidate("movie"), now=10)

    claim = claim_source(
        first, "discovered", worker="downloader-0", lease_seconds=10, now=20
    )
    assert claim is not None
    assert (
        claim_source(
            second, "discovered", worker="downloader-1", lease_seconds=10, now=25
        )
        is None
    )
    recovered = claim_source(
        second, "discovered", worker="downloader-1", lease_seconds=10, now=31
    )
    assert recovered is not None
    assert recovered["source_key"] == claim["source_key"]
    with pytest.raises(RuntimeError, match="no longer owned"):
        finish_source(
            first,
            claim["source_key"],
            worker="downloader-0",
            expected_state="discovered",
            next_state="downloaded",
            now=32,
        )


def test_claims_balance_in_flight_workers_across_platforms(tmp_path: Path) -> None:
    connection = connect_frontier(tmp_path)
    youtube = candidate("youtube-high")
    youtube[0]["source_platform"] = "youtube"
    vimeo = candidate("vimeo-low")
    vimeo[0]["source_platform"] = "vimeo"
    enqueue_source(connection, youtube, priority=100, now=10)
    enqueue_source(connection, vimeo, priority=1, now=11)

    first = claim_source(
        connection, "discovered", worker="download-0", lease_seconds=100, now=20
    )
    second = claim_source(
        connection, "discovered", worker="download-1", lease_seconds=100, now=20
    )

    assert first is not None
    assert first["platform"] == "youtube"
    assert second is not None
    assert second["platform"] == "vimeo"


def test_claims_use_provider_weights_for_active_leases(tmp_path: Path) -> None:
    connection = connect_frontier(tmp_path)
    for index in range(4):
        dailymotion = candidate(f"dm-{index}")
        youtube = candidate(f"yt-{index}")
        youtube[0]["source_platform"] = "youtube"
        enqueue_source(connection, dailymotion, now=10 + index)
        enqueue_source(connection, youtube, now=20 + index)

    claimed = [
        claim_source(
            connection,
            "discovered",
            worker=f"download-{index}",
            lease_seconds=100,
            now=30,
            provider_weights={"dailymotion": 4, "youtube": 1},
        )
        for index in range(5)
    ]

    assert all(item is not None for item in claimed)
    assert [item["platform"] for item in claimed].count("dailymotion") == 4
    assert [item["platform"] for item in claimed].count("youtube") == 1


def test_claims_enforce_provider_active_transfer_cap(tmp_path: Path) -> None:
    connection = connect_frontier(tmp_path)
    for index in range(6):
        youtube = candidate(f"yt-cap-{index}")
        youtube[0]["source_platform"] = "youtube"
        vimeo = candidate(f"vimeo-cap-{index}")
        vimeo[0]["source_platform"] = "vimeo"
        enqueue_source(connection, youtube, priority=100, now=10 + index)
        enqueue_source(connection, vimeo, priority=1, now=20 + index)

    claimed = [
        claim_source(
            connection,
            "discovered",
            worker=f"download-{index}",
            lease_seconds=100,
            now=30,
            provider_weights={"youtube": 10, "vimeo": 1},
            provider_max_active={"youtube": 2},
        )
        for index in range(6)
    ]

    assert all(item is not None for item in claimed)
    assert [item["platform"] for item in claimed].count("youtube") == 2
    assert [item["platform"] for item in claimed].count("vimeo") == 4


def test_claims_rate_limit_one_provider_without_idling_others(tmp_path: Path) -> None:
    connection = connect_frontier(tmp_path)
    for index in range(3):
        youtube = candidate(f"yt-rate-{index}")
        youtube[0]["source_platform"] = "youtube"
        vimeo = candidate(f"vimeo-rate-{index}")
        vimeo[0]["source_platform"] = "vimeo"
        enqueue_source(connection, youtube, priority=100, now=10 + index)
        enqueue_source(connection, vimeo, priority=1, now=20 + index)

    first = claim_source(
        connection,
        "discovered",
        worker="download-0",
        lease_seconds=100,
        now=30,
        provider_min_claim_interval_seconds={"youtube": 1},
    )
    throttled = claim_source(
        connection,
        "discovered",
        worker="download-1",
        lease_seconds=100,
        now=30.5,
        provider_min_claim_interval_seconds={"youtube": 1},
    )
    resumed = claim_source(
        connection,
        "discovered",
        worker="download-2",
        lease_seconds=100,
        now=31,
        provider_min_claim_interval_seconds={"youtube": 1},
    )

    assert first is not None and first["platform"] == "youtube"
    assert throttled is not None and throttled["platform"] == "vimeo"
    assert resumed is not None and resumed["platform"] == "youtube"


def test_provider_circuit_skips_open_provider_and_allows_one_recovery_probe(
    tmp_path: Path,
) -> None:
    connection = connect_frontier(tmp_path)
    youtube = candidate("youtube")
    youtube[0]["source_platform"] = "youtube"
    dailymotion = candidate("dailymotion")
    enqueue_source(connection, youtube, priority=100, now=10)
    enqueue_source(connection, dailymotion, priority=1, now=11)
    opened = provider_circuit_record_failure(
        connection,
        "youtube",
        "HTTP 402",
        failure_threshold=1,
        cooldown_seconds=300,
        now=20,
    )
    assert opened["state"] == "open"

    healthy = claim_source(
        connection,
        "discovered",
        worker="download-0",
        lease_seconds=30,
        respect_provider_circuits=True,
        now=21,
    )
    assert healthy is not None
    assert healthy["platform"] == "dailymotion"
    finish_source(
        connection,
        healthy["source_key"],
        worker="download-0",
        expected_state="discovered",
        next_state="downloaded",
        now=22,
    )

    probe = claim_source(
        connection,
        "discovered",
        worker="download-1",
        lease_seconds=30,
        respect_provider_circuits=True,
        now=321,
    )
    assert probe is not None
    assert probe["platform"] == "youtube"
    assert provider_circuit_snapshot(connection, now=321)["youtube"]["state"] == (
        "half_open"
    )
    assert (
        claim_source(
            connection,
            "discovered",
            worker="download-2",
            lease_seconds=30,
            respect_provider_circuits=True,
            now=322,
        )
        is None
    )
    provider_circuit_record_success(connection, "youtube", now=323)
    circuit = provider_circuit_snapshot(connection, now=323)["youtube"]
    assert circuit["state"] == "closed"
    assert circuit["consecutive_failures"] == 0


def test_restarted_worker_reclaims_its_own_unexpired_lease(tmp_path: Path) -> None:
    connection = connect_frontier(tmp_path)
    enqueue_source(connection, candidate("movie"), now=10)
    original = claim_source(
        connection, "discovered", worker="download-0", lease_seconds=100, now=20
    )

    reclaimed = claim_source(
        connection, "discovered", worker="download-0", lease_seconds=100, now=30
    )

    assert original is not None
    assert reclaimed is not None
    assert reclaimed["source_key"] == original["source_key"]
    assert reclaimed["lease_expires_at"] == 130
    assert (
        claim_source(
            connection, "discovered", worker="download-1", lease_seconds=100, now=40
        )
        is None
    )


def test_startup_release_expires_all_claims_for_worker(tmp_path: Path) -> None:
    connection = connect_frontier(tmp_path)
    enqueue_source(connection, candidate("movie"), now=10)
    claim = claim_source(
        connection, "discovered", worker="download-0", lease_seconds=100, now=20
    )
    assert claim is not None

    released = release_worker_leases(
        connection, worker="download-0", state="discovered", now=30
    )
    reclaimed = claim_source(
        connection, "discovered", worker="download-1", lease_seconds=100, now=31
    )

    assert released == 1
    assert reclaimed is not None
    assert reclaimed["source_key"] == claim["source_key"]


def test_retry_backoff_and_exhaustion_are_durable(tmp_path: Path) -> None:
    connection = connect_frontier(tmp_path)
    enqueue_source(connection, candidate("movie"), now=10)
    claim = claim_source(
        connection, "discovered", worker="downloader", lease_seconds=10, now=20
    )
    assert claim is not None
    assert (
        retry_source(
            connection,
            claim["source_key"],
            worker="downloader",
            expected_state="discovered",
            error="temporary network failure",
            backoff_seconds=30,
            max_attempts=2,
            now=21,
        )
        == "discovered"
    )
    assert (
        claim_source(
            connection, "discovered", worker="downloader", lease_seconds=10, now=40
        )
        is None
    )
    claim = claim_source(
        connection, "discovered", worker="downloader", lease_seconds=10, now=52
    )
    assert claim is not None
    assert (
        retry_source(
            connection,
            claim["source_key"],
            worker="downloader",
            expected_state="discovered",
            error="still failing",
            backoff_seconds=60,
            max_attempts=2,
            now=53,
        )
        == "rejected"
    )
    assert frontier_counts(connection)["rejected"] == 1


def test_transition_and_snapshot_report_stage_timing(tmp_path: Path) -> None:
    connection = connect_frontier(tmp_path)
    enqueue_source(connection, candidate("movie"), now=100)
    claim = claim_source(
        connection, "discovered", worker="downloader", lease_seconds=30, now=110
    )
    assert claim is not None
    finish_source(
        connection,
        claim["source_key"],
        worker="downloader",
        expected_state="discovered",
        next_state="downloaded",
        details={"bytes": 12_000_000, "download_seconds": 2.0},
        now=122,
    )
    snapshot = frontier_snapshot(tmp_path, window_minutes=15, now=130)
    assert snapshot["enabled"] is True
    assert snapshot["counts"]["downloaded"] == 1
    assert snapshot["oldest_ready_minutes"]["downloaded"] == pytest.approx(
        8 / 60, abs=0.001
    )
    assert snapshot["stages"]["download"] == {
        "events": 1,
        "per_minute": pytest.approx(1 / 15, abs=1e-4),
        "active_events": 1,
        "active_per_minute": pytest.approx(1 / 15, abs=1e-4),
        "audio_hours_per_hour": pytest.approx(2 / 3, abs=1e-4),
        "audio_basis": "successful_source_duration",
        "outcomes": {"success": 1},
        "duration_p50_seconds": 12.0,
        "duration_p95_seconds": 12.0,
        "window_minutes": 15,
    }
    platform = snapshot["platforms"]["dailymotion"]
    assert platform["states"]["downloaded"] == 1
    assert platform["download_attempts"] == 1
    assert platform["download_success_percent"] == 100.0
    assert platform["download_megabytes_per_second"] == 6.0
    assert platform["source_audio_hours_per_wall_hour"] > 0
    throughput = snapshot["download_throughput"]
    assert throughput["completed_file_megabytes_per_second"] == pytest.approx(
        12 / 900, abs=0.001
    )
    assert throughput["source_audio_hours_per_wall_hour"] == pytest.approx(
        2 / 3, abs=0.001
    )
    assert throughput["host_network"]["receive_megabytes_per_second"] is None


def test_snapshot_excludes_cache_bookkeeping_from_work_latency(tmp_path: Path) -> None:
    connection = connect_frontier(tmp_path)
    enqueue_source(connection, candidate("cached"), now=100)
    connection.execute(
        """INSERT INTO source_stage_events(
        source_key,stage,outcome,worker,started_at,finished_at,
        duration_seconds,details_json) VALUES(?,?,?,?,?,?,?,?)""",
        (
            "dailymotion:cached",
            "scan",
            "cache_reused",
            "cache",
            "1970-01-01T00:01:50+00:00",
            "1970-01-01T00:02:00+00:00",
            0.0,
            "{}",
        ),
    )

    stage = frontier_snapshot(tmp_path, window_minutes=15, now=130)["stages"]["scan"]

    assert stage["events"] == 1
    assert stage["active_events"] == 0
    assert stage["duration_p95_seconds"] is None


def test_invalid_update_rolls_back_transition(tmp_path: Path) -> None:
    connection = connect_frontier(tmp_path)
    enqueue_source(connection, candidate("movie"), now=10)
    claim = claim_source(
        connection, "discovered", worker="downloader", lease_seconds=30, now=20
    )
    assert claim is not None
    with pytest.raises(ValueError, match="Unsupported"):
        finish_source(
            connection,
            claim["source_key"],
            worker="downloader",
            expected_state="discovered",
            next_state="downloaded",
            updates={"state": "complete"},
            now=21,
        )
    row = connection.execute("SELECT state FROM source_jobs").fetchone()
    assert row["state"] == "discovered"


def test_frontier_connections_use_wal(tmp_path: Path) -> None:
    connection = connect_frontier(tmp_path)
    mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    assert isinstance(connection, sqlite3.Connection)


def test_worker_heartbeat_becomes_stale_without_progress(tmp_path: Path) -> None:
    connection = connect_frontier(tmp_path)
    heartbeat_worker(
        connection,
        "download-0",
        stage="download",
        details={"last_status": "high_water"},
        now=100,
    )
    current = frontier_snapshot(tmp_path, now=120)
    stale = frontier_snapshot(tmp_path, now=131)

    assert current["workers"][0]["state"] == "running"
    assert current["workers"][0]["activity"] == "idle"
    assert current["workers"][0]["details"] == {"last_status": "high_water"}
    assert stale["workers"][0]["state"] == "stale"
    assert stale["workers"][0]["activity"] == "stale"


def test_active_lease_marks_long_running_worker_as_working(tmp_path: Path) -> None:
    connection = connect_frontier(tmp_path)
    enqueue_source(connection, candidate("movie"), now=80)
    heartbeat_worker(
        connection,
        "download-0",
        stage="download",
        details={"last_status": "idle"},
        now=80,
    )
    claim_source(
        connection,
        "discovered",
        worker="download-0",
        lease_seconds=120,
        now=100,
    )

    worker = frontier_snapshot(tmp_path, now=120)["workers"][0]

    assert worker["state"] == "stale"
    assert worker["activity"] == "working"
    assert worker["heartbeat_age_seconds"] == 40.0


def test_download_worker_detail_reads_live_process_lazily(tmp_path: Path) -> None:
    from sam_audio_pipeline.source_frontier import download_worker_detail

    item = candidate("movie")
    item[0]["title"] = "Movie scene"
    staging = tmp_path / ".source-download-test"
    staging.mkdir()
    (staging / "source.webm.part").write_bytes(b"x" * 2_000_000)
    proc_root = tmp_path / "proc"
    process = proc_root / "123"
    process.mkdir(parents=True)
    command = (
        "sam-media-direct-download-test yt-dlp --config-locations proxy-0042.conf "
        f"-o {staging}/source.%(ext)s https://example.test/movie"
    )
    (process / "cmdline").write_bytes(command.replace(" ", "\0").encode())
    observed_at = process.stat().st_ctime + 10.0
    connection = connect_frontier(tmp_path)
    enqueue_source(connection, item, now=observed_at - 20.0)
    heartbeat_worker(
        connection, "download-0", stage="download", now=observed_at - 10.0
    )
    claim_source(
        connection,
        "discovered",
        worker="download-0",
        lease_seconds=120,
        now=observed_at - 5.0,
    )

    detail = download_worker_detail(
        tmp_path, "download-0", now=observed_at, proc_root=proc_root
    )

    assert detail is not None
    assert detail["activity"] == "working"
    assert detail["source"]["title"] == "Movie scene"
    assert detail["transfer"]["status"] == "downloading"
    assert detail["transfer"]["bytes"] == 2_000_000
    assert detail["transfer"]["average_megabytes_per_second"] == 0.2
    assert detail["transfer"]["proxy_slot"] == 42
