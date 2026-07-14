from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sam_audio_pipeline.source_frontier import (
    claim_source,
    connect_frontier,
    enqueue_source,
    finish_source,
    frontier_counts,
    frontier_snapshot,
    heartbeat_worker,
    release_worker_leases,
    retry_source,
)


def candidate(video_id: str) -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"{video_id}:0",
            "video_id": video_id,
            "source_platform": "dailymotion",
            "source_url": f"https://example.test/{video_id}",
            "clip_start_seconds": 0.0,
        }
    ]


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
        details={"bytes": 123},
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
        "outcomes": {"success": 1},
        "duration_p50_seconds": 12.0,
        "duration_p95_seconds": 12.0,
    }


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

    stage = frontier_snapshot(tmp_path, window_minutes=15, now=130)["stages"][
        "scan"
    ]

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
    assert current["workers"][0]["details"] == {"last_status": "high_water"}
    assert stale["workers"][0]["state"] == "stale"
