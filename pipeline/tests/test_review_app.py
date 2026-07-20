from __future__ import annotations

import json
import sqlite3
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from sam_audio_pipeline.review_app import (
    ContinuousProgressStore,
    PipelineProgressStore,
    ReviewStore,
    TrainingQualityOverrideUpdate,
    TrainingSnapshotStore,
    create_review_app,
)

ALICE = {"reviewer_id": "reviewer-alice", "reviewer_name": "Alice"}
BOB = {"reviewer_id": "reviewer-bob", "reviewer_name": "Bob"}


def _wav(path: Path) -> None:
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(2)
        destination.setsampwidth(2)
        destination.setframerate(48_000)
        destination.writeframes(b"\0\0\0\0" * 32)


def _dataset(tmp_path: Path) -> Path:
    audio = tmp_path / "balanced-audio"
    audio.mkdir()
    for filename in ("one.wav", "two.wav"):
        _wav(audio / filename)
    validation = {
        "background_bucket": "effects_ambience_led",
        "speech_coverage": 0.8,
        "background_coverage": 1.0,
        "overlap_coverage": 0.8,
        "vocal_music_coverage": 0.0,
        "windows": [
            {
                "top_labels": [
                    {"name": "Speech"},
                    {"name": "Vehicle"},
                ]
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "name": "Test review set",
                "balanced_listening_subset": {
                    "local_directory": "balanced-audio",
                    "filenames": ["two.wav", "one.wav"],
                },
                "records": [
                    {
                        "local_path": f"audio/{filename}",
                        "title": f"Title {filename}",
                        "source_url": f"https://example.test/{filename}",
                        "m2d_validation": validation,
                    }
                    for filename in ("one.wav", "two.wav")
                ],
            }
        )
    )
    return tmp_path


def _catalog_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "snapshot"
    dataset.mkdir()
    _dataset(dataset)
    from sam_audio_pipeline.continuous_dataset import connect

    connection = connect(tmp_path)
    for index, filename in enumerate(("one.wav", "two.wav"), start=1):
        digest = f"{index:064x}"
        record = {
            "continuous_filename": filename,
            "title": f"Catalog {filename}",
            "source_url": f"https://example.test/catalog/{filename}",
        }
        validation = {
            "accepted": True,
            "background_bucket": "effects_ambience_led",
            "windows": [],
        }
        connection.execute(
            """INSERT INTO records(
            sha256,candidate_id,filename,platform,video_id,clip_start,
            record_json,discovered_at) VALUES(?,?,?,?,?,?,?,?)""",
            (
                digest,
                f"candidate-{index}",
                filename,
                "test",
                f"video-{index}",
                0.0,
                json.dumps(record),
                "2026-07-16T00:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO m2d_scores VALUES(?,?,?,?)",
            (filename, 1, json.dumps(validation), "2026-07-16T00:00:00+00:00"),
        )
        connection.execute(
            "INSERT INTO asr_scores VALUES(?,?,?,?)",
            (filename, 1, json.dumps({"accepted": True}), "2026-07-16T00:00:00+00:00"),
        )
        connection.execute(
            "INSERT INTO accepted(sha256,accepted_at) VALUES(?,?)",
            (digest, "2026-07-16T00:00:00+00:00"),
        )
    connection.commit()
    connection.close()
    return dataset


def _training_workspace(tmp_path: Path) -> tuple[Path, str, str]:
    workspace = tmp_path / "training"
    workspace.mkdir()
    snapshot_id = "v1-00000001-00000001"
    record_id = "a" * 64
    metadata = {
        "record_id": record_id,
        "quality": {
            "bucket": "success",
            "failure_reasons": [],
            "review_reasons": [],
            "signals": {"dialogue_word_count": 5},
        },
        "source": {"title": "Test training source"},
        "separation": {
            "sam": {"verification_status": "success"},
            "reconstruction": {"similarity_score": 98.5},
        },
        "scene_description": {
            "parsed": {
                "description": "Rain and traffic fill a city street.",
                "timeline": [{"time_seconds": [0, 2], "events": ["Rain", "Traffic"]}],
            }
        },
        "dialogue_transcription": {"transcript": "We should leave now."},
    }
    connection = sqlite3.connect(workspace / "training-dataset.sqlite3")
    connection.executescript(
        """
        CREATE TABLE snapshots(
            end_sequence INTEGER PRIMARY KEY,
            snapshot_id TEXT UNIQUE,
            record_count INTEGER,
            manifest_sha256 TEXT,
            s3_prefix TEXT,
            published_at TEXT
        );
        CREATE TABLE records(
            sequence INTEGER PRIMARY KEY,
            job_id INTEGER,
            record_id TEXT,
            quality_bucket TEXT,
            record_json TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO snapshots VALUES(?,?,?,?,?,?)",
        (
            1,
            snapshot_id,
            1,
            "manifest-digest",
            f"s3://test-bucket/training/snapshots/{snapshot_id}/",
            "2026-07-19T00:00:00+00:00",
        ),
    )
    connection.execute(
        "INSERT INTO records VALUES(?,?,?,?,?)",
        (1, 1, record_id, "success", json.dumps(metadata)),
    )
    connection.commit()
    connection.close()
    snapshot_dir = workspace / "snapshots" / snapshot_id
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "READY.json").write_text(
        json.dumps(
            {
                "quality_buckets": {"success": 1},
                "verification_status": "passed",
                "immutable": True,
            }
        )
    )
    record_root = workspace / "work" / "000000000001"
    record_root.mkdir(parents=True)
    for filename in ("original.wav", "dialogue.wav", "background.wav"):
        _wav(record_root / filename)
    return workspace, snapshot_id, record_id


def test_review_app_persists_decisions_and_serves_audio(tmp_path: Path):
    dataset = _dataset(tmp_path)
    store = ReviewStore(dataset, audio_directory="balanced-audio")
    client = TestClient(create_review_app(store))

    state = client.get("/api/state").json()
    assert "clips" not in state
    assert state["summary"]["total"] == 2
    assert state["summary"]["reviewed"] == 0
    assert state["summary"]["available"] == 2
    assert client.get("/api/clips/two.wav").json()["filename"] == "two.wav"
    assert client.get("/clip/two.wav").status_code == 200
    assert client.get("/api/audio/two.wav").content.startswith(b"RIFF")

    assert client.post("/api/claims/two.wav", json=ALICE).status_code == 200
    response = client.put(
        "/api/reviews/two.wav",
        json={**ALICE, "decision": "perfect"},
    )
    assert response.status_code == 200
    assert response.json()["summary"]["perfect"] == 1

    assert client.post("/api/claims/one.wav", json=BOB).status_code == 200
    response = client.put(
        "/api/reviews/one.wav",
        json={
            **BOB,
            "decision": "not_ok",
            "reasons": ["lacking_music", "too_low_quality"],
            "note": "Mostly isolated dialogue",
        },
    )
    assert response.status_code == 200
    assert response.json()["summary"]["not_ok"] == 1

    reloaded = ReviewStore(dataset, audio_directory="balanced-audio")
    assert reloaded.reviews["two.wav"]["decision"] == "perfect"
    assert reloaded.reviews["two.wav"]["reviewer_name"] == "Alice"
    assert reloaded.reviews["one.wav"]["reasons"] == [
        "lacking_music",
        "too_low_quality",
    ]
    assert "two.wav,perfect" in client.get("/api/export.csv").text


def test_training_snapshot_review_serves_one_record_and_three_stems(
    tmp_path: Path,
) -> None:
    review = tmp_path / "review"
    review.mkdir()
    workspace, snapshot_id, record_id = _training_workspace(tmp_path)
    client = TestClient(
        create_review_app(
            ReviewStore(_dataset(review), audio_directory="balanced-audio"),
            training_store=TrainingSnapshotStore(workspace),
        )
    )

    assert client.get("/training").status_code == 200
    assert "Training Snapshot Review" in client.get("/training").text
    assert client.get("/training/balance").status_code == 200
    assert "Training Dataset Balance" in client.get("/training/balance").text
    snapshots = client.get("/api/training/snapshots").json()
    assert snapshots["total_records"] == 1
    assert snapshots["snapshots"][0]["quality_buckets"] == {"success": 1}
    record = client.get(
        f"/api/training/snapshots/{snapshot_id}/records?position=0"
    ).json()
    assert record["record_id"] == record_id
    assert record["scene_description"]["parsed"]["description"].startswith("Rain")
    assert record["dialogue_transcription"]["transcript"] == "We should leave now."
    assert client.get(f"/training/{snapshot_id}/{record_id}").status_code == 200
    for stem in ("original", "dialogue", "background"):
        response = client.get(f"/api/training/audio/{snapshot_id}/{record_id}/{stem}")
        assert response.status_code == 200
        assert response.content.startswith(b"RIFF")
    assert (
        client.get(
            f"/api/training/snapshots/{snapshot_id}/records?position=1"
        ).status_code
        == 404
    )


def test_training_quality_override_is_separate_and_preserves_automated_record(
    tmp_path: Path,
) -> None:
    review = tmp_path / "review"
    review.mkdir()
    workspace, snapshot_id, record_id = _training_workspace(tmp_path)
    database_path = workspace / "training-dataset.sqlite3"
    with sqlite3.connect(database_path) as connection:
        original_record_json = connection.execute(
            "SELECT record_json FROM records WHERE record_id=?", (record_id,)
        ).fetchone()[0]
    client = TestClient(
        create_review_app(
            ReviewStore(_dataset(review), audio_directory="balanced-audio"),
            training_store=TrainingSnapshotStore(workspace),
        )
    )
    endpoint = (
        f"/api/training/snapshots/{snapshot_id}/records/{record_id}/quality-override"
    )

    response = client.put(
        endpoint,
        json={
            "decision": "failure",
            "issues": {
                "background_spill": True,
                "dialogue_leakage": False,
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["automated_quality_bucket"] == "success"
    assert response.json()["effective_quality_bucket"] == "failure"
    assert response.json()["manual_quality_override"]["issues"] == {
        "background_spill": True,
        "dialogue_leakage": False,
    }

    record = client.get(
        f"/api/training/snapshots/{snapshot_id}/records?record_id={record_id}"
    ).json()
    assert record["quality_bucket"] == "success"
    assert record["quality"]["bucket"] == "success"
    assert record["automated_quality_bucket"] == "success"
    assert record["effective_quality_bucket"] == "failure"
    assert record["manual_quality_override"]["decision"] == "failure"

    reloaded = TrainingSnapshotStore(workspace).record(snapshot_id, record_id=record_id)
    assert reloaded["manual_quality_override"]["issues"]["dialogue_leakage"] is False
    with sqlite3.connect(database_path) as connection:
        assert (
            connection.execute(
                "SELECT record_json FROM records WHERE record_id=?", (record_id,)
            ).fetchone()[0]
            == original_record_json
        )

    response = client.delete(endpoint)
    assert response.status_code == 200
    assert response.json()["manual_quality_override"] is None
    cleared = client.get(
        f"/api/training/snapshots/{snapshot_id}/records?record_id={record_id}"
    ).json()
    assert cleared["effective_quality_bucket"] == "success"
    assert cleared["manual_quality_override"] is None


def test_training_quality_override_validates_record_and_issue_name(
    tmp_path: Path,
) -> None:
    review = tmp_path / "review"
    review.mkdir()
    workspace, snapshot_id, record_id = _training_workspace(tmp_path)
    client = TestClient(
        create_review_app(
            ReviewStore(_dataset(review), audio_directory="balanced-audio"),
            training_store=TrainingSnapshotStore(workspace),
        )
    )
    base = f"/api/training/snapshots/{snapshot_id}/records"
    assert (
        client.put(
            f"{base}/{record_id}/quality-override",
            json={"decision": "success", "issues": {"not a valid issue": False}},
        ).status_code
        == 422
    )
    assert (
        client.put(
            f"{base}/missing/quality-override",
            json={"decision": "success", "issues": {}},
        ).status_code
        == 404
    )


def test_training_balance_summary_uses_newest_records_tags_and_manual_bucket(
    tmp_path: Path,
) -> None:
    review = tmp_path / "review"
    review.mkdir()
    workspace, snapshot_id, record_id = _training_workspace(tmp_path)
    database_path = workspace / "training-dataset.sqlite3"
    with sqlite3.connect(database_path) as connection:
        metadata = json.loads(
            connection.execute(
                "SELECT record_json FROM records WHERE record_id=?", (record_id,)
            ).fetchone()[0]
        )
        metadata["source"].update(
            {
                "source_platform": "dailymotion",
                "video_id": "video-1",
            }
        )
        metadata["background_tagger"] = {
            "background_bucket": "effects_ambience_led",
            "cinematic_music_coverage": 0.2,
            "cinematic_sfx_coverage": 0.8,
            "overlap_coverage": 0.7,
        }
        metadata["dialogue_transcription"].update(
            {"word_count": 5, "duration_after_vad_seconds": 15.0}
        )
        metadata["scene_description"]["parsed"].update(
            {
                "global_tags": ["Footsteps", "Rain"],
                "sound_effects": ["Footsteps", "Door"],
            }
        )
        connection.executescript(
            """
            CREATE TABLE caption_revision_state(
                key TEXT PRIMARY KEY,value TEXT,updated_at TEXT
            );
            CREATE TABLE jobs(
                id INTEGER PRIMARY KEY,separation_status TEXT,description_json TEXT
            );
            CREATE TABLE caption_v2_records(
                record_id TEXT,quality_bucket TEXT,status TEXT,metadata_json TEXT,
                uploaded_at TEXT
            );
            """
        )
        connection.executemany(
            "INSERT INTO caption_revision_state VALUES(?,?,?)",
            [
                ("active_revision", "4", "now"),
                ("target_source_sequence", "1", "now"),
            ],
        )
        connection.execute(
            "INSERT INTO caption_v2_records VALUES(?,?,?,?,?)",
            (record_id, "success", "complete", json.dumps(metadata), "published"),
        )
        connection.commit()

    store = TrainingSnapshotStore(workspace)
    store.set_quality_override(
        snapshot_id,
        record_id,
        TrainingQualityOverrideUpdate(
            decision="failure", issues={"background_spill": False}
        ),
    )
    summary = store._read_balance_summary()

    assert summary["records"] == 1
    assert summary["automated_buckets"][0]["label"] == "success"
    assert summary["effective_buckets"][0]["label"] == "failure"
    assert summary["scopes"]["success"]["records"] == 0
    failure = summary["scopes"]["failure"]
    assert failure["records"] == 1
    assert failure["unique_source_videos"] == 1
    assert {item["label"] for item in failure["tags"]} == {
        "Footsteps",
        "Rain",
        "Door",
    }
    assert (
        next(item for item in failure["tags"] if item["label"] == "Footsteps")["role"]
        == "both"
    )
    assert (
        next(
            item
            for item in failure["families"]
            if item["label"] == "Footsteps and movement"
        )["records"]
        == 1
    )
    assert failure["dialogue_words"][0]["label"] == "Sparse (0–19 words)"
    assert failure["dialogue_words"][0]["records"] == 1

    store._cached_balance_summary = summary
    store._balance_expires_at = float("inf")
    client = TestClient(
        create_review_app(
            ReviewStore(_dataset(review), audio_directory="balanced-audio"),
            training_store=store,
        )
    )
    response = client.get("/api/training/balance")
    assert response.status_code == 200
    assert response.json()["scopes"]["failure"]["records"] == 1
    assert "tags" not in response.json()["scopes"]["failure"]
    tags = client.get(
        "/api/training/balance/tags",
        params={"scope": "failure", "query": "foot", "role": "both"},
    ).json()
    assert tags["total"] == 1
    assert tags["items"][0]["label"] == "Footsteps"


def test_training_transformation_summary_counts_stages_hours_and_latest_buckets(
    tmp_path: Path,
):
    workspace, _, _ = _training_workspace(tmp_path)
    connection = sqlite3.connect(workspace / "training-dataset.sqlite3")
    connection.executescript(
        """
        CREATE TABLE jobs(
            id INTEGER PRIMARY KEY,
            separation_status TEXT NOT NULL,
            description_json TEXT
        );
        CREATE TABLE caption_revision_state(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE caption_v2_records(
            sequence INTEGER PRIMARY KEY,
            status TEXT NOT NULL,
            quality_bucket TEXT,
            uploaded_at TEXT
        );
        """
    )
    connection.executemany(
        "INSERT INTO jobs VALUES(?,?,?)",
        [
            (1, "complete", None),
            (2, "complete", None),
            (3, "pending", None),
            (
                4,
                "complete",
                json.dumps(
                    {
                        "schema_version": 3,
                        "policy": "af_next_description_timeline_v3",
                    }
                ),
            ),
        ],
    )
    connection.execute(
        "INSERT INTO caption_revision_state VALUES('active_revision','3','now')"
    )
    connection.execute(
        """INSERT INTO caption_revision_state
        VALUES('target_source_sequence','1','now')"""
    )
    connection.execute(
        "INSERT INTO records VALUES(?,?,?,?,?)",
        (2, 4, "post-target", "failure", "{}"),
    )
    connection.executemany(
        "INSERT INTO caption_v2_records VALUES(?,?,?,?)",
        [
            (1, "complete", "success", "published"),
            (2, "pending", None, None),
        ],
    )
    connection.commit()
    connection.close()

    summary = TrainingSnapshotStore(workspace).transformation_summary(source_clips=120)

    assert summary["revision"] == 3
    assert summary["source"] == {"clips": 120, "audio_hours": 1.0}
    assert summary["registered"] == {"clips": 4, "audio_hours": 0.033}
    assert summary["stemmed"]["clips"] == 3
    assert summary["review_clips"]["clips"] == 2
    assert summary["latest_transformed"]["clips"] == 2
    assert summary["latest_published"]["clips"] == 2
    assert summary["buckets"] == {
        "success": {"clips": 1, "audio_hours": 0.008},
        "review": {"clips": 0, "audio_hours": 0.0},
        "failure": {"clips": 1, "audio_hours": 0.008},
    }


def test_training_snapshot_review_includes_caption_v2_snapshots(
    tmp_path: Path,
) -> None:
    workspace, _, _ = _training_workspace(tmp_path)
    snapshot_id = "v2-00000001-00000001"
    record_id = "caption-v2-record"
    metadata = {
        "record_id": record_id,
        "quality": {
            "bucket": "review",
            "failure_reasons": [],
            "review_reasons": ["manual_check"],
        },
        "scene_description": {
            "policy": "af_next_description_timeline_v2",
            "parsed": {
                "description": "A detailed rainy street ambience changes over time.",
                "timeline": [{"start_seconds": 0, "end_seconds": 30, "events": []}],
            },
        },
        "dialogue_transcription": {"transcript": "A test sentence."},
    }
    connection = sqlite3.connect(workspace / "training-dataset.sqlite3")
    connection.executescript(
        """
        CREATE TABLE caption_v2_snapshots(
            end_sequence INTEGER PRIMARY KEY,snapshot_id TEXT,record_count INTEGER,
            manifest_sha256 TEXT,s3_prefix TEXT,published_at TEXT
        );
        CREATE TABLE caption_snapshot_membership(
            snapshot_id TEXT,record_id TEXT,source_sequence INTEGER,job_id INTEGER,
            quality_bucket TEXT,metadata_json TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO caption_v2_snapshots VALUES(?,?,?,?,?,?)",
        (
            1,
            snapshot_id,
            1,
            "v2-digest",
            f"s3://test-bucket/training-v2/snapshots/{snapshot_id}/",
            "2026-07-19T01:00:00+00:00",
        ),
    )
    connection.execute(
        "INSERT INTO caption_snapshot_membership VALUES(?,?,?,?,?,?)",
        (snapshot_id, record_id, 1, 2, "review", json.dumps(metadata)),
    )
    connection.commit()
    connection.close()
    ready_root = workspace / "caption-v2-snapshots" / snapshot_id
    ready_root.mkdir(parents=True)
    (ready_root / "READY.json").write_text(
        json.dumps(
            {
                "quality_buckets": {"review": 1},
                "verification_status": "passed",
                "immutable": True,
            }
        )
    )
    record_root = workspace / "work" / "000000000002"
    record_root.mkdir(parents=True)
    for filename in ("original.wav", "dialogue.wav", "background.wav"):
        _wav(record_root / filename)

    store = TrainingSnapshotStore(workspace)
    snapshots = store.snapshots()
    assert snapshots["total_records"] == 2
    assert snapshots["snapshots"][0]["snapshot_id"] == snapshot_id
    assert snapshots["snapshots"][0]["dataset_version"] == 2
    record = store.record(snapshot_id, position=0)
    assert record["record_id"] == record_id
    assert record["quality_bucket"] == "review"
    local, bucket, key = store.audio_location(snapshot_id, record_id, "background")
    assert local == record_root / "background.wav"
    assert bucket == "test-bucket"
    assert f"/{snapshot_id}/review/{record_id}/background.wav" in f"/{key}"


def test_catalog_review_claims_one_random_clip_without_loading_catalog(
    tmp_path: Path,
) -> None:
    store = ReviewStore(_catalog_dataset(tmp_path), audio_directory="balanced-audio")
    client = TestClient(create_review_app(store))

    state = client.get("/api/state").json()
    assert state["summary"]["total"] == 2
    assert "clips" not in state
    assert store.filenames == []
    response = client.post("/api/claims/next", json=ALICE)
    assert response.status_code == 200
    payload = response.json()
    assert payload["clip"]["filename"] == payload["filename"]
    assert payload["clip"]["title"].startswith("Catalog ")
    assert payload["clip"]["claim"]["reviewer_id"] == ALICE["reviewer_id"]


def test_progress_dashboard_reports_live_pipeline_funnel(tmp_path: Path):
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    dataset = _dataset(review_dir)
    batch = tmp_path / "cinematic-dm-raw-20260715"
    batch.mkdir()
    (batch / "attempts.jsonl").write_text("{}\n{}\n{}\n")
    (batch / "manifest.json").write_text(
        json.dumps(
            {
                "target_records": 10,
                "accepted_records": 3,
                "records": [
                    {"local_path": f"audio/{name}"}
                    for name in ("one.wav", "two.wav", "bad.wav")
                ],
            }
        )
    )
    (batch / "m2d-validation.jsonl").write_text(
        "".join(
            json.dumps({"filename": name, "accepted": accepted}) + "\n"
            for name, accepted in (
                ("one.wav", True),
                ("two.wav", True),
                ("bad.wav", False),
            )
        )
    )
    (batch / "asr-validation.jsonl").write_text(
        json.dumps({"filename": "one.wav", "accepted": True}) + "\n"
    )
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "audit.json").write_text(
        json.dumps({"record_count": 1, "all_requirements_pass": False})
    )
    progress = PipelineProgressStore([batch], final_dir=final_dir, target=1000)
    client = TestClient(
        create_review_app(
            ReviewStore(dataset, audio_directory="balanced-audio"), progress
        )
    )

    assert client.get("/progress").status_code == 200
    assert "Dataset Pipeline Progress" in client.get("/progress").text
    assert "Download workers" in client.get("/progress").text
    payload = client.get("/api/progress").json()
    assert payload["stage"] == "downloading"
    assert payload["active_stages"] == ["acquisition", "m2d", "speech"]
    assert payload["batches"][0]["workers"] == {
        "acquisition": "running",
        "m2d": "running",
        "speech": "running",
    }
    assert payload["totals"] == {
        "attempts": 3,
        "downloaded": 3,
        "m2d_scored": 3,
        "m2d_accepted": 2,
        "asr_scored": 1,
        "asr_accepted": 1,
        "combined_eligible": 1,
    }
    assert payload["final"]["materialized"] == 1
    assert payload["review_snapshot"]["materialized"] == 2

    (final_dir / "audit.json").write_text(
        json.dumps({"record_count": 1000, "all_requirements_pass": True})
    )
    completed = client.get("/api/progress").json()
    assert completed["stage"] == "complete"
    assert completed["active_stages"] == []

    with (batch / "attempts.jsonl").open("a") as destination:
        destination.write("{}\n")
    assert client.get("/api/progress").json()["totals"]["attempts"] == 4

    empty_future_batch = tmp_path / "cinematic-dm-raw-20260716"
    empty_future_batch.mkdir()
    future_progress = PipelineProgressStore([batch, empty_future_batch])
    assert future_progress.snapshot()["batches"][1]["status"] == "waiting"
    assert future_progress.snapshot()["stage"] == "downloading"


def test_progress_dashboard_is_optional(tmp_path: Path):
    client = TestClient(
        create_review_app(
            ReviewStore(_dataset(tmp_path), audio_directory="balanced-audio")
        )
    )
    assert client.get("/progress").status_code == 404
    assert client.get("/api/progress").status_code == 404


def test_continuous_progress_store_serves_cached_snapshot(monkeypatch, tmp_path: Path):
    calls = 0

    def snapshot(workspace: Path, snapshot_size: int) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "workspace": str(workspace),
            "snapshot_size": snapshot_size,
            "counts": {"accepted": 42},
        }

    monkeypatch.setattr(
        "sam_audio_pipeline.continuous_dataset.progress_snapshot", snapshot
    )
    store = ContinuousProgressStore(
        tmp_path,
        snapshot_size=2500,
        refresh_seconds=60,
    )

    assert store.snapshot()["snapshot_size"] == 2500
    assert store.snapshot()["snapshot_size"] == 2500
    assert calls == 1

    review_root = tmp_path / "review"
    review_root.mkdir()
    dataset = _dataset(review_root)
    client = TestClient(
        create_review_app(
            ReviewStore(dataset, audio_directory="balanced-audio"),
            store,
        )
    )
    assert client.get("/api/progress").json()["review_snapshot"]["materialized"] == 42
    monkeypatch.setattr(
        store,
        "download_worker_detail",
        lambda worker: {"worker": worker, "activity": "idle"},
    )
    assert client.get("/api/progress/download-workers/download-7").json() == {
        "worker": "download-7",
        "activity": "idle",
    }
    assert client.get("/api/progress/download-workers/not-a-worker").status_code == 404


def test_continuous_progress_store_starts_from_persisted_snapshot(
    monkeypatch, tmp_path: Path
):
    cached = {
        "mode": "continuous",
        "updated_at": "cached",
        "counts": {"accepted": 7},
        "source_frontier": {"discovery_strategies": {"lane": {"sources": 3}}},
    }
    (tmp_path / ContinuousProgressStore.CACHE_FILENAME).write_text(json.dumps(cached))
    calls: list[dict[str, object]] = []

    def snapshot(
        workspace: Path, snapshot_size: int, **kwargs: object
    ) -> dict[str, object]:
        calls.append(kwargs)
        return {
            **cached,
            "updated_at": "fresh",
            "snapshot_size": snapshot_size,
        }

    monkeypatch.setattr(
        "sam_audio_pipeline.continuous_dataset.progress_snapshot", snapshot
    )
    store = ContinuousProgressStore(tmp_path, refresh_seconds=60)

    assert calls == []
    assert store.snapshot()["updated_at"] == "cached"


def test_review_store_discovers_new_manifest_records_without_restart(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    store = ReviewStore(dataset, audio_directory="balanced-audio")
    assert store.state()["summary"]["total"] == 2

    _wav(dataset / "balanced-audio" / "three.wav")
    manifest = json.loads((dataset / "manifest.json").read_text())
    manifest.pop("balanced_listening_subset")
    manifest["records"].append(
        {
            "local_path": "balanced-audio/three.wav",
            "title": "New streamed clip",
            "m2d_validation": {},
        }
    )
    temporary = dataset / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest))
    temporary.replace(dataset / "manifest.json")

    state = store.state()
    assert state["summary"]["total"] == 3
    assert store.clip("three.wav")["filename"] == "three.wav"


def test_not_ok_requires_reason_and_other_requires_note(tmp_path: Path):
    client = TestClient(
        create_review_app(
            ReviewStore(_dataset(tmp_path), audio_directory="balanced-audio")
        )
    )
    assert (
        client.put(
            "/api/reviews/one.wav", json={**ALICE, "decision": "not_ok"}
        ).status_code
        == 422
    )
    response = client.put(
        "/api/reviews/one.wav",
        json={**ALICE, "decision": "not_ok", "reasons": ["other"]},
    )
    assert response.status_code == 422
    assert "requires a note" in response.json()["detail"]


def test_clear_review_returns_clip_to_unreviewed(tmp_path: Path):
    client = TestClient(
        create_review_app(
            ReviewStore(_dataset(tmp_path), audio_directory="balanced-audio")
        )
    )
    client.post("/api/claims/one.wav", json=ALICE)
    client.put("/api/reviews/one.wav", json={**ALICE, "decision": "good"})
    response = client.request("DELETE", "/api/reviews/one.wav", json=ALICE)
    assert response.status_code == 200
    assert response.json()["summary"]["unreviewed"] == 2


def test_two_reviewers_receive_distinct_atomic_random_claims(tmp_path: Path):
    store = ReviewStore(_dataset(tmp_path), audio_directory="balanced-audio")

    def claim(identity: dict[str, str]) -> str | None:
        from sam_audio_pipeline.review_app import ClaimNextRequest

        return store.claim_next(ClaimNextRequest(**identity))

    with ThreadPoolExecutor(max_workers=2) as executor:
        alice_future = executor.submit(claim, ALICE)
        bob_future = executor.submit(claim, BOB)
    alice_clip = alice_future.result()
    bob_clip = bob_future.result()
    assert alice_clip is not None
    assert bob_clip is not None
    assert alice_clip != bob_clip
    assert store.claims[alice_clip]["reviewer_name"] == "Alice"
    assert store.claims[bob_clip]["reviewer_name"] == "Bob"


def test_other_reviewer_cannot_take_or_submit_claimed_clip(tmp_path: Path):
    client = TestClient(
        create_review_app(
            ReviewStore(_dataset(tmp_path), audio_directory="balanced-audio")
        )
    )
    assert client.post("/api/claims/one.wav", json=ALICE).status_code == 200
    conflict = client.post("/api/claims/one.wav", json=BOB)
    assert conflict.status_code == 409
    assert "Alice" in conflict.json()["detail"]
    submit = client.put("/api/reviews/one.wav", json={**BOB, "decision": "good"})
    assert submit.status_code == 409


def test_release_and_expired_claim_return_clip_to_queue(tmp_path: Path):
    store = ReviewStore(_dataset(tmp_path), audio_directory="balanced-audio")
    client = TestClient(create_review_app(store))
    client.post("/api/claims/one.wav", json=ALICE)
    response = client.request("DELETE", "/api/claims/one.wav", json=ALICE)
    assert response.status_code == 200
    assert client.post("/api/claims/one.wav", json=BOB).status_code == 200

    store.claims["one.wav"]["expires_at"] = "2000-01-01T00:00:00+00:00"
    assert client.post("/api/claims/one.wav", json=ALICE).status_code == 200
