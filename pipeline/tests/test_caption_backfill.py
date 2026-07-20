from __future__ import annotations

import json
from pathlib import Path

from sam_audio_pipeline.caption_backfill import (
    _claim,
    _publication_contract_reasons,
    prepare_revision,
    recover_leases,
    requeue_invalid_complete_once,
    reuse_existing_once,
    seed_once,
    status,
)
from sam_audio_pipeline.training_dataset import (
    BACKGROUND_ONLY_ENDING,
    connect,
    enqueue_job,
)


def test_publication_contract_blocks_non_current_and_hard_caption_failures() -> None:
    valid = {
        "schema_version": 4,
        "policy": "af_next_description_timeline_v4",
        "parsed": {
            "description": (
                "A stable mechanical room tone combines a low electrical hum, "
                "soft ventilation noise, and intermittent metallic vibration. "
                + "The layered texture remains quiet and spatially broad while " * 6
                + BACKGROUND_ONLY_ENDING
            ),
            "timeline": [
                {
                    "start_seconds": 0,
                    "end_seconds": 30,
                    "events": ["A stable mechanical ambience continues."],
                }
            ],
        },
    }
    assert _publication_contract_reasons(json.dumps(valid)) == []

    invalid = {
        **valid,
        "schema_version": 2,
        "parsed": {"description": "A short caption.", "timeline": []},
    }
    assert _publication_contract_reasons(json.dumps(invalid)) == [
        "scene_description_too_short",
        "scene_description_wrong_schema_version",
        "scene_timeline_empty",
    ]


def test_seed_marks_current_captions_complete_and_legacy_pending(
    tmp_path: Path,
) -> None:
    connection = connect(tmp_path)
    for index, policy in enumerate(
        ("af_next_captioner_m2d_grounded_v1", "af_next_description_timeline_v4"),
        start=1,
    ):
        enqueue_job(
            connection,
            source_key=f"source-{index}",
            source_kind="continuous",
            source_ref=f"/source-{index}.wav",
            source_sha256=str(index) * 64,
            source={},
        )
        job_id = connection.execute(
            "SELECT id FROM jobs WHERE source_key=?", (f"source-{index}",)
        ).fetchone()[0]
        description = {
            "schema_version": 4 if policy.endswith("_v4") else 1,
            "policy": policy,
            "parsed": {
                "description": (
                    "A stable mechanical room tone combines a low electrical hum, "
                    "soft ventilation noise, and intermittent metallic vibration. "
                    + "The layered texture remains quiet and spatially broad while "
                    * 6
                    + BACKGROUND_ONLY_ENDING
                ),
                "timeline": [
                    {
                        "start_seconds": 0,
                        "end_seconds": 30,
                        "events": ["A stable mechanical ambience continues."],
                    }
                ],
            },
        }
        connection.execute(
            "UPDATE jobs SET description_json=? WHERE id=?",
            (json.dumps(description), job_id),
        )
        connection.execute(
            """INSERT INTO records(job_id,record_id,quality_bucket,record_json,
            created_at,uploaded_at) VALUES(?,?,?,?,?,?)""",
            (
                job_id,
                f"record-{index}",
                "review",
                json.dumps({"record_id": f"record-{index}"}),
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
    connection.execute(
        """INSERT INTO snapshots(end_sequence,snapshot_id,record_count,
        manifest_sha256,s3_prefix,published_at) VALUES(?,?,?,?,?,?)""",
        (
            2,
            "v1-test",
            2,
            "a" * 64,
            "s3://bucket/v1/snapshots/v1-test/",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    connection.commit()
    connection.close()

    assert seed_once(tmp_path) == 2
    assert seed_once(tmp_path) == 0

    connection = connect(tmp_path)
    rows = list(
        connection.execute(
            """SELECT job_id,status,quality_bucket,description_json
            FROM caption_v2_records ORDER BY job_id"""
        )
    )
    connection.close()
    assert rows[0]["status"] == "pending"
    assert rows[0]["quality_bucket"] is None
    assert rows[0]["description_json"] is None
    assert rows[1]["status"] == "complete"
    assert rows[1]["quality_bucket"] == "review"
    assert status(tmp_path) == {
        "records": {"complete": 1, "pending": 1},
        "published": 0,
        "snapshots": 0,
    }

    connection = connect(tmp_path)
    connection.execute(
        """UPDATE caption_v2_records SET status='running',lease_owner='old',
        lease_expires_at=9999999999 WHERE status='pending'"""
    )
    connection.commit()
    connection.close()
    assert recover_leases(tmp_path) == 1
    assert status(tmp_path)["records"] == {"complete": 1, "pending": 1}

    connection = connect(tmp_path)
    with connection:
        connection.execute(
            """UPDATE caption_v2_records SET description_json=?,
            status='complete' WHERE job_id=2""",
            (
                json.dumps(
                    {
                        "schema_version": 4,
                        "policy": "af_next_description_timeline_v4",
                        "parsed": {
                            "description": "A short caption.",
                            "timeline": [],
                        },
                    }
                ),
            ),
        )
    connection.close()
    assert requeue_invalid_complete_once(tmp_path) == 1
    assert status(tmp_path)["records"] == {"pending": 2}


def test_claim_prioritizes_fresh_work_over_repeated_contract_retries(
    tmp_path: Path,
) -> None:
    connection = connect(tmp_path)
    for index in (1, 2):
        enqueue_job(
            connection,
            source_key=f"source-{index}",
            source_kind="continuous",
            source_ref=f"/source-{index}.wav",
            source_sha256=str(index) * 64,
            source={},
        )
    job_ids = [row[0] for row in connection.execute("SELECT id FROM jobs ORDER BY id")]
    connection.commit()
    connection.close()
    status(tmp_path)  # Initialize the revision tables.
    connection = connect(tmp_path)
    with connection:
        connection.executemany(
            """INSERT INTO caption_v2_records(
            job_id,source_sequence,record_id,source_quality_bucket,
            source_s3_prefix,status,attempts,created_at,updated_at)
            VALUES(?,?,?,?,?,'pending',?,?,?)""",
            (
                (job_ids[0], 1, "retry", "review", "s3://bucket/old", 5, "now", "now"),
                (job_ids[1], 2, "fresh", "review", "s3://bucket/new", 0, "now", "now"),
            ),
        )
    connection.close()

    claimed = _claim(tmp_path, "worker")

    assert claimed is not None
    assert claimed["record_id"] == "fresh"


def test_seed_honors_fixed_revision_source_boundary(tmp_path: Path) -> None:
    connection = connect(tmp_path)
    description = {
        "schema_version": 4,
        "policy": "af_next_description_timeline_v4",
        "parsed": {
            "description": (
                "A stable mechanical room tone combines a low electrical hum, "
                "soft ventilation noise, and intermittent metallic vibration. "
                + "The layered texture remains quiet and spatially broad while " * 6
                + BACKGROUND_ONLY_ENDING
            ),
            "timeline": [
                {
                    "start_seconds": 0,
                    "end_seconds": 30,
                    "events": ["A stable mechanical ambience continues."],
                }
            ],
        },
    }
    for index in (1, 2):
        enqueue_job(
            connection,
            source_key=f"bounded-{index}",
            source_kind="continuous",
            source_ref=f"/bounded-{index}.wav",
            source_sha256=str(index) * 64,
            source={},
        )
        job_id = connection.execute(
            "SELECT id FROM jobs WHERE source_key=?", (f"bounded-{index}",)
        ).fetchone()[0]
        connection.execute(
            "UPDATE jobs SET description_json=? WHERE id=?",
            (json.dumps(description), job_id),
        )
        connection.execute(
            """INSERT INTO records(job_id,record_id,quality_bucket,record_json,
            uploaded_at,created_at) VALUES(?,?,?,?,?,?)""",
            (
                job_id,
                f"bounded-record-{index}",
                "success",
                json.dumps({"record_id": f"bounded-record-{index}"}),
                "now",
                "now",
            ),
        )
    connection.execute(
        "INSERT INTO snapshots VALUES(?,?,?,?,?,?)",
        (2, "v1-bounded", 2, "a" * 64, "s3://bucket/bounded/", "now"),
    )
    connection.commit()
    connection.close()
    status(tmp_path)
    connection = connect(tmp_path)
    with connection:
        connection.execute(
            """INSERT INTO caption_revision_state(key,value,updated_at)
            VALUES('target_source_sequence','1','now')"""
        )
    connection.close()

    assert seed_once(tmp_path) == 1
    connection = connect(tmp_path)
    assert [
        row[0]
        for row in connection.execute(
            "SELECT source_sequence FROM caption_v2_records ORDER BY source_sequence"
        )
    ] == [1]
    connection.close()


def test_reuse_existing_reparses_saved_model_response_without_gpu(
    tmp_path: Path,
) -> None:
    connection = connect(tmp_path)
    enqueue_job(
        connection,
        source_key="legacy",
        source_kind="continuous",
        source_ref="/legacy.wav",
        source_sha256="a" * 64,
        source={},
    )
    job_id = connection.execute("SELECT id FROM jobs").fetchone()[0]
    raw = json.dumps(
        {
            "description": " ".join(["mechanical ambience changes gradually"] * 18),
            "timeline": [
                {
                    "start_seconds": 0,
                    "end_seconds": 15,
                    "events": "Music",
                },
                {
                    "start_seconds": 15,
                    "end_seconds": 30,
                    "events": "Silence",
                },
            ],
            "global_tags": ["machinery"],
            "music": None,
            "ambience": "industrial room",
            "sound_effects": ["metal impact"],
        }
    )
    legacy = {
        "policy": "af_next_captioner_m2d_grounded_v1",
        "raw_text": raw,
        "parsed": {"description": "Old caption."},
    }
    connection.execute(
        """UPDATE jobs SET description_json=?,separation_json=?,tag_json=?,asr_json=?
        WHERE id=?""",
        (
            json.dumps(legacy),
            json.dumps(
                {
                    "sam": {
                        "verification_status": "success",
                        "stages": {"stage1": {"verification": {}}},
                    },
                    "reconstruction": {"similarity_score": 95},
                }
            ),
            json.dumps(
                {
                    "speech_coverage": 0,
                    "vocal_music_coverage": 0,
                    "windows": [],
                }
            ),
            json.dumps(
                {
                    "transcript": "A sufficiently long spoken sentence.",
                    "accepted": True,
                    "detected_language": "en",
                }
            ),
            job_id,
        ),
    )
    connection.execute(
        """INSERT INTO records(job_id,record_id,quality_bucket,record_json,
        created_at,uploaded_at) VALUES(?,?,?,?,?,?)""",
        (
            job_id,
            "legacy-record",
            "review",
            json.dumps({"record_id": "legacy-record"}),
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    connection.execute(
        """INSERT INTO snapshots(end_sequence,snapshot_id,record_count,
        manifest_sha256,s3_prefix,published_at) VALUES(?,?,?,?,?,?)""",
        (
            1,
            "v1-test",
            1,
            "b" * 64,
            "s3://bucket/v1/snapshots/v1-test/",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    connection.commit()
    connection.close()

    assert seed_once(tmp_path) == 2  # one registration plus one CPU reuse
    assert reuse_existing_once(tmp_path) == 0

    connection = connect(tmp_path)
    row = connection.execute("SELECT * FROM caption_v2_records").fetchone()
    connection.close()
    description = json.loads(row["description_json"])
    assert row["status"] == "complete"
    assert row["attempts"] == 0
    assert description["policy"] == "af_next_description_timeline_v4"
    assert description["revision"]["original_policy"].endswith("_v1")
    assert description["parsed"]["description"].endswith(BACKGROUND_ONLY_ENDING)
    assert description["parse"]["timeline_contract_repair"] == (
        "grounded_saved_short_label_expansion_v1"
    )
    assert [item["events"] for item in description["parsed"]["timeline"]] == [
        ["Music remains audible throughout this interval."],
        ["Near-silence persists throughout this interval."],
    ]

    assert prepare_revision(tmp_path) is True
    assert prepare_revision(tmp_path) is False
    connection = connect(tmp_path)
    reset = connection.execute("SELECT * FROM caption_v2_records").fetchone()
    connection.close()
    assert reset["status"] == "reparse"
    assert json.loads(reset["description_json"])["raw_text"] == raw
    assert reset["uploaded_at"] is None
