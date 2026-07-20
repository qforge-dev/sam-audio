from __future__ import annotations

import json
import re
import sys
import wave
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from sam_audio_pipeline.training_dataset import (
    BACKGROUND_ONLY_ENDING,
    OUTPUT_FILENAMES,
    _background_evidence,
    _caption_completion,
    _caption_prompt,
    _claim,
    _description_evaluation,
    _fit_training_frame_count,
    _format_scene_description,
    _generate_caption_with_contract,
    _m2d_fallback_description,
    _m2d_known_failure_description,
    _record_root,
    _retry_transient_upstream_failure,
    _validate_record_quality,
    _verify_snapshot_manifest,
    connect,
    enqueue_job,
    import_jsonl,
    package_once,
    promote_known_failures_once,
    publish_due_once,
    quality_evaluation,
    recover_leases,
    separate_once,
    sync_inbox,
)


def test_training_frame_contract_pads_and_trims_to_exactly_30_seconds(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audio.wav"

    def write_frames(count: int) -> None:
        with wave.open(str(path), "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(48_000)
            output.writeframes(b"\1\0\2\0" * count)

    for count in (30 * 48_000 - 17, 30 * 48_000 + 23):
        write_frames(count)
        _fit_training_frame_count(path)
        with wave.open(str(path), "rb") as source:
            assert source.getnframes() == 30 * 48_000
            assert source.getnchannels() == 2
            assert source.getframerate() == 48_000


def _quality(bucket: str) -> dict[str, object]:
    return {
        "bucket": bucket,
        "failure_reasons": ["dialogue_transcript_empty"] if bucket == "failure" else [],
        "review_reasons": ["possible_dialogue_spill_in_background"]
        if bucket == "review"
        else [],
    }


def test_directory_inbox_imports_new_and_changed_files_incrementally(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    package = tmp_path / "inbox" / "package-a"
    package.mkdir(parents=True)
    (package / "one.wav").write_bytes(b"first")

    assert sync_inbox(workspace, package.parent) == 1
    assert sync_inbox(workspace, package.parent) == 0
    (package / "two.wav").write_bytes(b"second")
    assert sync_inbox(workspace, package.parent) == 1
    (package / "one.wav").write_bytes(b"changed")
    assert sync_inbox(workspace, package.parent) == 1

    connection = connect(workspace)
    rows = connection.execute("SELECT source_ref FROM jobs ORDER BY id").fetchall()
    connection.close()
    assert len(rows) == 3
    assert all(Path(row["source_ref"]).is_file() for row in rows)
    assert all(
        str(workspace / "input-packages" / "files") in row["source_ref"] for row in rows
    )


def test_expired_running_stage_is_reclaimed_after_worker_crash(
    tmp_path: Path,
) -> None:
    connection = connect(tmp_path)
    enqueue_job(
        connection,
        source_key="source-a",
        source_kind="package",
        source_ref="/tmp/source.wav",
        source_sha256="a" * 64,
        source={},
    )

    first = _claim(connection, "separation", "worker-a", lease_seconds=-1)
    second = _claim(connection, "separation", "worker-b", lease_seconds=60)

    assert first is not None
    assert second is not None
    assert second["id"] == first["id"]
    assert second["lease_owner"] == "worker-b"
    assert second["separation_attempts"] == 2
    connection.close()


def test_upstream_outage_does_not_consume_quality_attempt(tmp_path: Path) -> None:
    connection = connect(tmp_path)
    enqueue_job(
        connection,
        source_key="source-a",
        source_kind="package",
        source_ref="/tmp/source.wav",
        source_sha256="a" * 64,
        source={},
    )
    job = _claim(connection, "separation", "worker-a", lease_seconds=60)
    assert job is not None

    with connection:
        transient = _retry_transient_upstream_failure(
            connection,
            int(job["id"]),
            "separation",
            RuntimeError("ConnectError: [Errno 111] Connection refused"),
        )

    row = connection.execute(
        "SELECT separation_status,separation_attempts,attempts FROM jobs"
    ).fetchone()
    connection.close()
    assert transient is True
    assert dict(row) == {
        "separation_status": "retry",
        "separation_attempts": 0,
        "attempts": 0,
    }


def test_elastic_separation_worker_pauses_for_description_backlog(
    tmp_path: Path,
) -> None:
    connection = connect(tmp_path)
    enqueue_job(
        connection,
        source_key="ready-for-caption",
        source_kind="package",
        source_ref="/tmp/complete.wav",
        source_sha256="a" * 64,
        source={},
    )
    enqueue_job(
        connection,
        source_key="waiting-for-separation",
        source_kind="package",
        source_ref="/tmp/pending.wav",
        source_sha256="b" * 64,
        source={},
    )
    connection.execute(
        """UPDATE jobs SET separation_status='complete',tag_status='complete',
        asr_status='complete',description_status='pending'
        WHERE source_key='ready-for-caption'"""
    )
    connection.commit()
    connection.close()

    result = separate_once(
        tmp_path,
        sam_api_url="http://unused",
        bucket=None,
        worker="separation-3",
        elastic_worker_from=3,
        description_backlog_high=1,
    )

    assert result is None
    connection = connect(tmp_path)
    row = connection.execute(
        "SELECT state,details_json FROM workers WHERE worker='separation-3'"
    ).fetchone()
    pending = connection.execute(
        """SELECT separation_status FROM jobs
        WHERE source_key='waiting-for-separation'"""
    ).fetchone()[0]
    connection.close()
    assert row["state"] == "paused"
    assert json.loads(row["details_json"])["reason"] == "description_backpressure"
    assert pending == "pending"


def test_supervisor_restart_recovers_an_unexpired_lease(tmp_path: Path) -> None:
    connection = connect(tmp_path)
    enqueue_job(
        connection,
        source_key="source-a",
        source_kind="package",
        source_ref="/tmp/source.wav",
        source_sha256="a" * 64,
        source={},
    )
    claimed = _claim(connection, "separation", "dead-worker", lease_seconds=7200)
    assert claimed is not None
    connection.close()

    assert recover_leases(tmp_path) == 1
    connection = connect(tmp_path)
    row = connection.execute("SELECT * FROM jobs").fetchone()
    assert row["separation_status"] == "retry"
    assert row["lease_owner"] is None
    assert row["error"] == "worker_restarted"
    reclaimed = _claim(connection, "separation", "new-worker", lease_seconds=60)
    assert reclaimed is not None
    assert reclaimed["id"] == claimed["id"]
    connection.close()


def test_jsonl_import_retries_a_result_that_arrives_before_stage_commit(
    tmp_path: Path,
) -> None:
    connection = connect(tmp_path)
    enqueue_job(
        connection,
        source_key="source-a",
        source_kind="package",
        source_ref="/tmp/source.wav",
        source_sha256="a" * 64,
        source={},
    )
    job_id = int(connection.execute("SELECT id FROM jobs").fetchone()[0])
    results = tmp_path / "tag.jsonl"
    results.write_text(json.dumps({"filename": f"{job_id:012d}.wav"}) + "\n")

    assert import_jsonl(tmp_path, results, "tag") == 0
    assert connection.execute("SELECT byte_offset FROM offsets").fetchone()[0] == 0
    with connection:
        connection.execute(
            "UPDATE jobs SET separation_status='complete' WHERE id=?", (job_id,)
        )
    assert import_jsonl(tmp_path, results, "tag") == 1
    row = connection.execute(
        "SELECT tag_status,tag_json FROM jobs WHERE id=?", (job_id,)
    ).fetchone()
    assert row["tag_status"] == "complete"
    assert json.loads(row["tag_json"])["filename"] == f"{job_id:012d}.wav"
    connection.close()


def test_zip_inbox_never_commits_an_unsafe_partial_extraction(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    archive = inbox / "package.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.wav", b"unsafe")

    with pytest.raises(ValueError, match="Unsafe archive member"):
        sync_inbox(workspace, inbox)
    extraction_root = workspace / "input-packages"
    assert not any(
        path.is_dir() and not path.name.startswith(".")
        for path in extraction_root.iterdir()
    )

    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("audio/safe.wav", b"safe")
    assert sync_inbox(workspace, inbox) == 1
    connection = connect(workspace)
    assert connection.execute("SELECT COUNT(*) FROM packages").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    connection.close()


def _record(
    workspace: Path,
    *,
    job_id: int,
    sequence: int,
    bucket: str,
) -> None:
    root = _record_root(workspace, job_id)
    root.mkdir(parents=True)
    for filename in OUTPUT_FILENAMES:
        (root / filename).write_bytes(f"{job_id}:{filename}".encode())
    connection = connect(workspace)
    source_key = f"source:{job_id}"
    now = "2026-07-18T00:00:00+00:00"
    with connection:
        connection.execute(
            """INSERT INTO jobs(id,source_key,source_kind,source_ref,source_json,
            quality_bucket,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)""",
            (job_id, source_key, "package", "/tmp/a.wav", "{}", bucket, now, now),
        )
        connection.execute(
            """INSERT INTO records(sequence,job_id,record_id,quality_bucket,
            record_json,created_at) VALUES(?,?,?,?,?,?)""",
            (
                sequence,
                job_id,
                f"record-{job_id}",
                bucket,
                json.dumps(
                    {
                        "record_id": f"record-{job_id}",
                        "quality": _quality(bucket),
                    }
                ),
                now,
            ),
        )
    connection.close()


def test_packager_keeps_a_quality_failure_as_a_complete_record(
    tmp_path: Path,
) -> None:
    connection = connect(tmp_path)
    enqueue_job(
        connection,
        source_key="package:source-a",
        source_kind="package",
        source_ref="/tmp/source.wav",
        source_sha256="a" * 64,
        source={"filename": "source.wav"},
    )
    job_id = int(connection.execute("SELECT id FROM jobs").fetchone()[0])
    root = _record_root(tmp_path, job_id)
    root.mkdir(parents=True)
    for name in ("original.wav", "dialogue.wav", "background.wav"):
        (root / name).write_bytes(f"audio:{name}".encode())
    with connection:
        connection.execute(
            """UPDATE jobs SET separation_status='complete',tag_status='complete',
            asr_status='complete',description_status='complete',package_status='pending',
            separation_json=?,tag_json=?,asr_json=?,description_json=? WHERE id=?""",
            (
                json.dumps(
                    {
                        "sam": {"verification_status": "success"},
                        "reconstruction": {"similarity_score": 92.0},
                    }
                ),
                json.dumps({"speech_coverage": 0.0, "vocal_music_coverage": 0.0}),
                json.dumps(
                    {
                        "transcript": "",
                        "detected_language": "en",
                        "accepted": False,
                    }
                ),
                json.dumps(
                    {
                        "parsed": {
                            "description": (
                                "A detailed mechanical ambience evolves over the "
                                "full scene. "
                                + "Texture changes gradually. "
                                * 8
                                + "The requested background contains no dialogue, "
                                "intelligible speech, narration, or vocals."
                            )
                        },
                        "validation": {"review_reasons": []},
                    }
                ),
                job_id,
            ),
        )
    connection.close()

    result = package_once(tmp_path, worker="package-test")

    assert result == {"status": "complete", "job_id": job_id, "bucket": "failure"}
    connection = connect(tmp_path)
    row = connection.execute("SELECT * FROM records").fetchone()
    record = json.loads(row["record_json"])
    assert row["quality_bucket"] == "failure"
    assert "dialogue_transcript_empty" in record["quality"]["failure_reasons"]
    assert set(record["artifacts"]) == set(OUTPUT_FILENAMES)
    scene = (root / "scene_description.txt").read_text()
    assert scene.startswith("DESCRIPTION:\n")
    assert "\n\nTIMELINE:\n" in scene
    assert (root / "dialogue_transcript.txt").read_text() == "\n"
    connection.close()


class _FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_file(self, path: str, bucket: str, key: str) -> None:
        self.objects[f"s3://{bucket}/{key}"] = Path(path).read_bytes()


def test_quality_validation_rejects_mismatched_bucket(tmp_path: Path) -> None:
    _record(tmp_path, job_id=1, sequence=1, bucket="review")
    connection = connect(tmp_path)
    with connection:
        connection.execute(
            "UPDATE records SET quality_bucket='success' WHERE sequence=1"
        )
    row = connection.execute("SELECT * FROM records").fetchone()

    with pytest.raises(ValueError, match="quality mismatch"):
        _validate_record_quality(row)
    connection.close()


def test_description_contract_marks_weak_caption_for_review() -> None:
    result = _description_evaluation(
        {
            "description": "A quiet room tone.",
            "timeline": [],
            "global_tags": ["indoors"],
            "sound_effects": [],
        }
    )

    assert result["status"] == "review"
    assert result["review_reasons"] == [
        "scene_description_too_short",
        "scene_timeline_empty",
        "background_only_dialogue_assertion_missing",
        "background_only_vocal_assertion_missing",
    ]


def test_description_contract_enforces_prompt_size_limits() -> None:
    ending = (
        "The requested background contains no dialogue, intelligible speech, "
        "narration, or vocals."
    )
    too_long = _description_evaluation(
        {
            "description": " ".join(["sound"] * 141) + f" {ending}",
            "timeline": [
                {
                    "start_seconds": 0,
                    "end_seconds": 30,
                    "events": ["A detailed continuous sound remains audible."],
                }
            ],
        }
    )
    too_many_events = _description_evaluation(
        {
            "description": " ".join(["sound"] * 80) + f" {ending}",
            "timeline": [{"events": ["event"]}] * 7,
        }
    )

    assert too_long["review_reasons"] == ["scene_description_too_long"]
    assert too_many_events["review_reasons"] == ["scene_timeline_too_many_events"]


def test_description_contract_rejects_underdescribed_timeline_interval() -> None:
    result = _description_evaluation(
        {
            "description": (
                "A broad mechanical ambience combines a steady low rumble, "
                "metallic impacts, and a distant resonant drone. "
                + "The layered background remains detailed and spatially wide. " * 6
                + BACKGROUND_ONLY_ENDING
            ),
            "timeline": [
                {
                    "start_seconds": 0,
                    "end_seconds": 15,
                    "events": [
                        "A deep mechanical rumble grows beneath metallic impacts."
                    ],
                },
                {
                    "start_seconds": 15,
                    "end_seconds": 30,
                    "events": ["Additional."],
                },
            ],
        }
    )

    assert "scene_timeline_underdescribed" in result["review_reasons"]


def test_caption_normalization_bounds_description_and_timeline() -> None:
    parsed, metadata = _caption_completion(
        json.dumps(
            {
                "description": " ".join(["sound"] * 180),
                "timeline": [
                    {
                        "start_seconds": index,
                        "end_seconds": index + 1,
                        "events": f"event {index}",
                    }
                    for index in range(8)
                ],
                "global_tags": [],
                "music": None,
                "ambience": None,
                "sound_effects": [],
            }
        ),
        {},
    )

    assert metadata["format"] == "strict_json_v1"
    assert len(re.findall(r"[\w']+", parsed["description"])) == 140
    assert parsed["description"].endswith(BACKGROUND_ONLY_ENDING)
    assert len(parsed["timeline"]) == 6


def test_caption_prompt_excludes_permissive_speech_and_vocal_hints() -> None:
    tag = {
        "windows": [
            {
                "start_seconds": 0,
                "end_seconds": 2,
                "music_score": 0.8,
                "background_score": 0.9,
                "speech_score": 0.7,
                "top_labels": [
                    {"name": "Speech", "probability": 0.7},
                    {"name": "Singing", "probability": 0.6},
                    {"name": "Music", "probability": 0.5},
                    {"name": "Engine", "probability": 0.4},
                ],
            }
        ]
    }

    evidence = _background_evidence(tag)
    prompt = _caption_prompt(tag)

    assert evidence == [
        {
            "t": [0.0, 2.0],
            "tags": [["Music", 0.5], ["Engine", 0.4]],
            "music": 0.8,
            "background": 0.9,
        }
    ]
    assert '"Speech"' not in prompt
    assert '"Singing"' not in prompt
    assert "native [start-end] format" in prompt
    assert "4-6 contiguous chronological intervals" in prompt
    assert "dense timestamped acoustic analysis" in prompt


def test_sparse_static_caption_is_expanded_without_inventing_transitions() -> None:
    parsed, _ = _caption_completion(
        "[0-30] Distant traffic hum.",
        {
            "global_tags": ["Vehicle", "Traffic noise", "Rumble"],
            "windows": [],
        },
    )
    evaluation = _description_evaluation(parsed)

    assert evaluation["status"] == "success"
    assert evaluation["signals"]["description_word_count"] >= 80
    assert len(parsed["timeline"]) == 1
    assert "No separately timed transition is evident" in parsed["description"]


def test_caption_timeline_merges_adjacent_duplicate_events() -> None:
    parsed, _ = _caption_completion(
        "\n".join(
            [
                "[0-5] A suspenseful low-frequency drone continues.",
                "[5-10] A suspenseful low-frequency drone continues.",
                "[10-20] A suspenseful low-frequency drone continues.",
                "[20-30] A suspenseful low-frequency drone continues.",
            ]
        ),
        {"windows": []},
    )

    assert parsed["timeline"] == [
        {
            "start_seconds": 0.0,
            "end_seconds": 30.0,
            "events": ["A suspenseful low-frequency drone continues."],
        }
    ]
    assert parsed["description"].casefold().count(
        "suspenseful low-frequency drone"
    ) == 1


def test_caption_generation_retries_hard_contract_failure(tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def ask(
            self, audio_path: Path, prompt: str, *, max_new_tokens: int
        ) -> dict[str, str]:
            self.prompts.append(prompt)
            text = (
                "[0-30] The timestamped acoustic tagger identifies machinery."
                if len(self.prompts) == 1
                else "[0-30] A stable low mechanical hum fills the room."
            )
            return {"model": "fake", "text": text}

    client = FakeClient()
    result = _generate_caption_with_contract(
        client, tmp_path / "background.wav", {"windows": []}
    )

    assert result["attempts"] == 2
    assert result["prior_contract_failures"] == [
        ["scene_description_contains_model_boilerplate"]
    ]
    assert result["validation"]["status"] == "success"
    assert "previous attempt did not satisfy" in client.prompts[1].casefold()


def test_caption_generation_repairs_short_grounded_labels_after_retry(
    tmp_path: Path,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def ask(
            self, audio_path: Path, prompt: str, *, max_new_tokens: int
        ) -> dict[str, str]:
            self.calls += 1
            return {
                "model": "fake",
                "text": "[0-15] Music.\n[15-30] Silence.",
            }

    client = FakeClient()
    result = _generate_caption_with_contract(
        client, tmp_path / "background.wav", {"windows": []}
    )

    assert client.calls == 2
    assert result["attempts"] == 2
    assert result["validation"]["status"] == "success"
    assert result["parse"]["timeline_contract_repaired_intervals"] == 2
    assert result["parsed"]["timeline"] == [
        {
            "start_seconds": 0.0,
            "end_seconds": 15.0,
            "events": ["Music remains audible throughout this interval."],
        },
        {
            "start_seconds": 15.0,
            "end_seconds": 30.0,
            "events": ["Near-silence persists throughout this interval."],
        },
    ]


def test_native_caption_drops_truncated_terminal_event() -> None:
    parsed, _ = _caption_completion(
        "\n".join(
            (
                "<t>0-20</t> A dense mechanical rhythm continues.",
                "<t>20-30</t> Sound of heavy.",
                "<t>30-31</t> .",
                "<t>31-32</t> Sound.",
            )
        ),
        {"windows": []},
    )

    assert parsed["timeline"] == [
        {
            "start_seconds": 0.0,
            "end_seconds": 30.0,
            "events": ["A dense mechanical rhythm continues."],
        }
    ]


def test_section_caption_parser_and_public_formatter_match_contract() -> None:
    completion = """DESCRIPTION:
A low mechanical drone fills the space while thin metallic rattles move across the
stereo field. A distant engine-like pulse grows gradually, with a soft airy wash
behind it and occasional sharp impacts near the right edge. The texture becomes
denser through the middle before easing into a quieter hum near the end. The final
seconds retain a faint resonant tail and a small click at center.

TIMELINE:
- 00:00-00:08.0 — A low centered drone begins beneath light metallic rattling.
- 00:08.0-00:21.0 — The pulse grows louder and impacts cross toward the right.
- 00:21.0-00:30.0 — The texture recedes into a faint hum and short central click.
"""

    parsed, metadata = _caption_completion(completion, {"windows": []})
    rendered = _format_scene_description(parsed)

    assert metadata["format"] == "description_timeline_sections_v2"
    assert len(parsed["timeline"]) == 3
    assert rendered.startswith("DESCRIPTION:\n")
    assert "\n\nTIMELINE:\n" in rendered
    assert "- 00:00-00:08.0 — A low centered drone" in rendered
    assert "- 00:21.0-00:30.0 — The texture recedes" in rendered
    assert rendered.endswith(".\n")


def test_description_contract_rejects_classifier_boilerplate() -> None:
    result = _description_evaluation(
        {
            "description": (
                "The timestamped acoustic tagger identifies machinery. "
                + "A steady detailed mechanical texture continues. " * 10
                + BACKGROUND_ONLY_ENDING
            ),
            "timeline": [{"events": ["Mechanical sound."]}],
        }
    )

    assert "scene_description_contains_model_boilerplate" in result["review_reasons"]


def test_m2d_caption_fallback_is_useful_but_forces_failure_bucket() -> None:
    fallback = _m2d_fallback_description(
        {
            "cinematic_music_coverage": 0.5,
            "windows": [
                {
                    "start_seconds": 0,
                    "end_seconds": 2,
                    "music_score": 0.8,
                    "background_score": 0.9,
                    "speech_score": 0.0,
                    "top_labels": [
                        {"name": "Music", "probability": 0.8},
                        {"name": "Engine", "probability": 0.6},
                    ],
                }
            ],
        },
        RuntimeError("caption API unavailable"),
    )

    assert fallback["validation"]["status"] == "failure"
    assert fallback["validation"]["failure_reasons"] == [
        "audio_flamingo_generation_failed"
    ]
    assert fallback["parsed"]["timeline"][0]["events"] == ["Music", "Engine"]
    assert "no dialogue" in fallback["parsed"]["description"].casefold()


def test_known_failure_caption_fast_path_does_not_invent_caption_failure() -> None:
    result = _m2d_known_failure_description(
        {
            "windows": [
                {
                    "start_seconds": 0,
                    "end_seconds": 2,
                    "top_labels": [
                        {"name": "Engine", "probability": 0.8},
                    ],
                }
            ]
        }
    )

    assert result["parse"]["generation_skipped"] is True
    assert result["validation"].get("failure_reasons") is None
    assert result["parsed"]["timeline"][0]["events"] == ["Engine"]


def test_known_failure_promoter_skips_caption_gpu_and_preserves_viable_rows(
    tmp_path: Path,
) -> None:
    connection = connect(tmp_path)
    for source_key, sam_status in (("failed", "failure"), ("viable", "success")):
        enqueue_job(
            connection,
            source_key=source_key,
            source_kind="continuous",
            source_ref=f"/{source_key}.wav",
            source_sha256=source_key,
            source={},
        )
        connection.execute(
            """UPDATE jobs SET separation_status='complete',tag_status='complete',
            asr_status='complete',description_status='pending',package_status='pending',
            separation_json=?,tag_json=?,asr_json=? WHERE source_key=?""",
            (
                json.dumps(
                    {
                        "sam": {"verification_status": sam_status},
                        "reconstruction": {"similarity_score": 95},
                    }
                ),
                json.dumps({"windows": []}),
                json.dumps(
                    {
                        "transcript": "A valid foreground sentence exists.",
                        "accepted": True,
                        "detected_language": "en",
                    }
                ),
                source_key,
            ),
        )
    connection.commit()
    connection.close()

    result = promote_known_failures_once(tmp_path)

    connection = connect(tmp_path)
    rows = {
        row["source_key"]: row
        for row in connection.execute(
            "SELECT source_key,description_status,description_json FROM jobs"
        )
    }
    connection.close()
    assert result == {"scanned": 2, "promoted": 1}
    assert rows["failed"]["description_status"] == "complete"
    description = json.loads(rows["failed"]["description_json"])
    assert description["parse"]["generation_skipped"] is True
    assert description["known_failure_reasons"] == ["sam_voice_separation_failure"]
    assert rows["viable"]["description_status"] == "pending"
    assert rows["viable"]["description_json"] is None
    assert promote_known_failures_once(tmp_path) == {"scanned": 0, "promoted": 0}


def test_native_caption_fallback_omits_dialogue_text() -> None:
    parsed, metadata = _caption_completion(
        "\n".join(
            (
                '<t>0-1</t> Speaker 1: "secret words." Background: Traffic hum.',
                "<t>1-2</t> Background noise: A car passes from left to right.",
            )
        ),
        {
            "cinematic_music_coverage": 0.0,
            "windows": [
                {
                    "top_labels": [
                        {"name": "Traffic noise, roadway noise", "probability": 0.8}
                    ]
                }
            ],
        },
    )

    assert metadata["format"] == "audio_flamingo_native_timeline_v2"
    assert metadata["speech_mentions_omitted"] == 1
    assert "secret words" not in parsed["description"]
    assert "traffic hum" in parsed["description"].casefold()
    assert "no dialogue" in parsed["description"].casefold()
    assert parsed["timeline"][0]["events"] == ["Traffic hum."]


def test_zero_length_native_timestamps_keep_rich_model_prose() -> None:
    parsed, metadata = _caption_completion(
        "\n".join(
            (
                "<t>0-0</t> A continuous low-frequency mechanical hum fills a "
                "large reverberant room.",
                "<t>0-0</t> A sharp impact blooms into a long metallic decay.",
            )
        ),
        {
            "windows": [
                {
                    "start_seconds": 0,
                    "end_seconds": 30,
                    "top_labels": [
                        {"name": "Mechanisms", "probability": 0.9},
                        {"name": "Metallic impact", "probability": 0.8},
                    ],
                }
            ]
        },
    )

    assert metadata["timeline_fallback"] == "m2d_timestamped_non_dialogue_v1"
    assert "mechanical hum" in parsed["description"].casefold()
    assert "metallic decay" in parsed["description"].casefold()
    assert "m2d acoustic evidence" not in parsed["description"].casefold()
    assert _description_evaluation(parsed)["status"] == "success"


def test_bracket_native_caption_is_preserved_and_merged_to_six_regions() -> None:
    completion = "\n".join(
        f"[{index * 3}–{(index + 1) * 3}] A detailed mechanical event {index} "
        "moves across the stereo field and changes intensity."
        for index in range(10)
    )

    parsed, metadata = _caption_completion(completion, {"windows": []})
    rendered = _format_scene_description(parsed)

    assert metadata["format"] == "audio_flamingo_native_timeline_v2"
    assert len(parsed["timeline"]) == 5
    assert parsed["timeline"][0]["start_seconds"] == 0
    assert parsed["timeline"][-1]["end_seconds"] == 30
    assert "M2D acoustic evidence" not in parsed["description"]
    assert "event 0" in parsed["description"]
    assert "- 00:00-00:06.0" in rendered


def test_short_caption_expands_from_timed_model_events_and_covers_clip() -> None:
    completion = json.dumps(
        {
            "description": "A tense mechanical ambience builds.",
            "timeline": [
                {
                    "start_seconds": 3,
                    "end_seconds": 9,
                    "events": (
                        "A low engine drone grows beneath metallic scraping and "
                        "distant impacts."
                    ),
                },
                {
                    "start_seconds": 12,
                    "end_seconds": 24,
                    "events": (
                        "The drone becomes louder while sharp clicks cross the "
                        "stereo field."
                    ),
                },
                {
                    "start_seconds": 25,
                    "end_seconds": 28,
                    "events": (
                        "The machinery recedes into a broad resonant tail and "
                        "one final low impact."
                    ),
                },
            ],
            "global_tags": ["engine", "machinery", "metallic impact"],
            "music": None,
            "ambience": "industrial",
            "sound_effects": ["impact"],
        }
    )

    parsed, _ = _caption_completion(completion, {"windows": []})
    validation = _description_evaluation(parsed)

    assert len(re.findall(r"[\w']+", parsed["description"])) >= 80
    assert parsed["timeline"][0]["start_seconds"] == 0
    assert parsed["timeline"][-1]["end_seconds"] == 30
    assert all(
        previous["end_seconds"] == following["start_seconds"]
        for previous, following in zip(
            parsed["timeline"], parsed["timeline"][1:], strict=False
        )
    )
    assert validation["status"] == "success"


def test_long_caption_truncates_at_a_sentence_boundary() -> None:
    sentence = (
        "A layered mechanical drone carries metallic impacts across a broad "
        "reverberant stereo space. "
    )
    completion = json.dumps(
        {
            "description": sentence * 16 + "The unfinished fragment continues",
            "timeline": [
                {
                    "start_seconds": 0,
                    "end_seconds": 30,
                    "events": "The drone and impacts continue with changing weight.",
                }
            ],
            "global_tags": ["machinery", "metallic impact"],
            "music": None,
            "ambience": "reverberant machinery",
            "sound_effects": ["metallic impact"],
        }
    )

    parsed, _ = _caption_completion(completion, {"windows": []})

    body = parsed["description"].removesuffix(BACKGROUND_ONLY_ENDING).rstrip()
    assert body.endswith("space.")
    assert len(re.findall(r"[\w']+", parsed["description"])) <= 140


def test_truncated_json_recovers_background_without_transcription() -> None:
    truncated = (
        '{"description":"Two voices converse. A deep train rumble moves from '
        'left to right with metallic wheel clatter.",'
        '"timeline":[{"start_seconds":0,"end_seconds":8,'
        '"events":"A steady low mechanical rumble."}],'
        '"global_tags":["train","rumble"],"music":null,'
        '"ambience":"Reverberant rail carriage",'
        '"sound_effects":["wheel clatter"],'
        '"transcription":"unfinished direct speech'
    )

    parsed, metadata = _caption_completion(truncated, {"windows": []})

    assert metadata["format"] == "partial_json_recovery_v1"
    assert metadata["speech_mentions_omitted"] == 1
    assert "voices converse" not in parsed["description"].casefold()
    assert "deep train rumble" in parsed["description"].casefold()
    assert "no dialogue" in parsed["description"].casefold()
    assert parsed["timeline"][0]["events"] == ["A steady low mechanical rumble."]
    assert "transcription" not in parsed


def test_partial_json_salvages_valid_timeline_items_around_malformed_item() -> None:
    malformed = (
        '{"description":"A mechanical room hum shifts under repeated impacts and '
        'a broad resonant tail.","timeline":['
        '{"start_seconds":0,"end_seconds":10,"events":"A low machine hum."},'
        '{"start_seconds":10,"end_seconds":20,"events":"A broken "quote."},'
        '{"start_seconds":20,"end_seconds":30,"events":"A metallic tail fades."}'
        '],"global_tags":["machinery","impact"]'
    )

    parsed, metadata = _caption_completion(malformed, {"windows": []})

    assert metadata["format"] == "partial_json_recovery_v1"
    assert metadata["recovered_timeline_events"] == 2
    assert parsed["timeline"] == [
        {
            "start_seconds": 0.0,
            "end_seconds": 20.0,
            "events": ["A low machine hum."],
        },
        {
            "start_seconds": 20.0,
            "end_seconds": 30.0,
            "events": ["A metallic tail fades."],
        },
    ]


def test_missing_caption_timeline_uses_bounded_timestamped_m2d_evidence() -> None:
    completion = json.dumps(
        {
            "description": "A continuous mechanical environment remains active.",
            "timeline": [],
            "global_tags": ["machinery"],
            "music": None,
            "ambience": "mechanical room",
            "sound_effects": ["metal clicks"],
        }
    )
    tag = {
        "windows": [
            {
                "start_seconds": index * 3,
                "end_seconds": index * 3 + 2,
                "top_labels": [
                    {"name": "Speech", "probability": 0.9},
                    {"name": f"Machine {index}", "probability": 0.8},
                ],
            }
            for index in range(10)
        ]
    }

    parsed, metadata = _caption_completion(completion, tag)

    assert len(parsed["timeline"]) == 6
    assert all("Speech" not in item["events"] for item in parsed["timeline"])
    assert parsed["timeline"][0]["start_seconds"] == 0
    assert parsed["timeline"][-1]["end_seconds"] == 30
    assert all(
        previous["end_seconds"] == following["start_seconds"]
        for previous, following in zip(
            parsed["timeline"], parsed["timeline"][1:], strict=False
        )
    )
    assert metadata["timeline_fallback"] == "m2d_timestamped_non_dialogue_v1"
    assert metadata["grounded_timeline_events"] == 6


def test_valid_json_enforces_background_only_contract() -> None:
    completion = json.dumps(
        {
            "description": (
                "A stable mechanical hum fills the room. Indistinct human murmurs "
                "appear briefly. A metallic latch clicks on the right."
            ),
            "timeline": [
                {
                    "start_seconds": 0,
                    "end_seconds": 10,
                    "events": "Mechanical hum, human murmurs, metallic click",
                }
            ],
            "global_tags": ["machinery", "speech"],
            "music": None,
            "ambience": "Mechanical hum, murmurs",
            "sound_effects": ["metallic click", "voices"],
        }
    )

    parsed, metadata = _caption_completion(completion, {"windows": []})

    assert metadata["format"] == "strict_json_v1"
    assert metadata["speech_mentions_omitted"] >= 4
    assert "murmur" not in parsed["description"].casefold()
    assert "no dialogue" in parsed["description"].casefold()
    assert parsed["timeline"][0]["events"] == ["Mechanical hum, metallic click"]
    assert parsed["global_tags"] == ["machinery"]
    assert parsed["ambience"] == "Mechanical hum"
    assert parsed["sound_effects"] == ["metallic click"]


@pytest.mark.parametrize(
    ("speech_coverage", "transcript", "expected"),
    (
        (0.0, "This is clearly spoken dialogue.", "success"),
        (0.2, "This is clearly spoken dialogue.", "review"),
        (0.0, "", "failure"),
    ),
)
def test_three_bucket_quality_policy(
    speech_coverage: float,
    transcript: str,
    expected: str,
) -> None:
    result = quality_evaluation(
        {
            "separation_json": json.dumps(
                {
                    "sam": {"verification_status": "success"},
                    "reconstruction": {"similarity_score": 90.0},
                }
            ),
            "tag_json": json.dumps(
                {
                    "speech_coverage": speech_coverage,
                    "vocal_music_coverage": 0.0,
                }
            ),
            "asr_json": json.dumps(
                {"transcript": transcript, "detected_language": "en"}
            ),
            "description_json": json.dumps(
                {
                    "parsed": {"description": "Detailed background scene."},
                    "validation": {"review_reasons": []},
                }
            ),
        }
    )

    assert result["bucket"] == expected


def test_asr_confidence_gate_can_fail_a_nonempty_transcript() -> None:
    result = quality_evaluation(
        {
            "separation_json": json.dumps(
                {
                    "sam": {"verification_status": "success"},
                    "reconstruction": {"similarity_score": 90.0},
                }
            ),
            "tag_json": json.dumps(
                {"speech_coverage": 0.0, "vocal_music_coverage": 0.0}
            ),
            "asr_json": json.dumps(
                {
                    "transcript": "These words decoded poorly.",
                    "detected_language": "en",
                    "accepted": False,
                    "rejection_reasons": ["low_transcription_confidence"],
                }
            ),
            "description_json": json.dumps(
                {
                    "parsed": {"description": "Detailed background scene."},
                    "validation": {"review_reasons": []},
                }
            ),
        }
    )

    assert result["bucket"] == "failure"
    assert "dialogue_asr_quality_gate_failed" in result["failure_reasons"]
    assert result["signals"]["dialogue_asr_rejection_reasons"] == [
        "low_transcription_confidence"
    ]


@pytest.mark.parametrize(
    ("judge_quality", "similarity", "foreground_speech", "expected"),
    (
        (4.2545, 90.13, 0.034, "review"),
        (4.19, 90.13, 0.034, "failure"),
        (4.2545, 84.9, 0.034, "failure"),
        (4.2545, 90.13, 0.06, "failure"),
    ),
)
def test_near_threshold_sam_failure_requires_independent_clean_evidence(
    judge_quality: float,
    similarity: float,
    foreground_speech: float,
    expected: str,
) -> None:
    result = quality_evaluation(
        {
            "separation_json": json.dumps(
                {
                    "sam": {
                        "verification_status": "failure",
                        "stages": {
                            "stage1": {
                                "verification": {
                                    "judge_quality_score": judge_quality,
                                }
                            }
                        },
                    },
                    "reconstruction": {"similarity_score": similarity},
                }
            ),
            "tag_json": json.dumps(
                {
                    "speech_coverage": 1.0,
                    "strong_speech_coverage": 0.0,
                    "foreground_speech_coverage": foreground_speech,
                    "synthetic_speech_coverage": 0.0,
                    "vocal_music_coverage": 0.0,
                }
            ),
            "asr_json": json.dumps(
                {
                    "transcript": "This is clearly spoken dialogue.",
                    "detected_language": "en",
                    "accepted": True,
                }
            ),
            "description_json": json.dumps(
                {
                    "parsed": {"description": "Detailed background scene."},
                    "validation": {"review_reasons": []},
                }
            ),
        }
    )

    assert result["bucket"] == expected
    assert result["signals"]["sam_near_failure_threshold"] is (expected == "review")
    if expected == "review":
        assert "sam_voice_separation_near_failure_threshold" in result["review_reasons"]
        assert "sam_voice_separation_failure" not in result["failure_reasons"]
    else:
        assert "sam_voice_separation_failure" in result["failure_reasons"]


def test_quality_ignores_permissive_candidates_without_confirmed_speech() -> None:
    result = quality_evaluation(
        {
            "separation_json": json.dumps(
                {
                    "sam": {"verification_status": "success"},
                    "reconstruction": {"similarity_score": 90.0},
                }
            ),
            "tag_json": json.dumps(
                {
                    "speech_coverage": 1.0,
                    "strong_speech_coverage": 0.0,
                    "foreground_speech_coverage": 0.0,
                    "synthetic_speech_coverage": 0.0,
                    "vocal_music_coverage": 0.0,
                }
            ),
            "asr_json": json.dumps(
                {
                    "transcript": "This is clearly spoken dialogue.",
                    "detected_language": "en",
                }
            ),
            "description_json": json.dumps(
                {
                    "parsed": {"description": "Detailed background scene."},
                    "validation": {"review_reasons": []},
                }
            ),
        }
    )

    assert result["bucket"] == "success"
    assert result["signals"]["background_speech_coverage"] == 0.0
    assert result["signals"]["background_speech_candidate_coverage"] == 1.0


@pytest.mark.parametrize(
    ("confirmed_speech", "expected"), ((0.0, "success"), (0.05, "review"))
)
def test_captioner_speech_judge_requires_independent_m2d_support(
    confirmed_speech: float, expected: str
) -> None:
    result = quality_evaluation(
        {
            "separation_json": json.dumps(
                {
                    "sam": {"verification_status": "success"},
                    "reconstruction": {"similarity_score": 90.0},
                }
            ),
            "tag_json": json.dumps(
                {
                    "speech_coverage": 1.0,
                    "strong_speech_coverage": confirmed_speech,
                    "foreground_speech_coverage": 0.0,
                    "synthetic_speech_coverage": 0.0,
                    "vocal_music_coverage": 0.0,
                }
            ),
            "asr_json": json.dumps(
                {
                    "transcript": "This is clearly spoken dialogue.",
                    "detected_language": "en",
                }
            ),
            "description_json": json.dumps(
                {
                    "parsed": {"description": "Detailed background scene."},
                    "parse": {"speech_mentions_omitted": 1},
                    "validation": {
                        "review_reasons": ["captioner_detected_speech_in_background"]
                    },
                }
            ),
        }
    )

    assert result["bucket"] == expected
    assert result["signals"]["captioner_speech_mentions_omitted"] == 1
    assert result["signals"]["captioner_speech_supported_by_m2d"] is bool(
        confirmed_speech
    )


@pytest.mark.parametrize(
    ("active_windows", "coverage", "expected_bucket", "expected_reason"),
    (
        (1, 1 / 29, "success", None),
        (2, 2 / 29, "review", "possible_vocal_music_in_background"),
        (10, 10 / 29, "failure", "vocal_music_in_background"),
    ),
)
def test_vocal_music_judge_tolerates_one_calibrated_noise_window(
    active_windows: int,
    coverage: float,
    expected_bucket: str,
    expected_reason: str | None,
) -> None:
    result = quality_evaluation(
        {
            "separation_json": json.dumps(
                {
                    "sam": {"verification_status": "success"},
                    "reconstruction": {"similarity_score": 90.0},
                }
            ),
            "tag_json": json.dumps(
                {
                    "speech_coverage": 0.0,
                    "strong_speech_coverage": 0.0,
                    "vocal_music_coverage": coverage,
                    "vocal_music_active_windows": active_windows,
                    "window_count": 29,
                    "duration_scaled_window_requirements": {"vocal_music_max": 1},
                }
            ),
            "asr_json": json.dumps(
                {
                    "transcript": "This is clearly spoken dialogue.",
                    "detected_language": "en",
                }
            ),
            "description_json": json.dumps(
                {
                    "parsed": {"description": "Detailed background scene."},
                    "validation": {"review_reasons": []},
                }
            ),
        }
    )

    assert result["bucket"] == expected_bucket
    reasons = result[
        "failure_reasons" if expected_bucket == "failure" else "review_reasons"
    ]
    if expected_reason is None:
        assert reasons == []
    else:
        assert expected_reason in reasons


def test_snapshots_include_every_bucket_and_use_sequence_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buckets = ("success", "review", "failure", "failure", "success", "review")
    sequences = (2, 5, 9, 12, 20, 21)
    for job_id, (sequence, bucket) in enumerate(
        zip(sequences, buckets, strict=True), start=1
    ):
        _record(
            tmp_path,
            job_id=job_id,
            sequence=sequence,
            bucket=bucket,
        )
    fake_s3 = _FakeS3()
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda _: fake_s3))
    monkeypatch.setattr(
        "sam_audio_pipeline.training_dataset._s3_object_exists",
        lambda *_: False,
    )

    result = publish_due_once(
        tmp_path,
        bucket="dataset-bucket",
        prefix="training-v1",
        snapshot_size=3,
        upload_workers=1,
    )

    assert result == {
        "snapshots": 2,
        "published_records": 6,
        "cleaned_records": 6,
    }
    manifests = sorted((tmp_path / "snapshots").glob("*/manifest.json"))
    assert [path.parent.name for path in manifests] == [
        "v1-00000002-00000009",
        "v1-00000012-00000021",
    ]
    first = json.loads(manifests[0].read_text())
    assert first["all_quality_buckets_included"] is True
    assert first["training_default_filter"] == "quality.bucket == 'success'"
    assert first["quality_buckets"] == {
        "failure": 1,
        "review": 1,
        "success": 1,
    }
    assert first["verification"] == {
        "policy": "immutable_training_snapshot_audit_v1",
        "status": "passed",
        "record_count": 3,
        "unique_record_ids": 3,
        "unique_record_sequences": 3,
        "complete_artifact_records": 3,
        "quality_buckets": {"failure": 1, "review": 1, "success": 1},
        "all_quality_buckets_included": True,
        "reference_generation": False,
    }
    assert [record["quality"]["bucket"] for record in first["records"]] == [
        "success",
        "review",
        "failure",
    ]
    assert [
        record["snapshot_membership"]["record_sequence"] for record in first["records"]
    ] == [2, 5, 9]
    assert all(
        record["snapshot_membership"]["snapshot_id"] == "v1-00000002-00000009"
        for record in first["records"]
    )
    assert any(
        "/snapshots/v1-00000002-00000009/failure/record-3/metadata.json" in key
        for key in fake_s3.objects
    )
    assert any(
        "/snapshots/v1-00000002-00000009/review/record-2/metadata.json" in key
        for key in fake_s3.objects
    )
    assert not _record_root(tmp_path, 1).exists()
    connection = connect(tmp_path)
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM records WHERE cleaned_at IS NOT NULL"
        ).fetchone()[0]
        == 6
    )
    connection.close()


def test_snapshot_audit_rejects_wrong_artifact_bucket() -> None:
    manifest = {
        "snapshot_id": "v1-00000001-00000001",
        "snapshot_record_count": 1,
        "quality_buckets": {"review": 1},
        "all_quality_buckets_included": True,
        "reference_generation": False,
        "records": [
            {
                "record_id": "record-a",
                "quality": {"bucket": "review"},
                "snapshot_membership": {
                    "snapshot_id": "v1-00000001-00000001",
                    "record_sequence": 1,
                    "quality_bucket": "review",
                },
                "s3_artifacts": {
                    name: f"s3://bucket/failure/record-a/{name}"
                    for name in OUTPUT_FILENAMES
                },
            }
        ],
    }

    with pytest.raises(ValueError, match="wrong bucket"):
        _verify_snapshot_manifest(manifest)
