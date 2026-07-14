import csv
import hashlib
import json
import wave
from argparse import Namespace
from pathlib import Path

import numpy as np

from sam_audio_pipeline.m2d_validator import (
    _m2d_asr_allowlist,
    evaluate_asr,
    evaluate_probabilities,
    load_label_families,
    materialize_accepted,
    merge_materialized,
)


def test_asr_gate_requires_decodable_foreground_voice() -> None:
    accepted = evaluate_asr(
        transcript="They have a magnet plane",
        duration_after_vad=3.9,
        average_log_probability=-0.49,
        no_speech_probability=0.15,
    )
    assert accepted["accepted"] is True

    chatter_hallucination = evaluate_asr(
        transcript="Okay see you there",
        duration_after_vad=9.4,
        average_log_probability=-0.89,
        no_speech_probability=0.23,
    )
    assert chatter_hallucination["accepted"] is False
    assert "low_transcription_confidence" in chatter_hallucination["rejection_reasons"]

    non_english = evaluate_asr(
        transcript="Namaste dosto",
        duration_after_vad=4.0,
        average_log_probability=-0.2,
        no_speech_probability=0.1,
        detected_language="hi",
        language_probability=0.98,
    )
    assert non_english["accepted"] is False
    assert "non_english_speech" in non_english["rejection_reasons"]


def test_m2d_gate_rejects_repeated_confident_synthetic_speech() -> None:
    labels = [
        {"mid": "speech", "display_name": "Speech"},
        {"mid": "music", "display_name": "Music"},
        {"mid": "sfx", "display_name": "Vehicle"},
        {"mid": "/m/0brhx", "display_name": "Speech synthesizer"},
    ]
    families = {
        "speech": {0},
        "foreground_speech": {0},
        "synthetic_speech": {3},
        "music": {1},
        "nonmusic_background": {2},
        "background": {1, 2},
        "human": {0, 3},
        "vocal_music": set(),
    }
    probabilities = np.asarray([[0.35, 0.20, 0.15, 0.30]] * 9)

    result = evaluate_probabilities(probabilities, labels, families)

    assert result["accepted"] is False
    assert result["synthetic_speech_active_windows"] == 9
    assert "synthetic_speech_present" in result["rejection_reasons"]


def test_evaluate_probabilities_requires_temporal_overlap():
    labels = [
        {"mid": "speech", "display_name": "Speech"},
        {"mid": "music", "display_name": "Music"},
        {"mid": "sfx", "display_name": "Vehicle"},
        {"mid": "other", "display_name": "Other"},
    ]
    families = {
        "speech": {0},
        "music": {1},
        "nonmusic_background": {2},
        "background": {1, 2},
        "human": {0},
        "vocal_music": {3},
    }
    overlapping = np.asarray([[0.45, 0.35, 0.199, 0.001]] * 9)
    result = evaluate_probabilities(overlapping, labels, families)
    assert result["accepted"] is True
    assert result["background_bucket"] == "music_led"
    assert result["overlap_active_windows"] == 9

    singing = np.asarray([[0.40, 0.30, 0.10, 0.20]] * 9)
    result = evaluate_probabilities(singing, labels, families)
    assert result["accepted"] is False
    assert "vocal_music_present" in result["rejection_reasons"]

    disjoint = np.asarray(
        [[0.80, 0.001, 0.001, 0.001]] * 4
        + [[0.001, 0.50, 0.40, 0.001]] * 5
    )
    result = evaluate_probabilities(disjoint, labels, families)
    assert result["accepted"] is False
    assert "insufficient_dialogue_background_overlap" in result["rejection_reasons"]

    weak_false_positive = np.asarray([[0.02, 0.50, 0.47, 0.01]] * 9)
    result = evaluate_probabilities(weak_false_positive, labels, families)
    assert result["speech_active_windows"] == 9
    assert result["strong_speech_active_windows"] == 0
    assert result["accepted"] is False
    assert "insufficient_strong_speech" in result["rejection_reasons"]


def test_cinematic_policy_requires_independent_music_and_sfx() -> None:
    labels = [
        {"mid": "speech", "display_name": "Speech"},
        {"mid": "music", "display_name": "Music"},
        {"mid": "sfx", "display_name": "Vehicle"},
        {"mid": "other", "display_name": "Other"},
    ]
    families = {
        "speech": {0},
        "music": {1},
        "nonmusic_background": {2},
        "background": {1, 2},
        "human": {0},
        "vocal_music": {3},
    }
    complete_mix = np.asarray([[0.45, 0.30, 0.249, 0.001]] * 9)
    result = evaluate_probabilities(
        complete_mix, labels, families, require_cinematic_mix=True
    )
    assert result["accepted"] is True
    assert result["cinematic_mix_pass"] is True

    music_without_effects = np.asarray([[0.45, 0.548, 0.001, 0.001]] * 9)
    result = evaluate_probabilities(
        music_without_effects, labels, families, require_cinematic_mix=True
    )
    assert result["accepted"] is False
    assert "insufficient_cinematic_sfx" in result["rejection_reasons"]

    effects_without_music = np.asarray([[0.45, 0.001, 0.548, 0.001]] * 9)
    result = evaluate_probabilities(
        effects_without_music, labels, families, require_cinematic_mix=True
    )
    assert result["accepted"] is False
    assert "insufficient_cinematic_music" in result["rejection_reasons"]


def test_load_label_families_uses_ontology_descendants(tmp_path: Path):
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=["index", "mid", "display_name"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {"index": 0, "mid": "/m/music-child", "display_name": "Score"},
                {"index": 1, "mid": "/m/speech-child", "display_name": "Dialogue"},
                {"index": 2, "mid": "/m/thing-child", "display_name": "Vehicle"},
            ]
        )
    ontology = tmp_path / "ontology.json"
    ontology.write_text(
        json.dumps(
            [
                {"id": "/m/04rlf", "child_ids": ["/m/music-child"]},
                {"id": "/m/09x0r", "child_ids": ["/m/speech-child"]},
                {"id": "/m/0dgw9r", "child_ids": ["/m/09x0r"]},
                {"id": "/t/dd00041", "child_ids": ["/m/thing-child"]},
            ]
        )
    )
    ordered, families = load_label_families(labels, ontology)
    mids = [item["mid"] for item in ordered]
    assert mids.index("/m/speech-child") in families["speech"]
    assert mids.index("/m/music-child") in families["music"]
    assert mids.index("/m/thing-child") in families["nonmusic_background"]
    assert families["vocal_music"] == set()


def _wav(path: Path) -> None:
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(2)
        destination.setsampwidth(2)
        destination.setframerate(48_000)
        destination.writeframes(b"\0\0\0\0" * 16)


def _strong_voice_windows() -> list[dict[str, float | int]]:
    return [
        {
            "speech_score": 0.5,
            "speech_rank": 1,
            "foreground_speech_score": 0.05,
            "foreground_speech_rank": 2,
        }
        for _ in range(9)
    ]


def test_asr_allowlist_contains_only_current_m2d_passes(tmp_path: Path) -> None:
    results = tmp_path / "m2d.jsonl"
    results.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                {
                    "filename": "pass.wav",
                    "accepted": True,
                    "rejection_reasons": [],
                    "windows": _strong_voice_windows(),
                },
                {
                    "filename": "fail.wav",
                    "accepted": False,
                    "rejection_reasons": ["insufficient_speech"],
                    "windows": [],
                },
            )
        )
    )

    allowed, scored_count = _m2d_asr_allowlist(
        results, require_cinematic_mix=False
    )

    assert allowed == {"pass.wav"}
    assert scored_count == 2


def test_materialize_preserves_source_and_links_only_accepted(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _wav(source / "ok.wav")
    _wav(source / "no.wav")
    _wav(source / "weak.wav")
    results = tmp_path / "results.jsonl"
    results.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "filename": "ok.wav",
                        "accepted": True,
                        "background_bucket": "music_led",
                        "rejection_reasons": [],
                        "windows": _strong_voice_windows(),
                    }
                ),
                json.dumps(
                    {
                        "filename": "ok-effects.wav",
                        "accepted": True,
                        "background_bucket": "effects_ambience_led",
                        "rejection_reasons": [],
                        "windows": _strong_voice_windows(),
                    }
                ),
                    json.dumps(
                        {
                            "filename": "no.wav",
                        "accepted": False,
                        "background_bucket": "music_led",
                        "rejection_reasons": ["insufficient_speech"],
                            "windows": [],
                        }
                    ),
                    json.dumps(
                        {
                            "filename": "weak.wav",
                            "accepted": True,
                            "background_bucket": "music_led",
                            "rejection_reasons": [],
                            "windows": [
                                {"speech_score": 0.02, "speech_rank": 10}
                                for _ in range(9)
                            ],
                        }
                    ),
            ]
        )
        + "\n"
    )
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "records": [
                    {"local_path": "audio/ok.wav"},
                    {"local_path": "audio/ok-effects.wav"},
                    {"local_path": "audio/no.wav"},
                    {"local_path": "audio/weak.wav"},
                ]
            }
        )
    )
    _wav(source / "ok-effects.wav")
    output = tmp_path / "accepted"
    (output / "audio").mkdir(parents=True)
    _wav(output / "audio" / "stale.wav")
    materialize_accepted(
        Namespace(
            input_dir=source,
            results=results,
            source_manifest=source_manifest,
            output_dir=output,
        )
    )
    assert (source / "ok.wav").exists()
    assert (source / "no.wav").exists()
    assert (output / "audio" / "ok.wav").exists()
    assert not (output / "audio" / "no.wav").exists()
    assert not (output / "audio" / "weak.wav").exists()
    assert not (output / "audio" / "stale.wav").exists()
    audit = json.loads((output / "audit.json").read_text())
    assert audit["accepted_record_count"] == 2
    assert audit["balanced_audio_files"] == 2
    assert audit["original_dataset_preserved"] is True


def test_materialize_intersects_m2d_and_asr_acceptance(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _wav(source / "voice.wav")
    _wav(source / "chatter.wav")
    results = tmp_path / "m2d.jsonl"
    results.write_text(
        "\n".join(
            json.dumps(
                {
                    "filename": filename,
                    "accepted": True,
                    "background_bucket": "music_led",
                    "rejection_reasons": [],
                    "windows": _strong_voice_windows(),
                }
            )
            for filename in ("voice.wav", "chatter.wav")
        )
        + "\n"
    )
    asr_results = tmp_path / "asr.jsonl"
    asr_results.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "filename": "voice.wav",
                        "accepted": True,
                        "rejection_reasons": [],
                    }
                ),
                json.dumps(
                    {
                        "filename": "chatter.wav",
                        "accepted": False,
                        "rejection_reasons": ["low_transcription_confidence"],
                    }
                ),
            )
        )
        + "\n"
    )
    source_manifest = tmp_path / "manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "records": [
                    {"local_path": "audio/voice.wav"},
                    {"local_path": "audio/chatter.wav"},
                ]
            }
        )
    )
    output = tmp_path / "accepted"
    materialize_accepted(
        Namespace(
            input_dir=source,
            results=results,
            asr_results=asr_results,
            source_manifest=source_manifest,
            output_dir=output,
        )
    )

    assert (output / "audio" / "voice.wav").exists()
    assert not (output / "audio" / "chatter.wav").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["accepted_record_count"] == 1
    assert manifest["records"][0]["asr_validation"]["accepted"] is True
    assert manifest["foreground_voice_rejection_reason_counts"] == {
        "low_transcription_confidence": 1
    }


def _validated_batch(
    root: Path, records: list[tuple[str, str, float]]
) -> Path:
    audio = root / "audio"
    audio.mkdir(parents=True)
    manifest_records = []
    m2d_records = []
    asr_records = []
    for filename, video_id, start in records:
        path = audio / filename
        _wav(path)
        with path.open("ab") as destination:
            destination.write(filename.encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_records.append(
            {
                "candidate_id": f"{video_id}:{round(start * 1000)}",
                "video_id": video_id,
                "source_platform": "dailymotion",
                "source_url": f"https://www.dailymotion.com/video/{video_id}",
                "title": "English Movie Scene HD",
                "duration_seconds": 120,
                "uploader": "Movie Scenes",
                "clip_start_seconds": start,
                "local_path": f"audio/{filename}",
                "sha256": digest,
            }
        )
        m2d_records.append(
            {
                "filename": filename,
                "accepted": True,
                "background_bucket": "mixed_music_and_effects",
                "rejection_reasons": [],
                "windows": _strong_voice_windows(),
            }
        )
        asr_records.append(
            {"filename": filename, "accepted": True, "rejection_reasons": []}
        )
    (root / "manifest.json").write_text(json.dumps({"records": manifest_records}))
    (root / "m2d-validation.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in m2d_records)
    )
    (root / "asr-validation.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in asr_records)
    )
    return root


def test_merge_materialized_is_exact_diverse_and_deduplicated(tmp_path: Path) -> None:
    first = _validated_batch(
        tmp_path / "first",
        [("one.wav", "video-a", 10.0), ("two.wav", "video-b", 20.0)],
    )
    second = _validated_batch(
        tmp_path / "second",
        [("overlap.wav", "video-a", 15.0), ("three.wav", "video-c", 30.0)],
    )
    output = tmp_path / "final"

    merge_materialized(
        Namespace(
            batch=[first, second],
            output_dir=output,
            accepted_limit=3,
            max_clips_per_video=1,
            seed=7,
            require_cinematic_mix=False,
        )
    )

    manifest = json.loads((output / "manifest.json").read_text())
    audit = json.loads((output / "audit.json").read_text())
    assert manifest["accepted_record_count"] == 3
    assert len({record["video_id"] for record in manifest["records"]}) == 3
    assert audit["record_count"] == 3
    assert audit["unique_sha256_count"] == 3
    assert audit["all_requirements_pass"] is True
