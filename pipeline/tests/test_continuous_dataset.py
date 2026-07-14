from __future__ import annotations

import hashlib
import json
import math
import wave
from pathlib import Path

import numpy as np

from sam_audio_pipeline.continuous_dataset import (
    _snapshot_manifest,
    assemble_once,
    autoscale_decision,
    catalog_records,
    connect,
    progress_snapshot,
    promote_once,
)
from sam_audio_pipeline.review_app import ReviewStore


def _wav(path: Path) -> None:
    sample_rate = 48_000
    timeline = np.arange(sample_rate) / sample_rate
    samples = np.column_stack(
        (
            0.2 * np.sin(2 * math.pi * 440 * timeline),
            0.2 * np.sin(2 * math.pi * 554 * timeline),
        )
    )
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(2)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(np.rint(samples * 32767).astype("<i2").tobytes())


def _autoscale(**overrides: object) -> dict[str, object]:
    settings: dict[str, object] = {
        "download_concurrency": 8,
        "asr_concurrency": 1,
        "m2d_backlog": 0,
        "asr_backlog": 0,
        "cpu_percent": 40.0,
        "gpu_free_mb": 14_500.0,
        "download_min": 2,
        "download_max": 8,
        "asr_min": 1,
        "asr_max": 2,
        "cpu_low": 55.0,
        "cpu_high": 85.0,
        "m2d_backlog_high": 64,
        "asr_backlog_high": 8,
        "gpu_reserve_mb": 12_000.0,
    }
    settings.update(overrides)
    return autoscale_decision(**settings)  # type: ignore[arg-type]


def test_autoscaler_scales_asr_only_for_a_real_backlog_with_headroom() -> None:
    decision = _autoscale(asr_backlog=8, m2d_backlog=64)

    assert decision["asr_concurrency"] == 2
    assert decision["download_concurrency"] == 8
    assert decision["actions"] == ["increase_asr"]


def test_autoscaler_reduces_producer_pressure_when_resources_are_constrained() -> None:
    cpu = _autoscale(cpu_percent=95.0)
    no_gpu_room = _autoscale(asr_backlog=8, gpu_free_mb=5_000.0)

    assert cpu["download_concurrency"] == 7
    assert cpu["actions"] == ["reduce_download_for_cpu"]
    assert no_gpu_room["download_concurrency"] == 7
    assert no_gpu_room["actions"] == ["reduce_download_for_asr"]


def test_autoscaler_reclaims_idle_asr_then_increases_acquisition() -> None:
    idle_asr = _autoscale(asr_concurrency=2)
    source_starved = _autoscale(download_concurrency=7, cpu_percent=30.0)

    assert idle_asr["asr_concurrency"] == 1
    assert idle_asr["actions"] == ["decrease_idle_asr"]
    assert source_starved["download_concurrency"] == 8
    assert source_starved["actions"] == ["increase_download"]


def test_independent_workers_promote_score_and_assemble_incrementally(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    run = runs / "run-1"
    audio = run / "audio"
    audio.mkdir(parents=True)
    _wav(audio / "source.wav")
    digest = hashlib.sha256((audio / "source.wav").read_bytes()).hexdigest()
    record = {
        "candidate_id": "movie:5000",
        "video_id": "movie",
        "source_platform": "dailymotion",
        "source_url": "https://example.test/movie",
        "title": "English Movie Scene HD",
        "duration_seconds": 120,
        "clip_start_seconds": 5.0,
        "clip_end_seconds": 35.0,
        "retrieval_status": "success",
        "quality_rejections": [],
        "source_format": {
            "sample_rate_hz": 48_000,
            "channels": 2,
            "bitrate_kbps": 160,
        },
        "local_path": "audio/source.wav",
        "sha256": digest,
    }
    (run / "manifest.json").write_text(json.dumps({"records": [record]}))
    workspace = tmp_path / "workspace"

    assert promote_once(runs, workspace) == 1
    assert promote_once(runs, workspace) == 0
    filename = f"{digest}.wav"
    windows = [
        {
            "speech_score": 0.2,
            "speech_rank": 1,
            "foreground_speech_score": 0.02,
            "foreground_speech_rank": 1,
            "synthetic_speech_score": 0.0,
            "synthetic_speech_rank": 100,
            "music_score": 0.04,
            "music_rank": 1,
            "nonmusic_background_score": 0.04,
            "nonmusic_background_rank": 1,
        }
        for _ in range(29)
    ]
    (workspace / "m2d-validation.jsonl").write_text(
        json.dumps(
            {
                "filename": filename,
                "accepted": True,
                "rejection_reasons": [],
                "cinematic_mix_required": True,
                "windows": windows,
            }
        )
        + "\n"
    )
    (workspace / "asr-validation.jsonl").write_text(
        json.dumps(
            {
                "filename": filename,
                "accepted": True,
                "policy": "foreground_voice_faster_whisper_v3",
                "detected_language": "en",
            }
        )
        + "\n"
    )

    assert assemble_once(workspace) == 1
    assert assemble_once(workspace) == 0
    manifest = json.loads((workspace / "accepted" / "manifest.json").read_text())
    assert manifest["continuous"] is True
    assert manifest["clip_seconds"] == 30.0
    assert manifest["accepted_record_count"] == 1
    assert manifest["manual_review_is_acceptance_gate"] is False
    assert manifest["records"] == []
    assert (workspace / "accepted" / "audio" / filename).is_file()
    connection = connect(workspace)
    records = catalog_records(connection)
    connection.close()
    assert len(records) == 1
    assert records[0]["catalog_sequence"] == 1
    review = ReviewStore(workspace / "accepted", audio_directory="audio")
    assert review.state()["summary"]["total"] == 1
    assert review.state()["clips"][0]["filename"] == filename

    snapshot_dir = workspace / "snapshots" / "test"
    _snapshot_manifest(workspace, 1, 1, snapshot_dir)
    snapshot = json.loads((snapshot_dir / "manifest.json").read_text())
    assert snapshot["snapshot_sequence_start"] == 1
    assert snapshot["snapshot_sequence_end"] == 1
    assert len(snapshot["records"]) == 1
    progress = progress_snapshot(workspace)
    assert progress["counts"]["downloaded"] == 1
    assert progress["counts"]["accepted"] == 1
    assert progress["counts"]["rejected_total"] == 0
    assert progress["next_snapshot"]["remaining"] == 4999
    assert progress["throughput"]["download"]["audio_minutes_per_minute"] > 0
    assert progress["flow"]["state"] == "balanced"
    assert progress["flow"]["processed_audio_hours_per_wall_hour"] > 0
    assert progress["flow"]["accepted_audio_hours_per_wall_hour"] > 0
    assert progress["flow"]["rolling_yield_percent"] == 100.0
    assert progress["flow"]["stalled_stages"] == []
    assert progress["goal"]["target_audio_hours"] == 10_000
    assert progress["goal"]["estimated_completion_at"] is not None
