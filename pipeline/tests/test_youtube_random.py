from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from sam_audio_pipeline.youtube_random import (
    _candidate_allowed,
    analyze_wav,
    build_queries,
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


def test_candidate_filter_rejects_short_live_and_pure_audio_results() -> None:
    valid = {"id": "video", "title": "City festival vlog", "duration": 120}

    assert _candidate_allowed(valid)
    assert not _candidate_allowed({**valid, "duration": 20})
    assert not _candidate_allowed({**valid, "live_status": "is_live"})
    assert not _candidate_allowed({**valid, "title": "Official Audio"})
    assert not _candidate_allowed({**valid, "title": "Song (Official Video)"})


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
