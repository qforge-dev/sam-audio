from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from sam_audio_pipeline.source_scanner import (
    MAX_REGION_VOCAL_MUSIC_COVERAGE,
    MIN_REGION_FOREGROUND_SPEECH_COVERAGE,
    M2DSourceScanner,
    region_passes_confidence_gate,
    select_candidate_regions,
)


def test_proxy_creation_preserves_stereo_with_one_ffmpeg_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[str] = []

    def fake_run(command, **_kwargs) -> None:
        observed.extend(command)

    monkeypatch.setattr("sam_audio_pipeline.source_scanner.subprocess.run", fake_run)
    scanner = object.__new__(M2DSourceScanner)
    scanner.sample_rate = 16_000

    scanner.create_proxy(tmp_path / "source.mp4", tmp_path / "proxy.flac")

    assert observed[observed.index("-ac") + 1] == "2"
    assert observed[observed.index("-threads") + 1] == "1"


def _labels() -> list[dict[str, str]]:
    return [
        {"mid": f"/test/{index}", "display_name": name}
        for index, name in enumerate(
            ("speech", "music", "effects", "singing", "other", "human", "synthetic")
        )
    ]


def _families() -> dict[str, set[int]]:
    return {
        "speech": {0},
        "foreground_speech": {0},
        "synthetic_speech": {6},
        "music": {1},
        "nonmusic_background": {2},
        "background": {1, 2},
        "human": {0, 3, 5},
        "vocal_music": {3},
    }


def _passing_probabilities(windows: int) -> np.ndarray:
    values = np.full((windows, 7), 0.001, dtype=np.float32)
    values[:, 0] = 0.40
    values[:, 1] = 0.20
    values[:, 2] = 0.15
    values[:, 4] = 0.10
    values[:, 5] = 0.05
    return values


def test_source_scan_selects_ranked_non_overlapping_regions() -> None:
    regions = select_candidate_regions(
        _passing_probabilities(89),
        _labels(),
        _families(),
        clip_seconds=30,
        max_regions=10,
        region_hop_seconds=5,
    )

    assert [item["start_seconds"] for item in regions] == [0.0, 30.0, 60.0]
    assert all(item["evidence"]["accepted"] for item in regions)
    assert all("windows" not in item["evidence"] for item in regions)


def test_source_scan_selects_regions_from_sparse_long_source_windows() -> None:
    regions = select_candidate_regions(
        _passing_probabilities(13),
        _labels(),
        _families(),
        clip_seconds=30,
        max_regions=10,
        window_hop_seconds=5,
    )

    assert [item["start_seconds"] for item in regions] == [0.0, 30.0]


def test_source_scan_rejects_vocal_music_regions() -> None:
    probabilities = _passing_probabilities(59)
    probabilities[:, 3] = 0.30

    regions = select_candidate_regions(
        probabilities,
        _labels(),
        _families(),
        clip_seconds=30,
        max_regions=10,
    )

    assert regions == []


def test_source_scan_reports_inference_slot_wait() -> None:
    scanner = object.__new__(M2DSourceScanner)
    scanner.labels = _labels()
    scanner.families = _families()
    scanner.sample_rate = 16_000
    scanner.inference_concurrency = 4
    scanner._inference_slots = threading.BoundedSemaphore(4)
    scanner.target_scan_windows = 3_600
    scanner.long_source_inference_concurrency = 2
    scanner._long_source_inference_slots = threading.BoundedSemaphore(2)
    scanner._proxy_window_hop_seconds = lambda _proxy: 1.0
    scanner._probabilities = lambda _proxy, **_kwargs: (
        _passing_probabilities(29),
        29,
        1.25,
        1.0,
    )

    result = scanner.scan(None, clip_seconds=30, max_regions=1)

    assert result["m2d_inference_concurrency"] == 4
    assert result["inference_wait_seconds"] >= 0
    assert result["scan_seconds"] == 1.25


def test_source_scan_adapts_window_hop_for_very_long_sources() -> None:
    scanner = object.__new__(M2DSourceScanner)
    scanner.sample_rate = 16_000
    scanner.target_scan_windows = 3_600
    scanner.max_window_hop_seconds = 5.0

    assert scanner._window_hop_seconds(60 * 60 * 16_000) == 1.0
    assert scanner._window_hop_seconds(2 * 60 * 60 * 16_000) == 2.0
    assert scanner._window_hop_seconds(5 * 60 * 60 * 16_000) == 5.0
    assert scanner._window_hop_seconds(12 * 60 * 60 * 16_000) == 5.0


def test_region_confidence_gate_requires_measured_foreground_speech_margin() -> None:
    assert region_passes_confidence_gate(
        {
            "evidence": {
                "foreground_speech_coverage": (
                    MIN_REGION_FOREGROUND_SPEECH_COVERAGE
                )
            }
        }
    )
    assert not region_passes_confidence_gate(
        {
            "evidence": {
                "foreground_speech_coverage": (
                    MIN_REGION_FOREGROUND_SPEECH_COVERAGE - 0.01
                )
            }
        }
    )
    assert not region_passes_confidence_gate(
        {
            "evidence": {
                "foreground_speech_coverage": (
                    MIN_REGION_FOREGROUND_SPEECH_COVERAGE
                ),
                "vocal_music_coverage": MAX_REGION_VOCAL_MUSIC_COVERAGE + 0.01,
            }
        }
    )


def test_whole_source_stereo_metric_detects_dual_mono() -> None:
    sample_rate = 16_000
    timeline = np.arange(sample_rate, dtype=np.float32) / sample_rate
    left = 0.2 * np.sin(2 * np.pi * 440 * timeline)

    class FakeAudio:
        channels = 2
        samplerate = sample_rate

        def __init__(self, samples: np.ndarray) -> None:
            self.samples = samples

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self, *_args, **_kwargs) -> np.ndarray:
            samples, self.samples = self.samples, np.empty((0, 2))
            return samples

    scanner = object.__new__(M2DSourceScanner)
    scanner.sample_rate = sample_rate
    scanner.soundfile = type(
        "DualMonoSoundFile",
        (),
        {"SoundFile": lambda *_args: FakeAudio(np.column_stack((left, left)))},
    )
    assert scanner.stereo_metrics(None)["side_to_total_db"] <= -100.0
    right = 0.2 * np.sin(2 * np.pi * 554 * timeline)
    scanner.soundfile = type(
        "StereoSoundFile",
        (),
        {"SoundFile": lambda *_args: FakeAudio(np.column_stack((left, right)))},
    )
    assert scanner.stereo_metrics(None)["side_to_total_db"] > -10.0
