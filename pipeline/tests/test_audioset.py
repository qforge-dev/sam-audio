from __future__ import annotations

import json
from pathlib import Path

from sam_audio_pipeline.audioset import _acquire, select_random_manifest


def test_seeded_random_manifest_preserves_audioset_timestamps(tmp_path: Path) -> None:
    cache = tmp_path / "metadata"
    cache.mkdir()
    (cache / "ontology.json").write_text(
        json.dumps(
            [
                {"id": "/m/dog", "name": "Dog"},
                {"id": "/m/rain", "name": "Rain"},
            ]
        )
    )
    (cache / "eval_segments.csv").write_text(
        "# YTID, start_seconds, end_seconds, positive_labels\n"
        'video-dog, 12.500, 22.500, "/m/dog"\n'
    )
    (cache / "balanced_train_segments.csv").write_text(
        "# YTID, start_seconds, end_seconds, positive_labels\n"
        'video-rain, 40.000, 50.000, "/m/rain"\n'
    )

    first = select_random_manifest(cache, 2, seed=7, candidate_multiplier=1)
    second = select_random_manifest(cache, 2, seed=7, candidate_multiplier=1)

    assert first == second
    assert {item["video_id"] for item in first} == {"video-dog", "video-rain"}
    assert {(item["start_seconds"], item["end_seconds"]) for item in first} == {
        (12.5, 22.5),
        (40.0, 50.0),
    }
    assert {tuple(item["label_names"]) for item in first} == {("Dog",), ("Rain",)}


def test_acquire_downloads_only_the_published_audioset_time_range(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[list[str]] = []

    def capture(command: list[str], *, check: bool) -> None:
        assert check is True
        calls.append(command)

    monkeypatch.setattr("sam_audio_pipeline.audioset.subprocess.run", capture)

    _acquire(
        {
            "source_url": "https://www.youtube.com/watch?v=video-dog",
            "start_seconds": 12.5,
            "end_seconds": 22.5,
        },
        tmp_path / "clip.wav",
    )

    assert calls[0][calls[0].index("--download-sections") + 1] == "*12.5-22.5"
    assert "--force-keyframes-at-cuts" in calls[0]
    assert calls[1][calls[1].index("-t") + 1] == "10.0"
