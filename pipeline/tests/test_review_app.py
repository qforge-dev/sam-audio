from __future__ import annotations

import json
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from sam_audio_pipeline.review_app import ReviewStore, create_review_app


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


def test_review_app_persists_decisions_and_serves_audio(tmp_path: Path):
    dataset = _dataset(tmp_path)
    store = ReviewStore(dataset, audio_directory="balanced-audio")
    client = TestClient(create_review_app(store))

    state = client.get("/api/state").json()
    assert [clip["filename"] for clip in state["clips"]] == [
        "two.wav",
        "one.wav",
    ]
    assert state["summary"] == {
        "total": 2,
        "reviewed": 0,
        "unreviewed": 2,
        "good": 0,
        "perfect": 0,
        "not_ok": 0,
    }
    assert client.get("/clip/two.wav").status_code == 200
    assert client.get("/api/audio/two.wav").content.startswith(b"RIFF")

    response = client.put(
        "/api/reviews/two.wav",
        json={"decision": "perfect"},
    )
    assert response.status_code == 200
    assert response.json()["summary"]["perfect"] == 1

    response = client.put(
        "/api/reviews/one.wav",
        json={
            "decision": "not_ok",
            "reasons": ["lacking_music", "too_low_quality"],
            "note": "Mostly isolated dialogue",
        },
    )
    assert response.status_code == 200
    assert response.json()["summary"]["not_ok"] == 1

    reloaded = ReviewStore(dataset, audio_directory="balanced-audio")
    assert reloaded.reviews["two.wav"]["decision"] == "perfect"
    assert reloaded.reviews["one.wav"]["reasons"] == [
        "lacking_music",
        "too_low_quality",
    ]
    assert "two.wav,perfect" in client.get("/api/export.csv").text


def test_not_ok_requires_reason_and_other_requires_note(tmp_path: Path):
    client = TestClient(
        create_review_app(
            ReviewStore(_dataset(tmp_path), audio_directory="balanced-audio")
        )
    )
    assert (
        client.put(
            "/api/reviews/one.wav", json={"decision": "not_ok"}
        ).status_code
        == 422
    )
    response = client.put(
        "/api/reviews/one.wav",
        json={"decision": "not_ok", "reasons": ["other"]},
    )
    assert response.status_code == 422
    assert "requires a note" in response.json()["detail"]


def test_clear_review_returns_clip_to_unreviewed(tmp_path: Path):
    client = TestClient(
        create_review_app(
            ReviewStore(_dataset(tmp_path), audio_directory="balanced-audio")
        )
    )
    client.put("/api/reviews/one.wav", json={"decision": "good"})
    response = client.delete("/api/reviews/one.wav")
    assert response.status_code == 200
    assert response.json()["summary"]["unreviewed"] == 2
