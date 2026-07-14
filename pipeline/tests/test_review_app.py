from __future__ import annotations

import json
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from sam_audio_pipeline.review_app import ReviewStore, create_review_app

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


def test_review_app_persists_decisions_and_serves_audio(tmp_path: Path):
    dataset = _dataset(tmp_path)
    store = ReviewStore(dataset, audio_directory="balanced-audio")
    client = TestClient(create_review_app(store))

    state = client.get("/api/state").json()
    assert [clip["filename"] for clip in state["clips"]] == [
        "two.wav",
        "one.wav",
    ]
    assert state["summary"]["total"] == 2
    assert state["summary"]["reviewed"] == 0
    assert state["summary"]["available"] == 2
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
