from __future__ import annotations

import hashlib
import shutil
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sam_audio_pipeline.reconstruction import (
    backfill_job,
    join_reconstructed_chunks,
    join_stereo_stems,
)


def write_wav(path: Path, sample_rate: int, samples: np.ndarray) -> None:
    values = np.asarray(samples, dtype=np.float64)
    encoded = np.clip(np.rint(values * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(encoded.tobytes())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_identity_reconstruction_is_exact_and_scores_100(tmp_path: Path) -> None:
    rate = 8_000
    time = np.arange(rate) / rate
    original = np.column_stack(
        (
            0.2 * np.sin(2 * np.pi * 440 * time),
            0.12 * np.sin(2 * np.pi * 660 * time),
        )
    )
    original_path = tmp_path / "original.wav"
    joined_path = tmp_path / "joined.wav"
    write_wav(original_path, rate, original)

    result = join_stereo_stems(
        original_path, {"sfx": original_path}, joined_path
    )

    assert digest(joined_path) == digest(original_path)
    assert result.metrics["similarity_score"] == 100.0
    assert result.metrics["waveform_correlation"] == 1.0
    assert result.metrics["channel_similarity"] == {
        "left": 100.0,
        "right": 100.0,
    }


def test_similarity_is_phase_and_level_sensitive(tmp_path: Path) -> None:
    rate = 8_000
    time = np.arange(rate) / rate
    tone = 0.2 * np.sin(2 * np.pi * 440 * time)
    original = np.column_stack((tone, tone))
    original_path = tmp_path / "original.wav"
    half_path = tmp_path / "half.wav"
    inverted_path = tmp_path / "inverted.wav"
    write_wav(original_path, rate, original)
    write_wav(half_path, rate, original * 0.5)
    write_wav(inverted_path, rate, original * -1.0)

    half = join_stereo_stems(
        original_path, {"music": half_path}, tmp_path / "half-joined.wav"
    )
    inverted = join_stereo_stems(
        original_path,
        {"music": inverted_path},
        tmp_path / "inverted-joined.wav",
    )

    assert half.metrics["similarity_score"] == pytest.approx(80.0, abs=0.02)
    assert half.metrics["level_delta_db"] == pytest.approx(-6.0206, abs=0.01)
    assert inverted.metrics["similarity_score"] == 0.0
    assert inverted.metrics["waveform_correlation"] == pytest.approx(-1.0)


def test_joined_chunks_overlap_into_one_exact_source(tmp_path: Path) -> None:
    rate = 8_000
    time = np.arange(rate * 2) / rate
    original = np.column_stack(
        (
            0.2 * np.sin(2 * np.pi * 330 * time),
            0.16 * np.sin(2 * np.pi * 550 * time),
        )
    )
    original_path = tmp_path / "original.wav"
    first_path = tmp_path / "first.wav"
    second_path = tmp_path / "second.wav"
    write_wav(original_path, rate, original)
    write_wav(first_path, rate, original[: rate * 5 // 4])
    write_wav(second_path, rate, original[rate * 3 // 4 :])

    result = join_reconstructed_chunks(
        original_path,
        [("000000", 0.0, first_path), ("000001", 0.75, second_path)],
        tmp_path / "source-joined.wav",
    )

    assert result.metrics["similarity_score"] == 100.0
    assert result.metrics["chunk_count"] == 2
    assert result.metrics["coverage_fraction"] == 1.0


class BackfillAWS:
    def __init__(self, original: Path, stem: Path):
        self.files = {"chunk.wav": original, "voice.stereo.wav": stem}
        self.items = [
            {
                "PK": "JOB#job-1",
                "SK": "SOURCE#source-1",
                "entity": "source",
                "job_id": "job-1",
                "source_id": "source-1",
                "filename": "source.wav",
                "status": "chunked",
                "s3_key": "source.wav",
            },
            {
                "PK": "JOB#job-1",
                "SK": "CHUNK#source-1#000000",
                "entity": "chunk",
                "job_id": "job-1",
                "source_id": "source-1",
                "chunk_id": "000000",
                "status": "complete",
                "s3_key": "chunk.wav",
            },
            {
                "PK": "JOB#job-1",
                "SK": "STEM#source-1#000000#voice",
                "entity": "stem",
                "source_id": "source-1",
                "chunk_id": "000000",
                "stem_type": "voice",
                "stereo_s3_key": "voice.stereo.wav",
            },
        ]
        self.files["source.wav"] = original
        self.uploads: dict[str, bytes] = {}

    def query_partition(self, _: str) -> list[dict[str, Any]]:
        return self.items

    def object_exists(self, key: str) -> bool:
        return key in self.uploads

    def download_file(self, key: str, destination: Path) -> None:
        if key in self.uploads:
            destination.write_bytes(self.uploads[key])
        else:
            shutil.copyfile(self.files[key], destination)

    def upload_file(self, path: Path, key: str, _: str) -> None:
        self.uploads[key] = path.read_bytes()

    def update(self, _: str, sk: str, values: dict[str, Any]) -> None:
        next(item for item in self.items if item["SK"] == sk).update(values)


def test_backfill_persists_joined_audio_and_metrics(tmp_path: Path) -> None:
    rate = 8_000
    time = np.arange(rate) / rate
    tone = 0.2 * np.sin(2 * np.pi * 440 * time)
    original = np.column_stack((tone, tone * 0.5))
    original_path = tmp_path / "original.wav"
    write_wav(original_path, rate, original)
    aws = BackfillAWS(original_path, original_path)

    summary = backfill_job(object(), aws, "job-1")

    chunk_reconstruction = aws.items[1]["reconstruction"]
    source_reconstruction = aws.items[0]["reconstruction"]
    assert summary == {
        "chunks": 1,
        "skipped": 0,
        "missing_stereo": 0,
        "sources": 1,
        "source_skipped": 0,
        "source_incomplete": 0,
    }
    assert chunk_reconstruction["metrics"]["similarity_score"] == 100.0
    assert chunk_reconstruction["s3_key"].endswith("000000.joined.stereo.wav")
    assert source_reconstruction["metrics"]["similarity_score"] == 100.0
    assert source_reconstruction["s3_key"].endswith("source.joined.stereo.wav")
    assert aws.uploads[chunk_reconstruction["s3_key"]] == original_path.read_bytes()
