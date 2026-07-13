from __future__ import annotations

import json
from pathlib import Path

import pytest

from sam_audio_pipeline.submit import successful_sources


def test_successful_sources_preserves_manifest_provenance(tmp_path: Path) -> None:
    audio = tmp_path / "audio" / "clip.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"wav")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "retrieval_status": "success",
                    "local_path": "audio/clip.wav",
                    "start_seconds": 12.0,
                    "end_seconds": 22.0,
                },
                {
                    "retrieval_status": "unavailable",
                    "local_path": "audio/missing.wav",
                },
            ]
        )
    )

    sources = successful_sources(manifest)

    assert sources == [
        (
            audio,
            {
                "retrieval_status": "success",
                "local_path": "audio/clip.wav",
                "start_seconds": 12.0,
                "end_seconds": 22.0,
            },
        )
    ]


def test_successful_sources_rejects_missing_success_file(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([{"retrieval_status": "success", "local_path": "missing.wav"}])
    )

    with pytest.raises(FileNotFoundError):
        successful_sources(manifest)
