from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from sam_audio_pipeline.model_client import _extract_archive


def test_extract_archive_replaces_stale_retry_directory(tmp_path: Path) -> None:
    archive_path = tmp_path / "separation.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("metadata.json", "{}")
        archive.writestr("voice.wav", b"voice")

    destination = tmp_path / "extracted"
    destination.mkdir()
    (destination / "partial-download.wav").write_bytes(b"stale")

    _extract_archive(archive_path, destination)

    assert (destination / "metadata.json").read_text() == "{}"
    assert (destination / "voice.wav").read_bytes() == b"voice"
    assert not (destination / "partial-download.wav").exists()
    assert not list(tmp_path.glob(".extracted-*"))


def test_unsafe_archive_preserves_prior_complete_extraction(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "separation.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.wav", b"unsafe")

    destination = tmp_path / "extracted"
    destination.mkdir()
    (destination / "metadata.json").write_text("previous")

    with pytest.raises(ValueError, match="Unsafe archive member"):
        _extract_archive(archive_path, destination)

    assert (destination / "metadata.json").read_text() == "previous"
    assert not (tmp_path / "escape.wav").exists()
    assert not list(tmp_path.glob(".extracted-*"))
