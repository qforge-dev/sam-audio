from __future__ import annotations

import shutil
import struct
import wave
from pathlib import Path
from typing import Any

from sam_audio_pipeline.audio_profile import backfill_audio_profiles


class ProfileAWS:
    def __init__(self, source: Path):
        self.source = source
        self.item: dict[str, Any] = {
            "PK": "JOB#job-1",
            "SK": "SOURCE#source-1",
            "entity": "source",
            "source_id": "source-1",
            "filename": "original.wav",
            "s3_key": "original.wav",
        }

    def query_partition(self, _: str) -> list[dict[str, Any]]:
        return [self.item]

    def download_file(self, _: str, destination: Path) -> None:
        shutil.copyfile(self.source, destination)

    def object_exists(self, _: str) -> bool:
        return True

    def update(self, _: str, __: str, values: dict[str, Any]) -> None:
        self.item.update(values)


def write_stereo_wav(path: Path) -> None:
    frames = bytearray()
    for value in range(800):
        sample = (value % 100) * 100
        frames.extend(struct.pack("<hh", sample, -sample))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(frames)


def test_audio_profile_backfill_is_durable_and_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "original.wav"
    write_stereo_wav(source)
    aws = ProfileAWS(source)

    first = backfill_audio_profiles(aws, ["job-1"])
    second = backfill_audio_profiles(aws, ["job-1"])

    assert first == {
        "jobs": 1,
        "sources": 1,
        "skipped": 0,
        "missing_source": 0,
        "failed": 0,
    }
    assert second == {
        "jobs": 1,
        "sources": 0,
        "skipped": 1,
        "missing_source": 0,
        "failed": 0,
    }
    assert aws.item["input_channels"] == 2
    assert aws.item["audio_profile"]["channel_label"] == "Stereo"
    assert aws.item["audio_profile"]["quality_tier"] == "lossless"
    assert aws.item["audio_profile"]["bit_depth"] == 16
