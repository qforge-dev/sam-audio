from __future__ import annotations

import hashlib
import shutil
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sam_audio_pipeline.config import Settings
from sam_audio_pipeline.handlers import SeparationHandler
from sam_audio_pipeline.schema import QueueTask, StemRecord
from sam_audio_pipeline.stereo import backfill_job, map_stems_to_stereo


def write_wav(path: Path, sample_rate: int, samples: np.ndarray) -> None:
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    encoded = np.rint(np.clip(values, -0.999, 0.999) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(values.shape[1])
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(encoded.tobytes())


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as source:
        rate = source.getframerate()
        channels = source.getnchannels()
        values = np.frombuffer(source.readframes(source.getnframes()), "<i2")
    return rate, values.astype(np.float32).reshape(-1, channels) / 32768.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stereo_mapping_preserves_raw_and_smoothly_moves_voice_left_to_right(
    tmp_path: Path,
) -> None:
    sample_rate = 8_000
    duration = 4.0
    time = np.arange(round(sample_rate * duration)) / sample_rate
    voice = 0.22 * np.sin(2 * np.pi * 440 * time)
    ambience = 0.08 * np.sin(2 * np.pi * 1_500 * time)
    pan = np.linspace(-0.65, 0.65, len(time))
    angle = (pan + 1.0) * np.pi / 4.0
    original = np.column_stack(
        (voice * np.cos(angle) + ambience, voice * np.sin(angle) + ambience)
    )
    original_path = tmp_path / "original.wav"
    voice_path = tmp_path / "voice.wav"
    ambience_path = tmp_path / "sfx.wav"
    write_wav(original_path, sample_rate, original)
    write_wav(voice_path, sample_rate, voice)
    write_wav(ambience_path, sample_rate, ambience)
    before = digest(voice_path)

    mapped = map_stems_to_stereo(
        original_path,
        {"voice": voice_path, "sfx": ambience_path},
        tmp_path / "mapped",
        band_count=16,
        smoothing_alpha=0.08,
        n_fft=512,
        hop=128,
    )

    assert digest(voice_path) == before
    rate, stereo = read_wav(mapped["voice"].path)
    assert rate == sample_rate
    assert stereo.shape == (len(voice), 2)
    quarter = len(stereo) // 4
    first_rms = np.sqrt(np.mean(stereo[:quarter] ** 2, axis=0))
    last_rms = np.sqrt(np.mean(stereo[-quarter:] ** 2, axis=0))
    assert first_rms[0] > first_rms[1] * 1.2
    assert last_rms[1] > last_rms[0] * 1.2
    mapping = mapped["voice"].metadata
    assert mapping["raw_channels"] == 1
    assert mapping["mapped_channels"] == 2
    assert mapping["pan_summary"]["start"] < -0.15
    assert mapping["pan_summary"]["end"] > 0.15
    curve = np.array([point["pan"] for point in mapping["pan_curve"]])
    assert np.max(np.abs(np.diff(curve))) < 0.35


def test_stereo_mapping_keeps_frequency_specific_position(tmp_path: Path) -> None:
    sample_rate = 8_000
    time = np.arange(sample_rate * 2) / sample_rate
    low = 0.16 * np.sin(2 * np.pi * 440 * time)
    high = 0.16 * np.sin(2 * np.pi * 2_000 * time)
    stem = low + high
    original = np.column_stack((low + 0.08 * high, 0.08 * low + high))
    original_path = tmp_path / "original.wav"
    stem_path = tmp_path / "voice.wav"
    write_wav(original_path, sample_rate, original)
    write_wav(stem_path, sample_rate, stem)

    result = map_stems_to_stereo(
        original_path,
        {"voice": stem_path},
        tmp_path / "mapped",
        band_count=16,
        smoothing_alpha=0.08,
        n_fft=512,
        hop=128,
    )["voice"]

    _, stereo = read_wav(result.path)
    spectrum = np.abs(np.fft.rfft(stereo, axis=0))
    frequencies = np.fft.rfftfreq(len(stereo), 1 / sample_rate)
    low_bin = int(np.argmin(np.abs(frequencies - 440)))
    high_bin = int(np.argmin(np.abs(frequencies - 2_000)))
    assert spectrum[low_bin, 0] > spectrum[low_bin, 1] * 1.1
    assert spectrum[high_bin, 1] > spectrum[high_bin, 0] * 1.1


def test_stereo_mapping_preserves_stem_tonal_balance(tmp_path: Path) -> None:
    sample_rate = 8_000
    time = np.arange(sample_rate * 2) / sample_rate
    low = 0.12 * np.sin(2 * np.pi * 220 * time)
    high = 0.12 * np.sin(2 * np.pi * 2_000 * time)
    stem = low + high
    original = np.column_stack((low + 0.15 * high, low + 0.15 * high))
    original_path = tmp_path / "original.wav"
    stem_path = tmp_path / "music.wav"
    write_wav(original_path, sample_rate, original)
    write_wav(stem_path, sample_rate, stem)

    result = map_stems_to_stereo(
        original_path,
        {"music": stem_path},
        tmp_path / "mapped",
        band_count=16,
        smoothing_alpha=0.08,
        n_fft=512,
        hop=128,
    )["music"]

    _, stereo = read_wav(result.path)
    raw_spectrum = np.abs(np.fft.rfft(stem))
    mapped_spectrum = np.abs(np.fft.rfft(np.mean(stereo, axis=1)))
    frequencies = np.fft.rfftfreq(len(stem), 1 / sample_rate)
    low_bin = int(np.argmin(np.abs(frequencies - 220)))
    high_bin = int(np.argmin(np.abs(frequencies - 2_000)))
    raw_ratio = raw_spectrum[high_bin] / raw_spectrum[low_bin]
    mapped_ratio = mapped_spectrum[high_bin] / mapped_spectrum[low_bin]
    assert mapped_ratio == pytest.approx(raw_ratio, rel=0.2)
    assert result.metadata["tonal_transfer"] == "broadband_gain_only"
    assert result.metadata["algorithm"] == "frequency_masked_pan_v2"


def test_stereo_passthrough_is_bit_identical_when_input_is_the_only_stem(
    tmp_path: Path,
) -> None:
    sample_rate = 8_000
    time = np.arange(sample_rate) / sample_rate
    tone = 0.2 * np.sin(2 * np.pi * 440 * time)
    original = np.column_stack((tone, tone * 0.6))
    original_path = tmp_path / "original.wav"
    write_wav(original_path, sample_rate, original)

    result = map_stems_to_stereo(
        original_path,
        {"sfx": original_path},
        tmp_path / "mapped",
        n_fft=512,
        hop=128,
    )["sfx"]

    assert result.path.read_bytes() == original_path.read_bytes()
    assert result.metadata["algorithm"] == "stereo_identity_passthrough_v1"
    assert result.metadata["tonal_transfer"] == "identity"
    assert result.metadata["raw_channels"] == 2


class BackfillAWS:
    def __init__(self, original: Path, stem: Path):
        self.files = {"chunk.wav": original, "voice.wav": stem}
        self.items = [
            {
                "PK": "JOB#job-1",
                "SK": "CHUNK#source-1#000000",
                "entity": "chunk",
                "status": "complete",
                "source_id": "source-1",
                "chunk_id": "000000",
                "s3_key": "chunk.wav",
            },
            {
                "PK": "JOB#job-1",
                "SK": "STEM#source-1#000000#voice",
                "entity": "stem",
                "source_id": "source-1",
                "chunk_id": "000000",
                "stem_type": "voice",
                "s3_key": "voice.wav",
            },
        ]
        self.uploads: dict[str, Path] = {}
        self.json_uploads: dict[str, Any] = {}

    def query_partition(self, _: str) -> list[dict[str, Any]]:
        return self.items

    def object_exists(self, key: str) -> bool:
        return key in self.uploads

    def download_file(self, key: str, destination: Path) -> None:
        shutil.copyfile(self.files[key], destination)

    def upload_file(self, path: Path, key: str, _: str) -> None:
        saved = path.parent / f"saved-{path.name}"
        shutil.copyfile(path, saved)
        self.uploads[key] = saved

    def upload_json(self, value: Any, key: str) -> None:
        self.json_uploads[key] = value

    def update(self, _: str, sk: str, values: dict[str, Any]) -> None:
        next(item for item in self.items if item["SK"] == sk).update(values)


def test_backfill_persists_companion_without_replacing_raw_stem(tmp_path: Path) -> None:
    sample_rate = 8_000
    time = np.arange(sample_rate) / sample_rate
    stem = 0.2 * np.sin(2 * np.pi * 440 * time)
    original = np.column_stack((stem, stem * 0.4))
    original_path = tmp_path / "original.wav"
    stem_path = tmp_path / "voice.wav"
    write_wav(original_path, sample_rate, original)
    write_wav(stem_path, sample_rate, stem)
    aws = BackfillAWS(original_path, stem_path)

    result = backfill_job(object(), aws, "job-1")

    record = aws.items[1]
    assert result == {"chunks": 1, "stems": 1, "skipped": 0}
    assert record["s3_key"] == "voice.wav"
    assert record["stereo_s3_key"].endswith("voice.stereo.wav")
    assert record["stereo_bytes"] > 0
    assert record["stereo_mapping"]["mapped_channels"] == 2
    assert aws.items[0]["stereo_mapped_stems"] == ["voice"]


def test_backfill_keeps_presence_passthrough_bit_identical(tmp_path: Path) -> None:
    sample_rate = 8_000
    time = np.arange(sample_rate) / sample_rate
    tone = 0.2 * np.sin(2 * np.pi * 440 * time)
    original = np.column_stack((tone, tone * 0.4))
    original_path = tmp_path / "original.wav"
    write_wav(original_path, sample_rate, original)
    aws = BackfillAWS(original_path, original_path)
    record = aws.items[1]
    record["stem_type"] = "sfx"
    record["model"] = "semantic_presence_passthrough"

    result = backfill_job(object(), aws, "job-1")

    assert result == {"chunks": 1, "stems": 1, "skipped": 0}
    assert record["stereo_sha256"] == digest(original_path)
    assert record["stereo_bytes"] == original_path.stat().st_size
    assert record["stereo_mapping"]["algorithm"] == "stereo_identity_passthrough_v1"


class SeparationAWS:
    def __init__(self, original: Path):
        self.original = original
        self.source = {
            "PK": "JOB#job-1",
            "SK": "SOURCE#source-1",
            "entity": "source",
            "job_id": "job-1",
            "source_id": "source-1",
            "filename": "original.wav",
            "status": "chunked",
            "s3_key": "source.wav",
        }
        self.chunk = {
            "PK": "JOB#job-1",
            "SK": "CHUNK#source-1#000000",
            "entity": "chunk",
            "job_id": "job-1",
            "source_id": "source-1",
            "chunk_id": "000000",
            "status": "queued",
            "s3_key": "chunk.wav",
        }
        self.records: list[StemRecord] = []
        self.uploads: list[str] = []

    def get(self, _: str, sk: str) -> dict[str, Any] | None:
        return self.chunk if sk == self.chunk["SK"] else None

    def update(self, _: str, sk: str, values: dict[str, Any]) -> None:
        if sk == self.chunk["SK"]:
            self.chunk.update(values)
        if sk == self.source["SK"]:
            self.source.update(values)

    def download_file(self, _: str, destination: Path) -> None:
        shutil.copyfile(self.original, destination)

    def upload_file(self, _: Path, key: str, __: str) -> None:
        self.uploads.append(key)

    def upload_json(self, _: Any, __: str) -> None:
        pass

    def put(self, _: dict[str, Any]) -> None:
        pass

    def put_stem(self, record: StemRecord) -> str:
        self.records.append(record)
        return "review"

    def put_model_task(self, *_: Any) -> bool:
        return False

    def query_partition(self, _: str) -> list[dict[str, Any]]:
        return [
            self.source,
            self.chunk,
            *[
                {"entity": "stem", "stem_type": record.stem_type}
                for record in self.records
            ],
        ]


class SeparationClient:
    def __init__(self, stems: dict[str, Path]):
        self.stems = stems

    def separate(self, *_: Any, **__: Any) -> SimpleNamespace:
        return SimpleNamespace(
            stems=self.stems,
            metadata={
                "requested_order": "music_first",
                "cascade_order": ["music", "voice"],
                "verification_status": "success",
                "verification": {
                    "stage_statuses": {"stage1": "success", "stage2": "success"}
                },
                "stages": {
                    "stage1": {"verification": {}},
                    "stage2": {"verification": {}},
                },
                "inference_timings_ms": {},
                "model": "test",
            },
        )


def pipeline_settings() -> Settings:
    return Settings(
        aws_region="us-east-1",
        bucket="bucket",
        table_name="table",
        ingest_queue_url="ingest",
        sam_queue_url="sam",
        flamingo_queue_url="flamingo",
        sam_api_url="http://127.0.0.1:8000",
        flamingo_api_url="http://127.0.0.1:8001",
        chunk_seconds=30,
        overlap_seconds=5,
        gate_peak_dbfs=-52,
        gate_rms_dbfs=-60,
        presign_seconds=3600,
    )


def test_separation_handler_persists_raw_and_mapped_companions(tmp_path: Path) -> None:
    sample_rate = 8_000
    time = np.arange(sample_rate) / sample_rate
    tone = 0.2 * np.sin(2 * np.pi * 440 * time)
    original = tmp_path / "original.wav"
    write_wav(original, sample_rate, np.column_stack((tone, tone * 0.4)))
    stems: dict[str, Path] = {}
    for stem_type, scale in (("music", 0.4), ("voice", 0.35), ("sfx", 0.25)):
        path = tmp_path / f"{stem_type}.wav"
        write_wav(path, sample_rate, tone * scale)
        stems[stem_type] = path
    aws = SeparationAWS(original)
    handler = SeparationHandler(pipeline_settings(), aws)
    handler.client = SeparationClient(stems)

    handler.handle(
        QueueTask(
            task_id="sam-1",
            task_type="separate_chunk",
            job_id="job-1",
            source_id="source-1",
            chunk_id="000000",
        )
    )

    assert len(aws.records) == 3
    assert len(aws.uploads) == 8
    assert sum(
        key.endswith(".stereo.wav") and not key.endswith(".joined.stereo.wav")
        for key in aws.uploads
    ) == 3
    assert sum(key.endswith(".joined.stereo.wav") for key in aws.uploads) == 2
    assert all(record.stereo_s3_key for record in aws.records)
    assert all(record.stereo_bytes > 0 for record in aws.records)
    assert all(record.stereo_mapping["mapped_channels"] == 2 for record in aws.records)
    assert aws.chunk["reconstruction"]["metrics"]["similarity_score"] > 0
    assert aws.source["reconstruction"]["metrics"]["similarity_score"] > 0


def test_separation_handler_bypasses_sam_for_sfx_only_presence(
    tmp_path: Path,
) -> None:
    sample_rate = 8_000
    time = np.arange(sample_rate) / sample_rate
    tone = 0.2 * np.sin(2 * np.pi * 440 * time)
    original = tmp_path / "original.wav"
    write_wav(original, sample_rate, np.column_stack((tone, tone * 0.5)))
    aws = SeparationAWS(original)
    handler = SeparationHandler(pipeline_settings(), aws)

    handler.handle(
        QueueTask(
            task_id="sam-sfx-only",
            task_type="separate_chunk",
            job_id="job-1",
            source_id="source-1",
            chunk_id="000000",
            targets=[],
        )
    )

    assert len(aws.records) == 1
    record = aws.records[0]
    assert record.stem_type == "sfx"
    assert record.model == "semantic_presence_passthrough"
    assert record.sha256 == record.stereo_sha256
    assert record.bytes == record.stereo_bytes
    assert record.stereo_mapping["algorithm"] == "stereo_identity_passthrough_v1"
    assert aws.chunk["reconstruction"]["metrics"]["similarity_score"] == 100.0
    assert aws.source["reconstruction"]["metrics"]["similarity_score"] == 100.0
    assert aws.chunk["stored_stems"] == ["sfx"]
