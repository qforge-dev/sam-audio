from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from sam_audio_pipeline.audio import chunk_audio, gate_wav, probe_audio_profile


def write_tone(
    path: Path, duration: float, amplitude: float, rate: int = 8_000
) -> None:
    frames = bytearray()
    for index in range(round(duration * rate)):
        value = round(amplitude * 32767 * math.sin(2 * math.pi * 440 * index / rate))
        frames.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(frames)


def test_gate_rejects_silence_and_keeps_audible_tone(tmp_path: Path) -> None:
    silence = tmp_path / "silence.wav"
    tone = tmp_path / "tone.wav"
    write_tone(silence, 1.0, 0.0)
    write_tone(tone, 1.0, 0.25)

    assert gate_wav(silence).audible is False
    audible = gate_wav(tone)
    assert audible.audible is True
    assert audible.peak_dbfs == pytest.approx(-12.04, abs=0.1)


def test_probe_audio_profile_reports_channel_layout_and_quality(tmp_path: Path) -> None:
    tone = tmp_path / "tone.wav"
    write_tone(tone, 1.0, 0.25)

    profile = probe_audio_profile(tone)

    assert profile["channel_label"] == "Mono"
    assert profile["is_stereo"] is False
    assert profile["channels"] == 1
    assert profile["sample_rate_hz"] == 8_000
    assert profile["bit_depth"] == 16
    assert profile["bitrate_bps"] == 128_000
    assert profile["codec"] == "pcm_s16le"
    assert profile["container"] == "wav"
    assert profile["lossless"] is True
    assert profile["quality_tier"] == "lossless"


def test_chunking_uses_30_seconds_with_5_second_overlap(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    write_tone(source, 65.0, 0.1)

    chunks = chunk_audio(source, tmp_path / "chunks")

    assert [(chunk.start_seconds, chunk.end_seconds) for chunk in chunks] == [
        (0.0, 30.0),
        (25.0, 55.0),
        (50.0, 65.0),
    ]
    assert all(chunk.gate.audible for chunk in chunks)
    assert all(len(chunk.sha256) == 64 for chunk in chunks)


def test_chunking_does_not_add_redundant_tail_at_exact_boundary(
    tmp_path: Path,
) -> None:
    source = tmp_path / "exact.wav"
    write_tone(source, 30.0, 0.1)

    chunks = chunk_audio(source, tmp_path / "exact-chunks")

    assert [(chunk.start_seconds, chunk.end_seconds) for chunk in chunks] == [
        (0.0, 30.0)
    ]
