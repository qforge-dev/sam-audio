"""FFmpeg-backed chunking and a deterministic PCM sound gate."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import GateMetrics


@dataclass(frozen=True)
class Chunk:
    index: int
    start_seconds: float
    end_seconds: float
    path: Path
    sha256: str
    gate: GateMetrics

    @property
    def chunk_id(self) -> str:
        return f"{self.index:06d}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def probe_duration(path: Path) -> float:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid audio duration for {path}: {duration}")
    return duration


def probe_channels(path: Path) -> int:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = json.loads(result.stdout).get("streams", [])
    channels = int(streams[0].get("channels") or 0) if streams else 0
    if channels <= 0:
        raise ValueError(f"Audio channel count is unavailable for {path}")
    return channels


LOSSLESS_CODECS = {
    "alac",
    "ape",
    "flac",
    "mlp",
    "truehd",
    "tta",
    "wavpack",
}


def _integer(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _floating(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _sample_format_bits(sample_format: str) -> int | None:
    digits = "".join(character for character in sample_format if character.isdigit())
    return _integer(digits)


def probe_audio_profile(path: Path) -> dict[str, Any]:
    """Return original-file channel and encoding facts from its first audio stream."""
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            (
                "stream=codec_name,codec_long_name,sample_fmt,sample_rate,channels,"
                "channel_layout,bits_per_sample,bits_per_raw_sample,bit_rate:"
                "format=format_name,format_long_name,bit_rate,duration,size"
            ),
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"No audio stream is available for {path}")
    stream = streams[0]
    container = payload.get("format") or {}
    channels = _integer(stream.get("channels"))
    sample_rate = _integer(stream.get("sample_rate"))
    if channels is None or sample_rate is None:
        raise ValueError(
            f"Audio channel/sample-rate metadata is unavailable for {path}"
        )
    codec = str(stream.get("codec_name") or "unknown")
    sample_format = str(stream.get("sample_fmt") or "")
    bit_depth = (
        _integer(stream.get("bits_per_raw_sample"))
        or _integer(stream.get("bits_per_sample"))
        or (_sample_format_bits(sample_format) if codec.startswith("pcm_") else None)
    )
    bitrate = _integer(stream.get("bit_rate")) or _integer(container.get("bit_rate"))
    lossless = codec.startswith("pcm_") or codec in LOSSLESS_CODECS
    if lossless and bit_depth and bit_depth >= 24 and sample_rate >= 88_200:
        quality_tier = "hi_res_lossless"
    elif lossless:
        quality_tier = "lossless"
    elif bitrate and bitrate >= 256_000:
        quality_tier = "high_bitrate_lossy"
    elif bitrate and bitrate >= 128_000:
        quality_tier = "standard_bitrate_lossy"
    elif bitrate:
        quality_tier = "low_bitrate_lossy"
    else:
        quality_tier = "compressed_unknown_bitrate"
    channel_label = (
        "Mono"
        if channels == 1
        else "Stereo"
        if channels == 2
        else f"{channels} channels"
    )
    return {
        "schema_version": 1,
        "channels": channels,
        "channel_layout": str(stream.get("channel_layout") or "unknown"),
        "channel_label": channel_label,
        "is_stereo": channels == 2,
        "sample_rate_hz": sample_rate,
        "bit_depth": bit_depth,
        "sample_format": sample_format or None,
        "bitrate_bps": bitrate,
        "codec": codec,
        "codec_name": str(stream.get("codec_long_name") or codec),
        "container": str(container.get("format_name") or "unknown"),
        "container_name": str(container.get("format_long_name") or "unknown"),
        "lossless": lossless,
        "quality_tier": quality_tier,
        "duration_seconds": _floating(container.get("duration")),
        "bytes": _integer(container.get("size")) or path.stat().st_size,
    }


def gate_wav(
    path: Path,
    *,
    peak_threshold_dbfs: float = -52.0,
    rms_threshold_dbfs: float = -60.0,
) -> GateMetrics:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        sample_width = source.getsampwidth()
        frame_count = source.getnframes()
        frames = source.readframes(frame_count)
    if sample_width != 2:
        raise ValueError(f"Sound gate requires PCM16 WAV, got {sample_width * 8}-bit")
    samples = array("h")
    samples.frombytes(frames)
    if not samples:
        peak = 0
        rms = 0.0
    else:
        peak = max(abs(sample) for sample in samples)
        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))

    def dbfs(amplitude: float) -> float:
        return max(-120.0, 20.0 * math.log10(max(amplitude, 1e-6) / 32768.0))

    peak_dbfs = dbfs(float(peak))
    rms_dbfs = dbfs(rms)
    duration = frame_count / sample_rate if sample_rate else 0.0
    return GateMetrics(
        audible=(
            duration > 0
            and peak_dbfs >= peak_threshold_dbfs
            and rms_dbfs >= rms_threshold_dbfs
        ),
        peak_dbfs=peak_dbfs,
        rms_dbfs=rms_dbfs,
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channels,
    )


def chunk_audio(
    source: Path,
    output_dir: Path,
    *,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 5.0,
    sample_rate: int = 48_000,
    peak_threshold_dbfs: float = -52.0,
    rms_threshold_dbfs: float = -60.0,
) -> list[Chunk]:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be in [0, chunk_seconds)")
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(source)
    step = chunk_seconds - overlap_seconds
    chunks: list[Chunk] = []
    index = 0
    start = 0.0
    while start < duration - 0.001:
        end = min(duration, start + chunk_seconds)
        output = output_dir / f"{index:06d}.wav"
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-ss",
                f"{start:.6f}",
                "-i",
                str(source),
                "-t",
                f"{end - start:.6f}",
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sample_rate),
                str(output),
            ]
        )
        gate = gate_wav(
            output,
            peak_threshold_dbfs=peak_threshold_dbfs,
            rms_threshold_dbfs=rms_threshold_dbfs,
        )
        chunks.append(
            Chunk(
                index=index,
                start_seconds=start,
                end_seconds=end,
                path=output,
                sha256=sha256_file(output),
                gate=gate,
            )
        )
        if end >= duration:
            break
        index += 1
        start += step
    return chunks
