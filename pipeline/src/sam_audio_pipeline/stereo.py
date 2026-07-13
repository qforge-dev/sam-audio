"""Frequency-aware stereo remapping for mono SAM Audio stems."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .audio import sha256_file
from .schema import utc_now

if TYPE_CHECKING:
    from .aws import PipelineAWS
    from .config import Settings

EPSILON = 1e-10


@dataclass(frozen=True)
class StereoMappedStem:
    path: Path
    sha256: str
    bytes: int
    metadata: dict[str, Any]


def _read_pcm16(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        sample_width = source.getsampwidth()
        frames = source.readframes(source.getnframes())
    if sample_width != 2:
        raise ValueError(f"Stereo mapper requires PCM16 WAV: {path}")
    values = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if not len(values) or len(values) % channels:
        raise ValueError(f"Stereo mapper received invalid PCM frames: {path}")
    return sample_rate, values.reshape(-1, channels)


def _write_pcm16(path: Path, sample_rate: int, samples: np.ndarray) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(samples), initial=0.0))
    limiter_gain = min(1.0, 0.999 / peak) if peak else 1.0
    limited = np.clip(samples * limiter_gain, -0.999, 0.999)
    encoded = np.rint(limited * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(encoded)
    return limiter_gain


def _aligned_mono(samples: np.ndarray, length: int) -> np.ndarray:
    mono = np.mean(samples, axis=1, dtype=np.float32)
    if len(mono) >= length:
        return mono[:length].copy()
    return np.pad(mono, (0, length - len(mono)))


def _stft(signal: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    edge = n_fft // 2
    padded = np.pad(signal, (edge, edge))
    remainder = (len(padded) - n_fft) % hop
    if remainder:
        padded = np.pad(padded, (0, hop - remainder))
    frame_count = 1 + (len(padded) - n_fft) // hop
    shape = (frame_count, n_fft)
    strides = (padded.strides[0] * hop, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        padded, shape=shape, strides=strides, writeable=False
    )
    window = np.hanning(n_fft).astype(np.float32)
    return np.fft.rfft(frames * window, axis=1).T


def _istft(spectrum: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float32)
    frames = np.fft.irfft(spectrum.T, n=n_fft, axis=1).real.astype(np.float32)
    frames *= window
    output_length = n_fft + hop * (len(frames) - 1)
    output = np.zeros(output_length, dtype=np.float32)
    normalization = np.zeros(output_length, dtype=np.float32)
    window_power = window * window
    for index, frame in enumerate(frames):
        start = index * hop
        output[start : start + n_fft] += frame
        normalization[start : start + n_fft] += window_power
    valid = normalization > 1e-8
    output[valid] /= normalization[valid]
    edge = n_fft // 2
    return output[edge : edge + length]


def _band_slices(sample_rate: int, n_fft: int, count: int) -> list[slice]:
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    edges = np.concatenate(([0.0], np.geomspace(60.0, sample_rate / 2, count)))
    indexes = np.searchsorted(frequencies, edges, side="left")
    indexes[0] = 0
    indexes[-1] = len(frequencies)
    slices: list[slice] = []
    for index in range(count):
        start = int(indexes[index])
        stop = max(start + 1, int(indexes[index + 1]))
        slices.append(slice(start, min(stop, len(frequencies))))
    return slices


def _bidirectional_ema(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    neutral: float,
) -> np.ndarray:
    scale = np.quantile(weights, 0.75, axis=1, keepdims=True)
    confidence = np.clip(weights / np.maximum(scale, EPSILON), 0.0, 1.0)

    def one_direction(reverse: bool) -> np.ndarray:
        result = np.empty_like(values, dtype=np.float32)
        order = (
            range(values.shape[1] - 1, -1, -1)
            if reverse
            else range(values.shape[1])
        )
        for band in range(values.shape[0]):
            reliable = np.flatnonzero(weights[band] > EPSILON)
            if len(reliable):
                initial_index = reliable[-1] if reverse else reliable[0]
                state = float(values[band, initial_index])
            else:
                state = neutral
            for frame in order:
                amount = alpha * float(confidence[band, frame])
                state += amount * (float(values[band, frame]) - state)
                result[band, frame] = state
        return result

    smoothed = (one_direction(False) + one_direction(True)) * 0.5
    if len(smoothed) > 1:
        padded = np.pad(smoothed, ((1, 1), (0, 0)), mode="edge")
        smoothed = (
            0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
        )
    return smoothed.astype(np.float32)


def _broadband_mapping(
    original: np.ndarray,
    stem: np.ndarray,
    *,
    frame_count: int,
    sample_rate: int,
    hop: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = max(hop, round(sample_rate * 0.1))
    half = window // 2
    raw = np.zeros(frame_count, dtype=np.float32)
    raw_gain_db = np.zeros(frame_count, dtype=np.float32)
    weights = np.zeros(frame_count, dtype=np.float32)
    for frame in range(frame_count):
        center = frame * hop
        start = max(0, center - half)
        stop = min(len(stem), center + half)
        target = stem[start:stop]
        target_power = float(np.dot(target, target))
        if target_power <= EPSILON:
            continue
        gain_left = abs(float(np.dot(target, original[start:stop, 0]))) / target_power
        gain_right = abs(float(np.dot(target, original[start:stop, 1]))) / target_power
        raw[frame] = (gain_right - gain_left) / max(
            gain_right + gain_left, EPSILON
        )
        raw_gain_db[frame] = np.clip(
            10.0 * np.log10(max(gain_left**2 + gain_right**2, EPSILON)),
            -6.0,
            6.0,
        )
        weights[frame] = target_power
    smoothed_pan = _bidirectional_ema(
        raw[None, :], weights[None, :], alpha=alpha, neutral=0.0
    )[0]
    smoothed_gain_db = _bidirectional_ema(
        raw_gain_db[None, :], weights[None, :], alpha=alpha, neutral=0.0
    )[0]
    return (
        np.sign(smoothed_pan) * np.sqrt(np.abs(smoothed_pan)),
        smoothed_gain_db,
        weights,
    )


def _curve(
    pan: np.ndarray,
    gain_db: np.ndarray,
    weights: np.ndarray,
    *,
    hop: int,
    sample_rate: int,
    max_points: int = 120,
) -> list[dict[str, float]]:
    weight_sum = np.sum(weights, axis=0)
    confidence = weights / np.maximum(weight_sum[None, :], EPSILON)
    pan_by_frame = np.where(
        weight_sum > EPSILON, np.sum(pan * confidence, axis=0), np.mean(pan, axis=0)
    )
    gain_by_frame = np.where(
        weight_sum > EPSILON,
        np.sum(gain_db * confidence, axis=0),
        np.mean(gain_db, axis=0),
    )
    frame_count = len(pan_by_frame)
    indexes = np.unique(
        np.linspace(
            0, max(frame_count - 1, 0), min(frame_count, max_points)
        ).astype(int)
    )
    return [
        {
            "time_seconds": round(float(index * hop / sample_rate), 3),
            "pan": round(float(pan_by_frame[index]), 4),
            "gain_db": round(float(gain_by_frame[index]), 3),
        }
        for index in indexes
    ]


def _identity_stereo_mapping(
    original_path: Path,
    output_path: Path,
    original: np.ndarray,
    sample_rate: int,
    *,
    smoothing_alpha: float,
    n_fft: int,
    hop: int,
) -> StereoMappedStem:
    mono = np.mean(original[:, :2], axis=1, dtype=np.float32)
    frame_count = _stft(mono, n_fft, hop).shape[1]
    pan, _, weights = _broadband_mapping(
        original[:, :2],
        mono,
        frame_count=frame_count,
        sample_rate=sample_rate,
        hop=hop,
        alpha=smoothing_alpha,
    )
    curve = _curve(
        pan[None, :],
        np.zeros((1, frame_count), dtype=np.float32),
        weights[None, :],
        hop=hop,
        sample_rate=sample_rate,
    )
    pan_values = [point["pan"] for point in curve] or [0.0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original_path, output_path)
    metadata = {
        "algorithm": "stereo_identity_passthrough_v1",
        "sample_rate": sample_rate,
        "source_channels": 2,
        "source_was_mono": False,
        "raw_channels": 2,
        "mapped_channels": 2,
        "frequency_bands": 0,
        "n_fft": n_fft,
        "hop_samples": hop,
        "smoothing": {
            "type": "bidirectional_ema",
            "alpha": smoothing_alpha,
            "frequency_kernel": [],
        },
        "tonal_transfer": "identity",
        "pan_summary": {
            "start": pan_values[0],
            "end": pan_values[-1],
            "mean": round(float(np.mean(pan_values)), 4),
            "minimum": min(pan_values),
            "maximum": max(pan_values),
        },
        "gain_summary_db": {"mean": 0.0, "minimum": 0.0, "maximum": 0.0},
        "limiter_gain": 1.0,
        "pan_curve": curve,
    }
    return StereoMappedStem(
        path=output_path,
        sha256=sha256_file(output_path),
        bytes=output_path.stat().st_size,
        metadata=metadata,
    )


def map_stems_to_stereo(
    original_path: Path,
    stem_paths: dict[str, Path],
    output_dir: Path,
    *,
    band_count: int = 32,
    smoothing_alpha: float = 0.03,
    n_fft: int = 2048,
    hop: int = 512,
) -> dict[str, StereoMappedStem]:
    """Transfer the original mix's smoothed pan/loudness map onto mono stems."""
    started = time.perf_counter()
    sample_rate, original = _read_pcm16(original_path)
    source_channels = int(original.shape[1])
    if original.shape[1] == 1:
        original = np.repeat(original, 2, axis=1)
        original_was_mono = True
    else:
        original = original[:, :2]
        original_was_mono = False
    sample_count = len(original)
    if not stem_paths:
        return {}
    mono_stems: dict[str, np.ndarray] = {}
    raw_channels: dict[str, int] = {}
    identity_stems: set[str] = set()
    for stem_type, path in stem_paths.items():
        stem_rate, samples = _read_pcm16(path)
        if stem_rate != sample_rate:
            raise ValueError(
                f"Sample-rate mismatch for {stem_type}: {stem_rate} != {sample_rate}"
            )
        raw_channels[stem_type] = int(samples.shape[1])
        mono_stems[stem_type] = _aligned_mono(samples, sample_count)
        if (
            source_channels == 2
            and raw_channels[stem_type] == 2
            and path.resolve() == original_path.resolve()
        ):
            identity_stems.add(stem_type)

    output_dir.mkdir(parents=True, exist_ok=True)
    mapped: dict[str, StereoMappedStem] = {}
    for stem_type in identity_stems:
        mapped[stem_type] = _identity_stereo_mapping(
            original_path,
            output_dir / f"{stem_type}.stereo.wav",
            original,
            sample_rate,
            smoothing_alpha=smoothing_alpha,
            n_fft=n_fft,
            hop=hop,
        )
    processable_stems = {
        stem_type: samples
        for stem_type, samples in mono_stems.items()
        if stem_type not in identity_stems
    }
    if not processable_stems:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        for result in mapped.values():
            result.metadata["processing_ms"] = round(elapsed_ms, 3)
        return mapped

    left_spectrum = _stft(original[:, 0], n_fft, hop)
    right_spectrum = _stft(original[:, 1], n_fft, hop)
    stem_spectra = {
        stem_type: _stft(samples, n_fft, hop)
        for stem_type, samples in processable_stems.items()
    }
    powers = {
        stem_type: np.abs(spectrum).astype(np.float32) ** 2
        for stem_type, spectrum in stem_spectra.items()
    }
    total_power = np.sum(np.stack(list(powers.values())), axis=0) + EPSILON
    left_power = np.abs(left_spectrum).astype(np.float32) ** 2
    right_power = np.abs(right_spectrum).astype(np.float32) ** 2
    bands = _band_slices(sample_rate, n_fft, band_count)
    for stem_type, stem_spectrum in stem_spectra.items():
        mask = powers[stem_type] / total_power
        frame_count = stem_spectrum.shape[1]
        raw_pan = np.zeros((band_count, frame_count), dtype=np.float32)
        weights = np.zeros_like(raw_pan)
        for band_index, frequencies in enumerate(bands):
            stem_band = stem_spectrum[frequencies]
            stem = np.sum(powers[stem_type][frequencies], axis=0)
            left = np.sum(left_power[frequencies], axis=0)
            right = np.sum(right_power[frequencies], axis=0)
            cross_left = np.sum(
                left_spectrum[frequencies] * np.conj(stem_band), axis=0
            )
            cross_right = np.sum(
                right_spectrum[frequencies] * np.conj(stem_band), axis=0
            )
            gain_left = np.abs(cross_left) / np.maximum(stem, EPSILON)
            gain_right = np.abs(cross_right) / np.maximum(stem, EPSILON)
            raw_pan[band_index] = (gain_right - gain_left) / np.maximum(
                gain_right + gain_left, EPSILON
            )
            coherence_left = np.abs(cross_left) ** 2 / np.maximum(
                left * stem, EPSILON
            )
            coherence_right = np.abs(cross_right) ** 2 / np.maximum(
                right * stem, EPSILON
            )
            dominance = np.sum(
                mask[frequencies] * powers[stem_type][frequencies], axis=0
            ) / np.maximum(stem, EPSILON)
            weights[band_index] = (
                stem
                * np.sqrt(np.maximum(coherence_left, coherence_right))
                * dominance
            )

        band_pan = _bidirectional_ema(
            raw_pan, weights, alpha=smoothing_alpha, neutral=0.0
        )
        band_pan = np.sign(band_pan) * np.sqrt(np.abs(band_pan))
        broadband_pan, broadband_gain_db, _ = _broadband_mapping(
            original,
            mono_stems[stem_type],
            frame_count=frame_count,
            sample_rate=sample_rate,
            hop=hop,
            alpha=smoothing_alpha,
        )
        band_weight_sum = np.sum(weights, axis=0)
        normalized_weights = weights / np.maximum(
            band_weight_sum[None, :], EPSILON
        )
        band_center = np.where(
            band_weight_sum > EPSILON,
            np.sum(band_pan * normalized_weights, axis=0),
            np.mean(band_pan, axis=0),
        )
        pan = np.clip(
            broadband_pan[None, :] + 0.25 * (band_pan - band_center[None, :]),
            -0.95,
            0.95,
        )
        gain_db = np.broadcast_to(
            broadband_gain_db[None, :], (band_count, frame_count)
        )
        gain = 10.0 ** (gain_db / 20.0)
        angle = (pan + 1.0) * (math.pi / 4.0)
        left_mapped = np.zeros_like(stem_spectrum)
        right_mapped = np.zeros_like(stem_spectrum)
        for band_index, frequencies in enumerate(bands):
            left_mapped[frequencies] = (
                stem_spectrum[frequencies]
                * gain[band_index]
                * np.cos(angle[band_index])
            )
            right_mapped[frequencies] = (
                stem_spectrum[frequencies]
                * gain[band_index]
                * np.sin(angle[band_index])
            )
        stereo = np.column_stack(
            (
                _istft(left_mapped, n_fft, hop, sample_count),
                _istft(right_mapped, n_fft, hop, sample_count),
            )
        )
        output_path = output_dir / f"{stem_type}.stereo.wav"
        limiter_gain = _write_pcm16(output_path, sample_rate, stereo)
        curve = _curve(pan, gain_db, weights, hop=hop, sample_rate=sample_rate)
        pan_values = [point["pan"] for point in curve] or [0.0]
        gain_values = [point["gain_db"] for point in curve] or [0.0]
        metadata = {
            "algorithm": "frequency_masked_pan_v2",
            "sample_rate": sample_rate,
            "source_channels": source_channels,
            "source_was_mono": original_was_mono,
            "raw_channels": raw_channels[stem_type],
            "mapped_channels": 2,
            "frequency_bands": band_count,
            "n_fft": n_fft,
            "hop_samples": hop,
            "smoothing": {
                "type": "bidirectional_ema",
                "alpha": smoothing_alpha,
                "frequency_kernel": [0.25, 0.5, 0.25],
            },
            "tonal_transfer": "broadband_gain_only",
            "pan_summary": {
                "start": pan_values[0],
                "end": pan_values[-1],
                "mean": round(float(np.mean(pan_values)), 4),
                "minimum": min(pan_values),
                "maximum": max(pan_values),
            },
            "gain_summary_db": {
                "mean": round(float(np.mean(gain_values)), 3),
                "minimum": min(gain_values),
                "maximum": max(gain_values),
            },
            "limiter_gain": round(limiter_gain, 6),
            "pan_curve": curve,
        }
        mapped[stem_type] = StereoMappedStem(
            path=output_path,
            sha256=sha256_file(output_path),
            bytes=output_path.stat().st_size,
            metadata=metadata,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    for result in mapped.values():
        result.metadata["processing_ms"] = round(elapsed_ms, 3)
    return mapped


def backfill_job(
    settings: Settings,
    aws: PipelineAWS,
    job_id: str,
    *,
    force: bool = False,
    max_chunks: int | None = None,
) -> dict[str, int]:
    items = aws.query_partition(f"JOB#{job_id}")
    chunks = [
        item
        for item in items
        if item.get("entity") == "chunk"
        and item.get("status") == "complete"
        and item.get("s3_key")
    ]
    stems_by_chunk: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item.get("entity") == "stem":
            stems_by_chunk[(str(item["source_id"]), str(item["chunk_id"]))].append(
                item
            )
    result = {"chunks": 0, "stems": 0, "skipped": 0}
    for chunk in chunks[:max_chunks]:
        source_id = str(chunk["source_id"])
        chunk_id = str(chunk["chunk_id"])
        stems = stems_by_chunk[(source_id, chunk_id)]
        pending = [
            stem
            for stem in stems
            if force
            or not stem.get("stereo_s3_key")
            or not aws.object_exists(str(stem["stereo_s3_key"]))
        ]
        if not pending:
            result["skipped"] += 1
            continue
        with tempfile.TemporaryDirectory(prefix="sam-stereo-backfill-") as temporary:
            root = Path(temporary)
            original = root / "original.wav"
            aws.download_file(str(chunk["s3_key"]), original)
            local_stems: dict[str, Path] = {}
            for stem in stems:
                if stem.get("model") == "semantic_presence_passthrough":
                    local_stems[str(stem["stem_type"])] = original
                    continue
                path = root / "raw" / f"{stem['stem_type']}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                aws.download_file(str(stem["s3_key"]), path)
                local_stems[str(stem["stem_type"])] = path
            mapped = map_stems_to_stereo(original, local_stems, root / "mapped")
            mapping_key = (
                f"jobs/{job_id}/metadata/{source_id}/{chunk_id}.stereo.json"
            )
            aws.upload_json(
                {"algorithm": "frequency_masked_pan_v2", "stems": {
                    stem_type: item.metadata for stem_type, item in mapped.items()
                }},
                mapping_key,
            )
            for stem in pending:
                stem_type = str(stem["stem_type"])
                mapped_stem = mapped[stem_type]
                stereo_key = (
                    f"jobs/{job_id}/stems/{source_id}/{chunk_id}/"
                    f"{stem_type}.stereo.wav"
                )
                aws.upload_file(mapped_stem.path, stereo_key, "audio/wav")
                aws.update(
                    f"JOB#{job_id}",
                    str(stem["SK"]),
                    {
                        "stereo_s3_key": stereo_key,
                        "stereo_sha256": mapped_stem.sha256,
                        "stereo_bytes": mapped_stem.bytes,
                        "stereo_mapping": mapped_stem.metadata,
                        "updated_at": utc_now(),
                    },
                )
                result["stems"] += 1
            aws.update(
                f"JOB#{job_id}",
                str(chunk["SK"]),
                {
                    "stereo_mapping_s3_key": mapping_key,
                    "stereo_mapped_stems": sorted(mapped),
                    "updated_at": utc_now(),
                },
            )
        result["chunks"] += 1
    return result


def main() -> None:
    from .aws import PipelineAWS
    from .config import Settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-chunks", type=int)
    args = parser.parse_args()
    settings = Settings.from_env()
    result = backfill_job(
        settings,
        PipelineAWS(settings),
        args.job_id,
        force=args.force,
        max_chunks=args.max_chunks,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
