"""Join mapped stereo stems and measure sample-aligned reconstruction fidelity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
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

EPSILON = 1e-12
PCM16_PEAK = 32767.0 / 32768.0


@dataclass(frozen=True)
class JoinedReconstruction:
    path: Path
    sha256: str
    bytes: int
    metrics: dict[str, Any]


def _read_stereo_pcm16(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        sample_width = source.getsampwidth()
        frames = source.readframes(source.getnframes())
    if channels != 2 or sample_width != 2:
        raise ValueError(f"Reconstruction requires stereo PCM16 WAV: {path}")
    encoded = np.frombuffer(frames, dtype="<i2")
    if not len(encoded) or len(encoded) % 2:
        raise ValueError(f"Reconstruction received invalid PCM frames: {path}")
    return sample_rate, encoded.astype(np.float64).reshape(-1, 2) / 32768.0


def _aligned(samples: np.ndarray, length: int) -> np.ndarray:
    if len(samples) >= length:
        return samples[:length].copy()
    return np.pad(samples, ((0, length - len(samples)), (0, 0)))


def _write_stereo_pcm16(
    path: Path, sample_rate: int, samples: np.ndarray
) -> tuple[float, np.ndarray]:
    peak = float(np.max(np.abs(samples), initial=0.0))
    limiter_gain = min(1.0, PCM16_PEAK / peak) if peak else 1.0
    limited = np.clip(samples * limiter_gain, -1.0, PCM16_PEAK)
    encoded = np.clip(
        np.rint(limited * 32768.0), -32768, 32767
    ).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(encoded.tobytes())
    return limiter_gain, encoded.astype(np.float64) / 32768.0


def _bounded_db(numerator: float, denominator: float) -> float:
    value = 10.0 * math.log10(max(numerator, EPSILON) / max(denominator, EPSILON))
    return max(-120.0, min(120.0, value))


def _agreement(original: np.ndarray, joined: np.ndarray) -> dict[str, float]:
    original_vector = original.reshape(-1)
    joined_vector = joined.reshape(-1)
    original_energy = float(np.dot(original_vector, original_vector))
    joined_energy = float(np.dot(joined_vector, joined_vector))
    dot = float(np.dot(original_vector, joined_vector))
    denominator = original_energy + joined_energy
    if denominator <= EPSILON:
        similarity = 1.0
    else:
        similarity = max(0.0, min(1.0, 2.0 * dot / denominator))
    correlation_denominator = math.sqrt(original_energy * joined_energy)
    correlation = (
        max(-1.0, min(1.0, dot / correlation_denominator))
        if correlation_denominator > EPSILON
        else (1.0 if denominator <= EPSILON else 0.0)
    )
    error = original_vector - joined_vector
    error_energy = float(np.dot(error, error))
    return {
        "similarity_score": round(similarity * 100.0, 4),
        "waveform_correlation": round(correlation, 6),
        "level_delta_db": round(_bounded_db(joined_energy, original_energy), 4),
        "error_to_signal_db": round(_bounded_db(error_energy, original_energy), 4),
        "snr_db": round(_bounded_db(original_energy, error_energy), 4),
        "normalized_rmse": round(
            math.sqrt(error_energy / max(original_energy, EPSILON)), 6
        ),
    }


def _finish_reconstruction(
    original: np.ndarray,
    joined: np.ndarray,
    sample_rate: int,
    output_path: Path,
    metadata: dict[str, Any],
    started: float,
) -> JoinedReconstruction:
    limiter_gain, stored_joined = _write_stereo_pcm16(
        output_path, sample_rate, joined
    )
    overall = _agreement(original, stored_joined)
    channels = {
        name: _agreement(original[:, index], stored_joined[:, index])[
            "similarity_score"
        ]
        for index, name in enumerate(("left", "right"))
    }
    metrics: dict[str, Any] = {
        "algorithm": "sample_aligned_stereo_agreement_v1",
        **overall,
        "channel_similarity": channels,
        "sample_rate": sample_rate,
        "sample_count": len(original),
        "duration_seconds": round(len(original) / sample_rate, 6),
        "limiter_gain": round(limiter_gain, 6),
        "score_semantics": (
            "100 * max(0, 2 * dot(original, joined) / "
            "(energy(original) + energy(joined))); sample-aligned, stereo, "
            "phase-sensitive, and level-sensitive."
        ),
        **metadata,
        "processing_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    return JoinedReconstruction(
        path=output_path,
        sha256=sha256_file(output_path),
        bytes=output_path.stat().st_size,
        metrics=metrics,
    )


def join_stereo_stems(
    original_path: Path,
    stem_paths: dict[str, Path],
    output_path: Path,
) -> JoinedReconstruction:
    """Sum stored stereo variants and score the stored joined PCM against input."""
    if not stem_paths:
        raise ValueError("At least one mapped stereo stem is required")
    started = time.perf_counter()
    sample_rate, original = _read_stereo_pcm16(original_path)
    joined = np.zeros_like(original)
    for stem_type, stem_path in sorted(stem_paths.items()):
        stem_rate, samples = _read_stereo_pcm16(stem_path)
        if stem_rate != sample_rate:
            raise ValueError(
                f"Sample-rate mismatch for {stem_type}: {stem_rate} != {sample_rate}"
            )
        joined += _aligned(samples, len(original))
    return _finish_reconstruction(
        original,
        joined,
        sample_rate,
        output_path,
        {
            "aggregation": "sum_stored_stereo_stems_v1",
            "stem_types": sorted(stem_paths),
        },
        started,
    )


def normalize_source_audio(
    source_path: Path, output_path: Path, *, sample_rate: int = 48_000
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            "-acodec",
            "pcm_s16le",
            str(output_path),
        ],
        check=True,
    )


def stereo_pcm16_sample_rate(path: Path) -> int:
    sample_rate, _ = _read_stereo_pcm16(path)
    return sample_rate


def join_reconstructed_chunks(
    normalized_original_path: Path,
    chunks: list[tuple[str, float, Path]],
    output_path: Path,
) -> JoinedReconstruction:
    """Overlap-average joined chunks into one source-length stereo artifact."""
    if not chunks:
        raise ValueError("At least one joined chunk is required")
    started = time.perf_counter()
    sample_rate, original = _read_stereo_pcm16(normalized_original_path)
    accumulated = np.zeros_like(original)
    weights = np.zeros(len(original), dtype=np.float64)
    included: list[str] = []
    for chunk_id, start_seconds, chunk_path in sorted(chunks):
        chunk_rate, samples = _read_stereo_pcm16(chunk_path)
        if chunk_rate != sample_rate:
            raise ValueError(
                f"Sample-rate mismatch for chunk {chunk_id}: "
                f"{chunk_rate} != {sample_rate}"
            )
        start = max(0, round(start_seconds * sample_rate))
        stop = min(len(original), start + len(samples))
        if stop <= start:
            continue
        length = stop - start
        accumulated[start:stop] += samples[:length]
        weights[start:stop] += 1.0
        included.append(chunk_id)
    covered = weights > 0
    joined = np.zeros_like(original)
    joined[covered] = accumulated[covered] / weights[covered, None]
    return _finish_reconstruction(
        original,
        joined,
        sample_rate,
        output_path,
        {
            "aggregation": "overlap_average_joined_chunks_v1",
            "chunk_ids": included,
            "chunk_count": len(included),
            "coverage_fraction": round(float(np.mean(covered)), 6),
        },
        started,
    )


def reconstruction_record(
    result: JoinedReconstruction, s3_key: str
) -> dict[str, Any]:
    return {
        "s3_key": s3_key,
        "sha256": result.sha256,
        "bytes": result.bytes,
        "metrics": result.metrics,
        "created_at": utc_now(),
    }


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
        if item.get("entity") == "stem" and item.get("stereo_s3_key"):
            stems_by_chunk[(str(item["source_id"]), str(item["chunk_id"]))].append(
                item
            )
    summary = {
        "chunks": 0,
        "skipped": 0,
        "missing_stereo": 0,
        "sources": 0,
        "source_skipped": 0,
        "source_incomplete": 0,
    }
    for chunk in chunks[:max_chunks]:
        existing = chunk.get("reconstruction", {})
        if (
            not force
            and existing.get("s3_key")
            and aws.object_exists(str(existing["s3_key"]))
        ):
            summary["skipped"] += 1
            continue
        source_id = str(chunk["source_id"])
        chunk_id = str(chunk["chunk_id"])
        stems = stems_by_chunk[(source_id, chunk_id)]
        if not stems:
            summary["missing_stereo"] += 1
            continue
        with tempfile.TemporaryDirectory(
            prefix="sam-reconstruction-backfill-"
        ) as temporary:
            root = Path(temporary)
            original = root / "original.wav"
            aws.download_file(str(chunk["s3_key"]), original)
            local_stems: dict[str, Path] = {}
            for stem in stems:
                stem_type = str(stem["stem_type"])
                path = root / "stereo" / f"{stem_type}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                aws.download_file(str(stem["stereo_s3_key"]), path)
                local_stems[stem_type] = path
            joined = join_stereo_stems(
                original, local_stems, root / "joined.stereo.wav"
            )
            joined_key = (
                f"jobs/{job_id}/reconstructions/{source_id}/"
                f"{chunk_id}.joined.stereo.wav"
            )
            aws.upload_file(joined.path, joined_key, "audio/wav")
            aws.update(
                f"JOB#{job_id}",
                str(chunk["SK"]),
                {
                    "reconstruction": reconstruction_record(joined, joined_key),
                    "updated_at": utc_now(),
                },
            )
        summary["chunks"] += 1
    refreshed = aws.query_partition(f"JOB#{job_id}")
    refreshed_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in refreshed:
        if item.get("entity") == "chunk":
            refreshed_chunks[str(item["source_id"])].append(item)
    sources = [item for item in refreshed if item.get("entity") == "source"]
    for source in sources:
        source_id = str(source["source_id"])
        source_chunks = sorted(
            refreshed_chunks[source_id],
            key=lambda item: float(item.get("start_seconds") or 0.0),
        )
        complete = [
            chunk for chunk in source_chunks if chunk.get("status") == "complete"
        ]
        reconstructed = [
            chunk
            for chunk in complete
            if chunk.get("reconstruction", {}).get("s3_key")
        ]
        if not reconstructed or len(reconstructed) != len(complete):
            if complete:
                summary["source_incomplete"] += 1
            continue
        existing = source.get("reconstruction", {})
        if (
            not force
            and existing.get("s3_key")
            and aws.object_exists(str(existing["s3_key"]))
        ):
            summary["source_skipped"] += 1
            continue
        with tempfile.TemporaryDirectory(
            prefix="sam-source-reconstruction-backfill-"
        ) as temporary:
            root = Path(temporary)
            suffix = Path(str(source.get("filename") or "source.audio")).suffix
            source_path = root / f"source{suffix or '.audio'}"
            normalized = root / "source.normalized.wav"
            aws.download_file(str(source["s3_key"]), source_path)
            local_chunks: list[tuple[str, float, Path]] = []
            for chunk in reconstructed:
                path = root / "chunks" / f"{chunk['chunk_id']}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                aws.download_file(
                    str(chunk["reconstruction"]["s3_key"]), path
                )
                local_chunks.append(
                    (
                        str(chunk["chunk_id"]),
                        float(chunk.get("start_seconds") or 0.0),
                        path,
                    )
                )
            normalize_source_audio(
                source_path,
                normalized,
                sample_rate=stereo_pcm16_sample_rate(local_chunks[0][2]),
            )
            joined = join_reconstructed_chunks(
                normalized,
                local_chunks,
                root / "source.joined.stereo.wav",
            )
            joined_key = (
                f"jobs/{job_id}/reconstructions/{source_id}/"
                "source.joined.stereo.wav"
            )
            aws.upload_file(joined.path, joined_key, "audio/wav")
            aws.update(
                f"JOB#{job_id}",
                str(source["SK"]),
                {
                    "reconstruction": reconstruction_record(joined, joined_key),
                    "updated_at": utc_now(),
                },
            )
        summary["sources"] += 1
    return summary


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
