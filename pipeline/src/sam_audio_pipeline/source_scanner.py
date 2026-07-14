"""Whole-source proxy scanning for high-yield cinematic clip acquisition."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .m2d_validator import (
    WINDOW_HOP_SECONDS,
    WINDOW_SECONDS,
    evaluate_probabilities,
    load_label_families,
)

SCAN_POLICY_VERSION = "whole_source_proxy_m2d_v1"
PROXY_SAMPLE_RATE = 16_000
PROXY_ACTIVITY_DBFS = -50.0
REGION_HOP_SECONDS = 5.0
MIN_REGION_FOREGROUND_SPEECH_COVERAGE = 0.48
MAX_REGION_VOCAL_MUSIC_COVERAGE = 0.11


def _region_score(evaluation: dict[str, Any]) -> float:
    """Rank passing regions by foreground dialogue and mixed-background evidence."""
    return (
        4.0 * float(evaluation["foreground_speech_coverage"])
        + 3.0 * float(evaluation["strong_speech_coverage"])
        + 2.0 * float(evaluation["overlap_coverage"])
        + float(evaluation["cinematic_music_coverage"])
        + float(evaluation["cinematic_sfx_coverage"])
        - 4.0 * float(evaluation["vocal_music_coverage"])
    )


def compact_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Keep selection evidence without duplicating per-second label payloads."""
    return {key: value for key, value in evaluation.items() if key != "windows"}


def region_passes_confidence_gate(region: dict[str, Any]) -> bool:
    """Keep enough high-confidence regions to saturate full-quality extraction."""
    evidence = region.get("evidence") or {}
    return (
        float(evidence.get("foreground_speech_coverage") or 0.0)
        >= MIN_REGION_FOREGROUND_SPEECH_COVERAGE
        and float(evidence.get("vocal_music_coverage") or 0.0)
        <= MAX_REGION_VOCAL_MUSIC_COVERAGE
    )


def select_candidate_regions(
    probabilities: np.ndarray,
    labels: list[dict[str, str]],
    families: dict[str, set[int]],
    *,
    clip_seconds: float,
    max_regions: int,
    region_hop_seconds: float = REGION_HOP_SECONDS,
) -> list[dict[str, Any]]:
    """Select high-confidence non-overlapping regions from one scored timeline."""
    if clip_seconds < WINDOW_SECONDS:
        raise ValueError("clip_seconds must cover at least one M2D window")
    if max_regions < 1:
        raise ValueError("max_regions must be positive")
    if region_hop_seconds <= 0:
        raise ValueError("region_hop_seconds must be positive")
    windows_per_region = (
        math.floor((clip_seconds - WINDOW_SECONDS) / WINDOW_HOP_SECONDS) + 1
    )
    region_hop_windows = max(1, round(region_hop_seconds / WINDOW_HOP_SECONDS))
    ranked: list[dict[str, Any]] = []
    for first in range(
        0,
        max(0, len(probabilities) - windows_per_region + 1),
        region_hop_windows,
    ):
        start = first * WINDOW_HOP_SECONDS
        evaluation = evaluate_probabilities(
            probabilities[first : first + windows_per_region],
            labels,
            families,
            starts=[
                start + offset * WINDOW_HOP_SECONDS
                for offset in range(windows_per_region)
            ],
            require_cinematic_mix=True,
        )
        if not evaluation["accepted"]:
            continue
        region = {
            "start_seconds": round(start, 3),
            "end_seconds": round(start + clip_seconds, 3),
            "score": round(_region_score(evaluation), 8),
            "evidence": compact_evaluation(evaluation),
        }
        if region_passes_confidence_gate(region):
            ranked.append(region)
    ranked.sort(key=lambda item: (-float(item["score"]), item["start_seconds"]))
    selected: list[dict[str, Any]] = []
    for region in ranked:
        start = float(region["start_seconds"])
        if any(
            abs(start - float(existing["start_seconds"])) < clip_seconds
            for existing in selected
        ):
            continue
        selected.append(region)
        if len(selected) >= max_regions:
            break
    return selected


class M2DSourceScanner:
    """Load M2D once and scan complete sources through bounded proxy batches."""

    def __init__(
        self,
        *,
        m2d_repo: Path,
        checkpoint: Path,
        class_labels: Path,
        ontology: Path,
        device: str = "cuda",
        batch_size: int = 128,
        inference_concurrency: int = 2,
    ) -> None:
        try:
            import soundfile as sf
            import torch
        except ImportError as error:
            raise RuntimeError(
                "Whole-source scanning requires torch and soundfile"
            ) from error
        sys.path.insert(0, str(m2d_repo / "examples"))
        from portable_m2d import PortableM2D  # type: ignore[import-not-found]

        self.torch = torch
        self.soundfile = sf
        self.labels, self.families = load_label_families(class_labels, ontology)
        self.model = PortableM2D(
            weight_file=str(checkpoint), num_classes=len(self.labels)
        )
        self.model = self.model.to(device).eval()
        self.device = device
        self.batch_size = max(1, batch_size)
        self.sample_rate = int(self.model.cfg.sample_rate)
        self.inference_concurrency = max(1, inference_concurrency)
        self._inference_slots = threading.BoundedSemaphore(
            self.inference_concurrency
        )

    def create_proxy(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-fflags",
                "+genpts",
                "-i",
                str(source),
                "-vn",
                "-threads",
                "1",
                "-ac",
                "2",
                "-ar",
                str(self.sample_rate),
                "-af",
                "aresample=async=1:first_pts=0",
                "-compression_level",
                "2",
                str(destination),
            ],
            check=True,
            timeout=1800,
        )

    def stereo_metrics(self, proxy: Path) -> dict[str, float]:
        """Measure whole-source stereo energy without loading the proxy at once."""
        side_energy = total_energy = 0.0
        frames = 0
        with self.soundfile.SoundFile(str(proxy)) as audio:
            if audio.channels != 2 or audio.samplerate != self.sample_rate:
                raise ValueError("Source proxy must be stereo at the M2D sample rate")
            while True:
                block = audio.read(
                    self.sample_rate * 60, dtype="float32", always_2d=True
                )
                if not len(block):
                    break
                side = (block[:, 0] - block[:, 1]) / 2.0
                side_energy += float(np.sum(np.square(side), dtype=np.float64))
                total_energy += float(np.sum(np.square(block), dtype=np.float64))
                frames += len(block)
        side_mean = side_energy / max(1, frames)
        total_mean = total_energy / max(1, frames * 2)
        return {
            "side_to_total_db": round(
                10.0
                * math.log10(max(side_mean, 1e-12) / max(total_mean, 1e-12)),
                4,
            )
        }

    def _probabilities(self, proxy: Path) -> tuple[np.ndarray, int, float]:
        started = time.perf_counter()
        window_frames = round(WINDOW_SECONDS * self.sample_rate)
        model_frames = max(window_frames, 3 * self.sample_rate)
        hop_frames = round(WINDOW_HOP_SECONDS * self.sample_rate)
        all_probabilities: list[np.ndarray] = []
        active_windows = 0
        with self.soundfile.SoundFile(str(proxy)) as audio:
            if audio.channels != 2 or audio.samplerate != self.sample_rate:
                raise ValueError("Proxy must be stereo at the M2D sample rate")
            total_frames = len(audio)
            window_count = max(
                0, math.floor((total_frames - window_frames) / hop_frames) + 1
            )
            for first in range(0, window_count, self.batch_size):
                count = min(self.batch_size, window_count - first)
                block_start = first * hop_frames
                block_frames = (count - 1) * hop_frames + window_frames
                audio.seek(block_start)
                block = audio.read(block_frames, dtype="float32", always_2d=True)
                if len(block) < block_frames:
                    block = np.pad(
                        block, ((0, block_frames - len(block)), (0, 0))
                    )
                mono = np.mean(block, axis=1)
                windows = np.stack(
                    [
                        mono[offset * hop_frames : offset * hop_frames + window_frames]
                        for offset in range(count)
                    ]
                )
                rms = np.sqrt(np.mean(np.square(windows), axis=1))
                dbfs = 20.0 * np.log10(np.maximum(rms, 1e-12))
                active = np.flatnonzero(dbfs >= PROXY_ACTIVITY_DBFS)
                batch_probabilities = np.zeros(
                    (count, len(self.labels)), dtype=np.float32
                )
                if len(active):
                    inference = windows[active]
                    if model_frames != window_frames:
                        inference = np.asarray(
                            [np.resize(window, model_frames) for window in inference],
                            dtype=np.float32,
                        )
                    batch = self.torch.from_numpy(inference).to(self.device)
                    with self.torch.inference_mode():
                        values = self.model(batch).softmax(dim=-1).cpu().numpy()
                    batch_probabilities[active] = values
                    active_windows += len(active)
                all_probabilities.append(batch_probabilities)
        probabilities = (
            np.concatenate(all_probabilities)
            if all_probabilities
            else np.empty((0, len(self.labels)), dtype=np.float32)
        )
        return probabilities, active_windows, time.perf_counter() - started

    def scan(
        self,
        proxy: Path,
        *,
        clip_seconds: float,
        max_regions: int,
    ) -> dict[str, Any]:
        with self._inference_slots:
            probabilities, active_windows, inference_seconds = self._probabilities(
                proxy
            )
        regions = select_candidate_regions(
            probabilities,
            self.labels,
            self.families,
            clip_seconds=clip_seconds,
            max_regions=max_regions,
        )
        return {
            "policy": SCAN_POLICY_VERSION,
            "proxy_sample_rate_hz": self.sample_rate,
            "proxy_activity_threshold_dbfs": PROXY_ACTIVITY_DBFS,
            "m2d_windows": len(probabilities),
            "m2d_inference_concurrency": self.inference_concurrency,
            "active_proxy_windows": active_windows,
            "scan_seconds": round(inference_seconds, 3),
            "regions": regions,
        }


def load_cached_scan(path: Path, *, clip_seconds: float) -> dict[str, Any] | None:
    try:
        result = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if (
        result.get("policy") != SCAN_POLICY_VERSION
        or float(result.get("clip_seconds") or 0.0) != clip_seconds
    ):
        return None
    return result
