"""Validate dialogue-over-background clips with the official M2D tagger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .source_diversity import (
    DEFAULT_MAX_CLIPS_PER_SOURCE,
    record_source_clip_budget,
    source_diversity_policy,
)

logger = logging.getLogger(__name__)

SPEECH_ROOT = "/m/09x0r"
FOREGROUND_SPEECH_MIDS = {
    "/m/05zppz",  # Male speech
    "/m/02zsn",  # Female speech
    "/m/0ytgt",  # Child speech
    "/m/01h8n0",  # Conversation
    "/m/02qldy",  # Narration / monologue
}
SYNTHETIC_SPEECH_MID = "/m/0brhx"
MUSIC_ROOT = "/m/04rlf"
HUMAN_ROOT = "/m/0dgw9r"
BACKGROUND_ROOTS = (
    "/m/0jbk",  # Animal
    "/m/059j3w",  # Natural sounds
    "/t/dd00041",  # Sounds of things
    "/t/dd00098",  # Source-ambiguous sounds
    "/t/dd00123",  # Channel, environment and background
)
VOCAL_MUSIC_ROOTS = (
    "/m/015lz1",  # Singing (includes choir, chant, rapping, and voice types)
    "/m/02fxyj",  # Humming
    "/m/05lls",  # Opera
    "/m/0y4f8",  # Vocal music (includes a capella)
    "/m/074ft",  # Song
)

WINDOW_SECONDS = 2.0
WINDOW_HOP_SECONDS = 1.0
MIN_WINDOW_PROBABILITY = 0.004
MAX_ACTIVE_RANK = 15
MIN_SPEECH_WINDOWS = 3
MIN_STRONG_SPEECH_PROBABILITY = 0.10
MAX_STRONG_SPEECH_RANK = 5
MIN_STRONG_SPEECH_WINDOWS = 5
MIN_FOREGROUND_SPEECH_PROBABILITY = 0.005
MAX_FOREGROUND_SPEECH_RANK = 15
MIN_FOREGROUND_SPEECH_WINDOWS = 4
MIN_SYNTHETIC_SPEECH_PROBABILITY = 0.20
MAX_SYNTHETIC_SPEECH_RANK = 5
MAX_SYNTHETIC_SPEECH_WINDOWS = 1
MIN_ASR_VAD_SECONDS = 1.5
MIN_ASR_WORDS = 2
MIN_ASR_AVG_LOGPROB = -0.80
MAX_ASR_NO_SPEECH_PROBABILITY = 0.50
REQUIRED_ASR_LANGUAGE = "en"
MIN_ASR_LANGUAGE_PROBABILITY = 0.80
MIN_CINEMATIC_MUSIC_PROBABILITY = 0.01
MAX_CINEMATIC_MUSIC_RANK = 15
MIN_CINEMATIC_MUSIC_WINDOWS = 4
MIN_CINEMATIC_SFX_PROBABILITY = 0.02
MAX_CINEMATIC_SFX_RANK = 5
MIN_CINEMATIC_SFX_WINDOWS = 1
MIN_BACKGROUND_WINDOWS = 4
MIN_OVERLAP_WINDOWS = 3
MAX_VOCAL_MUSIC_WINDOWS = 1
TOP_LABELS = 8
POLICY_VERSION = "spoken_dialogue_instrumental_background_m2d_v5"
CINEMATIC_POLICY_VERSION = "cinematic_dialogue_music_sfx_duration_aware_m2d_v4"
ASR_POLICY_VERSION = "foreground_voice_faster_whisper_v3"


def _duration_requirements(window_count: int) -> dict[str, int]:
    """Scale the nine-window calibration to the evaluated clip duration."""
    if window_count < 1:
        raise ValueError("At least one audio window is required")
    baseline_windows = 9

    def minimum(value: int) -> int:
        return max(1, math.ceil(value * window_count / baseline_windows))

    def maximum(value: int) -> int:
        return max(0, math.floor(value * window_count / baseline_windows))

    return {
        "speech": minimum(MIN_SPEECH_WINDOWS),
        "strong_speech": minimum(MIN_STRONG_SPEECH_WINDOWS),
        "foreground_speech": minimum(MIN_FOREGROUND_SPEECH_WINDOWS),
        "cinematic_music": minimum(MIN_CINEMATIC_MUSIC_WINDOWS),
        "cinematic_sfx": minimum(MIN_CINEMATIC_SFX_WINDOWS),
        "background": minimum(MIN_BACKGROUND_WINDOWS),
        "overlap": minimum(MIN_OVERLAP_WINDOWS),
        "synthetic_speech_max": maximum(MAX_SYNTHETIC_SPEECH_WINDOWS),
        "vocal_music_max": maximum(MAX_VOCAL_MUSIC_WINDOWS),
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _belongs_to_shard(filename: str, shard_index: int, shard_count: int) -> bool:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Shard index must be within the positive shard count")
    value = int.from_bytes(hashlib.sha256(filename.encode("utf-8")).digest()[:8], "big")
    return value % shard_count == shard_index


def _descendants(root: str, children: dict[str, list[str]]) -> set[str]:
    found = {root}
    pending = [root]
    while pending:
        for child in children.get(pending.pop(), []):
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def load_label_families(
    class_labels_path: Path, ontology_path: Path
) -> tuple[list[dict[str, str]], dict[str, set[int]]]:
    """Load the class order used by the official M2D tagging example."""
    with class_labels_path.open(newline="", encoding="utf-8") as source:
        labels = sorted(csv.DictReader(source), key=lambda item: item["mid"])
    ontology = json.loads(ontology_path.read_text())
    children = {item["id"]: item.get("child_ids", []) for item in ontology}
    mids = [item["mid"] for item in labels]

    def indices(roots: tuple[str, ...]) -> set[int]:
        family: set[str] = set()
        for root in roots:
            family.update(_descendants(root, children))
        return {index for index, mid in enumerate(mids) if mid in family}

    speech = indices((SPEECH_ROOT,))
    foreground_speech = {
        index for index, mid in enumerate(mids) if mid in FOREGROUND_SPEECH_MIDS
    }
    synthetic_speech = {
        index for index, mid in enumerate(mids) if mid == SYNTHETIC_SPEECH_MID
    }
    music = indices((MUSIC_ROOT,))
    human = indices((HUMAN_ROOT,))
    nonmusic_background = indices(BACKGROUND_ROOTS)
    vocal_music = indices(VOCAL_MUSIC_ROOTS)
    return labels, {
        "speech": speech,
        "foreground_speech": foreground_speech,
        "synthetic_speech": synthetic_speech,
        "music": music,
        "nonmusic_background": nonmusic_background,
        "background": music | nonmusic_background,
        "human": human,
        "vocal_music": vocal_music,
    }


def _family_evidence(probabilities: np.ndarray, family: set[int]) -> tuple[float, int]:
    if not family:
        return 0.0, len(probabilities) + 1
    ordered = np.argsort(probabilities)[::-1]
    ranks = np.empty_like(ordered)
    ranks[ordered] = np.arange(1, len(ordered) + 1)
    family_indices = np.fromiter(family, dtype=np.int64)
    return (
        float(np.max(probabilities[family_indices], initial=0.0)),
        int(np.min(ranks[family_indices], initial=len(probabilities) + 1)),
    )


def evaluate_probabilities(
    probabilities: np.ndarray,
    labels: list[dict[str, str]],
    families: dict[str, set[int]],
    *,
    starts: list[float] | None = None,
    require_cinematic_mix: bool = False,
) -> dict[str, Any]:
    """Evaluate actual temporal overlap from M2D window probabilities."""
    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        raise ValueError("Expected [windows, AudioSet classes] probabilities")
    if starts is None:
        starts = [index * WINDOW_HOP_SECONDS for index in range(len(probabilities))]
    windows: list[dict[str, Any]] = []
    for start, row in zip(starts, probabilities, strict=True):
        evidence = {
            name: _family_evidence(
                row,
                families.get(
                    name,
                    families["speech"] if name == "foreground_speech" else set(),
                ),
            )
            for name in (
                "speech",
                "foreground_speech",
                "synthetic_speech",
                "music",
                "nonmusic_background",
                "background",
                "vocal_music",
            )
        }
        speech_active = (
            evidence["speech"][0] >= MIN_WINDOW_PROBABILITY
            and evidence["speech"][1] <= MAX_ACTIVE_RANK
        )
        strong_speech_active = (
            evidence["speech"][0] >= MIN_STRONG_SPEECH_PROBABILITY
            and evidence["speech"][1] <= MAX_STRONG_SPEECH_RANK
        )
        foreground_speech_active = (
            evidence["foreground_speech"][0] >= MIN_FOREGROUND_SPEECH_PROBABILITY
            and evidence["foreground_speech"][1] <= MAX_FOREGROUND_SPEECH_RANK
        )
        synthetic_speech_active = (
            evidence["synthetic_speech"][0] >= MIN_SYNTHETIC_SPEECH_PROBABILITY
            and evidence["synthetic_speech"][1] <= MAX_SYNTHETIC_SPEECH_RANK
        )
        cinematic_music_active = (
            evidence["music"][0] >= MIN_CINEMATIC_MUSIC_PROBABILITY
            and evidence["music"][1] <= MAX_CINEMATIC_MUSIC_RANK
        )
        cinematic_sfx_active = (
            evidence["nonmusic_background"][0] >= MIN_CINEMATIC_SFX_PROBABILITY
            and evidence["nonmusic_background"][1] <= MAX_CINEMATIC_SFX_RANK
        )
        background_active = (
            evidence["background"][0] >= MIN_WINDOW_PROBABILITY
            and evidence["background"][1] <= MAX_ACTIVE_RANK
        )
        vocal_music_active = (
            evidence["vocal_music"][0] >= MIN_WINDOW_PROBABILITY
            and evidence["vocal_music"][1] <= MAX_ACTIVE_RANK
        )
        top = np.argsort(row)[::-1][:TOP_LABELS]
        windows.append(
            {
                "start_seconds": round(float(start), 3),
                "end_seconds": round(float(start + WINDOW_SECONDS), 3),
                "speech_score": round(evidence["speech"][0], 8),
                "speech_rank": evidence["speech"][1],
                "music_score": round(evidence["music"][0], 8),
                "music_rank": evidence["music"][1],
                "nonmusic_background_score": round(
                    evidence["nonmusic_background"][0], 8
                ),
                "nonmusic_background_rank": evidence["nonmusic_background"][1],
                "background_score": round(evidence["background"][0], 8),
                "background_rank": evidence["background"][1],
                "vocal_music_score": round(evidence["vocal_music"][0], 8),
                "vocal_music_rank": evidence["vocal_music"][1],
                "speech_active": speech_active,
                "strong_speech_active": strong_speech_active,
                "foreground_speech_score": round(evidence["foreground_speech"][0], 8),
                "foreground_speech_rank": evidence["foreground_speech"][1],
                "foreground_speech_active": foreground_speech_active,
                "synthetic_speech_score": round(evidence["synthetic_speech"][0], 8),
                "synthetic_speech_rank": evidence["synthetic_speech"][1],
                "synthetic_speech_active": synthetic_speech_active,
                "cinematic_music_active": cinematic_music_active,
                "cinematic_sfx_active": cinematic_sfx_active,
                "background_active": background_active,
                "vocal_music_active": vocal_music_active,
                "overlap_active": speech_active and background_active,
                "top_labels": [
                    {
                        "mid": labels[index]["mid"],
                        "name": labels[index]["display_name"],
                        "probability": round(float(row[index]), 8),
                    }
                    for index in top
                ],
            }
        )

    speech_windows = sum(item["speech_active"] for item in windows)
    strong_speech_windows = sum(item["strong_speech_active"] for item in windows)
    foreground_speech_windows = sum(
        item["foreground_speech_active"] for item in windows
    )
    synthetic_speech_windows = sum(item["synthetic_speech_active"] for item in windows)
    cinematic_music_windows = sum(item["cinematic_music_active"] for item in windows)
    cinematic_sfx_windows = sum(item["cinematic_sfx_active"] for item in windows)
    background_windows = sum(item["background_active"] for item in windows)
    overlap_windows = sum(item["overlap_active"] for item in windows)
    vocal_music_windows = sum(item["vocal_music_active"] for item in windows)
    overlap_music = sum(
        item["music_score"] for item in windows if item["overlap_active"]
    )
    overlap_nonmusic = sum(
        item["nonmusic_background_score"] for item in windows if item["overlap_active"]
    )
    if overlap_music > overlap_nonmusic * 1.25:
        background_bucket = "music_led"
    elif overlap_nonmusic > overlap_music * 1.25:
        background_bucket = "effects_ambience_led"
    else:
        background_bucket = "mixed_music_and_effects"

    window_count = len(windows)
    # Preserve the policy calibrated on nine windows while scaling its required
    # temporal coverage to clips of any duration (notably the new 30 s clips).
    required = _duration_requirements(window_count)
    rejections: list[str] = []
    if speech_windows < required["speech"]:
        rejections.append("insufficient_speech")
    if strong_speech_windows < required["strong_speech"]:
        rejections.append("insufficient_strong_speech")
    if foreground_speech_windows < required["foreground_speech"]:
        rejections.append("insufficient_foreground_speech")
    if synthetic_speech_windows > required["synthetic_speech_max"]:
        rejections.append("synthetic_speech_present")
    if require_cinematic_mix:
        if cinematic_music_windows < required["cinematic_music"]:
            rejections.append("insufficient_cinematic_music")
        if cinematic_sfx_windows < required["cinematic_sfx"]:
            rejections.append("insufficient_cinematic_sfx")
    if background_windows < required["background"]:
        rejections.append("insufficient_background")
    if overlap_windows < required["overlap"]:
        rejections.append("insufficient_dialogue_background_overlap")
    if vocal_music_windows > required["vocal_music_max"]:
        rejections.append("vocal_music_present")
    return {
        "accepted": not rejections,
        "policy": (
            CINEMATIC_POLICY_VERSION if require_cinematic_mix else POLICY_VERSION
        ),
        "rejection_reasons": rejections,
        "background_bucket": background_bucket,
        "window_count": window_count,
        "duration_scaled_window_requirements": required,
        "speech_active_windows": speech_windows,
        "strong_speech_active_windows": strong_speech_windows,
        "foreground_speech_active_windows": foreground_speech_windows,
        "synthetic_speech_active_windows": synthetic_speech_windows,
        "cinematic_music_active_windows": cinematic_music_windows,
        "cinematic_sfx_active_windows": cinematic_sfx_windows,
        "background_active_windows": background_windows,
        "overlap_active_windows": overlap_windows,
        "vocal_music_active_windows": vocal_music_windows,
        "speech_coverage": round(speech_windows / window_count, 6),
        "strong_speech_coverage": round(strong_speech_windows / window_count, 6),
        "foreground_speech_coverage": round(
            foreground_speech_windows / window_count, 6
        ),
        "synthetic_speech_coverage": round(synthetic_speech_windows / window_count, 6),
        "cinematic_music_coverage": round(cinematic_music_windows / window_count, 6),
        "cinematic_sfx_coverage": round(cinematic_sfx_windows / window_count, 6),
        "cinematic_mix_required": require_cinematic_mix,
        "cinematic_mix_pass": (
            cinematic_music_windows >= required["cinematic_music"]
            and cinematic_sfx_windows >= required["cinematic_sfx"]
        ),
        "background_coverage": round(background_windows / window_count, 6),
        "overlap_coverage": round(overlap_windows / window_count, 6),
        "vocal_music_coverage": round(vocal_music_windows / window_count, 6),
        "windows": windows,
    }


def _audio_windows(
    waveform: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, list[float]]:
    duration = len(waveform) / sample_rate
    if duration < WINDOW_SECONDS:
        wanted = int(round(WINDOW_SECONDS * sample_rate))
        waveform = np.pad(waveform, (0, wanted - len(waveform)))
        duration = WINDOW_SECONDS
    length = int(round(WINDOW_SECONDS * sample_rate))
    starts = list(np.arange(0.0, duration - WINDOW_SECONDS + 1e-6, WINDOW_HOP_SECONDS))
    windows = [
        waveform[round(start * sample_rate) : round(start * sample_rate) + length]
        for start in starts
    ]
    minimum = 3 * sample_rate
    windows = [
        np.resize(window, minimum) if len(window) < minimum else window
        for window in windows
    ]
    return np.asarray(windows, dtype=np.float32), [float(value) for value in starts]


def score_directory(args: argparse.Namespace) -> None:
    try:
        import librosa
        import torch
    except ImportError as error:
        raise RuntimeError(
            "M2D scoring requires torch, librosa, timm, einops, and nnAudio"
        ) from error

    sys.path.insert(0, str(args.m2d_repo / "examples"))
    from portable_m2d import PortableM2D  # type: ignore[import-not-found]

    labels, families = load_label_families(args.class_labels, args.ontology)
    model = PortableM2D(weight_file=str(args.checkpoint), num_classes=len(labels))
    model = model.to(args.device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if args.output.exists() and not args.overwrite:
        for line in args.output.read_text().splitlines():
            if line.strip():
                existing.add(str(json.loads(line)["filename"]))
    mode = "w" if args.overwrite else "a"
    follow = bool(getattr(args, "follow", False))
    producer_done = getattr(args, "producer_done", None)
    poll_seconds = max(0.1, float(getattr(args, "poll_seconds", 2.0)))
    shard_index = int(getattr(args, "shard_index", 0))
    shard_count = int(getattr(args, "shard_count", 1))
    metadata = {
        "model": "nttcslab/m2d",
        "checkpoint": args.checkpoint.parent.name,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "m2d_repository_commit": args.m2d_commit,
        "policy": (
            CINEMATIC_POLICY_VERSION if args.require_cinematic_mix else POLICY_VERSION
        ),
        "window_seconds": WINDOW_SECONDS,
        "window_hop_seconds": WINDOW_HOP_SECONDS,
        "minimum_window_probability": MIN_WINDOW_PROBABILITY,
        "maximum_active_rank": MAX_ACTIVE_RANK,
        "minimum_speech_windows": MIN_SPEECH_WINDOWS,
        "minimum_strong_speech_probability": MIN_STRONG_SPEECH_PROBABILITY,
        "maximum_strong_speech_rank": MAX_STRONG_SPEECH_RANK,
        "minimum_strong_speech_windows": MIN_STRONG_SPEECH_WINDOWS,
        "minimum_foreground_speech_probability": (MIN_FOREGROUND_SPEECH_PROBABILITY),
        "maximum_foreground_speech_rank": MAX_FOREGROUND_SPEECH_RANK,
        "minimum_foreground_speech_windows": MIN_FOREGROUND_SPEECH_WINDOWS,
        "minimum_synthetic_speech_probability": (MIN_SYNTHETIC_SPEECH_PROBABILITY),
        "maximum_synthetic_speech_rank": MAX_SYNTHETIC_SPEECH_RANK,
        "maximum_synthetic_speech_windows": MAX_SYNTHETIC_SPEECH_WINDOWS,
        "minimum_cinematic_music_probability": MIN_CINEMATIC_MUSIC_PROBABILITY,
        "maximum_cinematic_music_rank": MAX_CINEMATIC_MUSIC_RANK,
        "minimum_cinematic_music_windows": MIN_CINEMATIC_MUSIC_WINDOWS,
        "minimum_cinematic_sfx_probability": MIN_CINEMATIC_SFX_PROBABILITY,
        "maximum_cinematic_sfx_rank": MAX_CINEMATIC_SFX_RANK,
        "minimum_cinematic_sfx_windows": MIN_CINEMATIC_SFX_WINDOWS,
        "minimum_background_windows": MIN_BACKGROUND_WINDOWS,
        "minimum_overlap_windows": MIN_OVERLAP_WINDOWS,
        "maximum_vocal_music_windows": MAX_VOCAL_MUSIC_WINDOWS,
        "worker_shard_index": shard_index,
        "worker_shard_count": shard_count,
    }
    processed = 0
    with args.output.open(mode, encoding="utf-8") as destination:
        while True:
            files = sorted(args.input_dir.glob(args.glob))
            files = [
                path
                for path in files
                if _belongs_to_shard(path.name, shard_index, shard_count)
            ]
            if args.limit:
                files = files[: args.limit]
            pending = [path for path in files if path.name not in existing]
            if not pending:
                if not follow or (producer_done and producer_done.is_file()):
                    break
                time.sleep(poll_seconds)
                continue
            for path in pending:
                waveform, _ = librosa.load(path, mono=True, sr=model.cfg.sample_rate)
                windows, starts = _audio_windows(waveform, model.cfg.sample_rate)
                batch = torch.from_numpy(windows).to(args.device)
                with torch.inference_mode():
                    probabilities = model(batch).softmax(dim=-1).cpu().numpy()
                result = {
                    "filename": path.name,
                    "scored_at": _now(),
                    "m2d": metadata,
                    **evaluate_probabilities(
                        probabilities,
                        labels,
                        families,
                        starts=starts,
                        require_cinematic_mix=args.require_cinematic_mix,
                    ),
                }
                destination.write(json.dumps(result, separators=(",", ":")) + "\n")
                destination.flush()
                existing.add(path.name)
                processed += 1
                if processed % 25 == 0:
                    completed = sum(path.name in existing for path in files)
                    logger.info(
                        "M2D scored %d/%d currently discovered clips",
                        completed,
                        len(files),
                    )
            if not follow:
                break
    logger.info("Wrote %d new validation records to %s", processed, args.output)


def evaluate_asr(
    *,
    transcript: str,
    duration_after_vad: float,
    average_log_probability: float,
    no_speech_probability: float,
    detected_language: str = REQUIRED_ASR_LANGUAGE,
    language_probability: float = 1.0,
) -> dict[str, Any]:
    """Require decodable foreground speech rather than generic chatter labels."""
    word_count = len(re.findall(r"[A-Za-z]+", transcript))
    reasons: list[str] = []
    if duration_after_vad < MIN_ASR_VAD_SECONDS:
        reasons.append("insufficient_voice_activity")
    if word_count < MIN_ASR_WORDS:
        reasons.append("insufficient_decoded_words")
    if average_log_probability < MIN_ASR_AVG_LOGPROB:
        reasons.append("low_transcription_confidence")
    if no_speech_probability > MAX_ASR_NO_SPEECH_PROBABILITY:
        reasons.append("high_no_speech_probability")
    if detected_language != REQUIRED_ASR_LANGUAGE:
        reasons.append("non_english_speech")
    if language_probability < MIN_ASR_LANGUAGE_PROBABILITY:
        reasons.append("low_language_confidence")
    return {
        "accepted": not reasons,
        "policy": ASR_POLICY_VERSION,
        "rejection_reasons": reasons,
        "transcript": transcript.strip(),
        "word_count": word_count,
        "duration_after_vad_seconds": round(duration_after_vad, 6),
        "best_average_log_probability": round(average_log_probability, 8),
        "lowest_no_speech_probability": round(no_speech_probability, 8),
        "detected_language": detected_language,
        "language_probability": round(language_probability, 8),
    }


def _m2d_asr_allowlist(
    path: Path, *, require_cinematic_mix: bool
) -> tuple[set[str], int]:
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            latest[str(item["filename"])] = item
    allowed = {
        filename
        for filename, item in latest.items()
        if _enforce_current_voice_gate(
            item, require_cinematic_mix=require_cinematic_mix
        ).get("accepted")
    }
    return allowed, len(latest)


class _M2DAllowlistTail:
    """Incrementally follow M2D JSONL output without repeatedly rereading it."""

    def __init__(self, path: Path, *, require_cinematic_mix: bool):
        self.path = path
        self.require_cinematic_mix = require_cinematic_mix
        self.identity: tuple[int, int] | None = None
        self.offset = 0
        self.latest: dict[str, bool] = {}

    def refresh(self) -> tuple[set[str], int]:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return set(), 0
        identity = (stat.st_dev, stat.st_ino)
        if self.identity != identity or stat.st_size < self.offset:
            self.identity = identity
            self.offset = 0
            self.latest = {}
        with self.path.open("rb") as source:
            source.seek(self.offset)
            while True:
                line_start = source.tell()
                line = source.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    source.seek(line_start)
                    break
                try:
                    item = json.loads(line)
                    filename = str(item["filename"])
                except (json.JSONDecodeError, KeyError, TypeError, UnicodeError):
                    continue
                self.latest[filename] = bool(
                    _enforce_current_voice_gate(
                        item,
                        require_cinematic_mix=self.require_cinematic_mix,
                    ).get("accepted")
                )
            self.offset = source.tell()
        return (
            {filename for filename, accepted in self.latest.items() if accepted},
            len(self.latest),
        )


def _runtime_asr_concurrency(path: Path | None, maximum: int) -> int:
    if not path:
        return max(1, maximum)
    try:
        payload = json.loads(path.read_text())
        value = payload.get("asr_concurrency")
        if value is None:
            value = payload.get("limits", {}).get("asr_concurrency")
        return max(1, min(maximum, int(value)))
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        return 1


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _pending_asr_probe_requests(
    request_dir: Path | None,
    result_dir: Path | None,
) -> list[tuple[Path, Path, Path, dict[str, Any]]]:
    """Return valid proxy-ASR requests, resolving malformed ones as failures."""
    if request_dir is None or result_dir is None:
        return []
    request_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[Path, Path, Path, dict[str, Any]]] = []
    for request_path in sorted(request_dir.glob("*.json")):
        try:
            request = json.loads(request_path.read_text())
            request_id = str(request["request_id"])
            if request_id != request_path.stem:
                raise ValueError("request id does not match filename")
            audio_path = Path(str(request["audio_path"]))
            result_path = result_dir / f"{request_id}.json"
            if result_path.exists():
                request_path.unlink(missing_ok=True)
                continue
            if not audio_path.is_file():
                _write_json_atomic(
                    result_path,
                    {
                        "request_id": request_id,
                        "accepted": False,
                        "error": "probe_audio_missing",
                        "rejection_reasons": ["probe_audio_missing"],
                    },
                )
                request_path.unlink(missing_ok=True)
                continue
            pending.append((request_path, result_path, audio_path, request))
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
            request_path.unlink(missing_ok=True)
    return pending


def _transcribe_asr_file(
    model: Any,
    path: Path,
    args: argparse.Namespace,
    *,
    shard_index: int,
    shard_count: int,
    beam_size: int | None = None,
) -> dict[str, Any]:
    effective_beam_size = args.beam_size if beam_size is None else beam_size
    segments_source, info = model.transcribe(
        str(path),
        language=None,
        beam_size=effective_beam_size,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    segments = list(segments_source)
    transcript = " ".join(segment.text for segment in segments).strip()
    best_log_probability = max(
        (float(segment.avg_logprob) for segment in segments), default=-99.0
    )
    lowest_no_speech = min(
        (float(segment.no_speech_prob) for segment in segments), default=1.0
    )
    return {
        "filename": path.name,
        "scored_at": _now(),
        "asr": {
            # Keep a stable identity in metadata when the runtime loads an
            # offline snapshot by absolute path.
            "model": getattr(args, "model_label", None) or args.model,
            "device": args.device,
            "compute_type": args.compute_type,
            "beam_size": effective_beam_size,
            "worker_shard_index": shard_index,
            "worker_shard_count": shard_count,
        },
        **evaluate_asr(
            transcript=transcript,
            duration_after_vad=float(info.duration_after_vad),
            average_log_probability=best_log_probability,
            no_speech_probability=lowest_no_speech,
            detected_language=str(info.language),
            language_probability=float(info.language_probability),
        ),
        "segments": [
            {
                "start_seconds": round(float(segment.start), 3),
                "end_seconds": round(float(segment.end), 3),
                "text": segment.text.strip(),
                "average_log_probability": round(float(segment.avg_logprob), 8),
                "no_speech_probability": round(float(segment.no_speech_prob), 8),
            }
            for segment in segments
        ],
    }


def score_asr_directory(args: argparse.Namespace) -> None:
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError(
            "ASR scoring requires faster-whisper in the runtime environment"
        ) from error

    max_inference_workers = max(1, int(getattr(args, "max_inference_workers", 1)))
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=max(0, int(getattr(args, "cpu_threads", 0))),
        num_workers=max_inference_workers,
        download_root=(str(args.download_root) if args.download_root else None),
    )
    m2d_results = getattr(args, "m2d_results", None)
    m2d_results_dir = getattr(args, "m2d_results_dir", None)
    m2d_paths = ([m2d_results] if m2d_results else []) + (
        sorted(m2d_results_dir.glob("*.jsonl")) if m2d_results_dir else []
    )
    allowlist_tails = [
        _M2DAllowlistTail(
            path,
            require_cinematic_mix=bool(getattr(args, "require_cinematic_mix", False)),
        )
        for path in m2d_paths
    ]
    model_identity = getattr(args, "model_label", None) or args.model
    # Accept both the stable label and load path. This also makes a cutover from
    # path-based metadata idempotent after --model-label is introduced.
    model_identities = {str(args.model), str(model_identity)}
    existing: set[str] = set()
    if args.output.exists() and not args.overwrite:
        for line in args.output.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if (
                item.get("policy") == ASR_POLICY_VERSION
                and item.get("detected_language") is not None
                and str(item.get("asr", {}).get("model")) in model_identities
            ):
                existing.add(str(item["filename"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    follow = bool(getattr(args, "follow", False))
    producer_done = getattr(args, "producer_done", None)
    poll_seconds = max(0.1, float(getattr(args, "poll_seconds", 2.0)))
    shard_index = int(getattr(args, "shard_index", 0))
    shard_count = int(getattr(args, "shard_count", 1))
    processed = 0
    probe_processed = 0
    previous_allowed_count = -1
    control_file = getattr(args, "autoscale_control", None)
    with (
        args.output.open(mode, encoding="utf-8") as destination,
        ThreadPoolExecutor(max_workers=max_inference_workers) as executor,
    ):
        while True:
            files = sorted(args.input_dir.glob(args.glob))
            files = [
                path
                for path in files
                if _belongs_to_shard(path.name, shard_index, shard_count)
            ]
            if allowlist_tails:
                allowed: set[str] = set()
                scored_count = 0
                for tail in allowlist_tails:
                    tail_allowed, tail_scored = tail.refresh()
                    allowed.update(tail_allowed)
                    scored_count += tail_scored
                files = [path for path in files if path.name in allowed]
                if len(allowed) != previous_allowed_count:
                    logger.info(
                        "ASR queue has %d M2D passes from %d scored filenames",
                        len(files),
                        scored_count,
                    )
                    previous_allowed_count = len(allowed)
            if args.limit:
                files = files[: args.limit]
            pending = [path for path in files if path.name not in existing]
            probe_requests = _pending_asr_probe_requests(
                getattr(args, "probe_requests_dir", None),
                getattr(args, "probe_results_dir", None),
            )
            if not pending and not probe_requests:
                if not follow or (producer_done and producer_done.is_file()):
                    break
                time.sleep(poll_seconds)
                continue
            concurrency = _runtime_asr_concurrency(control_file, max_inference_workers)
            # A probe unblocks up to eight future extracts. Give it one slot while
            # retaining one final-validation slot, and use both slots for probes
            # only when the normal ASR queue is empty.
            if probe_requests and max_inference_workers >= 2:
                concurrency = max(concurrency, 2)
            jobs: list[tuple[str, Any]] = []
            if probe_requests:
                probe_slots = concurrency if not pending else 1
                jobs.extend(("probe", item) for item in probe_requests[:probe_slots])
            remaining_slots = concurrency - len(jobs)
            jobs.extend(("final", path) for path in pending[:remaining_slots])
            futures: dict[Any, tuple[str, Any]] = {}
            for kind, item in jobs:
                path = item[2] if kind == "probe" else item
                futures[
                    executor.submit(
                        _transcribe_asr_file,
                        model,
                        path,
                        args,
                        shard_index=shard_index,
                        shard_count=shard_count,
                        beam_size=(1 if kind == "probe" else None),
                    )
                ] = (kind, item)
            for future in as_completed(futures):
                kind, item = futures[future]
                if kind == "probe":
                    request_path, result_path, _, request = item
                    try:
                        result = {
                            **future.result(),
                            "request_id": request["request_id"],
                            "probe": {
                                "video_id": request.get("video_id"),
                                "start_seconds": request.get("start_seconds"),
                                "policy": request.get("policy"),
                            },
                        }
                    except Exception as error:
                        result = {
                            "request_id": request.get("request_id"),
                            "accepted": False,
                            "error": f"{type(error).__name__}: {error}",
                            "rejection_reasons": ["probe_inference_failed"],
                        }
                    _write_json_atomic(result_path, result)
                    request_path.unlink(missing_ok=True)
                    probe_processed += 1
                    if probe_processed % 25 == 0:
                        logger.info(
                            "Completed %d proxy-ASR source probes", probe_processed
                        )
                    continue
                result = future.result()
                destination.write(json.dumps(result, separators=(",", ":")) + "\n")
                destination.flush()
                existing.add(str(result["filename"]))
                processed += 1
                if processed % 25 == 0:
                    completed = sum(path.name in existing for path in files)
                    logger.info(
                        "ASR scored %d/%d currently eligible clips",
                        completed,
                        len(files),
                    )
            if not follow:
                break
    logger.info("Wrote %d new ASR records to %s", processed, args.output)


def _enforce_current_voice_gate(
    result: dict[str, Any], *, require_cinematic_mix: bool = False
) -> dict[str, Any]:
    """Apply current voice and optional cinematic gates to M2D results."""
    result = dict(result)
    require_cinematic_mix = bool(
        require_cinematic_mix or result.get("cinematic_mix_required")
    )
    windows = []
    for source in result.get("windows", []):
        window = dict(source)
        window["strong_speech_active"] = (
            float(window.get("speech_score", 0.0)) >= MIN_STRONG_SPEECH_PROBABILITY
            and int(window.get("speech_rank", 10_000)) <= MAX_STRONG_SPEECH_RANK
        )
        foreground_score = float(window.get("foreground_speech_score", 0.0))
        foreground_rank = int(window.get("foreground_speech_rank", 10_000))
        if "foreground_speech_score" not in window:
            foreground_labels = [
                item
                for item in window.get("top_labels", [])
                if item.get("mid") in FOREGROUND_SPEECH_MIDS
            ]
            if foreground_labels:
                strongest = max(
                    foreground_labels,
                    key=lambda item: float(item.get("probability", 0.0)),
                )
                foreground_score = float(strongest.get("probability", 0.0))
                foreground_rank = next(
                    index
                    for index, item in enumerate(window.get("top_labels", []), 1)
                    if item is strongest
                )
        window["foreground_speech_score"] = foreground_score
        window["foreground_speech_rank"] = foreground_rank
        window["foreground_speech_active"] = (
            foreground_score >= MIN_FOREGROUND_SPEECH_PROBABILITY
            and foreground_rank <= MAX_FOREGROUND_SPEECH_RANK
        )
        synthetic_score = float(window.get("synthetic_speech_score", 0.0))
        synthetic_rank = int(window.get("synthetic_speech_rank", 10_000))
        if "synthetic_speech_score" not in window:
            for index, item in enumerate(window.get("top_labels", []), 1):
                if item.get("mid") == SYNTHETIC_SPEECH_MID:
                    synthetic_score = float(item.get("probability", 0.0))
                    synthetic_rank = index
                    break
        window["synthetic_speech_score"] = synthetic_score
        window["synthetic_speech_rank"] = synthetic_rank
        window["synthetic_speech_active"] = (
            synthetic_score >= MIN_SYNTHETIC_SPEECH_PROBABILITY
            and synthetic_rank <= MAX_SYNTHETIC_SPEECH_RANK
        )
        window["cinematic_music_active"] = (
            float(window.get("music_score", 0.0)) >= MIN_CINEMATIC_MUSIC_PROBABILITY
            and int(window.get("music_rank", 10_000)) <= MAX_CINEMATIC_MUSIC_RANK
        )
        window["cinematic_sfx_active"] = (
            float(window.get("nonmusic_background_score", 0.0))
            >= MIN_CINEMATIC_SFX_PROBABILITY
            and int(window.get("nonmusic_background_rank", 10_000))
            <= MAX_CINEMATIC_SFX_RANK
        )
        windows.append(window)
    strong_speech_windows = sum(
        bool(window["strong_speech_active"]) for window in windows
    )
    foreground_speech_windows = sum(
        bool(window["foreground_speech_active"]) for window in windows
    )
    synthetic_speech_windows = sum(
        bool(window["synthetic_speech_active"]) for window in windows
    )
    cinematic_music_windows = sum(
        bool(window["cinematic_music_active"]) for window in windows
    )
    cinematic_sfx_windows = sum(
        bool(window["cinematic_sfx_active"]) for window in windows
    )
    # Legacy rejected rows may not carry window evidence; keep them rejected
    # while allowing append-only consumers to continue past the row.
    required = _duration_requirements(len(windows) or 9)
    cinematic_mix_present = (
        cinematic_music_windows >= required["cinematic_music"]
        and cinematic_sfx_windows >= required["cinematic_sfx"]
    )
    reasons = list(dict.fromkeys(result.get("rejection_reasons", [])))
    strong_voice_present = strong_speech_windows >= required["strong_speech"]
    foreground_voice_present = (
        foreground_speech_windows >= required["foreground_speech"]
    )
    if not strong_voice_present and "insufficient_strong_speech" not in reasons:
        reasons.append("insufficient_strong_speech")
    if not foreground_voice_present and "insufficient_foreground_speech" not in reasons:
        reasons.append("insufficient_foreground_speech")
    if (
        synthetic_speech_windows > required["synthetic_speech_max"]
        and "synthetic_speech_present" not in reasons
    ):
        reasons.append("synthetic_speech_present")
    if require_cinematic_mix:
        if (
            cinematic_music_windows < required["cinematic_music"]
            and "insufficient_cinematic_music" not in reasons
        ):
            reasons.append("insufficient_cinematic_music")
        if (
            cinematic_sfx_windows < required["cinematic_sfx"]
            and "insufficient_cinematic_sfx" not in reasons
        ):
            reasons.append("insufficient_cinematic_sfx")
    previous_policy = result.get("policy")
    policy = CINEMATIC_POLICY_VERSION if require_cinematic_mix else POLICY_VERSION
    result.update(
        {
            "accepted": (
                bool(result.get("accepted"))
                and strong_voice_present
                and foreground_voice_present
                and synthetic_speech_windows <= required["synthetic_speech_max"]
                and (not require_cinematic_mix or cinematic_mix_present)
            ),
            "policy": policy,
            "rejection_reasons": reasons,
            "duration_scaled_window_requirements": required,
            "strong_speech_active_windows": strong_speech_windows,
            "strong_speech_coverage": round(
                strong_speech_windows / max(1, len(windows)), 6
            ),
            "foreground_speech_active_windows": foreground_speech_windows,
            "foreground_speech_coverage": round(
                foreground_speech_windows / max(1, len(windows)), 6
            ),
            "synthetic_speech_active_windows": synthetic_speech_windows,
            "synthetic_speech_coverage": round(
                synthetic_speech_windows / max(1, len(windows)), 6
            ),
            "cinematic_music_active_windows": cinematic_music_windows,
            "cinematic_sfx_active_windows": cinematic_sfx_windows,
            "cinematic_music_coverage": round(
                cinematic_music_windows / max(1, len(windows)), 6
            ),
            "cinematic_sfx_coverage": round(
                cinematic_sfx_windows / max(1, len(windows)), 6
            ),
            "cinematic_mix_required": require_cinematic_mix,
            "cinematic_mix_pass": cinematic_mix_present,
            "windows": windows,
        }
    )
    if previous_policy and previous_policy != policy:
        result["policy_migrated_from"] = previous_policy
    return result


def _materialize_audio(
    input_dir: Path, output_dir: Path, selected: list[dict[str, Any]]
) -> None:
    output_dir.mkdir(exist_ok=True)
    selected_names = {item["filename"] for item in selected}
    for stale in output_dir.glob("*.wav"):
        if stale.name not in selected_names:
            stale.unlink()
    for result in selected:
        source = input_dir / result["filename"]
        destination = output_dir / result["filename"]
        if destination.exists():
            continue
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def materialize_accepted(args: argparse.Namespace) -> None:
    require_cinematic_mix = bool(getattr(args, "require_cinematic_mix", False))
    results = [
        _enforce_current_voice_gate(
            json.loads(line),
            require_cinematic_mix=require_cinematic_mix,
        )
        for line in args.results.read_text().splitlines()
        if line.strip()
    ]
    asr_results_path = getattr(args, "asr_results", None)
    asr_by_filename: dict[str, dict[str, Any]] = {}
    if asr_results_path:
        asr_by_filename = {
            item["filename"]: item
            for item in (
                json.loads(line)
                for line in asr_results_path.read_text().splitlines()
                if line.strip()
            )
        }
    accepted = [
        item
        for item in results
        if item.get("accepted")
        and (
            not asr_results_path
            or asr_by_filename.get(item["filename"], {}).get("accepted")
        )
    ]
    accepted_limit = getattr(args, "accepted_limit", None)
    if accepted_limit:
        accepted = accepted[:accepted_limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    _materialize_audio(args.input_dir, audio_dir, accepted)

    music_led = [item for item in accepted if item["background_bucket"] == "music_led"]
    nonmusic_led = [
        item for item in accepted if item["background_bucket"] != "music_led"
    ]
    balance_size = min(len(music_led), len(nonmusic_led))
    balanced = music_led[:balance_size] + nonmusic_led[:balance_size]
    balanced.sort(key=lambda item: item["filename"])
    balanced_dir = args.output_dir / "balanced-audio"
    _materialize_audio(args.input_dir, balanced_dir, balanced)

    source_manifest = json.loads(args.source_manifest.read_text())
    source_by_name = {
        Path(record["local_path"]).name: record
        for record in source_manifest.get("records", [])
    }
    records: list[dict[str, Any]] = []
    for index, result in enumerate(accepted):
        original = dict(source_by_name.get(result["filename"], {}))
        original.update(
            {
                "record_index": index,
                "local_path": f"audio/{result['filename']}",
                "m2d_validation": result,
                "asr_validation": asr_by_filename.get(result["filename"]),
            }
        )
        records.append(original)
    rejection_counts = Counter(
        reason for result in results for reason in result.get("rejection_reasons", [])
    )
    asr_rejection_counts = Counter(
        reason
        for result in asr_by_filename.values()
        for reason in result.get("rejection_reasons", [])
    )
    bucket_counts = Counter(item["background_bucket"] for item in accepted)
    manifest = {
        "schema_version": 1,
        "name": "M2D-validated dialogue over active background",
        "created_at": _now(),
        "source_manifest": str(args.source_manifest),
        "source_record_count": len(results),
        "accepted_record_count": len(records),
        "acceptance_rate": round(len(records) / max(1, len(results)), 6),
        "validator": "nttcslab/m2d AudioSet fine-tuned tagger",
        "foreground_voice_validator": ("faster-whisper" if asr_results_path else None),
        "foreground_voice_policy": (ASR_POLICY_VERSION if asr_results_path else None),
        "policy": (
            CINEMATIC_POLICY_VERSION if require_cinematic_mix else POLICY_VERSION
        ),
        "background_bucket_counts": dict(sorted(bucket_counts.items())),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "foreground_voice_rejection_reason_counts": dict(
            sorted(asr_rejection_counts.items())
        ),
        "balanced_listening_subset": {
            "policy": "equal_music_led_and_nonmusic_led_v1",
            "record_count": len(balanced),
            "music_led_count": balance_size,
            "nonmusic_led_count": balance_size,
            "local_directory": "balanced-audio",
            "filenames": [item["filename"] for item in balanced],
        },
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    audit = {
        key: manifest[key]
        for key in (
            "created_at",
            "source_record_count",
            "accepted_record_count",
            "acceptance_rate",
            "validator",
            "policy",
            "background_bucket_counts",
            "rejection_reason_counts",
            "foreground_voice_rejection_reason_counts",
        )
    }
    audit["original_dataset_preserved"] = True
    audit["materialized_audio_files"] = len(list(audio_dir.glob("*.wav")))
    audit["balanced_audio_files"] = len(list(balanced_dir.glob("*.wav")))
    audit["balanced_listening_subset"] = manifest["balanced_listening_subset"]
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    normalized_results = "".join(
        json.dumps(result, separators=(",", ":")) + "\n" for result in results
    )
    (args.output_dir / "m2d-validation.jsonl").write_text(normalized_results)


def merge_materialized(args: argparse.Namespace) -> None:
    """Build one exact, deduplicated final set from validated acquisition batches."""
    from .youtube_random import _candidate_allowed

    candidates: list[dict[str, Any]] = []
    batch_summaries: list[dict[str, Any]] = []
    for batch_index, batch_dir in enumerate(args.batch):
        manifest = json.loads((batch_dir / "manifest.json").read_text())
        m2d_lines = (batch_dir / "m2d-validation.jsonl").read_text().splitlines()
        m2d_by_name = {
            item["filename"]: _enforce_current_voice_gate(
                item,
                require_cinematic_mix=args.require_cinematic_mix,
            )
            for item in (json.loads(line) for line in m2d_lines if line.strip())
        }
        asr_lines = (batch_dir / "asr-validation.jsonl").read_text().splitlines()
        asr_by_name = {
            item["filename"]: item
            for item in (json.loads(line) for line in asr_lines if line.strip())
        }
        combined_count = 0
        for source_record in manifest.get("records", []):
            filename = Path(source_record["local_path"]).name
            m2d = m2d_by_name.get(filename)
            asr = asr_by_name.get(filename)
            if not m2d or not asr or not m2d.get("accepted") or not asr.get("accepted"):
                continue
            if not _candidate_allowed(source_record, profile="cinematic"):
                continue
            source_path = batch_dir / str(source_record["local_path"])
            if not source_path.exists():
                continue
            combined_count += 1
            candidates.append(
                {
                    "batch_index": batch_index,
                    "batch_dir": batch_dir,
                    "source_path": source_path,
                    "filename": filename,
                    "source_record": source_record,
                    "m2d_validation": m2d,
                    "asr_validation": asr,
                }
            )
        batch_summaries.append(
            {
                "batch_index": batch_index,
                "path": str(batch_dir),
                "source_record_count": len(manifest.get("records", [])),
                "combined_pass_count": combined_count,
            }
        )

    random.Random(args.seed).shuffle(candidates)
    selected: list[dict[str, Any]] = []
    hashes: set[str] = set()
    candidate_ids: set[str] = set()
    starts_by_video: dict[tuple[str, str], list[float]] = {}
    video_counts: Counter[tuple[str, str]] = Counter()
    video_budgets: dict[tuple[str, str], int] = {}
    skip_counts: Counter[str] = Counter()
    for item in candidates:
        source_record = item["source_record"]
        platform = str(source_record.get("source_platform") or "unknown")
        video_id = str(source_record.get("video_id") or "unknown")
        video_key = (platform, video_id)
        candidate_id = str(
            source_record.get("candidate_id")
            or f"{video_id}:{round(float(source_record['clip_start_seconds']) * 1000)}"
        )
        digest = str(source_record.get("sha256") or _sha256(item["source_path"]))
        start = float(source_record["clip_start_seconds"])
        if digest in hashes:
            skip_counts["duplicate_sha256"] += 1
            continue
        if candidate_id in candidate_ids:
            skip_counts["duplicate_candidate"] += 1
            continue
        if any(
            abs(start - existing) < 10.0
            for existing in starts_by_video.get(video_key, [])
        ):
            skip_counts["overlapping_source_interval"] += 1
            continue
        source_content_minutes_per_hour = getattr(
            args, "source_content_minutes_per_hour", None
        )
        source_budget = args.max_clips_per_video
        if source_content_minutes_per_hour is not None:
            source_budget = record_source_clip_budget(
                source_record,
                clip_seconds=30.0,
                base_clips=args.max_clips_per_video,
                content_minutes_per_hour=source_content_minutes_per_hour,
                max_clips=getattr(
                    args,
                    "max_duration_scaled_clips_per_video",
                    DEFAULT_MAX_CLIPS_PER_SOURCE,
                ),
            )
        video_budgets[video_key] = source_budget
        if video_counts[video_key] >= source_budget:
            skip_counts["source_video_cap"] += 1
            continue
        item["sha256"] = digest
        selected.append(item)
        hashes.add(digest)
        candidate_ids.add(candidate_id)
        starts_by_video.setdefault(video_key, []).append(start)
        video_counts[video_key] += 1
        if len(selected) >= args.accepted_limit:
            break

    if len(selected) < args.accepted_limit:
        raise RuntimeError(
            f"Only {len(selected)}/{args.accepted_limit} clips survive combined "
            "validation and final deduplication"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    selected_names: set[str] = set()
    for index, item in enumerate(selected):
        filename = item["filename"]
        if filename in selected_names:
            filename = f"batch{item['batch_index']:02d}-{filename}"
        selected_names.add(filename)
        destination = audio_dir / filename
        if not destination.exists():
            try:
                os.link(item["source_path"], destination)
            except OSError:
                shutil.copy2(item["source_path"], destination)
        if _sha256(destination) != item["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch after materializing {filename}")
        record = dict(item["source_record"])
        record.update(
            {
                "record_index": index,
                "local_path": f"audio/{filename}",
                "source_batch_index": item["batch_index"],
                "m2d_validation": item["m2d_validation"],
                "asr_validation": item["asr_validation"],
            }
        )
        records.append(record)
    for stale in audio_dir.glob("*.wav"):
        if stale.name not in selected_names:
            stale.unlink()

    final_manifest = {
        "schema_version": 1,
        "name": f"Cinematic dialogue + music + SFX final {args.accepted_limit}",
        "created_at": _now(),
        "selection_seed": args.seed,
        "target_records": args.accepted_limit,
        "accepted_record_count": len(records),
        "maximum_clips_per_source_video": args.max_clips_per_video,
        "source_batches": batch_summaries,
        "source_candidate_count": len(candidates),
        "deduplication_skip_counts": dict(sorted(skip_counts.items())),
        "m2d_policy": CINEMATIC_POLICY_VERSION,
        "foreground_voice_policy": ASR_POLICY_VERSION,
        "explicit_metadata_policy": "cinematic_source_exclusions_v1",
        "records": records,
    }
    if getattr(args, "source_content_minutes_per_hour", None) is not None:
        final_manifest["source_diversity"] = source_diversity_policy(
            clip_seconds=30.0,
            base_clips=args.max_clips_per_video,
            content_minutes_per_hour=args.source_content_minutes_per_hour,
            max_clips=args.max_duration_scaled_clips_per_video,
        )
    temporary_manifest = args.output_dir / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(final_manifest, indent=2) + "\n")
    os.replace(temporary_manifest, args.output_dir / "manifest.json")
    audit = {
        "verified_at": _now(),
        "target_records": args.accepted_limit,
        "record_count": len(records),
        "audio_file_count": len(list(audio_dir.glob("*.wav"))),
        "unique_sha256_count": len(hashes),
        "unique_source_video_count": len(video_counts),
        "maximum_observed_clips_per_source_video": max(video_counts.values()),
        "all_requirements_pass": (
            len(records) == args.accepted_limit
            and len(list(audio_dir.glob("*.wav"))) == args.accepted_limit
            and len(hashes) == args.accepted_limit
            and all(
                count <= video_budgets[video] for video, count in video_counts.items()
            )
            and all(record["m2d_validation"]["accepted"] for record in records)
            and all(record["asr_validation"]["accepted"] for record in records)
        ),
        "source_batches": batch_summaries,
        "deduplication_skip_counts": dict(sorted(skip_counts.items())),
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if not audit["all_requirements_pass"]:
        raise RuntimeError("Final merged dataset audit failed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score", help="Score WAV files with M2D")
    score.add_argument("--input-dir", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--m2d-repo", type=Path, required=True)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--class-labels", type=Path, required=True)
    score.add_argument("--ontology", type=Path, required=True)
    score.add_argument("--m2d-commit", default="unknown")
    score.add_argument("--device", default="cuda")
    score.add_argument("--glob", default="*.wav")
    score.add_argument("--limit", type=int)
    score.add_argument("--overwrite", action="store_true")
    score.add_argument("--require-cinematic-mix", action="store_true")
    score.add_argument("--follow", action="store_true")
    score.add_argument("--producer-done", type=Path)
    score.add_argument("--poll-seconds", type=float, default=2.0)
    score.add_argument("--shard-index", type=int, default=0)
    score.add_argument("--shard-count", type=int, default=1)
    score.set_defaults(handler=score_directory)

    asr_score = subparsers.add_parser(
        "asr-score", help="Confirm decodable foreground voice with faster-whisper"
    )
    asr_score.add_argument("--input-dir", type=Path, required=True)
    asr_score.add_argument("--output", type=Path, required=True)
    asr_score.add_argument("--model", default="small")
    asr_score.add_argument(
        "--model-label",
        help=(
            "Stable model identity written to metadata and used for resume "
            "checks when --model is an offline snapshot path"
        ),
    )
    asr_score.add_argument("--device", default="cuda")
    asr_score.add_argument("--compute-type", default="float16")
    asr_score.add_argument("--download-root", type=Path)
    asr_score.add_argument("--m2d-results", type=Path)
    asr_score.add_argument("--m2d-results-dir", type=Path)
    asr_score.add_argument("--require-cinematic-mix", action="store_true")
    asr_score.add_argument("--beam-size", type=int, default=5)
    asr_score.add_argument("--glob", default="*.wav")
    asr_score.add_argument("--limit", type=int)
    asr_score.add_argument("--overwrite", action="store_true")
    asr_score.add_argument("--follow", action="store_true")
    asr_score.add_argument("--producer-done", type=Path)
    asr_score.add_argument("--poll-seconds", type=float, default=2.0)
    asr_score.add_argument("--shard-index", type=int, default=0)
    asr_score.add_argument("--shard-count", type=int, default=1)
    asr_score.add_argument("--max-inference-workers", type=int, default=1)
    asr_score.add_argument("--cpu-threads", type=int, default=0)
    asr_score.add_argument("--autoscale-control", type=Path)
    asr_score.add_argument("--probe-requests-dir", type=Path)
    asr_score.add_argument("--probe-results-dir", type=Path)
    asr_score.set_defaults(handler=score_asr_directory)

    materialize = subparsers.add_parser(
        "materialize", help="Create a folder containing only accepted clips"
    )
    materialize.add_argument("--input-dir", type=Path, required=True)
    materialize.add_argument("--results", type=Path, required=True)
    materialize.add_argument("--asr-results", type=Path)
    materialize.add_argument("--source-manifest", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.add_argument("--require-cinematic-mix", action="store_true")
    materialize.add_argument("--accepted-limit", type=int)
    materialize.set_defaults(handler=materialize_accepted)

    merge = subparsers.add_parser(
        "merge-materialize",
        help="Merge validated batches into one exact, deduplicated final set",
    )
    merge.add_argument("--batch", type=Path, action="append", required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--accepted-limit", type=int, default=1000)
    merge.add_argument("--max-clips-per-video", type=int, default=3)
    merge.add_argument("--source-content-minutes-per-hour", type=float)
    merge.add_argument(
        "--max-duration-scaled-clips-per-video",
        type=int,
        default=DEFAULT_MAX_CLIPS_PER_SOURCE,
    )
    merge.add_argument("--seed", type=int, default=20260715)
    merge.add_argument("--require-cinematic-mix", action="store_true")
    merge.set_defaults(handler=merge_materialized)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
