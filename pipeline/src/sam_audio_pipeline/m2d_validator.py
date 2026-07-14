"""Validate dialogue-over-background clips with the official M2D tagger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SPEECH_ROOT = "/m/09x0r"
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
MIN_BACKGROUND_WINDOWS = 4
MIN_OVERLAP_WINDOWS = 3
MAX_VOCAL_MUSIC_WINDOWS = 1
TOP_LABELS = 8
POLICY_VERSION = "spoken_dialogue_instrumental_background_m2d_v3"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    music = indices((MUSIC_ROOT,))
    human = indices((HUMAN_ROOT,))
    nonmusic_background = indices(BACKGROUND_ROOTS)
    vocal_music = indices(VOCAL_MUSIC_ROOTS)
    return labels, {
        "speech": speech,
        "music": music,
        "nonmusic_background": nonmusic_background,
        "background": music | nonmusic_background,
        "human": human,
        "vocal_music": vocal_music,
    }


def _family_evidence(
    probabilities: np.ndarray, family: set[int]
) -> tuple[float, int]:
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
) -> dict[str, Any]:
    """Evaluate actual temporal overlap from M2D window probabilities."""
    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        raise ValueError("Expected [windows, AudioSet classes] probabilities")
    if starts is None:
        starts = [index * WINDOW_HOP_SECONDS for index in range(len(probabilities))]
    windows: list[dict[str, Any]] = []
    for start, row in zip(starts, probabilities, strict=True):
        evidence = {
            name: _family_evidence(row, families[name])
            for name in (
                "speech",
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
    background_windows = sum(item["background_active"] for item in windows)
    overlap_windows = sum(item["overlap_active"] for item in windows)
    vocal_music_windows = sum(item["vocal_music_active"] for item in windows)
    overlap_music = sum(
        item["music_score"] for item in windows if item["overlap_active"]
    )
    overlap_nonmusic = sum(
        item["nonmusic_background_score"]
        for item in windows
        if item["overlap_active"]
    )
    if overlap_music > overlap_nonmusic * 1.25:
        background_bucket = "music_led"
    elif overlap_nonmusic > overlap_music * 1.25:
        background_bucket = "effects_ambience_led"
    else:
        background_bucket = "mixed_music_and_effects"

    rejections: list[str] = []
    if speech_windows < MIN_SPEECH_WINDOWS:
        rejections.append("insufficient_speech")
    if strong_speech_windows < MIN_STRONG_SPEECH_WINDOWS:
        rejections.append("insufficient_strong_speech")
    if background_windows < MIN_BACKGROUND_WINDOWS:
        rejections.append("insufficient_background")
    if overlap_windows < MIN_OVERLAP_WINDOWS:
        rejections.append("insufficient_dialogue_background_overlap")
    if vocal_music_windows > MAX_VOCAL_MUSIC_WINDOWS:
        rejections.append("vocal_music_present")
    window_count = len(windows)
    return {
        "accepted": not rejections,
        "policy": POLICY_VERSION,
        "rejection_reasons": rejections,
        "background_bucket": background_bucket,
        "window_count": window_count,
        "speech_active_windows": speech_windows,
        "strong_speech_active_windows": strong_speech_windows,
        "background_active_windows": background_windows,
        "overlap_active_windows": overlap_windows,
        "vocal_music_active_windows": vocal_music_windows,
        "speech_coverage": round(speech_windows / window_count, 6),
        "strong_speech_coverage": round(strong_speech_windows / window_count, 6),
        "background_coverage": round(background_windows / window_count, 6),
        "overlap_coverage": round(overlap_windows / window_count, 6),
        "vocal_music_coverage": round(vocal_music_windows / window_count, 6),
        "windows": windows,
    }


def _audio_windows(
    waveform: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, list[float]]:
    wanted = int(round(10.0 * sample_rate))
    waveform = waveform[:wanted]
    if len(waveform) < wanted:
        waveform = np.pad(waveform, (0, wanted - len(waveform)))
    length = int(round(WINDOW_SECONDS * sample_rate))
    starts = list(np.arange(0.0, 10.0 - WINDOW_SECONDS + 1e-6, WINDOW_HOP_SECONDS))
    windows = [
        waveform[
            round(start * sample_rate) : round(start * sample_rate) + length
        ]
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
    files = sorted(args.input_dir.glob(args.glob))
    if args.limit:
        files = files[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if args.output.exists() and not args.overwrite:
        for line in args.output.read_text().splitlines():
            if line.strip():
                existing.add(str(json.loads(line)["filename"]))
    mode = "w" if args.overwrite else "a"
    metadata = {
        "model": "nttcslab/m2d",
        "checkpoint": args.checkpoint.parent.name,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "m2d_repository_commit": args.m2d_commit,
        "policy": POLICY_VERSION,
        "window_seconds": WINDOW_SECONDS,
        "window_hop_seconds": WINDOW_HOP_SECONDS,
        "minimum_window_probability": MIN_WINDOW_PROBABILITY,
        "maximum_active_rank": MAX_ACTIVE_RANK,
        "minimum_speech_windows": MIN_SPEECH_WINDOWS,
        "minimum_strong_speech_probability": MIN_STRONG_SPEECH_PROBABILITY,
        "maximum_strong_speech_rank": MAX_STRONG_SPEECH_RANK,
        "minimum_strong_speech_windows": MIN_STRONG_SPEECH_WINDOWS,
        "minimum_background_windows": MIN_BACKGROUND_WINDOWS,
        "minimum_overlap_windows": MIN_OVERLAP_WINDOWS,
        "maximum_vocal_music_windows": MAX_VOCAL_MUSIC_WINDOWS,
    }
    processed = 0
    with args.output.open(mode, encoding="utf-8") as destination:
        for index, path in enumerate(files, start=1):
            if path.name in existing:
                continue
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
                    probabilities, labels, families, starts=starts
                ),
            }
            destination.write(json.dumps(result, separators=(",", ":")) + "\n")
            destination.flush()
            processed += 1
            if index % 25 == 0 or index == len(files):
                logger.info("M2D scored %d/%d clips", index, len(files))
    logger.info("Wrote %d new validation records to %s", processed, args.output)


def _enforce_current_voice_gate(result: dict[str, Any]) -> dict[str, Any]:
    """Apply the strong-voice gate to current and legacy M2D results."""
    result = dict(result)
    windows = []
    for source in result.get("windows", []):
        window = dict(source)
        window["strong_speech_active"] = (
            float(window.get("speech_score", 0.0))
            >= MIN_STRONG_SPEECH_PROBABILITY
            and int(window.get("speech_rank", 10_000)) <= MAX_STRONG_SPEECH_RANK
        )
        windows.append(window)
    strong_speech_windows = sum(
        bool(window["strong_speech_active"]) for window in windows
    )
    strong_voice_present = strong_speech_windows >= MIN_STRONG_SPEECH_WINDOWS
    reasons = list(dict.fromkeys(result.get("rejection_reasons", [])))
    if not strong_voice_present and "insufficient_strong_speech" not in reasons:
        reasons.append("insufficient_strong_speech")
    previous_policy = result.get("policy")
    result.update(
        {
            "accepted": bool(result.get("accepted")) and strong_voice_present,
            "policy": POLICY_VERSION,
            "rejection_reasons": reasons,
            "strong_speech_active_windows": strong_speech_windows,
            "strong_speech_coverage": round(
                strong_speech_windows / max(1, len(windows)), 6
            ),
            "windows": windows,
        }
    )
    if previous_policy and previous_policy != POLICY_VERSION:
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
    results = [
        _enforce_current_voice_gate(json.loads(line))
        for line in args.results.read_text().splitlines()
        if line.strip()
    ]
    accepted = [item for item in results if item.get("accepted")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    _materialize_audio(args.input_dir, audio_dir, accepted)

    music_led = [
        item for item in accepted if item["background_bucket"] == "music_led"
    ]
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
            }
        )
        records.append(original)
    rejection_counts = Counter(
        reason
        for result in results
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
        "policy": POLICY_VERSION,
        "background_bucket_counts": dict(sorted(bucket_counts.items())),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
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
    score.set_defaults(handler=score_directory)

    materialize = subparsers.add_parser(
        "materialize", help="Create a folder containing only accepted clips"
    )
    materialize.add_argument("--input-dir", type=Path, required=True)
    materialize.add_argument("--results", type=Path, required=True)
    materialize.add_argument("--source-manifest", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.set_defaults(handler=materialize_accepted)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
