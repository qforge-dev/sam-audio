"""Evaluate scene-caption prompt variants against representative saved stems.

The lab is read-only: it never changes job, record, or snapshot state.  It emits
one JSONL row per model response plus a ranked summary so prompt changes can be
measured on real pipeline audio before deployment.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .flamingo_client import AudioFlamingoClient
from .training_dataset import (
    CAPTION_MAX_NEW_TOKENS,
    _background_evidence,
    _caption_completion,
    _caption_prompt,
    _description_evaluation,
    _format_scene_description,
    _prepare_caption_audio,
    _record_root,
    connect,
)


def _prompt_variants(tag: dict[str, Any]) -> dict[str, str]:
    grounded = _caption_prompt(tag)
    evidence_marker = "The hints are fallible"
    audio_only = grounded.split(evidence_marker, 1)[0].rstrip()
    audio_only += (
        " Base every detail on the audio itself; no auxiliary acoustic hints are "
        "provided."
    )
    transition_focused = grounded.replace(
        "Produce a dense timestamped acoustic analysis",
        "First identify every meaningful acoustic transition, then produce a dense "
        "timestamped acoustic analysis",
    )
    return {
        "native_v2_grounded": grounded,
        "native_v2_audio_only": audio_only,
        "native_v2_transition_focused": transition_focused,
    }


def _timeline_coverage(timeline: list[dict[str, Any]]) -> float:
    intervals = sorted(
        (
            max(0.0, float(item["start_seconds"])),
            min(30.0, float(item["end_seconds"])),
        )
        for item in timeline
    )
    covered = 0.0
    cursor = 0.0
    for start, end in intervals:
        if end <= start:
            continue
        covered += max(0.0, end - max(start, cursor))
        cursor = max(cursor, end)
    return min(1.0, covered / 30.0)


def evaluate_caption(
    parsed: dict[str, Any],
    parse: dict[str, Any],
    tag: dict[str, Any],
) -> dict[str, Any]:
    validation = _description_evaluation(parsed)
    timeline = parsed.get("timeline") or []
    rendered = _format_scene_description(parsed)
    normalized = rendered.casefold()
    tags = [
        str(item[0]).casefold()
        for window in _background_evidence(tag)
        for item in window["tags"]
    ]
    unique_tags = list(dict.fromkeys(tags))
    matched_tags = [value for value in unique_tags if value in normalized]
    detailed_events = sum(
        1
        for item in timeline
        if len(" ".join(item.get("events") or []).split()) >= 5
    )
    coverage = _timeline_coverage(timeline)
    style_dimensions = {
        "spatial": any(
            token in normalized
            for token in (
                "left",
                "right",
                "center",
                "stereo",
                "distant",
                "nearby",
                "wide",
                "spacious",
                "reverber",
            )
        ),
        "dynamics": any(
            token in normalized
            for token in (
                "loud",
                "soft",
                "quiet",
                "intens",
                "grows",
                "rises",
                "fades",
                "recedes",
                "thins",
            )
        ),
        "texture": any(
            token in normalized
            for token in (
                "texture",
                "drone",
                "pulse",
                "metallic",
                "sustained",
                "resonant",
                "percuss",
                "hum",
                "wash",
            )
        ),
        "transitions": any(
            token in normalized
            for token in (
                "begins",
                "builds",
                "changes",
                "shifts",
                "returns",
                "drops",
                "ends",
                "transition",
            )
        ),
    }
    score = 0.0
    score += (
        20
        if parse.get("format")
        in {
            "description_timeline_sections_v2",
            "audio_flamingo_native_timeline_v2",
        }
        else 0
    )
    score += 20 if validation["status"] == "success" else 0
    score += 15 * coverage
    score += 15 * min(1.0, detailed_events / max(1, len(timeline)))
    score += 10 if 2 <= len(timeline) <= 6 else 0
    score += 5 if not parse.get("speech_mentions_omitted") else 0
    score += 5 * min(1.0, len(matched_tags) / max(1, min(3, len(unique_tags))))
    score += 2.5 * sum(style_dimensions.values())
    return {
        "score": round(score, 2),
        "validation": validation,
        "timeline_coverage": round(coverage, 3),
        "detailed_timeline_events": detailed_events,
        "matched_m2d_tags": matched_tags,
        "m2d_tag_match_ratio": round(
            len(matched_tags) / max(1, len(unique_tags)), 3
        ),
        "style_dimensions": style_dimensions,
        "style_dimension_count": sum(style_dimensions.values()),
    }


def _sample_jobs(workspace: Path, count: int) -> list[dict[str, Any]]:
    connection = connect(workspace)
    candidates = [
        dict(row)
        for row in connection.execute(
            """SELECT id,quality_bucket,tag_json,description_json FROM jobs
            WHERE tag_json IS NOT NULL AND description_status='complete'
            ORDER BY id DESC LIMIT 5000"""
        )
        if (_record_root(workspace, int(row["id"])) / "background.wav").exists()
    ]
    connection.close()
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_bucket[str(row.get("quality_bucket") or "unclassified")].append(row)
    selected: list[dict[str, Any]] = []
    buckets = ("success", "review", "failure", "unclassified")
    while len(selected) < count and any(by_bucket.values()):
        for bucket in buckets:
            if by_bucket[bucket] and len(selected) < count:
                selected.append(by_bucket[bucket].pop(0))
    return selected


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank prompt variants from stored responses without rerunning inference."""
    summary: dict[str, Any] = {
        "samples": len({int(item["job_id"]) for item in results}),
        "responses": len(results),
    }
    variants: dict[str, Any] = {}
    for variant in _prompt_variants({}):
        subset = [item for item in results if item["variant"] == variant]
        if not subset:
            continue
        variants[variant] = {
            "mean_score": round(
                sum(item["evaluation"]["score"] for item in subset) / len(subset), 2
            ),
            "contract_success": sum(
                item["evaluation"]["validation"]["status"] == "success"
                for item in subset
            ),
            "contract_success_rate": round(
                sum(
                    item["evaluation"]["validation"]["status"] == "success"
                    for item in subset
                )
                / len(subset),
                3,
            ),
            "contract_failure_reasons": dict(
                Counter(
                    reason
                    for item in subset
                    if item["evaluation"]["validation"]["status"] != "success"
                    for reason in item["evaluation"]["validation"].get(
                        "review_reasons", []
                    )
                )
            ),
            "parse_formats": dict(Counter(item["parse"]["format"] for item in subset)),
            "mean_seconds": round(
                sum(item["processing_seconds"] for item in subset) / len(subset), 3
            ),
            "mean_description_words": round(
                sum(
                    item["evaluation"]["validation"]["signals"][
                        "description_word_count"
                    ]
                    for item in subset
                )
                / len(subset),
                2,
            ),
            "mean_timeline_events": round(
                sum(
                    item["evaluation"]["validation"]["signals"][
                        "timeline_event_count"
                    ]
                    for item in subset
                )
                / len(subset),
                2,
            ),
            "mean_style_dimensions": round(
                sum(
                    item["evaluation"]["style_dimension_count"] for item in subset
                )
                / len(subset),
                2,
            ),
        }
    summary["variants"] = variants
    summary["ranking_policy"] = (
        "contract_success_rate, then mean_score, then lower mean_seconds"
    )
    summary["winner"] = max(
        variants,
        key=lambda value: (
            variants[value]["contract_success_rate"],
            variants[value]["mean_score"],
            -variants[value]["mean_seconds"],
        ),
    )
    return summary


def run_lab(
    workspace: Path,
    *,
    api_url: str,
    sample_count: int,
    output: Path,
) -> dict[str, Any]:
    rows = _sample_jobs(workspace, sample_count)
    if not rows:
        raise RuntimeError("No completed jobs with local background.wav artifacts")
    client = AudioFlamingoClient(api_url)
    results: list[dict[str, Any]] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as destination:
        for row in rows:
            tag = json.loads(row["tag_json"])
            source = _record_root(workspace, int(row["id"])) / "background.wav"
            with tempfile.TemporaryDirectory(prefix="caption-lab-") as temporary:
                prepared = Path(temporary) / "background.wav"
                _prepare_caption_audio(source, prepared)
                for variant, prompt in _prompt_variants(tag).items():
                    started = time.perf_counter()
                    response = client.ask(
                        prepared, prompt, max_new_tokens=CAPTION_MAX_NEW_TOKENS
                    )
                    parsed, parse = _caption_completion(str(response["text"]), tag)
                    result = {
                        "job_id": int(row["id"]),
                        "quality_bucket": row["quality_bucket"],
                        "variant": variant,
                        "processing_seconds": round(time.perf_counter() - started, 3),
                        "model": response.get("model"),
                        "raw_text": response.get("text"),
                        "parse": parse,
                        "parsed": parsed,
                        "rendered": _format_scene_description(parsed),
                        "evaluation": evaluate_caption(parsed, parse, tag),
                    }
                    destination.write(json.dumps(result, ensure_ascii=False) + "\n")
                    destination.flush()
                    results.append(result)
    summary = summarize_results(results)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8001")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_lab(
                args.workspace,
                api_url=args.api_url,
                sample_count=max(1, args.samples),
                output=args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
