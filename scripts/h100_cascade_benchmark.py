#!/usr/bin/env python3
"""Benchmark a two-stage SAM-Audio cascade on an H100.

Stage 1 separates a broad music prompt. Stage 2 runs on the Stage 1 residual
with a voices prompt. This is intended for testing whether removing music first
reduces spillover into the voices output.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
import torchaudio


DTYPES = {
    "fp32": torch.float32,
    "tf32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="facebook/sam-audio-small-tv")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--stage1-prompt", default="music soundtrack")
    parser.add_argument("--stage2-prompt", default="human voices")
    parser.add_argument("--dtype-policy", choices=sorted(DTYPES), default="tf32")
    parser.add_argument(
        "--compile-transformer",
        choices=["none", "default", "reduce-overhead", "max-autotune"],
        default="none",
    )
    parser.add_argument("--predict-spans", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive-initial-candidates", type=int, default=4)
    parser.add_argument("--adaptive-max-candidates", type=int, default=8)
    parser.add_argument("--adaptive-margin", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--record-cold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-warm-seconds", type=float, default=60.0)
    parser.add_argument("--min-warm-runs", type=int, default=3)
    parser.add_argument(
        "--disable-visual-ranker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Avoid loading ImageBind visual reranker for audio-only text benchmarks.",
    )
    return parser.parse_args()


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_precision(policy: str) -> torch.dtype:
    dtype = DTYPES[policy]
    if policy == "tf32":
        try:
            torch.backends.fp32_precision = "tf32"
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.fp32_precision = "tf32"
        except Exception:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    elif policy == "fp32":
        torch.set_float32_matmul_precision("highest")
    return dtype


def keep_rankers_fp32(model: torch.nn.Module) -> None:
    for name in ("text_ranker", "visual_ranker"):
        ranker = getattr(model, name, None)
        if ranker is not None:
            ranker.float()


def compile_transformer_if_requested(model: Any, mode: str) -> None:
    if mode == "none":
        return
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable in this PyTorch build")
    if hasattr(model, "compile_h100"):
        model.compile_h100(mode=mode)
    else:
        model.transformer = torch.compile(model.transformer, mode=mode)


def save_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().float().cpu()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(str(path), wav, sample_rate)


def separate_adaptive(
    *,
    model: Any,
    processor: Any,
    audio_path: Path,
    prompt: str,
    args: argparse.Namespace,
    dtype: torch.dtype,
    seed: int,
) -> tuple[Any, float]:
    set_seed(seed)
    batch = processor(audios=[str(audio_path)], descriptions=[prompt]).to("cuda")
    if dtype != torch.float32:
        batch.audios = batch.audios.to(dtype)
    cuda_sync()
    start = now_ms()
    result = model.separate_adaptive_rerank(
        batch,
        predict_spans=args.predict_spans,
        initial_candidates=args.adaptive_initial_candidates,
        max_candidates=args.adaptive_max_candidates,
        margin=args.adaptive_margin,
    )
    cuda_sync()
    return result, now_ms() - start


def run_cascade(
    *,
    model: Any,
    processor: Any,
    args: argparse.Namespace,
    dtype: torch.dtype,
    run_dir: Path,
    metrics_path: Path,
    phase: str,
    run_index: int,
) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_start = now_ms()
    artifact_dir = run_dir / "artifacts" / phase / f"{run_index:04d}"
    temp_dir = run_dir / "tmp" / phase
    temp_dir.mkdir(parents=True, exist_ok=True)

    stage1, stage1_ms = separate_adaptive(
        model=model,
        processor=processor,
        audio_path=args.audio_path,
        prompt=args.stage1_prompt,
        args=args,
        dtype=dtype,
        seed=args.seed + run_index * 10,
    )
    stage1_target_path = artifact_dir / "stage1_music_soundtrack_target.wav"
    stage1_residual_path = artifact_dir / "stage1_music_soundtrack_residual.wav"
    save_wav(stage1_target_path, stage1.target[0], processor.audio_sampling_rate)
    save_wav(stage1_residual_path, stage1.residual[0], processor.audio_sampling_rate)

    residual_input_path = temp_dir / f"{run_index:04d}_music_residual_input.wav"
    save_wav(residual_input_path, stage1.residual[0], processor.audio_sampling_rate)

    stage2, stage2_ms = separate_adaptive(
        model=model,
        processor=processor,
        audio_path=residual_input_path,
        prompt=args.stage2_prompt,
        args=args,
        dtype=dtype,
        seed=args.seed + run_index * 10 + 1,
    )
    stage2_target_path = artifact_dir / "stage2_human_voices_from_music_residual_target.wav"
    stage2_residual_path = artifact_dir / "stage2_human_voices_from_music_residual_residual.wav"
    save_wav(stage2_target_path, stage2.target[0], processor.audio_sampling_rate)
    save_wav(stage2_residual_path, stage2.residual[0], processor.audio_sampling_rate)

    row = {
        "status": "ok",
        "run_name": args.run_name,
        "phase": phase,
        "run_index": run_index,
        "model_id": args.model_id,
        "audio_path": str(args.audio_path),
        "stage1_prompt": args.stage1_prompt,
        "stage2_prompt": args.stage2_prompt,
        "dtype_policy": args.dtype_policy,
        "compile_transformer": args.compile_transformer,
        "predict_spans": args.predict_spans,
        "adaptive_initial_candidates": args.adaptive_initial_candidates,
        "adaptive_max_candidates": args.adaptive_max_candidates,
        "adaptive_margin": args.adaptive_margin,
        "stage1_music_ms": round(stage1_ms, 3),
        "stage2_voice_ms": round(stage2_ms, 3),
        "total_ms": round(now_ms() - total_start, 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 4),
        "artifacts": {
            "stage1_target": str(stage1_target_path),
            "stage1_residual": str(stage1_residual_path),
            "stage2_target": str(stage2_target_path),
            "stage2_residual": str(stage2_residual_path),
            "stage2_input": str(residual_input_path),
        },
    }
    with metrics_path.open("a") as fout:
        fout.write(json.dumps(row, default=str, sort_keys=True) + "\n")
    return row


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "run_name",
        "phase",
        "count",
        "mean_total_ms",
        "p50_total_ms",
        "p95_total_ms",
        "mean_stage1_music_ms",
        "mean_stage2_voice_ms",
        "max_peak_allocated_gb",
        "max_peak_reserved_gb",
    ]
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for phase in ("cold", "warm"):
            phase_rows = [row for row in rows if row["phase"] == phase]
            if not phase_rows:
                continue
            totals = [float(row["total_ms"]) for row in phase_rows]
            p95 = sorted(totals)[max(0, int(0.95 * (len(totals) - 1)))]
            writer.writerow(
                {
                    "run_name": phase_rows[0]["run_name"],
                    "phase": phase,
                    "count": len(phase_rows),
                    "mean_total_ms": round(statistics.mean(totals), 3),
                    "p50_total_ms": round(statistics.median(totals), 3),
                    "p95_total_ms": round(p95, 3),
                    "mean_stage1_music_ms": round(
                        statistics.mean(row["stage1_music_ms"] for row in phase_rows), 3
                    ),
                    "mean_stage2_voice_ms": round(
                        statistics.mean(row["stage2_voice_ms"] for row in phase_rows), 3
                    ),
                    "max_peak_allocated_gb": max(
                        row["peak_allocated_gb"] for row in phase_rows
                    ),
                    "max_peak_reserved_gb": max(
                        row["peak_reserved_gb"] for row in phase_rows
                    ),
                }
            )


def main() -> int:
    args = parse_args()
    args.run_name = args.run_name or (
        f"cascade-{args.dtype_policy}-adaptive"
        f"{'-compile-' + args.compile_transformer if args.compile_transformer != 'none' else ''}"
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for meaningful H100 benchmarking")

    dtype = configure_precision(args.dtype_policy)
    run_dir = args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.csv"
    metrics_path.unlink(missing_ok=True)

    from sam_audio import SAMAudio, SAMAudioProcessor

    load_start = now_ms()
    model_kwargs: dict[str, Any] = {}
    if args.disable_visual_ranker:
        model_kwargs["visual_ranker"] = None
    model = SAMAudio.from_pretrained(
        args.model_id,
        proxies=None,
        resume_download=False,
        **model_kwargs,
    ).eval().cuda()
    if dtype != torch.float32:
        model = model.to(dtype)
        keep_rankers_fp32(model)
    compile_transformer_if_requested(model, args.compile_transformer)
    processor = SAMAudioProcessor.from_pretrained(args.model_id)
    load_ms = now_ms() - load_start

    if not hasattr(model, "separate_adaptive_rerank"):
        raise RuntimeError("This branch does not implement adaptive reranking")

    manifest = {
        "run_name": args.run_name,
        "model_id": args.model_id,
        "audio_path": str(args.audio_path),
        "stage1_prompt": args.stage1_prompt,
        "stage2_prompt": args.stage2_prompt,
        "load_ms": round(load_ms, 3),
        "args": vars(args),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, default=str, indent=2, sort_keys=True)
    )

    rows = []
    if args.record_cold:
        rows.append(
            run_cascade(
                model=model,
                processor=processor,
                args=args,
                dtype=dtype,
                run_dir=run_dir,
                metrics_path=metrics_path,
                phase="cold",
                run_index=0,
            )
        )

    for warmup_index in range(args.warmup):
        run_cascade(
            model=model,
            processor=processor,
            args=args,
            dtype=dtype,
            run_dir=run_dir,
            metrics_path=metrics_path,
            phase="warmup",
            run_index=warmup_index,
        )

    warm_start = now_ms()
    warm_runs = 0
    while True:
        rows.append(
            run_cascade(
                model=model,
                processor=processor,
                args=args,
                dtype=dtype,
                run_dir=run_dir,
                metrics_path=metrics_path,
                phase="warm",
                run_index=warm_runs,
            )
        )
        warm_runs += 1
        warm_elapsed_seconds = (now_ms() - warm_start) / 1000.0
        if warm_runs >= args.min_warm_runs and warm_elapsed_seconds >= args.min_warm_seconds:
            break

    write_summary(summary_path, rows)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
