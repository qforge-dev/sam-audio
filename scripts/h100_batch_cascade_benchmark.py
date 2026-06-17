#!/usr/bin/env python3
"""Benchmark batch-parallel two-stage SAM-Audio cascade inference on an H100."""

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
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="facebook/sam-audio-small-tv")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-benchmark-files", type=int, default=None)
    parser.add_argument("--stage1-prompt", default="music soundtrack")
    parser.add_argument("--stage2-prompt", default="human voices")
    parser.add_argument("--dtype-policy", choices=sorted(DTYPES), default="tf32")
    parser.add_argument(
        "--compile-transformer",
        choices=["none", "default", "reduce-overhead", "max-autotune"],
        default="default",
    )
    parser.add_argument("--predict-spans", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive-initial-candidates", type=int, default=4)
    parser.add_argument("--adaptive-max-candidates", type=int, default=8)
    parser.add_argument("--adaptive-margin", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
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


def top_level_mp3s(input_dir: Path) -> list[Path]:
    files = sorted(path for path in input_dir.glob("*.mp3") if path.is_file())
    if len(files) < 2:
        raise ValueError(f"Need at least two top-level MP3 files under {input_dir}")
    return files


def batches(items: list[Path], batch_size: int) -> list[list[Path]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def save_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().float().cpu()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(str(path), wav, sample_rate)


def sanitize_stem(path: Path) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in path.stem)


def separate_adaptive_batch(
    *,
    model: Any,
    processor: Any,
    audio_paths: list[Path],
    prompt: str,
    args: argparse.Namespace,
    dtype: torch.dtype,
    seed: int,
) -> tuple[Any, float]:
    set_seed(seed)
    batch = processor(
        audios=[str(path) for path in audio_paths],
        descriptions=[prompt] * len(audio_paths),
    ).to("cuda")
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


def run_batch(
    *,
    model: Any,
    processor: Any,
    args: argparse.Namespace,
    dtype: torch.dtype,
    run_dir: Path,
    metrics_path: Path,
    audio_paths: list[Path],
    phase: str,
    batch_index: int,
) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_start = now_ms()
    batch_dir = run_dir / "artifacts" / phase / f"batch_{batch_index:04d}"
    temp_dir = run_dir / "tmp" / phase / f"batch_{batch_index:04d}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    stage1, stage1_ms = separate_adaptive_batch(
        model=model,
        processor=processor,
        audio_paths=audio_paths,
        prompt=args.stage1_prompt,
        args=args,
        dtype=dtype,
        seed=args.seed + batch_index * 10,
    )

    residual_inputs: list[Path] = []
    artifacts: list[dict[str, str]] = []
    for item_index, audio_path in enumerate(audio_paths):
        item_dir = batch_dir / f"{item_index:02d}_{sanitize_stem(audio_path)}"
        temp_input = temp_dir / f"{item_index:02d}_{sanitize_stem(audio_path)}_music_residual_input.wav"
        stage1_target = item_dir / "stage1_music_soundtrack_target.wav"
        stage1_residual = item_dir / "stage1_music_soundtrack_residual.wav"
        save_wav(stage1_target, stage1.target[item_index], processor.audio_sampling_rate)
        save_wav(stage1_residual, stage1.residual[item_index], processor.audio_sampling_rate)
        save_wav(temp_input, stage1.residual[item_index], processor.audio_sampling_rate)
        residual_inputs.append(temp_input)
        artifacts.append(
            {
                "source": str(audio_path),
                "stage1_target": str(stage1_target),
                "stage1_residual": str(stage1_residual),
                "stage2_input": str(temp_input),
            }
        )

    stage2, stage2_ms = separate_adaptive_batch(
        model=model,
        processor=processor,
        audio_paths=residual_inputs,
        prompt=args.stage2_prompt,
        args=args,
        dtype=dtype,
        seed=args.seed + batch_index * 10 + 1,
    )

    for item_index, item in enumerate(artifacts):
        item_dir = Path(item["stage1_target"]).parent
        stage2_target = item_dir / "stage2_human_voices_from_music_residual_target.wav"
        stage2_residual = item_dir / "stage2_human_voices_from_music_residual_residual.wav"
        save_wav(stage2_target, stage2.target[item_index], processor.audio_sampling_rate)
        save_wav(stage2_residual, stage2.residual[item_index], processor.audio_sampling_rate)
        item["stage2_target"] = str(stage2_target)
        item["stage2_residual"] = str(stage2_residual)

    row = {
        "status": "ok",
        "run_name": args.run_name,
        "phase": phase,
        "batch_size": args.batch_size,
        "batch_index": batch_index,
        "chunk_count": len(audio_paths),
        "chunk_files": [path.name for path in audio_paths],
        "stage1_music_ms": round(stage1_ms, 3),
        "stage2_voice_ms": round(stage2_ms, 3),
        "total_ms": round(now_ms() - total_start, 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 4),
        "artifacts": artifacts,
    }
    write_jsonl(metrics_path, row)
    return row


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as fout:
        fout.write(json.dumps(row, default=str, sort_keys=True) + "\n")


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    benchmark_files: list[Path],
    wall_ms: float,
    load_ms: float,
    status: str,
    failure_reason: str | None = None,
) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("phase") == "benchmark"]
    totals = [float(row["total_ms"]) for row in ok_rows]
    processed_chunks = sum(int(row["chunk_count"]) for row in ok_rows)
    fields = [
        "run_name",
        "status",
        "failure_reason",
        "batch_size",
        "benchmark_chunk_count",
        "processed_chunk_count",
        "batch_count",
        "load_ms",
        "total_wall_ms",
        "chunks_per_sec",
        "chunks_per_min",
        "mean_batch_ms",
        "p50_batch_ms",
        "p95_batch_ms",
        "mean_stage1_music_ms",
        "mean_stage2_voice_ms",
        "max_peak_allocated_gb",
        "max_peak_reserved_gb",
    ]
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "run_name": args.run_name,
                "status": status,
                "failure_reason": failure_reason or "",
                "batch_size": args.batch_size,
                "benchmark_chunk_count": len(benchmark_files),
                "processed_chunk_count": processed_chunks,
                "batch_count": len(ok_rows),
                "load_ms": round(load_ms, 3),
                "total_wall_ms": round(wall_ms, 3),
                "chunks_per_sec": round(processed_chunks / (wall_ms / 1000.0), 4)
                if wall_ms > 0
                else "",
                "chunks_per_min": round(processed_chunks / (wall_ms / 60000.0), 3)
                if wall_ms > 0
                else "",
                "mean_batch_ms": round(statistics.mean(totals), 3) if totals else "",
                "p50_batch_ms": round(statistics.median(totals), 3) if totals else "",
                "p95_batch_ms": round(sorted(totals)[max(0, int(0.95 * (len(totals) - 1)))], 3)
                if totals
                else "",
                "mean_stage1_music_ms": round(
                    statistics.mean(float(row["stage1_music_ms"]) for row in ok_rows), 3
                )
                if ok_rows
                else "",
                "mean_stage2_voice_ms": round(
                    statistics.mean(float(row["stage2_voice_ms"]) for row in ok_rows), 3
                )
                if ok_rows
                else "",
                "max_peak_allocated_gb": max((row["peak_allocated_gb"] for row in ok_rows), default=""),
                "max_peak_reserved_gb": max((row["peak_reserved_gb"] for row in ok_rows), default=""),
            }
        )


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    args.run_name = args.run_name or f"batch-cascade-b{args.batch_size}"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for meaningful H100 benchmarking")

    audio_files = top_level_mp3s(args.input_dir)
    warmup_file = audio_files[0]
    benchmark_files = audio_files[1:]
    if args.max_benchmark_files is not None:
        benchmark_files = benchmark_files[: args.max_benchmark_files]
    if not benchmark_files:
        raise ValueError("No benchmark files remain after reserving the first MP3 for warmup")

    run_dir = args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.csv"
    metrics_path.unlink(missing_ok=True)

    dtype = configure_precision(args.dtype_policy)

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
        "input_dir": str(args.input_dir),
        "warmup_file": warmup_file.name,
        "benchmark_files": [path.name for path in benchmark_files],
        "load_ms": round(load_ms, 3),
        "args": vars(args),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, default=str, indent=2, sort_keys=True)
    )

    rows: list[dict[str, Any]] = []
    run_batch(
        model=model,
        processor=processor,
        args=args,
        dtype=dtype,
        run_dir=run_dir,
        metrics_path=metrics_path,
        audio_paths=[warmup_file],
        phase="warmup",
        batch_index=0,
    )

    wall_start = now_ms()
    status = "ok"
    failure_reason = None
    try:
        for batch_index, batch_files in enumerate(batches(benchmark_files, args.batch_size)):
            rows.append(
                run_batch(
                    model=model,
                    processor=processor,
                    args=args,
                    dtype=dtype,
                    run_dir=run_dir,
                    metrics_path=metrics_path,
                    audio_paths=batch_files,
                    phase="benchmark",
                    batch_index=batch_index,
                )
            )
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        failure_reason = str(exc).splitlines()[0]
        write_jsonl(
            metrics_path,
            {
                "status": status,
                "phase": "benchmark",
                "batch_size": args.batch_size,
                "failure_reason": failure_reason,
                "processed_batches_before_failure": len(rows),
            },
        )
    except Exception as exc:
        status = "error"
        failure_reason = f"{type(exc).__name__}: {exc}"
        write_jsonl(
            metrics_path,
            {
                "status": status,
                "phase": "benchmark",
                "batch_size": args.batch_size,
                "failure_reason": failure_reason,
                "processed_batches_before_failure": len(rows),
            },
        )
        raise
    finally:
        wall_ms = now_ms() - wall_start
        write_summary(
            summary_path,
            args=args,
            rows=rows,
            benchmark_files=benchmark_files,
            wall_ms=wall_ms,
            load_ms=load_ms,
            status=status,
            failure_reason=failure_reason,
        )

    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
