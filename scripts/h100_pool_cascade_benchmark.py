#!/usr/bin/env python3
"""Benchmark queue-based parallel two-stage SAM-Audio cascade inference on an H100."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import multiprocessing as mp
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

_WORKER: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="facebook/sam-audio-small-tv")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--workers", type=int, required=True)
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


def sanitize_stem(path: Path) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in path.stem)


def save_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().float().cpu()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(str(path), wav, sample_rate)


def separate_adaptive_one(
    *,
    audio_path: Path,
    prompt: str,
    seed: int,
) -> tuple[Any, float]:
    args = _WORKER["args"]
    model = _WORKER["model"]
    processor = _WORKER["processor"]
    dtype = _WORKER["dtype"]

    set_seed(seed)
    batch = processor(audios=[str(audio_path)], descriptions=[prompt]).to("cuda")
    if dtype != torch.float32:
        batch.audios = batch.audios.to(dtype)
    cuda_sync()
    start = now_ms()
    result = model.separate_adaptive_rerank(
        batch,
        predict_spans=args["predict_spans"],
        initial_candidates=args["adaptive_initial_candidates"],
        max_candidates=args["adaptive_max_candidates"],
        margin=args["adaptive_margin"],
    )
    cuda_sync()
    return result, now_ms() - start


def run_cascade_item(
    *,
    audio_path: Path,
    chunk_index: int,
    phase: str,
    seed: int,
) -> dict[str, Any]:
    args = _WORKER["args"]
    processor = _WORKER["processor"]
    worker_id = _WORKER["worker_id"]
    run_dir = Path(args["run_dir"])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_start = now_ms()
    safe_name = sanitize_stem(audio_path)
    item_dir = run_dir / "artifacts" / phase / f"{chunk_index:04d}_{safe_name}"
    temp_dir = run_dir / "tmp" / phase / f"worker_{worker_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    stage1, stage1_ms = separate_adaptive_one(
        audio_path=audio_path,
        prompt=args["stage1_prompt"],
        seed=seed,
    )
    stage1_target = item_dir / "stage1_music_soundtrack_target.wav"
    stage1_residual = item_dir / "stage1_music_soundtrack_residual.wav"
    stage2_input = temp_dir / f"{chunk_index:04d}_{safe_name}_music_residual_input.wav"
    save_wav(stage1_target, stage1.target[0], processor.audio_sampling_rate)
    save_wav(stage1_residual, stage1.residual[0], processor.audio_sampling_rate)
    save_wav(stage2_input, stage1.residual[0], processor.audio_sampling_rate)

    stage2, stage2_ms = separate_adaptive_one(
        audio_path=stage2_input,
        prompt=args["stage2_prompt"],
        seed=seed + 1,
    )
    stage2_target = item_dir / "stage2_human_voices_from_music_residual_target.wav"
    stage2_residual = item_dir / "stage2_human_voices_from_music_residual_residual.wav"
    save_wav(stage2_target, stage2.target[0], processor.audio_sampling_rate)
    save_wav(stage2_residual, stage2.residual[0], processor.audio_sampling_rate)

    return {
        "status": "ok",
        "run_name": args["run_name"],
        "phase": phase,
        "worker_id": worker_id,
        "pid": os.getpid(),
        "chunk_index": chunk_index,
        "chunk_file": audio_path.name,
        "stage1_music_ms": round(stage1_ms, 3),
        "stage2_voice_ms": round(stage2_ms, 3),
        "total_ms": round(now_ms() - total_start, 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 4),
        "artifacts": {
            "source": str(audio_path),
            "stage1_target": str(stage1_target),
            "stage1_residual": str(stage1_residual),
            "stage2_input": str(stage2_input),
            "stage2_target": str(stage2_target),
            "stage2_residual": str(stage2_residual),
        },
    }


def init_worker(args: dict[str, Any], warmup_path: str, worker_counter: Any) -> None:
    with worker_counter.get_lock():
        worker_id = int(worker_counter.value)
        worker_counter.value += 1

    torch.cuda.set_device(0)
    dtype = configure_precision(args["dtype_policy"])

    from sam_audio import SAMAudio, SAMAudioProcessor

    load_start = now_ms()
    model_kwargs: dict[str, Any] = {}
    if args["disable_visual_ranker"]:
        model_kwargs["visual_ranker"] = None
    model = SAMAudio.from_pretrained(
        args["model_id"],
        proxies=None,
        resume_download=False,
        **model_kwargs,
    ).eval().cuda()
    if dtype != torch.float32:
        model = model.to(dtype)
        keep_rankers_fp32(model)
    compile_transformer_if_requested(model, args["compile_transformer"])
    processor = SAMAudioProcessor.from_pretrained(args["model_id"])
    load_ms = now_ms() - load_start

    if not hasattr(model, "separate_adaptive_rerank"):
        raise RuntimeError("This branch does not implement adaptive reranking")

    _WORKER.update(
        {
            "args": args,
            "dtype": dtype,
            "model": model,
            "processor": processor,
            "worker_id": worker_id,
            "load_ms": load_ms,
        }
    )
    warmup_start = now_ms()
    warmup_row = run_cascade_item(
        audio_path=Path(warmup_path),
        chunk_index=0,
        phase="warmup",
        seed=args["seed"] + worker_id * 1000,
    )
    warmup_row["load_ms"] = round(load_ms, 3)
    warmup_row["warmup_wall_ms"] = round(now_ms() - warmup_start, 3)
    _WORKER["warmup_row"] = warmup_row


def worker_ready() -> dict[str, Any]:
    return dict(_WORKER["warmup_row"])


def worker_process_chunk(chunk_index: int, audio_path: str, seed: int) -> dict[str, Any]:
    try:
        return run_cascade_item(
            audio_path=Path(audio_path),
            chunk_index=chunk_index,
            phase="benchmark",
            seed=seed,
        )
    except torch.cuda.OutOfMemoryError as exc:
        return {
            "status": "oom",
            "run_name": _WORKER["args"]["run_name"],
            "phase": "benchmark",
            "worker_id": _WORKER.get("worker_id"),
            "pid": os.getpid(),
            "chunk_index": chunk_index,
            "chunk_file": Path(audio_path).name,
            "failure_reason": str(exc).splitlines()[0],
        }
    except Exception as exc:
        return {
            "status": "error",
            "run_name": _WORKER["args"]["run_name"],
            "phase": "benchmark",
            "worker_id": _WORKER.get("worker_id"),
            "pid": os.getpid(),
            "chunk_index": chunk_index,
            "chunk_file": Path(audio_path).name,
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as fout:
        fout.write(json.dumps(row, default=str, sort_keys=True) + "\n")


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    benchmark_files: list[Path],
    ready_ms: float,
    wall_ms: float,
    status: str,
    failure_reason: str | None = None,
) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("phase") == "benchmark"]
    benchmark_rows = [row for row in rows if row.get("phase") == "benchmark"]
    totals = [float(row["total_ms"]) for row in ok_rows]
    processed_chunks = len(ok_rows)
    fields = [
        "run_name",
        "status",
        "failure_reason",
        "workers",
        "benchmark_chunk_count",
        "processed_chunk_count",
        "failed_chunk_count",
        "pool_ready_ms",
        "benchmark_wall_ms",
        "chunks_per_sec",
        "chunks_per_min",
        "mean_chunk_ms",
        "p50_chunk_ms",
        "p95_chunk_ms",
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
                "workers": args.workers,
                "benchmark_chunk_count": len(benchmark_files),
                "processed_chunk_count": processed_chunks,
                "failed_chunk_count": len([row for row in benchmark_rows if row.get("status") != "ok"]),
                "pool_ready_ms": round(ready_ms, 3),
                "benchmark_wall_ms": round(wall_ms, 3),
                "chunks_per_sec": round(processed_chunks / (wall_ms / 1000.0), 4)
                if wall_ms > 0
                else "",
                "chunks_per_min": round(processed_chunks / (wall_ms / 60000.0), 3)
                if wall_ms > 0
                else "",
                "mean_chunk_ms": round(statistics.mean(totals), 3) if totals else "",
                "p50_chunk_ms": round(statistics.median(totals), 3) if totals else "",
                "p95_chunk_ms": round(sorted(totals)[max(0, int(0.95 * (len(totals) - 1)))], 3)
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
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    args.run_name = args.run_name or f"pool-cascade-w{args.workers}"
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

    worker_args = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "model_id": args.model_id,
        "stage1_prompt": args.stage1_prompt,
        "stage2_prompt": args.stage2_prompt,
        "dtype_policy": args.dtype_policy,
        "compile_transformer": args.compile_transformer,
        "predict_spans": args.predict_spans,
        "adaptive_initial_candidates": args.adaptive_initial_candidates,
        "adaptive_max_candidates": args.adaptive_max_candidates,
        "adaptive_margin": args.adaptive_margin,
        "seed": args.seed,
        "disable_visual_ranker": args.disable_visual_ranker,
    }
    manifest = {
        "run_name": args.run_name,
        "model_id": args.model_id,
        "input_dir": str(args.input_dir),
        "warmup_file": warmup_file.name,
        "benchmark_files": [path.name for path in benchmark_files],
        "args": vars(args),
        "pool_model": "one process and one loaded model per worker",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, default=str, indent=2, sort_keys=True)
    )

    rows: list[dict[str, Any]] = []
    status = "ok"
    failure_reason = None
    ready_start = now_ms()
    wall_ms = 0.0

    ctx = mp.get_context("spawn")
    worker_counter = ctx.Value("i", 0)
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=init_worker,
            initargs=(worker_args, str(warmup_file), worker_counter),
        ) as executor:
            ready_futures = [executor.submit(worker_ready) for _ in range(args.workers)]
            for future in concurrent.futures.as_completed(ready_futures):
                row = future.result()
                rows.append(row)
                write_jsonl(metrics_path, row)
            ready_ms = now_ms() - ready_start

            wall_start = now_ms()
            futures = [
                executor.submit(
                    worker_process_chunk,
                    chunk_index,
                    str(audio_path),
                    args.seed + chunk_index * 10,
                )
                for chunk_index, audio_path in enumerate(benchmark_files, start=1)
            ]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                rows.append(row)
                write_jsonl(metrics_path, row)
                if row.get("status") != "ok" and status == "ok":
                    status = row["status"]
                    failure_reason = row.get("failure_reason", "")
            wall_ms = now_ms() - wall_start
    except concurrent.futures.process.BrokenProcessPool as exc:
        status = "error"
        failure_reason = f"BrokenProcessPool: {exc}"
        ready_ms = now_ms() - ready_start
        write_jsonl(
            metrics_path,
            {
                "status": status,
                "phase": "pool",
                "workers": args.workers,
                "failure_reason": failure_reason,
            },
        )
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        failure_reason = str(exc).splitlines()[0]
        ready_ms = now_ms() - ready_start
        write_jsonl(
            metrics_path,
            {
                "status": status,
                "phase": "pool",
                "workers": args.workers,
                "failure_reason": failure_reason,
            },
        )
    else:
        if "ready_ms" not in locals():
            ready_ms = now_ms() - ready_start

    write_summary(
        summary_path,
        args=args,
        rows=rows,
        benchmark_files=benchmark_files,
        ready_ms=ready_ms,
        wall_ms=wall_ms,
        status=status,
        failure_reason=failure_reason,
    )
    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
