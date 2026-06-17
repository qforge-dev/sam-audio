from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
import torchaudio


DTYPES = {
    "tf32": torch.float32,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def configure_precision(policy: str) -> torch.dtype:
    dtype = DTYPES[policy]
    try:
        torch.backends.fp32_precision = "tf32" if policy == "tf32" else "ieee"
        torch.backends.cuda.matmul.fp32_precision = (
            "tf32" if policy == "tf32" else "ieee"
        )
        torch.backends.cudnn.fp32_precision = "tf32" if policy == "tf32" else "ieee"
    except Exception:
        torch.backends.cuda.matmul.allow_tf32 = policy == "tf32"
        torch.backends.cudnn.allow_tf32 = policy == "tf32"
    torch.set_float32_matmul_precision("high" if policy == "tf32" else "highest")
    return dtype


def save_wav(path: Path, wav: torch.Tensor, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().float().cpu()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(str(path), wav, sample_rate)


def load_audio_tensor(path: Path, sample_rate: int, pin_memory: bool) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if pin_memory and torch.cuda.is_available():
        wav = wav.pin_memory()
    return wav


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in path.stem)


def write_jsonl(path: Path, row: dict[str, Any]):
    with path.open("a") as f:
        f.write(json.dumps(row, default=str, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-active-requests", type=int, default=16)
    parser.add_argument("--max-benchmark-files", type=int, default=None)
    parser.add_argument("--preprocess-workers", type=int, default=12)
    parser.add_argument("--postprocess-workers", type=int, default=4)
    parser.add_argument("--input-decode-workers", type=int, default=0)
    parser.add_argument("--artifact-writer-workers", type=int, default=0)
    parser.add_argument("--predecode-inputs", action="store_true")
    parser.add_argument("--async-artifacts", action="store_true")
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--dtype-policy", choices=sorted(DTYPES), default="tf32")
    parser.add_argument("--rankers-fp32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage1-steps", type=int, default=16)
    parser.add_argument("--stage2-steps", type=int, default=16)
    parser.add_argument("--stage1-initial-candidates", type=int, default=4)
    parser.add_argument("--stage1-max-candidates", type=int, default=8)
    parser.add_argument("--stage1-margin", type=float, default=0.05)
    parser.add_argument("--stage2-initial-candidates", type=int, default=4)
    parser.add_argument("--stage2-max-candidates", type=int, default=8)
    parser.add_argument("--stage2-margin", type=float, default=0.05)
    parser.add_argument(
        "--compile-transformer",
        choices=["none", "default", "reduce-overhead"],
        default="none",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    dtype = configure_precision(args.dtype_policy)

    files = sorted(path for path in args.input_dir.glob("*.mp3") if path.is_file())
    if len(files) < 2:
        raise ValueError("Need at least one warmup MP3 and one benchmark MP3")
    warmup_file = files[0]
    benchmark_files = files[1:]
    if args.max_benchmark_files is not None:
        benchmark_files = benchmark_files[: args.max_benchmark_files]

    run_dir = args.out_dir / args.run_name
    artifact_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.csv"
    metrics_path.unlink(missing_ok=True)

    from sam_audio import (
        ContinuousBatcherConfig,
        ContinuousSAMAudioBatcher,
        SAMAudio,
        SAMAudioProcessor,
    )

    load_start = now_ms()
    model = SAMAudio.from_pretrained(
        args.model_id,
        proxies=None,
        resume_download=False,
        visual_ranker=None,
    ).eval().cuda()
    if dtype != torch.float32:
        model = model.to(dtype)
        if args.rankers_fp32:
            for name in ("text_ranker", "visual_ranker"):
                ranker = getattr(model, name, None)
                if ranker is not None:
                    ranker.float()
    if args.compile_transformer != "none":
        if hasattr(model, "compile_h100"):
            model.compile_h100(mode=args.compile_transformer)
        else:
            model.transformer = torch.compile(model.transformer, mode=args.compile_transformer)
    processor = SAMAudioProcessor.from_pretrained(args.model_id)
    load_ms = now_ms() - load_start

    decode_futures = {}
    decode_start = now_ms()
    decode_pool = None
    if args.predecode_inputs:
        decode_workers = args.input_decode_workers or min(8, len(files))
        decode_pool = ThreadPoolExecutor(
            max_workers=max(1, decode_workers),
            thread_name_prefix="sam-audio-input-decode",
        )
        for path in files:
            decode_futures[path] = decode_pool.submit(
                load_audio_tensor,
                path,
                processor.audio_sampling_rate,
                True,
            )

    config = ContinuousBatcherConfig(
        max_batch_size=1,
        max_active_requests=args.max_active_requests,
        max_queue_size=128,
        preprocess_workers=args.preprocess_workers,
        postprocess_workers=args.postprocess_workers,
        fixed_midpoint_steps=max(args.stage1_steps, args.stage2_steps),
        predict_spans=True,
        initial_candidates=4,
        max_candidates=8,
        margin=0.05,
        dtype=dtype,
        pin_memory=True,
        non_blocking_transfer=True,
    )

    per_stage = {
        "stage1": {
            "description": "music soundtrack",
            "fixed_midpoint_steps": args.stage1_steps,
            "initial_candidates": args.stage1_initial_candidates,
            "max_candidates": args.stage1_max_candidates,
            "margin": args.stage1_margin,
        },
        "stage2": {
            "description": "human voices",
            "fixed_midpoint_steps": args.stage2_steps,
            "initial_candidates": args.stage2_initial_candidates,
            "max_candidates": args.stage2_max_candidates,
            "margin": args.stage2_margin,
        },
    }
    manifest = {
        "run_name": args.run_name,
        "model_id": args.model_id,
        "warmup_file": warmup_file.name,
        "benchmark_files": [p.name for p in benchmark_files],
        "load_ms": round(load_ms, 3),
        "base_config": {
            k: str(v)
            for k, v in config.__dict__.items()
            if k != "completion_callback"
        },
        "per_stage": per_stage,
        "compile_transformer": args.compile_transformer,
        "dtype_policy": args.dtype_policy,
        "model_dtype": str(dtype),
        "rankers_fp32": args.rankers_fp32,
        "predecode_inputs": args.predecode_inputs,
        "input_decode_workers": args.input_decode_workers,
        "async_artifacts": args.async_artifacts,
        "artifact_writer_workers": args.artifact_writer_workers,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    rows = []
    artifact_pool = None
    artifact_futures = []
    if args.async_artifacts:
        writer_workers = args.artifact_writer_workers or 4
        artifact_pool = ThreadPoolExecutor(
            max_workers=max(1, writer_workers),
            thread_name_prefix="sam-audio-artifact-writer",
        )

    def get_audio(path: Path) -> str | torch.Tensor:
        if not args.predecode_inputs:
            return str(path)
        return decode_futures[path].result()

    def queue_wav(path: Path, wav: torch.Tensor):
        if args.skip_artifacts:
            return
        if artifact_pool is None:
            save_wav(path, wav, processor.audio_sampling_rate)
            return
        artifact_futures.append(
            artifact_pool.submit(save_wav, path, wav, processor.audio_sampling_rate)
        )

    with ContinuousSAMAudioBatcher(model, processor, config) as batcher:
        warm_start = now_ms()
        warm = batcher.separate_cascade(
            audio=get_audio(warmup_file),
            stage1_description=per_stage["stage1"]["description"],
            stage2_description=per_stage["stage2"]["description"],
            seed=1234,
            stage1_fixed_midpoint_steps=args.stage1_steps,
            stage2_fixed_midpoint_steps=args.stage2_steps,
            stage1_initial_candidates=args.stage1_initial_candidates,
            stage1_max_candidates=args.stage1_max_candidates,
            stage1_margin=args.stage1_margin,
            stage2_initial_candidates=args.stage2_initial_candidates,
            stage2_max_candidates=args.stage2_max_candidates,
            stage2_margin=args.stage2_margin,
            timeout=args.timeout,
        )
        warm_dir = artifact_dir / "warmup" / f"0000_{safe_stem(warmup_file)}"
        queue_wav(
            warm_dir / "stage1_music_soundtrack_target.wav",
            warm.stage1.target[0],
        )
        queue_wav(
            warm_dir / "stage1_music_soundtrack_residual.wav",
            warm.stage1.residual[0],
        )
        queue_wav(
            warm_dir / "stage2_human_voices_from_music_residual_target.wav",
            warm.stage2.target[0],
        )
        queue_wav(
            warm_dir / "stage2_human_voices_from_music_residual_residual.wav",
            warm.stage2.residual[0],
        )
        warm_ms = now_ms() - warm_start
        write_jsonl(
            metrics_path,
            {
                "status": "ok",
                "phase": "warmup",
                "chunk_file": warmup_file.name,
                "total_ms": round(warm_ms, 3),
            },
        )

        torch.cuda.reset_peak_memory_stats()
        wall_start = now_ms()
        submit_wait_ms = 0.0
        futures = {}
        submit_times = {}
        for idx, audio_path in enumerate(benchmark_files, start=1):
            wait_start = now_ms()
            audio = get_audio(audio_path)
            submit_wait_ms += now_ms() - wait_start
            submit_times[idx] = now_ms()
            fut = batcher.submit_cascade(
                audio=audio,
                stage1_description=per_stage["stage1"]["description"],
                stage2_description=per_stage["stage2"]["description"],
                seed=1234 + idx * 10,
                stage1_fixed_midpoint_steps=args.stage1_steps,
                stage2_fixed_midpoint_steps=args.stage2_steps,
                stage1_initial_candidates=args.stage1_initial_candidates,
                stage1_max_candidates=args.stage1_max_candidates,
                stage1_margin=args.stage1_margin,
                stage2_initial_candidates=args.stage2_initial_candidates,
                stage2_max_candidates=args.stage2_max_candidates,
                stage2_margin=args.stage2_margin,
            )
            futures[fut] = (idx, audio_path)

        for future in as_completed(futures, timeout=args.timeout):
            idx, audio_path = futures[future]
            item_dir = artifact_dir / "benchmark" / f"{idx:04d}_{safe_stem(audio_path)}"
            row = {
                "phase": "benchmark",
                "chunk_index": idx,
                "chunk_file": audio_path.name,
            }
            try:
                result = future.result(timeout=1)
                queue_wav(
                    item_dir / "stage1_music_soundtrack_target.wav",
                    result.stage1.target[0],
                )
                queue_wav(
                    item_dir / "stage1_music_soundtrack_residual.wav",
                    result.stage1.residual[0],
                )
                queue_wav(
                    item_dir / "stage2_human_voices_from_music_residual_target.wav",
                    result.stage2.target[0],
                )
                queue_wav(
                    item_dir / "stage2_human_voices_from_music_residual_residual.wav",
                    result.stage2.residual[0],
                )
                row.update(
                    {
                        "status": "ok",
                        "total_ms": round(now_ms() - submit_times[idx], 3),
                        "artifacts_dir": str(item_dir),
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "status": "error",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "total_ms": round(now_ms() - submit_times[idx], 3),
                    }
                )
            rows.append(row)
            write_jsonl(metrics_path, row)
        wall_ms = now_ms() - wall_start
        stats = batcher.metrics()

    artifact_wait_start = now_ms()
    for future in artifact_futures:
        future.result()
    artifact_write_wait_ms = now_ms() - artifact_wait_start
    if artifact_pool is not None:
        artifact_pool.shutdown(wait=True)
    if decode_pool is not None:
        decode_pool.shutdown(wait=True)
    input_decode_wall_ms = now_ms() - decode_start if args.predecode_inputs else 0.0

    ok = [row for row in rows if row.get("status") == "ok"]
    totals = [float(row["total_ms"]) for row in ok]
    fields = [
        "run_name",
        "status",
        "failure_reason",
        "benchmark_chunk_count",
        "processed_chunk_count",
        "stage1_steps",
        "stage2_steps",
        "stage1_initial_candidates",
        "stage1_max_candidates",
        "stage1_margin",
        "stage2_initial_candidates",
        "stage2_max_candidates",
        "stage2_margin",
        "dtype_policy",
        "predecode_inputs",
        "input_decode_workers",
        "input_decode_wall_ms",
        "benchmark_submit_wait_ms",
        "async_artifacts",
        "artifact_writer_workers",
        "artifact_write_wait_ms",
        "load_ms",
        "warmup_ms",
        "benchmark_wall_ms",
        "chunks_per_min",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "gpu_batches",
        "generation_steps",
        "gpu_ready_starved",
        "preprocess_ms",
        "prepare_ms",
        "step_ms",
        "decode_ms",
        "score_ms",
        "postprocess_ms",
    ]
    status = "ok" if len(ok) == len(benchmark_files) else "error"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "run_name": args.run_name,
                "status": status,
                "failure_reason": "" if status == "ok" else "one or more request failures",
                "benchmark_chunk_count": len(benchmark_files),
                "processed_chunk_count": len(ok),
                "stage1_steps": args.stage1_steps,
                "stage2_steps": args.stage2_steps,
                "stage1_initial_candidates": args.stage1_initial_candidates,
                "stage1_max_candidates": args.stage1_max_candidates,
                "stage1_margin": args.stage1_margin,
                "stage2_initial_candidates": args.stage2_initial_candidates,
                "stage2_max_candidates": args.stage2_max_candidates,
                "stage2_margin": args.stage2_margin,
                "dtype_policy": args.dtype_policy,
                "predecode_inputs": args.predecode_inputs,
                "input_decode_workers": args.input_decode_workers,
                "input_decode_wall_ms": round(input_decode_wall_ms, 3),
                "benchmark_submit_wait_ms": round(submit_wait_ms, 3),
                "async_artifacts": args.async_artifacts,
                "artifact_writer_workers": args.artifact_writer_workers,
                "artifact_write_wait_ms": round(artifact_write_wait_ms, 3),
                "load_ms": round(load_ms, 3),
                "warmup_ms": round(warm_ms, 3),
                "benchmark_wall_ms": round(wall_ms, 3),
                "chunks_per_min": round(len(ok) / (wall_ms / 60000.0), 3)
                if wall_ms > 0
                else "",
                "mean_latency_ms": round(statistics.mean(totals), 3) if totals else "",
                "p50_latency_ms": round(statistics.median(totals), 3) if totals else "",
                "p95_latency_ms": round(
                    sorted(totals)[max(0, int(0.95 * (len(totals) - 1)))], 3
                )
                if totals
                else "",
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
                "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 4),
                "gpu_batches": stats.gpu_batches,
                "generation_steps": stats.generation_steps,
                "gpu_ready_starved": stats.gpu_ready_starved,
                "preprocess_ms": round(stats.preprocess_ms, 3),
                "prepare_ms": round(stats.prepare_ms, 3),
                "step_ms": round(stats.step_ms, 3),
                "decode_ms": round(stats.decode_ms, 3),
                "score_ms": round(stats.score_ms, 3),
                "postprocess_ms": round(stats.postprocess_ms, 3),
            }
        )
    print(summary_path.read_text())
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
