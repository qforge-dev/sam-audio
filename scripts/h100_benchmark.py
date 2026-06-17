#!/usr/bin/env python3
"""Run SAM-Audio inference benchmarks on an H100-class machine.

This script is intentionally self-contained so optimization branches can be
compared with the same command line, the same inputs, and the same output format.
It is safe to run on non-H100 machines for syntax/path validation, but meaningful
numbers require CUDA and the gated SAM-Audio checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
import torchaudio


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
DTYPES = {
    "fp32": torch.float32,
    "tf32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--prompts-json", type=Path, default=Path("benchmarks/prompts.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="facebook/sam-audio-small-tv")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dtype-policy", choices=sorted(DTYPES), default="fp32")
    parser.add_argument("--reranking", default="4", help="Integer candidate count or 'adaptive'.")
    parser.add_argument("--predict-spans", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit-audios", type=int, default=None)
    parser.add_argument("--max-audio-seconds", type=float, default=35.0)
    parser.add_argument("--max-save-items", type=int, default=3)
    parser.add_argument("--fixed-midpoint", action="store_true")
    parser.add_argument("--cache-conditioning", action="store_true")
    parser.add_argument("--compile-transformer", choices=["none", "default", "reduce-overhead", "max-autotune"], default="none")
    parser.add_argument("--sdpa-backend", choices=["auto", "flash", "cudnn"], default="auto")
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


def load_prompts(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        prompts = data
    elif isinstance(data, dict):
        prompts = data.get("prompts", data.get("items", []))
    else:
        raise ValueError(f"Unsupported prompts file shape: {type(data)!r}")

    result = []
    for item in prompts:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            value = item.get("prompt") or item.get("description") or item.get("text")
            if value:
                result.append(str(value))
        else:
            raise ValueError(f"Unsupported prompt item: {item!r}")
    if not result:
        raise ValueError(f"No prompts found in {path}")
    return result


def list_audio_files(audio_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(
        path
        for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    if limit is not None:
        files = files[:limit]
    if not files:
        raise ValueError(f"No audio files found under {audio_dir}")
    return files


def duration_seconds(path: Path) -> float | None:
    try:
        info = torchaudio.info(str(path))
        if info.sample_rate and info.num_frames:
            return info.num_frames / info.sample_rate
    except Exception:
        return None
    return None


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


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
    model.transformer = torch.compile(model.transformer, mode=mode)


def sdpa_context(name: str):
    if name == "auto":
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as exc:
        raise RuntimeError("This PyTorch build does not expose torch.nn.attention.sdpa_kernel") from exc
    backend = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }[name]
    return sdpa_kernel(backend)


def unsupported_reason(model: Any, args: argparse.Namespace) -> str | None:
    if args.fixed_midpoint and not getattr(model, "_h100_fixed_midpoint_supported", False):
        return "fixed midpoint is not implemented on this branch"
    if args.cache_conditioning and not (
        hasattr(model, "prepare_audio") and hasattr(model, "separate_prepared")
    ):
        return "conditioning cache is not implemented on this branch"
    if args.reranking == "adaptive" and not hasattr(model, "separate_adaptive_rerank"):
        return "adaptive reranking is not implemented on this branch"
    return None


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as fout:
        fout.write(json.dumps(row, sort_keys=True) + "\n")


def write_summary(path: Path, metrics: list[dict[str, Any]]) -> None:
    completed = [m for m in metrics if m.get("status") == "ok"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        groups[row["run_name"]].append(row)

    fields = [
        "run_name",
        "count",
        "mean_total_ms",
        "p50_total_ms",
        "p95_total_ms",
        "mean_inference_ms",
        "max_peak_allocated_gb",
        "max_peak_reserved_gb",
    ]
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for run_name, rows in sorted(groups.items()):
            totals = [float(r["total_ms"]) for r in rows]
            infs = [float(r["inference_ms"]) for r in rows]
            p95 = sorted(totals)[max(0, int(0.95 * (len(totals) - 1)))]
            writer.writerow(
                {
                    "run_name": run_name,
                    "count": len(rows),
                    "mean_total_ms": round(statistics.mean(totals), 3),
                    "p50_total_ms": round(statistics.median(totals), 3),
                    "p95_total_ms": round(p95, 3),
                    "mean_inference_ms": round(statistics.mean(infs), 3),
                    "max_peak_allocated_gb": max(r.get("peak_allocated_gb") or 0 for r in rows),
                    "max_peak_reserved_gb": max(r.get("peak_reserved_gb") or 0 for r in rows),
                }
            )


def save_outputs(
    result: Any,
    sample_rate: int,
    output_dir: Path,
    audio_path: Path,
    prompts: list[str],
    max_save_items: int,
) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    limit = min(max_save_items, len(prompts))
    for index in range(limit):
        target = result.target[index] if isinstance(result.target, list) else result.target[index]
        residual = result.residual[index] if isinstance(result.residual, list) else result.residual[index]
        if target.ndim == 1:
            target = target.unsqueeze(0)
        if residual.ndim == 1:
            residual = residual.unsqueeze(0)
        stem = f"{audio_path.stem}_{index:02d}_{short_hash(prompts[index])}"
        target_path = output_dir / f"{stem}_target.wav"
        residual_path = output_dir / f"{stem}_residual.wav"
        torchaudio.save(str(target_path), target.detach().float().cpu(), sample_rate)
        torchaudio.save(str(residual_path), residual.detach().float().cpu(), sample_rate)
        artifacts.append(
            {
                "prompt": prompts[index],
                "target": str(target_path),
                "residual": str(residual_path),
            }
        )
    return artifacts


def make_separate_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"predict_spans": args.predict_spans}
    if args.reranking != "adaptive":
        kwargs["reranking_candidates"] = int(args.reranking)
    if args.fixed_midpoint:
        kwargs["ode_opt"] = {"method": "fixed_midpoint", "options": {"step_size": 2 / 32}}
    return kwargs


def candidate_count(args: argparse.Namespace) -> int:
    if args.reranking == "adaptive":
        return 1
    return int(args.reranking)


def run_one(
    *,
    model: Any,
    processor: Any,
    audio_path: Path,
    prompts: list[str],
    args: argparse.Namespace,
    dtype: torch.dtype,
    run_dir: Path,
    metrics_path: Path,
    run_index: int,
) -> dict[str, Any]:
    device = "cuda"
    duration = duration_seconds(audio_path)
    if duration is not None and duration > args.max_audio_seconds:
        row = {
            "status": "skipped",
            "reason": f"duration {duration:.3f}s exceeds max {args.max_audio_seconds:.3f}s",
            "audio_path": str(audio_path),
            "duration_seconds": duration,
            "run_name": args.run_name,
        }
        write_jsonl(metrics_path, row)
        return row

    set_seed(args.seed + run_index)
    total_start = now_ms()
    process_start = now_ms()
    batch = processor(audios=[str(audio_path)] * len(prompts), descriptions=prompts).to(device)
    if dtype != torch.float32:
        batch.audios = batch.audios.to(dtype)
    cuda_sync()
    process_ms = now_ms() - process_start

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    inference_start = now_ms()
    with sdpa_context(args.sdpa_backend):
        if args.cache_conditioning:
            handle = model.prepare_audio(
                batch,
                candidates=candidate_count(args),
                predict_spans=args.predict_spans,
            )
            result = model.separate_prepared(handle, prompts=prompts, **make_separate_kwargs(args))
        elif args.reranking == "adaptive":
            result = model.separate_adaptive_rerank(batch, predict_spans=args.predict_spans)
        else:
            result = model.separate(batch, **make_separate_kwargs(args))
    cuda_sync()
    inference_ms = now_ms() - inference_start

    save_start = now_ms()
    artifact_dir = run_dir / "artifacts" / audio_path.stem
    artifacts = save_outputs(
        result,
        processor.audio_sampling_rate,
        artifact_dir,
        audio_path,
        prompts,
        args.max_save_items,
    )
    save_ms = now_ms() - save_start
    total_ms = now_ms() - total_start

    peak_allocated = None
    peak_reserved = None
    if torch.cuda.is_available():
        peak_allocated = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
        peak_reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 4)

    row = {
        "status": "ok",
        "run_name": args.run_name,
        "model_id": args.model_id,
        "audio_path": str(audio_path),
        "duration_seconds": duration,
        "prompt_count": len(prompts),
        "dtype_policy": args.dtype_policy,
        "reranking": args.reranking,
        "predict_spans": args.predict_spans,
        "fixed_midpoint": args.fixed_midpoint,
        "cache_conditioning": args.cache_conditioning,
        "compile_transformer": args.compile_transformer,
        "sdpa_backend": args.sdpa_backend,
        "process_ms": round(process_ms, 3),
        "inference_ms": round(inference_ms, 3),
        "save_ms": round(save_ms, 3),
        "total_ms": round(total_ms, 3),
        "peak_allocated_gb": peak_allocated,
        "peak_reserved_gb": peak_reserved,
        "artifacts": artifacts,
    }
    write_jsonl(metrics_path, row)
    return row


def main() -> int:
    args = parse_args()
    args.run_name = args.run_name or (
        f"{args.dtype_policy}_rerank-{args.reranking}"
        f"{'_fixed' if args.fixed_midpoint else ''}"
        f"{'_cache' if args.cache_conditioning else ''}"
        f"{'_compile-' + args.compile_transformer if args.compile_transformer != 'none' else ''}"
        f"{'_sdpa-' + args.sdpa_backend if args.sdpa_backend != 'auto' else ''}"
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for meaningful H100 benchmarking")

    run_dir = args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.csv"
    metrics_path.unlink(missing_ok=True)

    prompts = load_prompts(args.prompts_json)
    audio_files = list_audio_files(args.audio_dir, args.limit_audios)
    dtype = configure_precision(args.dtype_policy)

    from sam_audio import SAMAudio, SAMAudioProcessor

    load_start = now_ms()
    model = SAMAudio.from_pretrained(args.model_id, proxies=None, resume_download=False).eval().cuda()
    if dtype != torch.float32:
        model = model.to(dtype)
        keep_rankers_fp32(model)
    compile_transformer_if_requested(model, args.compile_transformer)
    processor = SAMAudioProcessor.from_pretrained(args.model_id)
    load_ms = now_ms() - load_start

    manifest = {
        "run_name": args.run_name,
        "model_id": args.model_id,
        "audio_dir": str(args.audio_dir),
        "prompts_json": str(args.prompts_json),
        "audio_count": len(audio_files),
        "prompt_count": len(prompts),
        "load_ms": round(load_ms, 3),
        "args": vars(args),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    reason = unsupported_reason(model, args)
    if reason is not None:
        row = {
            "status": "skipped",
            "reason": reason,
            "run_name": args.run_name,
            "model_id": args.model_id,
            "args": vars(args),
        }
        write_jsonl(metrics_path, row)
        write_summary(summary_path, [row])
        print(json.dumps(row, indent=2, sort_keys=True))
        return 0

    for warmup_index in range(args.warmup):
        set_seed(args.seed + warmup_index)
        batch = processor(audios=[str(audio_files[0])] * len(prompts), descriptions=prompts).to("cuda")
        if dtype != torch.float32:
            batch.audios = batch.audios.to(dtype)
        with sdpa_context(args.sdpa_backend):
            if args.cache_conditioning:
                handle = model.prepare_audio(
                    batch,
                    candidates=candidate_count(args),
                    predict_spans=args.predict_spans,
                )
                model.separate_prepared(handle, **make_separate_kwargs(args))
            elif args.reranking == "adaptive":
                model.separate_adaptive_rerank(batch, predict_spans=args.predict_spans)
            else:
                model.separate(batch, **make_separate_kwargs(args))
        cuda_sync()

    metrics = []
    for index, audio_path in enumerate(audio_files):
        metrics.append(
            run_one(
                model=model,
                processor=processor,
                audio_path=audio_path,
                prompts=prompts,
                args=args,
                dtype=dtype,
                run_dir=run_dir,
                metrics_path=metrics_path,
                run_index=index,
            )
        )
    write_summary(summary_path, metrics)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
