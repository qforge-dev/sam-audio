#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 AUDIO_DIR OUT_ROOT [PROMPTS_JSON]" >&2
  exit 2
fi

AUDIO_DIR="$1"
OUT_ROOT="$2"
PROMPTS_JSON="${3:-benchmarks/prompts.json}"
MODEL_ID="${SAM_AUDIO_MODEL_ID:-facebook/sam-audio-small-tv}"
SEED="${SAM_AUDIO_BENCH_SEED:-1234}"
WARMUP="${SAM_AUDIO_BENCH_WARMUP:-1}"
LIMIT_AUDIOS="${SAM_AUDIO_BENCH_LIMIT_AUDIOS:-}"
MAX_SAVE_ITEMS="${SAM_AUDIO_BENCH_MAX_SAVE_ITEMS:-3}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${OUT_ROOT%/}/${STAMP}"
mkdir -p "$RUN_DIR"

COMMON=(
  --audio-dir "$AUDIO_DIR"
  --prompts-json "$PROMPTS_JSON"
  --out-dir "$RUN_DIR"
  --model-id "$MODEL_ID"
  --seed "$SEED"
  --warmup "$WARMUP"
  --max-save-items "$MAX_SAVE_ITEMS"
)

if [[ -n "$LIMIT_AUDIOS" ]]; then
  COMMON+=(--limit-audios "$LIMIT_AUDIOS")
fi

run_case() {
  local name="$1"
  shift
  echo "==> ${name}"
  python scripts/h100_benchmark.py "${COMMON[@]}" --run-name "$name" "$@"
}

run_case baseline-fp32 --dtype-policy fp32 --reranking 4 --predict-spans
run_case tf32 --dtype-policy tf32 --reranking 4 --predict-spans
run_case rerank-1 --dtype-policy fp32 --reranking 1 --predict-spans
run_case rerank-8 --dtype-policy fp32 --reranking 8 --predict-spans
run_case fixed-midpoint --dtype-policy fp32 --reranking 4 --predict-spans --fixed-midpoint
run_case fixed-midpoint-cache --dtype-policy fp32 --reranking 4 --predict-spans --fixed-midpoint --cache-conditioning
run_case compile-default --dtype-policy fp32 --reranking 4 --predict-spans --compile-transformer default
run_case compile-reduce-overhead --dtype-policy fp32 --reranking 4 --predict-spans --compile-transformer reduce-overhead
run_case compile-max-autotune --dtype-policy fp32 --reranking 4 --predict-spans --compile-transformer max-autotune
run_case sdpa-flash --dtype-policy fp32 --reranking 4 --predict-spans --sdpa-backend flash
run_case adaptive-rerank --dtype-policy fp32 --reranking adaptive --predict-spans

python - <<'PY' "$RUN_DIR"
from pathlib import Path
import csv
import sys

root = Path(sys.argv[1])
rows = []
for summary in root.glob("*/summary.csv"):
    with summary.open() as fin:
        for row in csv.DictReader(fin):
            rows.append(row)

out = root / "combined_summary.csv"
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
with out.open("w", newline="") as fout:
    writer = csv.DictWriter(fout, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
print(f"Wrote {out}")
PY

echo "Benchmark run directory: $RUN_DIR"
