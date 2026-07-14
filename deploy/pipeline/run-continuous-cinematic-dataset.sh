#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT=${SAM_CONTINUOUS_DEPLOY_ROOT:-/home/ubuntu/sam-audio-deploy}
PIPELINE_ROOT="$DEPLOY_ROOT/pipeline"
PIPELINE_PYTHON="$PIPELINE_ROOT/.venv/bin/python"
MODEL_PYTHON="$DEPLOY_ROOT/.venv/bin/python"
WHISPER_PYTHON=${SAM_CONTINUOUS_WHISPER_PYTHON:-/home/ubuntu/whisper-venv/bin/python}
M2D_REPO=${SAM_CONTINUOUS_M2D_REPO:-/home/ubuntu/m2d}
M2D_CHECKPOINT=${SAM_CONTINUOUS_M2D_CHECKPOINT:-$M2D_REPO/weights/m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d/weights_ep69it3124-0.47929.pth}
CLASS_LABELS=${SAM_CONTINUOUS_CLASS_LABELS:-/home/ubuntu/m2d-validation/metadata/class_labels_indices.csv}
ONTOLOGY=${SAM_CONTINUOUS_ONTOLOGY:-/home/ubuntu/m2d-validation/metadata/ontology.json}
WORKSPACE=${SAM_CONTINUOUS_WORKSPACE:-/home/ubuntu/cinematic-continuous-30s}
RUNS_DIR="$WORKSPACE/acquisition-runs"
BUCKET=${SAM_CONTINUOUS_S3_BUCKET:?SAM_CONTINUOUS_S3_BUCKET is required}
S3_PREFIX=${SAM_CONTINUOUS_S3_PREFIX:-cinematic-dialogue-dataset}
DOWNLOAD_WORKERS=${SAM_CONTINUOUS_DOWNLOAD_WORKERS:-8}
ACQUISITION_PRODUCERS=${SAM_CONTINUOUS_ACQUISITION_PRODUCERS:-2}
SEARCH_WORKERS=${SAM_CONTINUOUS_SEARCH_WORKERS:-8}
M2D_WORKERS=${SAM_CONTINUOUS_M2D_WORKERS:-1}
ASR_WORKERS=${SAM_CONTINUOUS_ASR_WORKERS:-1}
UPLOAD_CONCURRENCY=${SAM_CONTINUOUS_UPLOAD_CONCURRENCY:-10}
AUTOSCALE_ENABLED=${SAM_CONTINUOUS_AUTOSCALE_ENABLED:-true}
DOWNLOAD_MIN=${SAM_CONTINUOUS_DOWNLOAD_MIN:-2}
ASR_CONCURRENCY_MIN=${SAM_CONTINUOUS_ASR_CONCURRENCY_MIN:-1}
ASR_CONCURRENCY_MAX=${SAM_CONTINUOUS_ASR_CONCURRENCY_MAX:-2}
ASR_CPU_THREADS=${SAM_CONTINUOUS_ASR_CPU_THREADS:-4}
CPU_LOW=${SAM_CONTINUOUS_CPU_LOW:-55}
CPU_HIGH=${SAM_CONTINUOUS_CPU_HIGH:-85}
CPU_EMERGENCY=${SAM_CONTINUOUS_CPU_EMERGENCY:-95}
M2D_BACKLOG_HIGH=${SAM_CONTINUOUS_M2D_BACKLOG_HIGH:-64}
ASR_BACKLOG_HIGH=${SAM_CONTINUOUS_ASR_BACKLOG_HIGH:-8}
# The second faster-whisper task shares the already-loaded model and needs far less
# than 12 GiB. Keep an 8 GiB guard for concurrent M2D source scans and model APIs.
GPU_RESERVE_MB=${SAM_CONTINUOUS_GPU_RESERVE_MB:-8000}
AUTOSCALE_COOLDOWN_SECONDS=${SAM_CONTINUOUS_AUTOSCALE_COOLDOWN_SECONDS:-60}
AUTOSCALE_INTERVAL_SECONDS=${SAM_CONTINUOUS_AUTOSCALE_INTERVAL_SECONDS:-10}
BASE_CLIPS_PER_VIDEO=${SAM_CONTINUOUS_BASE_CLIPS_PER_VIDEO:-16}
SOURCE_CONTENT_MINUTES_PER_HOUR=${SAM_CONTINUOUS_SOURCE_CONTENT_MINUTES_PER_HOUR:-10}
MAX_DURATION_SCALED_CLIPS_PER_VIDEO=${SAM_CONTINUOUS_MAX_DURATION_SCALED_CLIPS_PER_VIDEO:-60}
SOURCE_SCAN_ENABLED=${SAM_CONTINUOUS_SOURCE_SCAN_ENABLED:-true}
SOURCE_SCAN_BATCH_SIZE=${SAM_CONTINUOUS_SOURCE_SCAN_BATCH_SIZE:-128}
SOURCE_ASR_PROBE_MODE=${SAM_CONTINUOUS_SOURCE_ASR_PROBE_MODE:-enforce}
SOURCE_ASR_PROBE_TIMEOUT=${SAM_CONTINUOUS_SOURCE_ASR_PROBE_TIMEOUT:-120}
STAGED_ACQUISITION=${SAM_CONTINUOUS_STAGED_ACQUISITION:-false}

for value in "$DOWNLOAD_WORKERS" "$ACQUISITION_PRODUCERS" "$SEARCH_WORKERS" "$M2D_WORKERS" "$ASR_WORKERS" "$UPLOAD_CONCURRENCY"; do
  if (( value < 1 )); then
    echo "All worker counts must be positive" >&2
    exit 2
  fi
done
if (( DOWNLOAD_MIN < 1 || DOWNLOAD_MIN > DOWNLOAD_WORKERS )); then
  echo "Download bounds must satisfy 1 <= min <= max" >&2
  exit 2
fi
if (( ASR_CONCURRENCY_MIN < 1 || ASR_CONCURRENCY_MIN > ASR_CONCURRENCY_MAX )); then
  echo "ASR concurrency bounds must satisfy 1 <= min <= max" >&2
  exit 2
fi

export PYTHONPATH="$PIPELINE_ROOT/src"
export HF_HOME=${HF_HOME:-/home/ubuntu/.cache/huggingface}
mkdir -p "$RUNS_DIR" "$WORKSPACE/raw-audio" "$WORKSPACE/accepted/audio" \
  "$WORKSPACE/m2d-validation" "$WORKSPACE/asr-validation" \
  "$WORKSPACE/source-scans" "$WORKSPACE/source-asr-probe-requests" \
  "$WORKSPACE/source-asr-probe-results"
cd "$PIPELINE_ROOT"

configure_args=(
  --workspace "$WORKSPACE"
  --download-workers "$DOWNLOAD_WORKERS"
  --acquisition-producers "$ACQUISITION_PRODUCERS"
  --m2d-workers "$M2D_WORKERS"
  --asr-workers "$ASR_WORKERS"
  --upload-concurrency "$UPLOAD_CONCURRENCY"
  --download-min "$DOWNLOAD_MIN"
  --asr-concurrency-min "$ASR_CONCURRENCY_MIN"
  --asr-concurrency-max "$ASR_CONCURRENCY_MAX"
  --base-clips-per-video "$BASE_CLIPS_PER_VIDEO"
  --source-content-minutes-per-hour "$SOURCE_CONTENT_MINUTES_PER_HOUR"
  --max-duration-scaled-clips-per-video "$MAX_DURATION_SCALED_CLIPS_PER_VIDEO"
  --source-scan-batch-size "$SOURCE_SCAN_BATCH_SIZE"
)
if [[ "$SOURCE_SCAN_ENABLED" == "true" ]]; then
  configure_args+=(--source-scan-enabled)
fi
if [[ "$AUTOSCALE_ENABLED" == "true" ]]; then
  configure_args+=(--autoscaling-enabled)
fi
"$PIPELINE_PYTHON" -m sam_audio_pipeline.continuous_dataset configure \
  "${configure_args[@]}"

for ((index=0; index<M2D_WORKERS; index++)); do
  touch "$WORKSPACE/m2d-validation/worker-$index.jsonl"
done

heartbeat_loop() {
  local worker=$1
  while true; do
    "$PIPELINE_PYTHON" -m sam_audio_pipeline.continuous_dataset heartbeat \
      --workspace "$WORKSPACE" --worker "$worker" \
      --follow --interval-seconds 10 >/dev/null 2>&1 || true
    sleep 5
  done
}

restart_worker() {
  local worker=$1
  shift
  while true; do
    "$@" || true
    "$PIPELINE_PYTHON" -m sam_audio_pipeline.continuous_dataset heartbeat \
      --workspace "$WORKSPACE" --worker "$worker" --state restarting \
      >/dev/null 2>&1 || true
    sleep 5
  done
}

download_forever() {
  local producer_index=$1
  local seed_file="$WORKSPACE/next-seed-$producer_index"
  local legacy_seed_file="$WORKSPACE/next-seed"
  local workers_per_producer=$(((DOWNLOAD_WORKERS + ACQUISITION_PRODUCERS - 1) / ACQUISITION_PRODUCERS))
  local base_seed
  local seed
  base_seed=$(test -s "$legacy_seed_file" && tr -dc '0-9' < "$legacy_seed_file" || date -u +%Y%m%d)
  seed=$(test -s "$seed_file" && tr -dc '0-9' < "$seed_file" || echo $((base_seed + producer_index)))
  while true; do
    local run_dir="$RUNS_DIR/run-$seed"
    local next_seed=$((seed + ACQUISITION_PRODUCERS))
    local next_run_dir="$RUNS_DIR/run-$next_seed"
    local prefetch_pid
    local scan_args=()
    if [[ "$SOURCE_SCAN_ENABLED" == "true" ]]; then
      scan_args=(
        --scan-before-extract
        --source-scan-cache "$WORKSPACE/source-scans"
        --catalog "$WORKSPACE/catalog.sqlite3"
        --m2d-repo "$M2D_REPO"
        --m2d-checkpoint "$M2D_CHECKPOINT"
        --m2d-class-labels "$CLASS_LABELS"
        --m2d-ontology "$ONTOLOGY"
        --m2d-device cuda
        --m2d-batch-size "$SOURCE_SCAN_BATCH_SIZE"
        --source-asr-probe-mode "$SOURCE_ASR_PROBE_MODE"
        --source-asr-probe-requests "$WORKSPACE/source-asr-probe-requests"
        --source-asr-probe-results "$WORKSPACE/source-asr-probe-results"
        --source-asr-probe-timeout "$SOURCE_ASR_PROBE_TIMEOUT"
        --yt-dlp-python "$PIPELINE_PYTHON"
      )
    fi
    mkdir -p "$next_run_dir"
    nice -n 15 "$PIPELINE_PYTHON" -m sam_audio_pipeline.youtube_random \
      --output "$next_run_dir" \
      --source dailymotion \
      --profile cinematic \
      --clip-seconds 30 \
      --total 2000 \
      --seed "$next_seed" \
      --clips-per-video "$BASE_CLIPS_PER_VIDEO" \
      --source-content-minutes-per-hour "$SOURCE_CONTENT_MINUTES_PER_HOUR" \
      --max-clips-per-video "$MAX_DURATION_SCALED_CLIPS_PER_VIDEO" \
      --query-count 500 \
      --results-per-query 100 \
      --search-workers "$SEARCH_WORKERS" \
      --candidate-multiplier 1.8 \
      --discover-only >"$next_run_dir/prefetch.log" 2>&1 &
    prefetch_pid=$!
    nice -n 10 "$MODEL_PYTHON" -m sam_audio_pipeline.youtube_random \
      --output "$run_dir" \
      --source dailymotion \
      --profile cinematic \
      --clip-seconds 30 \
      --total 2000 \
      --seed "$seed" \
      --clips-per-video "$BASE_CLIPS_PER_VIDEO" \
      --source-content-minutes-per-hour "$SOURCE_CONTENT_MINUTES_PER_HOUR" \
      --max-clips-per-video "$MAX_DURATION_SCALED_CLIPS_PER_VIDEO" \
      --query-count 500 \
      --results-per-query 100 \
      --search-workers "$SEARCH_WORKERS" \
      --download-workers "$workers_per_producer" \
      --worker-limit-file "$WORKSPACE/autoscale-control.json" \
      --candidate-multiplier 1.8 \
      --max-attempts 40000 \
      "${scan_args[@]}" || true
    wait "$prefetch_pid" || true
    seed=$next_seed
    printf '%s\n' "$seed" > "$seed_file.tmp"
    mv "$seed_file.tmp" "$seed_file"
  done
}

shutdown() {
  trap - TERM INT EXIT
  jobs -pr | xargs -r kill 2>/dev/null || true
  # systemd uses KillMode=control-group, so grandchildren are terminated too.
  # Waiting on the shell job table here can spin on stale subprocess PIDs and
  # force every routine restart to hit TimeoutStopSec.
  exit 0
}
trap shutdown TERM INT EXIT

heartbeat_loop downloader &
if [[ "$STAGED_ACQUISITION" != "true" ]]; then
  for ((index=0; index<ACQUISITION_PRODUCERS; index++)); do
    restart_worker downloader download_forever "$index" &
  done
fi

heartbeat_loop promoter &
restart_worker promoter "$PIPELINE_PYTHON" -m sam_audio_pipeline.continuous_dataset \
  promote --runs-dir "$RUNS_DIR" --workspace "$WORKSPACE" --follow &

for ((index=0; index<M2D_WORKERS; index++)); do
  heartbeat_loop "m2d-$index" &
  restart_worker "m2d-$index" "$MODEL_PYTHON" \
    -m sam_audio_pipeline.m2d_validator score \
    --input-dir "$WORKSPACE/raw-audio" \
    --output "$WORKSPACE/m2d-validation/worker-$index.jsonl" \
    --m2d-repo "$M2D_REPO" \
    --checkpoint "$M2D_CHECKPOINT" \
    --class-labels "$CLASS_LABELS" \
    --ontology "$ONTOLOGY" \
    --m2d-commit 3d0c4de9447c404a8d3f9f37e04f53bc902e09b3 \
    --require-cinematic-mix --follow --poll-seconds 2 \
    --shard-index "$index" --shard-count "$M2D_WORKERS" &
done

for ((index=0; index<ASR_WORKERS; index++)); do
  probe_args=()
  if (( index == 0 )); then
    probe_args=(
      --probe-requests-dir "$WORKSPACE/source-asr-probe-requests"
      --probe-results-dir "$WORKSPACE/source-asr-probe-results"
    )
  fi
  heartbeat_loop "asr-$index" &
  restart_worker "asr-$index" "$WHISPER_PYTHON" \
    -m sam_audio_pipeline.m2d_validator asr-score \
    --input-dir "$WORKSPACE/raw-audio" \
    --output "$WORKSPACE/asr-validation/worker-$index.jsonl" \
    --m2d-results-dir "$WORKSPACE/m2d-validation" \
    --require-cinematic-mix \
    --model small \
    --download-root /home/ubuntu/.cache/huggingface/faster-whisper \
    --max-inference-workers "$ASR_CONCURRENCY_MAX" \
    --cpu-threads "$ASR_CPU_THREADS" \
    --autoscale-control "$WORKSPACE/autoscale-control.json" \
    "${probe_args[@]}" \
    --follow --poll-seconds 2 \
    --shard-index "$index" --shard-count "$ASR_WORKERS" &
done

if [[ "$AUTOSCALE_ENABLED" == "true" ]]; then
  restart_worker autoscaler "$PIPELINE_PYTHON" \
    -m sam_audio_pipeline.continuous_dataset autoscale \
    --workspace "$WORKSPACE" \
    --download-min "$DOWNLOAD_MIN" --download-max "$DOWNLOAD_WORKERS" \
    --asr-min "$ASR_CONCURRENCY_MIN" --asr-max "$ASR_CONCURRENCY_MAX" \
    --cpu-low "$CPU_LOW" --cpu-high "$CPU_HIGH" \
    --cpu-emergency "$CPU_EMERGENCY" \
    --m2d-backlog-high "$M2D_BACKLOG_HIGH" \
    --asr-backlog-high "$ASR_BACKLOG_HIGH" \
    --gpu-reserve-mb "$GPU_RESERVE_MB" \
    --cooldown-seconds "$AUTOSCALE_COOLDOWN_SECONDS" \
    --interval-seconds "$AUTOSCALE_INTERVAL_SECONDS" --follow &
fi

heartbeat_loop assembler &
restart_worker assembler "$PIPELINE_PYTHON" -m sam_audio_pipeline.continuous_dataset \
  assemble --workspace "$WORKSPACE" \
  --max-clips-per-video "$BASE_CLIPS_PER_VIDEO" \
  --source-content-minutes-per-hour "$SOURCE_CONTENT_MINUTES_PER_HOUR" \
  --max-duration-scaled-clips-per-video "$MAX_DURATION_SCALED_CLIPS_PER_VIDEO" \
  --follow &

heartbeat_loop snapshot_publisher &
restart_worker snapshot_publisher "$PIPELINE_PYTHON" \
  -m sam_audio_pipeline.continuous_dataset publish-due \
  --workspace "$WORKSPACE" --bucket "$BUCKET" --prefix "$S3_PREFIX" \
  --snapshot-size 5000 --upload-concurrency "$UPLOAD_CONCURRENCY" \
  --follow --poll-seconds 30 &

wait -n
exit 1
