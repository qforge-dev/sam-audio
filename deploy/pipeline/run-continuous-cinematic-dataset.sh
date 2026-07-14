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
SEARCH_WORKERS=${SAM_CONTINUOUS_SEARCH_WORKERS:-8}
M2D_WORKERS=${SAM_CONTINUOUS_M2D_WORKERS:-1}
ASR_WORKERS=${SAM_CONTINUOUS_ASR_WORKERS:-1}
UPLOAD_CONCURRENCY=${SAM_CONTINUOUS_UPLOAD_CONCURRENCY:-10}

for value in "$DOWNLOAD_WORKERS" "$SEARCH_WORKERS" "$M2D_WORKERS" "$ASR_WORKERS" "$UPLOAD_CONCURRENCY"; do
  if (( value < 1 )); then
    echo "All worker counts must be positive" >&2
    exit 2
  fi
done

export PYTHONPATH="$PIPELINE_ROOT/src"
export HF_HOME=${HF_HOME:-/home/ubuntu/.cache/huggingface}
mkdir -p "$RUNS_DIR" "$WORKSPACE/raw-audio" "$WORKSPACE/accepted/audio" \
  "$WORKSPACE/m2d-validation" "$WORKSPACE/asr-validation"
cd "$PIPELINE_ROOT"

"$PIPELINE_PYTHON" -m sam_audio_pipeline.continuous_dataset configure \
  --workspace "$WORKSPACE" \
  --download-workers "$DOWNLOAD_WORKERS" \
  --m2d-workers "$M2D_WORKERS" \
  --asr-workers "$ASR_WORKERS" \
  --upload-concurrency "$UPLOAD_CONCURRENCY"

for ((index=0; index<M2D_WORKERS; index++)); do
  touch "$WORKSPACE/m2d-validation/worker-$index.jsonl"
done

heartbeat_loop() {
  local worker=$1
  while true; do
    "$PIPELINE_PYTHON" -m sam_audio_pipeline.continuous_dataset heartbeat \
      --workspace "$WORKSPACE" --worker "$worker" >/dev/null 2>&1 || true
    sleep 10
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
  local seed_file="$WORKSPACE/next-seed"
  local seed
  seed=$(test -s "$seed_file" && tr -dc '0-9' < "$seed_file" || date -u +%Y%m%d)
  while true; do
    local run_dir="$RUNS_DIR/run-$seed"
    "$PIPELINE_PYTHON" -m sam_audio_pipeline.youtube_random \
      --output "$run_dir" \
      --source dailymotion \
      --profile cinematic \
      --clip-seconds 30 \
      --total 2000 \
      --seed "$seed" \
      --clips-per-video 24 \
      --query-count 500 \
      --results-per-query 100 \
      --search-workers "$SEARCH_WORKERS" \
      --download-workers "$DOWNLOAD_WORKERS" \
      --candidate-multiplier 1.8 \
      --max-attempts 40000 || true
    seed=$((seed + 1))
    printf '%s\n' "$seed" > "$seed_file.tmp"
    mv "$seed_file.tmp" "$seed_file"
  done
}

shutdown() {
  trap - TERM INT EXIT
  jobs -pr | xargs -r kill 2>/dev/null || true
  wait || true
}
trap shutdown TERM INT EXIT

heartbeat_loop downloader &
restart_worker downloader download_forever &

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
  heartbeat_loop "asr-$index" &
  restart_worker "asr-$index" "$WHISPER_PYTHON" \
    -m sam_audio_pipeline.m2d_validator asr-score \
    --input-dir "$WORKSPACE/raw-audio" \
    --output "$WORKSPACE/asr-validation/worker-$index.jsonl" \
    --m2d-results-dir "$WORKSPACE/m2d-validation" \
    --require-cinematic-mix \
    --model small \
    --download-root /home/ubuntu/.cache/huggingface/faster-whisper \
    --follow --poll-seconds 2 \
    --shard-index "$index" --shard-count "$ASR_WORKERS" &
done

heartbeat_loop assembler &
restart_worker assembler "$PIPELINE_PYTHON" -m sam_audio_pipeline.continuous_dataset \
  assemble --workspace "$WORKSPACE" --max-clips-per-video 24 --follow &

heartbeat_loop snapshot_publisher &
restart_worker snapshot_publisher "$PIPELINE_PYTHON" \
  -m sam_audio_pipeline.continuous_dataset publish-due \
  --workspace "$WORKSPACE" --bucket "$BUCKET" --prefix "$S3_PREFIX" \
  --snapshot-size 5000 --upload-concurrency "$UPLOAD_CONCURRENCY" \
  --follow --poll-seconds 30 &

wait -n
exit 1
