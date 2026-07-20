#!/usr/bin/env bash
set -uo pipefail

DEPLOY_ROOT=${SAM_TRAINING_DEPLOY_ROOT:-/home/ubuntu/sam-audio-deploy}
PIPELINE_ROOT="$DEPLOY_ROOT/pipeline"
PIPELINE_PYTHON="$PIPELINE_ROOT/.venv/bin/python"
MODEL_PYTHON=${SAM_TRAINING_MODEL_PYTHON:-$DEPLOY_ROOT/.venv/bin/python}
WHISPER_PYTHON=${SAM_TRAINING_WHISPER_PYTHON:-/home/ubuntu/whisper-venv/bin/python}
WORKSPACE=${SAM_TRAINING_WORKSPACE:-/home/ubuntu/dialogue-background-training-v1}
SOURCE_WORKSPACE=${SAM_TRAINING_SOURCE_WORKSPACE:-/home/ubuntu/cinematic-continuous-30s}
INBOX=${SAM_TRAINING_INBOX:-$WORKSPACE/inbox}
BUCKET=${SAM_TRAINING_S3_BUCKET:?SAM_TRAINING_S3_BUCKET is required}
SOURCE_PREFIX=${SAM_TRAINING_SOURCE_S3_PREFIX:-cinematic-dialogue-dataset}
OUTPUT_PREFIX=${SAM_TRAINING_OUTPUT_S3_PREFIX:-dialogue-background-training-v1}
SAM_API_URL=${SAM_TRAINING_SAM_API_URL:-http://127.0.0.1:8000}
SAM_API_URLS=${SAM_TRAINING_SAM_API_URLS:-$SAM_API_URL}
FLAMINGO_API_URL=${SAM_TRAINING_FLAMINGO_API_URL:-http://127.0.0.1:8001}
SEPARATION_WORKERS=${SAM_TRAINING_SEPARATION_WORKERS:-1}
ELASTIC_SEPARATION_WORKER_FROM=${SAM_TRAINING_ELASTIC_SEPARATION_WORKER_FROM:--1}
DESCRIPTION_BACKLOG_HIGH=${SAM_TRAINING_DESCRIPTION_BACKLOG_HIGH:-500}
DESCRIPTION_WORKERS=${SAM_TRAINING_DESCRIPTION_WORKERS:-1}
PACKAGE_WORKERS=${SAM_TRAINING_PACKAGE_WORKERS:-2}
SNAPSHOT_SIZE=${SAM_TRAINING_SNAPSHOT_SIZE:-1000}
UPLOAD_WORKERS=${SAM_TRAINING_UPLOAD_WORKERS:-8}
M2D_REPO=${SAM_TRAINING_M2D_REPO:-/home/ubuntu/m2d}
M2D_CHECKPOINT=${SAM_TRAINING_M2D_CHECKPOINT:-$M2D_REPO/weights/m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d/weights_ep69it3124-0.47929.pth}
CLASS_LABELS=${SAM_TRAINING_CLASS_LABELS:-/home/ubuntu/m2d-validation/metadata/class_labels_indices.csv}
ONTOLOGY=${SAM_TRAINING_ONTOLOGY:-/home/ubuntu/m2d-validation/metadata/ontology.json}
ASR_MODEL=${SAM_TRAINING_ASR_MODEL:-/home/ubuntu/.cache/huggingface/faster-whisper/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120}
ASR_WORKERS=${SAM_TRAINING_ASR_WORKERS:-2}
ASR_CPU_THREADS=${SAM_TRAINING_ASR_CPU_THREADS:-4}
M2D_DEVICE=${SAM_TRAINING_M2D_DEVICE:-cuda}
ASR_DEVICE=${SAM_TRAINING_ASR_DEVICE:-cuda}
ASR_COMPUTE_TYPE=${SAM_TRAINING_ASR_COMPUTE_TYPE:-float16}

export PYTHONPATH="$PIPELINE_ROOT/src"
export HF_HOME=${HF_HOME:-/home/ubuntu/.cache/huggingface}
mkdir -p "$WORKSPACE" "$INBOX" "$WORKSPACE/logs"
cd "$PIPELINE_ROOT"

# systemd has already terminated the previous control group. Requeue its leased
# jobs immediately instead of waiting for the conservative per-stage TTL.
"$PIPELINE_PYTHON" -m sam_audio_pipeline.training_dataset \
  --workspace "$WORKSPACE" recover-leases \
  >>"$WORKSPACE/logs/lease-recovery.log" 2>&1

restart_worker() {
  local name=$1
  shift
  while true; do
    "$@" >>"$WORKSPACE/logs/$name.log" 2>&1
    status=$?
    printf '%s worker=%s exit=%s restarting\n' "$(date --iso-8601=seconds)" \
      "$name" "$status" >>"$WORKSPACE/logs/supervisor.log"
    sleep 5
  done
}

wait_for_http() {
  local url=$1
  until curl --connect-timeout 2 --max-time 5 -fsS "$url" >/dev/null; do
    sleep 5
  done
}

run_separation() {
  local api_url=$1
  local worker_count=$2
  local worker_offset=$3
  wait_for_http "$api_url/healthz"
  exec "$PIPELINE_PYTHON" -m sam_audio_pipeline.training_dataset \
    --workspace "$WORKSPACE" separate \
    --sam-api-url "$api_url" --bucket "$BUCKET" \
    --workers "$worker_count" --worker-offset "$worker_offset" \
    --elastic-worker-from "$ELASTIC_SEPARATION_WORKER_FROM" \
    --description-backlog-high "$DESCRIPTION_BACKLOG_HIGH" --follow
}

run_description() {
  wait_for_http "$FLAMINGO_API_URL/healthz"
  exec "$PIPELINE_PYTHON" -m sam_audio_pipeline.training_dataset \
    --workspace "$WORKSPACE" describe \
    --flamingo-api-url "$FLAMINGO_API_URL" \
    --workers "$DESCRIPTION_WORKERS" --follow
}

shutdown() {
  trap - TERM INT EXIT
  jobs -pr | xargs -r kill 2>/dev/null || true
  exit 0
}
trap shutdown TERM INT EXIT

restart_worker sync "$PIPELINE_PYTHON" -m sam_audio_pipeline.training_dataset \
  --workspace "$WORKSPACE" sync \
  --source-workspace "$SOURCE_WORKSPACE" --source-s3-prefix "$SOURCE_PREFIX" \
  --inbox "$INBOX" --limit 5000 --follow &

IFS=',' read -r -a SAM_API_URL_LIST <<< "$SAM_API_URLS"
if (( ${#SAM_API_URL_LIST[@]} > SEPARATION_WORKERS )); then
  echo "SAM_TRAINING_SEPARATION_WORKERS must cover every SAM API URL" >&2
  exit 2
fi
worker_offset=0
for api_index in "${!SAM_API_URL_LIST[@]}"; do
  api_url=${SAM_API_URL_LIST[$api_index]//[[:space:]]/}
  worker_count=$((SEPARATION_WORKERS / ${#SAM_API_URL_LIST[@]}))
  if (( api_index < SEPARATION_WORKERS % ${#SAM_API_URL_LIST[@]} )); then
    worker_count=$((worker_count + 1))
  fi
  restart_worker "separation-$api_index" run_separation \
    "$api_url" "$worker_count" "$worker_offset" &
  worker_offset=$((worker_offset + worker_count))
done

restart_worker m2d "$MODEL_PYTHON" -m sam_audio_pipeline.m2d_validator score \
  --input-dir "$WORKSPACE/background-audio" \
  --output "$WORKSPACE/background-tags.jsonl" \
  --m2d-repo "$M2D_REPO" --checkpoint "$M2D_CHECKPOINT" \
  --class-labels "$CLASS_LABELS" --ontology "$ONTOLOGY" \
  --m2d-commit 3d0c4de9447c404a8d3f9f37e04f53bc902e09b3 \
  --device "$M2D_DEVICE" --follow --poll-seconds 2 &

restart_worker tag-import "$PIPELINE_PYTHON" \
  -m sam_audio_pipeline.training_dataset --workspace "$WORKSPACE" \
  import-jsonl --path "$WORKSPACE/background-tags.jsonl" --kind tag \
  --follow --poll-seconds 2 &

restart_worker asr "$WHISPER_PYTHON" -m sam_audio_pipeline.m2d_validator asr-score \
  --input-dir "$WORKSPACE/dialogue-audio" \
  --output "$WORKSPACE/dialogue-asr.jsonl" \
  --model "$ASR_MODEL" --model-label small \
  --device "$ASR_DEVICE" --compute-type "$ASR_COMPUTE_TYPE" \
  --download-root /home/ubuntu/.cache/huggingface/faster-whisper \
  --max-inference-workers "$ASR_WORKERS" --cpu-threads "$ASR_CPU_THREADS" \
  --follow --poll-seconds 2 &

restart_worker asr-import "$PIPELINE_PYTHON" \
  -m sam_audio_pipeline.training_dataset --workspace "$WORKSPACE" \
  import-jsonl --path "$WORKSPACE/dialogue-asr.jsonl" --kind asr \
  --follow --poll-seconds 2 &

restart_worker description run_description &

restart_worker package "$PIPELINE_PYTHON" -m sam_audio_pipeline.training_dataset \
  --workspace "$WORKSPACE" package --workers "$PACKAGE_WORKERS" --follow &

restart_worker snapshot-publisher "$PIPELINE_PYTHON" \
  -m sam_audio_pipeline.training_dataset --workspace "$WORKSPACE" publish \
  --bucket "$BUCKET" --prefix "$OUTPUT_PREFIX" \
  --snapshot-size "$SNAPSHOT_SIZE" --upload-workers "$UPLOAD_WORKERS" \
  --follow --poll-seconds 30 &

wait -n
exit 1
