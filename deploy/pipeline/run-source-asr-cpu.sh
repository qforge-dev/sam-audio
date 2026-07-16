#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT=${SAM_CONTINUOUS_DEPLOY_ROOT:-/home/ubuntu/sam-audio-deploy}
PIPELINE_ROOT="$DEPLOY_ROOT/pipeline"
WHISPER_PYTHON=${SAM_CONTINUOUS_SOURCE_ASR_CPU_PYTHON:-/home/ubuntu/whisper-cpu-venv/bin/python}
WORKSPACE=${SAM_CONTINUOUS_WORKSPACE:-/home/ubuntu/cinematic-continuous-30s}
SHARED_ROOT=${SAM_CONTINUOUS_SOURCE_ASR_SHARED_ROOT:-$WORKSPACE/source-work/.source-asr}
MODEL_LABEL=${SAM_CONTINUOUS_SOURCE_ASR_MODEL:-small}
MODEL=$MODEL_LABEL
WORKERS=${SAM_CONTINUOUS_SOURCE_ASR_CPU_WORKERS:-8}
CPU_THREADS=${SAM_CONTINUOUS_SOURCE_ASR_CPU_THREADS:-4}
MODEL_CACHE=${SAM_CONTINUOUS_SOURCE_ASR_MODEL_CACHE:-/home/ubuntu/.cache/huggingface/faster-whisper}
FINAL_ASR_ENABLED=${SAM_CONTINUOUS_SOURCE_ASR_CPU_FINAL_ENABLED:-false}

# Prefer the already-provisioned snapshot so service restarts never depend on
# a Hugging Face metadata request. Fall back to the model name for first-time
# provisioning when no compatible snapshot has been copied yet.
if [[ "$MODEL" == "small" ]]; then
  for snapshot in "$MODEL_CACHE"/models--Systran--faster-whisper-small/snapshots/*; do
    if [[ -d "$snapshot" ]]; then
      MODEL=$snapshot
      break
    fi
  done
fi

REQUESTS="$SHARED_ROOT/source-asr-probe-requests"
RESULTS="$SHARED_ROOT/source-asr-probe-results"
EMPTY_INPUT="$SHARED_ROOT/empty-input"
CONTINUOUS_ROOT="$SHARED_ROOT/continuous"
INPUT="$EMPTY_INPUT"
OUTPUT="$SHARED_ROOT/cpu-probe-worker.jsonl"
FINAL_ARGS=()
if [[ "$FINAL_ASR_ENABLED" == "true" ]]; then
  INPUT="$CONTINUOUS_ROOT/raw-audio"
  OUTPUT="$CONTINUOUS_ROOT/asr-validation/worker-0.jsonl"
  FINAL_ARGS=(
    --m2d-results-dir "$CONTINUOUS_ROOT/m2d-validation"
    --require-cinematic-mix
  )
fi

export PYTHONPATH="$PIPELINE_ROOT/src"
mkdir -p "$REQUESTS" "$RESULTS" "$EMPTY_INPUT" "$INPUT" \
  "$CONTINUOUS_ROOT/m2d-validation" "$(dirname "$OUTPUT")"
cd "$PIPELINE_ROOT"

exec "$WHISPER_PYTHON" -m sam_audio_pipeline.m2d_validator asr-score \
  --input-dir "$INPUT" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --model-label "$MODEL_LABEL" \
  --device cpu \
  --compute-type int8 \
  --download-root "$MODEL_CACHE" \
  --max-inference-workers "$WORKERS" \
  --cpu-threads "$CPU_THREADS" \
  --probe-requests-dir "$REQUESTS" \
  --probe-results-dir "$RESULTS" \
  "${FINAL_ARGS[@]}" \
  --follow --poll-seconds 0.25
