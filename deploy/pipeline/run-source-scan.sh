#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT=${SAM_CONTINUOUS_DEPLOY_ROOT:-/home/ubuntu/sam-audio-deploy}
PIPELINE_ROOT="$DEPLOY_ROOT/pipeline"
MODEL_PYTHON="$DEPLOY_ROOT/.venv/bin/python"
M2D_REPO=${SAM_CONTINUOUS_M2D_REPO:-/home/ubuntu/m2d}
M2D_CHECKPOINT=${SAM_CONTINUOUS_M2D_CHECKPOINT:-$M2D_REPO/weights/m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d/weights_ep69it3124-0.47929.pth}
CLASS_LABELS=${SAM_CONTINUOUS_CLASS_LABELS:-/home/ubuntu/m2d-validation/metadata/class_labels_indices.csv}
ONTOLOGY=${SAM_CONTINUOUS_ONTOLOGY:-/home/ubuntu/m2d-validation/metadata/ontology.json}
WORKSPACE=${SAM_CONTINUOUS_WORKSPACE:-/home/ubuntu/cinematic-continuous-30s}
SCAN_CACHE=${SAM_CONTINUOUS_SOURCE_SCAN_CACHE:-$WORKSPACE/source-scans}
WORKERS=${SAM_CONTINUOUS_SOURCE_SCAN_WORKERS:-4}
HIGH_WATER=${SAM_CONTINUOUS_SCANNED_HIGH_WATER:-64}
ASR_MODE=${SAM_CONTINUOUS_SOURCE_ASR_PROBE_MODE:-enforce}
ASR_TIMEOUT=${SAM_CONTINUOUS_SOURCE_ASR_PROBE_TIMEOUT:-120}
BATCH_SIZE=${SAM_CONTINUOUS_SOURCE_SCAN_BATCH_SIZE:-128}

export PYTHONPATH="$PIPELINE_ROOT/src"
export HF_HOME=${HF_HOME:-/home/ubuntu/.cache/huggingface}
mkdir -p "$SCAN_CACHE" "$WORKSPACE/source-asr-probe-requests" \
  "$WORKSPACE/source-asr-probe-results"
cd "$PIPELINE_ROOT"

exec nice -n 10 "$MODEL_PYTHON" -m sam_audio_pipeline.source_pipeline scan \
  --workspace "$WORKSPACE" \
  --scan-cache "$SCAN_CACHE" \
  --proxy-asr-request-dir "$WORKSPACE/source-asr-probe-requests" \
  --proxy-asr-result-dir "$WORKSPACE/source-asr-probe-results" \
  --proxy-asr-mode "$ASR_MODE" \
  --proxy-asr-timeout-seconds "$ASR_TIMEOUT" \
  --scanned-high-water "$HIGH_WATER" \
  --workers "$WORKERS" \
  --control-file "$WORKSPACE/source-stage-control.json" \
  --clip-seconds 30 \
  --m2d-repo "$M2D_REPO" \
  --checkpoint "$M2D_CHECKPOINT" \
  --class-labels "$CLASS_LABELS" \
  --ontology "$ONTOLOGY" \
  --device cuda \
  --batch-size "$BATCH_SIZE" \
  --inference-concurrency 2
