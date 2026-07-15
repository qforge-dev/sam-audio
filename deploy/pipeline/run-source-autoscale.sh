#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT=${SAM_CONTINUOUS_DEPLOY_ROOT:-/home/ubuntu/sam-audio-deploy}
PIPELINE_ROOT="$DEPLOY_ROOT/pipeline"
PIPELINE_PYTHON="$PIPELINE_ROOT/.venv/bin/python"
WORKSPACE=${SAM_CONTINUOUS_WORKSPACE:-/home/ubuntu/cinematic-continuous-30s}
DOWNLOAD_MAX=${SAM_CONTINUOUS_SOURCE_DOWNLOAD_WORKERS:-16}
SCAN_MAX=${SAM_CONTINUOUS_SOURCE_SCAN_WORKERS:-4}
EXTRACT_MAX=${SAM_CONTINUOUS_SOURCE_EXTRACT_WORKERS:-16}
REMOTE_TASKS=,${SAM_MEDIA_WORKER_REMOTE_TASKS:-},
CPU_EXEMPT_ARGS=()
if [[ "$REMOTE_TASKS" == *,download,* ]]; then
  CPU_EXEMPT_ARGS+=(--cpu-exempt-stage download)
fi

export PYTHONPATH="$PIPELINE_ROOT/src"
cd "$PIPELINE_ROOT"

exec "$PIPELINE_PYTHON" -m sam_audio_pipeline.source_pipeline autoscale \
  --workspace "$WORKSPACE" \
  --control-file "$WORKSPACE/source-stage-control.json" \
  --download-min 2 --download-max "$DOWNLOAD_MAX" --download-initial 8 \
  --scan-min 1 --scan-max "$SCAN_MAX" --scan-initial 2 \
  --extract-min 1 --extract-max "$EXTRACT_MAX" --extract-initial 4 \
  --cpu-low 55 --cpu-high 85 \
  --scan-backlog-high 16 \
  --extract-backlog-high 8 \
  --download-backlog-low 4 \
  "${CPU_EXEMPT_ARGS[@]}" \
  --interval-seconds 10
