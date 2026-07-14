#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT=${SAM_CONTINUOUS_DEPLOY_ROOT:-/home/ubuntu/sam-audio-deploy}
PIPELINE_ROOT="$DEPLOY_ROOT/pipeline"
PIPELINE_PYTHON="$PIPELINE_ROOT/.venv/bin/python"
WORKSPACE=${SAM_CONTINUOUS_WORKSPACE:-/home/ubuntu/cinematic-continuous-30s}
DOWNLOAD_MAX=${SAM_CONTINUOUS_SOURCE_DOWNLOAD_WORKERS:-16}
SCAN_MAX=${SAM_CONTINUOUS_SOURCE_SCAN_WORKERS:-4}
EXTRACT_MAX=${SAM_CONTINUOUS_SOURCE_EXTRACT_WORKERS:-16}

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
  --interval-seconds 10
