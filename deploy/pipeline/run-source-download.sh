#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT=${SAM_CONTINUOUS_DEPLOY_ROOT:-/home/ubuntu/sam-audio-deploy}
PIPELINE_ROOT="$DEPLOY_ROOT/pipeline"
PIPELINE_PYTHON="$PIPELINE_ROOT/.venv/bin/python"
WORKSPACE=${SAM_CONTINUOUS_WORKSPACE:-/home/ubuntu/cinematic-continuous-30s}
SOURCE_CACHE=${SAM_CONTINUOUS_SOURCE_CACHE:-$WORKSPACE/source-work}
WORKERS=${SAM_CONTINUOUS_SOURCE_DOWNLOAD_WORKERS:-16}
HIGH_WATER=${SAM_CONTINUOUS_DOWNLOADED_HIGH_WATER:-64}
HIGH_WATER_BYTES=${SAM_CONTINUOUS_DOWNLOADED_HIGH_WATER_BYTES:-8589934592}

export PYTHONPATH="$PIPELINE_ROOT/src"
mkdir -p "$SOURCE_CACHE"
cd "$PIPELINE_ROOT"

exec "$PIPELINE_PYTHON" -m sam_audio_pipeline.source_pipeline download \
  --workspace "$WORKSPACE" \
  --source-cache "$SOURCE_CACHE" \
  --workers "$WORKERS" \
  --downloaded-high-water "$HIGH_WATER" \
  --downloaded-high-water-bytes "$HIGH_WATER_BYTES" \
  --clip-seconds 30 \
  --control-file "$WORKSPACE/source-stage-control.json" \
  --yt-dlp-python "$PIPELINE_PYTHON"
