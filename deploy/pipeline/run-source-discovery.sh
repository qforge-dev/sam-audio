#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT=${SAM_CONTINUOUS_DEPLOY_ROOT:-/home/ubuntu/sam-audio-deploy}
PIPELINE_ROOT="$DEPLOY_ROOT/pipeline"
PIPELINE_PYTHON="$PIPELINE_ROOT/.venv/bin/python"
WORKSPACE=${SAM_CONTINUOUS_WORKSPACE:-/home/ubuntu/cinematic-continuous-30s}
DISCOVERY_DIR=${SAM_CONTINUOUS_DISCOVERY_DIR:-$WORKSPACE/source-discovery}
SEARCH_WORKERS=${SAM_CONTINUOUS_SEARCH_WORKERS:-8}
QUERY_COUNT=${SAM_CONTINUOUS_DISCOVERY_QUERY_COUNT:-50}
SUCCESS_PAUSE=${SAM_CONTINUOUS_DISCOVERY_SUCCESS_PAUSE_SECONDS:-30}
FAILURE_PAUSE=${SAM_CONTINUOUS_DISCOVERY_FAILURE_PAUSE_SECONDS:-60}
HIGH_WATER=${SAM_CONTINUOUS_DISCOVERED_HIGH_WATER:-4000}
MINIMUM_CANDIDATES=${SAM_CONTINUOUS_DISCOVERY_MINIMUM_CANDIDATES:-1}
BASE_CLIPS_PER_VIDEO=${SAM_CONTINUOUS_BASE_CLIPS_PER_VIDEO:-16}
SOURCE_CONTENT_MINUTES_PER_HOUR=${SAM_CONTINUOUS_SOURCE_CONTENT_MINUTES_PER_HOUR:-10}
MAX_CLIPS_PER_VIDEO=${SAM_CONTINUOUS_MAX_DURATION_SCALED_CLIPS_PER_VIDEO:-60}

export PYTHONPATH="$PIPELINE_ROOT/src"
mkdir -p "$DISCOVERY_DIR"
cd "$PIPELINE_ROOT"

# CPython 3.14's urllib response finalizers can retain descriptors after several
# hundred concurrent Dailymotion requests. Keep the controller persistent but
# isolate each discovery batch in a short-lived child so the OS closes every
# descriptor before the next seed. No model is loaded in this stage.
while true; do
  if ! "$PIPELINE_PYTHON" -m sam_audio_pipeline.source_pipeline discover \
    --workspace "$WORKSPACE" \
    --discovery-dir "$DISCOVERY_DIR" \
    --catalog "$WORKSPACE/catalog.sqlite3" \
    --source dailymotion \
    --profile cinematic \
    --clip-seconds 30 \
    --query-count "$QUERY_COUNT" \
    --results-per-query 100 \
    --search-workers "$SEARCH_WORKERS" \
    --minimum-candidates "$MINIMUM_CANDIDATES" \
    --clips-per-video "$BASE_CLIPS_PER_VIDEO" \
    --source-content-minutes-per-hour "$SOURCE_CONTENT_MINUTES_PER_HOUR" \
    --max-clips-per-video "$MAX_CLIPS_PER_VIDEO" \
    --discovered-high-water "$HIGH_WATER" \
    --scan-cache "$WORKSPACE/source-scans" \
    --cached-scan-high-water 64 \
    --once; then
    sleep "$FAILURE_PAUSE"
    continue
  fi
  sleep "$SUCCESS_PAUSE"
done
