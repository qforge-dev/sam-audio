#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT=${SAM_CONTINUOUS_DEPLOY_ROOT:-/home/ubuntu/sam-audio-deploy}
PIPELINE_ROOT="$DEPLOY_ROOT/pipeline"
PIPELINE_PYTHON="$PIPELINE_ROOT/.venv/bin/python"
WORKSPACE=${SAM_CONTINUOUS_WORKSPACE:-/home/ubuntu/cinematic-continuous-30s}
RUNS_DIR="$WORKSPACE/acquisition-runs"
SCAN_CACHE=${SAM_CONTINUOUS_SOURCE_SCAN_CACHE:-$WORKSPACE/source-scans}
WORKERS=${SAM_CONTINUOUS_SOURCE_EXTRACT_WORKERS:-16}
ASR_MODE=${SAM_CONTINUOUS_SOURCE_ASR_PROBE_MODE:-enforce}
ASR_TIMEOUT=${SAM_CONTINUOUS_SOURCE_ASR_PROBE_TIMEOUT:-120}
BASE_CLIPS_PER_VIDEO=${SAM_CONTINUOUS_BASE_CLIPS_PER_VIDEO:-16}
SOURCE_CONTENT_MINUTES_PER_HOUR=${SAM_CONTINUOUS_SOURCE_CONTENT_MINUTES_PER_HOUR:-10}
MAX_CLIPS_PER_VIDEO=${SAM_CONTINUOUS_MAX_DURATION_SCALED_CLIPS_PER_VIDEO:-60}
YOUTUBE_PROXY_CONFIG=${SAM_YOUTUBE_PROXY_CONFIG:-}

PROXY_ARGS=()
if [[ -n "$YOUTUBE_PROXY_CONFIG" ]]; then
  PROXY_ARGS+=(--youtube-proxy-config "$YOUTUBE_PROXY_CONFIG")
fi

export PYTHONPATH="$PIPELINE_ROOT/src"
mkdir -p "$RUNS_DIR" "$SCAN_CACHE"
cd "$PIPELINE_ROOT"

exec nice -n 10 "$PIPELINE_PYTHON" -m sam_audio_pipeline.source_pipeline extract \
  --workspace "$WORKSPACE" \
  --runs-dir "$RUNS_DIR" \
  --scan-cache "$SCAN_CACHE" \
  --catalog "$WORKSPACE/catalog.sqlite3" \
  --proxy-asr-request-dir "$WORKSPACE/source-asr-probe-requests" \
  --proxy-asr-result-dir "$WORKSPACE/source-asr-probe-results" \
  --proxy-asr-mode "$ASR_MODE" \
  --proxy-asr-timeout-seconds "$ASR_TIMEOUT" \
  --workers "$WORKERS" \
  --control-file "$WORKSPACE/source-stage-control.json" \
  --clip-seconds 30 \
  --run-target 2000 \
  --clips-per-video "$BASE_CLIPS_PER_VIDEO" \
  --source-content-minutes-per-hour "$SOURCE_CONTENT_MINUTES_PER_HOUR" \
  --max-clips-per-video "$MAX_CLIPS_PER_VIDEO" \
  --yt-dlp-python "$PIPELINE_PYTHON" \
  "${PROXY_ARGS[@]}"
