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
PLATFORM_HIGH_WATER=${SAM_CONTINUOUS_PLATFORM_DISCOVERED_HIGH_WATER:-600}
MINIMUM_CANDIDATES=${SAM_CONTINUOUS_DISCOVERY_MINIMUM_CANDIDATES:-1}
BASE_CLIPS_PER_VIDEO=${SAM_CONTINUOUS_BASE_CLIPS_PER_VIDEO:-16}
SOURCE_CONTENT_MINUTES_PER_HOUR=${SAM_CONTINUOUS_SOURCE_CONTENT_MINUTES_PER_HOUR:-10}
MAX_CLIPS_PER_VIDEO=${SAM_CONTINUOUS_MAX_DURATION_SCALED_CLIPS_PER_VIDEO:-60}
SOURCES=${SAM_CONTINUOUS_DISCOVERY_SOURCES:-${SAM_CONTINUOUS_DISCOVERY_SOURCE:-youtube,dailymotion,vimeo,tiktok,soundcloud,bilibili,internet_archive}}
YOUTUBE_PROXY_CONFIG=${SAM_YOUTUBE_PROXY_CONFIG:-}

IFS=',' read -r -a RAW_SOURCE_POOL <<< "$SOURCES"
SOURCE_POOL=()
for source in "${RAW_SOURCE_POOL[@]}"; do
  source=${source//[[:space:]]/}
  if [[ -n "$source" ]]; then
    SOURCE_POOL+=("$source")
  fi
done
if (( ${#SOURCE_POOL[@]} == 0 )); then
  echo "SAM_CONTINUOUS_DISCOVERY_SOURCES must contain at least one source" >&2
  exit 2
fi
SOURCE_INDEX_FILE="$WORKSPACE/source-discovery-next-provider"
SOURCE_INDEX=$(test -s "$SOURCE_INDEX_FILE" && tr -dc '0-9' < "$SOURCE_INDEX_FILE" || echo 0)

PROXY_ARGS=()
if [[ -n "$YOUTUBE_PROXY_CONFIG" ]]; then
  PROXY_ARGS+=(--youtube-proxy-config "$YOUTUBE_PROXY_CONFIG")
fi

export PYTHONPATH="$PIPELINE_ROOT/src"
mkdir -p "$DISCOVERY_DIR"
cd "$PIPELINE_ROOT"

# CPython 3.14's urllib response finalizers can retain descriptors after several
# hundred concurrent Dailymotion requests. Keep the controller persistent but
# isolate each discovery batch in a short-lived child so the OS closes every
# descriptor before the next seed. No model is loaded in this stage.
while true; do
  SOURCE=${SOURCE_POOL[$((SOURCE_INDEX % ${#SOURCE_POOL[@]}))]}
  PAUSE=$SUCCESS_PAUSE
  if ! "$PIPELINE_PYTHON" -m sam_audio_pipeline.source_pipeline discover \
    --workspace "$WORKSPACE" \
    --discovery-dir "$DISCOVERY_DIR" \
    --catalog "$WORKSPACE/catalog.sqlite3" \
    --source "$SOURCE" \
    "${PROXY_ARGS[@]}" \
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
    --platform-high-water "$PLATFORM_HIGH_WATER" \
    --scan-cache "$WORKSPACE/source-scans" \
    --cached-scan-high-water 64 \
    --once; then
    PAUSE=$FAILURE_PAUSE
  fi
  SOURCE_INDEX=$(((SOURCE_INDEX + 1) % ${#SOURCE_POOL[@]}))
  printf '%s\n' "$SOURCE_INDEX" > "$SOURCE_INDEX_FILE.tmp"
  mv "$SOURCE_INDEX_FILE.tmp" "$SOURCE_INDEX_FILE"
  sleep "$PAUSE"
done
