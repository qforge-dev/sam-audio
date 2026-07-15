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
DISCOVERY_PROCESSES=${SAM_CONTINUOUS_DISCOVERY_PROCESSES:-1}
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
if ! [[ "$DISCOVERY_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAM_CONTINUOUS_DISCOVERY_PROCESSES must be a positive integer" >&2
  exit 2
fi

PROXY_ARGS=()
if [[ -n "$YOUTUBE_PROXY_CONFIG" ]]; then
  PROXY_ARGS+=(--youtube-proxy-config "$YOUTUBE_PROXY_CONFIG")
fi

export PYTHONPATH="$PIPELINE_ROOT/src"
mkdir -p "$DISCOVERY_DIR"
cd "$PIPELINE_ROOT"

# CPython 3.14's urllib response finalizers can retain descriptors after several
# hundred concurrent Dailymotion requests. Keep each controller persistent but
# isolate every discovery batch in a short-lived child so the OS closes every
# descriptor before the next seed. Each controller owns its provider cursor and
# seed file; otherwise parallel controllers would repeat the same searches.
run_discovery_controller() {
  local worker_index=$1
  local source_index_file seed_file source_index initial_seed source pause
  if (( worker_index == 0 )); then
    # Preserve the original cursor and seed across an upgrade from one worker.
    source_index_file="$WORKSPACE/source-discovery-next-provider"
    seed_file="$WORKSPACE/source-discovery-next-seed"
    initial_seed=20260714
  else
    source_index_file="$WORKSPACE/source-discovery-next-provider-$worker_index"
    seed_file="$WORKSPACE/source-discovery-next-seed-$worker_index"
    initial_seed=$((20260714 + worker_index * 1000000))
  fi
  source_index=$(test -s "$source_index_file" && tr -dc '0-9' < "$source_index_file" || echo "$worker_index")

  while true; do
    source=${SOURCE_POOL[$((source_index % ${#SOURCE_POOL[@]}))]}
    pause=$SUCCESS_PAUSE
    if ! "$PIPELINE_PYTHON" -m sam_audio_pipeline.source_pipeline discover \
      --workspace "$WORKSPACE" \
      --discovery-dir "$DISCOVERY_DIR" \
      --catalog "$WORKSPACE/catalog.sqlite3" \
      --seed-file "$seed_file" \
      --seed "$initial_seed" \
      --source "$source" \
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
      pause=$FAILURE_PAUSE
    fi
    source_index=$(((source_index + 1) % ${#SOURCE_POOL[@]}))
    printf '%s\n' "$source_index" > "$source_index_file.tmp"
    mv "$source_index_file.tmp" "$source_index_file"
    sleep "$pause"
  done
}

pids=()
stop_controllers() {
  if (( ${#pids[@]} )); then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap stop_controllers EXIT INT TERM
for ((worker_index = 0; worker_index < DISCOVERY_PROCESSES; worker_index++)); do
  run_discovery_controller "$worker_index" &
  pids+=("$!")
done
wait "${pids[@]}"
