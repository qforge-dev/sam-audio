#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT=${SAM_CONTINUOUS_DEPLOY_ROOT:-/home/ubuntu/sam-audio-deploy}
PIPELINE_ROOT="$DEPLOY_ROOT/pipeline"
PIPELINE_PYTHON="$PIPELINE_ROOT/.venv/bin/python"
WORKSPACE=${SAM_CONTINUOUS_WORKSPACE:-/home/ubuntu/cinematic-continuous-30s}
DISCOVERY_DIR=${SAM_CONTINUOUS_DISCOVERY_DIR:-$WORKSPACE/source-discovery}
SEARCH_WORKERS=${SAM_CONTINUOUS_SEARCH_WORKERS:-8}
QUERY_COUNT=${SAM_CONTINUOUS_DISCOVERY_QUERY_COUNT:-50}
PROVIDER_QUERY_COUNTS=${SAM_CONTINUOUS_DISCOVERY_PROVIDER_QUERY_COUNTS:-}
SUCCESS_PAUSE=${SAM_CONTINUOUS_DISCOVERY_SUCCESS_PAUSE_SECONDS:-30}
FAILURE_PAUSE=${SAM_CONTINUOUS_DISCOVERY_FAILURE_PAUSE_SECONDS:-60}
HIGH_WATER_PAUSE=${SAM_CONTINUOUS_DISCOVERY_HIGH_WATER_PAUSE_SECONDS:-30}
EMPTY_MAX_PAUSE=${SAM_CONTINUOUS_DISCOVERY_EMPTY_MAX_PAUSE_SECONDS:-300}
HIGH_WATER=${SAM_CONTINUOUS_DISCOVERED_HIGH_WATER:-4000}
PLATFORM_HIGH_WATER=${SAM_CONTINUOUS_PLATFORM_DISCOVERED_HIGH_WATER:-600}
PLATFORM_HIGH_WATERS=${SAM_CONTINUOUS_PLATFORM_DISCOVERED_HIGH_WATERS:-}
MINIMUM_CANDIDATES=${SAM_CONTINUOUS_DISCOVERY_MINIMUM_CANDIDATES:-1}
BASE_CLIPS_PER_VIDEO=${SAM_CONTINUOUS_BASE_CLIPS_PER_VIDEO:-16}
SOURCE_CONTENT_MINUTES_PER_HOUR=${SAM_CONTINUOUS_SOURCE_CONTENT_MINUTES_PER_HOUR:-10}
MAX_CLIPS_PER_VIDEO=${SAM_CONTINUOUS_MAX_DURATION_SCALED_CLIPS_PER_VIDEO:-60}
SOURCES=${SAM_CONTINUOUS_DISCOVERY_SOURCES:-${SAM_CONTINUOUS_DISCOVERY_SOURCE:-youtube,dailymotion,vimeo,tiktok,soundcloud,bilibili,internet_archive}}
DISCOVERY_PROCESSES=${SAM_CONTINUOUS_DISCOVERY_PROCESSES:-1}
PROVIDER_WORKERS=${SAM_CONTINUOUS_DISCOVERY_PROVIDER_WORKERS:-}
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

# Keep each producer pinned to one provider. Rotating every controller through
# the same provider list made their long Bilibili turns synchronize, leaving no
# Dailymotion/Vimeo producer alive while downloaders drained the frontier.
SOURCE_ASSIGNMENTS=()
if [[ -n "$PROVIDER_WORKERS" ]]; then
  IFS=',' read -r -a RAW_PROVIDER_WORKERS <<< "$PROVIDER_WORKERS"
  for assignment in "${RAW_PROVIDER_WORKERS[@]}"; do
    assignment=${assignment//[[:space:]]/}
    source=${assignment%%=*}
    count=${assignment#*=}
    if [[ "$assignment" != *=* || -z "$source" || ! "$count" =~ ^[1-9][0-9]*$ ]]; then
      echo "SAM_CONTINUOUS_DISCOVERY_PROVIDER_WORKERS must use source=positive_count" >&2
      exit 2
    fi
    for ((index = 0; index < count; index++)); do
      SOURCE_ASSIGNMENTS+=("$source")
    done
  done
else
  for ((index = 0; index < DISCOVERY_PROCESSES; index++)); do
    SOURCE_ASSIGNMENTS+=("${SOURCE_POOL[$((index % ${#SOURCE_POOL[@]}))]}")
  done
fi
DISCOVERY_PROCESSES=${#SOURCE_ASSIGNMENTS[@]}

declare -A QUERY_COUNT_BY_PROVIDER=()
if [[ -n "$PROVIDER_QUERY_COUNTS" ]]; then
  IFS=',' read -r -a RAW_QUERY_COUNTS <<< "$PROVIDER_QUERY_COUNTS"
  for assignment in "${RAW_QUERY_COUNTS[@]}"; do
    assignment=${assignment//[[:space:]]/}
    source=${assignment%%=*}
    count=${assignment#*=}
    if [[ "$assignment" != *=* || -z "$source" || ! "$count" =~ ^[1-9][0-9]*$ ]]; then
      echo "SAM_CONTINUOUS_DISCOVERY_PROVIDER_QUERY_COUNTS must use source=positive_count" >&2
      exit 2
    fi
    QUERY_COUNT_BY_PROVIDER[$source]=$count
  done
fi

declare -A HIGH_WATER_BY_PROVIDER=()
if [[ -n "$PLATFORM_HIGH_WATERS" ]]; then
  IFS=',' read -r -a RAW_HIGH_WATERS <<< "$PLATFORM_HIGH_WATERS"
  for assignment in "${RAW_HIGH_WATERS[@]}"; do
    assignment=${assignment//[[:space:]]/}
    source=${assignment%%=*}
    count=${assignment#*=}
    if [[ "$assignment" != *=* || -z "$source" || ! "$count" =~ ^[1-9][0-9]*$ ]]; then
      echo "SAM_CONTINUOUS_PLATFORM_DISCOVERED_HIGH_WATERS must use source=positive_count" >&2
      exit 2
    fi
    HIGH_WATER_BY_PROVIDER[$source]=$count
  done
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
  local seed_file status_file initial_seed source pause query_count platform_high_water
  local batch_status inserted empty_streak=0
  if (( worker_index == 0 )); then
    # Preserve the original seed across an upgrade from one worker.
    seed_file="$WORKSPACE/source-discovery-next-seed"
    initial_seed=20260714
  else
    seed_file="$WORKSPACE/source-discovery-next-seed-$worker_index"
    initial_seed=$((20260714 + worker_index * 1000000))
  fi
  source=${SOURCE_ASSIGNMENTS[$worker_index]}
  status_file="$WORKSPACE/source-discovery-status-$worker_index.json"
  query_count=${QUERY_COUNT_BY_PROVIDER[$source]:-$QUERY_COUNT}
  platform_high_water=${HIGH_WATER_BY_PROVIDER[$source]:-$PLATFORM_HIGH_WATER}
  echo "Discovery worker $worker_index pinned to $source ($query_count queries, high water $platform_high_water)" >&2

  while true; do
    pause=$SUCCESS_PAUSE
    if "$PIPELINE_PYTHON" -m sam_audio_pipeline.source_pipeline discover \
      --workspace "$WORKSPACE" \
      --discovery-dir "$DISCOVERY_DIR" \
      --catalog "$WORKSPACE/catalog.sqlite3" \
      --seed-file "$seed_file" \
      --status-file "$status_file" \
      --seed "$initial_seed" \
      --source "$source" \
      "${PROXY_ARGS[@]}" \
      --profile cinematic \
      --clip-seconds 30 \
      --query-count "$query_count" \
      --results-per-query 100 \
      --search-workers "$SEARCH_WORKERS" \
      --minimum-candidates "$MINIMUM_CANDIDATES" \
      --clips-per-video "$BASE_CLIPS_PER_VIDEO" \
      --source-content-minutes-per-hour "$SOURCE_CONTENT_MINUTES_PER_HOUR" \
      --max-clips-per-video "$MAX_CLIPS_PER_VIDEO" \
      --discovered-high-water "$HIGH_WATER" \
      --platform-high-water "$platform_high_water" \
      --scan-cache "$WORKSPACE/source-scans" \
      --cached-scan-high-water 64 \
      --once; then
      read -r batch_status inserted < <(
        "$PIPELINE_PYTHON" -c \
          'import json,sys; p=json.load(open(sys.argv[1])); print(p.get("status","unknown"),int(p.get("inserted_sources") or 0))' \
          "$status_file"
      )
      if (( inserted > 0 )); then
        empty_streak=0
      elif [[ "$batch_status" == "high_water" ]]; then
        pause=$HIGH_WATER_PAUSE
      else
        empty_streak=$((empty_streak + 1))
        if (( empty_streak > 5 )); then
          empty_streak=5
        fi
        pause=$((FAILURE_PAUSE * (1 << (empty_streak - 1))))
        if (( pause > EMPTY_MAX_PAUSE )); then
          pause=$EMPTY_MAX_PAUSE
        fi
      fi
    else
      pause=$FAILURE_PAUSE
    fi
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
