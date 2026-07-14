#!/usr/bin/env bash
set -uo pipefail

PIPELINE_ROOT=/home/ubuntu/sam-audio-deploy/pipeline
PIPELINE_PYTHON="$PIPELINE_ROOT/.venv/bin/python"
MODEL_PYTHON=/home/ubuntu/sam-audio-deploy/.venv/bin/python
WHISPER_PYTHON=/home/ubuntu/whisper-venv/bin/python
M2D_REPO=/home/ubuntu/m2d
M2D_CHECKPOINT="$M2D_REPO/weights/m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d/weights_ep69it3124-0.47929.pth"
CLASS_LABELS=/home/ubuntu/m2d-validation/metadata/class_labels_indices.csv
ONTOLOGY=/home/ubuntu/m2d-validation/metadata/ontology.json
M2D_COMMIT=3d0c4de9447c404a8d3f9f37e04f53bc902e09b3
FINAL_DIR=/home/ubuntu/cinematic-final-1000-20260715
MATERIALIZE_LOG="${FINAL_DIR}.materialize.log"

export PYTHONPATH="$PIPELINE_ROOT/src"
export HF_HOME=/home/ubuntu/.cache/huggingface
cd "$PIPELINE_ROOT"

batch_at_target() {
  local batch_dir=$1
  local target=$2
  "$PIPELINE_PYTHON" - "$batch_dir/manifest.json" "$target" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
target = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
manifest = json.loads(path.read_text())
raise SystemExit(0 if len(manifest.get("records", [])) >= target else 1)
PY
}

acquire_batch() {
  local seed=$1
  local target=$2
  local batch_dir="/home/ubuntu/cinematic-dm-raw-$seed"
  # One producer downloads each source once and publishes completed WAV files
  # by atomic move. The two consumers can safely discover them immediately.
  "$PIPELINE_PYTHON" -m sam_audio_pipeline.youtube_random \
    --output "$batch_dir" \
    --source dailymotion \
    --profile cinematic \
    --total "$target" \
    --seed "$seed" \
    --clips-per-video 16 \
    --source-content-minutes-per-hour 10 \
    --max-clips-per-video 60 \
    --query-count 500 \
    --results-per-query 100 \
    --search-workers 8 \
    --download-workers 8 \
    --candidate-multiplier 1.8 \
    --max-attempts 40000 || true
}

follow_m2d() {
  local batch_dir=$1
  local producer_done=$2
  "$MODEL_PYTHON" -m sam_audio_pipeline.m2d_validator score \
    --input-dir "$batch_dir/audio" \
    --output "$batch_dir/m2d-validation.jsonl" \
    --m2d-repo "$M2D_REPO" \
    --checkpoint "$M2D_CHECKPOINT" \
    --class-labels "$CLASS_LABELS" \
    --ontology "$ONTOLOGY" \
    --m2d-commit "$M2D_COMMIT" \
    --require-cinematic-mix \
    --follow \
    --producer-done "$producer_done" \
    --poll-seconds 2
}

follow_asr() {
  local batch_dir=$1
  local producer_done=$2
  "$WHISPER_PYTHON" -m sam_audio_pipeline.m2d_validator asr-score \
    --input-dir "$batch_dir/audio" \
    --output "$batch_dir/asr-validation.jsonl" \
    --m2d-results "$batch_dir/m2d-validation.jsonl" \
    --require-cinematic-mix \
    --model small \
    --download-root /home/ubuntu/.cache/huggingface/faster-whisper \
    --follow \
    --producer-done "$producer_done" \
    --poll-seconds 2
}

try_materialize() {
  local args=()
  local seed
  for seed in 20260715 20260716 20260717 20260718 20260719; do
    local batch_dir="/home/ubuntu/cinematic-dm-raw-$seed"
    if [[ -s "$batch_dir/m2d-validation.jsonl" && -s "$batch_dir/asr-validation.jsonl" ]]; then
      args+=(--batch "$batch_dir")
    fi
  done
  if [[ ${#args[@]} -eq 0 ]]; then
    return 1
  fi
  "$PIPELINE_PYTHON" -m sam_audio_pipeline.m2d_validator merge-materialize \
    "${args[@]}" \
    --output-dir "$FINAL_DIR" \
    --accepted-limit 1000 \
    --max-clips-per-video 16 \
    --source-content-minutes-per-hour 10 \
    --max-duration-scaled-clips-per-video 60 \
    --seed 20260715 \
    --require-cinematic-mix
}

final_ready() {
  "$PIPELINE_PYTHON" - "$FINAL_DIR/audit.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
audit = json.loads(path.read_text())
ready = (
    audit.get("all_requirements_pass") is True
    and audit.get("record_count") == 1000
    and audit.get("audio_file_count") == 1000
)
raise SystemExit(0 if ready else 1)
PY
}

validated_line_count() {
  local total=0
  local seed
  for seed in 20260715 20260716 20260717 20260718 20260719; do
    local path="/home/ubuntu/cinematic-dm-raw-$seed/asr-validation.jsonl"
    if [[ -f "$path" ]]; then
      local count
      count=$(wc -l < "$path")
      total=$((total + count))
    fi
  done
  echo "$total"
}

watch_materialization() {
  local terminal_marker=$1
  local last_attempt_lines
  last_attempt_lines=$(validated_line_count)
  while true; do
    if final_ready; then
      return 0
    fi
    local current_lines
    current_lines=$(validated_line_count)
    if (( current_lines >= last_attempt_lines + 100 )); then
      if try_materialize >"$MATERIALIZE_LOG" 2>&1; then
        return 0
      fi
      last_attempt_lines=$current_lines
    fi
    if [[ -f "$terminal_marker" ]]; then
      if (( current_lines != last_attempt_lines )); then
        try_materialize >"$MATERIALIZE_LOG" 2>&1 || true
      fi
      final_ready
      return
    fi
    sleep 5
  done
}

run_streaming_batch() {
  local seed=$1
  local target=$2
  local batch_dir="/home/ubuntu/cinematic-dm-raw-$seed"
  local acquisition_done="$batch_dir/.acquisition-complete-$target"
  local m2d_done="$batch_dir/.m2d-complete-$target"
  local asr_done="$batch_dir/.asr-complete-$target"
  mkdir -p "$batch_dir"

  if ! batch_at_target "$batch_dir" "$target"; then
    rm -f "$acquisition_done" "$m2d_done" "$asr_done"
  fi

  (
    if batch_at_target "$batch_dir" "$target"; then
      echo "Acquisition checkpoint $seed/$target already complete"
    else
      acquire_batch "$seed" "$target"
    fi
    touch "$acquisition_done"
  ) &
  local acquisition_pid=$!

  (
    if [[ -f "$m2d_done" ]]; then
      echo "M2D checkpoint $seed/$target already complete"
    else
      follow_m2d "$batch_dir" "$acquisition_done"
      local status=$?
      touch "$m2d_done"
      exit "$status"
    fi
  ) &
  local m2d_pid=$!

  (
    if [[ -f "$asr_done" ]]; then
      echo "ASR checkpoint $seed/$target already complete"
    else
      follow_asr "$batch_dir" "$m2d_done"
      local status=$?
      touch "$asr_done"
      exit "$status"
    fi
  ) &
  local asr_pid=$!

  watch_materialization "$asr_done" &
  local materialize_pid=$!

  wait "$acquisition_pid" || true
  wait "$m2d_pid" || true
  wait "$asr_pid" || true
  wait "$materialize_pid" || true

  if final_ready; then
    return 0
  fi
  try_materialize >"$MATERIALIZE_LOG" 2>&1
}

if run_streaming_batch 20260715 7000; then
  exit 0
fi

for seed in 20260716 20260717 20260718 20260719; do
  for checkpoint in 2500 4500 7000; do
    if run_streaming_batch "$seed" "$checkpoint"; then
      exit 0
    fi
  done
done

echo "Five acquisition batches did not yield 1,000 final clips" >&2
exit 1
