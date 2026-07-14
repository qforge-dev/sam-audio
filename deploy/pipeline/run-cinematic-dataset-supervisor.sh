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

export PYTHONPATH="$PIPELINE_ROOT/src"
export HF_HOME=/home/ubuntu/.cache/huggingface
cd "$PIPELINE_ROOT"

wait_for_initial_batch() {
  while pgrep -f \
    "sam_audio_pipeline.youtube_random.*cinematic-dm-raw-20260715" \
    >/dev/null; do
    sleep 30
  done
}

acquire_batch() {
  local seed=$1
  local batch_dir="/home/ubuntu/cinematic-dm-raw-$seed"
  "$PIPELINE_PYTHON" -m sam_audio_pipeline.youtube_random \
    --output "$batch_dir" \
    --source dailymotion \
    --profile cinematic \
    --total 7000 \
    --seed "$seed" \
    --clips-per-video 12 \
    --query-count 500 \
    --results-per-query 100 \
    --search-workers 8 \
    --download-workers 16 \
    --candidate-multiplier 2.0 \
    --max-attempts 16000 || true
}

score_batch() {
  local batch_dir=$1
  "$MODEL_PYTHON" -m sam_audio_pipeline.m2d_validator score \
    --input-dir "$batch_dir/audio" \
    --output "$batch_dir/m2d-validation.jsonl" \
    --m2d-repo "$M2D_REPO" \
    --checkpoint "$M2D_CHECKPOINT" \
    --class-labels "$CLASS_LABELS" \
    --ontology "$ONTOLOGY" \
    --m2d-commit "$M2D_COMMIT" \
    --require-cinematic-mix
  "$WHISPER_PYTHON" -m sam_audio_pipeline.m2d_validator asr-score \
    --input-dir "$batch_dir/audio" \
    --output "$batch_dir/asr-validation.jsonl" \
    --model small \
    --download-root /home/ubuntu/.cache/huggingface/faster-whisper
}

try_materialize() {
  local args=()
  local seed
  for seed in 20260715 20260716 20260717; do
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
    --max-clips-per-video 3 \
    --seed 20260715 \
    --require-cinematic-mix
}

wait_for_initial_batch
acquire_batch 20260715
score_batch /home/ubuntu/cinematic-dm-raw-20260715
if try_materialize; then
  exit 0
fi

for seed in 20260716 20260717; do
  acquire_batch "$seed"
  score_batch "/home/ubuntu/cinematic-dm-raw-$seed"
  if try_materialize; then
    exit 0
  fi
done

echo "Three acquisition batches did not yield 1,000 final clips" >&2
exit 1
