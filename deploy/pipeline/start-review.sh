#!/usr/bin/env bash
set -euo pipefail

deploy_root="${SAM_PIPELINE_DEPLOY_ROOT:-/home/ubuntu/sam-audio-deploy}"
cd "${deploy_root}/pipeline"
export PYTHONPATH="${deploy_root}/pipeline/src${PYTHONPATH:+:${PYTHONPATH}}"

progress_args=()
if [[ -n "${SAM_REVIEW_PROGRESS_BATCH_DIRS:-}" ]]; then
  IFS=',' read -r -a progress_dirs <<< "${SAM_REVIEW_PROGRESS_BATCH_DIRS}"
  for progress_dir in "${progress_dirs[@]}"; do
    progress_args+=(--progress-batch-dir "${progress_dir}")
  done
  progress_args+=(--progress-target "${SAM_REVIEW_PROGRESS_TARGET:-1000}")
  if [[ -n "${SAM_REVIEW_PROGRESS_FINAL_DIR:-}" ]]; then
    progress_args+=(--progress-final-dir "${SAM_REVIEW_PROGRESS_FINAL_DIR}")
  fi
fi

exec "${SAM_PIPELINE_PYTHON:-${deploy_root}/pipeline/.venv/bin/python}" \
  -m sam_audio_pipeline.review_app \
  --dataset-dir "${SAM_REVIEW_DATASET_DIR:?SAM_REVIEW_DATASET_DIR is required}" \
  --audio-directory "${SAM_REVIEW_AUDIO_DIRECTORY:-balanced-audio}" \
  --annotations "${SAM_REVIEW_ANNOTATIONS:-${SAM_REVIEW_DATASET_DIR}/manual-review.json}" \
  --host "${SAM_REVIEW_HOST:-127.0.0.1}" \
  --port "${SAM_REVIEW_PORT:-18081}" \
  --claim-seconds "${SAM_REVIEW_CLAIM_SECONDS:-600}" \
  "${progress_args[@]}"
