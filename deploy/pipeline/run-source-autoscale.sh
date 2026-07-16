#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT=${SAM_CONTINUOUS_DEPLOY_ROOT:-/home/ubuntu/sam-audio-deploy}
PIPELINE_ROOT="$DEPLOY_ROOT/pipeline"
PIPELINE_PYTHON="$PIPELINE_ROOT/.venv/bin/python"
WORKSPACE=${SAM_CONTINUOUS_WORKSPACE:-/home/ubuntu/cinematic-continuous-30s}
DOWNLOAD_MAX=${SAM_CONTINUOUS_SOURCE_DOWNLOAD_WORKERS:-16}
DOWNLOAD_MIN=2
DOWNLOAD_INITIAL=8
DOWNLOAD_INDEPENDENT=${SAM_CONTINUOUS_SOURCE_DOWNLOAD_INDEPENDENT:-false}
SCAN_MAX=${SAM_CONTINUOUS_SOURCE_SCAN_WORKERS:-4}
EXTRACT_MAX=${SAM_CONTINUOUS_SOURCE_EXTRACT_WORKERS:-16}
REMOTE_TASKS=,${SAM_MEDIA_WORKER_REMOTE_TASKS:-},
SCAN_CPU_EXEMPT=${SAM_CONTINUOUS_SOURCE_SCAN_CPU_EXEMPT:-false}
CPU_EXEMPT_ARGS=()
if [[ "$DOWNLOAD_INDEPENDENT" == "true" ]]; then
  # The transfer pool is controlled only by provider circuits and disk safety.
  # Pin the reported limit at the configured pool size for observability; the
  # downloader itself intentionally does not consume this control file.
  DOWNLOAD_MIN=$DOWNLOAD_MAX
  DOWNLOAD_INITIAL=$DOWNLOAD_MAX
fi
if [[ "$REMOTE_TASKS" == *,download,* ]]; then
  CPU_EXEMPT_ARGS+=(--cpu-exempt-stage download)
fi
if [[ "$REMOTE_TASKS" == *,extract,* ]]; then
  CPU_EXEMPT_ARGS+=(--cpu-exempt-stage extract)
fi
if [[ "$SCAN_CPU_EXEMPT" == "true" && "$REMOTE_TASKS" == *,ffmpeg,* ]]; then
  # Whole-source decode/proxy work runs on the media host. Local M2D work is
  # independently bounded by the scanner's inference semaphore, so keeping
  # more scan threads in flight fills the remote CPU without multiplying GPU
  # inference concurrency.
  CPU_EXEMPT_ARGS+=(--cpu-exempt-stage scan)
fi

export PYTHONPATH="$PIPELINE_ROOT/src"
cd "$PIPELINE_ROOT"

exec "$PIPELINE_PYTHON" -m sam_audio_pipeline.source_pipeline autoscale \
  --workspace "$WORKSPACE" \
  --control-file "$WORKSPACE/source-stage-control.json" \
  --download-min "$DOWNLOAD_MIN" --download-max "$DOWNLOAD_MAX" --download-initial "$DOWNLOAD_INITIAL" \
  --scan-min 1 --scan-max "$SCAN_MAX" --scan-initial 2 \
  --extract-min 1 --extract-max "$EXTRACT_MAX" --extract-initial 4 \
  --cpu-low 55 --cpu-high 85 \
  --scan-backlog-high 16 \
  --extract-backlog-high 8 \
  --download-backlog-low 4 \
  "${CPU_EXEMPT_ARGS[@]}" \
  --interval-seconds 10
