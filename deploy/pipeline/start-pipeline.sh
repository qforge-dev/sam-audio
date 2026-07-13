#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/sam-audio-deploy/pipeline
set -a
source /etc/sam-audio-pipeline.env
set +a

exec /home/ubuntu/sam-audio-deploy/pipeline/.venv/bin/"$@"

