#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/sam-audio-deploy/pipeline

exec /home/ubuntu/sam-audio-deploy/pipeline/.venv/bin/"$@"
