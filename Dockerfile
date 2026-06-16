# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.14-slim-bookworm
ARG UV_VERSION=0.9.2

FROM ${PYTHON_IMAGE} AS runtime

ARG UV_VERSION
ARG UV_TORCH_BACKEND=cu129

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /uvx /usr/local/bin/

WORKDIR /opt/sam-audio

COPY pyproject.toml pylock.toml README.md LICENSE ./
COPY sam_audio ./sam_audio

RUN uv pip sync pylock.toml --system --torch-backend "${UV_TORCH_BACKEND}" --strict \
    && uv pip install --system --no-deps --no-build-isolation .

CMD ["python", "-c", "import torch, sam_audio; print(f'sam_audio image ready: torch {torch.__version__}')"]
