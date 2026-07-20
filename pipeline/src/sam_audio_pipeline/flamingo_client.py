"""Client for the local Audio Flamingo Next model service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx


class AudioFlamingoClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def ask(
        self,
        audio_path: Path,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "audio_path": str(audio_path),
            "prompt": prompt,
        }
        if max_new_tokens is not None:
            payload["max_new_tokens"] = max_new_tokens
        with httpx.Client(timeout=None) as client:
            response = client.post(
                f"{self.base_url}/v1/ask",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
