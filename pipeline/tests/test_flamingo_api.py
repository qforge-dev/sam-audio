from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("torch")

import sam_audio_pipeline.flamingo_api as flamingo_api
from sam_audio_pipeline.flamingo_api import AskRequest, CaptionBatcher


def test_caption_batcher_coalesces_concurrent_requests(monkeypatch) -> None:
    observed: list[list[str]] = []

    def fake_ask_many(requests: list[AskRequest]) -> list[str]:
        observed.append([request.audio_path for request in requests])
        return [f"caption:{request.audio_path}" for request in requests]

    monkeypatch.setattr(flamingo_api, "ask_many", fake_ask_many)
    batcher = CaptionBatcher(max_batch_size=2, wait_ms=100)
    requests = [
        AskRequest(audio_path=f"audio-{index}.wav", prompt="describe")
        for index in range(2)
    ]
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            outputs = list(executor.map(batcher.submit, requests))
    finally:
        batcher.close()

    assert outputs == ["caption:audio-0.wav", "caption:audio-1.wav"]
    assert observed == [["audio-0.wav", "audio-1.wav"]]
