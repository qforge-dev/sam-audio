from __future__ import annotations

import importlib.util
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_batching_module():
    package = types.ModuleType("sam_audio")
    package.__path__ = [str(ROOT / "sam_audio")]
    sys.modules.setdefault("sam_audio", package)

    processor = types.ModuleType("sam_audio.processor")
    processor.Anchor = tuple

    class Batch:
        def __init__(
            self,
            audios,
            sizes,
            wav_sizes,
            descriptions,
            hop_length,
            audio_sampling_rate,
            anchors=None,
            audio_pad_mask=None,
            masked_video=None,
        ):
            self.audios = audios
            self.sizes = sizes
            self.wav_sizes = wav_sizes
            self.descriptions = descriptions
            self.audio_pad_mask = audio_pad_mask
            self.masked_video = masked_video
            self.hop_length = hop_length
            self.audio_sampling_rate = audio_sampling_rate
            self.process_anchors(anchors)

        def to(self, device, non_blocking=False):
            self.audios = self.audios.to(device, non_blocking=non_blocking)
            self.sizes = self.sizes.to(device, non_blocking=non_blocking)
            self.wav_sizes = self.wav_sizes.to(device, non_blocking=non_blocking)
            self.anchor_ids = self.anchor_ids.to(device, non_blocking=non_blocking)
            self.anchor_alignment = self.anchor_alignment.to(
                device, non_blocking=non_blocking
            )
            if self.audio_pad_mask is not None:
                self.audio_pad_mask = self.audio_pad_mask.to(
                    device, non_blocking=non_blocking
                )
            return self

        def process_anchors(self, anchors):
            self.anchors = anchors
            self.anchor_ids = torch.zeros(self.audios.size(0), 2, dtype=torch.long)
            self.anchor_alignment = torch.zeros(
                self.audios.size(0), self.audio_pad_mask.size(-1), dtype=torch.long
            )

    processor.Batch = Batch
    sys.modules["sam_audio.processor"] = processor

    batching_spec = importlib.util.spec_from_file_location(
        "sam_audio.batching", ROOT / "sam_audio" / "batching.py"
    )
    batching = importlib.util.module_from_spec(batching_spec)
    sys.modules["sam_audio.batching"] = batching
    batching_spec.loader.exec_module(batching)
    return batching, processor


batching, processor_mod = load_batching_module()
Batch = processor_mod.Batch
ContinuousBatcherConfig = batching.ContinuousBatcherConfig
ContinuousSAMAudioBatcher = batching.ContinuousSAMAudioBatcher


@dataclass
class FakePrepared:
    batch: Batch
    candidates: int
    forward_args: dict[str, torch.Tensor]
    projected_static: torch.Tensor
    memory_base: torch.Tensor | None
    predict_spans: bool = False


@dataclass
class FakeResult:
    target: list[torch.Tensor]
    residual: list[torch.Tensor]
    noise: torch.Tensor


class FakeCodec:
    def feature_idx_to_wav_idx(self, sizes):
        return sizes


class FakeProcessor:
    audio_sampling_rate = 48_000

    def __init__(self):
        self.audio_inputs = []

    def __call__(self, audios, descriptions, anchors=None, masked_videos=None):
        del masked_videos
        wavs = []
        sizes = []
        for audio in audios:
            self.audio_inputs.append(audio)
            size = int(audio) if not torch.is_tensor(audio) else int(audio.numel())
            sizes.append(size)
            wavs.append(torch.ones(1, size))
        max_size = max(sizes)
        padded = [
            torch.nn.functional.pad(wav, (0, max_size - wav.size(-1))) for wav in wavs
        ]
        sizes_tensor = torch.tensor(sizes, dtype=torch.long)
        audio_pad_mask = torch.arange(max_size).expand(
            len(sizes), -1
        ) < sizes_tensor.unsqueeze(1)
        return Batch(
            audios=torch.stack(padded, dim=0),
            sizes=sizes_tensor,
            wav_sizes=sizes_tensor,
            descriptions=descriptions,
            audio_pad_mask=audio_pad_mask,
            anchors=anchors,
            masked_video=None,
            hop_length=1,
            audio_sampling_rate=self.audio_sampling_rate,
        )


class FakeModel(torch.nn.Module):
    def __init__(self, step_sleep_scale=0.0):
        super().__init__()
        self.param = torch.nn.Parameter(torch.empty(0))
        self.audio_codec = FakeCodec()
        self.step_sleep_scale = step_sleep_scale
        self.step_log = []

    def prepare_audio(self, batch, candidates=1, predict_spans=False):
        del predict_spans
        feature_length = int(batch.sizes.max().item())
        text_length = max(len(description.split()) for description in batch.descriptions)
        rows = batch.audios.size(0) * candidates
        audio_features = torch.zeros(rows, feature_length, 2)
        return FakePrepared(
            batch=batch,
            candidates=candidates,
            forward_args={
                "audio_features": audio_features,
                "text_features": torch.zeros(rows, text_length, 1),
                "text_mask": torch.ones(rows, text_length, dtype=torch.bool),
                "masked_video_features": torch.zeros(rows, 1, feature_length),
                "anchor_ids": torch.zeros(rows, 2, dtype=torch.long),
                "anchor_alignment": torch.zeros(rows, feature_length, dtype=torch.long),
                "audio_pad_mask": torch.ones(rows, feature_length, dtype=torch.bool),
            },
            projected_static=torch.zeros(rows, feature_length, 1),
            memory_base=torch.zeros(rows, 1, 1),
        )

    def continuous_fixed_midpoint_step(
        self, prepared, generated, step_index, steps, options=None
    ):
        del prepared, steps, options
        feature_length = generated.size(1)
        self.step_log.append(
            (feature_length, step_index, generated.size(0), time.perf_counter())
        )
        if self.step_sleep_scale:
            time.sleep(feature_length * self.step_sleep_scale)
        return generated + 1

    def decode_prepared_candidate_wavs(self, prepared, generated):
        del generated
        size = int(prepared.batch.sizes[0].item())
        target = [torch.full((prepared.candidates, 1), float(size))]
        residual = [torch.full((prepared.candidates, 1), -float(size))]
        return target, residual

    def _score_candidate_wavs(self, batch, target_wavs, candidates):
        del batch, target_wavs
        if candidates <= 1:
            return None
        return torch.arange(candidates, dtype=torch.float32).view(1, candidates)

    def _select_candidate_wavs(self, target_wavs, residual_wavs, scores, device):
        del scores
        return FakeResult(
            target=[target_wavs[0][0].to(device)],
            residual=[residual_wavs[0][0].to(device)],
            noise=torch.empty(0, device=device),
        )


def test_concurrent_submitters_resolve():
    model = FakeModel()
    cfg = ContinuousBatcherConfig(
        max_batch_size=2,
        max_active_requests=4,
        fixed_midpoint_steps=2,
        preprocess_workers=2,
        postprocess_workers=1,
        pin_memory=False,
    )
    batcher = ContinuousSAMAudioBatcher(model, FakeProcessor(), cfg)
    try:
        futures = [batcher.submit(4, f"request {idx}") for idx in range(6)]
        results = [future.result(timeout=5) for future in futures]
        assert len(results) == 6
        assert batcher.metrics().completed == 6
    finally:
        batcher.close()


def test_incompatible_conditioning_shapes_are_not_merged():
    model = FakeModel(step_sleep_scale=0.01)
    cfg = ContinuousBatcherConfig(
        max_batch_size=2,
        max_active_requests=2,
        fixed_midpoint_steps=2,
        preprocess_workers=2,
        postprocess_workers=1,
        pin_memory=False,
    )
    batcher = ContinuousSAMAudioBatcher(model, FakeProcessor(), cfg)
    try:
        short_text = batcher.submit(4, "music")
        long_text = batcher.submit(4, "human voices in foreground")

        assert short_text.result(timeout=5).target[0].item() == 4.0
        assert long_text.result(timeout=5).target[0].item() == 4.0
        assert batcher.metrics().completed == 2
    finally:
        batcher.close()


def test_cascade_accepts_per_stage_generation_settings():
    model = FakeModel()
    fake_processor = FakeProcessor()
    cfg = ContinuousBatcherConfig(
        max_batch_size=1,
        max_active_requests=1,
        fixed_midpoint_steps=16,
        initial_candidates=4,
        max_candidates=8,
        preprocess_workers=1,
        postprocess_workers=1,
        pin_memory=False,
    )
    batcher = ContinuousSAMAudioBatcher(model, fake_processor, cfg)
    try:
        result = batcher.separate_cascade(
            audio=3,
            stage1_description="music soundtrack",
            stage2_description="human voices in foreground",
            stage1_fixed_midpoint_steps=2,
            stage1_initial_candidates=2,
            stage1_max_candidates=2,
            stage2_fixed_midpoint_steps=1,
            stage2_initial_candidates=1,
            stage2_max_candidates=1,
            timeout=5,
        )

        assert result.stage1.target[0].item() == 3.0
        assert result.stage2.target[0].item() == 1.0
        assert any(row[1] == 1 for row in model.step_log)
    finally:
        batcher.close()


def test_short_request_completes_before_long_request_and_new_work_enters():
    model = FakeModel(step_sleep_scale=0.01)
    cfg = ContinuousBatcherConfig(
        max_batch_size=2,
        max_active_requests=2,
        fixed_midpoint_steps=2,
        preprocess_workers=1,
        postprocess_workers=1,
        pin_memory=False,
    )
    batcher = ContinuousSAMAudioBatcher(model, FakeProcessor(), cfg)
    try:
        short = batcher.submit(2, "short")
        long = batcher.submit(8, "long")
        queued_short = batcher.submit(2, "queued-short")

        first_result = short.result(timeout=5)
        queued_result = queued_short.result(timeout=5)
        assert first_result.target[0].item() == 2.0
        assert queued_result.target[0].item() == 2.0
        assert long.result(timeout=5).target[0].item() == 8.0

        second_short_start = [
            index
            for index, row in enumerate(model.step_log)
            if row[0] == 2 and row[1] == 0
        ][1]
        long_final_step = [
            index
            for index, row in enumerate(model.step_log)
            if row[0] == 8 and row[1] == 1
        ][0]
        assert second_short_start < long_final_step
    finally:
        batcher.close()


def test_backpressure_rejects_when_raw_queue_is_full():
    model = FakeModel(step_sleep_scale=0.05)
    cfg = ContinuousBatcherConfig(
        max_batch_size=1,
        max_active_requests=1,
        max_queue_size=1,
        fixed_midpoint_steps=3,
        preprocess_workers=1,
        postprocess_workers=1,
        pin_memory=False,
    )
    batcher = ContinuousSAMAudioBatcher(model, FakeProcessor(), cfg)
    try:
        batcher.submit(5, "first")
        rejected = False
        for idx in range(100):
            try:
                batcher.submit(5, f"queued {idx}", block=False)
            except Exception:
                rejected = True
                break
        assert rejected
    finally:
        batcher.close(wait=False)


def test_cascade_uses_in_memory_residual_for_stage2():
    model = FakeModel()
    fake_processor = FakeProcessor()
    cfg = ContinuousBatcherConfig(
        max_batch_size=1,
        max_active_requests=1,
        fixed_midpoint_steps=1,
        preprocess_workers=1,
        postprocess_workers=1,
        pin_memory=False,
    )
    batcher = ContinuousSAMAudioBatcher(model, fake_processor, cfg)
    try:
        result = batcher.separate_cascade(
            audio=3,
            stage1_description="music",
            stage2_description="voice",
            timeout=5,
        )
        assert result.stage1.residual[0].item() == -3.0
        assert result.stage2.target[0].item() == 1.0
        assert torch.is_tensor(fake_processor.audio_inputs[1])
    finally:
        batcher.close()
