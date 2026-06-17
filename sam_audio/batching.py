# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from __future__ import annotations

import concurrent.futures
import itertools
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sam_audio.processor import Anchor, Batch

if TYPE_CHECKING:
    from sam_audio.model.model import SeparationResult


_STOP = object()


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


@dataclass
class ContinuousBatcherConfig:
    max_batch_size: int = 4
    max_active_requests: Optional[int] = None
    max_queue_size: int = 64
    preprocess_workers: Optional[int] = None
    postprocess_workers: Optional[int] = None
    fixed_midpoint_steps: int = 16
    midpoint_options: dict[str, Any] = field(default_factory=dict)
    predict_spans: bool = False
    initial_candidates: int = 1
    max_candidates: int = 1
    margin: float = 0.05
    dtype: Optional[torch.dtype] = None
    pin_memory: bool = True
    non_blocking_transfer: bool = True
    scheduler_idle_sleep_ms: float = 1.0
    completion_callback: Optional[Callable[[SeparationResult], Any]] = None

    def __post_init__(self):
        if self.max_batch_size < 1:
            raise ValueError("`max_batch_size` must be >= 1")
        if self.max_queue_size < 1:
            raise ValueError("`max_queue_size` must be >= 1")
        if self.fixed_midpoint_steps < 1:
            raise ValueError("`fixed_midpoint_steps` must be >= 1")
        if self.initial_candidates < 1:
            raise ValueError("`initial_candidates` must be >= 1")
        if self.max_candidates < self.initial_candidates:
            raise ValueError("`max_candidates` must be >= `initial_candidates`")

        cpu_count = os.cpu_count() or 1
        available_workers = max(1, cpu_count - 4)
        if self.preprocess_workers is None:
            self.preprocess_workers = max(1, available_workers // 2)
        if self.postprocess_workers is None:
            self.postprocess_workers = max(
                1, available_workers - self.preprocess_workers
            )
        if self.max_active_requests is None:
            self.max_active_requests = max(self.max_batch_size, self.max_batch_size * 4)


@dataclass
class ContinuousBatcherMetrics:
    submitted: int = 0
    preprocessed: int = 0
    admitted: int = 0
    completed: int = 0
    failed: int = 0
    generation_steps: int = 0
    gpu_batches: int = 0
    preprocess_ms: float = 0.0
    prepare_ms: float = 0.0
    step_ms: float = 0.0
    decode_ms: float = 0.0
    score_ms: float = 0.0
    postprocess_ms: float = 0.0
    gpu_ready_starved: int = 0
    raw_queue_depth: int = 0
    gpu_ready_queue_depth: int = 0
    active_requests: int = 0
    postprocess_pending: int = 0


@dataclass
class CascadeResult:
    stage1: "SeparationResult"
    stage2: "SeparationResult"


@dataclass
class _RawRequest:
    request_id: int
    future: concurrent.futures.Future
    audio: str | torch.Tensor
    description: str
    anchors: Optional[list[Anchor]]
    masked_video: Optional[str | torch.Tensor]
    seed: Optional[int]
    fixed_midpoint_steps: Optional[int]
    initial_candidates: Optional[int]
    max_candidates: Optional[int]
    margin: Optional[float]
    submitted_ms: float


@dataclass
class _ReadyRequest:
    raw: _RawRequest
    batch: Batch
    feature_length: int
    ready_ms: float


@dataclass
class _ActiveRequest:
    raw: _RawRequest
    prepared: Any
    generated: torch.Tensor
    step_index: int
    candidates: int
    max_candidates: int
    margin: float
    fixed_midpoint_steps: int
    stage: str
    admitted_ms: float
    initial_target_wavs: Optional[list[torch.Tensor]] = None
    initial_residual_wavs: Optional[list[torch.Tensor]] = None
    initial_scores: Optional[torch.Tensor] = None

class ContinuousSAMAudioBatcher:
    """Thread-safe continuous batcher for one loaded SAMAudio model instance.

    The batcher owns all model/CUDA execution on a scheduler thread. Callers may
    submit from many threads; work is admitted to GPU generation between
    fixed-midpoint ODE steps.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        config: Optional[ContinuousBatcherConfig] = None,
    ):
        self.model = model
        self.processor = processor
        self.config = config or ContinuousBatcherConfig()
        self.device = self._infer_device()

        self._raw_queue: queue.Queue[_RawRequest | object] = queue.Queue(
            maxsize=self.config.max_queue_size
        )
        self._gpu_ready_queue: queue.Queue[_ReadyRequest | object] = queue.Queue(
            maxsize=self.config.max_queue_size
        )
        self._active: list[_ActiveRequest] = []
        self._request_ids = itertools.count()
        self._closed = threading.Event()
        self._metrics_lock = threading.Lock()
        self._metrics = ContinuousBatcherMetrics()
        self._postprocess_pending = 0

        self._preprocess_threads = [
            threading.Thread(
                target=self._preprocess_loop,
                name=f"sam-audio-preprocess-{idx}",
                daemon=True,
            )
            for idx in range(int(self.config.preprocess_workers or 1))
        ]
        self._postprocess_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(self.config.postprocess_workers or 1),
            thread_name_prefix="sam-audio-postprocess",
        )
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="sam-audio-gpu-scheduler",
            daemon=True,
        )

        for thread in self._preprocess_threads:
            thread.start()
        self._scheduler_thread.start()

    def submit(
        self,
        audio: str | torch.Tensor,
        description: str,
        anchors: Optional[list[Anchor]] = None,
        masked_video: Optional[str | torch.Tensor] = None,
        seed: Optional[int] = None,
        fixed_midpoint_steps: Optional[int] = None,
        initial_candidates: Optional[int] = None,
        max_candidates: Optional[int] = None,
        margin: Optional[float] = None,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> concurrent.futures.Future:
        if self._closed.is_set():
            raise RuntimeError("ContinuousSAMAudioBatcher is closed")

        future: concurrent.futures.Future = concurrent.futures.Future()
        request = _RawRequest(
            request_id=next(self._request_ids),
            future=future,
            audio=audio,
            description=description,
            anchors=anchors,
            masked_video=masked_video,
            seed=seed,
            fixed_midpoint_steps=fixed_midpoint_steps,
            initial_candidates=initial_candidates,
            max_candidates=max_candidates,
            margin=margin,
            submitted_ms=_now_ms(),
        )
        self._raw_queue.put(request, block=block, timeout=timeout)
        with self._metrics_lock:
            self._metrics.submitted += 1
            self._refresh_queue_metrics_locked()
        return future

    def separate(
        self,
        audio: str | torch.Tensor,
        description: str,
        anchors: Optional[list[Anchor]] = None,
        masked_video: Optional[str | torch.Tensor] = None,
        seed: Optional[int] = None,
        fixed_midpoint_steps: Optional[int] = None,
        initial_candidates: Optional[int] = None,
        max_candidates: Optional[int] = None,
        margin: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> SeparationResult:
        return self.submit(
            audio=audio,
            description=description,
            anchors=anchors,
            masked_video=masked_video,
            seed=seed,
            fixed_midpoint_steps=fixed_midpoint_steps,
            initial_candidates=initial_candidates,
            max_candidates=max_candidates,
            margin=margin,
        ).result(timeout=timeout)

    def submit_cascade(
        self,
        audio: str | torch.Tensor,
        stage1_description: str,
        stage2_description: str,
        stage1_anchors: Optional[list[Anchor]] = None,
        stage2_anchors: Optional[list[Anchor]] = None,
        masked_video: Optional[str | torch.Tensor] = None,
        seed: Optional[int] = None,
        stage1_fixed_midpoint_steps: Optional[int] = None,
        stage2_fixed_midpoint_steps: Optional[int] = None,
        stage1_initial_candidates: Optional[int] = None,
        stage1_max_candidates: Optional[int] = None,
        stage1_margin: Optional[float] = None,
        stage2_initial_candidates: Optional[int] = None,
        stage2_max_candidates: Optional[int] = None,
        stage2_margin: Optional[float] = None,
    ) -> concurrent.futures.Future:
        outer: concurrent.futures.Future = concurrent.futures.Future()
        stage1_future = self.submit(
            audio=audio,
            description=stage1_description,
            anchors=stage1_anchors,
            masked_video=masked_video,
            seed=seed,
            fixed_midpoint_steps=stage1_fixed_midpoint_steps,
            initial_candidates=stage1_initial_candidates,
            max_candidates=stage1_max_candidates,
            margin=stage1_margin,
        )

        def submit_stage2(done_future: concurrent.futures.Future):
            if outer.cancelled():
                return
            try:
                stage1 = done_future.result()
                residual = stage1.residual[0]
                if torch.is_tensor(residual) and residual.ndim == 1:
                    residual = residual.unsqueeze(0)
                stage2_future = self.submit(
                    audio=residual,
                    description=stage2_description,
                    anchors=stage2_anchors,
                    masked_video=masked_video,
                    seed=None if seed is None else seed + 1,
                    fixed_midpoint_steps=stage2_fixed_midpoint_steps,
                    initial_candidates=stage2_initial_candidates,
                    max_candidates=stage2_max_candidates,
                    margin=stage2_margin,
                )

                def finish(stage2_done: concurrent.futures.Future):
                    if outer.cancelled():
                        return
                    try:
                        outer.set_result(
                            CascadeResult(stage1=stage1, stage2=stage2_done.result())
                        )
                    except Exception as exc:
                        outer.set_exception(exc)

                stage2_future.add_done_callback(finish)
            except Exception as exc:
                outer.set_exception(exc)

        stage1_future.add_done_callback(submit_stage2)
        return outer

    def separate_cascade(
        self,
        audio: str | torch.Tensor,
        stage1_description: str,
        stage2_description: str,
        stage1_anchors: Optional[list[Anchor]] = None,
        stage2_anchors: Optional[list[Anchor]] = None,
        masked_video: Optional[str | torch.Tensor] = None,
        seed: Optional[int] = None,
        stage1_fixed_midpoint_steps: Optional[int] = None,
        stage2_fixed_midpoint_steps: Optional[int] = None,
        stage1_initial_candidates: Optional[int] = None,
        stage1_max_candidates: Optional[int] = None,
        stage1_margin: Optional[float] = None,
        stage2_initial_candidates: Optional[int] = None,
        stage2_max_candidates: Optional[int] = None,
        stage2_margin: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> CascadeResult:
        return self.submit_cascade(
            audio=audio,
            stage1_description=stage1_description,
            stage2_description=stage2_description,
            stage1_anchors=stage1_anchors,
            stage2_anchors=stage2_anchors,
            masked_video=masked_video,
            seed=seed,
            stage1_fixed_midpoint_steps=stage1_fixed_midpoint_steps,
            stage2_fixed_midpoint_steps=stage2_fixed_midpoint_steps,
            stage1_initial_candidates=stage1_initial_candidates,
            stage1_max_candidates=stage1_max_candidates,
            stage1_margin=stage1_margin,
            stage2_initial_candidates=stage2_initial_candidates,
            stage2_max_candidates=stage2_max_candidates,
            stage2_margin=stage2_margin,
        ).result(timeout=timeout)

    def close(self, wait: bool = True):
        if self._closed.is_set():
            return
        self._closed.set()
        if not wait:
            self._cancel_pending_raw()
        for _ in self._preprocess_threads:
            self._raw_queue.put(_STOP)
        if wait:
            for thread in self._preprocess_threads:
                thread.join()
            self._scheduler_thread.join()
            self._postprocess_pool.shutdown(wait=True)
        else:
            self._postprocess_pool.shutdown(wait=False, cancel_futures=True)

    def _cancel_pending_raw(self):
        while True:
            try:
                item = self._raw_queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, _RawRequest) and not item.future.done():
                item.future.set_exception(
                    RuntimeError("ContinuousSAMAudioBatcher closed")
                )
            self._raw_queue.task_done()

    def metrics(self) -> ContinuousBatcherMetrics:
        with self._metrics_lock:
            self._refresh_queue_metrics_locked()
            return ContinuousBatcherMetrics(**self._metrics.__dict__)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(wait=True)

    def _infer_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _preprocess_loop(self):
        while True:
            item = self._raw_queue.get()
            if item is _STOP:
                self._raw_queue.task_done()
                self._gpu_ready_queue.put(_STOP)
                return
            request = item
            assert isinstance(request, _RawRequest)
            start = _now_ms()
            try:
                batch = self.processor(
                    audios=[request.audio],
                    descriptions=[request.description],
                    anchors=[request.anchors] if request.anchors is not None else None,
                    masked_videos=(
                        [request.masked_video]
                        if request.masked_video is not None
                        else None
                    ),
                )
                self._pin_batch(batch)
                feature_length = int(batch.sizes.max().item())
                self._gpu_ready_queue.put(
                    _ReadyRequest(
                        raw=request,
                        batch=batch,
                        feature_length=feature_length,
                        ready_ms=_now_ms(),
                    )
                )
                with self._metrics_lock:
                    self._metrics.preprocessed += 1
                    self._metrics.preprocess_ms += _now_ms() - start
                    self._refresh_queue_metrics_locked()
            except Exception as exc:
                request.future.set_exception(exc)
                with self._metrics_lock:
                    self._metrics.failed += 1
            finally:
                self._raw_queue.task_done()

    def _pin_batch(self, batch: Batch):
        if not self.config.pin_memory or not torch.cuda.is_available():
            return
        for name in (
            "audios",
            "sizes",
            "wav_sizes",
            "audio_pad_mask",
            "anchor_ids",
            "anchor_alignment",
        ):
            value = getattr(batch, name, None)
            if torch.is_tensor(value):
                setattr(batch, name, value.pin_memory())
        if batch.masked_video is not None:
            batch.masked_video = [
                video.pin_memory() if torch.is_tensor(video) else video
                for video in batch.masked_video
            ]

    def _scheduler_loop(self):
        preprocess_stops = 0
        expected_stops = len(self._preprocess_threads)
        try:
            while True:
                self._admit_ready_requests(non_blocking=bool(self._active))
                if not self._active:
                    if self._closed.is_set() and preprocess_stops >= expected_stops:
                        break
                    try:
                        item = self._gpu_ready_queue.get(
                            timeout=self.config.scheduler_idle_sleep_ms / 1000.0
                        )
                    except queue.Empty:
                        with self._metrics_lock:
                            self._metrics.gpu_ready_starved += 1
                        continue
                    if item is _STOP:
                        preprocess_stops += 1
                        self._gpu_ready_queue.task_done()
                        continue
                    self._admit_one(item)
                    self._gpu_ready_queue.task_done()
                    continue

                group = self._select_step_group()
                if not group:
                    time.sleep(self.config.scheduler_idle_sleep_ms / 1000.0)
                    continue
                self._run_generation_step(group)
                self._complete_finished_generation()

                while True:
                    try:
                        item = self._gpu_ready_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is _STOP:
                        preprocess_stops += 1
                        self._gpu_ready_queue.task_done()
                        continue
                    if len(self._active) >= int(self.config.max_active_requests or 1):
                        self._gpu_ready_queue.put(item)
                        break
                    self._admit_one(item)
                    self._gpu_ready_queue.task_done()
        except Exception as exc:
            self._fail_all(exc)

    def _admit_ready_requests(self, non_blocking: bool):
        while len(self._active) < int(self.config.max_active_requests or 1):
            try:
                if non_blocking:
                    item = self._gpu_ready_queue.get_nowait()
                else:
                    item = self._gpu_ready_queue.get(
                        timeout=self.config.scheduler_idle_sleep_ms / 1000.0
                    )
            except queue.Empty:
                return
            if item is _STOP:
                self._gpu_ready_queue.task_done()
                self._gpu_ready_queue.put(_STOP)
                return
            self._admit_one(item)
            self._gpu_ready_queue.task_done()

    def _admit_one(self, item: _ReadyRequest | object):
        assert isinstance(item, _ReadyRequest)
        if item.raw.future.cancelled():
            return
        try:
            start = _now_ms()
            batch = item.batch.to(
                self.device, non_blocking=self.config.non_blocking_transfer
            )
            if self.config.dtype is not None and batch.audios.is_floating_point():
                batch.audios = batch.audios.to(self.config.dtype)
            if item.raw.seed is not None:
                torch.manual_seed(item.raw.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(item.raw.seed)
            initial_candidates = (
                item.raw.initial_candidates
                if item.raw.initial_candidates is not None
                else self.config.initial_candidates
            )
            max_candidates = (
                item.raw.max_candidates
                if item.raw.max_candidates is not None
                else self.config.max_candidates
            )
            fixed_midpoint_steps = (
                item.raw.fixed_midpoint_steps
                if item.raw.fixed_midpoint_steps is not None
                else self.config.fixed_midpoint_steps
            )
            margin = item.raw.margin if item.raw.margin is not None else self.config.margin
            if fixed_midpoint_steps < 1:
                raise ValueError("`fixed_midpoint_steps` must be >= 1")
            if initial_candidates < 1:
                raise ValueError("`initial_candidates` must be >= 1")
            if max_candidates < initial_candidates:
                raise ValueError("`max_candidates` must be >= `initial_candidates`")
            with torch.inference_mode():
                prepared = self.model.prepare_audio(
                    batch,
                    candidates=initial_candidates,
                    predict_spans=self.config.predict_spans,
                )
            generated = torch.randn_like(prepared.forward_args["audio_features"])
            self._active.append(
                _ActiveRequest(
                    raw=item.raw,
                    prepared=prepared,
                    generated=generated,
                    step_index=0,
                    candidates=initial_candidates,
                    max_candidates=max_candidates,
                    margin=margin,
                    fixed_midpoint_steps=fixed_midpoint_steps,
                    stage="initial",
                    admitted_ms=_now_ms(),
                )
            )
            with self._metrics_lock:
                self._metrics.admitted += 1
                self._metrics.prepare_ms += _now_ms() - start
                self._refresh_queue_metrics_locked()
        except Exception as exc:
            item.raw.future.set_exception(exc)
            with self._metrics_lock:
                self._metrics.failed += 1

    def _select_step_group(self) -> list[_ActiveRequest]:
        groups: dict[tuple[Any, ...], list[_ActiveRequest]] = {}
        for active in self._active:
            key = self._active_key(active)
            groups.setdefault(key, []).append(active)
        if not groups:
            return []
        candidates = sorted(
            groups.values(),
            key=lambda group: (
                min(item.step_index for item in group),
                min(item.admitted_ms for item in group),
                -len(group),
            ),
        )
        return candidates[0][: self.config.max_batch_size]

    def _active_key(self, active: _ActiveRequest) -> tuple[Any, ...]:
        def tensor_key(value):
            if value is None:
                return None
            return (tuple(value.shape[1:]), value.dtype, str(value.device))

        return (
            active.stage,
            active.candidates,
            active.max_candidates,
            active.fixed_midpoint_steps,
            active.step_index,
            tuple(
                (name, tensor_key(value))
                for name, value in sorted(active.prepared.forward_args.items())
            ),
            tensor_key(active.prepared.projected_static),
            tensor_key(active.prepared.memory_base),
        )

    def _run_generation_step(self, group: list[_ActiveRequest]):
        start = _now_ms()
        merged = self._merge_prepared([item.prepared for item in group])
        generated = torch.cat([item.generated for item in group], dim=0)
        with torch.inference_mode():
            stepped = self.model.continuous_fixed_midpoint_step(
                merged,
                generated,
                step_index=group[0].step_index,
                steps=group[0].fixed_midpoint_steps,
                options=self.config.midpoint_options,
            )
        offset = 0
        for active in group:
            rows = active.generated.size(0)
            active.generated = stepped[offset : offset + rows]
            active.step_index += 1
            offset += rows
        with self._metrics_lock:
            self._metrics.gpu_batches += 1
            self._metrics.generation_steps += len(group)
            self._metrics.step_ms += _now_ms() - start
            self._refresh_queue_metrics_locked()

    def _merge_prepared(self, prepared_items: list[Any]):
        first = prepared_items[0]
        forward_args = {}
        for key in first.forward_args:
            values = [prepared.forward_args[key] for prepared in prepared_items]
            if values[0] is None:
                forward_args[key] = None
            else:
                forward_args[key] = torch.cat(values, dim=0)
        memory_base = None
        if first.memory_base is not None:
            memory_base = torch.cat(
                [prepared.memory_base for prepared in prepared_items], dim=0
            )
        return type(first)(
            batch=first.batch,
            candidates=first.candidates,
            forward_args=forward_args,
            projected_static=torch.cat(
                [prepared.projected_static for prepared in prepared_items], dim=0
            ),
            memory_base=memory_base,
            predict_spans=first.predict_spans,
        )

    def _complete_finished_generation(self):
        finished = [
            active
            for active in self._active
            if active.step_index >= active.fixed_midpoint_steps
        ]
        for active in finished:
            self._active.remove(active)
            self._finish_generation(active)

    def _finish_generation(self, active: _ActiveRequest):
        try:
            start = _now_ms()
            with torch.inference_mode():
                target_wavs, residual_wavs = self.model.decode_prepared_candidate_wavs(
                    active.prepared, active.generated
                )
            with self._metrics_lock:
                self._metrics.decode_ms += _now_ms() - start

            score_start = _now_ms()
            with torch.inference_mode():
                scores = self.model._score_candidate_wavs(
                    active.prepared.batch, target_wavs, active.candidates
                )
            with self._metrics_lock:
                self._metrics.score_ms += _now_ms() - score_start

            if active.stage == "initial" and self._needs_extra_candidates(active, scores):
                extra_candidates = (
                    active.max_candidates - active.candidates
                )
                with torch.inference_mode():
                    extra_prepared = self.model.prepare_audio(
                        active.prepared.batch,
                        candidates=extra_candidates,
                        predict_spans=False,
                    )
                self._active.append(
                    _ActiveRequest(
                        raw=active.raw,
                        prepared=extra_prepared,
                        generated=torch.randn_like(
                            extra_prepared.forward_args["audio_features"]
                        ),
                        step_index=0,
                        candidates=extra_candidates,
                        max_candidates=active.max_candidates,
                        margin=active.margin,
                        fixed_midpoint_steps=active.fixed_midpoint_steps,
                        stage="extra",
                        admitted_ms=_now_ms(),
                        initial_target_wavs=target_wavs,
                        initial_residual_wavs=residual_wavs,
                        initial_scores=scores,
                    )
                )
                return

            if active.stage == "extra" and active.initial_scores is not None:
                if scores is not None:
                    target_wavs = [
                        torch.cat([initial, extra], dim=0)
                        for initial, extra in zip(
                            active.initial_target_wavs or [],
                            target_wavs,
                            strict=False,
                        )
                    ]
                    residual_wavs = [
                        torch.cat([initial, extra], dim=0)
                        for initial, extra in zip(
                            active.initial_residual_wavs or [],
                            residual_wavs,
                            strict=False,
                        )
                    ]
                    scores = torch.cat([active.initial_scores, scores], dim=1)
                else:
                    scores = active.initial_scores
                    target_wavs = active.initial_target_wavs or target_wavs
                    residual_wavs = active.initial_residual_wavs or residual_wavs

            with torch.inference_mode():
                result = self.model._select_candidate_wavs(
                    target_wavs,
                    residual_wavs,
                    scores,
                    active.generated.device,
                )
            result.noise = active.generated
            self._submit_postprocess(active.raw, result)
        except Exception as exc:
            active.raw.future.set_exception(exc)
            with self._metrics_lock:
                self._metrics.failed += 1

    def _needs_extra_candidates(
        self, active: _ActiveRequest, scores: Optional[torch.Tensor]
    ) -> bool:
        if scores is None:
            return False
        if active.candidates >= active.max_candidates:
            return False
        if active.candidates < 2:
            return True
        top2 = scores.topk(k=2, dim=1).values
        confident = (top2[:, 0] - top2[:, 1]).ge(active.margin)
        return not bool(confident.all())

    def _submit_postprocess(self, request: _RawRequest, result: SeparationResult):
        with self._metrics_lock:
            self._postprocess_pending += 1
            self._refresh_queue_metrics_locked()
        self._postprocess_pool.submit(self._postprocess_result, request, result)

    def _postprocess_result(self, request: _RawRequest, result: SeparationResult):
        start = _now_ms()
        try:
            result = type(result)(
                target=[
                    tensor.detach().float().cpu() if torch.is_tensor(tensor) else tensor
                    for tensor in result.target
                ],
                residual=[
                    tensor.detach().float().cpu() if torch.is_tensor(tensor) else tensor
                    for tensor in result.residual
                ],
                noise=(
                    result.noise.detach().cpu()
                    if torch.is_tensor(result.noise)
                    else result.noise
                ),
            )
            if self.config.completion_callback is not None:
                self.config.completion_callback(result)
            request.future.set_result(result)
            with self._metrics_lock:
                self._metrics.completed += 1
                self._metrics.postprocess_ms += _now_ms() - start
        except Exception as exc:
            request.future.set_exception(exc)
            with self._metrics_lock:
                self._metrics.failed += 1
        finally:
            with self._metrics_lock:
                self._postprocess_pending -= 1
                self._refresh_queue_metrics_locked()

    def _fail_all(self, exc: Exception):
        for active in self._active:
            if not active.raw.future.done():
                active.raw.future.set_exception(exc)
        while True:
            try:
                item = self._gpu_ready_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, _ReadyRequest) and not item.raw.future.done():
                item.raw.future.set_exception(exc)
            self._gpu_ready_queue.task_done()
        with self._metrics_lock:
            self._metrics.failed += len(self._active)
            self._active.clear()
            self._refresh_queue_metrics_locked()

    def _refresh_queue_metrics_locked(self):
        self._metrics.raw_queue_depth = self._raw_queue.qsize()
        self._metrics.gpu_ready_queue_depth = self._gpu_ready_queue.qsize()
        self._metrics.active_requests = len(self._active)
        self._metrics.postprocess_pending = self._postprocess_pending


__all__ = [
    "CascadeResult",
    "ContinuousBatcherConfig",
    "ContinuousBatcherMetrics",
    "ContinuousSAMAudioBatcher",
]
