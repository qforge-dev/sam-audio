# Continuous SAM-Audio Batching

This guide shows how to serve many independent SAM-Audio inference requests through
one loaded model instance. The main entry point is `ContinuousSAMAudioBatcher`.

The batcher is designed for H100-style serving workloads where loading multiple
model instances wastes GPU memory. It keeps one model resident on the GPU, accepts
requests from many caller threads, and schedules GPU work between fixed-midpoint
ODE steps.

## What It Does

- Owns one loaded `SAMAudio` model instance.
- Lets many threads call `submit(...)` concurrently.
- Uses CPU preprocess workers for audio/video decode, resample, masking, and
  `SAMAudioProcessor` batching.
- Uses one GPU scheduler thread for all model calls.
- Uses postprocess workers to copy outputs back to CPU and run optional callbacks.
- Admits new queued work between fixed-midpoint generation steps.
- Supports in-memory two-stage cascade inference so stage 1 residual audio can feed
  stage 2 without saving and reloading an intermediate WAV.

This is not the same as calling `model.separate(...)` with a static batch. Static
batches return only when every row in the batch is done. The continuous batcher can
complete one request and admit another request while a longer active request keeps
running.

## Basic Usage

```python
import torchaudio
import torch

from sam_audio import (
    ContinuousBatcherConfig,
    ContinuousSAMAudioBatcher,
    SAMAudio,
    SAMAudioProcessor,
)


model_id = "facebook/sam-audio-large"

model = SAMAudio.from_pretrained(model_id).eval().cuda()
processor = SAMAudioProcessor.from_pretrained(model_id)

config = ContinuousBatcherConfig(
    max_batch_size=4,
    max_active_requests=16,
    max_queue_size=128,
    fixed_midpoint_steps=16,
    predict_spans=False,
    initial_candidates=1,
    max_candidates=1,
    dtype=torch.float32,
)

with ContinuousSAMAudioBatcher(model, processor, config) as batcher:
    future = batcher.submit(
        audio="input.wav",
        description="human voice",
    )

    result = future.result(timeout=120)

torchaudio.save("target.wav", result.target[0], processor.audio_sampling_rate)
torchaudio.save("residual.wav", result.residual[0], processor.audio_sampling_rate)
```

`submit(...)` returns a `concurrent.futures.Future`. Use `future.result()` in a
blocking caller, or attach callbacks in a server.

## Blocking Convenience API

Use `separate(...)` when the caller does not need to manage a future directly:

```python
with ContinuousSAMAudioBatcher(model, processor, config) as batcher:
    result = batcher.separate(
        audio="input.wav",
        description="drums",
        timeout=120,
    )
```

## Concurrent Callers

The model itself should not be called from multiple application threads. Instead,
all application workers submit work to one batcher.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed


requests = [
    ("chunk_001.wav", "human voice"),
    ("chunk_002.wav", "music"),
    ("chunk_003.wav", "drums"),
    ("chunk_004.wav", "speech"),
]

with ContinuousSAMAudioBatcher(model, processor, config) as batcher:
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [
            pool.submit(
                lambda audio, description: batcher.separate(audio, description),
                audio,
                description,
            )
            for audio, description in requests
        ]

        for future in as_completed(futures):
            result = future.result()
            # Persist, upload, or pass result to the next pipeline stage.
```

All GPU/model execution still happens on the batcher's scheduler thread. The caller
threads only enqueue work and wait for results.

## Two-Stage Cascade Without Disk Round-Trips

For cascade workloads, use `submit_cascade(...)` or `separate_cascade(...)`.
Stage 1 residual audio is passed directly to stage 2 as an in-memory tensor.

```python
with ContinuousSAMAudioBatcher(model, processor, config) as batcher:
    cascade = batcher.separate_cascade(
        audio="mixture.wav",
        stage1_description="music soundtrack",
        stage2_description="human voices",
        timeout=240,
    )

stage1_music = cascade.stage1.target[0]
stage1_residual = cascade.stage1.residual[0]
stage2_voice = cascade.stage2.target[0]
```

Use this path for pipelines that currently save `stage1_residual.wav` and then
load it back for stage 2.

## Adaptive Reranking

The batcher supports shared adaptive-rerank settings at startup:

```python
config = ContinuousBatcherConfig(
    initial_candidates=4,
    max_candidates=8,
    margin=0.05,
    predict_spans=True,
    fixed_midpoint_steps=16,
)
```

The config values are defaults. Individual `submit(...)`, `separate(...)`, and
cascade calls may override fixed-midpoint step count, candidate counts, and
adaptive margin. The batcher uses those values as part of its compatibility key,
so requests with different tensor shapes or generation policies are not merged
into the same GPU step group.

Adaptive rerank is handled per request:

- Initial candidates are generated first.
- Confident requests complete immediately.
- Only unconfident requests continue into extra-candidate generation.

This avoids the static-batch behavior where one uncertain row can force extra work
for every row in the batch.

## Per-Stage Cascade Settings

Cascade calls can override generation and rerank settings independently for
stage 1 and stage 2. This keeps one deployed batcher while allowing cheaper
second-stage extraction when the first-stage residual is already simpler than the
original mixture.

```python
with ContinuousSAMAudioBatcher(model, processor, config) as batcher:
    cascade = batcher.separate_cascade(
        audio="chunk.wav",
        stage1_description="music soundtrack",
        stage2_description="human voices",
        stage1_fixed_midpoint_steps=16,
        stage1_initial_candidates=4,
        stage1_max_candidates=8,
        stage1_margin=0.05,
        stage2_fixed_midpoint_steps=12,
        stage2_initial_candidates=4,
        stage2_max_candidates=4,
        stage2_margin=0.05,
        timeout=240,
    )
```

On the H100 cascade benchmark, the best measured speed/quality candidate was:

- dtype policy: TF32/fp32, with rankers kept fp32
- prompt cascade: `music soundtrack` then `human voices`
- stage 1: 16 fixed-midpoint steps, adaptive 4 -> 8 candidates
- stage 2: 12 fixed-midpoint steps, capped at 4 candidates
- input path: predecode/resample/pin future chunks on CPU
- output path: write artifacts asynchronously outside the GPU scheduler

This beat blanket fp16 for throughput. fp16 reduced VRAM, but it was slower for
the full pipeline because ranking/scoring became more expensive.

## Tuning For H100 Servers

Start conservatively and tune with metrics:

```python
config = ContinuousBatcherConfig(
    max_batch_size=4,
    max_active_requests=16,
    max_queue_size=256,
    preprocess_workers=24,
    postprocess_workers=8,
    fixed_midpoint_steps=16,
    pin_memory=True,
    non_blocking_transfer=True,
)
```

Important knobs:

- `max_batch_size`: maximum compatible active requests per ODE step group.
- `max_active_requests`: total requests admitted into active GPU scheduling.
- `max_queue_size`: backpressure limit for raw and GPU-ready queues.
- `preprocess_workers`: CPU threads for `processor(...)`, audio/video decode, and
  resampling.
- `postprocess_workers`: CPU threads for D2H output copy and optional completion
  callback work.
- `fixed_midpoint_steps`: number of fixed midpoint ODE steps. The batcher admits
  queued work between these steps.
- `pin_memory`: pin CPU tensors before GPU transfer when CUDA is available.
- `non_blocking_transfer`: use non-blocking tensor transfers to the model device.

The default worker split reserves a few CPU cores for the scheduler, server, OS,
and monitoring:

```python
available_workers = max(1, os.cpu_count() - 4)
preprocess_workers = max(1, available_workers // 2)
postprocess_workers = max(1, available_workers - preprocess_workers)
```

On H100 instances, increase `preprocess_workers` until the GPU-ready queue is
usually non-empty. Increase `postprocess_workers` if result copying, WAV encoding,
uploads, or callbacks build up.

## Metrics

Call `metrics()` to inspect pipeline pressure:

```python
stats = batcher.metrics()

print(stats.submitted)
print(stats.completed)
print(stats.raw_queue_depth)
print(stats.gpu_ready_queue_depth)
print(stats.active_requests)
print(stats.gpu_batches)
print(stats.generation_steps)
print(stats.gpu_ready_starved)
```

Useful signals:

- `gpu_ready_starved` increasing quickly means the GPU scheduler is waiting for CPU
  preprocess.
- High `raw_queue_depth` means callers are producing work faster than CPU preprocess.
- High `postprocess_pending` means CPU output work is lagging behind GPU completion.
- Low `gpu_batches` with high request volume can mean requests are not shape
  compatible enough to batch together.

## Backpressure

`submit(...)` accepts the same blocking controls as `queue.Queue.put(...)`:

```python
future = batcher.submit(
    audio="input.wav",
    description="voice",
    block=False,
)
```

If the raw queue is full and `block=False`, `queue.Full` is raised. Use this in an
HTTP service to return a 429/503 instead of letting memory grow without bound.

## Completion Callback

Use `completion_callback` for lightweight postprocess hooks:

```python
def on_complete(result):
    # Keep this short. Heavy upload/write work should use your own executor.
    print(result.target[0].shape)


config = ContinuousBatcherConfig(completion_callback=on_complete)
```

The callback runs in the postprocess worker pool after tensors are detached and
copied to CPU.

## Current Limits

- Continuous batching v1 uses fixed-step midpoint generation.
- Arbitrary adaptive `torchdiffeq.odeint` solvers are not supported in continuous
  mode.
- New requests are admitted between ODE steps, not inside a single transformer
  kernel.
- The scheduler only groups compatible active states with the same stage, candidate
  count, tensor shape, dtype, and device.
- The model instance must be treated as owned by the batcher. Do not call
  `model.separate(...)` from other threads while the batcher is running.

## Recommended Serving Shape

Use one process per GPU:

1. Load one `SAMAudio` model and one `SAMAudioProcessor`.
2. Create one `ContinuousSAMAudioBatcher`.
3. Let HTTP/RPC workers call `batcher.submit(...)`.
4. Save or upload outputs from future callbacks or a separate application executor.
5. Export `batcher.metrics()` and system metrics to your monitoring stack.

For multiple GPUs, run one process and one batcher per GPU, then route requests
above them.
