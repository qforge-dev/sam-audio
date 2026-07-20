"""Single-model localhost API for Audio Flamingo Next."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import queue
import threading
import time
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

for _name in ("float8_e8m0fnu", "float8_e4m3fnuz", "float8_e5m2fnuz"):
    if not hasattr(torch, _name):
        setattr(torch, _name, torch.uint8)

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor  # noqa: E402

MODEL = os.environ.get("AFNEXT_MODEL", "nvidia/audio-flamingo-next-think-hf")
MAX_NEW_TOKENS = int(os.environ.get("AFNEXT_MAX_NEW_TOKENS", "512"))
GENERATION_HARD_LIMIT = int(
    os.environ.get("AFNEXT_GENERATION_HARD_LIMIT", "512")
)
MAX_BATCH_SIZE = int(os.environ.get("AFNEXT_MAX_BATCH_SIZE", "1"))
BATCH_WAIT_MS = float(os.environ.get("AFNEXT_BATCH_WAIT_MS", "20"))
HOST = os.environ.get("AFNEXT_HOST", "127.0.0.1")
PORT = int(os.environ.get("AFNEXT_PORT", "8001"))
logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    audio_path: str
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=MAX_NEW_TOKENS, ge=1, le=4096)


class ModelState:
    processor: Any | None = None
    model: Any | None = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None
    fatal_error: str | None = None
    batcher: CaptionBatcher | None = None


state = ModelState()
load_lock = threading.Lock()
inference_lock = threading.Lock()


def load_model() -> None:
    if state.model is not None:
        return
    with load_lock:
        if state.model is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("Audio Flamingo requires CUDA in this deployment")
        processor = AutoProcessor.from_pretrained(MODEL)
        config = AutoConfig.from_pretrained(MODEL)
        if getattr(config, "model_type", "") != "musicflamingo":
            raise RuntimeError(
                f"Unexpected Audio Flamingo model type: {config.model_type}"
            )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        state.processor = processor
        state.model = model
        state.device = next(model.parameters()).device
        state.dtype = next(model.parameters()).dtype


def ask_many(requests: list[AskRequest]) -> list[str]:
    if not requests:
        return []
    load_model()
    assert state.processor is not None
    assert state.model is not None
    assert state.device is not None
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.prompt},
                    {"type": "audio", "path": request.audio_path},
                ],
            }
        ]
        for request in requests
    ]
    batch = state.processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        padding=True,
    ).to(state.device)
    if "input_features" in batch:
        batch["input_features"] = batch["input_features"].to(state.dtype)
    amp = (
        torch.autocast("cuda", dtype=state.dtype)
        if state.dtype == torch.bfloat16
        else nullcontext()
    )
    with inference_lock, torch.inference_mode(), amp:
        generated = state.model.generate(
            **batch,
            # Music Flamingo's learned position table is 1,200 tokens. Keep the
            # hard limit below that boundary: 512 truncated structured captions
            # in production, while 768 leaves room for the complete JSON object.
            max_new_tokens=min(
                max(request.max_new_tokens for request in requests),
                GENERATION_HARD_LIMIT,
            ),
            repetition_penalty=1.15,
        )
    if bool(getattr(state.model.config, "is_encoder_decoder", False)):
        completion = generated
    else:
        prompt_length = batch["input_ids"].shape[1]
        completion = generated[:, prompt_length:]
    return [
        text.strip()
        for text in state.processor.batch_decode(
            completion, skip_special_tokens=True
        )
    ]


def ask(audio_path: Path, prompt: str, max_new_tokens: int) -> str:
    return ask_many(
        [
            AskRequest(
                audio_path=str(audio_path),
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        ]
    )[0]


class CaptionBatcher:
    def __init__(self, max_batch_size: int, wait_ms: float) -> None:
        self.max_batch_size = max(1, max_batch_size)
        self.wait_seconds = max(0.0, wait_ms / 1000.0)
        self.items: queue.Queue[
            tuple[AskRequest, concurrent.futures.Future[str]] | None
        ] = queue.Queue()
        self.thread = threading.Thread(
            target=self._run, name="audio-flamingo-batcher", daemon=True
        )
        self.thread.start()

    def submit(self, request: AskRequest) -> str:
        future: concurrent.futures.Future[str] = concurrent.futures.Future()
        self.items.put((request, future))
        return future.result()

    def close(self) -> None:
        self.items.put(None)
        self.thread.join()

    def _run(self) -> None:
        while True:
            first = self.items.get()
            if first is None:
                return
            batch = [first]
            deadline = time.monotonic() + self.wait_seconds
            while len(batch) < self.max_batch_size:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    item = self.items.get(timeout=timeout)
                except queue.Empty:
                    break
                if item is None:
                    self.items.put(None)
                    break
                batch.append(item)
            try:
                outputs = ask_many([request for request, _ in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError("Audio Flamingo batch output count mismatch")
                for (_, future), output in zip(batch, outputs, strict=True):
                    future.set_result(output)
            except Exception as error:
                for _, future in batch:
                    future.set_exception(error)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    state.batcher = CaptionBatcher(MAX_BATCH_SIZE, BATCH_WAIT_MS)
    try:
        yield
    finally:
        state.batcher.close()
        state.batcher = None


app = FastAPI(title="Audio Flamingo Next API", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
def health() -> dict[str, Any]:
    return {
        "status": (
            "unhealthy"
            if state.fatal_error
            else "ready" if state.model is not None else "loading"
        ),
        "model": MODEL,
        "device": str(state.device) if state.device else None,
        "dtype": str(state.dtype) if state.dtype else None,
        "generation_hard_limit": GENERATION_HARD_LIMIT,
        "max_batch_size": MAX_BATCH_SIZE,
        "batch_wait_ms": BATCH_WAIT_MS,
    }


def _fatal_cuda_error(error: Exception) -> bool:
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "cuda error",
            "device-side assert",
            "index out of bounds",
            "cublas_status",
            "cudnn_status",
        )
    )


def _restart_after_response() -> None:
    time.sleep(0.25)
    os._exit(70)


@app.post("/v1/ask")
def ask_endpoint(request: AskRequest) -> dict[str, Any]:
    path = Path(request.audio_path).resolve()
    if not path.is_file():
        raise HTTPException(status_code=422, detail="Audio file does not exist")
    try:
        if state.batcher is None:
            raise RuntimeError("Audio Flamingo batcher is not ready")
        text = state.batcher.submit(request)
    except Exception as error:
        if _fatal_cuda_error(error):
            state.fatal_error = f"{type(error).__name__}: {error}"[-1000:]
            logger.exception(
                "Fatal CUDA inference error; exiting so systemd can reload the model"
            )
            threading.Thread(target=_restart_after_response, daemon=True).start()
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"model": MODEL, "prompt": request.prompt, "text": text}


def main() -> None:
    uvicorn.run(
        "sam_audio_pipeline.flamingo_api:app",
        host=HOST,
        port=PORT,
        workers=1,
    )


if __name__ == "__main__":
    main()
