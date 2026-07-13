"""Single-model localhost API for Audio Flamingo Next."""

from __future__ import annotations

import os
import threading
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
MAX_NEW_TOKENS = int(os.environ.get("AFNEXT_MAX_NEW_TOKENS", "768"))


class AskRequest(BaseModel):
    audio_path: str
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=MAX_NEW_TOKENS, ge=1, le=4096)


class ModelState:
    processor: Any | None = None
    model: Any | None = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None


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
            device_map="auto",
            max_memory={0: "30GiB", "cpu": "120GiB"},
        ).eval()
        state.processor = processor
        state.model = model
        state.device = next(model.parameters()).device
        state.dtype = next(model.parameters()).dtype


def ask(audio_path: Path, prompt: str, max_new_tokens: int) -> str:
    load_model()
    assert state.processor is not None
    assert state.model is not None
    assert state.device is not None
    conversation = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": str(audio_path)},
                ],
            }
        ]
    ]
    batch = state.processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
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
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.15,
        )
    prompt_length = batch["input_ids"].shape[1]
    completion = generated[:, prompt_length:]
    return state.processor.batch_decode(completion, skip_special_tokens=True)[0].strip()


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="Audio Flamingo Next API", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
def health() -> dict[str, Any]:
    return {
        "status": "ready" if state.model is not None else "loading",
        "model": MODEL,
        "device": str(state.device) if state.device else None,
        "dtype": str(state.dtype) if state.dtype else None,
    }


@app.post("/v1/ask")
def ask_endpoint(request: AskRequest) -> dict[str, Any]:
    path = Path(request.audio_path).resolve()
    if not path.is_file():
        raise HTTPException(status_code=422, detail="Audio file does not exist")
    try:
        text = ask(path, request.prompt, request.max_new_tokens)
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"model": MODEL, "prompt": request.prompt, "text": text}


def main() -> None:
    uvicorn.run(
        "sam_audio_pipeline.flamingo_api:app",
        host="127.0.0.1",
        port=8001,
        workers=1,
    )


if __name__ == "__main__":
    main()
