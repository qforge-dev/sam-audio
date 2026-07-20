"""HTTP inference service for SAM Audio."""

import asyncio
import io
import json
import logging
import os
import tempfile
import time
import wave
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from sam_audio import (
    ContinuousBatcherConfig,
    ContinuousSAMAudioBatcher,
    SAMAudio,
    SAMAudioProcessor,
)

logger = logging.getLogger(__name__)

MODEL_ENV = "SAM_AUDIO_MODEL"
DEFAULT_MODEL = "facebook/sam-audio-small-tv"
MAX_UPLOAD_BYTES = int(os.environ.get("SAM_AUDIO_MAX_UPLOAD_MB", "200")) * 1024**2


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _configure_precision(policy: str) -> torch.dtype:
    if policy not in {"tf32", "fp32"}:
        raise ValueError("SAM_AUDIO_DTYPE_POLICY must be 'tf32' or 'fp32'")
    try:
        precision = "tf32" if policy == "tf32" else "ieee"
        torch.backends.fp32_precision = precision
        torch.backends.cuda.matmul.fp32_precision = precision
        torch.backends.cudnn.fp32_precision = precision
    except Exception:
        torch.backends.cuda.matmul.allow_tf32 = policy == "tf32"
        torch.backends.cudnn.allow_tf32 = policy == "tf32"
    torch.set_float32_matmul_precision("high" if policy == "tf32" else "highest")
    return torch.float32


class InferenceService:
    def __init__(self) -> None:
        self.model_id = os.environ.get(MODEL_ENV, DEFAULT_MODEL)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_policy = os.environ.get("SAM_AUDIO_DTYPE_POLICY", "tf32")
        self.stage1_prompt = os.environ.get(
            "SAM_AUDIO_PROMPT_STAGE1", "music soundtrack"
        )
        self.stage2_prompt = os.environ.get("SAM_AUDIO_PROMPT_STAGE2", "human voices")
        self.predict_spans = _env_bool("SAM_AUDIO_PREDICT_SPANS", True)
        self.stage1_steps = int(os.environ.get("SAM_AUDIO_STAGE1_STEPS", "16"))
        self.stage1_initial_candidates = int(
            os.environ.get("SAM_AUDIO_STAGE1_INITIAL_CANDIDATES", "4")
        )
        self.stage1_max_candidates = int(
            os.environ.get("SAM_AUDIO_STAGE1_MAX_CANDIDATES", "12")
        )
        self.stage1_margin = float(os.environ.get("SAM_AUDIO_STAGE1_MARGIN", "0.05"))
        self.stage2_steps = int(os.environ.get("SAM_AUDIO_STAGE2_STEPS", "16"))
        self.stage2_initial_candidates = int(
            os.environ.get("SAM_AUDIO_STAGE2_INITIAL_CANDIDATES", "4")
        )
        self.stage2_max_candidates = int(
            os.environ.get("SAM_AUDIO_STAGE2_MAX_CANDIDATES", "12")
        )
        self.stage2_margin = float(os.environ.get("SAM_AUDIO_STAGE2_MARGIN", "0.05"))
        self.candidate_increment = int(
            os.environ.get("SAM_AUDIO_CANDIDATE_INCREMENT", "4")
        )
        self.stage1_success_threshold = float(
            os.environ.get("SAM_AUDIO_STAGE1_SUCCESS_THRESHOLD", "4.4")
        )
        self.stage1_failure_threshold = float(
            os.environ.get("SAM_AUDIO_STAGE1_FAILURE_THRESHOLD", "4.1")
        )
        self.stage2_success_threshold = float(
            os.environ.get("SAM_AUDIO_STAGE2_SUCCESS_THRESHOLD", "4.5")
        )
        self.stage2_failure_threshold = float(
            os.environ.get("SAM_AUDIO_STAGE2_FAILURE_THRESHOLD", "4.3")
        )
        if self.stage1_failure_threshold > self.stage1_success_threshold:
            raise ValueError(
                "Stage 1 failure threshold cannot exceed success threshold"
            )
        if self.stage2_failure_threshold > self.stage2_success_threshold:
            raise ValueError(
                "Stage 2 failure threshold cannot exceed success threshold"
            )
        self.max_batch_size = int(os.environ.get("SAM_AUDIO_MAX_BATCH_SIZE", "1"))
        self.max_active_requests = int(
            os.environ.get("SAM_AUDIO_MAX_ACTIVE_REQUESTS", "16")
        )
        self.predecode_inputs = _env_bool("SAM_AUDIO_PREDECODE_INPUTS", True)
        self.async_outputs = _env_bool("SAM_AUDIO_ASYNC_OUTPUTS", True)
        self.disable_visual_ranker = _env_bool(
            "SAM_AUDIO_DISABLE_VISUAL_RANKER", False
        )
        self.request_timeout = float(os.environ.get("SAM_AUDIO_REQUEST_TIMEOUT", "900"))
        self.model: SAMAudio | None = None
        self.processor: SAMAudioProcessor | None = None
        self.batcher: ContinuousSAMAudioBatcher | None = None

    def load(self) -> None:
        dtype = _configure_precision(self.dtype_policy)
        logger.info(
            "Loading %s on %s with %s continuous cascade",
            self.model_id,
            self.device,
            self.dtype_policy,
        )
        self.processor = SAMAudioProcessor.from_pretrained(self.model_id)
        model_overrides = (
            {"visual_ranker": None} if self.disable_visual_ranker else {}
        )
        self.model = SAMAudio.from_pretrained(self.model_id, **model_overrides)
        self.model.eval().to(self.device)
        for name in ("text_ranker", "visual_ranker"):
            ranker = getattr(self.model, name, None)
            if ranker is not None:
                ranker.float()
        self.batcher = ContinuousSAMAudioBatcher(
            self.model,
            self.processor,
            ContinuousBatcherConfig(
                max_batch_size=self.max_batch_size,
                max_active_requests=self.max_active_requests,
                max_queue_size=128,
                fixed_midpoint_steps=max(self.stage1_steps, self.stage2_steps),
                predict_spans=self.predict_spans,
                initial_candidates=self.stage1_initial_candidates,
                max_candidates=self.stage1_max_candidates,
                candidate_increment=self.candidate_increment,
                margin=self.stage1_margin,
                dtype=dtype,
                pin_memory=True,
                non_blocking_transfer=True,
            ),
        )
        logger.info("Continuous cascade model ready")

    def close(self) -> None:
        if self.batcher is not None:
            self.batcher.close(wait=True)
            self.batcher = None

    def _predecode(self, audio_path: str) -> torch.Tensor:
        assert self.processor is not None
        audio, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.processor.audio_sampling_rate:
            audio = torchaudio.functional.resample(
                audio, sample_rate, self.processor.audio_sampling_rate
            )
        return audio

    def _config_for_kind(self, kind: str) -> dict[str, object]:
        by_kind = {
            "music": {
                "prompt": self.stage1_prompt,
                "steps": self.stage1_steps,
                "initial_candidates": self.stage1_initial_candidates,
                "max_candidates": self.stage1_max_candidates,
                "margin": self.stage1_margin,
                "success_threshold": self.stage1_success_threshold,
                "failure_threshold": self.stage1_failure_threshold,
            },
            "voice": {
                "prompt": self.stage2_prompt,
                "steps": self.stage2_steps,
                "initial_candidates": self.stage2_initial_candidates,
                "max_candidates": self.stage2_max_candidates,
                "margin": self.stage2_margin,
                "success_threshold": self.stage2_success_threshold,
                "failure_threshold": self.stage2_failure_threshold,
            },
        }
        if kind not in by_kind:
            raise ValueError("target must be 'music' or 'voice'")
        return by_kind[kind]

    def _cascade_configs(
        self, order: str
    ) -> tuple[dict[str, object], dict[str, object]]:
        kinds = {
            "music_first": ("music", "voice"),
            "voice_first": ("voice", "music"),
        }.get(order)
        if kinds is None:
            raise ValueError("order must be 'music_first' or 'voice_first'")
        return self._config_for_kind(kinds[0]), self._config_for_kind(kinds[1])

    def separate_single(
        self, audio_path: str, target: str
    ) -> tuple[dict[str, torch.Tensor], int, dict[str, object]]:
        if self.batcher is None or self.processor is None:
            raise RuntimeError("Model is not loaded")
        service_started = time.perf_counter()
        input_decode_started = time.perf_counter()
        audio: str | torch.Tensor = (
            self._predecode(audio_path) if self.predecode_inputs else audio_path
        )
        input_decode_ms = (time.perf_counter() - input_decode_started) * 1000
        config = self._config_for_kind(target)
        stage_started = time.perf_counter()
        result = self.batcher.separate(
            audio=audio,
            description=str(config["prompt"]),
            fixed_midpoint_steps=int(config["steps"]),
            initial_candidates=int(config["initial_candidates"]),
            max_candidates=int(config["max_candidates"]),
            margin=float(config["margin"]),
            quality_success_threshold=float(config["success_threshold"]),
            quality_failure_threshold=float(config["failure_threshold"]),
            timeout=self.request_timeout,
        )
        stage_ms = (time.perf_counter() - stage_started) * 1000
        stage_metadata = result.metadata or {}
        status = stage_metadata.get("verification", {}).get("status", "uncertain")
        target_file = f"stage1_{target}.wav"
        residual_file = "stage1_residual.wav"
        metadata: dict[str, object] = {
            "schema_version": 4,
            "verification_status": status,
            "verification": {
                "status": status,
                "stage_statuses": {"stage1": status},
                "processing_policy": (
                    f"Source-scene preflight requested only the {target} stage."
                ),
            },
            "model": self.model_id,
            "dtype_policy": self.dtype_policy,
            "predict_spans": self.predict_spans,
            "requested_order": f"{target}_only",
            "requested_targets": [target],
            "cascade_order": [target],
            "artifacts": {
                "stage1_target": target_file,
                "stage1_residual": residual_file,
                "canonical_stems": {
                    target: target_file,
                    "sfx": residual_file,
                },
            },
            "score_semantics": {
                "judge": (
                    "Continuous quality estimates where higher is better; these "
                    "are not calibrated probabilities or presence estimates."
                ),
                "candidate_margin": (
                    "Difference between the best and runner-up ensemble ranking "
                    "scores; this measures candidate preference, not correctness."
                ),
            },
            "inference_timings_ms": {
                "input_decode": input_decode_ms,
                "cascade": stage_ms,
                "service_total": (time.perf_counter() - service_started) * 1000,
                "stage1": stage_metadata.get("timings_ms", {}),
            },
            "stages": {
                "stage1": {
                    "prompt": config["prompt"],
                    "input": "original_audio",
                    "fixed_midpoint_steps": config["steps"],
                    **stage_metadata,
                }
            },
        }
        return (
            {
                target_file: result.target[0],
                residual_file: result.residual[0],
            },
            self.processor.audio_sampling_rate,
            metadata,
        )

    def separate_cascade(
        self, audio_path: str, order: str = "music_first"
    ) -> tuple[dict[str, torch.Tensor], int, dict[str, object]]:
        if self.batcher is None or self.processor is None:
            raise RuntimeError("Model is not loaded")
        service_started = time.perf_counter()
        input_decode_started = time.perf_counter()
        audio: str | torch.Tensor = (
            self._predecode(audio_path) if self.predecode_inputs else audio_path
        )
        input_decode_ms = (time.perf_counter() - input_decode_started) * 1000
        stage1_config, stage2_config = self._cascade_configs(order)
        cascade_started = time.perf_counter()
        result = self.batcher.separate_cascade(
            audio=audio,
            stage1_description=str(stage1_config["prompt"]),
            stage2_description=str(stage2_config["prompt"]),
            stage1_fixed_midpoint_steps=int(stage1_config["steps"]),
            stage1_initial_candidates=int(stage1_config["initial_candidates"]),
            stage1_max_candidates=int(stage1_config["max_candidates"]),
            stage1_margin=float(stage1_config["margin"]),
            stage1_quality_success_threshold=float(stage1_config["success_threshold"]),
            stage1_quality_failure_threshold=float(stage1_config["failure_threshold"]),
            stage2_fixed_midpoint_steps=int(stage2_config["steps"]),
            stage2_initial_candidates=int(stage2_config["initial_candidates"]),
            stage2_max_candidates=int(stage2_config["max_candidates"]),
            stage2_margin=float(stage2_config["margin"]),
            stage2_quality_success_threshold=float(stage2_config["success_threshold"]),
            stage2_quality_failure_threshold=float(stage2_config["failure_threshold"]),
            timeout=self.request_timeout,
        )
        cascade_ms = (time.perf_counter() - cascade_started) * 1000
        stage1_kind = _prompt_stem_kind(str(stage1_config["prompt"]))
        stage2_kind = _prompt_stem_kind(str(stage2_config["prompt"]))
        stage1_target_file = f"stage1_{stage1_kind}.wav"
        stage2_target_file = f"stage2_{stage2_kind}.wav"
        artifacts = {
            stage1_target_file: result.stage1.target[0],
            "stage1_residual.wav": result.stage1.residual[0],
            stage2_target_file: result.stage2.target[0],
            "stage2_residual.wav": result.stage2.residual[0],
        }
        stage1_metadata = result.stage1.metadata or {}
        stage2_metadata = result.stage2.metadata or {}
        stage_statuses = {
            "stage1": stage1_metadata.get("verification", {}).get(
                "status", "uncertain"
            ),
            "stage2": stage2_metadata.get("verification", {}).get(
                "status", "uncertain"
            ),
        }
        if "failure" in stage_statuses.values():
            final_status = "failure"
        elif "uncertain" in stage_statuses.values():
            final_status = "uncertain"
        else:
            final_status = "success"
        metadata: dict[str, object] = {
            "schema_version": 4,
            "verification_status": final_status,
            "verification": {
                "status": final_status,
                "stage_statuses": stage_statuses,
                "processing_policy": (
                    "Both stages always run; verification status never cancels "
                    "downstream separation."
                ),
            },
            "model": self.model_id,
            "dtype_policy": self.dtype_policy,
            "predict_spans": self.predict_spans,
            "requested_order": order,
            "requested_targets": [stage1_kind, stage2_kind],
            "cascade_order": [stage1_kind, stage2_kind],
            "artifacts": {
                "stage1_target": stage1_target_file,
                "stage1_residual": "stage1_residual.wav",
                "stage2_target": stage2_target_file,
                "stage2_residual": "stage2_residual.wav",
                "canonical_stems": {
                    stage1_kind: stage1_target_file,
                    stage2_kind: stage2_target_file,
                    "sfx": "stage2_residual.wav",
                },
            },
            "score_semantics": {
                "judge": (
                    "Continuous quality estimates where higher is better; these are "
                    "not calibrated probabilities."
                ),
                "candidate_margin": (
                    "Difference between the best and runner-up ensemble ranking "
                    "scores; this measures candidate preference, not correctness."
                ),
            },
            "inference_timings_ms": {
                "input_decode": input_decode_ms,
                "cascade": cascade_ms,
                "service_total": (time.perf_counter() - service_started) * 1000,
                "stage1": stage1_metadata.get("timings_ms", {}),
                "stage2": stage2_metadata.get("timings_ms", {}),
            },
            "stages": {
                "stage1": {
                    "prompt": stage1_config["prompt"],
                    "input": "original_audio",
                    "fixed_midpoint_steps": stage1_config["steps"],
                    **stage1_metadata,
                },
                "stage2": {
                    "prompt": stage2_config["prompt"],
                    "input": "stage1_residual",
                    "fixed_midpoint_steps": stage2_config["steps"],
                    **stage2_metadata,
                },
            },
        }
        return (
            artifacts,
            self.processor.audio_sampling_rate,
            metadata,
        )


service = InferenceService()


def _prompt_stem_kind(prompt: str) -> str:
    normalized = prompt.casefold()
    if any(token in normalized for token in ("music", "soundtrack")):
        return "music"
    if any(token in normalized for token in ("voice", "speech", "human")):
        return "voice"
    return "target"


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(service.load)
    yield
    await asyncio.to_thread(service.close)
    service.model = None
    service.processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="SAM Audio API", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
def health() -> dict[str, object]:
    return {
        "status": "ready" if service.batcher is not None else "starting",
        "model": service.model_id,
        "device": str(service.device),
        "cuda_available": torch.cuda.is_available(),
        "dtype_policy": service.dtype_policy,
        "predict_spans": service.predict_spans,
        "stage1": {
            "prompt": service.stage1_prompt,
            "steps": service.stage1_steps,
            "initial_candidates": service.stage1_initial_candidates,
            "max_candidates": service.stage1_max_candidates,
            "margin": service.stage1_margin,
            "candidate_increment": service.candidate_increment,
            "success_threshold": service.stage1_success_threshold,
            "failure_threshold": service.stage1_failure_threshold,
        },
        "stage2": {
            "prompt": service.stage2_prompt,
            "steps": service.stage2_steps,
            "initial_candidates": service.stage2_initial_candidates,
            "max_candidates": service.stage2_max_candidates,
            "margin": service.stage2_margin,
            "candidate_increment": service.candidate_increment,
            "success_threshold": service.stage2_success_threshold,
            "failure_threshold": service.stage2_failure_threshold,
        },
        "batcher": {
            "max_batch_size": service.max_batch_size,
            "max_active_requests": service.max_active_requests,
            "pin_memory": True,
            "non_blocking_transfer": True,
            "predecode_inputs": service.predecode_inputs,
            "async_outputs": service.async_outputs,
        },
        "supported_orders": ["music_first", "voice_first"],
        "supported_targets": ["music", "voice"],
    }


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    total = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Audio upload is too large")
            output.write(chunk)


def _wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    pcm = (
        audio.clamp(-1, 1)
        .mul(32767)
        .round()
        .to(torch.int16)
        .transpose(0, 1)
        .contiguous()
        .numpy()
        .tobytes()
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(audio.size(0))
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    return buffer.getvalue()


def _build_archive(
    artifacts: dict[str, torch.Tensor],
    sample_rate: int,
    metadata: dict[str, object],
) -> io.BytesIO:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for filename, artifact in artifacts.items():
            output.writestr(filename, _wav_bytes(artifact, sample_rate))
        output.writestr(
            "metadata.json",
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        )
    archive.seek(0)
    return archive


def _detailed_timing_headers(
    metadata: dict[str, object],
) -> tuple[list[str], dict[str, str]]:
    timings = metadata.get("inference_timings_ms", {})
    if not isinstance(timings, dict):
        return [], {}
    server_timings: list[str] = []
    headers: dict[str, str] = {}

    def add(server_name: str, header_name: str, value: object) -> None:
        if not isinstance(value, (int, float)):
            return
        duration = float(value)
        server_timings.append(f"{server_name};dur={duration:.1f}")
        headers[header_name] = f"{duration:.1f}"

    add("input-decode", "X-SAM-Audio-Input-Decode-Ms", timings.get("input_decode"))
    add("cascade", "X-SAM-Audio-Cascade-Ms", timings.get("cascade"))
    for stage_number in (1, 2):
        stage = timings.get(f"stage{stage_number}", {})
        if not isinstance(stage, dict):
            continue
        prefix = f"s{stage_number}"
        header_prefix = f"X-SAM-Audio-Stage{stage_number}"
        for key, label in (
            ("raw_queue_wait", "Raw-Queue-Wait"),
            ("preprocess", "Preprocess"),
            ("gpu_queue_wait", "GPU-Queue-Wait"),
            ("prepare", "Prepare"),
            ("generation", "Generation"),
            ("decode", "Decode"),
            ("clap", "CLAP"),
            ("judge", "Judge"),
            ("ensemble_combine", "Ensemble-Combine"),
            ("scoring", "Scoring"),
            ("selection", "Selection"),
            ("postprocess", "Postprocess"),
            ("stage_total", "Total"),
        ):
            add(
                f"{prefix}-{key.replace('_', '-')}",
                f"{header_prefix}-{label}-Ms",
                stage.get(key),
            )
    return server_timings, headers


@app.post("/v1/separate")
async def separate(
    audio: Annotated[UploadFile, File()],
    description: Annotated[str, Form(max_length=500)] = "",
    order: Annotated[str, Form()] = "music_first",
    targets: Annotated[str, Form()] = "music,voice",
) -> StreamingResponse:
    request_started = time.perf_counter()
    if order not in {"music_first", "voice_first"}:
        raise HTTPException(
            status_code=422,
            detail="order must be 'music_first' or 'voice_first'",
        )
    selected_targets = tuple(
        dict.fromkeys(value.strip().casefold() for value in targets.split(",") if value.strip())
    )
    if not selected_targets or any(
        target not in {"music", "voice"} for target in selected_targets
    ):
        raise HTTPException(
            status_code=422,
            detail="targets must contain music, voice, or both",
        )
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    try:
        with tempfile.TemporaryDirectory(prefix="sam-audio-") as directory:
            input_path = Path(directory) / f"input{suffix}"
            await _save_upload(audio, input_path)
            upload_finished = time.perf_counter()
            if len(selected_targets) == 1:
                artifacts, sample_rate, metadata = await asyncio.to_thread(
                    service.separate_single,
                    str(input_path),
                    selected_targets[0],
                )
            else:
                artifacts, sample_rate, metadata = await asyncio.to_thread(
                    service.separate_cascade,
                    str(input_path),
                    order,
                )
            inference_finished = time.perf_counter()
    finally:
        await audio.close()

    if description.strip():
        logger.info("Ignoring request description in configured cascade mode")
    upload_ms = (upload_finished - request_started) * 1000
    inference_ms = (inference_finished - upload_finished) * 1000
    metadata["request"] = {
        "submitted_description": description,
        "submitted_description_used": False,
        "requested_order": order,
        "requested_targets": list(selected_targets),
    }
    metadata["timings_ms"] = {
        "upload_handler": upload_ms,
        "inference": inference_ms,
        "note": "Packaging and full server timings are returned in HTTP headers.",
    }
    if service.async_outputs:
        archive = await asyncio.to_thread(
            _build_archive, artifacts, sample_rate, metadata
        )
    else:
        archive = _build_archive(artifacts, sample_rate, metadata)
    package_finished = time.perf_counter()

    package_ms = (package_finished - inference_finished) * 1000
    server_ms = (package_finished - request_started) * 1000
    logger.info(
        "request timing upload=%.1fms inference=%.1fms package=%.1fms total=%.1fms",
        upload_ms,
        inference_ms,
        package_ms,
        server_ms,
    )
    detailed_server_timings, detailed_headers = _detailed_timing_headers(metadata)
    verification = metadata.get("verification", {})
    stage_statuses = (
        verification.get("stage_statuses", {}) if isinstance(verification, dict) else {}
    )
    final_status = metadata.get("verification_status", "uncertain")
    server_timing = [
        f"upload;dur={upload_ms:.1f}",
        f"inference;dur={inference_ms:.1f}",
        *detailed_server_timings,
        f"package;dur={package_ms:.1f}",
    ]
    return StreamingResponse(
        archive,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="separation.zip"',
            "Server-Timing": ", ".join(server_timing),
            "X-SAM-Audio-Upload-Ms": f"{upload_ms:.1f}",
            "X-SAM-Audio-Inference-Ms": f"{inference_ms:.1f}",
            "X-SAM-Audio-Package-Ms": f"{package_ms:.1f}",
            "X-SAM-Audio-Server-Ms": f"{server_ms:.1f}",
            "X-SAM-Audio-Verification-Status": str(final_status),
            "X-SAM-Audio-Cascade-Order": order,
            "X-SAM-Audio-Stage1-Verification-Status": str(
                stage_statuses.get("stage1", "uncertain")
            ),
            "X-SAM-Audio-Stage2-Verification-Status": str(
                stage_statuses.get("stage2", "uncertain")
            ),
            **detailed_headers,
        },
    )


def main() -> None:
    uvicorn.run(
        "sam_audio.api:app",
        host=os.environ.get("SAM_AUDIO_HOST", "0.0.0.0"),
        port=int(os.environ.get("SAM_AUDIO_PORT", "8000")),
        workers=1,
    )


if __name__ == "__main__":
    main()
