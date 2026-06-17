# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from core.audio_visual_encoder import PEAudioFrame, PEAudioFrameTransform
from torchdiffeq import odeint

from sam_audio.model.align import AlignModalities
from sam_audio.model.base import BaseModel
from sam_audio.model.codec import DACVAE
from sam_audio.model.config import SAMAudioConfig
from sam_audio.model.text_encoder import T5TextEncoder
from sam_audio.model.transformer import DiT
from sam_audio.model.vision_encoder import PerceptionEncoder
from sam_audio.processor import Batch
from sam_audio.ranking import create_ranker

DFLT_ODE_OPT = {"method": "midpoint", "options": {"step_size": 2 / 32}}


def _fixed_midpoint_integrate(vector_field, y0: torch.Tensor, options: Dict[str, Any]):
    """Fixed-grid midpoint integration specialized for the default inference path."""
    t0 = float(options.get("t0", 0.0))
    t1 = float(options.get("t1", 1.0))
    if "steps" in options:
        steps = int(options["steps"])
        if steps <= 0:
            raise ValueError(f"`steps` must be positive, got {steps}")
        step_size = (t1 - t0) / steps
    else:
        step_size = float(options.get("step_size", 2 / 32))
        if step_size <= 0:
            raise ValueError(f"`step_size` must be positive, got {step_size}")
        steps = math.ceil((t1 - t0) / step_size)

    y = y0
    current_t = t0
    for _ in range(steps):
        dt = min(step_size, t1 - current_t)
        if dt <= 0:
            break
        start_t = y.new_tensor(current_t)
        midpoint_t = y.new_tensor(current_t + 0.5 * dt)
        k1 = vector_field(start_t, y)
        midpoint_y = y + 0.5 * dt * k1
        y = y + dt * vector_field(midpoint_t, midpoint_y)
        current_t += dt
    return y


class SinusoidalEmbedding(torch.nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        inv_freq = torch.exp(
            -math.log(theta) * torch.arange(half_dim).float() / half_dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, pos=None):
        if pos is None:
            seq_len, device = x.shape[1], x.device
            pos = torch.arange(seq_len, device=device)

        emb = torch.einsum("i, j -> i j", pos, self.inv_freq)
        emb = torch.cat((emb.cos(), emb.sin()), dim=-1)
        return emb


class EmbedAnchors(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, out_dim: int):
        super().__init__()
        self.embed = torch.nn.Embedding(
            num_embeddings + 1, embedding_dim, padding_idx=num_embeddings
        )
        self.gate = torch.nn.Parameter(torch.tensor([0.0]))
        self.proj = torch.nn.Linear(embedding_dim, out_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
    ):
        if anchor_ids is None:
            return x

        embs = self.embed(anchor_ids.gather(1, anchor_alignment))
        proj = self.proj(embs)
        return x + self.gate.tanh() * proj


@dataclass
class SeparationResult:
    target: torch.Tensor
    residual: torch.Tensor
    noise: torch.Tensor


@dataclass
class PreparedSeparation:
    batch: Batch
    candidates: int
    forward_args: Dict[str, torch.Tensor]
    projected_static: torch.Tensor
    memory_base: Optional[torch.Tensor]
    predict_spans: bool = False


class SAMAudio(BaseModel):
    config_cls = SAMAudioConfig
    revision = None
    _h100_fixed_midpoint_supported = True

    def __init__(self, cfg: SAMAudioConfig):
        super().__init__()
        self.audio_codec = DACVAE(cfg.audio_codec)
        self.text_encoder = T5TextEncoder(cfg.text_encoder)
        self.vision_encoder = PerceptionEncoder(cfg.vision_encoder)
        self.transformer = DiT(cfg.transformer)
        self.proj = torch.nn.Linear(cfg.in_channels, cfg.transformer.dim)
        self.align_masked_video = AlignModalities(
            cfg.vision_encoder.dim, cfg.transformer.dim
        )
        self.embed_anchors = EmbedAnchors(
            cfg.num_anchors, cfg.anchor_embedding_dim, cfg.transformer.dim
        )
        self.memory_proj = torch.nn.Linear(cfg.text_encoder.dim, cfg.transformer.dim)
        self.timestep_emb = SinusoidalEmbedding(cfg.transformer.dim)
        self.visual_ranker = create_ranker(cfg.visual_ranker)
        self.text_ranker = create_ranker(cfg.text_ranker)
        if cfg.span_predictor is not None:
            self.span_predictor = PEAudioFrame.from_config(
                cfg.span_predictor, pretrained=True
            )
            self.span_predictor_transform = PEAudioFrameTransform.from_config(
                cfg.span_predictor
            )

    @property
    def sample_rate(self):
        return self.audio_codec.sample_rate

    def align_inputs(
        self,
        noisy_audio,
        audio_features: torch.Tensor,
        masked_video_features: Optional[torch.Tensor] = None,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
    ):
        x = torch.cat(
            [
                noisy_audio,
                torch.zeros_like(audio_features),
                audio_features,
            ],
            dim=2,
        )

        projected = self.proj(x)
        aligned = self.align_masked_video(projected, masked_video_features)
        aligned = self.embed_anchors(aligned, anchor_ids, anchor_alignment)
        return aligned

    def _get_projected_static_inputs(
        self,
        audio_features: torch.Tensor,
        masked_video_features: Optional[torch.Tensor] = None,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
    ):
        feature_channels = audio_features.size(-1)
        expected_channels = feature_channels * 3
        if self.proj.in_features == expected_channels:
            audio_weight = self.proj.weight[
                :, 2 * feature_channels : 3 * feature_channels
            ]
            projected = F.linear(audio_features, audio_weight, self.proj.bias)
        else:
            static_input = torch.cat(
                [
                    torch.zeros_like(audio_features),
                    torch.zeros_like(audio_features),
                    audio_features,
                ],
                dim=2,
            )
            projected = self.proj(static_input)
        aligned = self.align_masked_video(projected, masked_video_features)
        return self.embed_anchors(aligned, anchor_ids, anchor_alignment)

    def _align_prepared_inputs(
        self,
        noisy_audio: torch.Tensor,
        projected_static: torch.Tensor,
    ):
        feature_channels = noisy_audio.size(-1)
        if self.proj.in_features >= feature_channels:
            noise_weight = self.proj.weight[:, :feature_channels]
            return projected_static + F.linear(noisy_audio, noise_weight, bias=None)
        return self.align_inputs(noisy_audio, noisy_audio.new_zeros(noisy_audio.shape))

    def forward(
        self,
        noisy_audio: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        time: torch.Tensor,
        masked_video_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
        audio_pad_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for the model.  Represents one function evaluation of the ODE.
        In the below descriptions, B is batch size, T is sequence length, C is channel size.
        Note that the size of C and T may vary across arguments (ex. text_features vs. audio_features),
        it is used only to designate a Channel or time/sequence-length dimension respectively.

        Args:
            noisy_audio (torch.Tensor): Noisy audio input tensor (being denoised).
            audio_features (torch.Tensor): Clean audio features [B x T x C].
            text_features (torch.Tensor): Encoded text features tensor [B x T x C].
            time (torch.Tensor): Timestep tensor for positional encoding [B].
            masked_video_features (Optional[torch.Tensor], optional): Masked video features tensor. [B x C x T].
            text_mask (Optional[torch.Tensor], optional): Padding mask for text features. [B x T].
            anchor_ids (Optional[torch.Tensor], optional): Anchor IDs tensor. Defaults to None [B x T].
            anchor_alignment (Optional[torch.Tensor], optional): Anchor alignment tensor. B x T.
            audio_pad_mask (Optional[torch.Tensor], optional): Padding mask for audio input. [B x T].

        Returns:
            torch.Tensor
        """
        aligned_inputs = self.align_inputs(
            noisy_audio,
            audio_features,
            masked_video_features=masked_video_features,
            anchor_ids=anchor_ids,
            anchor_alignment=anchor_alignment,
        )

        memory = timestep_emb = self.timestep_emb(time, pos=time).unsqueeze(1)
        if text_features is not None:
            memory = self.memory_proj(text_features) + timestep_emb

        return self.transformer(
            aligned_inputs,
            time,
            padding_mask=audio_pad_mask,
            memory=memory,
            memory_padding_mask=text_mask,
        )

    def forward_prepared(
        self,
        noisy_audio: torch.Tensor,
        prepared: PreparedSeparation,
        time: torch.Tensor,
    ):
        aligned_inputs = self._align_prepared_inputs(
            noisy_audio, prepared.projected_static
        )
        timestep_emb = self.timestep_emb(time, pos=time).unsqueeze(1)
        memory = timestep_emb
        if prepared.memory_base is not None:
            memory = prepared.memory_base + timestep_emb
        return self.transformer(
            aligned_inputs,
            time,
            padding_mask=prepared.forward_args["audio_pad_mask"],
            memory=memory,
            memory_padding_mask=prepared.forward_args["text_mask"],
        )

    def _get_audio_features(self, audios: torch.Tensor):
        audio_features = self.audio_codec(audios).transpose(1, 2)
        return torch.cat([audio_features, audio_features], dim=2)

    def _get_audio_features_dedup(self, audios: torch.Tensor):
        if audios.size(0) <= 1:
            return self._get_audio_features(audios)
        first = audios[:1]
        if torch.equal(audios, first.expand_as(audios)):
            features = self._get_audio_features(first)
            return features.expand(audios.size(0), *features.shape[1:])
        return self._get_audio_features(audios)

    def _get_video_features(self, video, audio_features):
        B, T, _ = audio_features.shape
        if video is None:
            return audio_features.new_zeros(B, self.vision_encoder.dim, T)
        else:
            return self.vision_encoder(video).transpose(1, 2)

    def _repeat_for_reranking(self, tensor, candidates):
        if candidates > 1:
            B = tensor.size(0)
            rest = tensor.shape[1:]
            return (
                tensor.unsqueeze(1)
                .expand(B, candidates, *rest)
                .reshape(B * candidates, *rest)
            )
        else:
            return tensor

    def _unrepeat_from_reranking(self, tensor, candidates):
        return tensor[::candidates]

    def _get_forward_args(self, batch: Batch, candidates: int = 1):
        audio_features = self._get_audio_features(batch.audios)
        text_features, text_mask = self.text_encoder(batch.descriptions)
        masked_video_features = self._get_video_features(
            batch.masked_video, audio_features
        )

        return {
            "audio_features": self._repeat_for_reranking(audio_features, candidates),
            "text_features": self._repeat_for_reranking(text_features, candidates),
            "text_mask": self._repeat_for_reranking(text_mask, candidates),
            "masked_video_features": self._repeat_for_reranking(
                masked_video_features, candidates
            ),
            "anchor_ids": self._repeat_for_reranking(batch.anchor_ids, candidates),
            "anchor_alignment": self._repeat_for_reranking(
                batch.anchor_alignment, candidates
            ),
            "audio_pad_mask": self._repeat_for_reranking(
                batch.audio_pad_mask, candidates
            ),
        }

    def prepare_audio(
        self,
        batch: Batch,
        candidates: int = 1,
        predict_spans: bool = False,
    ) -> PreparedSeparation:
        audio_features = self._get_audio_features_dedup(batch.audios)
        text_features, text_mask = self.text_encoder(batch.descriptions)
        masked_video_features = self._get_video_features(
            batch.masked_video, audio_features
        )

        forward_args = {
            "audio_features": self._repeat_for_reranking(audio_features, candidates),
            "text_features": self._repeat_for_reranking(text_features, candidates),
            "text_mask": self._repeat_for_reranking(text_mask, candidates),
            "masked_video_features": self._repeat_for_reranking(
                masked_video_features, candidates
            ),
            "anchor_ids": self._repeat_for_reranking(batch.anchor_ids, candidates),
            "anchor_alignment": self._repeat_for_reranking(
                batch.anchor_alignment, candidates
            ),
            "audio_pad_mask": self._repeat_for_reranking(
                batch.audio_pad_mask, candidates
            ),
        }

        if predict_spans and hasattr(self, "span_predictor") and batch.anchors is None:
            batch = self.predict_spans(
                batch=batch,
                audio_features=audio_features,
                audio_pad_mask=batch.audio_pad_mask,
            )
            forward_args["anchor_ids"] = self._repeat_for_reranking(
                batch.anchor_ids, candidates
            )
            forward_args["anchor_alignment"] = self._repeat_for_reranking(
                batch.anchor_alignment, candidates
            )

        memory_base = None
        if forward_args["text_features"] is not None:
            memory_base = self.memory_proj(forward_args["text_features"])
        projected_static = self._get_projected_static_inputs(
            forward_args["audio_features"],
            masked_video_features=forward_args["masked_video_features"],
            anchor_ids=forward_args["anchor_ids"],
            anchor_alignment=forward_args["anchor_alignment"],
        )
        return PreparedSeparation(
            batch=batch,
            candidates=candidates,
            forward_args=forward_args,
            projected_static=projected_static,
            memory_base=memory_base,
            predict_spans=predict_spans,
        )

    def predict_spans(
        self, batch: Batch, audio_features: torch.Tensor, audio_pad_mask: torch.Tensor
    ) -> Batch:
        input = self.span_predictor_transform(text=batch.descriptions).to(
            audio_features.device
        )
        output = self.span_predictor(
            input_features=audio_features[:, :, :128],
            padding_mask=audio_pad_mask,
            return_spans=True,
            **input,
        )
        anchors = [[["+"] + anchor for anchor in anchors] for anchors in output.spans]
        batch.process_anchors(anchors)
        return batch

    @torch.inference_mode()
    def separate(
        self,
        batch: Batch,
        noise: Optional[torch.Tensor] = None,
        ode_opt: Dict[str, Any] = DFLT_ODE_OPT,
        reranking_candidates: int = 1,
        predict_spans: bool = False,
    ) -> SeparationResult:
        # Encode audio
        forward_args = self._get_forward_args(batch, candidates=reranking_candidates)

        if predict_spans and hasattr(self, "span_predictor") and batch.anchors is None:
            batch = self.predict_spans(
                batch=batch,
                audio_features=self._unrepeat_from_reranking(
                    forward_args["audio_features"], reranking_candidates
                ),
                audio_pad_mask=self._unrepeat_from_reranking(
                    forward_args["audio_pad_mask"], reranking_candidates
                ),
            )

            # Refresh anchor conditioning created by predict_spans()
            forward_args.update(
                {
                    "anchor_ids": self._repeat_for_reranking(
                        batch.anchor_ids, reranking_candidates
                    ),
                    "anchor_alignment": self._repeat_for_reranking(
                        batch.anchor_alignment, reranking_candidates
                    ),
                }
            )

        audio_features = forward_args["audio_features"]
        B, T, C = audio_features.shape
        C = C // 2  # we stack audio_features, so the actual channels is half

        if noise is None:
            noise = torch.randn_like(audio_features)

        def vector_field(t, noisy_audio):
            res = self.forward(
                noisy_audio=noisy_audio,
                time=t.expand(noisy_audio.size(0)),
                **forward_args,
            )
            return res

        if ode_opt.get("method") == "fixed_midpoint":
            generated = _fixed_midpoint_integrate(
                vector_field, noise, ode_opt.get("options", {})
            )
        else:
            states = odeint(
                vector_field,
                noise,
                torch.tensor([0.0, 1.0], device=noise.device),
                **ode_opt,
            )
            generated = states[-1]
        generated_features = generated.transpose(1, 2)
        # generated_features has shape [B, 2C, T].  Reshape to stack along the batch dimension
        wavs = self.audio_codec.decode(generated_features.reshape(2 * B, C, T)).view(
            B, 2, -1
        )

        bsz = wavs.size(0) // reranking_candidates
        sizes = self.audio_codec.feature_idx_to_wav_idx(batch.sizes)
        target_wavs = self.unbatch(
            wavs[:, 0].view(bsz, reranking_candidates, -1), sizes
        )
        residual_wavs = self.unbatch(
            wavs[:, 1].view(bsz, reranking_candidates, -1), sizes
        )

        if (
            reranking_candidates > 1
            and batch.masked_video is not None
            and self.visual_ranker is not None
        ):
            scores = self.visual_ranker(
                extracted_audio=target_wavs,
                videos=batch.masked_video,
                sample_rate=self.audio_codec.sample_rate,
            )
            idxs = scores.argmax(dim=1)
        elif reranking_candidates > 1 and self.text_ranker is not None:
            input_audio = [
                audio[:, :size].expand(reranking_candidates, -1)
                for audio, size in zip(batch.audios, sizes, strict=False)
            ]
            scores = self.text_ranker(
                extracted_audio=target_wavs,
                input_audio=input_audio,
                descriptions=batch.descriptions,
                sample_rate=self.audio_codec.sample_rate,
            )
            idxs = scores.argmax(dim=1)
        else:
            idxs = torch.zeros(bsz, dtype=torch.long, device=noise.device)

        return SeparationResult(
            target=[wav[idx] for wav, idx in zip(target_wavs, idxs, strict=False)],
            residual=[
                wavs[idx] for wavs, idx in zip(residual_wavs, idxs, strict=False)
            ],
            noise=noise,
        )

    @torch.inference_mode()
    def separate_prepared(
        self,
        prepared: PreparedSeparation,
        prompts: Optional[list[str]] = None,
        noise: Optional[torch.Tensor] = None,
        ode_opt: Dict[str, Any] = DFLT_ODE_OPT,
        reranking_candidates: Optional[int] = None,
        predict_spans: bool = False,
    ) -> SeparationResult:
        del prompts
        if reranking_candidates is None:
            reranking_candidates = prepared.candidates
        if reranking_candidates != prepared.candidates:
            raise ValueError(
                "`reranking_candidates` must match the candidate count used by `prepare_audio`"
            )
        if predict_spans and not prepared.predict_spans and prepared.batch.anchors is None:
            prepared = self.prepare_audio(
                prepared.batch,
                candidates=prepared.candidates,
                predict_spans=True,
            )

        audio_features = prepared.forward_args["audio_features"]
        B, T, C = audio_features.shape
        C = C // 2

        if noise is None:
            noise = torch.randn_like(audio_features)

        def vector_field(t, noisy_audio):
            return self.forward_prepared(
                noisy_audio=noisy_audio,
                prepared=prepared,
                time=t.expand(noisy_audio.size(0)),
            )

        if ode_opt.get("method") == "fixed_midpoint":
            generated = _fixed_midpoint_integrate(
                vector_field, noise, ode_opt.get("options", {})
            )
        else:
            states = odeint(
                vector_field,
                noise,
                torch.tensor([0.0, 1.0], device=noise.device),
                **ode_opt,
            )
            generated = states[-1]
        generated_features = generated.transpose(1, 2)
        wavs = self.audio_codec.decode(generated_features.reshape(2 * B, C, T)).view(
            B, 2, -1
        )

        batch = prepared.batch
        bsz = wavs.size(0) // reranking_candidates
        sizes = self.audio_codec.feature_idx_to_wav_idx(batch.sizes)
        target_wavs = self.unbatch(
            wavs[:, 0].view(bsz, reranking_candidates, -1), sizes
        )
        residual_wavs = self.unbatch(
            wavs[:, 1].view(bsz, reranking_candidates, -1), sizes
        )

        if (
            reranking_candidates > 1
            and batch.masked_video is not None
            and self.visual_ranker is not None
        ):
            scores = self.visual_ranker(
                extracted_audio=target_wavs,
                videos=batch.masked_video,
                sample_rate=self.audio_codec.sample_rate,
            )
            idxs = scores.argmax(dim=1)
        elif reranking_candidates > 1 and self.text_ranker is not None:
            input_audio = [
                audio[:, :size].expand(reranking_candidates, -1)
                for audio, size in zip(batch.audios, sizes, strict=False)
            ]
            scores = self.text_ranker(
                extracted_audio=target_wavs,
                input_audio=input_audio,
                descriptions=batch.descriptions,
                sample_rate=self.audio_codec.sample_rate,
            )
            idxs = scores.argmax(dim=1)
        else:
            idxs = torch.zeros(bsz, dtype=torch.long, device=noise.device)

        return SeparationResult(
            target=[wav[idx] for wav, idx in zip(target_wavs, idxs, strict=False)],
            residual=[
                wavs[idx] for wavs, idx in zip(residual_wavs, idxs, strict=False)
            ],
            noise=noise,
        )

    def unbatch(self, wavs: torch.Tensor, sizes: torch.Tensor, time_dim: int = -1):
        result = []
        for row, size in zip(wavs, sizes, strict=False):
            result.append(row.narrow(dim=time_dim, start=0, length=size))
        return result

    def load_state_dict(self, state_dict, strict=True):
        if strict:
            missing_keys, unexpected_keys = super().load_state_dict(
                state_dict, strict=False
            )
            # We load this directly from HF, not in checkpoint
            skip_regex = re.compile(
                "(^text_encoder|^visual_ranker|^text_ranker|^span_predictor)"
            )
            missing_keys = [x for x in missing_keys if not re.search(skip_regex, x)]
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                raise RuntimeError(
                    f"Missing keys: {missing_keys}, unexpected_keys: {unexpected_keys}"
                )


__all__ = ["SAMAudio"]
