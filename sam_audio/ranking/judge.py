# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import os

import torch

from ..model.config import JudgeRankerConfig
from ..model.judge import SAMAudioJudgeModel
from ..processor import SAMAudioJudgeProcessor
from .ranker import Ranker, RankingResult


class JudgeRanker(Ranker):
    def __init__(self, config: JudgeRankerConfig):
        super().__init__()
        self.config = config
        self.default_weights = self._normalized_weights(
            "SAM_AUDIO_JUDGE_OVERALL_WEIGHT",
            "SAM_AUDIO_JUDGE_PRECISION_WEIGHT",
            0.4,
            0.6,
        )
        self.voice_weights = self._normalized_weights(
            "SAM_AUDIO_VOICE_JUDGE_OVERALL_WEIGHT",
            "SAM_AUDIO_VOICE_JUDGE_PRECISION_WEIGHT",
            0.5,
            0.5,
        )
        self.music_weights = self._normalized_weights(
            "SAM_AUDIO_MUSIC_JUDGE_OVERALL_WEIGHT",
            "SAM_AUDIO_MUSIC_JUDGE_PRECISION_WEIGHT",
            0.25,
            0.75,
        )
        self.model = SAMAudioJudgeModel.from_pretrained(config.checkpoint_or_model_id)
        self.processor = SAMAudioJudgeProcessor.from_pretrained(
            config.checkpoint_or_model_id
        )

    @staticmethod
    def _normalized_weights(
        overall_env: str,
        precision_env: str,
        overall_default: float,
        precision_default: float,
    ) -> tuple[float, float]:
        overall_weight = float(os.environ.get(overall_env, str(overall_default)))
        precision_weight = float(os.environ.get(precision_env, str(precision_default)))
        if overall_weight < 0 or precision_weight < 0:
            raise ValueError("Judge selection weights must be non-negative")
        total_weight = overall_weight + precision_weight
        if total_weight <= 0:
            raise ValueError("At least one Judge selection weight must be positive")
        return overall_weight / total_weight, precision_weight / total_weight

    def _weights_for_description(self, description: str) -> tuple[float, float, str]:
        normalized = description.lower()
        if "voice" in normalized or "speech" in normalized or "human" in normalized:
            return *self.voice_weights, "voice"
        if "music" in normalized or "soundtrack" in normalized:
            return *self.music_weights, "music"
        return *self.default_weights, "default"

    @torch.inference_mode()
    def score_with_details(
        self,
        input_audio: list[torch.Tensor],
        extracted_audio: list[torch.Tensor],
        descriptions: list[str],
        sample_rate: int = 48_000,
        **kwargs,
    ):
        bsz, ncandidates = len(input_audio), len(input_audio[0])
        input_seqs = [x[None] for candidates in input_audio for x in candidates]
        extracted_seqs = [x[None] for candidates in extracted_audio for x in candidates]
        repeated_descriptions = [x for x in descriptions for _ in range(ncandidates)]
        processed = self.processor(
            text=repeated_descriptions,
            input_audio=input_seqs,
            separated_audio=extracted_seqs,
            return_tensors="pt",
            padding=True,
            sampling_rate=sample_rate,
        )
        res = self.model(**processed.to(input_audio[0].device))
        overall_weight, precision_weight, weight_profile = (
            self._weights_for_description(descriptions[0])
        )
        details = {
            "overall": res.overall.view(bsz, ncandidates),
            "recall": res.recall.view(bsz, ncandidates),
            "precision": res.precision.view(bsz, ncandidates),
            "faithfulness": res.faithfulness.view(bsz, ncandidates),
        }
        selection_score = (
            overall_weight * details["overall"]
            + precision_weight * details["precision"]
        )
        details.update(
            {
                "selection_score": selection_score,
                "selection_overall_weight": overall_weight,
                "selection_precision_weight": precision_weight,
                "selection_weight_profile": weight_profile,
            }
        )
        return RankingResult(scores=selection_score, details={"judge": details})

    def forward(self, **kwargs):
        return self.score_with_details(**kwargs).scores
