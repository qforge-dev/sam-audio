# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import time
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch


@dataclass
class RankingResult:
    scores: torch.Tensor
    details: dict[str, dict[str, Any]]
    timings_ms: dict[str, float] = field(default_factory=dict)


def _synchronize(scores: Optional[torch.Tensor] = None) -> None:
    if not torch.cuda.is_available():
        return
    device = scores.device if torch.is_tensor(scores) else None
    if device is not None and device.type != "cuda":
        return
    torch.cuda.synchronize(device=device)


class Ranker(torch.nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, audio: list[torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Args:
            audio: (list[torch.Tensor]) where each element in the list corresponds to
                the candidates for the i'th generation (num_candidates, num_frames)
        Returns:
            (torch.Tensor) of shape (batch_size, num_candidates) correspoding to the ranking scores
        """
        pass

    def score_with_details(self, **kwargs) -> RankingResult:
        _synchronize()
        started = time.perf_counter()
        scores = self.forward(**kwargs)
        _synchronize(scores)
        elapsed_ms = (time.perf_counter() - started) * 1000
        name = self.__class__.__name__.removesuffix("Ranker").lower()
        return RankingResult(
            scores=scores,
            details={name: {"score": scores}},
            timings_ms={name: elapsed_ms},
        )


class EnsembleRanker(Ranker):
    def __init__(
        self,
        rankers: List[Ranker],
        weights: List[float],
        names: Optional[List[str]] = None,
    ):
        super().__init__()
        assert len(rankers) == len(weights)
        self.rankers = torch.nn.ModuleList(rankers)
        self.weights = weights
        self.names = names or [
            ranker.__class__.__name__.removesuffix("Ranker").lower()
            for ranker in rankers
        ]

    def forward(self, **kwargs) -> torch.Tensor:
        return self.score_with_details(**kwargs).scores

    def score_with_details(self, **kwargs) -> RankingResult:
        _synchronize()
        total_started = time.perf_counter()
        result = None
        details: dict[str, dict[str, Any]] = {}
        timings_ms: dict[str, float] = {}
        for name, weight, ranker in zip(
            self.names, self.weights, self.rankers, strict=False
        ):
            _synchronize(result)
            ranker_started = time.perf_counter()
            ranked = ranker.score_with_details(**kwargs)
            _synchronize(ranked.scores)
            timings_ms[name] = (time.perf_counter() - ranker_started) * 1000
            if result is None:
                result = weight * ranked.scores
            else:
                result += weight * ranked.scores
            ranker_details = next(iter(ranked.details.values()))
            details[name] = {**ranker_details, "ensemble_weight": weight}
        assert result is not None
        details["ensemble"] = {"score": result}
        _synchronize(result)
        total_ms = (time.perf_counter() - total_started) * 1000
        timings_ms["ensemble_combine"] = max(0.0, total_ms - sum(timings_ms.values()))
        timings_ms["ranking_total"] = total_ms
        return RankingResult(
            scores=result,
            details=details,
            timings_ms=timings_ms,
        )
