"""Duration-aware source diversity limits shared by dataset stages."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

DEFAULT_BASE_CLIPS_PER_SOURCE = 16
DEFAULT_SOURCE_CONTENT_MINUTES_PER_HOUR = 10.0
DEFAULT_MAX_CLIPS_PER_SOURCE = 60


def source_clip_budget(
    source_duration_seconds: float,
    *,
    clip_seconds: float,
    base_clips: int = DEFAULT_BASE_CLIPS_PER_SOURCE,
    content_minutes_per_hour: float = DEFAULT_SOURCE_CONTENT_MINUTES_PER_HOUR,
    max_clips: int = DEFAULT_MAX_CLIPS_PER_SOURCE,
) -> int:
    """Return a bounded clip budget that grows with source duration.

    The base gives short sources a small absolute allowance. Beyond that, the
    duration allowance grows at the configured content rate.
    """
    if clip_seconds <= 0:
        raise ValueError("clip_seconds must be positive")
    if base_clips < 1:
        raise ValueError("base_clips must be positive")
    if content_minutes_per_hour <= 0:
        raise ValueError("content_minutes_per_hour must be positive")
    if max_clips < base_clips:
        raise ValueError("max_clips must be at least base_clips")
    duration = max(0.0, float(source_duration_seconds))
    duration_content_seconds = duration * float(content_minutes_per_hour) / 60.0
    duration_clips = math.floor(duration_content_seconds / clip_seconds)
    return min(max_clips, max(base_clips, duration_clips))


def record_source_clip_budget(
    record: Mapping[str, Any],
    *,
    clip_seconds: float,
    base_clips: int = DEFAULT_BASE_CLIPS_PER_SOURCE,
    content_minutes_per_hour: float = DEFAULT_SOURCE_CONTENT_MINUTES_PER_HOUR,
    max_clips: int = DEFAULT_MAX_CLIPS_PER_SOURCE,
) -> int:
    return source_clip_budget(
        float(record.get("duration_seconds") or 0.0),
        clip_seconds=clip_seconds,
        base_clips=base_clips,
        content_minutes_per_hour=content_minutes_per_hour,
        max_clips=max_clips,
    )


def source_diversity_policy(
    *,
    clip_seconds: float,
    base_clips: int = DEFAULT_BASE_CLIPS_PER_SOURCE,
    content_minutes_per_hour: float = DEFAULT_SOURCE_CONTENT_MINUTES_PER_HOUR,
    max_clips: int = DEFAULT_MAX_CLIPS_PER_SOURCE,
) -> dict[str, Any]:
    return {
        "policy": "duration_scaled_source_budget_v1",
        "clip_seconds": clip_seconds,
        "base_clips_per_source": base_clips,
        "content_minutes_per_source_hour": content_minutes_per_hour,
        "maximum_clips_per_source": max_clips,
        "maximum_content_minutes_per_source": round(max_clips * clip_seconds / 60.0, 3),
    }
