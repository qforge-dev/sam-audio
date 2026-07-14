from __future__ import annotations

import pytest

from sam_audio_pipeline.source_diversity import source_clip_budget


def test_duration_scaled_budget_preserves_base_then_grows_for_long_sources() -> None:
    assert source_clip_budget(30 * 60, clip_seconds=30) == 16
    assert source_clip_budget(48 * 60, clip_seconds=30) == 16
    assert source_clip_budget(3600, clip_seconds=30) == 20
    assert source_clip_budget(2 * 3600, clip_seconds=30) == 40
    assert source_clip_budget(3 * 3600, clip_seconds=30) == 60


def test_duration_scaled_budget_has_an_absolute_diversity_guardrail() -> None:
    assert source_clip_budget(12 * 3600, clip_seconds=30) == 60
    assert source_clip_budget(48 * 3600, clip_seconds=30) == 60


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"clip_seconds": 0}, "clip_seconds"),
        ({"base_clips": 0}, "base_clips"),
        ({"content_minutes_per_hour": 0}, "content_minutes_per_hour"),
        ({"base_clips": 16, "max_clips": 15}, "max_clips"),
    ],
)
def test_duration_scaled_budget_rejects_invalid_policy(
    overrides: dict[str, float | int], message: str
) -> None:
    settings: dict[str, float | int] = {"clip_seconds": 30}
    settings.update(overrides)
    with pytest.raises(ValueError, match=message):
        source_clip_budget(3600, **settings)  # type: ignore[arg-type]
