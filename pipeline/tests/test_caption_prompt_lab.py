from __future__ import annotations

from sam_audio_pipeline.caption_prompt_lab import evaluate_caption, summarize_results


def test_evaluate_caption_rewards_contract_and_timeline_coverage() -> None:
    parsed = {
        "description": (
            "A low engine drone fills the center while metallic impacts move toward "
            "the right. The layered mechanical texture grows denser and louder, then "
            "eases into a soft resonant tail. A distant airy wash remains underneath "
            "the main pulse, giving the space a broad industrial character. Short "
            "clicks appear at the edges before the final hum settles quietly. The "
            "sound stays moderately loud through the middle, with crisp transients "
            "against a smooth low-frequency bed and a gradual closing decay. The "
            "requested background contains no dialogue, intelligible speech, "
            "narration, or vocals."
        ),
        "timeline": [
            {
                "start_seconds": 0,
                "end_seconds": 15,
                "events": ["A centered engine drone builds under metallic impacts."],
            },
            {
                "start_seconds": 15,
                "end_seconds": 30,
                "events": ["The impacts recede and a soft resonant hum remains."],
            },
        ],
        "global_tags": ["Engine", "Mechanisms"],
        "sound_effects": ["Metallic impact"],
    }
    tag = {
        "windows": [
            {
                "start_seconds": 0,
                "end_seconds": 30,
                "top_labels": [
                    {"name": "Engine", "probability": 0.9},
                    {"name": "Mechanisms", "probability": 0.8},
                ],
            }
        ]
    }

    result = evaluate_caption(
        parsed,
        {"format": "description_timeline_sections_v2"},
        tag,
    )

    assert result["timeline_coverage"] == 1.0
    assert result["detailed_timeline_events"] == 2
    assert "engine" in result["matched_m2d_tags"]
    assert result["style_dimensions"] == {
        "spatial": True,
        "dynamics": True,
        "texture": True,
        "transitions": True,
    }
    assert result["style_dimension_count"] == 4
    assert result["score"] >= 80


def test_prompt_summary_ranks_contract_success_before_style_score() -> None:
    def result(
        variant: str, *, score: float, status: str, seconds: float
    ) -> dict:
        return {
            "job_id": 1,
            "variant": variant,
            "processing_seconds": seconds,
            "parse": {"format": "audio_flamingo_native_timeline_v2"},
            "evaluation": {
                "score": score,
                "validation": {
                    "status": status,
                    "review_reasons": (
                        [] if status == "success" else ["scene_timeline_underdescribed"]
                    ),
                    "signals": {
                        "description_word_count": 100,
                        "timeline_event_count": 5,
                    },
                },
                "style_dimension_count": 4,
            },
        }

    summary = summarize_results(
        [
            result(
                "native_v2_grounded",
                score=90,
                status="success",
                seconds=20,
            ),
            result(
                "native_v2_audio_only",
                score=99,
                status="review",
                seconds=10,
            ),
        ]
    )

    assert summary["winner"] == "native_v2_grounded"
    assert summary["variants"]["native_v2_audio_only"][
        "contract_failure_reasons"
    ] == {"scene_timeline_underdescribed": 1}
