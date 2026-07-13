from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sam_audio_pipeline.handlers import (
    FlamingoHandler,
    IngestHandler,
    SeparationHandler,
    _scene_presence,
)
from sam_audio_pipeline.schema import QueueTask


def metadata(
    order: str,
    final: str,
    stage1: str,
    stage2: str,
    stage1_quality: float,
    stage2_quality: float,
) -> dict[str, Any]:
    cascade = ["music", "voice"] if order == "music_first" else ["voice", "music"]
    return {
        "requested_order": order,
        "cascade_order": cascade,
        "verification_status": final,
        "verification": {"stage_statuses": {"stage1": stage1, "stage2": stage2}},
        "stages": {
            "stage1": {
                "verification": {"judge_quality_score": stage1_quality},
            },
            "stage2": {
                "verification": {"judge_quality_score": stage2_quality},
            },
        },
        "inference_timings_ms": {"service_total": 1234.0},
    }


class FakeClient:
    def __init__(self, results: dict[str, dict[str, Any]]):
        self.results = results
        self.orders: list[str] = []

    def separate(
        self,
        _: Path,
        __: Path,
        *,
        order: str,
        targets: tuple[str, ...] = ("music", "voice"),
    ):
        self.orders.append(order)
        return SimpleNamespace(metadata=self.results[order])


def policy_handler(results: dict[str, dict[str, Any]]) -> SeparationHandler:
    handler = object.__new__(SeparationHandler)
    handler.client = FakeClient(results)
    return handler


def test_failure_retries_voice_first_and_selects_the_better_route(
    tmp_path: Path,
) -> None:
    handler = policy_handler(
        {
            "music_first": metadata(
                "music_first", "failure", "failure", "failure", 4.0, 4.1
            ),
            "voice_first": metadata(
                "voice_first", "uncertain", "success", "uncertain", 4.6, 4.3
            ),
        }
    )

    selected = handler._separate_with_policy(tmp_path / "input.wav", tmp_path)

    assert handler.client.orders == ["music_first", "voice_first"]
    routing = selected.metadata["adaptive_routing"]
    assert routing["selected_order"] == "voice_first"
    assert routing["trigger"] == "primary_failure_retry"
    assert len(routing["attempts"]) == 2
    assert routing["attempts"][1]["status_by_kind"] == {
        "voice": "success",
        "music": "uncertain",
    }


def test_successful_primary_route_does_not_retry(tmp_path: Path) -> None:
    handler = policy_handler(
        {
            "music_first": metadata(
                "music_first", "success", "success", "success", 4.8, 4.7
            )
        }
    )

    selected = handler._separate_with_policy(tmp_path / "input.wav", tmp_path)

    assert handler.client.orders == ["music_first"]
    assert selected.metadata["adaptive_routing"]["selected_order"] == "music_first"


def test_no_presence_targets_bypasses_sam_and_keeps_input_as_sfx(
    tmp_path: Path,
) -> None:
    handler = object.__new__(SeparationHandler)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"stereo-placeholder")

    selected = handler._separate_with_policy(
        input_path, tmp_path / "result", targets=()
    )

    assert selected.stems == {"sfx": input_path}
    assert selected.metadata["requested_targets"] == []
    assert selected.metadata["inference_timings_ms"]["service_total"] == 0.0
    assert (
        selected.metadata["adaptive_routing"]["trigger"]
        == "semantic_presence_no_targets"
    )
    assert selected.metadata["adaptive_routing"]["selected_order"] == "sfx_only"


class RefreshAWS:
    def __init__(self, items: list[dict[str, Any]]):
        self.items = items
        self.updates: list[dict[str, Any]] = []

    def query_partition(self, _: str) -> list[dict[str, Any]]:
        return self.items

    def update(self, _: str, sk: str, values: dict[str, Any]) -> None:
        assert sk == "META"
        self.updates.append(values)


def test_all_sound_gated_sources_finish_without_model_tasks() -> None:
    aws = RefreshAWS(
        [
            {"entity": "source", "status": "complete"},
            {"entity": "source", "status": "complete"},
            {"entity": "chunk", "status": "skipped"},
            {"entity": "chunk", "status": "skipped"},
        ]
    )
    handler = object.__new__(IngestHandler)
    handler.aws = aws

    handler._refresh_job("job-1")

    assert aws.updates[-1]["status"] == "complete"
    assert aws.updates[-1]["completed_sources"] == 2


def test_job_stays_running_when_any_source_has_audible_chunks() -> None:
    aws = RefreshAWS(
        [
            {"entity": "source", "status": "complete"},
            {"entity": "source", "status": "chunked"},
            {"entity": "chunk", "status": "skipped"},
            {"entity": "chunk", "status": "queued"},
        ]
    )
    handler = object.__new__(IngestHandler)
    handler.aws = aws

    handler._refresh_job("job-1")

    assert aws.updates[-1]["status"] == "running"


def test_scene_presence_parses_flamingo_python_dict_and_fails_open() -> None:
    parsed = _scene_presence(
        "{'has_music': False, 'has_voices': False, 'scene_description': 'writing'}"
    )
    assert parsed["parsed"] is True
    assert parsed["has_music"] is False
    assert parsed["has_voices"] is False
    assert _scene_presence("not structured") == {
        "has_music": True,
        "has_voices": True,
        "parsed": False,
        "raw_text": "not structured",
    }


class PresenceAWS:
    def __init__(self) -> None:
        self.items = [
            {
                "SK": "SOURCE#source-1",
                "entity": "source",
                "source_id": "source-1",
                "status": "analyzing",
            },
            {
                "SK": "CHUNK#source-1#000000",
                "entity": "chunk",
                "source_id": "source-1",
                "chunk_id": "000000",
                "status": "waiting_scene",
            },
        ]
        self.queued: list[QueueTask] = []

    def update(self, _: str, sk: str, values: dict[str, Any]) -> None:
        next(item for item in self.items if item["SK"] == sk).update(values)

    def query_partition(self, _: str) -> list[dict[str, Any]]:
        return self.items

    def send_task(self, _: str, task: QueueTask) -> None:
        self.queued.append(task)


def test_scene_preflight_queues_sfx_passthrough_when_targets_are_absent() -> None:
    aws = PresenceAWS()
    handler = object.__new__(FlamingoHandler)
    handler.aws = aws
    handler.settings = SimpleNamespace(sam_queue_url="sam")

    handler._schedule_separation(
        QueueTask(
            task_id="scene-1",
            task_type="describe_scene",
            job_id="job-1",
            source_id="source-1",
        ),
        {
            "model": "flamingo",
            "text": "{'has_music': False, 'has_voices': False}",
        },
    )

    assert aws.queued[0].targets == []
    chunk = aws.items[1]
    assert chunk["status"] == "queued"
    assert chunk["presence_gate"]["targets"] == []
    assert aws.items[0]["semantic_presence"]["parsed"] is True


def test_final_sam_chunk_closes_job_when_annotations_already_finished() -> None:
    aws = RefreshAWS(
        [
            {"entity": "source", "status": "chunked", "audible_chunk_count": 1},
            {"entity": "chunk", "status": "complete"},
            {"entity": "stem", "stem_type": "music"},
            {"entity": "stem", "stem_type": "voice"},
            {"entity": "stem", "stem_type": "sfx"},
            {"entity": "model_task", "status": "complete"},
            {"entity": "model_task", "status": "complete"},
            {"entity": "model_task", "status": "complete"},
        ]
    )
    handler = object.__new__(SeparationHandler)
    handler.aws = aws

    handler._refresh_job("job-1")

    assert aws.updates[-1]["status"] == "complete"
    assert aws.updates[-1]["completed_sources"] == 1
    assert aws.updates[-1]["completed_chunks"] == 1
    assert aws.updates[-1]["failed_chunks"] == 0
