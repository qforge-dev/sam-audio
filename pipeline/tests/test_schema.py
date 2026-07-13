from sam_audio_pipeline.schema import QueueTask, ReviewDecision, VerificationStatus


def test_queue_task_round_trip() -> None:
    task = QueueTask(
        task_id="task-1",
        task_type="separate_chunk",
        job_id="job-1",
        source_id="source-1",
        chunk_id="000001",
    )
    assert QueueTask.model_validate_json(task.model_dump_json()) == task


def test_review_and_verification_values_are_stable() -> None:
    assert list(ReviewDecision) == ["pass", "fail", "pending"]
    assert list(VerificationStatus) == ["success", "uncertain", "failure"]
