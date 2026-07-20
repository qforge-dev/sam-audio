from __future__ import annotations

import shlex

from sam_audio_pipeline.remote_media import (
    command_for_media_worker,
    shared_media_temp_root,
)


def test_media_command_is_local_without_remote_configuration(monkeypatch) -> None:
    monkeypatch.delenv("SAM_MEDIA_WORKER_SSH_TARGET", raising=False)
    monkeypatch.delenv("SAM_MEDIA_WORKER_REMOTE_TASKS", raising=False)

    command, timeout = command_for_media_worker(
        ["ffmpeg", "-i", "source.mp4", "proxy.flac"],
        task="ffmpeg",
        timeout=30,
    )

    assert command == ["ffmpeg", "-i", "source.mp4", "proxy.flac"]
    assert timeout == 30


def test_media_command_wraps_only_enabled_tasks_and_quotes_arguments(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SAM_MEDIA_WORKER_SSH_TARGET", "ubuntu@172.31.0.10")
    monkeypatch.setenv("SAM_MEDIA_WORKER_SSH_IDENTITY", "/home/ubuntu/.ssh/media")
    monkeypatch.setenv("SAM_MEDIA_WORKER_REMOTE_TASKS", "download,ffmpeg")
    original = [
        "yt-dlp",
        "-o",
        "/shared/source.%(ext)s",
        "https://example.test/watch?v=one&list=two",
    ]

    command, timeout = command_for_media_worker(original, task="download", timeout=90.2)

    assert command[0] == "ssh"
    assert "ControlMaster=no" in command
    assert not any(value.startswith("ControlPath=") for value in command)
    assert command[-2] == "ubuntu@172.31.0.10"
    assert command[command.index("-i") + 1] == "/home/ubuntu/.ssh/media"
    remote = command[-1]
    parsed = shlex.split(remote)
    assert parsed[:6] == [
        "sudo",
        "systemd-run",
        "--quiet",
        "--wait",
        "--pipe",
        "--collect",
    ]
    assert parsed[6].startswith("--unit=sam-media-download-")
    assert parsed[7:9] == ["--property=RuntimeMaxSec=91s", "--"]
    assert parsed[9:] == original
    assert timeout == 120.2


def test_media_command_can_avoid_transient_units_for_high_fanout_tasks(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SAM_MEDIA_WORKER_SSH_TARGET", "ubuntu@172.31.0.10")
    monkeypatch.setenv("SAM_MEDIA_WORKER_REMOTE_TASKS", "download,ffmpeg")
    monkeypatch.setenv("SAM_MEDIA_WORKER_DIRECT_TASKS", "download")
    original = ["yt-dlp", "https://example.test/watch?v=one&list=two"]

    command, timeout = command_for_media_worker(original, task="download", timeout=90.2)

    parsed = shlex.split(command[-1])
    assert parsed[:2] == ["bash", "-c"]
    bounded = shlex.split(parsed[2])
    assert bounded[:3] == ["exec", "-a", bounded[2]]
    assert bounded[2].startswith("sam-media-direct-download-")
    assert bounded[3:7] == [
        "timeout",
        "--signal=TERM",
        "--kill-after=15s",
        "91s",
    ]
    assert bounded[7:] == original
    assert "systemd-run" not in command[-1]
    assert timeout == 120.2


def test_media_command_keeps_unlisted_task_local(monkeypatch) -> None:
    monkeypatch.setenv("SAM_MEDIA_WORKER_SSH_TARGET", "ubuntu@172.31.0.10")
    monkeypatch.setenv("SAM_MEDIA_WORKER_REMOTE_TASKS", "ffmpeg")

    command, _ = command_for_media_worker(
        ["yt-dlp", "https://example.test"], task="download", timeout=30
    )

    assert command[0] == "yt-dlp"


def test_shared_media_temp_root_is_only_required_for_remote_task(
    monkeypatch, tmp_path
) -> None:
    shared = tmp_path / "shared"
    monkeypatch.setenv("SAM_MEDIA_WORKER_SSH_TARGET", "ubuntu@172.31.0.10")
    monkeypatch.setenv("SAM_MEDIA_WORKER_REMOTE_TASKS", "extract")
    monkeypatch.setenv("SAM_MEDIA_WORKER_SHARED_TMP", str(shared))

    assert shared_media_temp_root("download") is None
    assert shared_media_temp_root("extract") == shared
    assert shared.is_dir()


def test_shared_media_temp_root_rejects_missing_remote_path(monkeypatch) -> None:
    monkeypatch.setenv("SAM_MEDIA_WORKER_SSH_TARGET", "ubuntu@172.31.0.10")
    monkeypatch.setenv("SAM_MEDIA_WORKER_REMOTE_TASKS", "extract")
    monkeypatch.delenv("SAM_MEDIA_WORKER_SHARED_TMP", raising=False)

    try:
        shared_media_temp_root("extract")
    except RuntimeError as error:
        assert "SAM_MEDIA_WORKER_SHARED_TMP" in str(error)
    else:
        raise AssertionError("missing shared scratch path was accepted")


def test_cleanup_stops_task_units_and_removes_download_staging(
    monkeypatch,
) -> None:
    from sam_audio_pipeline import remote_media

    monkeypatch.setenv("SAM_MEDIA_WORKER_SSH_TARGET", "ubuntu@172.31.0.10")
    monkeypatch.setenv("SAM_MEDIA_WORKER_REMOTE_TASKS", "download,ffmpeg")
    monkeypatch.setenv("SAM_CONTINUOUS_SOURCE_STAGING_DIR", "/shared/.staging")
    observed: list[str] = []

    class Result:
        returncode = 0

    def fake_run(command, **_kwargs):
        observed.extend(command)
        return Result()

    monkeypatch.setattr(remote_media.subprocess, "run", fake_run)

    assert remote_media.cleanup_remote_media("download") == 0
    assert observed[-2] == "ubuntu@172.31.0.10"
    assert "sam-media-download-*" in observed[-1]
    assert "[s]am-media-direct-download-" in observed[-1]
    assert "/shared/.staging" in observed[-1]


def test_cleanup_stops_extract_units_and_removes_shared_scratch(monkeypatch) -> None:
    from sam_audio_pipeline import remote_media

    monkeypatch.setenv("SAM_MEDIA_WORKER_SSH_TARGET", "ubuntu@172.31.0.10")
    monkeypatch.setenv("SAM_MEDIA_WORKER_REMOTE_TASKS", "extract")
    monkeypatch.setenv("SAM_MEDIA_WORKER_SHARED_TMP", "/shared/.staging")
    observed: list[str] = []

    class Result:
        returncode = 0

    def fake_run(command, **_kwargs):
        observed.extend(command)
        return Result()

    monkeypatch.setattr(remote_media.subprocess, "run", fake_run)

    assert remote_media.cleanup_remote_media("extract") == 0
    assert "sam-media-extract-*" in observed[-1]
    assert "sam-source-extract-*" in observed[-1]
