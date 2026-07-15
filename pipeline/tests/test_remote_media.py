from __future__ import annotations

import shlex

from sam_audio_pipeline.remote_media import command_for_media_worker


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


def test_media_command_wraps_only_enabled_tasks_and_quotes_arguments(monkeypatch) -> None:
    monkeypatch.setenv("SAM_MEDIA_WORKER_SSH_TARGET", "ubuntu@172.31.0.10")
    monkeypatch.setenv("SAM_MEDIA_WORKER_SSH_IDENTITY", "/home/ubuntu/.ssh/media")
    monkeypatch.setenv("SAM_MEDIA_WORKER_REMOTE_TASKS", "download,ffmpeg")
    original = [
        "yt-dlp",
        "-o",
        "/shared/source.%(ext)s",
        "https://example.test/watch?v=one&list=two",
    ]

    command, timeout = command_for_media_worker(
        original, task="download", timeout=90.2
    )

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


def test_media_command_keeps_unlisted_task_local(monkeypatch) -> None:
    monkeypatch.setenv("SAM_MEDIA_WORKER_SSH_TARGET", "ubuntu@172.31.0.10")
    monkeypatch.setenv("SAM_MEDIA_WORKER_REMOTE_TASKS", "ffmpeg")

    command, _ = command_for_media_worker(
        ["yt-dlp", "https://example.test"], task="download", timeout=30
    )

    assert command[0] == "yt-dlp"


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
    assert "/shared/.staging" in observed[-1]
