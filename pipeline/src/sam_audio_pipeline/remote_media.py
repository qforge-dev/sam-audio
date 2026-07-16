"""Optional SSH execution for retryable media preparation commands."""

from __future__ import annotations

import math
import os
import shlex
import subprocess
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path


def _enabled_tasks() -> set[str]:
    return {
        value.strip().lower()
        for value in os.environ.get("SAM_MEDIA_WORKER_REMOTE_TASKS", "").split(",")
        if value.strip()
    }


def command_for_media_worker(
    command: Sequence[str], *, task: str, timeout: float
) -> tuple[list[str], float]:
    """Wrap a command for bounded SSH execution when its task is offloaded.

    Paths are intentionally left unchanged. The coordinator and media worker
    must expose the shared media workspace at the same absolute path.
    """

    target = os.environ.get("SAM_MEDIA_WORKER_SSH_TARGET", "").strip()
    if not target or task.lower() not in _enabled_tasks():
        return list(command), timeout

    remote_timeout = max(1, math.ceil(timeout))
    unit = f"sam-media-{task.lower()}-{uuid.uuid4().hex}"
    remote = shlex.join(
        [
            "sudo",
            "systemd-run",
            "--quiet",
            "--wait",
            "--pipe",
            "--collect",
            f"--unit={unit}",
            f"--property=RuntimeMaxSec={remote_timeout}s",
            "--",
            *command,
        ]
    )
    wrapped = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        # Every media command can be long-lived. Sharing all downloader and
        # ffmpeg workers through one OpenSSH control connection hits sshd's
        # per-connection MaxSessions limit (10 by default), leaving most of
        # the remote CPU idle and intermittently failing work with
        # "session open refused". Independent connections let worker
        # concurrency map cleanly to remote systemd units.
        "ControlMaster=no",
    ]
    identity = os.environ.get("SAM_MEDIA_WORKER_SSH_IDENTITY", "").strip()
    if identity:
        wrapped.extend(["-i", identity])
    wrapped.extend([target, remote])
    # Let the remote timeout terminate and reap ffmpeg/yt-dlp before the local
    # coordinator tears down the SSH process group.
    return wrapped, timeout + 30.0


def shared_media_temp_root(task: str) -> Path | None:
    """Return the shared scratch directory required by remote media commands.

    Remote commands intentionally use the same absolute paths as the
    coordinator.  Callers that need to exchange temporary outputs must create
    them below this directory, which is mounted on both hosts.
    """

    target = os.environ.get("SAM_MEDIA_WORKER_SSH_TARGET", "").strip()
    normalized = task.strip().lower()
    if not target or normalized not in _enabled_tasks():
        return None
    configured = os.environ.get("SAM_MEDIA_WORKER_SHARED_TMP", "").strip()
    if not configured:
        raise RuntimeError(
            f"SAM_MEDIA_WORKER_SHARED_TMP is required for remote {normalized} work"
        )
    path = Path(configured)
    if not path.is_absolute():
        raise ValueError("SAM_MEDIA_WORKER_SHARED_TMP must be an absolute path")
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_remote_media(task: str) -> int:
    """Stop every remote unit for one coordinator stage after service exit."""

    target = os.environ.get("SAM_MEDIA_WORKER_SSH_TARGET", "").strip()
    normalized = task.strip().lower()
    if not target or normalized not in _enabled_tasks():
        return 0
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
    ]
    identity = os.environ.get("SAM_MEDIA_WORKER_SSH_IDENTITY", "").strip()
    if identity:
        command.extend(["-i", identity])
    remote = f"sudo systemctl stop 'sam-media-{normalized}-*' 2>/dev/null || true"
    if normalized in {"download", "extract"}:
        staging_variable = (
            "SAM_CONTINUOUS_SOURCE_STAGING_DIR"
            if normalized == "download"
            else "SAM_MEDIA_WORKER_SHARED_TMP"
        )
        staging = os.environ.get(staging_variable, "").strip()
        if staging:
            prefix = (
                ".source-download-*"
                if normalized == "download"
                else "sam-source-extract-*"
            )
            remote += (
                f"; find {shlex.quote(staging)} -mindepth 1 -maxdepth 1 "
                f"-type d -name {shlex.quote(prefix)} -exec rm -rf -- {{}} +"
            )
    command.extend([target, remote])
    try:
        return subprocess.run(command, check=False, timeout=30).returncode
    except (OSError, subprocess.TimeoutExpired):
        return 1


def main(arguments: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if arguments is None else arguments)
    if len(values) == 2 and values[0] == "cleanup":
        return cleanup_remote_media(values[1])
    raise SystemExit("usage: python -m sam_audio_pipeline.remote_media cleanup TASK")


if __name__ == "__main__":
    raise SystemExit(main())
