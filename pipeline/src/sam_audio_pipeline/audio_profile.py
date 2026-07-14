"""Backfill original-file audio profiles for durable pipeline sources."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .audio import probe_audio_profile
from .schema import utc_now

if TYPE_CHECKING:
    from .aws import PipelineAWS

logger = logging.getLogger(__name__)


def backfill_audio_profiles(
    aws: PipelineAWS,
    job_ids: list[str],
    *,
    force: bool = False,
) -> dict[str, int]:
    summary = {
        "jobs": 0,
        "sources": 0,
        "skipped": 0,
        "missing_source": 0,
        "failed": 0,
    }
    for job_id in job_ids:
        items = aws.query_partition(f"JOB#{job_id}")
        sources = [item for item in items if item.get("entity") == "source"]
        summary["jobs"] += 1
        for source in sources:
            if source.get("audio_profile") and not force:
                summary["skipped"] += 1
                continue
            if not source.get("s3_key") or not aws.object_exists(
                str(source["s3_key"])
            ):
                summary["missing_source"] += 1
                continue
            try:
                with tempfile.TemporaryDirectory(
                    prefix="sam-audio-profile-backfill-"
                ) as temporary:
                    suffix = Path(str(source.get("filename") or "source.audio")).suffix
                    local_source = Path(temporary) / f"source{suffix or '.audio'}"
                    aws.download_file(str(source["s3_key"]), local_source)
                    profile = probe_audio_profile(local_source)
                aws.update(
                    f"JOB#{job_id}",
                    str(source["SK"]),
                    {
                        "audio_profile": profile,
                        "input_channels": int(profile["channels"]),
                        "updated_at": utc_now(),
                    },
                )
                summary["sources"] += 1
            except Exception:
                summary["failed"] += 1
                logger.exception(
                    "Failed to profile source %s in job %s",
                    source.get("source_id"),
                    job_id,
                )
    return summary


def _job_ids(aws: PipelineAWS, explicit: list[str], all_jobs: bool) -> list[str]:
    if explicit:
        return list(dict.fromkeys(explicit))
    if all_jobs:
        return [
            str(job["job_id"])
            for job in aws.query_index("JOBS", limit=None, newest_first=True)
        ]
    return []


def main() -> None:
    from .aws import PipelineAWS
    from .config import Settings

    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--job-id", action="append", default=[])
    selection.add_argument("--all", action="store_true", dest="all_jobs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    aws = PipelineAWS(settings)
    result: dict[str, Any] = backfill_audio_profiles(
        aws,
        _job_ids(aws, args.job_id, args.all_jobs),
        force=args.force,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
