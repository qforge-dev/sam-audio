"""Independent persistent stages for cinematic source acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import youtube_random
from .source_frontier import (
    claim_source,
    connect_frontier,
    discovery_strategy_admission,
    discovery_strategy_snapshot,
    downloaded_queue_bytes,
    enqueue_sources,
    finish_source,
    frontier_counts,
    frontier_platform_counts,
    frontier_snapshot,
    heartbeat_worker,
    provider_circuit_record_failure,
    provider_circuit_record_success,
    release_worker_leases,
    retry_source,
)
from .source_scanner import (
    SCAN_POLICY_VERSION,
    M2DSourceScanner,
    load_cached_scan,
    region_passes_confidence_gate,
)
from .youtube_random import (
    MIN_SIDE_TO_TOTAL_DB,
    MIN_SOURCE_SAMPLE_RATE,
    _cinematic_candidate_priority,
    _current_proxy_asr,
    _download_full_source_for_scan,
    _group_candidates_by_video,
    _load_attempts,
    _permanent_media_error,
    _preflight_source_for_scan,
    _probe_source_proxy_asr,
    _proxy_asr_blocks_extraction,
    _scan_cache_path,
    _scan_group_has_remaining_work,
    _source_format,
    acquire_scanned_source_group,
    discover_candidates,
    load_catalog_source_guidance,
    write_manifest,
)

logger = logging.getLogger(__name__)
STAGED_RUN_SEAL_FILENAME = ".sealed.json"


def _productive_expansion_seeds(
    guidance: dict[str, dict[str, Any]], *, seed: int, limit: int = 8
) -> list[dict[str, Any]]:
    """Rotate through proven sources instead of crawling arbitrary channels."""
    productive = [item for item in guidance.values() if int(item.get("accepted", 0))]
    productive.sort(
        key=lambda item: (
            int(item.get("accepted", 0)),
            int(item.get("asr_accepted", 0)),
            int(item.get("scored", 0)),
        ),
        reverse=True,
    )
    # Keep a productive pool but rotate parents so graph traversal does not
    # repeatedly rediscover the same first page around the all-time best source.
    pool = productive[: max(limit * 8, limit)]
    randomizer = random.Random(f"{seed}:accepted-parent-expansion")
    randomizer.shuffle(pool)
    return [dict(item["record"]) for item in pool[:limit]]


def _admit_discovery_groups(
    groups: list[list[dict[str, Any]]],
    metrics: dict[str, dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Apply per-lane probe limits and quality circuit breakers."""
    admissions: dict[str, dict[str, Any]] = {}
    remaining: dict[str, int | None] = {}
    admitted: list[list[dict[str, Any]]] = []
    for group in groups:
        key = str(group[0].get("discovery_quality_key") or "legacy")
        if key not in admissions:
            item = metrics.get(
                key,
                {
                    "key": key,
                    "active_sources": 0,
                    "scan_evaluated_sources": 0,
                    "scan_passed_sources": 0,
                    "final_records": 0,
                    "final_accepted": 0,
                },
            )
            admissions[key] = discovery_strategy_admission(item)
            remaining[key] = admissions[key].get("new_source_allowance")
        allowance = remaining[key]
        if allowance is not None and allowance <= 0:
            continue
        admitted.append(group)
        if allowance is not None:
            remaining[key] = allowance - 1
    for key, admission in admissions.items():
        admission["admitted_this_batch"] = sum(
            1
            for group in admitted
            if str(group[0].get("discovery_quality_key") or "legacy") == key
        )
    return admitted, admissions


def _called_process_error_text(error: subprocess.CalledProcessError) -> str:
    """Return a bounded failure description that keeps the useful tool output."""
    output = youtube_random._redact_proxy_credentials(
        "\n".join(
            part.strip()
            for part in (error.stdout, error.stderr)
            if part and part.strip()
        )
    )
    if len(output) > 2_000:
        output = output[-2_000:]
    summary = f"{type(error).__name__}: {error}"
    return f"{summary}: {output}" if output else summary


def _format_unavailable(error: subprocess.CalledProcessError) -> bool:
    output = f"{error.stdout or ''}\n{error.stderr or ''}".lower()
    return "requested format is not available" in output


def _heartbeat(
    workspace: Path,
    worker: str,
    stage: str,
    *,
    state: str = "running",
    details: dict[str, Any] | None = None,
) -> None:
    connection = None
    try:
        connection = connect_frontier(workspace)
        heartbeat_worker(
            connection, worker, stage=stage, state=state, details=details or {}
        )
    except sqlite3.OperationalError as error:
        if "locked" not in str(error).lower():
            raise
        # Telemetry is best effort. A synchronized heartbeat burst must never
        # terminate a producer or consumer worker that is otherwise healthy.
        logger.warning("%s heartbeat lock contention; continuing", worker)
    finally:
        if connection is not None:
            connection.close()


def _release_previous_worker_leases(workspace: Path, worker: str, state: str) -> None:
    connection = None
    try:
        connection = connect_frontier(workspace)
        released = release_worker_leases(connection, worker=worker, state=state)
    except sqlite3.OperationalError as error:
        if "locked" not in str(error).lower():
            raise
        # claim_source can reclaim an unexpired lease owned by the same stable
        # worker ID, so startup cleanup is safe to skip during lock contention.
        logger.warning("%s lease cleanup lock contention; continuing", worker)
        return
    finally:
        if connection is not None:
            connection.close()
    if released:
        logger.info("%s released %d previous %s leases", worker, released, state)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _read_seed(path: Path, fallback: int) -> int:
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, OSError, ValueError):
        return fallback


def _write_seed(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(f"{seed}\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class DiscoverySettings:
    workspace: Path
    discovery_dir: Path
    catalog: Path | None = None
    source: str = "dailymotion"
    profile: str = "cinematic"
    clip_seconds: float = 30.0
    query_count: int = 500
    results_per_query: int = 100
    search_workers: int = 8
    minimum_candidates: int = 3600
    clips_per_video: int = 16
    source_content_minutes_per_hour: float = 10.0
    max_clips_per_video: int = 60
    discovered_high_water: int = 4000
    platform_high_water: int = 600
    scan_cache: Path | None = None
    cached_scan_high_water: int = 64


@dataclass(frozen=True)
class DownloadSettings:
    workspace: Path
    source_cache: Path
    staging_dir: Path | None = None
    downloaded_high_water: int = 64
    downloaded_high_water_bytes: int = 8 * 1024**3
    minimum_free_bytes: int = 0
    lease_seconds: float = 7200.0
    max_attempts: int = 4
    retry_backoff_seconds: float = 15.0
    circuit_failure_threshold: int = 5
    circuit_cooldown_seconds: float = 300.0
    circuit_max_cooldown_seconds: float = 3600.0
    provider_weights: dict[str, float] | None = None
    control_file: Path | None = None


@dataclass(frozen=True)
class ScanSettings:
    workspace: Path
    scan_cache: Path
    proxy_asr_request_dir: Path
    proxy_asr_result_dir: Path
    proxy_asr_mode: str = "enforce"
    proxy_asr_timeout_seconds: float = 120.0
    scanned_high_water: int = 64
    lease_seconds: float = 7200.0
    max_attempts: int = 3
    retry_backoff_seconds: float = 15.0
    clip_seconds: float = 30.0
    control_file: Path | None = None


@dataclass(frozen=True)
class ExtractSettings:
    workspace: Path
    runs_dir: Path
    scan_cache: Path
    catalog: Path
    proxy_asr_request_dir: Path
    proxy_asr_result_dir: Path
    proxy_asr_mode: str = "enforce"
    proxy_asr_timeout_seconds: float = 120.0
    lease_seconds: float = 1800.0
    max_attempts: int = 4
    retry_backoff_seconds: float = 10.0
    clip_seconds: float = 30.0
    run_target: int = 2000
    clips_per_video: int = 16
    source_content_minutes_per_hour: float = 10.0
    max_clips_per_video: int = 60
    control_file: Path | None = None


def adopt_cached_scans(
    connection: Any,
    cache_dir: Path,
    *,
    clip_seconds: float,
    scanned_high_water: int,
    require_proxy_asr: bool = True,
    guidance: dict[str, dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Skip transfer/model work for compatible positive and negative caches."""
    counts = frontier_counts(connection)
    passing_capacity = max(0, scanned_high_water - counts["scanned"])
    promoted = rejected = completed = 0
    guidance = guidance or {}
    timestamp = datetime.now(UTC).isoformat()
    connection.execute("BEGIN IMMEDIATE")
    try:
        rows = connection.execute(
            """SELECT source_key,candidate_json,state FROM source_jobs
            WHERE state IN ('discovered','scanned') AND lease_owner IS NULL"""
        ).fetchall()
        for row in rows:
            candidates = json.loads(row["candidate_json"])
            cached = load_cached_scan(
                _scan_cache_path(cache_dir, candidates[0]),
                clip_seconds=clip_seconds,
            )
            if cached is None:
                continue
            regions = _passing_regions(cached)
            blocked = _proxy_asr_blocks_extraction(cached)
            probe = _current_proxy_asr(cached)
            probe_accepted = bool(probe and probe.get("accepted") is True)
            rejection_reasons = list(cached.get("rejection_reasons") or [])
            remaining = _scan_group_has_remaining_work(
                candidates,
                cache_dir=cache_dir,
                guidance=guidance,
                clip_seconds=clip_seconds,
            )
            if regions and not blocked and not remaining:
                state = "complete"
                reason = None
                outcome = "cache_exhausted"
            elif (
                row["state"] == "scanned"
                and regions
                and not blocked
                and (not require_proxy_asr or probe_accepted)
            ):
                continue
            elif (
                row["state"] == "scanned"
                and regions
                and not blocked
                and require_proxy_asr
                and not probe_accepted
            ):
                changed = connection.execute(
                    """UPDATE source_jobs SET state='discovered',available_at=?,
                    terminal_reason=NULL,updated_at=? WHERE source_key=?
                    AND state='scanned' AND lease_owner IS NULL""",
                    (time.time(), timestamp, row["source_key"]),
                ).rowcount
                if changed:
                    connection.execute(
                        """INSERT INTO source_stage_events(
                        source_key,stage,outcome,worker,started_at,finished_at,
                        duration_seconds,details_json) VALUES(?,?,?,?,?,?,?,?)""",
                        (
                            row["source_key"],
                            "scan",
                            "cache_probe_required",
                            "discovery-cache",
                            timestamp,
                            timestamp,
                            0.0,
                            "{}",
                        ),
                    )
                continue
            elif regions and not blocked and (not require_proxy_asr or probe_accepted):
                if passing_capacity <= 0:
                    continue
                state = "scanned"
                reason = None
                outcome = "cache_reused"
            else:
                state = "rejected"
                reason = (
                    str(rejection_reasons[0])
                    if rejection_reasons
                    else (
                        "source_proxy_asr_rejected"
                        if blocked
                        else "source_m2d_no_match"
                    )
                )
                outcome = "cache_rejected"
            summary = {
                "cache_path": str(_scan_cache_path(cache_dir, candidates[0])),
                "passing_regions": len(regions),
                "cache_reused": True,
            }
            changed = connection.execute(
                """UPDATE source_jobs SET state=?,available_at=?,terminal_reason=?,
                scan_json=?,updated_at=? WHERE source_key=? AND state=?
                AND lease_owner IS NULL""",
                (
                    state,
                    time.time(),
                    reason,
                    json.dumps(summary, separators=(",", ":")),
                    timestamp,
                    row["source_key"],
                    row["state"],
                ),
            ).rowcount
            if not changed:
                continue
            if state == "scanned":
                passing_capacity -= 1
                promoted += 1
            elif state == "complete":
                completed += 1
            else:
                rejected += 1
            connection.execute(
                """INSERT INTO source_stage_events(
                source_key,stage,outcome,worker,started_at,finished_at,
                duration_seconds,details_json) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    row["source_key"],
                    "scan",
                    outcome,
                    "discovery-cache",
                    timestamp,
                    timestamp,
                    0.0,
                    json.dumps(summary, separators=(",", ":")),
                ),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return {"promoted": promoted, "rejected": rejected, "completed": completed}


def discover_into_frontier_once(
    settings: DiscoverySettings,
    *,
    seed: int,
) -> dict[str, Any]:
    """Discover one reproducible batch and enqueue only frontier capacity."""
    started = time.perf_counter()
    connection = connect_frontier(settings.workspace)
    guidance = (
        load_catalog_source_guidance(settings.catalog, platform=settings.source)
        if settings.catalog
        else {}
    )
    expansion_seeds = _productive_expansion_seeds(guidance, seed=seed)
    cache_adoption = {"promoted": 0, "rejected": 0, "completed": 0}
    if settings.scan_cache is not None:
        cache_adoption = adopt_cached_scans(
            connection,
            settings.scan_cache,
            clip_seconds=settings.clip_seconds,
            scanned_high_water=settings.cached_scan_high_water,
            guidance=guidance,
        )
    before = frontier_counts(connection)
    before_platforms = frontier_platform_counts(connection)
    platform_counts = before_platforms.get(
        settings.source,
        {
            state: 0
            for state in ("discovered", "downloaded", "scanned", "complete", "rejected")
        },
    )
    platform_active = sum(
        platform_counts.get(state, 0)
        for state in ("discovered", "downloaded", "scanned")
    )
    capacity = min(
        max(0, settings.discovered_high_water - before["discovered"]),
        max(0, settings.platform_high_water - platform_active),
    )
    if capacity == 0:
        connection.close()
        return {
            "seed": seed,
            "status": "high_water",
            "inserted_sources": 0,
            "frontier": before,
            "platform": settings.source,
            "platform_counts": platform_counts,
            "cache_adoption": cache_adoption,
            "duration_seconds": round(time.perf_counter() - started, 3),
        }
    batch_dir = settings.discovery_dir / f"batch-{seed}"
    minimum_candidates = min(
        settings.minimum_candidates,
        max(settings.clips_per_video, capacity * settings.clips_per_video),
    )
    previous_clip_seconds = youtube_random.CLIP_SECONDS
    try:
        youtube_random.CLIP_SECONDS = settings.clip_seconds
        candidates = discover_candidates(
            batch_dir,
            seed=seed,
            query_count=settings.query_count,
            results_per_query=settings.results_per_query,
            workers=settings.search_workers,
            minimum_candidates=minimum_candidates,
            profile=settings.profile,
            clips_per_video=settings.clips_per_video,
            source_content_minutes_per_hour=settings.source_content_minutes_per_hour,
            max_clips_per_video=settings.max_clips_per_video,
            source=settings.source,
            expansion_seeds=expansion_seeds,
        )
    finally:
        youtube_random.CLIP_SECONDS = previous_clip_seconds
    groups = _group_candidates_by_video(candidates, grouped=True)
    groups.sort(key=lambda group: _cinematic_candidate_priority(group[0]), reverse=True)
    eligible = [
        group
        for group in groups
        if settings.scan_cache is None
        or _scan_group_has_remaining_work(
            group,
            cache_dir=settings.scan_cache,
            guidance=guidance,
            clip_seconds=settings.clip_seconds,
        )
    ]
    # Serialize the final capacity/admission decision. Multiple discovery
    # processes run concurrently, so a read-then-enqueue sequence would allow
    # every process to consume the same per-strategy probe allowance.
    connection.execute("BEGIN IMMEDIATE")
    try:
        known = {
            str(row["source_key"])
            for row in connection.execute("SELECT source_key FROM source_jobs")
        }
        unseen = [
            group
            for group in eligible
            if f"{group[0].get('source_platform') or 'unknown'}:{group[0]['video_id']}"
            not in known
        ]
        strategy_metrics = discovery_strategy_snapshot(
            connection,
            catalog_path=settings.catalog,
            platform=settings.source,
        )
        admitted, strategy_admission = _admit_discovery_groups(unseen, strategy_metrics)
        selected = admitted[:capacity]
        inserted = enqueue_sources(
            connection,
            selected,
            priority=lambda group: _cinematic_candidate_priority(group[0]),
        )
        after = frontier_counts(connection)
        after_platform = frontier_platform_counts(connection).get(settings.source, {})
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    connection.close()
    result = {
        "seed": seed,
        "status": "completed",
        "candidate_count": len(candidates),
        "minimum_candidates": minimum_candidates,
        "unique_source_count": len(groups),
        "unseen_source_count": len(unseen),
        "admitted_source_count": len(admitted),
        "selected_source_count": len(selected),
        "inserted_sources": inserted,
        "frontier": after,
        "platform": settings.source,
        "platform_counts": after_platform,
        "cache_adoption": cache_adoption,
        "expansion_seed_count": len(expansion_seeds),
        "strategy_admission": strategy_admission,
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    _atomic_json(batch_dir / "frontier-result.json", result)
    return result


def run_discovery(args: argparse.Namespace) -> None:
    youtube_random.YOUTUBE_PROXY_CONFIG = getattr(args, "youtube_proxy_config", None)
    settings = DiscoverySettings(
        workspace=args.workspace,
        discovery_dir=args.discovery_dir,
        catalog=getattr(args, "catalog", None),
        source=args.source,
        profile=args.profile,
        clip_seconds=args.clip_seconds,
        query_count=args.query_count,
        results_per_query=args.results_per_query,
        search_workers=args.search_workers,
        minimum_candidates=args.minimum_candidates,
        clips_per_video=args.clips_per_video,
        source_content_minutes_per_hour=args.source_content_minutes_per_hour,
        max_clips_per_video=args.max_clips_per_video,
        discovered_high_water=args.discovered_high_water,
        platform_high_water=args.platform_high_water,
        scan_cache=args.scan_cache,
        cached_scan_high_water=args.cached_scan_high_water,
    )
    seed_file = args.seed_file or args.workspace / "source-discovery-next-seed"
    seed = _read_seed(seed_file, args.seed)
    _heartbeat(
        args.workspace,
        "discovery",
        "discovery",
        details={"seed": seed, "source": args.source},
    )
    while True:
        try:
            result = discover_into_frontier_once(settings, seed=seed)
        except Exception as error:
            result = {
                "seed": seed,
                "status": "retry",
                "inserted_sources": 0,
                "frontier": {},
                "duration_seconds": 0.0,
                "error": f"{type(error).__name__}: {error}",
            }
            _atomic_json(
                args.workspace / "source-discovery-status.json",
                {**result, "observed_at": time.time()},
            )
            logger.exception(
                "Discovery source %s seed %d failed; retrying", args.source, seed
            )
            if args.once:
                # The service controller runs one fresh child per batch. Do not
                # let one deterministically empty/rate-limited seed poison the
                # persistent frontier forever.
                _write_seed(seed_file, seed + 1)
                raise
            time.sleep(args.retry_seconds)
            continue
        _atomic_json(
            args.workspace / "source-discovery-status.json",
            {**result, "observed_at": time.time()},
        )
        _heartbeat(
            args.workspace,
            "discovery",
            "discovery",
            details={
                "seed": seed,
                "source": args.source,
                "status": result["status"],
                "inserted_sources": result["inserted_sources"],
            },
        )
        logger.info(
            "Discovery source %s seed %d: %s; inserted=%d frontier=%s in %.1fs",
            args.source,
            seed,
            result["status"],
            result["inserted_sources"],
            result["frontier"],
            result["duration_seconds"],
        )
        if result["status"] == "completed":
            seed += 1
            _write_seed(seed_file, seed)
        if args.once:
            return
        time.sleep(
            args.high_water_poll_seconds
            if result["status"] == "high_water"
            else args.poll_seconds
        )


def _source_cache_key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _stage_worker_enabled(
    path: Path | None,
    *,
    stage: str,
    worker_index: int,
    default_limit: int,
) -> bool:
    if path is None:
        return worker_index < default_limit
    try:
        control = json.loads(path.read_text())
        limit = int((control.get("limits") or {}).get(stage, default_limit))
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        limit = default_limit
    return worker_index < max(1, default_limit if limit < 1 else limit)


def _existing_download(target_dir: Path) -> dict[str, Any] | None:
    try:
        metadata = json.loads((target_dir / "metadata.json").read_text())
        source = target_dir / str(metadata["filename"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
        return None
    if not source.is_file() or source.stat().st_size == 0:
        return None
    return {**metadata, "downloaded_path": str(source)}


def _publish_download(staging: Path, target: Path) -> None:
    """Atomically publish a download, including across staging filesystems."""
    if staging.stat().st_dev == target.parent.stat().st_dev:
        os.replace(staging, target)
        return
    publishing = Path(tempfile.mkdtemp(prefix=".source-publish-", dir=target.parent))
    shutil.rmtree(publishing)
    try:
        shutil.copytree(staging, publishing)
        os.replace(publishing, target)
        shutil.rmtree(staging)
    finally:
        shutil.rmtree(publishing, ignore_errors=True)


def _reject_download(
    connection: Any,
    job: dict[str, Any],
    *,
    worker: str,
    reason: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    finish_source(
        connection,
        job["source_key"],
        worker=worker,
        expected_state="discovered",
        next_state="rejected",
        outcome="rejected",
        terminal_reason=reason,
        details={"reason": reason, **(details or {})},
    )
    return {"status": "rejected", "reason": reason, "source_key": job["source_key"]}


def download_source_once(
    settings: DownloadSettings,
    *,
    worker: str,
) -> dict[str, Any] | None:
    """Claim, transfer, validate, and atomically publish one source proxy."""
    connection = connect_frontier(settings.workspace)
    settings.source_cache.mkdir(parents=True, exist_ok=True)
    source_free_bytes = shutil.disk_usage(settings.source_cache).free
    if source_free_bytes <= settings.minimum_free_bytes:
        connection.close()
        return {
            "status": "high_water",
            "reason": "minimum_free_bytes",
            "free_bytes": source_free_bytes,
        }
    if (
        frontier_counts(connection)["downloaded"] >= settings.downloaded_high_water
        or downloaded_queue_bytes(connection) >= settings.downloaded_high_water_bytes
    ):
        connection.close()
        return {"status": "high_water"}
    job = claim_source(
        connection,
        "discovered",
        worker=worker,
        lease_seconds=settings.lease_seconds,
        respect_provider_circuits=True,
        provider_weights=settings.provider_weights,
    )
    if job is None:
        connection.close()
        return None
    base = job["candidates"][0]
    source_platform = str(base.get("source_platform") or "unknown")
    retrieval_base = {
        **base,
        "_youtube_proxy_attempt": max(0, int(job["stage_attempts"]) - 1),
    }
    target_dir = settings.source_cache / _source_cache_key(job["source_key"])
    settings.source_cache.mkdir(parents=True, exist_ok=True)
    staging: Path | None = None
    try:
        existing = _existing_download(target_dir)
        if existing is not None:
            provider_circuit_record_success(connection, source_platform)
            finish_source(
                connection,
                job["source_key"],
                worker=worker,
                expected_state="discovered",
                next_state="downloaded",
                updates={
                    "downloaded_path": existing["downloaded_path"],
                    "download_json": existing,
                },
                outcome="recovered",
                details={"bytes": existing.get("bytes", 0)},
            )
            connection.close()
            return {"status": "recovered", **existing}
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_info = _preflight_source_for_scan(retrieval_base)
        if not target_info.get("quality_format_available", True):
            provider_circuit_record_success(connection, source_platform)
            result = _reject_download(
                connection,
                job,
                worker=worker,
                reason="source_high_quality_format_unavailable",
                details={"available_proxy_format_id": target_info.get("format_id")},
            )
            connection.close()
            return result
        staging_root = settings.staging_dir or settings.source_cache
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".source-download-", dir=staging_root))
        transfer_started = time.perf_counter()
        source, info = _download_full_source_for_scan(retrieval_base, staging)
        transfer_seconds = time.perf_counter() - transfer_started
        source_format = _source_format(
            source, info, f"{source_platform}-source-frontier"
        )
        if int(source_format.get("channels") or 0) != 2:
            provider_circuit_record_success(connection, source_platform)
            result = _reject_download(
                connection,
                job,
                worker=worker,
                reason="source_not_stereo",
                details={"source_format": source_format},
            )
            connection.close()
            return result
        if int(source_format.get("sample_rate_hz") or 0) < MIN_SOURCE_SAMPLE_RATE:
            provider_circuit_record_success(connection, source_platform)
            result = _reject_download(
                connection,
                job,
                worker=worker,
                reason="source_sample_rate",
                details={"source_format": source_format},
            )
            connection.close()
            return result
        metadata = {
            "schema_version": 1,
            "source_key": job["source_key"],
            "filename": source.name,
            "bytes": source.stat().st_size,
            "download_seconds": round(transfer_seconds, 3),
            "source_format": source_format,
            "extraction_format_id": target_info.get("format_id"),
            "proxy_format_id": info.get("format_id"),
            "downloaded_at": time.time(),
        }
        _atomic_json(staging / "metadata.json", metadata)
        _publish_download(staging, target_dir)
        staging = None
        downloaded_path = str(target_dir / source.name)
        persisted = {**metadata, "downloaded_path": downloaded_path}
        provider_circuit_record_success(connection, source_platform)
        finish_source(
            connection,
            job["source_key"],
            worker=worker,
            expected_state="discovered",
            next_state="downloaded",
            updates={
                "downloaded_path": downloaded_path,
                "download_json": persisted,
            },
            details={
                "bytes": metadata["bytes"],
                "download_seconds": metadata["download_seconds"],
            },
        )
        connection.close()
        return {"status": "downloaded", **persisted}
    except subprocess.CalledProcessError as error:
        if _permanent_media_error(error):
            provider_circuit_record_success(connection, source_platform)
            result = _reject_download(
                connection,
                job,
                worker=worker,
                reason="source_permanently_unavailable",
            )
            connection.close()
            return result
        if _format_unavailable(error):
            provider_circuit_record_success(connection, source_platform)
            result = _reject_download(
                connection,
                job,
                worker=worker,
                reason="source_high_quality_format_unavailable",
                details={"error": _called_process_error_text(error)},
            )
            connection.close()
            return result
        message = _called_process_error_text(error)
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    attempt = int(job["stage_attempts"])
    circuit = provider_circuit_record_failure(
        connection,
        source_platform,
        message,
        failure_threshold=settings.circuit_failure_threshold,
        cooldown_seconds=settings.circuit_cooldown_seconds,
        max_cooldown_seconds=settings.circuit_max_cooldown_seconds,
    )
    state = retry_source(
        connection,
        job["source_key"],
        worker=worker,
        expected_state="discovered",
        error=message,
        backoff_seconds=min(
            300.0, settings.retry_backoff_seconds * (2 ** max(0, attempt - 1))
        ),
        max_attempts=settings.max_attempts,
    )
    connection.close()
    return {
        "status": state,
        "error": message,
        "source_key": job["source_key"],
        "provider_circuit": circuit,
    }


def _download_loop(
    settings: DownloadSettings,
    *,
    worker: str,
    worker_index: int,
    maximum_workers: int,
    poll_seconds: float,
) -> None:
    _release_previous_worker_leases(settings.workspace, worker, "discovered")
    last_heartbeat = 0.0
    while True:
        if not _stage_worker_enabled(
            settings.control_file,
            stage="download",
            worker_index=worker_index,
            default_limit=maximum_workers,
        ):
            if time.monotonic() - last_heartbeat >= 10.0:
                _heartbeat(
                    settings.workspace,
                    worker,
                    "download",
                    state="paused",
                    details={"last_status": "autoscaled_down"},
                )
                last_heartbeat = time.monotonic()
            time.sleep(poll_seconds)
            continue
        try:
            result = download_source_once(settings, worker=worker)
        except sqlite3.OperationalError as error:
            if "locked" not in str(error).lower():
                raise
            logger.warning("%s frontier lock contention; retrying", worker)
            result = {"status": "frontier_locked"}
        if time.monotonic() - last_heartbeat >= 10.0:
            _heartbeat(
                settings.workspace,
                worker,
                "download",
                details={"last_status": (result or {}).get("status", "idle")},
            )
            last_heartbeat = time.monotonic()
        if result is None or result.get("status") == "high_water":
            time.sleep(poll_seconds)
            continue
        logger.info("%s %s", worker, result)


def run_downloaders(args: argparse.Namespace) -> None:
    youtube_random.CLIP_SECONDS = args.clip_seconds
    youtube_random.YTDLP_PYTHON = str(args.yt_dlp_python)
    youtube_random.YOUTUBE_PROXY_CONFIG = getattr(args, "youtube_proxy_config", None)
    settings = DownloadSettings(
        workspace=args.workspace,
        source_cache=args.source_cache,
        staging_dir=args.staging_dir,
        downloaded_high_water=args.downloaded_high_water,
        downloaded_high_water_bytes=args.downloaded_high_water_bytes,
        minimum_free_bytes=args.minimum_free_bytes,
        lease_seconds=args.lease_seconds,
        max_attempts=args.max_attempts,
        retry_backoff_seconds=args.retry_backoff_seconds,
        circuit_failure_threshold=args.circuit_failure_threshold,
        circuit_cooldown_seconds=args.circuit_cooldown_seconds,
        circuit_max_cooldown_seconds=args.circuit_max_cooldown_seconds,
        provider_weights=args.provider_weights,
        control_file=args.control_file,
    )
    threads = [
        threading.Thread(
            target=_download_loop,
            kwargs={
                "settings": settings,
                "worker": f"download-{index}",
                "worker_index": index,
                "maximum_workers": args.workers,
                "poll_seconds": args.poll_seconds,
            },
            name=f"source-download-{index}",
        )
        for index in range(args.workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def _passing_regions(scan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        region
        for region in scan.get("regions", [])
        if region_passes_confidence_gate(region)
    ]


def _finish_scan_rejection(
    connection: Any,
    job: dict[str, Any],
    *,
    worker: str,
    reason: str,
    cache_path: Path,
    scan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "cache_path": str(cache_path),
        "passing_regions": len(_passing_regions(scan or {})),
    }
    finish_source(
        connection,
        job["source_key"],
        worker=worker,
        expected_state="downloaded",
        next_state="rejected",
        outcome="rejected",
        terminal_reason=reason,
        updates={"scan_json": summary},
        details={"reason": reason, **summary},
    )
    return {"status": "rejected", "reason": reason, "source_key": job["source_key"]}


def scan_source_once(
    settings: ScanSettings,
    scanner: Any,
    *,
    worker: str,
) -> dict[str, Any] | None:
    """Claim one downloaded source, scan it, and discard the bulky transfer."""
    connection = connect_frontier(settings.workspace)
    if frontier_counts(connection)["scanned"] >= settings.scanned_high_water:
        connection.close()
        return {"status": "high_water"}
    job = claim_source(
        connection,
        "downloaded",
        worker=worker,
        lease_seconds=settings.lease_seconds,
    )
    if job is None:
        connection.close()
        return None
    base = job["candidates"][0]
    source = Path(str(job["downloaded_path"] or ""))
    work_dir = source.parent
    cache_path = _scan_cache_path(settings.scan_cache, base)
    settings.scan_cache.mkdir(parents=True, exist_ok=True)
    completed = False
    try:
        cached = load_cached_scan(cache_path, clip_seconds=settings.clip_seconds)
        proxy: Path | None = None
        if cached is None:
            if not source.is_file():
                raise FileNotFoundError(f"Downloaded source is missing: {source}")
            download = job.get("download_json") or {}
            proxy = work_dir / "proxy.flac"
            proxy_started = time.perf_counter()
            scanner.create_proxy(source, proxy)
            proxy_seconds = time.perf_counter() - proxy_started
            stereo_metrics = scanner.stereo_metrics(proxy)
            common = {
                "policy": SCAN_POLICY_VERSION,
                "clip_seconds": settings.clip_seconds,
                "video_id": str(base["video_id"]),
                "source_format": download.get("source_format"),
                "extraction_format_id": download.get("extraction_format_id"),
                "source_metadata": {
                    "uploader": base.get("uploader"),
                    "search_query": base.get("search_query"),
                    "title": base.get("title"),
                },
                "source_stereo_metrics": stereo_metrics,
                "download_seconds": download.get("download_seconds"),
                "proxy_seconds": round(proxy_seconds, 3),
                "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "claimed_starts": [],
            }
            if float(stereo_metrics["side_to_total_db"]) < MIN_SIDE_TO_TOTAL_DB:
                cached = {
                    **common,
                    "rejection_reasons": ["source_dual_mono"],
                    "regions": [],
                }
            else:
                source_budget = int(base.get("source_clip_budget") or 16)
                cached = {
                    **scanner.scan(
                        proxy,
                        clip_seconds=settings.clip_seconds,
                        max_regions=max(60, source_budget * 3),
                    ),
                    **common,
                }
                regions = _passing_regions(cached)
                if regions and settings.proxy_asr_mode != "off":
                    try:
                        probe = _probe_source_proxy_asr(
                            proxy,
                            regions,
                            work_dir,
                            video_id=str(base["video_id"]),
                            request_dir=settings.proxy_asr_request_dir,
                            result_dir=settings.proxy_asr_result_dir,
                            timeout_seconds=settings.proxy_asr_timeout_seconds,
                        )
                    except TimeoutError as error:
                        probe = {
                            "policy": youtube_random.SOURCE_ASR_PROBE_POLICY,
                            "status": "timeout",
                            "accepted": None,
                            "checked_regions": [],
                            "error": str(error),
                        }
                    probe["enforced"] = settings.proxy_asr_mode == "enforce"
                    cached["proxy_asr"] = probe
            _atomic_json(cache_path, cached)
        regions = _passing_regions(cached)
        if (
            regions
            and settings.proxy_asr_mode != "off"
            and _current_proxy_asr(cached) is None
        ):
            if not source.is_file():
                raise FileNotFoundError(
                    f"Downloaded source is missing for ASR probe: {source}"
                )
            if proxy is None:
                proxy = work_dir / "proxy.flac"
                scanner.create_proxy(source, proxy)
            try:
                probe = _probe_source_proxy_asr(
                    proxy,
                    regions,
                    work_dir,
                    video_id=str(base["video_id"]),
                    request_dir=settings.proxy_asr_request_dir,
                    result_dir=settings.proxy_asr_result_dir,
                    timeout_seconds=settings.proxy_asr_timeout_seconds,
                )
            except TimeoutError as error:
                probe = {
                    "policy": youtube_random.SOURCE_ASR_PROBE_POLICY,
                    "status": "timeout",
                    "accepted": None,
                    "checked_regions": [],
                    "error": str(error),
                }
            probe["enforced"] = settings.proxy_asr_mode == "enforce"
            cached["proxy_asr"] = probe
            _atomic_json(cache_path, cached)
        elif regions and settings.proxy_asr_mode == "enforce":
            probe = _current_proxy_asr(cached)
            if probe is not None and not probe.get("enforced"):
                probe["enforced"] = True
                cached["proxy_asr"] = probe
                _atomic_json(cache_path, cached)
        if cached.get("rejection_reasons"):
            reason = str(cached["rejection_reasons"][0])
            result = _finish_scan_rejection(
                connection,
                job,
                worker=worker,
                reason=reason,
                cache_path=cache_path,
                scan=cached,
            )
        elif _proxy_asr_blocks_extraction(cached):
            result = _finish_scan_rejection(
                connection,
                job,
                worker=worker,
                reason="source_proxy_asr_rejected",
                cache_path=cache_path,
                scan=cached,
            )
        elif not regions:
            result = _finish_scan_rejection(
                connection,
                job,
                worker=worker,
                reason="source_m2d_no_match",
                cache_path=cache_path,
                scan=cached,
            )
        else:
            summary = {
                "cache_path": str(cache_path),
                "passing_regions": len(regions),
                "m2d_windows": cached.get("m2d_windows"),
                "scan_seconds": cached.get("scan_seconds"),
                "proxy_seconds": cached.get("proxy_seconds"),
                "proxy_asr_accepted": (cached.get("proxy_asr") or {}).get("accepted"),
            }
            finish_source(
                connection,
                job["source_key"],
                worker=worker,
                expected_state="downloaded",
                next_state="scanned",
                updates={"scan_json": summary},
                details=summary,
            )
            result = {"status": "scanned", "source_key": job["source_key"], **summary}
        completed = True
        connection.close()
        shutil.rmtree(work_dir, ignore_errors=True)
        return result
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        attempt = int(job["stage_attempts"])
        state = retry_source(
            connection,
            job["source_key"],
            worker=worker,
            expected_state="downloaded",
            error=message,
            backoff_seconds=min(
                300.0, settings.retry_backoff_seconds * (2 ** max(0, attempt - 1))
            ),
            max_attempts=settings.max_attempts,
        )
        connection.close()
        return {"status": state, "error": message, "source_key": job["source_key"]}
    finally:
        if completed:
            # The compatible scan cache is now the durable source artifact.
            shutil.rmtree(work_dir, ignore_errors=True)


def _scan_loop(
    settings: ScanSettings,
    scanner: Any,
    *,
    worker: str,
    worker_index: int,
    maximum_workers: int,
    poll_seconds: float,
) -> None:
    _release_previous_worker_leases(settings.workspace, worker, "downloaded")
    last_heartbeat = 0.0
    while True:
        if not _stage_worker_enabled(
            settings.control_file,
            stage="scan",
            worker_index=worker_index,
            default_limit=maximum_workers,
        ):
            if time.monotonic() - last_heartbeat >= 10.0:
                _heartbeat(
                    settings.workspace,
                    worker,
                    "scan",
                    state="paused",
                    details={"last_status": "autoscaled_down"},
                )
                last_heartbeat = time.monotonic()
            time.sleep(poll_seconds)
            continue
        try:
            result = scan_source_once(settings, scanner, worker=worker)
        except sqlite3.OperationalError as error:
            if "locked" not in str(error).lower():
                raise
            logger.warning("%s frontier lock contention; retrying", worker)
            result = {"status": "frontier_locked"}
        if time.monotonic() - last_heartbeat >= 10.0:
            _heartbeat(
                settings.workspace,
                worker,
                "scan",
                details={"last_status": (result or {}).get("status", "idle")},
            )
            last_heartbeat = time.monotonic()
        if result is None or result.get("status") == "high_water":
            time.sleep(poll_seconds)
            continue
        logger.info("%s %s", worker, result)


def run_scanners(args: argparse.Namespace) -> None:
    youtube_random.CLIP_SECONDS = args.clip_seconds
    scanner = M2DSourceScanner(
        m2d_repo=args.m2d_repo,
        checkpoint=args.checkpoint,
        class_labels=args.class_labels,
        ontology=args.ontology,
        device=args.device,
        batch_size=args.batch_size,
        inference_concurrency=args.inference_concurrency,
    )
    settings = ScanSettings(
        workspace=args.workspace,
        scan_cache=args.scan_cache,
        proxy_asr_request_dir=args.proxy_asr_request_dir,
        proxy_asr_result_dir=args.proxy_asr_result_dir,
        proxy_asr_mode=args.proxy_asr_mode,
        proxy_asr_timeout_seconds=args.proxy_asr_timeout_seconds,
        scanned_high_water=args.scanned_high_water,
        lease_seconds=args.lease_seconds,
        max_attempts=args.max_attempts,
        retry_backoff_seconds=args.retry_backoff_seconds,
        clip_seconds=args.clip_seconds,
        control_file=args.control_file,
    )
    threads = [
        threading.Thread(
            target=_scan_loop,
            kwargs={
                "settings": settings,
                "scanner": scanner,
                "worker": f"scan-{index}",
                "worker_index": index,
                "maximum_workers": args.workers,
                "poll_seconds": args.poll_seconds,
            },
            name=f"source-scan-{index}",
        )
        for index in range(args.workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


class ExtractRunWriter:
    """Own rotating manifests for exactly one sequential extraction thread."""

    def __init__(self, settings: ExtractSettings, worker_index: int) -> None:
        self.settings = settings
        self.worker_index = worker_index
        self.sequence = self._latest_sequence()
        self.run_dir: Path
        self.attempts_path: Path
        self.attempts: list[dict[str, Any]]
        self._open_sequence()

    def _latest_sequence(self) -> int:
        prefix = f"run-staged-{self.worker_index}-"
        values = []
        for path in self.settings.runs_dir.glob(f"{prefix}*"):
            try:
                values.append(int(path.name.removeprefix(prefix)))
            except ValueError:
                continue
        return max(values, default=0)

    def _open_sequence(self) -> None:
        while True:
            self.run_dir = (
                self.settings.runs_dir
                / f"run-staged-{self.worker_index}-{self.sequence:06d}"
            )
            if not (self.run_dir / STAGED_RUN_SEAL_FILENAME).exists():
                break
            self.sequence += 1
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.attempts_path = self.run_dir / "attempts.jsonl"
        self.attempts, _ = _load_attempts(self.attempts_path, self.run_dir)

    @property
    def success_count(self) -> int:
        return sum(item.get("retrieval_status") == "success" for item in self.attempts)

    def prepare(self) -> Path:
        if (
            not self.run_dir.is_dir()
            or (self.run_dir / STAGED_RUN_SEAL_FILENAME).exists()
        ):
            if (self.run_dir / STAGED_RUN_SEAL_FILENAME).exists():
                self.sequence += 1
            self._open_sequence()
        if self.success_count > self.settings.run_target - 8:
            self.sequence += 1
            self._open_sequence()
        return self.run_dir

    def append(self, results: list[dict[str, Any]]) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.attempts_path.open("a", encoding="utf-8") as output:
            for result in results:
                output.write(json.dumps(result, separators=(",", ":")) + "\n")
        self.attempts.extend(results)
        write_manifest(
            self.run_dir,
            self.attempts,
            target=self.settings.run_target,
            seed=self.worker_index * 1_000_000 + self.sequence,
            profile="cinematic",
            clips_per_video=self.settings.clips_per_video,
            source_content_minutes_per_hour=(
                self.settings.source_content_minutes_per_hour
            ),
            max_clips_per_video=self.settings.max_clips_per_video,
            source="source_frontier",
        )
        sealed_run = self.run_dir
        _atomic_json(
            sealed_run / STAGED_RUN_SEAL_FILENAME,
            {
                "schema_version": 1,
                "sealed_at": datetime.now(UTC).isoformat(),
                "worker_index": self.worker_index,
                "sequence": self.sequence,
                "record_count": len(self.attempts),
            },
        )
        self.sequence += 1
        self._open_sequence()
        return sealed_run


def _commit_extraction_claims(
    cache_dir: Path,
    group: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> int:
    """Commit attempted scan regions only after their manifest is durable."""
    starts = [
        float(result["clip_start_seconds"])
        for result in results
        if result.get("selection") == "whole_source_proxy_scan"
        and result.get("retrieval_status") in {"success", "rejected", "unavailable"}
    ]
    if not starts:
        return 0
    cache_path = _scan_cache_path(cache_dir, group[0])
    cached = json.loads(cache_path.read_text())
    cached["claimed_starts"] = sorted(
        set(float(value) for value in cached.get("claimed_starts", [])) | set(starts)
    )
    _atomic_json(cache_path, cached)
    return len(starts)


def extract_source_once(
    settings: ExtractSettings,
    writer: ExtractRunWriter,
    *,
    worker: str,
) -> dict[str, Any] | None:
    """Claim one passing scan and publish its selected full-quality excerpts."""
    connection = connect_frontier(settings.workspace)
    job = claim_source(
        connection,
        "scanned",
        worker=worker,
        lease_seconds=settings.lease_seconds,
    )
    if job is None:
        connection.close()
        return None
    group = job["candidates"]
    source_platform = str(group[0].get("source_platform") or "youtube")
    guidance = load_catalog_source_guidance(settings.catalog, platform=source_platform)
    try:
        results = acquire_scanned_source_group(
            group,
            writer.prepare(),
            scanner=None,
            cache_dir=settings.scan_cache,
            guidance=guidance,
            proxy_asr_mode=settings.proxy_asr_mode,
            proxy_asr_request_dir=settings.proxy_asr_request_dir,
            proxy_asr_result_dir=settings.proxy_asr_result_dir,
            proxy_asr_timeout_seconds=settings.proxy_asr_timeout_seconds,
            defer_claim_commit=True,
            youtube_proxy_attempt=max(0, int(job["stage_attempts"]) - 1),
        )
        statuses = {str(item.get("retrieval_status")) for item in results}
        if not results or "source_scan_unavailable" in statuses:
            error = next(
                (
                    str(
                        item.get("error")
                        or (item.get("source_scan") or {}).get("error")
                        or "source scan unavailable"
                    )
                    for item in results
                    if item.get("retrieval_status") == "source_scan_unavailable"
                ),
                "source scan lease was unavailable",
            )
            raise RuntimeError(error)
        sealed_run_dir = writer.append(results)
        _commit_extraction_claims(settings.scan_cache, group, results)
        successful = sum(item.get("retrieval_status") == "success" for item in results)
        remaining = _scan_group_has_remaining_work(
            group,
            cache_dir=settings.scan_cache,
            guidance=guidance,
        )
        next_state = "scanned" if remaining else "complete"
        finish_source(
            connection,
            job["source_key"],
            worker=worker,
            expected_state="scanned",
            next_state=next_state,
            outcome="partial" if remaining else "success",
            details={
                "clips_published": successful,
                "result_statuses": sorted(statuses),
                "remaining_regions": remaining,
                "run_dir": str(sealed_run_dir),
            },
        )
        connection.close()
        return {
            "status": next_state,
            "source_key": job["source_key"],
            "clips_published": successful,
            "result_statuses": sorted(statuses),
        }
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        attempt = int(job["stage_attempts"])
        state = retry_source(
            connection,
            job["source_key"],
            worker=worker,
            expected_state="scanned",
            error=message,
            backoff_seconds=min(
                120.0, settings.retry_backoff_seconds * (2 ** max(0, attempt - 1))
            ),
            max_attempts=settings.max_attempts,
        )
        connection.close()
        return {"status": state, "error": message, "source_key": job["source_key"]}


def _extract_loop(
    settings: ExtractSettings,
    *,
    worker_index: int,
    maximum_workers: int,
    poll_seconds: float,
) -> None:
    worker = f"extract-{worker_index}"
    _release_previous_worker_leases(settings.workspace, worker, "scanned")
    writer = ExtractRunWriter(settings, worker_index)
    last_heartbeat = 0.0
    while True:
        if not _stage_worker_enabled(
            settings.control_file,
            stage="extract",
            worker_index=worker_index,
            default_limit=maximum_workers,
        ):
            if time.monotonic() - last_heartbeat >= 10.0:
                _heartbeat(
                    settings.workspace,
                    worker,
                    "extract",
                    state="paused",
                    details={"last_status": "autoscaled_down"},
                )
                last_heartbeat = time.monotonic()
            time.sleep(poll_seconds)
            continue
        try:
            result = extract_source_once(settings, writer, worker=worker)
        except sqlite3.OperationalError as error:
            if "locked" not in str(error).lower():
                raise
            logger.warning("%s frontier lock contention; retrying", worker)
            result = {"status": "frontier_locked"}
        if time.monotonic() - last_heartbeat >= 10.0:
            _heartbeat(
                settings.workspace,
                worker,
                "extract",
                details={"last_status": (result or {}).get("status", "idle")},
            )
            last_heartbeat = time.monotonic()
        if result is None:
            time.sleep(poll_seconds)
            continue
        logger.info("%s %s", worker, result)


def run_extractors(args: argparse.Namespace) -> None:
    youtube_random.CLIP_SECONDS = args.clip_seconds
    youtube_random.YTDLP_PYTHON = str(args.yt_dlp_python)
    youtube_random.YOUTUBE_PROXY_CONFIG = getattr(args, "youtube_proxy_config", None)
    settings = ExtractSettings(
        workspace=args.workspace,
        runs_dir=args.runs_dir,
        scan_cache=args.scan_cache,
        catalog=args.catalog,
        proxy_asr_request_dir=args.proxy_asr_request_dir,
        proxy_asr_result_dir=args.proxy_asr_result_dir,
        proxy_asr_mode=args.proxy_asr_mode,
        proxy_asr_timeout_seconds=args.proxy_asr_timeout_seconds,
        lease_seconds=args.lease_seconds,
        max_attempts=args.max_attempts,
        retry_backoff_seconds=args.retry_backoff_seconds,
        clip_seconds=args.clip_seconds,
        run_target=args.run_target,
        clips_per_video=args.clips_per_video,
        source_content_minutes_per_hour=args.source_content_minutes_per_hour,
        max_clips_per_video=args.max_clips_per_video,
        control_file=args.control_file,
    )
    threads = [
        threading.Thread(
            target=_extract_loop,
            kwargs={
                "settings": settings,
                "worker_index": index,
                "maximum_workers": args.workers,
                "poll_seconds": args.poll_seconds,
            },
            name=f"source-extract-{index}",
        )
        for index in range(args.workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def source_autoscale_decision(
    limits: dict[str, int],
    *,
    counts: dict[str, int],
    cpu_percent: float,
    bounds: dict[str, tuple[int, int]],
    cpu_low: float,
    cpu_high: float,
    scan_backlog_high: int,
    extract_backlog_high: int,
    download_backlog_low: int,
    cpu_exempt_stages: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Take one conservative stage-concurrency step from frontier pressure."""
    result = {
        stage: max(bounds[stage][0], min(bounds[stage][1], int(limits[stage])))
        for stage in ("download", "scan", "extract")
    }
    actions: list[str] = []
    if cpu_percent >= cpu_high:
        for stage in ("download", "scan", "extract"):
            if stage not in cpu_exempt_stages and result[stage] > bounds[stage][0]:
                result[stage] -= 1
                actions.append(f"reduce_{stage}_for_cpu")
                break
    elif (
        counts["scanned"] >= extract_backlog_high
        and result["extract"] < bounds["extract"][1]
    ):
        result["extract"] += 1
        actions.append("increase_extract")
    elif (
        counts["downloaded"] >= scan_backlog_high and result["scan"] < bounds["scan"][1]
    ):
        result["scan"] += 1
        actions.append("increase_scan")
    elif (
        counts["discovered"] > 0
        and counts["downloaded"]
        <= (
            scan_backlog_high
            if "download" in cpu_exempt_stages
            else download_backlog_low
        )
        and ("download" in cpu_exempt_stages or cpu_percent < cpu_low)
        and result["download"] < bounds["download"][1]
    ):
        result["download"] += 1
        actions.append("increase_download")
    return {"limits": result, "actions": actions or ["hold"]}


def run_source_autoscaler_once(args: argparse.Namespace) -> dict[str, Any]:
    from .continuous_dataset import _cpu_percent

    control_path = args.control_file
    try:
        previous = json.loads(control_path.read_text())
        previous_limits = previous.get("limits") or {}
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        previous_limits = {}
    limits = {
        "download": int(previous_limits.get("download", args.download_initial)),
        "scan": int(previous_limits.get("scan", args.scan_initial)),
        "extract": int(previous_limits.get("extract", args.extract_initial)),
    }
    connection = connect_frontier(args.workspace)
    counts = frontier_counts(connection)
    connection.close()
    cpu = _cpu_percent(args.cpu_sample_seconds)
    bounds = {
        "download": (args.download_min, args.download_max),
        "scan": (args.scan_min, args.scan_max),
        "extract": (args.extract_min, args.extract_max),
    }
    decision = source_autoscale_decision(
        limits,
        counts=counts,
        cpu_percent=cpu,
        bounds=bounds,
        cpu_low=args.cpu_low,
        cpu_high=args.cpu_high,
        scan_backlog_high=args.scan_backlog_high,
        extract_backlog_high=args.extract_backlog_high,
        download_backlog_low=args.download_backlog_low,
        cpu_exempt_stages=frozenset(args.cpu_exempt_stage),
    )
    payload = {
        "schema_version": 1,
        "observed_at": time.time(),
        "cpu_percent": round(cpu, 2),
        "counts": counts,
        "bounds": {stage: list(value) for stage, value in bounds.items()},
        **decision,
    }
    _atomic_json(control_path, payload)
    _heartbeat(
        args.workspace,
        "source-autoscaler",
        "autoscaler",
        details={
            "cpu_percent": payload["cpu_percent"],
            "limits": payload["limits"],
            "decision": payload["actions"],
        },
    )
    return payload


def run_source_autoscaler(args: argparse.Namespace) -> None:
    while True:
        status = run_source_autoscaler_once(args)
        logger.info(
            "Source autoscaler CPU %.1f%% queues=%s limits=%s decision=%s",
            status["cpu_percent"],
            status["counts"],
            status["limits"],
            status["actions"],
        )
        if args.once:
            return
        time.sleep(args.interval_seconds)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _provider_weights(value: str) -> dict[str, float]:
    """Parse provider=weight pairs used for weighted download concurrency."""
    weights: dict[str, float] = {}
    if not value.strip():
        return weights
    for item in value.split(","):
        try:
            platform, raw_weight = item.split("=", 1)
            weight = float(raw_weight)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "provider weights must use platform=positive_number"
            ) from error
        platform = platform.strip()
        if not platform or weight <= 0:
            raise argparse.ArgumentTypeError(
                "provider weights must use platform=positive_number"
            )
        weights[platform] = weight
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    discover = commands.add_parser("discover")
    discover.add_argument("--workspace", type=Path, required=True)
    discover.add_argument("--discovery-dir", type=Path, required=True)
    discover.add_argument("--catalog", type=Path)
    discover.add_argument("--seed-file", type=Path)
    discover.add_argument("--seed", type=int, default=20260714)
    discover.add_argument(
        "--source",
        choices=youtube_random.SUPPORTED_DISCOVERY_SOURCES,
        default="dailymotion",
    )
    discover.add_argument("--youtube-proxy-config", type=Path)
    discover.add_argument("--profile", default="cinematic")
    discover.add_argument("--clip-seconds", type=float, default=30)
    discover.add_argument("--query-count", type=_positive_int, default=500)
    discover.add_argument("--results-per-query", type=_positive_int, default=100)
    discover.add_argument("--search-workers", type=_positive_int, default=8)
    discover.add_argument("--minimum-candidates", type=_positive_int, default=3600)
    discover.add_argument("--clips-per-video", type=_positive_int, default=16)
    discover.add_argument("--source-content-minutes-per-hour", type=float, default=10)
    discover.add_argument("--max-clips-per-video", type=_positive_int, default=60)
    discover.add_argument("--discovered-high-water", type=_positive_int, default=4000)
    discover.add_argument("--platform-high-water", type=_positive_int, default=600)
    discover.add_argument("--scan-cache", type=Path)
    discover.add_argument("--cached-scan-high-water", type=_positive_int, default=64)
    discover.add_argument("--poll-seconds", type=float, default=2)
    discover.add_argument("--high-water-poll-seconds", type=float, default=10)
    discover.add_argument("--retry-seconds", type=float, default=5)
    discover.add_argument("--once", action="store_true")

    download = commands.add_parser("download")
    download.add_argument("--workspace", type=Path, required=True)
    download.add_argument("--source-cache", type=Path, required=True)
    download.add_argument("--staging-dir", type=Path)
    download.add_argument("--workers", type=_positive_int, default=8)
    download.add_argument("--downloaded-high-water", type=_positive_int, default=64)
    download.add_argument(
        "--downloaded-high-water-bytes", type=_positive_int, default=8 * 1024**3
    )
    download.add_argument(
        "--minimum-free-bytes",
        type=int,
        default=0,
        help="Stop claiming new transfers when source storage reaches this free-space floor",
    )
    download.add_argument("--lease-seconds", type=float, default=7200)
    download.add_argument("--max-attempts", type=_positive_int, default=4)
    download.add_argument("--retry-backoff-seconds", type=float, default=15)
    download.add_argument("--circuit-failure-threshold", type=_positive_int, default=5)
    download.add_argument("--circuit-cooldown-seconds", type=float, default=300)
    download.add_argument("--circuit-max-cooldown-seconds", type=float, default=3600)
    download.add_argument(
        "--provider-weights",
        type=_provider_weights,
        default={},
        help="Comma-separated provider=weight concurrency shares",
    )
    download.add_argument("--poll-seconds", type=float, default=1)
    download.add_argument("--clip-seconds", type=float, default=30)
    download.add_argument("--yt-dlp-python", type=Path, required=True)
    download.add_argument("--youtube-proxy-config", type=Path)
    download.add_argument("--control-file", type=Path)

    scan = commands.add_parser("scan")
    scan.add_argument("--workspace", type=Path, required=True)
    scan.add_argument("--scan-cache", type=Path, required=True)
    scan.add_argument("--proxy-asr-request-dir", type=Path, required=True)
    scan.add_argument("--proxy-asr-result-dir", type=Path, required=True)
    scan.add_argument(
        "--proxy-asr-mode", choices=("off", "shadow", "enforce"), default="enforce"
    )
    scan.add_argument("--proxy-asr-timeout-seconds", type=float, default=120)
    scan.add_argument("--scanned-high-water", type=_positive_int, default=64)
    scan.add_argument("--workers", type=_positive_int, default=4)
    scan.add_argument("--lease-seconds", type=float, default=7200)
    scan.add_argument("--max-attempts", type=_positive_int, default=3)
    scan.add_argument("--retry-backoff-seconds", type=float, default=15)
    scan.add_argument("--poll-seconds", type=float, default=1)
    scan.add_argument("--clip-seconds", type=float, default=30)
    scan.add_argument("--m2d-repo", type=Path, required=True)
    scan.add_argument("--checkpoint", type=Path, required=True)
    scan.add_argument("--class-labels", type=Path, required=True)
    scan.add_argument("--ontology", type=Path, required=True)
    scan.add_argument("--device", default="cuda")
    scan.add_argument("--batch-size", type=_positive_int, default=128)
    scan.add_argument("--inference-concurrency", type=_positive_int, default=2)
    scan.add_argument("--control-file", type=Path)

    extract = commands.add_parser("extract")
    extract.add_argument("--workspace", type=Path, required=True)
    extract.add_argument("--runs-dir", type=Path, required=True)
    extract.add_argument("--scan-cache", type=Path, required=True)
    extract.add_argument("--catalog", type=Path, required=True)
    extract.add_argument("--proxy-asr-request-dir", type=Path, required=True)
    extract.add_argument("--proxy-asr-result-dir", type=Path, required=True)
    extract.add_argument(
        "--proxy-asr-mode", choices=("off", "shadow", "enforce"), default="enforce"
    )
    extract.add_argument("--proxy-asr-timeout-seconds", type=float, default=120)
    extract.add_argument("--workers", type=_positive_int, default=8)
    extract.add_argument("--lease-seconds", type=float, default=1800)
    extract.add_argument("--max-attempts", type=_positive_int, default=4)
    extract.add_argument("--retry-backoff-seconds", type=float, default=10)
    extract.add_argument("--poll-seconds", type=float, default=1)
    extract.add_argument("--clip-seconds", type=float, default=30)
    extract.add_argument("--run-target", type=_positive_int, default=2000)
    extract.add_argument("--clips-per-video", type=_positive_int, default=16)
    extract.add_argument("--source-content-minutes-per-hour", type=float, default=10)
    extract.add_argument("--max-clips-per-video", type=_positive_int, default=60)
    extract.add_argument("--yt-dlp-python", type=Path, required=True)
    extract.add_argument("--youtube-proxy-config", type=Path)
    extract.add_argument("--control-file", type=Path)

    autoscale = commands.add_parser("autoscale")
    autoscale.add_argument("--workspace", type=Path, required=True)
    autoscale.add_argument("--control-file", type=Path, required=True)
    autoscale.add_argument("--download-min", type=_positive_int, default=2)
    autoscale.add_argument("--download-max", type=_positive_int, default=16)
    autoscale.add_argument("--download-initial", type=_positive_int, default=8)
    autoscale.add_argument("--scan-min", type=_positive_int, default=1)
    autoscale.add_argument("--scan-max", type=_positive_int, default=4)
    autoscale.add_argument("--scan-initial", type=_positive_int, default=2)
    autoscale.add_argument("--extract-min", type=_positive_int, default=1)
    autoscale.add_argument("--extract-max", type=_positive_int, default=8)
    autoscale.add_argument("--extract-initial", type=_positive_int, default=4)
    autoscale.add_argument("--cpu-low", type=float, default=55)
    autoscale.add_argument("--cpu-high", type=float, default=85)
    autoscale.add_argument("--scan-backlog-high", type=_positive_int, default=16)
    autoscale.add_argument("--extract-backlog-high", type=_positive_int, default=16)
    autoscale.add_argument("--download-backlog-low", type=int, default=4)
    autoscale.add_argument("--cpu-sample-seconds", type=float, default=0.25)
    autoscale.add_argument(
        "--cpu-exempt-stage",
        action="append",
        choices=("download", "scan", "extract"),
        default=[],
        help="Stage whose workers execute off-host and do not consume local CPU",
    )
    autoscale.add_argument("--interval-seconds", type=float, default=10)
    autoscale.add_argument("--once", action="store_true")

    status = commands.add_parser("status")
    status.add_argument("--workspace", type=Path, required=True)
    status.add_argument("--window-minutes", type=float, default=15)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.command == "discover":
        run_discovery(args)
    elif args.command == "download":
        run_downloaders(args)
    elif args.command == "scan":
        run_scanners(args)
    elif args.command == "extract":
        run_extractors(args)
    elif args.command == "autoscale":
        run_source_autoscaler(args)
    elif args.command == "status":
        print(
            json.dumps(
                frontier_snapshot(args.workspace, window_minutes=args.window_minutes),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
