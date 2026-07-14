# Independent acquisition pipeline rollout

## Objective

Replace the batch-shaped acquisition loop with a persistent, bounded producer /
consumer pipeline. Discovery, source transfer, whole-source M2D scanning, source
ASR probing, and full-quality clip extraction must continue independently. A
slow source or a discovery refresh must not stop unrelated stages.

The downstream raw-clip M2D, clip ASR, assembly, review, and snapshot pipeline
remains unchanged. The existing continuous catalog stays authoritative.

## Baseline

Measured on the production host at `2026-07-14T19:49Z`:

| Window | Raw audio h/h | Accepted audio h/h |
| --- | ---: | ---: |
| 5 minutes | 11.41 | 7.08 |
| 15 minutes | 7.06 | 4.37 |
| 60 minutes | 7.09 | 4.09 |

Other observations:

- M2D, ASR, and assembly queues were all empty.
- CPU was 43%, GPU utilization was 0%, and the autoscaler reported
  `source_yield`.
- Recent five-minute source-scan cohorts ranged from 0.13 to 4.09 candidate
  regions per source.
- Both batch acquisition producers periodically entered search and model-load
  startup together. Their short productive bursts therefore did not represent
  sustained throughput.
- Over the trailing hour, 845 raw clips became 487 accepted clips: a 57.6%
  end-to-end yield.

The primary target is sustained accepted throughput. A higher raw rate that
only increases rejection or queue depth is not a win.

## Target architecture

```text
discovery worker
      |
      v
 durable source frontier (SQLite/WAL, priority + leases)
      |
      +--> source download pool --> downloaded queue
                                  |
                                  v
                    resident M2D scan worker --> scanned queue
                                                   |
                                                   v
                                  full-quality extraction pool
                                                   |
                                                   v
                                         existing raw clip queue
                                                   |
                                      M2D -> ASR -> assembly
```

Source ASR requests continue to use the already-loaded faster-whisper worker,
but they are initiated by the scan stage and no longer couple discovery or
source transfer to extraction.

## Durable state contract

The frontier database is stored at
`$SAM_CONTINUOUS_WORKSPACE/source-frontier.sqlite3` in WAL mode.

Each source has one row keyed by `(platform, video_id)` and moves through:

```text
discovered -> downloaded -> scanned -> complete
       \            \          \
        +-------------+-----------> rejected
```

Workers claim rows with an expiring lease inside `BEGIN IMMEDIATE`. A crashed
worker does not lose a source: another worker may reclaim it after expiry.
Transient failures return to the same stage with bounded exponential backoff.
Permanent media, format, stereo, M2D, or enforced source-ASR failures are
terminal and retain their reason.

Every transition appends a stage event with start/end time, duration, outcome,
and small diagnostic payload. Queue depth, oldest age, lease count, completion
rate, rejection rate, and stage duration percentiles are exposed in the
progress API.

## Invariants

1. Discovery is idempotent: a source is inserted once and may receive a better
   priority or fresher metadata without resetting completed work.
2. A source can have only one active lease.
3. Downloads are written to a temporary path and atomically renamed.
4. Scan JSON is written atomically and remains compatible with the existing
   `whole_source_proxy_m2d_v1` cache.
5. Full source/proxy files are removed after a durable scan result; queue limits
   bound disk use while scanning is slower than transfer.
6. Extraction uses the existing per-source claim and catalog-guidance rules, so
   restart or concurrent workers cannot duplicate a clip interval.
7. Extracted WAVs are published through the existing manifest promoter and are
   not visible downstream until quality checks and hashes are complete.
8. The old batch acquisition path remains an environment-controlled rollback
   option until the staged path passes its sustained comparison.

## Rollout and measurement gates

### Phase 0: baseline — complete

Record the 5/15/60-minute funnel, queue depths, CPU/GPU use, source-region
yield, source-transfer latency, and producer lifecycle. The values above are
the rollout baseline.

### Phase 1: durable frontier and shadow discovery

- Add the frontier schema, lease/retry primitives, metrics, and tests.
- Run discovery in shadow mode until at least 1,000 unique sources are queued.
- Do not change the production clip path.

Gate: no duplicate keys, lease recovery is proven, and discovery keeps the
configured high-water mark without unbounded growth.

### Phase 2: independent source transfer

- Start a source download pool against the frontier.
- Bound `downloaded` sources by count and bytes so the scanner applies
  backpressure without blocking discovery.
- Preserve high-quality format, channel, and sample-rate preflight checks.

Gate: measure at least 15 minutes; report sources/minute, Mbps, download p50/p95,
retry rate, queue depth, CPU, and disk use. Downstream production remains on the
old path.

### Phase 3: resident whole-source scanner

- Load M2D once for the lifetime of the scanner service.
- Convert downloaded media to the 16 kHz stereo proxy, validate stereo energy,
  scan it, and run the source-ASR probe.
- Persist the compatible scan cache and delete bulky source work files.

Gate: measure at least 15 minutes; the downloaded queue must drain or remain
bounded, M2D reloads must be zero during normal operation, and scan p95 plus
regions/source must be visible.

### Phase 4: independent full-quality extraction

- Consume only passing scans.
- Download selected HD sections concurrently and publish normalized clips into
  rotating acquisition manifests.
- Keep the old downstream promoter, clip M2D, ASR, and assembler unchanged.

Gate: canary the staged extractor while the old extractor is disabled. Verify
no duplicate candidate IDs or hashes and compare 15-minute accepted throughput
with the baseline.

### Phase 5: sustained production comparison

- Run staged acquisition for at least 60 minutes after queues reach steady
  state.
- Compare accepted h/h, raw h/h, final yield, source-stage idle fraction,
  queue-age percentiles, CPU/GPU use, and errors against Phase 0.
- Retain the staged path only if it improves sustained accepted throughput or
  materially reduces burst/idle time without reducing output quality.

Initial success target: at least 6 accepted audio h/h over 60 minutes (46%
above the 4.09 baseline) with no growing downstream queue. The longer-term
10-day target requires more source supply or hosts and is outside this single
host scheduling refactor.

## Rollback

`SAM_CONTINUOUS_STAGED_ACQUISITION=false` starts the previous batch producers.
The frontier and compatible scan cache may remain on disk. Rollback does not
alter the continuous catalog or remove already-published clips.

## Measurement log

| Phase | UTC interval | Result | Notes |
| --- | --- | --- | --- |
| 0 | 2026-07-14 18:49–19:49 | 4.09 accepted h/h | Empty downstream queues; source-yield limited |
| 1 | 2026-07-14 20:06–20:19 | passed | 1,000 unique sources, 0 duplicate keys/leases, SQLite integrity `ok`; shadow only |
| 2a | 2026-07-14 20:19–20:36 | failed, corrected | 172 events: 76 success, 25 terminal rejection, 71 retry; 4.08 GB transferred; p50 2.84 s, p95 24.09 s |
| 3 | 2026-07-14 20:26–20:44 | passed with cache migration | Resident model stayed loaded; fresh scans were normally 6–7 s end to end and the downloaded queue remained bounded |
| 4a | 2026-07-14 20:37–20:55 | failed supply gate | Atomic handoff succeeded and duplicate candidate/hash counts stayed zero, but only 6 raw / 3 accepted clips arrived while exhausted caches were drained |
| 4b | 2026-07-14 20:55–21:10 | passed 15-minute gate | 24.50 raw h/h, 13.80 accepted h/h, 56.1% yield; broader discovery produced 344 unseen sources in 7.6 s |
| 5 | from 2026-07-14 21:06 | measuring | Final restart-safe build held unchanged for the clean 60-minute comparison |

Phase 1 found two issues before production handoff. The first shadow process
inherited the sampler's 10-second module default; the disposable frontier was
reset and discovery now applies/restores an explicit 30-second value. CPython
3.14 also retained Dailymotion HTTP response descriptors across large search
batches. Discovery now has a persistent shell controller but runs each batch in
a short-lived child, so the OS closes every descriptor between seeds. Empty or
rate-limited seeds advance rather than poisoning the frontier. Finally,
capacity selection filters known source keys before truncating, allowing the
frontier to reach its exact source-count high-water mark.

Phase 2a exposed classification bugs rather than a transfer-capacity problem.
`yt-dlp` responses for unavailable formats, private media, and media with no
formats were being retried as transient failures. These are now terminal and
bounded stderr is retained for real network/CDN retries. The transfer p95
dropped from 24.09 to 13.46 seconds in the first post-fix sample; the clean
15-minute sample is recorded separately once complete.

Phase 3 first encountered 2,459 existing scan-cache transitions. Cache
bookkeeping is now reported separately from active work so zero-second cache
adoption cannot distort scan latency. Cached positives are accepted only when
the current enforced source-ASR probe passed; otherwise the source returns to
the scan/probe path. Cached negative and exhausted sources terminate without
model work.

Phase 4a showed that concurrency was no longer the immediate limit. The old
query space returned 6,932 candidates but zero usable unseen sources. Two
batch-era behaviors amplified the stall: non-empty search batches below 3,600
candidates were discarded, and discovery's restored 10-second module default
made valid 30-second caches look absent during catalog filtering. Clip duration
is now explicit, every non-empty discovery batch is useful, and catalog history
filters exhausted sources before transfer.

The durable discovery producer now runs 50-query slices with a 30-second
success interval and a 60-second failure cooldown. This avoids Dailymotion API
rate-limit oscillation. Query coverage includes gameplay, story mode, NPC and
mission dialogue, party banter, game movies, and episodic scenes. The first
post-change slice found 597 eligible source videos, 344 of them unseen, in 7.6
seconds. With supply restored, the live bottleneck moved to full-quality
section extraction; its autoscaled ceiling was raised from 8 to 16 while CPU
remained below the configured pressure threshold.

At the 15-minute production gate the new path delivered 735 raw clips and 412
accepted clips: 24.50 raw audio h/h and 13.80 accepted audio h/h. This is 3.37x
the 4.09 h/h baseline and exceeds the initial 6 h/h success target by 2.3x.
M2D, ASR, and assembly returned to zero pending items at the gate, and catalog
queries found zero duplicate candidate IDs, raw hashes, or accepted hashes.
The final restart-safety change lets stable worker IDs reclaim their leases and
explicitly expires claims left by an earlier process before autoscaler gating.
