# Pipeline operations

## Live deployment

- AWS region: `us-east-1`
- CloudFormation stack: `sam-audio-pipeline`
- GPU instance: `i-0aed4af178083ce58` (`p5.4xlarge`, H100 80 GB)
- Host: `ec2-44-211-31-38.compute-1.amazonaws.com`
- Private artifact bucket: `sam-audio-pipeline-artifactbucket-ndb3jfc3asyk`
- DynamoDB table: `sam-audio-pipeline-PipelineTable-1C6R2H8D5C4TW`
- Remote application root: `/home/ubuntu/sam-audio-deploy`

The AWS names are identifiers, not credentials. The EC2 instance profile grants
only the S3, DynamoDB, and SQS operations declared in CloudFormation.

## Private access

The panel/API binds to remote localhost port 8080. Open a local tunnel:

```bash
ssh -fN -o ExitOnForwardFailure=yes \
  -L 18080:127.0.0.1:8080 \
  ubuntu@ec2-44-211-31-38.compute-1.amazonaws.com
```

Then open `http://127.0.0.1:18080/`. The model APIs remain private on remote
ports 8000 (SAM) and 8001 (Audio Flamingo).

## Services

```bash
ssh ubuntu@ec2-44-211-31-38.compute-1.amazonaws.com \
  'systemctl is-active sam-audio-api audio-flamingo-next \
   sam-pipeline-api sam-pipeline-ingest sam-pipeline-sam \
   sam-pipeline-flamingo'
```

The reconciliation timer repairs durable tasks whose SQS message was lost or
expired. It can also be invoked safely on demand:

```bash
ssh ubuntu@ec2-44-211-31-38.compute-1.amazonaws.com \
  'sudo systemctl start sam-pipeline-reconcile.service'
```

## Queue and job health

```bash
curl -fsS http://127.0.0.1:18080/v1/overview | jq
curl -fsS http://127.0.0.1:18080/v1/jobs/JOB_ID | jq
curl -fsS http://127.0.0.1:18080/v1/review/stats | jq
curl -fsS http://127.0.0.1:18080/v1/datasets/DATASET_ID/overview | jq
```

DynamoDB is authoritative. A successful HTTP upload confirmation only means the
source tasks are durable and queued; callers should poll the job endpoint rather
than keep an inference request open.

## Stereo mapping

The pipeline accepts exactly two-channel stereo sources. Mono and multichannel
uploads remain visible in the dataset but finish with `skip_reason` set to
`non_stereo_input`; they create no chunks, stems, or model tasks.

For accepted sources, Audio Flamingo scene analysis runs before SAM. Its
`has_music` and `has_voices` booleans select zero, one, or both SAM targets. A
malformed scene response deliberately falls back to both targets. Judge/CLAP
scores remain separation-quality evidence and are not used as presence scores.

New audible stems automatically keep both versions:

- `{stem}.wav` is the untouched mono model output used by review and existing
  integrations.
- `{stem}.stereo.wav` is the frequency-aware stereo/loudness reconstruction.
- The stem record and chunk metadata contain the mapping settings and
  pan/loudness curve; a backfill also writes `{chunk}.stereo.json`.

In **Split data**, select a source and use **Raw / Stereo mapped** above the
players. Stereo mapped is selected by default; Raw remains available. The
original player never changes. The maps below the players show the
smoothed trajectory; left is below the center line and right is above it.
Version 2 applies frequency-specific panning but only broadband gain, preventing
the mapper from making stems bass-heavy. A source with neither music nor voices
is stored as a single SFX stem; Raw and Stereo mapped are PCM-identical and the
map is labelled `identity · no EQ`.

Backfill a completed job after deploying the mapper. The command is idempotent
and skips stems that already have a mapped object unless `--force` is supplied:

```bash
ssh ubuntu@ec2-44-211-31-38.compute-1.amazonaws.com \
  "sudo bash -c 'set -a; source /etc/sam-audio-pipeline.env; set +a; \
   cd /home/ubuntu/sam-audio-deploy/pipeline; \
   .venv/bin/sam-pipeline-stereo-backfill --job-id JOB_ID'"
```

Use `--max-chunks N` for a canary. Mapping is CPU-only and does not require a
SAM or Audio Flamingo restart.

## Joined reconstruction and similarity

After stereo mapping and the output sound gate, the pipeline sums only the
stored stereo variants into `{chunk}.joined.stereo.wav`. After all chunks for a
record are terminal, it places those joins back on the source timeline and
overlap-averages them into `source.joined.stereo.wav`. The primary dataset score
compares that stored source PCM directly with the normalized stereo original,
without gain fitting or time alignment:

`100 × max(0, 2 × dot(original, joined) / (energy(original) + energy(joined)))`

This makes 100 an exact reconstruction; missing-output silence against an
audible original, opposite polarity, or another non-positive match scores 0.
The source record also retains waveform correlation,
left/right scores, level delta, normalized RMSE, SNR, coverage, and final limiter
gain. Chunk records retain the same diagnostics plus the included stem types so
a low source score can be localized. In **Split data**, the dataset histogram
shows the source-score distribution, the source detail plays Original beside
Joined, and each chunk exposes its normalized original and diagnostic join.

Backfill completed jobs from existing stereo companions without rerunning SAM or
Audio Flamingo:

```bash
ssh ubuntu@ec2-44-211-31-38.compute-1.amazonaws.com \
  "sudo bash -c 'set -a; source /etc/sam-audio-pipeline.env; set +a; \
   cd /home/ubuntu/sam-audio-deploy/pipeline; \
   .venv/bin/sam-pipeline-reconstruction-backfill --job-id JOB_ID'"
```

The command is idempotent; use `--force` to regenerate an existing joined
artifact or `--max-chunks N` for a canary.

## Original audio profiles

Ingestion probes the first audio stream before the stereo-only gate and stores
the original channel count/layout, codec and container, sample rate, bit depth
when the codec exposes it, bitrate, and lossless/lossy quality tier. **Split
data** shows these facts in both the source list and source header, including for
mono inputs that were skipped before model work.

Backfill existing sources directly from their durable originals:

```bash
ssh ubuntu@ec2-44-211-31-38.compute-1.amazonaws.com \
  "sudo bash -c 'set -a; source /etc/sam-audio-pipeline.env; set +a; \
   cd /home/ubuntu/sam-audio-deploy/pipeline; \
   .venv/bin/sam-pipeline-audio-profile-backfill --all'"
```

The command is idempotent; pass one or more `--job-id JOB_ID` arguments instead
of `--all`, or use `--force` to refresh already-profiled sources.

## Dashboard URLs

Dashboard state is encoded in clean paths and is safe to refresh or share:

```text
/                                      pipeline and upload overview
/review                                manual review queue
/data/{dataset_id}                     dataset explorer
/data/{dataset_id}/jobs/{job_id}       job within a dataset
/data/{dataset_id}/jobs/{job_id}/sources/{source_id}
                                       exact sound detail
```

Selecting a tab, job, dataset, or sound updates browser history. Back and
forward navigation restore the corresponding view and selected record.

## General YouTube random datasets

This sampler searches YouTube directly and does not read AudioSet manifests or
timestamps. Randomized, reproducible queries favor program audio containing
voices, music, and environmental effects while excluding obvious music-only,
sleep, ambience, and playlist results. Each video ID contributes at most one
clip.

```bash
cd pipeline
uv run sam-pipeline-youtube-random \
  --output ~/Downloads/youtube-random-1000-20260714 \
  --total 1000 \
  --seed 20260714 \
  --query-count 400 \
  --results-per-query 15 \
  --search-workers 8 \
  --download-workers 8 \
  --candidate-multiplier 2 \
  --max-attempts 2500
```

Acquisition is resumable. `attempts.jsonl` records every accepted, rejected, or
unavailable candidate; `metadata/candidates.json` and `metadata/search.json`
preserve search provenance; `manifest.json` contains only accepted records and
hashes. `audit.json` is written only after reopening and verifying the complete
dataset. The gate requires source audio at 44.1 kHz or better and at least 120
kbps, then verifies exact ten-second PCM16/48 kHz stereo output, real left/right
difference, loudness, silent-frame/run limits, clipping, uniqueness, and SHA-256.

Re-run the same command to continue an interrupted build. Audit an existing set
without downloading:

```bash
uv run sam-pipeline-youtube-random \
  --output ~/Downloads/youtube-random-1000-20260714 \
  --total 1000 --verify-only
```

## AudioSet validation batches

Acquire a reproducible random sample from the official AudioSet segment CSVs.
Unavailable YouTube videos are retained as failed provenance entries and
replacement candidates are tried until the requested successful count exists:

```bash
cd pipeline
uv run sam-pipeline-audioset \
  --output ~/Downloads/audioset-random-100 \
  --total 100 \
  --seed 20260713 \
  --s3-bucket sam-audio-pipeline-artifactbucket-ndb3jfc3asyk \
  --s3-prefix references/audioset/random-100-v1
```

Submit every successful manifest entry as one persistent dataset job while
retaining its AudioSet labels and YouTube timestamps:

```bash
uv run sam-pipeline-submit \
  --api http://127.0.0.1:18080 \
  --manifest ~/Downloads/audioset-random-100/manifest.json \
  --dataset-name "AudioSet random 100 · seed 20260713"
```

The verified seeded run is available in the panel as dataset
`991945c1f082446a9ff66d482838f964`; its completed job is
`3f0233f2442f4044a0f2967c68cb5a89`. The private reference prefix contains the
100 timestamped WAVs plus `manifest.json` and `manifest.sha256`.

## Review controls

The panel autoplays the next `uncertain`/`failure` stem. Keys are:

- `1`: assertion correct / pass
- `2`: assertion incorrect / fail
- `3`: leave pending for later verification
- `Space`: play or pause

Human decisions append review history and change `effective_status`; they never
erase `automatic_status`, Judge/CLAP evidence, timings, or adaptive-route data.

## Deployment safety

The repository and both Docker build contexts exclude model checkpoints. Models
are downloaded separately to `/home/ubuntu/models` and referenced at runtime.
When syncing systemd launchers, preserve executable modes. Stop only the SAM
queue consumer while restarting the SAM model API; queued messages remain safe
in SQS and task state remains safe in DynamoDB.
