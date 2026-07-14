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

## Continuous 30-second cinematic dataset

`sam-cinematic-continuous.service` runs a permanent producer/consumer graph.
Acquisition publishes quality-gated files atomically; resident M2D and ASR
workers consume independent deterministic shards; the assembler writes the
append-only accepted catalog; and the publisher releases an exhaustive,
immutable S3 snapshot at every 5,000 accepted clips. Manual review is diagnostic
feedback and never blocks acceptance.

The SQLite/WAL catalog is authoritative; the live JSON manifest is intentionally
constant-size. Review queries the newest 5,000 accepted records directly from
the catalog so the page stays responsive while the global count grows into the
millions. Each S3 snapshot contains only its new 5,000-record sequence range
(for example `v2-00000001-00005000`), avoiding quadratic cumulative manifests.

The live dataset is under `/home/ubuntu/cinematic-continuous-30s`. Review it at
`http://127.0.0.1:18081/` and inspect queues, worker health, rolling throughput,
the 10,000-hour ETA, and S3 releases at
`http://127.0.0.1:18081/progress`.

Pool sizes are independent settings in `/etc/sam-cinematic-continuous.env`:

```text
SAM_CONTINUOUS_SEARCH_WORKERS=8
SAM_CONTINUOUS_DOWNLOAD_WORKERS=8
SAM_CONTINUOUS_M2D_WORKERS=1
SAM_CONTINUOUS_ASR_WORKERS=1
SAM_CONTINUOUS_UPLOAD_CONCURRENCY=10
```

Restart `sam-cinematic-continuous.service` after changing a count. Filename hash
sharding prevents two M2D or ASR processes from claiming the same clip, and the
SQLite/WAL catalog makes replay and worker-count changes idempotent. Promoter,
assembler, and snapshot publisher remain single lightweight coordinators.

Throughput uses a rolling 60-minute window. `audio min/min` is clip duration
processed per wall-clock minute and has the same numeric value as audio
hours/wall-clock hour. The 10,000-hour estimate uses accepted throughput, not
download or model throughput, and is intentionally unavailable until at least
one clip has passed every gate.

The frozen 10-second baseline is published at:

```text
s3://sam-audio-pipeline-artifactbucket-ndb3jfc3asyk/cinematic-dialogue-dataset/snapshots/v1-00001000/
```

Audio is content-addressed under `cinematic-dialogue-dataset/audio/{sha256}.wav`.
Snapshot metadata is published first and `READY.json` last; consumers must
require the ready marker and validate `manifest.sha256`.

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

### Cinematic mixed-audio acquisition

The cinematic profile is based on the completed 216-clip human-review findings
in [`HUMAN_REVIEW_FINDINGS_20260714.md`](HUMAN_REVIEW_FINDINGS_20260714.md).
It targets raw movie/TV/animated scenes, game cutscenes, short films, and
produced news packages while rejecting review/reaction/vlog/tutorial/AI-voice
metadata. It may retain up to three non-overlapping excerpts from one promising
source:

```bash
uv run sam-pipeline-youtube-random \
  --output /data/cinematic-raw-20260715 \
  --total 3500 \
  --seed 20260715 \
  --profile cinematic \
  --clips-per-video 3 \
  --query-count 1600 \
  --results-per-query 15 \
  --candidate-multiplier 1.7 \
  --max-attempts 6000
```

If YouTube rejects the downloader host's public address, use the same gated
acquisition pipeline against Dailymotion rather than weakening the selection
policy or using account cookies:

```bash
uv run sam-pipeline-youtube-random \
  --output /data/cinematic-raw-20260715 \
  --source dailymotion \
  --total 3500 \
  --seed 20260715 \
  --profile cinematic \
  --clips-per-video 12 \
  --query-count 500 \
  --results-per-query 100 \
  --candidate-multiplier 2 \
  --max-attempts 9000
```

The Dailymotion path uses the public search API for metadata and `yt-dlp` only
for the selected time sections. The same title, uploader, description, and tag
exclusions apply, including explicit India/Indian and Indian-language source
metadata. It does not infer a speaker's ethnicity or nationality from audio.

Score with `--require-cinematic-mix` so dialogue, music, and non-music SFX are
independent requirements. Materialize with the same flag and
`--accepted-limit 1000` to create an exact 1,000-record result. The final
materialized set retains source URLs, exact timestamps, query provenance, and
the human-feedback-derived policy version.

### M2D dialogue/background validation

The technical YouTube gate does not prove that spoken dialogue and background
activity occur in the selected ten seconds. Validate candidates with the
official M2D AudioSet-fine-tuned tagger before using them. The validator scores
overlapping two-second windows and requires speech, a non-human background
class, and at least three windows in which both are active. Singing, choir,
chant, rapping, humming, opera, a capella, vocal music, and song evidence may
appear in at most one window, preventing sung vocals from satisfying the speech
requirement. It separately records music-led, effects/ambience-led, and mixed
instrumental backgrounds.

The model repository, checkpoint, AudioSet label CSV, and ontology are runtime
inputs rather than files baked into this repository or its Docker image:

```bash
uv run sam-pipeline-m2d-validate score \
  --input-dir /data/youtube-random-1000/audio \
  --output /data/youtube-random-1000/m2d-validation.jsonl \
  --m2d-repo /models/m2d \
  --checkpoint /models/m2d/weights_ep69it3124-0.47929.pth \
  --class-labels /models/audioset/class_labels_indices.csv \
  --ontology /models/audioset/ontology.json \
  --m2d-commit 3d0c4de9447c404a8d3f9f37e04f53bc902e09b3
```

Create a separate listening-test folder containing only accepted audio. The
command hard-links files when possible and never removes the source dataset:

```bash
uv run sam-pipeline-m2d-validate materialize \
  --input-dir ~/Downloads/youtube-random-1000-20260714/audio \
  --results ~/Downloads/m2d-validation.jsonl \
  --source-manifest ~/Downloads/youtube-random-1000-20260714/manifest.json \
  --output-dir ~/Downloads/youtube-dialogue-background-m2d-ok-20260714
```

The output includes `manifest.json`, `audit.json`, the complete M2D JSONL, and
the accepted WAVs under `audio/`. `balanced-audio/` is a deterministic listening
subset with equal music-led and non-music-led counts. Every record keeps the
per-window scores, ranks, top labels, temporal coverage, rejection reasons,
exact M2D checkpoint hash, and validator policy version.

Policy v5 requires audible voice evidence in at least five 2-second windows:
the M2D speech-family probability must be at least `0.10` and rank within the
top five labels in each counted window. This strong gate is separate from the
older low-confidence speech diagnostic and rejects clips tagged as speech only
because of weak background evidence. Materializing an older M2D JSONL applies
the current voice gate from its stored per-window scores without rerunning the
model.

M2D also records foreground-speech subclass evidence, but it is diagnostic: on
the cinematic pilot those sublabels were too sparse to use as a mandatory
gate. Confirm audible foreground voice with faster-whisper instead:

```bash
PYTHONPATH=/app/pipeline/src /models/whisper-venv/bin/python \
  -m sam_audio_pipeline.m2d_validator asr-score \
  --input-dir /data/cinematic-raw-20260715/audio \
  --output /data/cinematic-raw-20260715/asr-validation.jsonl \
  --model small \
  --download-root /models/faster-whisper
```

The ASR policy requires at least 1.5 seconds of VAD-positive audio, at least two
decoded words, best segment average log probability of `-0.80` or better, and
no-speech probability no higher than `0.50`. Against all 216 completed reviews,
it rejected 13 of 14 clips explicitly marked `lacking_voice` while retaining 86
of 110 Good/Perfect clips. The M2D strong speech check remains required as an
independent guard.

The validator detects the spoken language rather than forcing an English
decode. Final clips must be detected as English with probability at least
`0.80`. This is a language-content filter; it does not infer a speaker's
nationality or ethnicity from their voice.

The same M2D pass rejects synthetic narration when Speech Synthesizer scores at
least `0.20` within the top five labels in two or more windows. This threshold
comes directly from the reviewed `ai voice` failure, where the class scored
`0.24`–`0.72` in all nine windows.

Pass both validation files when creating the exact final set:

```bash
uv run sam-pipeline-m2d-validate materialize \
  --input-dir /data/cinematic-raw-20260715/audio \
  --results /data/cinematic-raw-20260715/m2d-validation.jsonl \
  --asr-results /data/cinematic-raw-20260715/asr-validation.jsonl \
  --source-manifest /data/cinematic-raw-20260715/manifest.json \
  --output-dir /data/cinematic-final-1000-20260715 \
  --require-cinematic-mix \
  --accepted-limit 1000
```

### Manual listening review

Start the local review app on the balanced M2D subset:

```bash
cd pipeline
uv run sam-pipeline-review \
  --dataset-dir ~/Downloads/youtube-dialogue-background-m2d-ok-20260714 \
  --audio-directory balanced-audio \
  --port 18081
```

Open `http://127.0.0.1:18081/`. Each browser first asks for a reviewer name,
then receives a random unreviewed clip. Each clip can be marked **Good**, **Perfect**,
or **Not OK**. Not OK supports multiple reasons: lacking music, lacking
background audio/SFX, singing or vocal music, speech that is not dialogue, low
quality, low volume, distortion/clipping, wrong voice/background balance, or an
Other reason with a required note. Keyboard shortcuts are shown in the app.
Press `X` to open Not OK, `1`–`9`/`0` to toggle its rejection reasons, `Enter`
to save, or `Esc` to cancel. This keeps multi-reason tagging keyboard-only.

Progress is written atomically after every decision to
`manual-review.json` inside the dataset directory, so closing or refreshing the
browser does not lose work. Each clip also has a refreshable `/clip/{filename}`
URL. `Export CSV` downloads a flat table suitable for analysis and later policy
tuning. Use `--annotations /some/path.json` to store annotations elsewhere, or
`--audio-directory audio` to review all M2D-accepted clips instead of the
balanced subset.

The app supports multiple reviewers against one annotation file. Assignment is
an atomic random lease: the server never gives a live clip to two reviewers,
rejects stale or conflicting submissions with HTTP 409, renews an open clip in
the browser, and releases an abandoned clip after 10 minutes. Saving or skipping
immediately assigns another random clip. Reviewer identity and attribution are
included in JSON and CSV exports. Change the lease with `--claim-seconds`.

For a shared server, install `deploy/pipeline/sam-pipeline-review.service`, set
`SAM_REVIEW_DATASET_DIR` in `/etc/sam-audio-review.env`, and install the Nginx
template after replacing `__REVIEW_PATH__` with a random path. Keep the Python
service bound to `127.0.0.1`; only Nginx should be public. The frontend derives
its API and audio URLs from the Nginx path prefix, so refreshable clip links also
work through the shared URL.

If the EC2 security group cannot accept port 80, run the optional
`sam-audio-review-tunnel.service`. It exposes the Nginx origin through an
outbound Cloudflare Quick Tunnel without changing inbound firewall rules. Quick
Tunnel hostnames are temporary and can change when the tunnel restarts; use a
named Cloudflare Tunnel and DNS hostname for a permanent production URL.

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
