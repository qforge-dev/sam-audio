# SAM Audio stem pipeline architecture

## Scope

The first production slice accepts one or many long audio files, processes them
as durable jobs, and stores reviewable music, voice, and SFX stems. Recursive
separation of individual sound effects is deliberately out of scope.

The default separation order is **music first, then voices**. The second stage
receives the first stage residual. A failed or uncertain stage does not cancel
later work; every produced stem keeps its own automatic verification state. If
the complete music-first cascade fails, the worker retries **voices first, then
music** and retains the route with the stronger final/stage status and Judge
quality evidence. Ties retain the default order. Both attempts and the selection
rule are persisted under `adaptive_routing`.

## Flow

1. The control API appends an upload job to a persistent dataset and returns S3 upload URLs. API requests finish
   after durable acceptance; they do not stay open while inference is queued.
2. An ingestion worker normalizes each source with FFmpeg and creates 30-second
   chunks with 5-second overlap. The final short chunk is retained.
3. A sound gate measures peak and RMS level. Empty or extremely quiet chunks are
   marked `skipped` without consuming GPU time.
4. Each audible chunk is sent to the single SAM Audio queue. One GPU worker
   receives one message at a time and calls the local `sam-audio-small-tv` API.
5. SAM Audio separates `music soundtrack` from the original chunk, then
   `human voices` from the music residual. Candidate expansion and Judge/CLAP
   evidence determine `success`, `uncertain`, or `failure` independently for
   each stage. A failed complete cascade triggers the bounded reverse-order
   attempt described above; it never recursively separates the SFX residual.
6. Music, voice, and final residual (SFX) WAVs, inference metadata, timings, and
   review assertions are stored in S3 and indexed in DynamoDB.
7. Description and transcription tasks share one Audio Flamingo queue so that a
   single loaded model processes one item at a time. It creates one source-scene
   analysis per audible source, one description per music stem, and one
   speaker-labelled transcript per voice stem. Tasks exist in DynamoDB before
   SQS delivery and are reconciled after worker/message loss.
8. A browser dashboard shows job/queue progress and exact remaining review
   counts. Its reviewer mode pulls the next `uncertain` or `failure` assertion,
   starts audio automatically, displays the prompt/assertion, and accepts
   one-key decisions.
9. The dataset explorer reports source count, duration, input/chunk/stem storage,
   processing counts, stem mix, and review backlog. Each source displays the
   original recording first, then only the music/voice/SFX stems that passed the
   output sound gate, grouped by chunk, followed by its processing route and
   model annotations.

## AWS resources

- One private, versioned, encrypted S3 bucket for sources, chunks, stems,
  metadata, and review artifacts.
- One DynamoDB table using `PK`/`SK`, plus a review-status index. It is the
  authoritative state store for jobs, chunks, stems, tasks, and review history.
- Three standard SQS queues with dead-letter queues:
  - CPU ingestion/orchestration
  - SAM Audio GPU inference
  - Audio Flamingo description/transcription
- One least-privilege EC2 instance role for the pipeline host.

SQS retention is finite (AWS caps it), so DynamoDB is authoritative. A
reconciler can re-enqueue unfinished database tasks, giving jobs durable,
no-request-timeout semantics even if a message expires or a worker restarts.

There is no fixed dataset-size or file-count limit in the application contract.
Clients may append new upload jobs to an existing dataset or create another
dataset. Each HTTP batch is finite for practical upload handling, while the
dataset and its processing history remain durable.

## Concurrency and backpressure

Queue depth is not constrained by the HTTP server. The SAM queue and Audio
Flamingo queue each have exactly one active consumer and one model task in
flight. CPU chunking and uploads may run concurrently, but the GPU model service
continues to use `max_batch_size=1`.

## Storage keys

Objects use stable, content-address-aware prefixes:

```text
jobs/{job_id}/sources/{source_id}/{filename}
jobs/{job_id}/chunks/{source_id}/{chunk_index}.wav
jobs/{job_id}/stems/{source_id}/{chunk_index}/{music|voice|sfx}.wav
jobs/{job_id}/metadata/{source_id}/{chunk_index}.json
```

Source and stem records include SHA-256 hashes. A stem record includes the
source chunk hash, prompt, model/settings fingerprint, automated status,
reviewer status, S3 version/key, timings, and raw Judge/CLAP evidence.

## Review contract

Automatic state is never overwritten by a person. Review records append a
human decision (`pass`, `fail`, or `pending`) with timestamp and optional note.
The effective bucket is derived from the latest human decision when present,
otherwise from the automatic `success`, `uncertain`, or `failure` state.

Reviewer assertions are concrete and binary:

- music target: "This should contain only music."
- music residual: "This should contain no music."
- voice target: "This should contain only human voices."
- voice residual / SFX: "This should contain no music and no voices."

## AudioSet reference set

Reference examples are selected from the official AudioSet ontology and
balanced/evaluation segment CSVs. AudioSet identifies 10-second regions in
YouTube videos; it does not distribute a raw-audio archive. The acquisition tool
therefore stores a reproducible manifest (AudioSet MID, video ID, time range,
source URL, retrieval result, SHA-256, and license/provenance fields) and fetches
available clips only at deployment time. Raw clips stay in the private pipeline
S3 bucket and are never committed to this repository.

The initial calibration set covers at least Music (`/m/04rlf`), Speech
(`/m/09x0r`), Silence (`/m/028v0c`), and representative ambience/noise classes.
These references support score calibration and regression tests; they do not
replace per-output Judge/CLAP evidence or human review.

The deployed `references/audioset/v1/` set contains eight 10-second WAVs
(music, speech, silence, dog, engine, door, explosion, and water), a provenance
manifest, and a manifest checksum. AudioSet metadata is CC BY 4.0 and its
ontology is CC BY-SA 4.0; the underlying YouTube media retains its source rights
and therefore remains private rather than being redistributed with the code.

The acquisition CLI also supports a seeded random cross-ontology sample with
`--total` and `--seed`. It shuffles official evaluation/balanced metadata,
downloads the published YouTube time ranges, records successful and unavailable
source IDs, hashes every retrieved WAV, and can upload the private set and
manifest to S3. The companion manifest submitter retains video ID, labels,
source start/end timestamps, licenses, and checksums on each pipeline source
record.

## Model licensing

SAM Audio remains subject to this repository's SAM license. The Audio Flamingo
code repository is MIT-licensed, while the deployed Audio Flamingo checkpoint is
under NVIDIA's non-commercial model license. A production/commercial deployment
must resolve that checkpoint license separately; keeping weights out of the
images does not change the model license.

## Security and exposure

All S3 objects and databases remain private. The initial web service binds to
the GPU host and is accessed through an SSH tunnel. Public exposure requires an
authenticated reverse proxy or load balancer and is not enabled implicitly.
