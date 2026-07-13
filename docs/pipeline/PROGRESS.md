# Pipeline progress

Last updated: 2026-07-13

## Target

Build and deploy an AWS-backed, durable batch pipeline for long audio files with
music-first then voice separation, sound gates, persisted stems/metadata,
single-consumer model queues, operational visibility, and fast manual review.
Recursive separation of individual SFX is excluded from this milestone.

Datasets are persistent and accept any number of successive upload jobs; `100`
was an example batch size, not a workflow limit.

## Milestones

- [x] Validate AWS credentials and the H100 host connection.
- [x] Audit the existing SAM Audio API, timings, adaptive candidates, and
  three-bucket verification output.
- [x] Record the production flow, data ownership, queue semantics, and review
  contract in `ARCHITECTURE.md`.
- [x] Make the model cascade prompt-aware and deploy music-first defaults.
- [x] Add the lightweight pipeline control-plane package and tests.
- [x] Add CloudFormation for S3, DynamoDB, SQS/DLQs, and the EC2 role.
- [x] Implement durable ingestion, 30s/5s chunking, and the sound gate.
- [x] Implement the one-at-a-time SAM queue worker and stem persistence.
- [x] Implement Audio Flamingo queue tasks for scene/music descriptions and
  voice transcription with diarization.
- [x] Acquire and persist a reproducible AudioSet reference/calibration set.
- [x] Implement job/queue dashboard and keyboard-first review UI.
- [x] Add exact review backlog counts, dataset size/processing metrics, and the
  original-plus-stems source explorer.
- [x] Add frequency-aware stereo/loudness reconstruction while preserving every
  raw mono stem, plus a dashboard playback toggle and smoothed trajectory plots.
- [x] Require stereo input, gate SAM targets with source-scene presence, support
  single-stage inference, and use identity pass-through for pure SFX sources.
- [x] Add persistent datasets, successive upload jobs, and reconciliation of
  uploaded/stale work from DynamoDB back into SQS.
- [x] Provision AWS, deploy services, and run a multi-file end-to-end test.
- [x] Verify recovery/reconciliation, review decisions, and all stored artifacts.
- [x] Commit and push the verified implementation to `qforge/main`.

## Current evidence

- AWS caller: account `088543363904`, IAM user `michal`, region `us-east-1`.
- GPU host: `i-0aed4af178083ce58`, `p5.4xlarge`, H100 80 GB, reachable by SSH.
- Model services: `sam-audio-small-tv` in TF32/fp32 and Audio Flamingo Next in
  BF16 are both active on the H100. Their checkpoints live under
  `/home/ubuntu/models` and are not part of an image.
- SAM candidates expand 4 -> 8 -> 12. The default route is music-first; a failed
  cascade gets one voice-first retry and both attempts are persisted.
- AWS stack `sam-audio-pipeline` is live with a private/versioned/encrypted S3
  bucket, point-in-time-recoverable DynamoDB table, three SQS queues and DLQs,
  and a least-privilege instance profile attached to the H100 host.
- AudioSet calibration set `references/audioset/v1/` contains 8 private WAVs,
  `manifest.json`, and `manifest.sha256`.
- Seeded AudioSet validation set `references/audioset/random-100-v1/` contains
  100 private WAVs from 100 unique YouTube videos plus the manifest/checksum.
  All 100 records retain their official AudioSet start/end timestamps and
  labels; 28 unavailable candidates remain in the manifest as provenance. The
  successful clips total 996.795 seconds, and the 102 reference objects total
  191,551,793 bytes.
- Dataset `991945c1f082446a9ff66d482838f964`, job
  `3f0233f2442f4044a0f2967c68cb5a89`, completed all 100 sources and chunks with
  zero failed chunks. It initially stored 284 audible stems (94 music, 91 voice,
  99 SFX)
  and omitted 16 below-gate outputs (6 music, 9 voice, 1 SFX). The selected
  route was music-first for 63 sounds and voice-first for 37.
- After the presence repair, the completed random dataset contains 282 stems
  and 540,037,650 bytes of mapped stereo companions. All 282 companions were
  force-migrated to stereo v2 (or the identity path) without replacing their
  raw keys. Automatic verification places 159 stems in success, 54 in
  uncertain, and 69 in failure, leaving 123 review items.
- Browser verification confirms the review screen shows exact remaining,
  failure, and uncertain counts; the dataset screen shows 100 sounds and its
  storage breakdown; gated stems are absent from the stacked track view; the
  repaired writing source shows `sfx only` plus `identity · no EQ`; and the mono
  canary shows the explicit stereo-input skip reason.
- Live reference job `ae31001eec754ee6850e3727134de84e` processed the supplied
  30-second `chunk_ours/og.wav` through the normal production handler and
  created raw plus stereo music, voice, and SFX objects. The rendered dashboard
  defaults to Stereo mapped, swaps only stem URLs when Raw is selected, and
  displays three 32-band/EMA-0.03 trajectory plots.
- The mapper was separately checked against the previously supplied raw
  `humanvoices.wav`: its smoothed pan moves from `-0.33` (left) to `+0.39`
  (right), matching the attached chunk-zero analysis. All three 30-second stems
  map in about 0.58 seconds locally; the fresh remote job mapped its three stems
  in about 1.02 seconds.
- Source `099_writing_mHf7COLGcD0_170.wav` exposed both failure modes: its scene
  annotation correctly said no music and no voices, but the old cascade still
  created three `success` stems, and stereo v1 reduced music high-frequency
  energy from 8.7% to 0.8%. Stereo v2 retains 7.8% on the same raw stem. The live
  source was repaired from its stored scene evidence and now has one SFX-only
  stem; raw, mapped, and original normalized PCM are sample-identical. The
  dataset now indexes 282 stems (93 music, 90 voice, 99 SFX).
- Live mono-filter job `5ec26b1bb28b420a9245e21d475d26be` completed with
  `non_stereo_input`, one input channel, zero chunks, zero stems, and zero model
  tasks. A live `targets=voice` API canary produced only voice and residual
  artifacts in one 12.8-second SAM stage.
- Multi-file job `bd85486f9d314c258956cac9b81a730f` completed: one 30-second
  audible source produced exactly one chunk and three stems; one silent source
  was sound-gated without GPU work; all three Audio Flamingo tasks completed.
- That job triggered both cascade orders. Music-first scored `18.03` versus
  `8.11` for voice-first, so the policy retained music-first and stored both
  timing/score summaries.
- Recovery job `1d07979ee96a45a98cf220b102fd7006` was uploaded without the
  completion callback. Reconciliation found the S3 object, enqueued ingestion,
  and completed it as sound-gated. All three queues were empty afterward.
- Local and remote validation: 36 pipeline tests pass; the existing 12 SAM
  batching tests also pass.
- Docker build execution remains unverified because neither the local Mac nor
  the current H100 host has a Docker daemon. Both Dockerfiles remain model-free.

## Decisions and open deployment work

- DynamoDB is the job/stem/review source of truth; SQS is transport, not state.
- S3 stores all audio and metadata; the model image continues to exclude model
  weights.
- The control plane is a separate lightweight package/image from the SAM model
  server.
- The H100 instance uses the stack-managed least-privilege IAM instance profile;
  no long-lived AWS key is stored in the service environment.
- The initial panel remains private behind an SSH tunnel until authentication is
  explicitly configured.
