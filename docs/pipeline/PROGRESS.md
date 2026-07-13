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
- Multi-file job `bd85486f9d314c258956cac9b81a730f` completed: one 30-second
  audible source produced exactly one chunk and three stems; one silent source
  was sound-gated without GPU work; all three Audio Flamingo tasks completed.
- That job triggered both cascade orders. Music-first scored `18.03` versus
  `8.11` for voice-first, so the policy retained music-first and stored both
  timing/score summaries.
- Recovery job `1d07979ee96a45a98cf220b102fd7006` was uploaded without the
  completion callback. Reconciliation found the S3 object, enqueued ingestion,
  and completed it as sound-gated. All three queues were empty afterward.
- Remote validation: 13 pipeline tests and 12 SAM batching tests pass.
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
