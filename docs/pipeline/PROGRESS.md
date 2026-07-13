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
- [ ] Make the model cascade prompt-aware and deploy music-first defaults.
- [x] Add the lightweight pipeline control-plane package and tests.
- [x] Add CloudFormation for S3, DynamoDB, SQS/DLQs, and the EC2 role.
- [x] Implement durable ingestion, 30s/5s chunking, and the sound gate.
- [x] Implement the one-at-a-time SAM queue worker and stem persistence.
- [ ] Implement Audio Flamingo queue tasks for scene/music descriptions and
  voice transcription with diarization.
- [ ] Acquire and persist a reproducible AudioSet reference/calibration set.
- [x] Implement job/queue dashboard and keyboard-first review UI.
- [x] Add persistent datasets, successive upload jobs, and reconciliation of
  uploaded/stale work from DynamoDB back into SQS.
- [ ] Provision AWS, deploy services, and run a multi-file end-to-end test.
- [ ] Verify recovery/reconciliation, review decisions, and all stored artifacts.
- [ ] Merge and push the verified implementation to `qforge/main`.

## Current evidence

- AWS caller: account `088543363904`, IAM user `michal`, region `us-east-1`.
- GPU host: `i-0aed4af178083ce58`, `p5.4xlarge`, H100 80 GB, reachable by SSH.
- Model service: active on the GPU host, `sam-audio-small-tv`, TF32/fp32,
  candidates 4 -> 8 -> 12, one model batch at a time.
- Current mismatch: deployed cascade is voice-first. The target contract is
  music-first, then voices; this must be changed and verified before pipeline use.

## Decisions and open deployment work

- DynamoDB is the job/stem/review source of truth; SQS is transport, not state.
- S3 stores all audio and metadata; the model image continues to exclude model
  weights.
- The control plane is a separate lightweight package/image from the SAM model
  server.
- The existing H100 instance has no IAM instance profile. The infrastructure
  milestone will create and attach a least-privilege profile before workers use
  AWS resources.
- The initial panel remains private behind an SSH tunnel until authentication is
  explicitly configured.
