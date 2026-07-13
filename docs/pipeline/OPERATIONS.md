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
```

DynamoDB is authoritative. A successful HTTP upload confirmation only means the
source tasks are durable and queued; callers should poll the job endpoint rather
than keep an inference request open.

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
