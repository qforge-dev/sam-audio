#!/usr/bin/env bash
set -euo pipefail

STACK_NAME=${SAM_PIPELINE_STACK_NAME:-sam-audio-pipeline}
AWS_REGION=${AWS_REGION:-us-east-1}
INSTANCE_ID=${SAM_PIPELINE_INSTANCE_ID:-i-0aed4af178083ce58}

aws cloudformation deploy \
  --region "$AWS_REGION" \
  --stack-name "$STACK_NAME" \
  --template-file "$(dirname "$0")/cloudformation.yaml" \
  --parameter-overrides PipelineInstanceId="$INSTANCE_ID" \
  --capabilities CAPABILITY_IAM \
  --no-fail-on-empty-changeset

PROFILE_NAME=$(aws cloudformation describe-stacks \
  --region "$AWS_REGION" \
  --stack-name "$STACK_NAME" \
  --query 'Stacks[0].Outputs[?OutputKey==`InstanceProfileName`].OutputValue | [0]' \
  --output text)

ASSOCIATION_ID=$(aws ec2 describe-iam-instance-profile-associations \
  --region "$AWS_REGION" \
  --filters Name=instance-id,Values="$INSTANCE_ID" \
  --query 'IamInstanceProfileAssociations[?State!=`disassociated`].AssociationId | [0]' \
  --output text)

if [[ -z "$ASSOCIATION_ID" || "$ASSOCIATION_ID" == "None" ]]; then
  aws ec2 associate-iam-instance-profile \
    --region "$AWS_REGION" \
    --instance-id "$INSTANCE_ID" \
    --iam-instance-profile Name="$PROFILE_NAME" >/dev/null
else
  aws ec2 replace-iam-instance-profile-association \
    --region "$AWS_REGION" \
    --association-id "$ASSOCIATION_ID" \
    --iam-instance-profile Name="$PROFILE_NAME" >/dev/null
fi

aws cloudformation describe-stacks \
  --region "$AWS_REGION" \
  --stack-name "$STACK_NAME" \
  --query 'Stacks[0].Outputs[].{Key:OutputKey,Value:OutputValue}' \
  --output table
