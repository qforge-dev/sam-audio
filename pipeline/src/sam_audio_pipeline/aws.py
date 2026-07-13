"""Small DynamoDB, S3, and SQS adapters used by the API and workers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from .config import Settings
from .schema import QueueTask, ReviewDecision, StemRecord, utc_now


def _ddb(value: Any) -> Any:
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {key: _ddb(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_ddb(item) for item in value]
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class ReceivedTask:
    task: QueueTask
    receipt_handle: str
    receive_count: int


class PipelineAWS:
    def __init__(self, settings: Settings):
        session = boto3.Session(region_name=settings.aws_region)
        self.settings = settings
        self.s3 = session.client("s3")
        self.sqs = session.client("sqs")
        self.table = session.resource("dynamodb").Table(settings.table_name)

    def put(self, item: dict[str, Any]) -> None:
        self.table.put_item(Item=_ddb(item))

    def get(self, pk: str, sk: str) -> dict[str, Any] | None:
        item = self.table.get_item(Key={"PK": pk, "SK": sk}).get("Item")
        return _plain(item) if item else None

    def update(
        self,
        pk: str,
        sk: str,
        values: dict[str, Any],
        *,
        condition: str | None = None,
    ) -> None:
        names = {f"#n{index}": name for index, name in enumerate(values)}
        data = {
            f":v{index}": _ddb(value) for index, value in enumerate(values.values())
        }
        expression = "SET " + ", ".join(
            f"{name_key} = :v{index}" for index, name_key in enumerate(names)
        )
        kwargs: dict[str, Any] = {
            "Key": {"PK": pk, "SK": sk},
            "UpdateExpression": expression,
            "ExpressionAttributeNames": names,
            "ExpressionAttributeValues": data,
        }
        if condition:
            kwargs["ConditionExpression"] = condition
        self.table.update_item(**kwargs)

    def query_partition(self, pk: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {"KeyConditionExpression": Key("PK").eq(pk)}
        while True:
            response = self.table.query(**kwargs)
            items.extend(_plain(item) for item in response.get("Items", []))
            if "LastEvaluatedKey" not in response:
                return items
            kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

    def query_index(
        self, index_pk: str, *, limit: int = 50, newest_first: bool = False
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {
            "IndexName": "GSI1",
            "KeyConditionExpression": Key("GSI1PK").eq(index_pk),
            "ScanIndexForward": not newest_first,
        }
        while len(items) < limit:
            kwargs["Limit"] = limit - len(items)
            response = self.table.query(**kwargs)
            items.extend(_plain(item) for item in response.get("Items", []))
            if "LastEvaluatedKey" not in response:
                break
            kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
        return items

    def ensure_dataset(self, dataset_id: str, name: str) -> dict[str, Any]:
        existing = self.get(f"DATASET#{dataset_id}", "META")
        if existing:
            return existing
        now = utc_now()
        item = {
            "PK": f"DATASET#{dataset_id}",
            "SK": "META",
            "entity": "dataset",
            "dataset_id": dataset_id,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "GSI1PK": "DATASETS",
            "GSI1SK": f"{now}#{dataset_id}",
        }
        self.put(item)
        return item

    def create_job(self, job_id: str, dataset_id: str, source_count: int) -> None:
        now = utc_now()
        self.put(
            {
                "PK": f"JOB#{job_id}",
                "SK": "META",
                "entity": "job",
                "job_id": job_id,
                "dataset_id": dataset_id,
                "status": "uploading",
                "source_count": source_count,
                "completed_sources": 0,
                "created_at": now,
                "updated_at": now,
                "GSI1PK": "JOBS",
                "GSI1SK": f"{now}#{job_id}",
            }
        )
        self.put(
            {
                "PK": f"DATASET#{dataset_id}",
                "SK": f"JOB#{now}#{job_id}",
                "entity": "dataset_job",
                "dataset_id": dataset_id,
                "job_id": job_id,
                "status": "uploading",
                "source_count": source_count,
                "created_at": now,
            }
        )

    def create_source(
        self, job_id: str, source_id: str, filename: str, s3_key: str
    ) -> None:
        self.put(
            {
                "PK": f"JOB#{job_id}",
                "SK": f"SOURCE#{source_id}",
                "entity": "source",
                "job_id": job_id,
                "source_id": source_id,
                "filename": filename,
                "s3_key": s3_key,
                "status": "uploading",
                "created_at": utc_now(),
            }
        )

    def put_stem(self, stem: StemRecord) -> str:
        review_id = f"{stem.job_id}:{stem.source_id}:{stem.chunk_id}:{stem.stem_type}"
        item = stem.model_dump(mode="json")
        item.update(
            {
                "PK": f"JOB#{stem.job_id}",
                "SK": f"STEM#{stem.source_id}#{stem.chunk_id}#{stem.stem_type}",
                "entity": "stem",
                "review_id": review_id,
            }
        )
        if stem.automatic_status != "success":
            item.update(
                {
                    "GSI1PK": f"REVIEW#{stem.automatic_status}",
                    "GSI1SK": f"{stem.created_at}#{review_id}",
                }
            )
        self.put(item)
        return review_id

    def record_review(
        self,
        stem_item: dict[str, Any],
        decision: ReviewDecision,
        note: str,
        reviewer: str,
    ) -> None:
        now = utc_now()
        review_id = stem_item["review_id"]
        self.put(
            {
                "PK": f"REVIEW#{review_id}",
                "SK": f"DECISION#{now}",
                "entity": "review_decision",
                "review_id": review_id,
                "decision": decision,
                "note": note,
                "reviewer": reviewer,
                "created_at": now,
            }
        )
        effective = {
            ReviewDecision.PASS: "success",
            ReviewDecision.FAIL: "failure",
            ReviewDecision.PENDING: "uncertain",
        }[decision]
        values: dict[str, Any] = {
            "human_decision": decision,
            "effective_status": effective,
            "reviewed_at": now,
        }
        if decision == ReviewDecision.PENDING:
            values["GSI1PK"] = "REVIEW#uncertain"
            values["GSI1SK"] = f"{now}#{review_id}"
        else:
            values["GSI1PK"] = "REVIEWED"
            values["GSI1SK"] = f"{now}#{review_id}"
        self.update(stem_item["PK"], stem_item["SK"], values)

    def presign_upload(self, key: str, content_type: str) -> str:
        return self.s3.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": self.settings.bucket,
                "Key": key,
                "ContentType": content_type,
            },
            ExpiresIn=self.settings.presign_seconds,
        )

    def presign_download(self, key: str, expires: int = 3600) -> str:
        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.settings.bucket, "Key": key},
            ExpiresIn=expires,
        )

    def object_exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.settings.bucket, Key=key)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
                return False
            raise
        return True

    def upload_file(self, path: Path, key: str, content_type: str) -> None:
        self.s3.upload_file(
            str(path),
            self.settings.bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )

    def download_file(self, key: str, path: Path) -> None:
        self.s3.download_file(self.settings.bucket, key, str(path))

    def upload_json(self, value: dict[str, Any], key: str) -> None:
        self.s3.put_object(
            Bucket=self.settings.bucket,
            Key=key,
            Body=(json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
            ContentType="application/json",
        )

    def send_task(self, queue_url: str, task: QueueTask) -> None:
        self.sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=task.model_dump_json(),
        )

    def receive_task(
        self, queue_url: str, *, visibility_timeout: int = 900
    ) -> ReceivedTask | None:
        response = self.sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            VisibilityTimeout=visibility_timeout,
            AttributeNames=["ApproximateReceiveCount"],
        )
        messages = response.get("Messages", [])
        if not messages:
            return None
        message = messages[0]
        return ReceivedTask(
            task=QueueTask.model_validate_json(message["Body"]),
            receipt_handle=message["ReceiptHandle"],
            receive_count=int(
                message.get("Attributes", {}).get("ApproximateReceiveCount", 1)
            ),
        )

    def extend_visibility(
        self, queue_url: str, receipt_handle: str, seconds: int = 900
    ) -> None:
        self.sqs.change_message_visibility(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle,
            VisibilityTimeout=seconds,
        )

    def delete_task(self, queue_url: str, receipt_handle: str) -> None:
        self.sqs.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle,
        )

    def queue_metrics(self, queue_url: str) -> dict[str, int]:
        response = self.sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=[
                "ApproximateNumberOfMessages",
                "ApproximateNumberOfMessagesNotVisible",
                "ApproximateNumberOfMessagesDelayed",
            ],
        )
        attributes = response.get("Attributes", {})
        return {
            "queued": int(attributes.get("ApproximateNumberOfMessages", 0)),
            "in_flight": int(
                attributes.get("ApproximateNumberOfMessagesNotVisible", 0)
            ),
            "delayed": int(attributes.get("ApproximateNumberOfMessagesDelayed", 0)),
        }
