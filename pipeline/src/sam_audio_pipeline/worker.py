"""Long-poll one task at a time from a selected model/CPU queue."""

from __future__ import annotations

import argparse
import logging
import signal
import threading
from collections.abc import Callable

from .aws import PipelineAWS, ReceivedTask
from .config import Settings
from .handlers import IngestHandler, Reconciler, SeparationHandler
from .schema import QueueTask

logger = logging.getLogger(__name__)


class VisibilityHeartbeat:
    def __init__(
        self,
        aws: PipelineAWS,
        queue_url: str,
        receipt_handle: str,
        *,
        interval_seconds: int = 300,
        visibility_seconds: int = 900,
    ):
        self.aws = aws
        self.queue_url = queue_url
        self.receipt_handle = receipt_handle
        self.interval_seconds = interval_seconds
        self.visibility_seconds = visibility_seconds
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> VisibilityHeartbeat:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stopped.set()
        self.thread.join(timeout=5)

    def _run(self) -> None:
        while not self.stopped.wait(self.interval_seconds):
            self.aws.extend_visibility(
                self.queue_url,
                self.receipt_handle,
                self.visibility_seconds,
            )


def run_worker(
    aws: PipelineAWS,
    queue_url: str,
    handler: Callable[[QueueTask], None],
) -> None:
    stopped = threading.Event()

    def stop(*_: object) -> None:
        stopped.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    logger.info("Worker ready; queue=%s concurrency=1", queue_url)
    while not stopped.is_set():
        received: ReceivedTask | None = aws.receive_task(queue_url)
        if received is None:
            continue
        logger.info(
            "Processing task=%s type=%s receive_count=%d",
            received.task.task_id,
            received.task.task_type,
            received.receive_count,
        )
        try:
            with VisibilityHeartbeat(aws, queue_url, received.receipt_handle):
                handler(received.task)
        except Exception:
            logger.exception("Task failed and will return to the queue")
            continue
        aws.delete_task(queue_url, received.receipt_handle)
        logger.info("Completed task=%s", received.task.task_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", choices=("ingest", "sam", "flamingo", "reconcile"), required=True
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings.from_env()
    aws = PipelineAWS(settings)
    if args.role == "reconcile":
        recovered = Reconciler(settings, aws).run_once()
        logger.info("Reconciliation complete: %s", recovered)
        return
    if args.role == "ingest":
        handler = IngestHandler(settings, aws).handle
        queue_url = settings.ingest_queue_url
    elif args.role == "sam":
        handler = SeparationHandler(settings, aws).handle
        queue_url = settings.sam_queue_url
    else:
        raise SystemExit(
            "Audio Flamingo worker is not deployed yet; refusing to consume its "
            "durable queue."
        )
    run_worker(aws, queue_url, handler)


if __name__ == "__main__":
    main()
