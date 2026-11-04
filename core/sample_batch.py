"""Worker-pool tasks that cover more than one sample: Parquet shards and task batches.

``process_parquet_shard`` runs every row of one ``ShardJob`` through the
image or diffusion processor. ``batch_worker`` executes a list of
``(callable, *args)`` task tuples in one pool call.
"""

import logging
import random
from typing import Any

import pandas as pd

from core.error_types import PARQUET_READ_ERRORS, SAMPLE_PROCESSING_ERRORS
from core.sample_diffusion import process_diffusion
from core.sample_jobs import ImageJob, ShardJob
from core.sample_processor import process_image

logger = logging.getLogger(__name__)


def _status_record(name: str, source: str, task: str, status: str) -> dict[str, Any]:
    """Registry record for a row that produced no sample (dropped, skipped or failed)."""
    return {"name": name, "source": source, "task": task, "split": "skipped",
            "hash": status, "nima_score": 1.0, "size": 0}


def _process_shard_row(job: ShardJob, offset: int, row: Any,
                       columns: Any) -> dict[str, Any]:
    """Process one Parquet row; a status record stands in for rows without a sample."""
    key = job.key.at(offset)
    name = key.name
    task = job.context.task

    if job.keep_prob < 1.0 and random.random() > job.keep_prob:
        return _status_record(name, key.slug, task, "dropped")
    if name in job.existing:
        return _status_record(name, key.slug, task, "skipped")
    img_bytes = getattr(row, "image", getattr(row, "pixel_values", None))
    if img_bytes is None:
        return _status_record(name, key.slug, task, "skipped")

    split = "train" if random.random() < job.train_prob else "val"
    if task == "diffusion":
        res = process_diffusion(img_bytes, key.prefix, key.slug, key.idx, split)
    else:
        row_dict = {k: getattr(row, k) for k in columns}
        res = process_image(ImageJob(img_bytes, key, job.context, row_dict, split))
    return res or _status_record(name, key.slug, task, "failed")


def process_parquet_shard(job: ShardJob) -> list[dict[str, Any] | None]:
    """Process an entire Parquet shard as one worker task.

    An unreadable shard yields one ``None`` per planned row: progress still
    advances by ``num_rows``, nothing is registered, and the next run retries
    the shard.
    """
    try:
        df = pd.read_parquet(job.pq_path)
    except PARQUET_READ_ERRORS as e:
        logger.warning("Skipping corrupted virtual parquet shard %s: %s", job.pq_path, e)
        return [None] * job.num_rows
    return [_process_shard_row(job, offset, row, df.columns)
            for offset, row in enumerate(df.itertuples())]


def batch_worker(tasks: list[tuple[Any, ...]]) -> list[Any]:
    """Execute a list of (func, *args) in one worker call to amortize IPC."""
    results: list[Any] = []
    for task_func, *task_args in tasks:
        try:
            res = task_func(*task_args)
            if task_func.__name__ == "process_parquet_shard":
                results.extend(res)
            else:
                results.append(res)
        except SAMPLE_PROCESSING_ERRORS as exc:
            logger.warning(
                "Worker task failed: %s with args %s: %s",
                getattr(task_func, "__name__", str(task_func)),
                task_args,
                exc,
            )
            results.append(None)
    return results
