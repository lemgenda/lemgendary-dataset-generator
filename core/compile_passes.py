"""The compile passes of one sample-based manifold.

- PASS 1 runs the planned tasks (a serial warm-up, then the worker pool)
  and writes every sample record into the registry.
- Style clustering groups the CLIP latents with MiniBatchKMeans.
- PASS 2 writes ``index.json`` (and WebDataset shards for diffusion) and
  any extra container formats requested with ``--also-format``.
"""

import argparse
import json
import logging
import os
import pickle
import random
import sqlite3
from concurrent.futures import (
    FIRST_COMPLETED,
    BrokenExecutor,
    CancelledError,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from webdataset.writer import TarWriter as WdsTarWriter

from core.compile_registry import insert_rows, registry_row
from core.compile_tasks import TaskItem, count_one
from core.compiler_core import CONFIG, batch_worker
from formats.base import parse_also_format
from tools.migrate_manifold_format import migrate_manifold

logger = logging.getLogger(__name__)

_WARMUP_SAMPLES = 500
_WARMUP_COMMIT_EVERY = 100
_BYTES_PER_GB = 1024**3


# ─── PASS 1: extraction ─────────────────────────────────────────────────────
@dataclass
class ExtractionRun:
    """The pool, registry and limits PASS 1 of one manifold runs with."""

    conn: sqlite3.Connection
    executor: ThreadPoolExecutor
    max_workers: int
    max_gb: float
    no_hash: bool
    batch_size: int


@dataclass
class _PoolState:
    """Mutable bookkeeping of the parallel phase."""

    pending: set[Future] = field(default_factory=set)
    seen_hashes: set[Any] = field(default_factory=set)
    compiled_bytes: int = 0


def _warmup(run: ExtractionRun, all_tasks: list[TaskItem], pbar: Any) -> int:
    """2026 Resilience: SAFE-START WARMUP (SOTA v6.3) - run the first tasks serially."""
    warmup_limit = min(_WARMUP_SAMPLES, len(all_tasks))
    if warmup_limit <= 0:
        return warmup_limit
    print(
        f"[SAFE-START] Warming up manifold (Serial Pass: {warmup_limit} samples)..."
    )
    for i in range(warmup_limit):
        task_args = all_tasks[i]
        res = task_args[0](*task_args[1:])
        pbar.update(1)
        if res:
            if not isinstance(res, list):
                res = [res]
            batch_entries = [registry_row(r) for r in res if r]
            if batch_entries:
                insert_rows(run.conn, batch_entries)
        if i % _WARMUP_COMMIT_EVERY == _WARMUP_COMMIT_EVERY - 1:
            run.conn.commit()
    print("[SAFE-START] Warmup complete. Engaging Parallel Matrix.")
    return warmup_limit


def _absorb(run: ExtractionRun, state: _PoolState, batch_results: list[Any],
            batch_entries: list[tuple[Any, ...]]) -> None:
    """Dedup a finished batch into ``batch_entries``; cancel pending work past ``max_gb``."""
    for res in batch_results:
        if not res:
            continue
        if CONFIG["enable_dedup"] and not run.no_hash and res["hash"] in state.seen_hashes:
            continue
        if res["hash"]:
            state.seen_hashes.add(res["hash"])
        state.compiled_bytes += res.get("size", 0)
        batch_entries.append(registry_row(res))
        if (state.compiled_bytes / _BYTES_PER_GB) >= run.max_gb:
            for f in state.pending:
                f.cancel()
            state.pending.clear()
            break


def _submit_next(run: ExtractionRun, state: _PoolState, batch_iter: Any, count: int) -> None:
    """Submit up to ``count`` more batches to the pool."""
    for _ in range(count):
        try:
            batch = next(batch_iter)
            state.pending.add(run.executor.submit(batch_worker, batch))
        except StopIteration:
            break


def _run_pool(run: ExtractionRun, state: _PoolState, remaining_tasks: list[TaskItem],
              pbar: Any) -> None:
    """Feed the remaining tasks to the pool in batches and register the results."""
    task_batches = [
        remaining_tasks[i : i + run.batch_size]
        for i in range(0, len(remaining_tasks), run.batch_size)
    ]
    batch_iter = iter(task_batches)
    _submit_next(run, state, batch_iter, min(run.max_workers * 4, len(task_batches)))

    while state.pending:
        done, state.pending = wait(state.pending, return_when=FIRST_COMPLETED)
        batch_entries: list[tuple[Any, ...]] = []
        for future in done:
            try:
                batch_results = future.result()
                pbar.update(len(batch_results))
                _absorb(run, state, batch_results, batch_entries)
            except (BrokenExecutor, CancelledError, pickle.PicklingError) as e:
                logger.error("Worker batch failed: %s", e)

        if batch_entries:
            insert_rows(run.conn, batch_entries)
            run.conn.commit()

        _submit_next(run, state, batch_iter, len(done))


def run_extraction(run: ExtractionRun, all_tasks: list[TaskItem], existing_count: int,
                   desc_label: str) -> int:
    """PASS 1: process every task and register the records; return the compiled bytes."""
    state = _PoolState()
    total_items_to_process = sum(count_one(t) for t in all_tasks)
    with tqdm(
        total=total_items_to_process + existing_count,
        initial=existing_count,
        desc=desc_label,
        smoothing=0.1,
    ) as pbar:
        warmup_limit = _warmup(run, all_tasks, pbar)
        try:
            _run_pool(run, state, all_tasks[warmup_limit:], pbar)
        except KeyboardInterrupt:
            os._exit(1)
    return state.compiled_bytes


# ─── Style clustering ───────────────────────────────────────────────────────
def cluster_styles(conn: sqlite3.Connection) -> None:
    """STEP 2: Style Clustering of the CLIP latents into ``cluster_id``."""
    print("[STYLING] Commencing Style Clustering on all extracted latents...")
    cursor = conn.execute("SELECT id, clip_latent FROM registry WHERE clip_latent IS NOT NULL")
    ids, latents = [], []
    for row in cursor:
        _lat = np.frombuffer(row[1], dtype=np.float32)
        if len(_lat) > 0:
            ids.append(row[0])
            latents.append(_lat)

    if not (latents and len(latents) > 0 and len(latents[0]) > 0):
        print("[STYLING] No valid style latents found. Skipping clustering (Pure Human Mode).")
        return
    latent_matrix = np.stack(latents)
    n_clusters = int(CONFIG.get("n_style_clusters", 16))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit(latent_matrix)
    labels_list = [int(x) for x in np.asarray(kmeans.labels_, dtype=np.int64)]
    for i, cid in tqdm(
        zip(ids, labels_list), total=len(ids), desc="[STYLING] Updating Clusters"
    ):
        conn.execute("UPDATE registry SET cluster_id = ? WHERE id = ?", (cid, i))
    conn.commit()


# ─── PASS 2: index and shards ───────────────────────────────────────────────
def _index_record(row: Any) -> dict[str, Any]:
    """The ``index.json`` entry of one registry row."""
    return {
        "id": row[0],
        "name": row[1],
        "source": row[2],
        "task": row[3],
        "split": row[4],
        "hash": row[5],
        "nima_score": row[6],
        "caption": row[7],
        "style_tag": row[8],
        "cluster_id": row[11],
    }


def _write_shard_sample(sink: Any, res: dict[str, Any], img_bytes: Any) -> None:
    """Write one diffusion sample into its source's WebDataset shard."""
    sink.write(
        {
            "__key__": res["name"],
            "jpg": img_bytes,
            "txt": res["caption"],
            "json": json.dumps(
                {
                    "style": res["style_tag"],
                    "cluster": res["cluster_id"],
                    "source": res["source"],
                }
            ),
        }
    )


def _index_source(conn: sqlite3.Connection, source: str, sink: Any) -> list[dict[str, Any]]:
    """Index entries of one source, writing its diffusion samples into ``sink``."""
    cursor = conn.execute(
        "SELECT * FROM registry WHERE source = ? ORDER BY cluster_id, id", (source,)
    )
    entries = []
    for row in cursor.fetchall():
        res = _index_record(row)
        if res["task"] == "diffusion" and row[10] and sink:
            _write_shard_sample(sink, res, row[10])
        entries.append(res)
    return entries


def write_index(conn: sqlite3.Connection, output_root: Path,
                affixes: tuple[str, str]) -> list[dict[str, Any]]:
    """PASS 2: Balanced Interleaving & Sharding per Dataset; return the shuffled index."""
    print("[SHARD] Commencing PASS 2: Multi-Domain Balanced Sharding...")
    prefix_str, suffix_str = affixes

    shard_dir = None
    has_diffusion = (
        conn.execute("SELECT 1 FROM registry WHERE task = 'diffusion' LIMIT 1").fetchone()
        is not None
    )
    if has_diffusion:
        shard_dir = output_root / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)

    unique_sources = [
        r[0] for r in conn.execute("SELECT DISTINCT source FROM registry").fetchall()
    ]

    final_index = []
    for source in unique_sources:
        sink = None
        if has_diffusion and shard_dir is not None:
            shard_name = f"{prefix_str}{source}{suffix_str}.tar"
            print(f"[SHARD] Writing {shard_name}...")
            sink = WdsTarWriter(str(shard_dir / shard_name))
        final_index.extend(_index_source(conn, source, sink))
        if sink:
            sink.close()

    random.seed(42)
    random.shuffle(final_index)
    with open(output_root / "index.json", "w", encoding="utf-8") as f:
        json.dump(final_index, f, indent=2)
    return final_index


def write_containers(args: argparse.Namespace, output_root: Path) -> tuple[str, ...]:
    """Write the ``--also-format`` containers; return the packed ones written successfully."""
    if not getattr(args, "also_format", None):
        return ()
    try:
        extra_formats = parse_also_format(args.also_format)
        if not extra_formats:
            return ()
        print(
            "[CONTAINER] Writing additional container format(s):"
            f" {', '.join(extra_formats)}..."
        )
        container_rc = migrate_manifold(
            root=output_root,
            formats=extra_formats,
            force_duplicate=getattr(args, "force_duplicate", False),
            accept_space_loss=getattr(args, "accept_space_loss", False),
        )
        if container_rc == 0:
            return tuple(f for f in extra_formats if f != "directory")
        print(
            f"[ERROR] Container writer exited with code {container_rc}; loose"
            " layout kept."
        )
    except (ImportError, OSError, ValueError) as e:
        logger.error("Failed writing additional container format: %s", e)
    return ()
