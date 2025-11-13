"""Optimized multi-threaded batch processor for retrieval and reranking."""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from src.utils import logger

T = TypeVar("T")
R = TypeVar("R")


class BatchProcessor:
    """Multi-threaded batch processor with prefetching and resource optimization."""

    def __init__(
        self,
        num_workers: int = 4,
        prefetch_batches: int = 2,
        max_queue_size: int = 10,
    ) -> None:
        self.num_workers = max(1, num_workers)
        self.prefetch_batches = max(1, prefetch_batches)
        self.max_queue_size = max(2, max_queue_size)
        self._executor: Optional[ThreadPoolExecutor] = None

    def __enter__(self) -> BatchProcessor:
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix="batch_worker")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._executor:
            self._executor.shutdown(wait=True)

    def process_batches(
        self,
        batches: List[List[T]],
        process_fn: Callable[[List[T]], R],
        batch_timeout: float = 300.0,
    ) -> List[R]:
        """Process batches in parallel with prefetching."""
        if not batches:
            return []

        if self._executor is None:
            raise RuntimeError("BatchProcessor must be used as context manager")

        results: List[Optional[R]] = [None] * len(batches)
        futures: Dict[Any, int] = {}

        for batch_idx, batch in enumerate(batches):
            future = self._executor.submit(process_fn, batch)
            futures[future] = batch_idx

        for future in as_completed(futures, timeout=batch_timeout * len(batches)):
            batch_idx = futures[future]
            try:
                results[batch_idx] = future.result(timeout=batch_timeout)
            except Exception as exc:
                logger.error(f"Batch {batch_idx} processing failed: {exc}")
                results[batch_idx] = None

        return [r for r in results if r is not None]


class PrefetchingBatchProcessor:
    """Batch processor with prefetching queue for optimal GPU utilization."""

    def __init__(
        self,
        process_fn: Callable[[List[T]], R],
        batch_size: int = 64,
        num_workers: int = 2,
        prefetch_batches: int = 2,
    ) -> None:
        self.process_fn = process_fn
        self.batch_size = max(1, batch_size)
        self.num_workers = max(1, num_workers)
        self.prefetch_batches = max(1, prefetch_batches)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._queue: Optional[queue.Queue] = None
        self._stop_event = threading.Event()

    def __enter__(self) -> PrefetchingBatchProcessor:
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix="prefetch_worker")
        self._queue = queue.Queue(maxsize=self.prefetch_batches * 2)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._stop_event.set()
        if self._executor:
            self._executor.shutdown(wait=True)

    def process_stream(
        self,
        items: List[T],
        timeout: float = 300.0,
    ) -> List[R]:
        """Process items in batches with prefetching."""
        if not items:
            return []

        if self._executor is None or self._queue is None:
            raise RuntimeError("PrefetchingBatchProcessor must be used as context manager")

        total_batches = (len(items) + self.batch_size - 1) // self.batch_size
        results: List[R] = []
        futures: List[Any] = []

        def prefetch_worker():
            for batch_start in range(0, len(items), self.batch_size):
                if self._stop_event.is_set():
                    break
                batch = items[batch_start : batch_start + self.batch_size]
                future = self._executor.submit(self.process_fn, batch)
                try:
                    self._queue.put((batch_start // self.batch_size, future), timeout=timeout)
                except queue.Full:
                    logger.warning("Prefetch queue full, waiting...")
                    self._queue.put((batch_start // self.batch_size, future), timeout=timeout * 2)

        prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        prefetch_thread.start()

        batch_results: Dict[int, R] = {}
        completed = 0

        while completed < total_batches:
            try:
                batch_idx, future = self._queue.get(timeout=timeout)
                try:
                    result = future.result(timeout=timeout)
                    batch_results[batch_idx] = result
                    completed += 1
                except Exception as exc:
                    logger.error(f"Batch {batch_idx} failed: {exc}")
                    completed += 1
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                logger.warning("Prefetch queue timeout, continuing...")

        prefetch_thread.join(timeout=5.0)

        for idx in sorted(batch_results.keys()):
            results.append(batch_results[idx])

        return results


def optimize_batch_size(
    base_batch_size: int,
    num_items: int,
    min_batch: int = 8,
    max_batch: int = 256,
) -> int:
    """Optimize batch size based on number of items and hardware."""
    if num_items <= min_batch:
        return min_batch

    optimal = base_batch_size
    while optimal < num_items and optimal < max_batch:
        if num_items % optimal == 0:
            break
        optimal += 1

    return min(max(min_batch, optimal), max_batch, num_items)


def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """Split list into chunks of specified size."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

