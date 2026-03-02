"""Free-threaded Python (3.13+) helpers.

Runtime detection of GIL status with fallback. When running on Python 3.13+
with free-threading enabled, allows true parallel execution of:
- Tokenization + vision processing
- Transform chain (sentence_divider, emotion_extractor) + next token generation

On GIL-enabled Python, falls back to sequential execution.
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger(__name__)


def is_free_threaded() -> bool:
    """Check if Python is running with free-threading (no GIL)."""
    if hasattr(sys, "_is_gil_enabled"):
        return not sys._is_gil_enabled()
    return False


# Global flag, checked once at import time
FREE_THREADED = is_free_threaded()


class ParallelExecutor:
    """Execute CPU-bound tasks in parallel when free-threading is available.

    On GIL-enabled Python, tasks run sequentially (no benefit from threads
    for CPU-bound work, and avoids thread overhead).
    """

    def __init__(self, max_workers: int = 4) -> None:
        self._use_threads = FREE_THREADED
        self._pool: ThreadPoolExecutor | None = None
        if self._use_threads:
            self._pool = ThreadPoolExecutor(max_workers=max_workers)
            logger.info(
                "ParallelExecutor using %d threads (free-threaded Python)",
                max_workers,
            )
        else:
            logger.info("ParallelExecutor using sequential execution (GIL enabled)")

    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> Future:
        """Submit a task for execution.

        Returns a Future that resolves to the result.
        In sequential mode, the task is executed immediately.
        """
        if self._pool is not None:
            return self._pool.submit(fn, *args, **kwargs)

        # Sequential fallback: execute immediately, wrap in a resolved Future
        future: Future = Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future

    def map(self, fn: Callable, *iterables) -> list:
        """Map a function over iterables in parallel.

        Returns results in order.
        """
        if self._pool is not None:
            return list(self._pool.map(fn, *iterables))
        return [fn(*args) for args in zip(*iterables)]

    def run_parallel(self, *tasks: Callable[[], Any]) -> list:
        """Run multiple zero-argument callables in parallel.

        Returns results in task order.
        """
        if self._pool is not None:
            futures = [self._pool.submit(task) for task in tasks]
            return [f.result() for f in futures]
        return [task() for task in tasks]

    def shutdown(self) -> None:
        """Shutdown the thread pool."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
