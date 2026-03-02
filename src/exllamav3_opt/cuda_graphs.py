"""CUDA graph capture for single-token decode steps.

Captures model.forward(batch=1, seq=1) as a CUDA graph for replay.
Static inputs (input_ids, block_table, cache_seqlens) are updated in-place
before each replay. Re-captures when block_table shape changes (page boundary).

Caveats:
- FlashInfer paged attention may not be graph-capturable; test early.
- Sampling is done OUTSIDE the graph (data-dependent control flow).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Number of warmup iterations before capturing
_WARMUP_ITERS = 3
# Re-capture interval (when crossing page boundary)
_RECAPTURE_INTERVAL = 256


class CUDAGraphRunner:
    """Manages CUDA graph capture and replay for decode steps.

    Usage:
        runner = CUDAGraphRunner(model, pool)
        runner.warmup(cache, embeddings)  # Must call once before first use

        # In decode loop:
        if runner.is_ready:
            logits = runner.replay(token_id, kv_position)
        else:
            logits = model.forward(...)  # fallback to eager
    """

    def __init__(self, model, pool) -> None:
        self.model = model
        self.pool = pool
        self.device = model.device

        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_logits: torch.Tensor | None = None
        self._captured_num_pages: int = -1
        self._is_ready = False
        self._supported = True  # Set to False if capture fails

    @property
    def is_ready(self) -> bool:
        return self._is_ready and self._supported

    def warmup(self, cache, num_pages: int) -> None:
        """Run warmup iterations and capture the CUDA graph.

        Args:
            cache: Cache object for attention.
            num_pages: Current number of pages needed.
        """
        if not self._supported:
            return

        try:
            self._warmup_and_capture(cache, num_pages)
        except Exception as e:
            logger.warning("CUDA graph capture failed: %s — falling back to eager", e)
            self._supported = False

    def _warmup_and_capture(self, cache, num_pages: int) -> None:
        """Warmup + capture implementation."""
        use_mrope = "mrope" in self.model.caps

        block_table = self.pool.get_block_table(num_pages)

        params = {
            "attn_mode": "flashinfer",
            "block_table": block_table,
            "cache": cache,
            "cache_seqlens": self.pool.cache_seqlens,
        }
        if use_mrope:
            params["positions"] = self.pool.cache_seqlens.clone()

        # Warmup iterations (must run on same stream)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            for _ in range(_WARMUP_ITERS):
                _ = self.model.forward(input_ids=self.pool.input_ids, params=params)

        torch.cuda.current_stream().wait_stream(s)

        # Capture graph
        self._graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self._graph, stream=s):
            self._static_logits = self.model.forward(
                input_ids=self.pool.input_ids, params=params
            )

        self._captured_num_pages = num_pages
        self._is_ready = True
        logger.debug("CUDA graph captured for %d pages", num_pages)

    def needs_recapture(self, kv_position: int) -> bool:
        """Check if we need to re-capture (page boundary crossed)."""
        from exllamav3.constants import PAGE_SIZE

        num_pages = (kv_position + 1 + PAGE_SIZE - 1) // PAGE_SIZE
        return num_pages != self._captured_num_pages

    def replay(self, token_id: int, kv_position: int) -> torch.Tensor:
        """Replay the captured CUDA graph.

        Updates static inputs in-place, replays, returns logits.
        """
        from exllamav3.constants import PAGE_SIZE

        # Update static inputs in-place (they're the same tensors used during capture)
        self.pool.set_input_id(token_id)
        self.pool.set_cache_seqlen(kv_position)

        # Replay
        self._graph.replay()

        return self._static_logits

    def recapture(self, cache, kv_position: int) -> None:
        """Re-capture graph for new page count."""
        from exllamav3.constants import PAGE_SIZE

        if not self._supported:
            return

        num_pages = (kv_position + 1 + PAGE_SIZE - 1) // PAGE_SIZE
        self._is_ready = False

        try:
            self._warmup_and_capture(cache, num_pages)
        except Exception as e:
            logger.warning("CUDA graph re-capture failed: %s", e)
            self._supported = False

    def reset(self) -> None:
        """Clear captured graph."""
        self._graph = None
        self._static_logits = None
        self._captured_num_pages = -1
        self._is_ready = False
