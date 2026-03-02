"""System prompt KV cache.

After the first prefill of a system prompt, snapshots the KV cache pages
to CPU pinned memory. Subsequent generations with the same prefix skip
the prefill for the cached portion and restore from the CPU snapshot.

Cost: ~151 MB CPU RAM for 36 layers x 2048 tokens (negligible).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from exllamav3.cache.cache import Cache

logger = logging.getLogger(__name__)


class PrefixCache:
    """CPU-side snapshot of KV cache pages for a fixed system prompt prefix."""

    def __init__(self) -> None:
        self._cached_ids: torch.Tensor | None = None  # (1, prefix_len) on CPU
        self._cached_len: int = 0
        self._snapshots: dict[int, tuple[torch.Tensor, ...]] = {}  # layer_idx -> tensors

    @property
    def is_captured(self) -> bool:
        return self._cached_len > 0

    @torch.no_grad()
    def capture(
        self,
        input_ids: torch.Tensor,
        cache: Cache,
        kv_position: int,
    ) -> None:
        """Snapshot KV cache to CPU pinned memory after first prefill.

        Args:
            input_ids: Full input token IDs (1, seq_len).
            cache: The Cache object with populated KV data.
            kv_position: Number of tokens stored in cache (= seq_len after prefill).
        """
        self._cached_ids = input_ids.cpu().clone()
        self._cached_len = kv_position
        self._snapshots.clear()

        # Snapshot each cache layer's tensors to CPU pinned memory
        for layer_idx, layer in cache.layers.items():
            gpu_tensors = layer.get_tensors()
            cpu_copies = []
            for t in gpu_tensors:
                # Pinned memory for async transfer back to GPU
                cpu_t = torch.empty_like(t, device="cpu").pin_memory()
                cpu_t.copy_(t, non_blocking=True)
                cpu_copies.append(cpu_t)
            self._snapshots[layer_idx] = tuple(cpu_copies)

        # Wait for all copies to complete
        if gpu_tensors:
            torch.cuda.synchronize(gpu_tensors[0].device)

        logger.info(
            "Prefix cache captured: %d tokens, %d layers, %.1f MB",
            kv_position,
            len(self._snapshots),
            sum(
                sum(t.nbytes for t in tensors)
                for tensors in self._snapshots.values()
            )
            / 1024
            / 1024,
        )

    def get_cached_length(self, input_ids: torch.Tensor) -> int:
        """Return number of tokens that match the cached prefix.

        Args:
            input_ids: Current input token IDs (1, seq_len).

        Returns:
            Number of matching prefix tokens (0 if no match).
        """
        if not self.is_captured or self._cached_ids is None:
            return 0

        current = input_ids.cpu()
        cached = self._cached_ids

        # Compare up to the cached length
        compare_len = min(current.shape[-1], cached.shape[-1], self._cached_len)
        if compare_len == 0:
            return 0

        current_prefix = current[0, :compare_len]
        cached_prefix = cached[0, :compare_len]

        if torch.equal(current_prefix, cached_prefix):
            return compare_len
        return 0

    @torch.no_grad()
    def restore_to_cache(self, cache: Cache) -> None:
        """Copy KV snapshot from CPU back to GPU cache pages.

        Uses torch.no_grad() to allow in-place copy_ even when called
        inside torch.inference_mode() (which model.forward() enables).

        Args:
            cache: The Cache object to restore into.
        """
        if not self.is_captured:
            return

        for layer_idx, cpu_tensors in self._snapshots.items():
            gpu_tensors = cache.layers[layer_idx].get_tensors()
            for gpu_t, cpu_t in zip(gpu_tensors, cpu_tensors):
                gpu_t.copy_(cpu_t, non_blocking=True)

    def invalidate(self) -> None:
        """Clear the cached prefix (e.g. when system prompt changes)."""
        self._cached_ids = None
        self._cached_len = 0
        self._snapshots.clear()
        logger.info("Prefix cache invalidated")
