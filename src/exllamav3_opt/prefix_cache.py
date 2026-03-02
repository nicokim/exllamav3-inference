"""System prompt KV cache.

After the first prefill of a system prompt, snapshots the KV cache pages
and recurrent state (GatedDeltaNet) to CPU pinned memory. Subsequent
generations with the same prefix skip the prefill for the cached portion
and restore from the CPU snapshot.

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
        self._recurrent_snapshots: dict[int, tuple] = {}  # layer_idx -> (rec, conv, position)

    @property
    def is_captured(self) -> bool:
        return self._cached_len > 0

    @torch.no_grad()
    def capture(
        self,
        input_ids: torch.Tensor,
        cache: Cache,
        kv_position: int,
        recurrent_states: dict | None = None,
    ) -> None:
        """Snapshot KV cache and recurrent state to CPU after first prefill.

        Args:
            input_ids: Full input token IDs (1, seq_len).
            cache: The Cache object with populated KV data.
            kv_position: Number of tokens stored in cache (= seq_len after prefill).
            recurrent_states: GatedDeltaNet recurrent states dict (layer_idx -> state).
        """
        self._cached_ids = input_ids.cpu().clone()
        self._cached_len = kv_position
        self._snapshots.clear()
        self._recurrent_snapshots.clear()

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

        # Snapshot recurrent state (GatedDeltaNet conv + recurrent state)
        if recurrent_states:
            for layer_idx, rs in recurrent_states.items():
                rec = (
                    rs.last_recurrent_state.cpu().clone()
                    if rs.last_recurrent_state is not None
                    else None
                )
                conv = rs.last_conv_state.cpu().clone() if rs.last_conv_state is not None else None
                pos = rs.position if hasattr(rs, "position") else 0
                self._recurrent_snapshots[layer_idx] = (rec, conv, pos)

        logger.info(
            "Prefix cache captured: %d tokens, %d KV layers, %d recurrent layers, %.1f MB",
            kv_position,
            len(self._snapshots),
            len(self._recurrent_snapshots),
            sum(sum(t.nbytes for t in tensors) for tensors in self._snapshots.values())
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

    @torch.inference_mode()
    def restore_to_cache(self, cache: Cache) -> None:
        """Copy KV snapshot from CPU back to GPU cache pages.

        Uses inference_mode() because KV cache tensors are inference
        tensors (created inside model.forward's inference_mode context)
        and can only be modified in-place within inference_mode.

        Args:
            cache: The Cache object to restore into.
        """
        if not self.is_captured:
            return

        for layer_idx, cpu_tensors in self._snapshots.items():
            gpu_tensors = cache.layers[layer_idx].get_tensors()
            for gpu_t, cpu_t in zip(gpu_tensors, cpu_tensors):
                gpu_t.copy_(cpu_t, non_blocking=True)

    def restore_recurrent_states(self, recurrent_states: dict) -> None:
        """Restore recurrent state snapshots to GPU.

        Args:
            recurrent_states: Current recurrent states dict to restore into.
        """
        if not self._recurrent_snapshots:
            return

        for layer_idx, (rec, conv, pos) in self._recurrent_snapshots.items():
            if layer_idx not in recurrent_states:
                continue
            rs = recurrent_states[layer_idx]
            if rec is not None:
                device = (
                    rs.last_recurrent_state.device
                    if rs.last_recurrent_state is not None
                    else "cuda"
                )
                rs.last_recurrent_state = rec.to(device)
            if conv is not None:
                device = rs.last_conv_state.device if rs.last_conv_state is not None else "cuda"
                rs.last_conv_state = conv.to(device)
            if hasattr(rs, "position"):
                rs.position = pos

    def invalidate(self) -> None:
        """Clear the cached prefix (e.g. when system prompt changes)."""
        self._cached_ids = None
        self._cached_len = 0
        self._snapshots.clear()
        self._recurrent_snapshots.clear()
        logger.info("Prefix cache invalidated")
