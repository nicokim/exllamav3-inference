"""Speculative pre-generation for VAD integration.

Saves a lightweight checkpoint of the current generation state (just the
current page's KV data + sequence length) so generation can start speculatively
while VAD is still confirming the user's utterance. If the prediction is wrong,
rolls back cheaply.

Memory cost: only 1 page snapshot (~256 tokens worth of KV), not the full cache.
"""

from __future__ import annotations

import logging

import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exllamav3.cache.cache import Cache

PAGE_SIZE = 256  # Must match exllamav3.constants.PAGE_SIZE

logger = logging.getLogger(__name__)


class SpeculativeCheckpoint:
    """Lightweight checkpoint for speculative generation rollback."""

    def __init__(self) -> None:
        self._saved = False
        self._kv_position: int = 0
        self._page_snapshots: dict[int, list[torch.Tensor]] = {}  # layer_idx -> tensors

    @property
    def is_saved(self) -> bool:
        return self._saved

    def save(self, cache: Cache, kv_position: int) -> None:
        """Snapshot the current page's KV data.

        Only saves the page that kv_position falls into, not the entire cache.
        """
        self._kv_position = kv_position
        self._page_snapshots.clear()

        page_idx = kv_position // PAGE_SIZE

        for layer_idx, layer in cache.layers.items():
            tensors = layer.get_tensors()
            snapshots = []
            for t in tensors:
                # Only snapshot the current page
                if t.dim() >= 2 and t.shape[0] > page_idx:
                    page_data = t[page_idx].clone()
                else:
                    page_data = t.clone()
                snapshots.append(page_data)
            self._page_snapshots[layer_idx] = snapshots

        self._saved = True
        logger.debug(
            "Speculative checkpoint saved at kv_position=%d (page %d)",
            kv_position,
            page_idx,
        )

    def rollback(self, cache: Cache) -> int:
        """Restore the saved page and return the checkpoint's kv_position.

        Returns:
            The kv_position to reset to.
        """
        if not self._saved:
            raise RuntimeError("No checkpoint to rollback to")

        page_idx = self._kv_position // PAGE_SIZE

        for layer_idx, snapshots in self._page_snapshots.items():
            tensors = cache.layers[layer_idx].get_tensors()
            for t, snapshot in zip(tensors, snapshots):
                if t.dim() >= 2 and t.shape[0] > page_idx:
                    t[page_idx].copy_(snapshot)
                else:
                    t.copy_(snapshot)

        kv_pos = self._kv_position
        logger.debug("Speculative rollback to kv_position=%d", kv_pos)
        return kv_pos

    def commit(self) -> None:
        """Discard the checkpoint (speculation was correct)."""
        self._saved = False
        self._kv_position = 0
        self._page_snapshots.clear()
        logger.debug("Speculative checkpoint committed (discarded)")
