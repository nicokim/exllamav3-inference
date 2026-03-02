"""Pre-allocated tensor pool for single-user inference.

Eliminates per-iteration tensor allocation by providing a fixed set of
GPU and pinned-CPU tensors sized for batch=1 decode.
"""

from __future__ import annotations

import torch

PAGE_SIZE = 256  # Must match exllamav3.constants.PAGE_SIZE


class TensorPool:
    """Owns all reusable tensors for SlimGenerator's decode loop.

    Tensors are allocated once at init and updated in-place each step.
    """

    def __init__(
        self,
        device: torch.device,
        hidden_size: int,
        vocab_size: int,
        max_pages: int,
    ) -> None:
        self.device = device
        self.page_size = PAGE_SIZE
        self.max_pages = max_pages

        padded_vocab = ((vocab_size + 31) // 32) * 32

        # GPU tensors — decode step (batch=1, seq=1)
        self.input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.cache_seqlens = torch.zeros((1,), dtype=torch.int32, device=device)
        self.logits = torch.empty((1, 1, padded_vocab), dtype=torch.half, device=device)

        # Block table: (1, max_pages) — page indices for this sequence
        self.block_table = torch.zeros((1, max_pages), dtype=torch.int32, device=device)

        # CPU pinned tensors for async transfer
        self.input_ids_cpu = torch.zeros((1, 1), dtype=torch.long).pin_memory()
        self.sample_id_cpu = torch.zeros((1,), dtype=torch.long).pin_memory()

        # Prefill input_ids (variable length, allocated lazily)
        self._prefill_ids: torch.Tensor | None = None

        # Pre-allocated tensor for tokenizer.decode() (avoids per-iteration allocation)
        self.decode_token = torch.zeros((1,), dtype=torch.long)

        # Pre-allocated output for fused sampling kernel
        self.sample_output = torch.zeros((1,), dtype=torch.long, device=device)

    def setup_pages(self, num_pages: int) -> None:
        """Initialize block_table with sequential page indices (pre-allocated)."""
        assert num_pages <= self.max_pages
        self.block_table[0, :num_pages] = torch.arange(num_pages, dtype=torch.int32)
        if num_pages < self.max_pages:
            self.block_table[0, num_pages:] = 0

    def set_cache_seqlen(self, length: int) -> None:
        """Update cache sequence length in-place."""
        self.cache_seqlens[0] = length

    def set_input_id(self, token_id: int) -> None:
        """Set the single decode token in-place."""
        self.input_ids[0, 0] = token_id

    def get_prefill_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Return input_ids tensor for prefill on the correct device."""
        if ids.device != self.device:
            ids = ids.to(self.device, non_blocking=True)
        self._prefill_ids = ids
        return self._prefill_ids

    def get_block_table(self, num_pages: int) -> torch.Tensor:
        """Return block_table view for the current number of pages."""
        return self.block_table[:, :num_pages]
