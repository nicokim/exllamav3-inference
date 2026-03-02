"""FP8 E4M3FN KV cache layer.

Implements the CacheLayer interface with FP8 quantized storage:
- K and V stored as torch.float8_e4m3fn
- Per-head-per-token scales stored as torch.float16
- ~50% VRAM savings vs FP16 cache (e.g. 293 MB vs 576 MB for 8192 tokens)

Requires compiled CUDA kernels (fp8_cache_kernels.cu).
"""

from __future__ import annotations

import torch

PAGE_SIZE = 256  # Must match exllamav3.constants.PAGE_SIZE

# Scratch buffers for dequantization (reused across calls)
_FP8_SCRATCH: dict[tuple[str, int | None, tuple], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_fp8_scratch(
    device: torch.device, shape: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get or create reusable scratch buffers for FP8 dequantization."""
    key = (device.type, device.index, shape)
    if key not in _FP8_SCRATCH:
        k = torch.empty(shape, dtype=torch.half, device=device)
        v = torch.empty(shape, dtype=torch.half, device=device)
        _FP8_SCRATCH[key] = (k, v)
    return _FP8_SCRATCH[key]


class CacheLayer_fp8:
    """FP8 E4M3FN quantized KV cache layer.

    Follows the same interface as upstream CacheLayer for drop-in compatibility.

    Storage layout:
        qk, qv: (num_pages, PAGE_SIZE, num_kv_heads, head_dim) as float8_e4m3fn
        sk, sv: (num_pages, PAGE_SIZE, num_kv_heads)           as float16
    """

    def __init__(
        self,
        config,
        attention,
        cache_id: int,
        max_num_tokens: int,
        **kwargs,
    ) -> None:
        self.cache_id = cache_id
        self.max_num_tokens = max_num_tokens
        self.attention = attention

        self.num_kv_heads = attention.num_kv_heads
        self.head_dim = attention.head_dim
        self.layer_idx = attention.layer_idx

        self.num_pages = max_num_tokens // PAGE_SIZE
        self.device: torch.device | None = None

        # FP8 quantized storage
        self.qk: torch.Tensor | None = None
        self.qv: torch.Tensor | None = None
        # Per-head-per-token scales
        self.sk: torch.Tensor | None = None
        self.sv: torch.Tensor | None = None

        # Shape for scratch buffers (full FP16 shape)
        self.fp16_shape = (self.num_pages, PAGE_SIZE, self.num_kv_heads, self.head_dim)
        self.scale_shape = (self.num_pages, PAGE_SIZE, self.num_kv_heads)

    def alloc(self, device: torch.device) -> None:
        """Allocate FP8 storage on GPU."""
        self.device = device
        fp8_shape = (self.num_pages, PAGE_SIZE, self.num_kv_heads, self.head_dim)

        self.qk = torch.zeros(fp8_shape, dtype=torch.float8_e4m3fn, device=device)
        self.qv = torch.zeros(fp8_shape, dtype=torch.float8_e4m3fn, device=device)
        self.sk = torch.zeros(self.scale_shape, dtype=torch.half, device=device)
        self.sv = torch.zeros(self.scale_shape, dtype=torch.half, device=device)

    def free(self) -> None:
        """Deallocate storage."""
        self.qk = None
        self.qv = None
        self.sk = None
        self.sv = None
        self.device = None

    def get_kv(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize FP8 -> FP16 for attention computation.

        Returns scratch buffers (k, v) in FP16, valid until the next call.
        """
        from exllamav3_opt._ext import dequant_fp8_cache_paged

        k_scratch, v_scratch = _get_fp8_scratch(self.device, self.fp16_shape)

        dequant_fp8_cache_paged(
            self.qk, self.sk, k_scratch,
            self.qv, self.sv, v_scratch,
            cache_seqlens, block_table,
            PAGE_SIZE,
        )

        return k_scratch, v_scratch

    def update_kv(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int,
    ) -> None:
        """Quantize incoming FP16 K,V -> FP8 and store in cache pages."""
        from exllamav3_opt._ext import quant_fp8_cache_paged

        quant_fp8_cache_paged(
            k, self.qk, self.sk,
            v, self.qv, self.sv,
            cache_seqlens, block_table,
            length,
        )

    def copy_page(
        self,
        source: CacheLayer_fp8,
        from_page: int,
        to_page: int,
        num_tokens: int,
    ) -> None:
        """Copy a page from another cache layer (for beam search / speculative)."""
        self.qk[to_page, :num_tokens].copy_(source.qk[from_page, :num_tokens])
        self.qv[to_page, :num_tokens].copy_(source.qv[from_page, :num_tokens])
        self.sk[to_page, :num_tokens].copy_(source.sk[from_page, :num_tokens])
        self.sv[to_page, :num_tokens].copy_(source.sv[from_page, :num_tokens])

    def get_tensors(self) -> list[torch.Tensor]:
        """Return all storage tensors (for serialization / prefix cache snapshot)."""
        return [self.qk, self.qv, self.sk, self.sv]

    def storage_size(self) -> int:
        """Total GPU memory used by this layer in bytes."""
        if self.qk is None:
            return 0
        return (
            self.qk.nbytes + self.qv.nbytes +
            self.sk.nbytes + self.sv.nbytes
        )

    def overhead_size(self) -> int:
        """Extra memory needed for dequantization scratch buffers."""
        # Two FP16 tensors of full shape
        per_tensor = (
            self.num_pages * PAGE_SIZE * self.num_kv_heads * self.head_dim * 2
        )
        return per_tensor * 2

    def get_kv_alloc_placeholder(self):
        """Placeholder for layer-split simulation."""
        return None

    def tp_export(self, plan):
        """Tensor parallel export (not supported for FP8 cache)."""
        raise NotImplementedError("FP8 cache does not support tensor parallelism")
