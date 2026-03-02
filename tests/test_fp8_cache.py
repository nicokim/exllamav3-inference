"""Tests for FP8 cache quantization.

Validates round-trip error FP16 -> FP8 -> FP16 is within acceptable bounds.
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.requires_cuda
class TestFP8CacheKernels:
    """Tests for FP8 cache CUDA kernels."""

    def test_round_trip_error(self, device):
        """Round-trip quantize/dequantize error < 0.05 max abs."""
        try:
            from exllamav3_opt._ext import dequant_fp8_cache_paged, quant_fp8_cache_paged
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        num_pages = 4
        page_size = 256
        num_heads = 4
        head_dim = 128
        bsz = 1
        new_tokens = 16

        # Random FP16 K and V inputs
        k_in = torch.randn(bsz, new_tokens, num_heads, head_dim, dtype=torch.half, device=device)
        v_in = torch.randn(bsz, new_tokens, num_heads, head_dim, dtype=torch.half, device=device)

        # FP8 storage
        shape = (num_pages, page_size, num_heads, head_dim)
        scale_shape = (num_pages, page_size, num_heads)

        k_out = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
        v_out = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
        k_scales = torch.zeros(scale_shape, dtype=torch.half, device=device)
        v_scales = torch.zeros(scale_shape, dtype=torch.half, device=device)

        # Block table: sequential pages
        block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
        cache_seqlens = torch.zeros(bsz, dtype=torch.int32, device=device)

        # Quantize
        quant_fp8_cache_paged(
            k_in, k_out, k_scales,
            v_in, v_out, v_scales,
            cache_seqlens, block_table,
            new_tokens,
        )

        # Dequantize to scratch
        k_deq = torch.zeros(shape, dtype=torch.half, device=device)
        v_deq = torch.zeros(shape, dtype=torch.half, device=device)

        # Update seqlens to reflect stored tokens
        cache_seqlens[0] = new_tokens

        dequant_fp8_cache_paged(
            k_out, k_scales, k_deq,
            v_out, v_scales, v_deq,
            cache_seqlens, block_table,
            page_size,
        )

        # Compare: extract the tokens we stored (page 0, positions 0:new_tokens)
        k_original = k_in[0]  # (new_tokens, num_heads, head_dim)
        k_restored = k_deq[0, :new_tokens]  # (new_tokens, num_heads, head_dim)

        v_original = v_in[0]
        v_restored = v_deq[0, :new_tokens]

        k_error = (k_original.float() - k_restored.float()).abs().max().item()
        v_error = (v_original.float() - v_restored.float()).abs().max().item()

        assert k_error < 0.05, f"K round-trip error too large: {k_error}"
        assert v_error < 0.05, f"V round-trip error too large: {v_error}"

    def test_zero_values(self, device):
        """FP8 handles zero values correctly."""
        try:
            from exllamav3_opt._ext import dequant_fp8_cache_paged, quant_fp8_cache_paged
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        num_pages = 1
        page_size = 256
        num_heads = 2
        head_dim = 64
        bsz = 1
        new_tokens = 4

        k_in = torch.zeros(bsz, new_tokens, num_heads, head_dim, dtype=torch.half, device=device)
        v_in = torch.zeros(bsz, new_tokens, num_heads, head_dim, dtype=torch.half, device=device)

        shape = (num_pages, page_size, num_heads, head_dim)
        scale_shape = (num_pages, page_size, num_heads)

        k_out = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
        v_out = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
        k_scales = torch.zeros(scale_shape, dtype=torch.half, device=device)
        v_scales = torch.zeros(scale_shape, dtype=torch.half, device=device)

        block_table = torch.zeros(bsz, num_pages, dtype=torch.int32, device=device)
        cache_seqlens = torch.zeros(bsz, dtype=torch.int32, device=device)

        quant_fp8_cache_paged(
            k_in, k_out, k_scales,
            v_in, v_out, v_scales,
            cache_seqlens, block_table,
            new_tokens,
        )

        k_deq = torch.zeros(shape, dtype=torch.half, device=device)
        v_deq = torch.zeros(shape, dtype=torch.half, device=device)
        cache_seqlens[0] = new_tokens

        dequant_fp8_cache_paged(
            k_out, k_scales, k_deq,
            v_out, v_scales, v_deq,
            cache_seqlens, block_table,
            page_size,
        )

        assert k_deq[0, :new_tokens].abs().max().item() == 0.0
        assert v_deq[0, :new_tokens].abs().max().item() == 0.0
