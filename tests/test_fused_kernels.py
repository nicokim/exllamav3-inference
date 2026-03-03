"""Tests for fused CUDA kernels.

Validates output matches unfused reference implementations.
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.requires_cuda
class TestFusedRMSNormResidual:

    def test_matches_unfused(self, device):
        """Fused RMSNorm+Residual matches sequential unfused ops."""
        try:
            from exllamav3_opt._ext import fused_rmsnorm_residual
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        dim = 3584  # Qwen3.5-27B hidden size
        rows = 4
        epsilon = 1e-6

        x = torch.randn(rows, dim, dtype=torch.half, device=device)
        attn_out = torch.randn(rows, dim, dtype=torch.half, device=device)
        weight = torch.randn(dim, dtype=torch.half, device=device)

        # Reference: unfused
        x_ref = x.clone()
        x_ref += attn_out
        rms = torch.sqrt(torch.mean(x_ref.float() ** 2, dim=-1, keepdim=True) + epsilon)
        y_ref = (x_ref.float() / rms * weight.float()).half()

        # Fused kernel
        x_fused = x.clone()
        y_fused = torch.empty_like(x_fused)
        fused_rmsnorm_residual(x_fused, attn_out, weight, y_fused, epsilon, 0.0)

        # x should be modified in-place (residual add)
        x_expected = x + attn_out
        assert torch.allclose(x_fused.float(), x_expected.float(), atol=1e-3), \
            f"Residual add mismatch: max diff = {(x_fused - x_expected).abs().max()}"

        # y should match normalized output
        assert torch.allclose(y_fused.float(), y_ref.float(), atol=1e-2), \
            f"RMSNorm output mismatch: max diff = {(y_fused - y_ref).abs().max()}"


@pytest.mark.requires_cuda
class TestFusedSampling:

    def test_argmax_mode(self, device):
        """Temperature=0 (argmax) returns the correct token."""
        try:
            from exllamav3_opt._ext import fused_sample
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        vocab_size = 152064
        logits = torch.randn(1, vocab_size, dtype=torch.half, device=device)
        output_id = torch.zeros(1, dtype=torch.long, device=device)

        fused_sample(logits, output_id, 0.0, 0, 0)

        expected = logits.float().argmax().item()
        assert output_id.item() == expected

    def test_temperature_sampling(self, device):
        """Non-zero temperature produces valid token IDs."""
        try:
            from exllamav3_opt._ext import fused_sample
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        vocab_size = 152064
        logits = torch.randn(1, vocab_size, dtype=torch.half, device=device)
        output_id = torch.zeros(1, dtype=torch.long, device=device)

        fused_sample(logits, output_id, 0.6, 20, 42)

        token_id = output_id.item()
        assert 0 <= token_id < vocab_size

    def test_top_k_constraint(self, device):
        """With top_k=1, should always pick the argmax."""
        try:
            from exllamav3_opt._ext import fused_sample
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        vocab_size = 1000
        logits = torch.randn(1, vocab_size, dtype=torch.half, device=device)
        output_id = torch.zeros(1, dtype=torch.long, device=device)

        # Very high temperature but top_k=1 should still pick argmax
        fused_sample(logits, output_id, 10.0, 1, 123)

        expected = logits.float().argmax().item()
        assert output_id.item() == expected

    def test_deterministic_with_same_seed(self, device):
        """Same seed produces same result."""
        try:
            from exllamav3_opt._ext import fused_sample
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        vocab_size = 152064
        logits = torch.randn(1, vocab_size, dtype=torch.half, device=device)

        output1 = torch.zeros(1, dtype=torch.long, device=device)
        output2 = torch.zeros(1, dtype=torch.long, device=device)

        fused_sample(logits, output1, 0.6, 20, 42)
        fused_sample(logits, output2, 0.6, 20, 42)

        assert output1.item() == output2.item()
