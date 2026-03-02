"""Tests for PrefixCache."""

from __future__ import annotations

import pytest
import torch


class TestPrefixCacheUnit:
    """Unit tests for prefix cache logic (no model needed)."""

    def test_initial_state(self):
        from exllamav3_opt.prefix_cache import PrefixCache

        pc = PrefixCache()
        assert not pc.is_captured
        assert pc.get_cached_length(torch.tensor([[1, 2, 3]])) == 0

    def test_invalidate(self):
        from exllamav3_opt.prefix_cache import PrefixCache

        pc = PrefixCache()
        # Manually set state
        pc._cached_ids = torch.tensor([[1, 2, 3]])
        pc._cached_len = 3
        pc._recurrent_snapshots = {0: (torch.zeros(1), torch.zeros(1), 3)}
        assert pc.is_captured

        pc.invalidate()
        assert not pc.is_captured
        assert len(pc._recurrent_snapshots) == 0

    def test_get_cached_length_full_match(self):
        from exllamav3_opt.prefix_cache import PrefixCache

        pc = PrefixCache()
        pc._cached_ids = torch.tensor([[10, 20, 30, 40, 50]])
        pc._cached_len = 5

        # Full prefix match
        result = pc.get_cached_length(torch.tensor([[10, 20, 30, 40, 50, 60, 70]]))
        assert result == 5

    def test_get_cached_length_no_match(self):
        from exllamav3_opt.prefix_cache import PrefixCache

        pc = PrefixCache()
        pc._cached_ids = torch.tensor([[10, 20, 30]])
        pc._cached_len = 3

        # Different prefix
        result = pc.get_cached_length(torch.tensor([[99, 20, 30, 40]]))
        assert result == 0

    def test_get_cached_length_shorter_input(self):
        from exllamav3_opt.prefix_cache import PrefixCache

        pc = PrefixCache()
        pc._cached_ids = torch.tensor([[10, 20, 30, 40, 50]])
        pc._cached_len = 5

        # Input shorter than cached prefix — compares only overlap
        result = pc.get_cached_length(torch.tensor([[10, 20]]))
        assert result == 2

    def test_restore_recurrent_states(self):
        """Recurrent state snapshots are restored to target states."""
        from exllamav3_opt.prefix_cache import PrefixCache

        pc = PrefixCache()

        # Simulate a GDN_RecurrentState with the same attributes
        class FakeRecurrentState:
            def __init__(self):
                self.last_recurrent_state = None
                self.last_conv_state = None
                self.position = 0

        # Manually populate a snapshot (as if captured after prefill)
        rec_tensor = torch.randn(1, 4, 8, 16)
        conv_tensor = torch.randn(1, 32, 4)
        pc._recurrent_snapshots = {0: (rec_tensor, conv_tensor, 42)}

        # Create target states to restore into
        target = {0: FakeRecurrentState()}
        pc.restore_recurrent_states(target)

        assert target[0].last_recurrent_state is not None
        assert torch.equal(target[0].last_recurrent_state.cpu(), rec_tensor)
        assert torch.equal(target[0].last_conv_state.cpu(), conv_tensor)
        assert target[0].position == 42

    def test_restore_recurrent_states_empty(self):
        """Restore with no snapshots is a no-op."""
        from exllamav3_opt.prefix_cache import PrefixCache

        pc = PrefixCache()

        class FakeRecurrentState:
            def __init__(self):
                self.last_recurrent_state = None
                self.last_conv_state = None
                self.position = 0

        target = {0: FakeRecurrentState()}
        pc.restore_recurrent_states(target)

        # Should remain unchanged
        assert target[0].last_recurrent_state is None
        assert target[0].position == 0


@pytest.mark.requires_model
class TestPrefixCacheIntegration:
    def test_capture_and_restore(self, loaded_model):
        """Capture KV snapshot and verify restore produces same output."""
        from exllamav3_opt.generator import SlimGenerator
        from exllamav3_opt.prefix_cache import PrefixCache

        model, tokenizer, cache = loaded_model
        prefix_cache = PrefixCache()
        gen = SlimGenerator(model, cache, tokenizer, prefix_cache=prefix_cache)

        prompt = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        )

        # First generation — captures prefix cache
        result1 = gen.generate(
            prompt,
            max_new_tokens=10,
            add_bos=True,
            encode_special_tokens=True,
            seed=42,
        )

        assert prefix_cache.is_captured

        # Second generation — should use prefix cache
        result2 = gen.generate(
            prompt,
            max_new_tokens=10,
            add_bos=True,
            encode_special_tokens=True,
            seed=42,
        )

        # With same seed and prefix, should produce same output
        assert result1 == result2
