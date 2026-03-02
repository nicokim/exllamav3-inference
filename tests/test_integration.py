"""Tests for OptimizedLLM integration."""

from __future__ import annotations

import asyncio

import pytest


@pytest.mark.requires_model
class TestOptimizedLLM:

    @pytest.fixture
    def llm(self, model_path):
        from exllamav3_opt.integration import LLMConfig, OptimizedLLM

        config = LLMConfig(
            model_repo="test",
            max_new_tokens=20,
            cache_size=2048,
        )
        llm = OptimizedLLM(config)
        llm.load(model_path, enable_prefix_cache=True)
        return llm

    def test_generate(self, llm):
        """Async generate returns a non-empty string."""
        result = asyncio.run(llm.generate("Hello, how are you?"))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_stream(self, llm):
        """Async stream yields token strings."""
        async def _stream():
            tokens = []
            async for token in await llm.stream("Hello"):  # noqa: ASYNC110
                tokens.append(token)
            return tokens

        tokens = asyncio.run(_stream())
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_strategy_property(self, llm):
        """Strategy property returns a valid strategy."""
        strategy = llm.strategy
        assert strategy is not None
        assert hasattr(strategy, "get_sampler")
        assert hasattr(strategy, "get_stop_conditions")
        assert hasattr(strategy, "build_prompt")


class TestLLMConfig:
    """Test LLMConfig without model."""

    def test_default_config(self):
        from exllamav3_opt.integration import LLMConfig

        config = LLMConfig()
        assert config.model_repo == ""
        assert config.max_new_tokens == 256
        assert config.cache_size == 2048

    def test_custom_config(self):
        from exllamav3_opt.integration import LLMConfig

        config = LLMConfig(
            model_repo="test/model",
            max_new_tokens=100,
            cache_size=4096,
            temperature=0.8,
        )
        assert config.model_repo == "test/model"
        assert config.max_new_tokens == 100
        assert config.temperature == 0.8


class TestFallbackStrategy:
    """Test fallback strategy without model."""

    def test_build_prompt(self):
        from exllamav3_opt.integration import _FallbackQwen35VLStrategy

        strategy = _FallbackQwen35VLStrategy()
        prompt = strategy.build_prompt(
            "Hello",
            system_prompt="You are helpful.",
            history=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hey!"}],
        )

        assert "<|im_start|>system" in prompt
        assert "You are helpful." in prompt
        assert "<|im_start|>user\nHello<|im_end|>" in prompt
        assert "<|im_start|>assistant\n" in prompt

    def test_build_prompt_with_aliases(self):
        from exllamav3_opt.integration import _FallbackQwen35VLStrategy

        strategy = _FallbackQwen35VLStrategy()
        prompt = strategy.build_prompt(
            "What is this?",
            embedding_aliases=["<image>"],
        )

        assert "<image>What is this?" in prompt

    def test_stop_conditions(self):
        from exllamav3_opt.integration import _FallbackQwen35VLStrategy

        strategy = _FallbackQwen35VLStrategy()
        assert strategy.get_add_bos() is True
        assert strategy.get_cache_size() == 8192
