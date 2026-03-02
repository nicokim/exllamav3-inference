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
        ids = llm.build_prompt([{"role": "user", "content": "Hello, how are you?"}])
        result = asyncio.run(llm.generate(ids))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_stream(self, llm):
        """Async stream yields token strings."""
        async def _stream():
            tokens = []
            ids = llm.build_prompt([{"role": "user", "content": "Hello"}])
            async for token in await llm.stream(ids):  # noqa: ASYNC110
                tokens.append(token)
            return tokens

        tokens = asyncio.run(_stream())
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)


class TestLLMConfig:
    """Test LLMConfig without model."""

    def test_default_config(self):
        from exllamav3_opt.integration import LLMConfig

        config = LLMConfig()
        assert config.model_repo == ""
        assert config.max_new_tokens == 256
        assert config.cache_size == 2048
        assert config.temperature == 0.8
        assert config.top_k == 50

    def test_custom_config(self):
        from exllamav3_opt.integration import LLMConfig

        config = LLMConfig(
            model_repo="test/model",
            max_new_tokens=100,
            cache_size=4096,
            temperature=0.6,
            top_k=20,
        )
        assert config.model_repo == "test/model"
        assert config.max_new_tokens == 100
        assert config.temperature == 0.6
        assert config.top_k == 20
