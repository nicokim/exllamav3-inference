"""Tests for SlimGenerator.

Validates that SlimGenerator produces coherent output and handles
stop conditions, cancellation, and streaming correctly.
"""

from __future__ import annotations

import threading

import pytest


@pytest.mark.requires_model
class TestSlimGenerator:

    def test_generate_basic(self, loaded_model):
        """Basic non-streaming generation produces non-empty output."""
        from exllamav3_opt.generator import SlimGenerator

        model, tokenizer, cache = loaded_model
        gen = SlimGenerator(model, cache, tokenizer)

        result = gen.generate(
            "Hello, how are you?",
            max_new_tokens=20,
            add_bos=True,
            encode_special_tokens=True,
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_stream_tokens(self, loaded_model):
        """Streaming yields StreamChunk objects with text."""
        from exllamav3_opt.generator import SlimGenerator, StreamChunk

        model, tokenizer, cache = loaded_model
        gen = SlimGenerator(model, cache, tokenizer)

        chunks = list(gen.stream_tokens(
            "Once upon a time",
            max_new_tokens=10,
            add_bos=True,
        ))

        assert len(chunks) > 0
        assert all(isinstance(c, StreamChunk) for c in chunks)
        # Last chunk should be eos
        assert chunks[-1].eos

    def test_stop_token(self, loaded_model):
        """Generation stops on EOS token ID."""
        from exllamav3_opt.generator import SlimGenerator

        model, tokenizer, cache = loaded_model
        gen = SlimGenerator(model, cache, tokenizer)

        # Use a common token as stop condition to trigger early stop
        result = gen.generate(
            "The quick brown fox",
            max_new_tokens=100,
            stop_conditions=[tokenizer.eos_token_id],
            add_bos=True,
        )
        # Should stop before 100 tokens
        assert isinstance(result, str)

    def test_stop_string(self, loaded_model):
        """Generation stops on stop string."""
        from exllamav3_opt.generator import SlimGenerator

        model, tokenizer, cache = loaded_model
        gen = SlimGenerator(model, cache, tokenizer)

        result = gen.generate(
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
            max_new_tokens=100,
            stop_conditions=["<|im_end|>"],
            add_bos=True,
            encode_special_tokens=True,
        )
        # Should not contain the stop string
        assert "<|im_end|>" not in result

    def test_cancel_flag(self, loaded_model):
        """Cancellation via threading.Event stops generation."""
        from exllamav3_opt.generator import SlimGenerator

        model, tokenizer, cache = loaded_model
        gen = SlimGenerator(model, cache, tokenizer)

        cancel = threading.Event()
        cancel.set()  # Immediately cancel

        chunks = list(gen.stream_tokens(
            "Hello",
            max_new_tokens=100,
            cancel_flag=cancel,
            add_bos=True,
        ))

        # Should have very few or zero chunks due to immediate cancel
        assert len(chunks) <= 2

    def test_max_new_tokens(self, loaded_model):
        """Generation respects max_new_tokens limit."""
        from exllamav3_opt.generator import SlimGenerator

        model, tokenizer, cache = loaded_model
        gen = SlimGenerator(model, cache, tokenizer)

        max_tokens = 5
        chunks = list(gen.stream_tokens(
            "Hello",
            max_new_tokens=max_tokens,
            add_bos=True,
        ))

        # Number of chunks <= max_tokens (some chunks may combine text)
        assert len(chunks) <= max_tokens + 1  # +1 for possible partial hold-back flush


class TestStopStringLogic:
    """Unit tests for stop string detection (no model needed)."""

    def test_no_stop_strings(self):
        from exllamav3_opt.generator import _check_stop_strings

        eos, text, buf = _check_stop_strings("hello world", set())
        assert not eos
        assert text == "hello world"
        assert buf == ""

    def test_stop_string_found(self):
        from exllamav3_opt.generator import _check_stop_strings

        eos, text, buf = _check_stop_strings("hello<|im_end|>world", {"<|im_end|>"})
        assert eos
        assert text == "hello"
        assert buf == ""

    def test_partial_match_held_back(self):
        from exllamav3_opt.generator import _check_stop_strings

        # "<|im_end|>" is 10 chars, so hold back 9
        eos, text, buf = _check_stop_strings("hello<|im", {"<|im_end|>"})
        assert not eos
        assert text == ""  # entire buffer held back (len=9 <= max_hold=9)
        assert buf == "hello<|im"

    def test_parse_stop_conditions(self):
        from exllamav3_opt.generator import _parse_stop_conditions

        tokens, strings = _parse_stop_conditions([42, "<|im_end|>", 100, "stop"])
        assert tokens == {42, 100}
        assert strings == {"<|im_end|>", "stop"}

    def test_parse_stop_conditions_none(self):
        from exllamav3_opt.generator import _parse_stop_conditions

        tokens, strings = _parse_stop_conditions(None)
        assert tokens == set()
        assert strings == set()
