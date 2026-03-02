"""Decode benchmark: tokens/s comparison between SlimGenerator and upstream Generator.

Usage:
    EXLLAMAV3_MODEL_PATH=/path/to/model python benchmarks/bench_decode.py
"""

from __future__ import annotations

import os
import sys
import time

import torch


def bench_slim(model, cache, tokenizer, prompt: str, max_tokens: int, warmup: int = 2):
    from exllamav3_opt.generator import SlimGenerator

    gen = SlimGenerator(model, cache, tokenizer)

    # Warmup
    for _ in range(warmup):
        gen.generate(prompt, max_new_tokens=5, add_bos=True, encode_special_tokens=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = gen.generate(
        prompt, max_new_tokens=max_tokens, add_bos=True, encode_special_tokens=True
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    tokens = len(tokenizer.encode(result, encode_special_tokens=True)[0])
    return tokens, t1 - t0


def bench_upstream(model, cache, tokenizer, prompt: str, max_tokens: int, warmup: int = 2):
    from exllamav3.generator.generator import Generator

    gen = Generator(model, cache, tokenizer)

    # Warmup
    for _ in range(warmup):
        gen.generate(prompt, max_new_tokens=5, add_bos=True, encode_special_tokens=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = gen.generate(
        prompt, max_new_tokens=max_tokens, add_bos=True, encode_special_tokens=True
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    tokens = len(tokenizer.encode(result, encode_special_tokens=True)[0])
    return tokens, t1 - t0


def main():
    model_path = os.environ.get("EXLLAMAV3_MODEL_PATH")
    if not model_path:
        print("Set EXLLAMAV3_MODEL_PATH to run benchmark")
        sys.exit(1)

    from exllamav3 import Cache, Config, Model, Tokenizer

    config = Config.from_directory(model_path)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=2048)
    model.load(progressbar=True)
    tokenizer = Tokenizer.from_config(config)

    prompt = (
        "<|im_start|>user\nExplain quantum computing in simple terms."
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    max_tokens = 128

    print(f"\nBenchmark: decode {max_tokens} tokens")
    print("=" * 50)

    # SlimGenerator
    slim_tokens, slim_time = bench_slim(model, cache, tokenizer, prompt, max_tokens)
    slim_tps = slim_tokens / slim_time
    print(f"SlimGenerator:    {slim_tokens} tokens in {slim_time:.3f}s = {slim_tps:.1f} tok/s")

    # Upstream Generator
    up_tokens, up_time = bench_upstream(model, cache, tokenizer, prompt, max_tokens)
    up_tps = up_tokens / up_time
    print(f"Upstream Gen:     {up_tokens} tokens in {up_time:.3f}s = {up_tps:.1f} tok/s")

    speedup = (slim_tps / up_tps - 1) * 100
    print(f"\nSpeedup: {speedup:+.1f}%")


if __name__ == "__main__":
    main()
