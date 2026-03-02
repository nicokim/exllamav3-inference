"""Prefill benchmark: measures time savings from prefix cache.

Usage:
    EXLLAMAV3_MODEL_PATH=/path/to/model python benchmarks/bench_prefill.py
"""

from __future__ import annotations

import os
import sys
import time

import torch


def main():
    model_path = os.environ.get("EXLLAMAV3_MODEL_PATH")
    if not model_path:
        print("Set EXLLAMAV3_MODEL_PATH to run benchmark")
        sys.exit(1)

    from exllamav3 import Cache, Config, Model, Tokenizer
    from exllamav3_opt.generator import SlimGenerator
    from exllamav3_opt.prefix_cache import PrefixCache

    config = Config.from_directory(model_path)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    model.load(progressbar=True)
    tokenizer = Tokenizer.from_config(config)

    system_prompt = "You are a helpful AI assistant. You answer questions accurately and concisely." * 10
    user_msg = "What is the capital of France?"

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    input_ids = tokenizer.encode(prompt, encode_special_tokens=True, add_bos=True)
    prompt_len = input_ids.shape[-1]

    print(f"\nBenchmark: prefill with {prompt_len} tokens")
    print("=" * 50)

    # Without prefix cache
    gen_no_cache = SlimGenerator(model, cache, tokenizer)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gen_no_cache.generate(prompt, max_new_tokens=1, add_bos=True, encode_special_tokens=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    no_cache_time = t1 - t0
    print(f"Without prefix cache: {no_cache_time:.3f}s")

    # With prefix cache — first run (capture)
    prefix_cache = PrefixCache()
    gen_cached = SlimGenerator(model, cache, tokenizer, prefix_cache=prefix_cache)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gen_cached.generate(prompt, max_new_tokens=1, add_bos=True, encode_special_tokens=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    first_run_time = t1 - t0
    print(f"First run (capture): {first_run_time:.3f}s")

    # With prefix cache — second run (restore)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gen_cached.generate(prompt, max_new_tokens=1, add_bos=True, encode_special_tokens=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    cached_time = t1 - t0
    print(f"With prefix cache:   {cached_time:.3f}s")

    reduction = (1 - cached_time / no_cache_time) * 100
    print(f"\nPrefill time reduction: {reduction:.0f}%")


if __name__ == "__main__":
    main()
