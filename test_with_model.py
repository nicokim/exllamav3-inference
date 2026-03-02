"""Benchmark: Upstream Generator vs SlimGenerator.

Proper A/B comparison:
1. Loads model once
2. Warmup run for each generator (excluded from timing)
3. N timed runs of 256 tokens each, no stop conditions (force full decode)
4. Reports decode-only tok/s (prefill time separated)

Usage:
    python test_with_model.py                  # default: fused_norm ON
    python test_with_model.py --no-fused-norm  # disable fused norm
    python test_with_model.py --fp8-cache      # use FP8 KV cache
"""

from __future__ import annotations

import argparse
import time
import statistics

import torch

MODEL_PATH = "/home/kimox/.cache/huggingface/hub/models--kohai-channel--kohai-vl-27b-v2-EXL3/snapshots/072a9f85e0e15e80c65e4680792d78c016cb073b"

PROMPT = "<|im_start|>system\nEres Kohai, una VTuber AI amigable que habla en español.<|im_end|>\n<|im_start|>user\nHola Kohai! ¿Cómo estás hoy?<|im_end|>\n<|im_start|>assistant\n"

MAX_TOKENS = 256
NUM_RUNS = 3


def bench_upstream(gen, tokenizer, sampler):
    """Run upstream Generator, return (total_time, num_tokens)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = gen.generate(
        prompt=PROMPT,
        max_new_tokens=MAX_TOKENS,
        sampler=sampler,
        stop_conditions=[],  # no stop — force full decode
        add_bos=True,
        encode_special_tokens=True,
        completion_only=True,
        seed=42,
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    n_tok = len(tokenizer.encode(result, encode_special_tokens=True)[0])
    return t1 - t0, n_tok, result


def bench_slim(gen, tokenizer, sampler):
    """Run SlimGenerator streaming, return (prefill_time, decode_time, total_time, num_tokens)."""
    parts = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    first_token_time = None
    for chunk in gen.stream_tokens(
        PROMPT,
        max_new_tokens=MAX_TOKENS,
        sampler=sampler,
        stop_conditions=[],  # no stop — force full decode
        add_bos=True,
        encode_special_tokens=True,
        seed=42,
    ):
        if first_token_time is None:
            torch.cuda.synchronize()
            first_token_time = time.perf_counter()
        parts.append(chunk.text)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Count tokens by re-encoding the result (same method as upstream bench)
    result = "".join(parts)
    n_tok = len(tokenizer.encode(result, encode_special_tokens=True)[0])

    prefill_time = first_token_time - t0 if first_token_time else 0
    decode_time = t1 - first_token_time if first_token_time else t1 - t0
    total_time = t1 - t0
    return prefill_time, decode_time, total_time, n_tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fused-norm", action="store_true", help="Disable fused RMSNorm+Residual")
    parser.add_argument("--fp8-cache", action="store_true", help="Use FP8 KV cache (VRAM savings)")
    args = parser.parse_args()

    from exllamav3 import Cache, Config, Model, Tokenizer
    from exllamav3.generator.generator import Generator
    from exllamav3.generator.sampler import ComboSampler

    print("=" * 60)
    print("Loading model...")
    print("=" * 60)

    config = Config.from_directory(MODEL_PATH)
    model = Model.from_config(config)

    cache_kwargs = {}
    if args.fp8_cache:
        from exllamav3_opt.fp8_cache import CacheLayer_fp8
        cache_kwargs["layer_type"] = CacheLayer_fp8
        print("  KV cache: FP8 E4M3FN")
    else:
        print("  KV cache: FP16 (default)")

    cache = Cache(model, max_num_tokens=4096, **cache_kwargs)
    model.load(progressbar=True)
    tokenizer = Tokenizer.from_config(config)

    sampler = ComboSampler(temperature=0.6, top_p=0.95, top_k=20)

    # ---- Upstream Generator ----
    print("\n" + "=" * 60)
    print(f"UPSTREAM Generator — {NUM_RUNS} runs x {MAX_TOKENS} tokens")
    print("=" * 60)

    upstream_gen = Generator(model, cache, tokenizer)

    # Warmup
    print("  Warmup...", end="", flush=True)
    bench_upstream(upstream_gen, tokenizer, sampler)
    print(" done")

    upstream_speeds = []
    for i in range(NUM_RUNS):
        total, n_tok, result = bench_upstream(upstream_gen, tokenizer, sampler)
        speed = n_tok / total
        upstream_speeds.append(speed)
        print(f"  Run {i+1}: {n_tok} tok in {total:.3f}s = {speed:.1f} tok/s")
        if i == 0:
            print(f"  Output: {result[:80]}...")

    # ---- SlimGenerator ----
    use_fused = not args.no_fused_norm
    label = "SLIM Generator"
    if use_fused:
        label += " + fused norm"
    if args.fp8_cache:
        label += " + FP8 cache"

    print("\n" + "=" * 60)
    print(f"{label} — {NUM_RUNS} runs x {MAX_TOKENS} tokens")
    print("=" * 60)

    from exllamav3_opt.generator import SlimGenerator
    slim_gen = SlimGenerator(model, cache, tokenizer, use_fused_norm=use_fused)

    # Warmup
    print("  Warmup...", end="", flush=True)
    bench_slim(slim_gen, tokenizer, sampler)
    print(" done")

    slim_speeds = []
    slim_decode_speeds = []
    slim_prefills = []
    for i in range(NUM_RUNS):
        prefill, decode, total, n_tok = bench_slim(slim_gen, tokenizer, sampler)
        speed = n_tok / total
        decode_speed = (n_tok - 1) / decode if decode > 0 else 0
        slim_speeds.append(speed)
        slim_decode_speeds.append(decode_speed)
        slim_prefills.append(prefill)
        print(f"  Run {i+1}: {n_tok} tok in {total:.3f}s = {speed:.1f} tok/s "
              f"(prefill {prefill*1000:.0f}ms, decode {decode_speed:.1f} tok/s)")

    # ---- Comparison ----
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    up_mean = statistics.mean(upstream_speeds)
    sl_mean = statistics.mean(slim_speeds)
    sl_decode_mean = statistics.mean(slim_decode_speeds)
    sl_prefill_mean = statistics.mean(slim_prefills)

    print(f"  Upstream total:    {up_mean:.1f} tok/s")
    print(f"  Slim total:        {sl_mean:.1f} tok/s")
    print(f"  Slim decode-only:  {sl_decode_mean:.1f} tok/s")
    print(f"  Slim avg prefill:  {sl_prefill_mean*1000:.0f} ms")
    print(f"  Speedup (total):   {(sl_mean/up_mean - 1)*100:+.1f}%")

    print(f"\n  VRAM used: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
    print(f"  VRAM peak: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")


if __name__ == "__main__":
    main()
