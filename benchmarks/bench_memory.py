"""Memory benchmark: compares VRAM usage between FP16 and FP8 KV cache.

Usage:
    python benchmarks/bench_memory.py
"""

from __future__ import annotations

import torch


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda:0")

    # Qwen3.5-27B config
    num_layers = 36
    num_kv_heads = 4
    head_dim = 128
    page_size = 256
    max_tokens = 8192
    num_pages = max_tokens // page_size

    print(f"Config: {num_layers} layers, {num_kv_heads} KV heads, head_dim={head_dim}")
    print(f"Max tokens: {max_tokens} ({num_pages} pages of {page_size})")
    print("=" * 50)

    # FP16 cache size per layer
    fp16_per_layer = num_pages * page_size * num_kv_heads * head_dim * 2  # bytes
    fp16_total = fp16_per_layer * num_layers * 2  # K + V
    print("\nFP16 KV cache:")
    print(f"  Per layer (K+V): {fp16_per_layer * 2 / 1024 / 1024:.1f} MB")
    print(f"  Total:           {fp16_total / 1024 / 1024:.1f} MB")

    # FP8 cache size per layer
    fp8_per_layer_kv = num_pages * page_size * num_kv_heads * head_dim * 1  # 1 byte per fp8
    fp8_per_layer_scales = num_pages * page_size * num_kv_heads * 2  # fp16 scales
    fp8_per_layer = fp8_per_layer_kv + fp8_per_layer_scales
    fp8_total = fp8_per_layer * num_layers * 2  # K + V
    print("\nFP8 E4M3 KV cache:")
    print(f"  Per layer (K+V): {fp8_per_layer * 2 / 1024 / 1024:.1f} MB")
    print(f"  Total:           {fp8_total / 1024 / 1024:.1f} MB")

    # Overhead: dequant scratch buffers (shared, not per-layer)
    scratch = num_pages * page_size * num_kv_heads * head_dim * 2 * 2  # 2 fp16 tensors
    print(f"  Scratch (shared): {scratch / 1024 / 1024:.1f} MB")

    saving = fp16_total - fp8_total
    saving_pct = saving / fp16_total * 100
    print(f"\nSaving: {saving / 1024 / 1024:.1f} MB ({saving_pct:.0f}%)")

    # Verify with actual allocations
    print("\n--- Actual VRAM usage ---")

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    base_mem = torch.cuda.memory_allocated(device)

    # Allocate FP16 cache
    fp16_k = torch.zeros(num_layers, num_pages, page_size, num_kv_heads, head_dim,
                         dtype=torch.half, device=device)
    fp16_v = torch.zeros_like(fp16_k)
    fp16_actual = torch.cuda.memory_allocated(device) - base_mem
    print(f"FP16 actual: {fp16_actual / 1024 / 1024:.1f} MB")
    del fp16_k, fp16_v
    torch.cuda.empty_cache()

    base_mem = torch.cuda.memory_allocated(device)
    # Allocate FP8 cache
    fp8_k = torch.zeros(num_layers, num_pages, page_size, num_kv_heads, head_dim,
                        dtype=torch.float8_e4m3fn, device=device)
    fp8_v = torch.zeros_like(fp8_k)
    fp8_sk = torch.zeros(num_layers, num_pages, page_size, num_kv_heads,
                         dtype=torch.half, device=device)
    fp8_sv = torch.zeros_like(fp8_sk)
    fp8_actual = torch.cuda.memory_allocated(device) - base_mem
    print(f"FP8 actual:  {fp8_actual / 1024 / 1024:.1f} MB")
    del fp8_k, fp8_v, fp8_sk, fp8_sv

    actual_saving = fp16_actual - fp8_actual
    actual_pct = actual_saving / fp16_actual * 100
    print(f"Actual saving: {actual_saving / 1024 / 1024:.1f} MB ({actual_pct:.0f}%)")


if __name__ == "__main__":
    main()
