# exllamav3-inference

[![CI](https://github.com/nicokim/exllamav3-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/nicokim/exllamav3-inference/actions/workflows/ci.yml)

Single-user optimized inference wrapper for [ExLlamaV3](https://github.com/turboderp-org/exllamav3). Replaces the upstream 1000+ line Generator with a minimal prefill-decode loop, custom CUDA kernels, and optional FP8 KV cache.

Tested with Qwen3.5-VL-27B @ 3.0bpw EXL3 on RTX 5090.

## Features

- **SlimGenerator**: direct prefill->decode loop, no job queue, no page table, no defrag
- **Fused RMSNorm+Residual**: CUDA kernel fusing `x += attn_out; y = rmsnorm(x)` into a single launch (36 fewer kernel launches per token)
- **Fused Sampling**: temperature + top-k + Gumbel noise + argmax in one kernel
- **FP8 KV Cache**: E4M3FN quantized KV storage (~50% VRAM savings)
- **PrefixCache**: snapshots system prompt KV + recurrent state (GatedDeltaNet) to CPU pinned memory
- **OptimizedLLM**: high-level async wrapper with chat template and vision support

## Requirements

- Python >= 3.13
- CUDA >= 12.8
- PyTorch >= 2.6
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (installed separately)

## Install

```bash
# Install exllamav3 first
pip install --no-build-isolation -e /path/to/exllamav3

# Install this package (compiles CUDA kernels)
MAX_JOBS=1 uv pip install --no-build-isolation -e .
```

## Usage

### OptimizedLLM (high-level API)

```python
import asyncio
from exllamav3_opt.integration import LLMConfig, OptimizedLLM

config = LLMConfig(
    max_new_tokens=256,
    cache_size=4096,
    temperature=0.6,
    top_k=20,
)
llm = OptimizedLLM(config)
llm.load("/path/to/model")

# build_prompt uses the model's chat template (via HF tokenizer)
ids = llm.build_prompt([
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"},
])

# Async generate
result = asyncio.run(llm.generate(ids))
print(result)

# Async streaming
async def main():
    async for token in await llm.stream(ids):
        print(token, end="", flush=True)

asyncio.run(main())
```

### SlimGenerator (low-level API)

```python
from exllamav3 import Cache, Config, Model, Tokenizer
from exllamav3_opt.generator import SlimGenerator

config = Config.from_directory("/path/to/model")
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=4096)
model.load()
tokenizer = Tokenizer.from_config(config)

gen = SlimGenerator(model, cache, tokenizer)

# Text prompt (encoded internally)
for chunk in gen.stream_tokens("Hello!", max_new_tokens=256):
    print(chunk.text, end="", flush=True)

# Or pass pre-tokenized input_ids directly
ids = tokenizer.hf_chat_template(
    [{"role": "user", "content": "Hello!"}],
    add_generation_prompt=True,
)
for chunk in gen.stream_tokens(input_ids=ids, max_new_tokens=256):
    print(chunk.text, end="", flush=True)
```

### FP8 KV Cache

```python
from exllamav3_opt.fp8_cache import CacheLayer_fp8

cache = Cache(model, max_num_tokens=8192, layer_type=CacheLayer_fp8)
gen = SlimGenerator(model, cache, tokenizer)
```

### Fused Sampling

```python
gen = SlimGenerator(
    model, cache, tokenizer,
    use_fused_sampling=True,
    fused_temperature=0.6,
    fused_top_k=20,
)
```

## Benchmark

RTX 5090, Qwen3.5-VL-27B @ 3.0bpw EXL3:

| Configuration | tok/s | vs upstream |
|---|---|---|
| Upstream Generator | 46.8 | baseline |
| SlimGenerator + fused norm | 48.3 (decode) | +3.2% |

## Project Structure

```
src/exllamav3_opt/
  _ext/                     # CUDA kernels
    fused_rmsnorm_residual.cu  # fused residual add + RMSNorm
    fused_sampling.cu          # fused temperature + top-k + Gumbel + argmax
    fp8_cache_kernels.cu       # FP8 E4M3FN quant/dequant
    bindings.cpp               # pybind11 bindings
  generator.py              # SlimGenerator + fused norm monkey-patch
  fp8_cache.py              # CacheLayer_fp8 (CacheLayer ABC)
  prefix_cache.py           # PrefixCache
  tensor_pool.py            # Pre-allocated tensor pool
  integration.py            # OptimizedLLM (async wrapper)
  compile.py                # torch.compile wrappers
```
