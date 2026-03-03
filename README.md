# exllamav3-inference

[![CI](https://github.com/nicokim/exllamav3-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/nicokim/exllamav3-inference/actions/workflows/ci.yml)

Self-contained optimized inference for [ExLlamaV3](https://github.com/turboderp-org/exllamav3) models. Vendorizes the ExLlamaV3 runtime (from [lesj0610/exllamav3](https://github.com/lesj0610/exllamav3) `feat/flashinfer-backend-migration`) so there is no external dependency — a single `pip install` compiles everything.

Replaces the upstream 1000+ line Generator with a minimal prefill-decode loop, custom CUDA kernels, and optional FP8 KV cache.

## Features

- **Vendorized ExLlamaV3**: all inference modules included — no separate install needed
- **SlimGenerator**: direct prefill->decode loop, no job queue, no page table, no defrag
- **Fused RMSNorm+Residual**: CUDA kernel fusing `x += attn_out; y = rmsnorm(x)` into a single launch (36 fewer kernel launches per token)
- **Fused Sampling**: temperature + top-k + Gumbel noise + argmax in one kernel
- **FP8 KV Cache**: E4M3FN quantized KV storage (~50% VRAM savings)
- **PrefixCache**: snapshots system prompt KV + recurrent state (GatedDeltaNet) to CPU pinned memory
- **OptimizedLLM**: high-level async wrapper with chat template and vision support
- **Vision + Video**: multimodal inference with image and multi-frame video embeddings

## Requirements

- Python >= 3.13
- CUDA >= 12.8
- PyTorch >= 2.6

## Install

```bash
# Single install — compiles both exllamav3_ext and exllamav3_opt_ext CUDA kernels
MAX_JOBS=1 uv pip install --no-build-isolation -e .

# Optional: vision support
uv pip install pillow
```

## Usage

### OptimizedLLM (high-level API)

```python
import asyncio
from exllamav3_opt.integration import LLMConfig, OptimizedLLM

config = LLMConfig(
    model_repo="your-org/your-model-EXL3",
    model_revision="bpw3.0",
    max_new_tokens=256,
    cache_size=4096,
    temperature=0.6,
    top_k=20,
)
llm = OptimizedLLM(config, hf_token="...")

model_path = llm.download()
llm.load(model_path)

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

### Vision / Video

```python
from PIL import Image

# Load vision model
vision_model = Model.from_config(config, component="vision")
vision_model.load()

# Single image
image = Image.open("photo.jpg").convert("RGB")
image.thumbnail((512, 512))
emb = vision_model.get_image_embeddings(tokenizer, image)

# Build prompt with embedding alias
prompt = f"<|im_start|>user\n{emb.text_alias}Describe this image<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer.encode(prompt, add_bos=True, encode_special_tokens=True, embeddings=[emb])

response = gen.generate(input_ids=input_ids, embeddings=[emb], max_new_tokens=256)

# Video (multiple frames)
frames = [Image.open(f"frame_{i}.jpg").convert("RGB") for i in range(4)]
frame_embs = vision_model.get_image_embeddings(tokenizer, frames)  # returns list
aliases = "".join(e.text_alias for e in frame_embs)
prompt = f"<|im_start|>user\n{aliases}What happens in this video?<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer.encode(prompt, add_bos=True, encode_special_tokens=True, embeddings=frame_embs)
response = gen.generate(input_ids=input_ids, embeddings=frame_embs, max_new_tokens=300)
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

## Project Structure

```
src/exllamav3/                # Vendorized ExLlamaV3 runtime (inference-only subset)
  architecture/               # ~43 model architectures (Qwen3.5, Llama, DeepSeek, etc.)
  modules/                    # Attention, MLP, GatedDeltaNet, linear, norms
  cache/                      # KV cache layers (fp16, quant, recurrent)
  model/                      # Model loading and config
  tokenizer/                  # Tokenizer + MMEmbedding (multimodal)
  exllamav3_ext/              # CUDA kernels (~52 sources)
  util/                       # RoPE, progress, tensors, vision

src/exllamav3_opt/            # Optimized inference layer
  _ext/                       # Custom CUDA kernels
    fused_rmsnorm_residual.cu # Fused residual add + RMSNorm
    fused_sampling.cu         # Fused temperature + top-k + Gumbel + argmax
    fp8_cache_kernels.cu      # FP8 E4M3FN quant/dequant
    bindings.cpp              # pybind11 bindings
  generator.py                # SlimGenerator + fused norm monkey-patch
  fp8_cache.py                # CacheLayer_fp8 (CacheLayer ABC)
  prefix_cache.py             # PrefixCache
  tensor_pool.py              # Pre-allocated tensor pool
  integration.py              # OptimizedLLM (async wrapper)
  compile.py                  # torch.compile wrappers
```
