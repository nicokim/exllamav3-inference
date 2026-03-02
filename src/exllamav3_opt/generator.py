"""SlimGenerator: single-user, single-sequence inference loop.

Replaces ExLlamaV3's 1000+ line Generator with a direct loop:
  tokenize -> prefill -> sample first token from prefill logits ->
  while not eos: forward(1 token) -> sample -> yield

No job queue, no page table, no defrag, no ref counting, no logit mapping,
no filter pool, no draft model orchestration.
"""

from __future__ import annotations

import logging
import random
import threading
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from exllamav3.cache.cache import Cache
    from exllamav3.generator.sampler import Sampler
    from exllamav3.model.model import Model
    from exllamav3.tokenizer import MMEmbedding
    from exllamav3.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

# Constants (avoid importing exllamav3 at module level)
PAGE_SIZE = 256  # Must match exllamav3.constants.PAGE_SIZE


@dataclass
class StreamChunk:
    """A single token from the streaming generation."""

    text: str
    token_id: int
    eos: bool


class SlimGenerator:
    """Minimal generator optimised for single-user, single-sequence inference.

    Key simplifications over upstream Generator:
    - No dynamic batching / job queue — always batch=1
    - Pages pre-allocated sequentially (no page table, no defrag)
    - Tensors from TensorPool reused across calls (zero allocation in decode loop)
    - Supports stop conditions (token IDs + strings), sampler chain, MMEmbeddings
    - Cancellation via threading.Event
    """

    def __init__(
        self,
        model: Model,
        cache: Cache,
        tokenizer: Tokenizer,
        prefix_cache=None,
        use_fused_sampling: bool = False,
        fused_temperature: float = 1.0,
        fused_top_k: int = 0,
        compile_lm_head: bool = False,
        use_fused_norm: bool = True,
    ) -> None:
        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        self.prefix_cache = prefix_cache

        self.device = model.output_device

        # Model properties
        self.hidden_size = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        self.padded_vocab = ((self.vocab_size + 31) // 32) * 32

        # Cache properties
        self.max_num_tokens = cache.max_num_tokens
        self.max_pages = self.max_num_tokens // PAGE_SIZE

        # Model capabilities
        self.use_mrope = "mrope" in model.caps
        self.has_recurrent = model.caps.get("recurrent_states", False)

        # Initialize recurrent state (GatedDeltaNet layers in Qwen3.5 etc.)
        self._recurrent_state: dict | None = None
        if self.has_recurrent:
            rl = model.get_recurrent_layers()
            self._recurrent_state = {m.layer_idx: m.new_recurrent_state() for m in rl}

        # Pre-allocate tensor pool
        from exllamav3_opt.tensor_pool import TensorPool
        self.pool = TensorPool(
            device=self.device,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            max_pages=self.max_pages,
        )

        # Pre-allocate all pages sequentially (no dynamic paging)
        self.pool.setup_pages(self.max_pages)

        # Pre-computed token→text lookup (list index vs tokenizer.decode())
        self._id_to_piece: list[str] = tokenizer.get_id_to_piece_list()

        # Fused sampling: CUDA kernel replaces ComboSampler when only
        # temperature + top_k are needed (no top_p, min_p, etc.)
        self._use_fused_sampling = False
        if use_fused_sampling:
            try:
                from exllamav3_opt._ext import fused_sample as _fused_sample
                self._fused_sample = _fused_sample
                self._fused_temperature = fused_temperature
                self._fused_top_k = fused_top_k
                self._use_fused_sampling = True
                logger.info(
                    "Fused sampling enabled (temp=%.2f, top_k=%d)",
                    fused_temperature, fused_top_k,
                )
            except ImportError:
                logger.warning("Fused sampling requested but CUDA extension not compiled")

        # torch.compile the LM head for faster decode (opt-in due to warmup cost)
        if compile_lm_head:
            try:
                from exllamav3_opt.compile import compile_lm_head as _compile
                _compile(model)
            except Exception as e:
                logger.debug("LM head compile skipped: %s", e)

        # Fused RMSNorm+Residual: monkey-patch TransformerBlock.forward()
        # to fuse `x += attn_out; y = rmsnorm(x)` into a single kernel
        self._use_fused_norm = False
        if use_fused_norm:
            try:
                self._use_fused_norm = _patch_transformer_blocks(model)
            except Exception as e:
                logger.warning("Fused norm patch failed: %s", e)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        sampler: Sampler | None = None,
        stop_conditions: list[int | str] | None = None,
        add_bos: bool = False,
        encode_special_tokens: bool = True,
        completion_only: bool = True,
        embeddings: list[MMEmbedding] | None = None,
        **kwargs,
    ) -> str:
        """Non-streaming generation. Returns the full completion string."""
        parts = []
        for chunk in self.stream_tokens(
            prompt,
            max_new_tokens=max_new_tokens,
            sampler=sampler,
            stop_conditions=stop_conditions,
            add_bos=add_bos,
            encode_special_tokens=encode_special_tokens,
            embeddings=embeddings,
        ):
            parts.append(chunk.text)
        return "".join(parts)

    def stream_tokens(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        sampler: Sampler | None = None,
        stop_conditions: list[int | str] | None = None,
        add_bos: bool = False,
        encode_special_tokens: bool = True,
        embeddings: list[MMEmbedding] | None = None,
        cancel_flag: threading.Event | None = None,
        seed: int | None = None,
    ) -> Generator[StreamChunk, None, None]:
        """Core streaming generation loop.

        Flow:
        1. Encode prompt
        2. Check prefix cache for system prompt hit
        3. Prefill remaining tokens -> get logits for last position
        4. Sample first token from prefill logits
        5. Decode loop: forward(1 token) -> sample -> yield
        """
        if sampler is None:
            from exllamav3.generator.sampler.presets import DefaultSampler
            sampler = DefaultSampler()

        stop_tokens, stop_strings = _parse_stop_conditions(stop_conditions)

        # Encode prompt
        input_ids = self.tokenizer.encode(
            prompt,
            encode_special_tokens=encode_special_tokens,
            add_bos=add_bos,
            embeddings=embeddings,
        )
        seq_len = input_ids.shape[-1]

        # Check prefix cache
        cached_len = 0
        if self.prefix_cache is not None:
            cached_len = self.prefix_cache.get_cached_length(input_ids)
            if cached_len > 0:
                self.prefix_cache.restore_to_cache(self.cache)
                logger.debug("Prefix cache hit: %d tokens", cached_len)

        # Reset recurrent state for new generation
        if self.has_recurrent:
            rl = self.model.get_recurrent_layers()
            self._recurrent_state = {m.layer_idx: m.new_recurrent_state() for m in rl}

        # RNG for sampling
        rng_val = seed if seed is not None else random.getrandbits(32)

        # --- PREFILL ---
        prefill_ids = input_ids[:, cached_len:]
        prefill_len = prefill_ids.shape[-1]

        if prefill_len == 0:
            raise RuntimeError("Full input was cached — no tokens to prefill")

        num_pages_needed = (cached_len + prefill_len + PAGE_SIZE - 1) // PAGE_SIZE
        block_table = self.pool.get_block_table(num_pages_needed)
        self.pool.set_cache_seqlen(cached_len)

        prefill_ids_gpu = self.pool.get_prefill_ids(prefill_ids)

        params = self._build_params(block_table, embeddings)
        params["last_tokens_only"] = 1
        prefill_logits = self.model.forward(input_ids=prefill_ids_gpu, params=params)

        kv_position = seq_len

        # Capture prefix cache on first run
        if self.prefix_cache is not None and cached_len == 0:
            self.prefix_cache.capture(input_ids, self.cache, kv_position)

        # --- SAMPLE FIRST TOKEN FROM PREFILL LOGITS ---
        logits = prefill_logits[:, -1:, :self.vocab_size]
        token_id = self._sample(logits, sampler, rng_val)
        rng_val = (rng_val + 1) & 0xFFFFFFFF

        # Check stop on first token
        if token_id in stop_tokens:
            yield StreamChunk(text="", token_id=token_id, eos=True)
            return

        # Decode token to text and check stop strings
        id_to_piece = self._id_to_piece
        text = id_to_piece[token_id]
        text_buffer = text
        eos, text_out, text_buffer = _check_stop_strings(text_buffer, stop_strings)

        if text_out:
            yield StreamChunk(text=text_out, token_id=token_id, eos=eos)
        if eos:
            return

        # --- DECODE LOOP ---
        # Pre-allocate params dict once; only block_table changes (at page boundaries)
        # cache_seqlens is a tensor updated in-place via pool.set_cache_seqlen()
        current_num_pages = (kv_position + 1 + PAGE_SIZE - 1) // PAGE_SIZE
        decode_block_table = self.pool.get_block_table(current_num_pages)
        decode_params: dict = {
            "attn_mode": "flashinfer",
            "block_table": decode_block_table,
            "cache": self.cache,
            "cache_seqlens": self.pool.cache_seqlens,
        }
        if self.use_mrope:
            decode_params["positions"] = self.pool.cache_seqlens.clone()
        if self._recurrent_state is not None:
            decode_params["recurrent_states"] = self._recurrent_state

        pool_input_ids = self.pool.input_ids

        for _ in range(1, max_new_tokens):
            if cancel_flag is not None and cancel_flag.is_set():
                if text_buffer:
                    yield StreamChunk(text=text_buffer, token_id=token_id, eos=True)
                return

            # Update block_table only when crossing a page boundary
            new_num_pages = (kv_position + 1 + PAGE_SIZE - 1) // PAGE_SIZE
            if new_num_pages != current_num_pages:
                current_num_pages = new_num_pages
                decode_block_table = self.pool.get_block_table(current_num_pages)
                decode_params["block_table"] = decode_block_table

            self.pool.set_cache_seqlen(kv_position)
            self.pool.set_input_id(token_id)

            # Forward single token
            decode_logits = self.model.forward(
                input_ids=pool_input_ids, params=decode_params
            )
            kv_position += 1

            # Sample
            logits = decode_logits[:, 0:1, :self.vocab_size]
            token_id = self._sample(logits, sampler, rng_val)
            rng_val = (rng_val + 1) & 0xFFFFFFFF

            # Check stop token
            if token_id in stop_tokens:
                if text_buffer:
                    yield StreamChunk(text=text_buffer, token_id=0, eos=False)
                yield StreamChunk(text="", token_id=token_id, eos=True)
                return

            # Decode and check stop strings
            text = id_to_piece[token_id]
            text_buffer += text
            eos, text_out, text_buffer = _check_stop_strings(text_buffer, stop_strings)

            if text_out:
                yield StreamChunk(text=text_out, token_id=token_id, eos=eos)
            if eos:
                return

        # Max tokens reached — flush buffer
        if text_buffer:
            yield StreamChunk(text=text_buffer, token_id=token_id, eos=True)
        else:
            yield StreamChunk(text="", token_id=token_id, eos=True)

    def _build_params(
        self,
        block_table: torch.Tensor,
        embeddings: list[MMEmbedding] | None = None,
    ) -> dict:
        """Build the params dict for model.forward() (used for prefill)."""
        params = {
            "attn_mode": "flashinfer",
            "block_table": block_table,
            "cache": self.cache,
            "cache_seqlens": self.pool.cache_seqlens,
        }
        if embeddings:
            params["indexed_embeddings"] = embeddings
        if self.use_mrope:
            params["positions"] = self.pool.cache_seqlens.clone()
        if self._recurrent_state is not None:
            params["recurrent_states"] = self._recurrent_state
        return params

    def _sample(self, logits: torch.Tensor, sampler: Sampler, rng_val: int) -> int:
        """Sample a token from logits. Uses fused kernel when available."""
        if self._use_fused_sampling:
            # Fused CUDA kernel: temperature + top-k + Gumbel + argmax in 1 launch
            self._fused_sample(
                logits.view(1, -1),
                self.pool.sample_output,
                self._fused_temperature,
                self._fused_top_k,
                rng_val,
            )
            return self.pool.sample_output.item()
        next_token = sampler.forward(logits, rand_u32=rng_val)
        return next_token.item()

    def reset(self) -> None:
        """Reset state for a new generation (clear KV cache positions)."""
        self.pool.set_cache_seqlen(0)
        if self.has_recurrent:
            rl = self.model.get_recurrent_layers()
            self._recurrent_state = {m.layer_idx: m.new_recurrent_state() for m in rl}


def _parse_stop_conditions(
    conditions: list[int | str] | None,
) -> tuple[set[int], set[str]]:
    """Split stop conditions into token IDs and strings."""
    stop_tokens: set[int] = set()
    stop_strings: set[str] = set()
    if conditions:
        for c in conditions:
            if isinstance(c, int):
                stop_tokens.add(c)
            elif isinstance(c, str):
                stop_strings.add(c)
    return stop_tokens, stop_strings


def _check_stop_strings(
    buffer: str, stop_strings: set[str]
) -> tuple[bool, str, str]:
    """Check if buffer contains any stop string.

    Returns (eos, text_to_yield, remaining_buffer).

    Holds back characters at the end that could be the start of a stop string.
    """
    if not stop_strings:
        return False, buffer, ""

    # Check for complete stop string matches
    for s in stop_strings:
        idx = buffer.find(s)
        if idx != -1:
            return True, buffer[:idx], ""

    # Hold back characters that could be a partial match
    max_hold = max(len(s) for s in stop_strings) - 1
    if max_hold <= 0:
        return False, buffer, ""

    if len(buffer) <= max_hold:
        return False, "", buffer

    return False, buffer[:-max_hold], buffer[-max_hold:]


def _patch_transformer_blocks(model: Model) -> bool:
    """Monkey-patch TransformerBlock.forward() to fuse residual add + RMSNorm.

    Replaces the pattern ``x += y; y = mlp_norm(x)`` (2 kernel launches)
    with a single fused_rmsnorm_residual kernel call per layer.
    36 layers × 1 fusion = 36 fewer kernel launches per token.

    Returns True if the patch was applied.
    """
    from exllamav3.modules.transformer import TransformerBlock
    from exllamav3.util.tensor import to2 as _to2

    from exllamav3_opt._ext import fused_rmsnorm_residual as _fused_kern

    # Set per-instance flag: can this block's mlp_norm be fused?
    patched = 0
    for module in model.modules:
        if not isinstance(module, TransformerBlock):
            continue
        norm = module.mlp_norm
        module._can_fuse_norm = (
            module.mlp is not None
            and norm is not None
            and hasattr(norm, "rms_norm_eps")
            and not getattr(norm, "span_heads", False)
            and not getattr(norm, "unweighted", False)
            and norm.weight is not None
        )
        if module._can_fuse_norm:
            patched += 1

    if patched == 0:
        logger.info("No fuseable TransformerBlocks found — skipping patch")
        return False

    # Replace forward at class level (all instances share it)
    def _patched_forward(self, x, params, out_dtype=None):
        if self.attn:
            if self.attn_norm:
                y = self.attn_norm.forward(x, params, out_dtype=torch.half)
            else:
                y = x.half()
            y = self.attn.forward(y, params)
            if params.get("prefill"):
                return x
            if self.attn_post_norm:
                y = self.attn_post_norm.forward(y, params)

            # Fused path: x += y; y = (w + bias) * rmsnorm(x)
            if self._can_fuse_norm and x.dtype == torch.half:
                dim = x.shape[-1]
                _fused_kern(
                    x.view(-1, dim), y.view(-1, dim),
                    self.mlp_norm.weight.data,
                    y.view(-1, dim),
                    self.mlp_norm.rms_norm_eps,
                    self.mlp_norm.constant_bias,
                )
                y = y.view_as(x)
            else:
                x += y

        if self.mlp:
            _fused = self.attn and self._can_fuse_norm and x.dtype == torch.half
            if not _fused:
                if self.mlp_norm:
                    y = self.mlp_norm.forward(x, params, out_dtype=torch.half)
                else:
                    y = x.half()
            y = self.mlp.forward(y, params)
            if self.mlp_post_norm:
                y = self.mlp_post_norm.forward(y, params)
            x += y

        return _to2(x, out_dtype, self.out_dtype)

    TransformerBlock.forward = _patched_forward
    logger.info("Fused RMSNorm+Residual patch applied to %d/%d blocks", patched, len(model.modules))
    return True
