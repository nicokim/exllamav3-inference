"""OptimizedLLM: high-level async inference wrapper.

Provides async generate/stream with vision support.
Internally uses SlimGenerator.
"""

from __future__ import annotations

import asyncio
import functools
import io
import logging
import threading
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from pathlib import Path

from huggingface_hub import snapshot_download

if TYPE_CHECKING:
    import torch

    from exllamav3_opt.prefix_cache import PrefixCache

logger = logging.getLogger(__name__)


class LLMConfig:
    """Configuration for OptimizedLLM."""

    def __init__(
        self,
        model_repo: str = "",
        model_revision: str = "bpw3.0",
        max_new_tokens: int = 256,
        cache_size: int = 2048,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        min_p: float = 0.0,
    ) -> None:
        self.model_repo = model_repo
        self.model_revision = model_revision
        self.max_new_tokens = max_new_tokens
        self.cache_size = cache_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p


class OptimizedLLM:
    """High-level async inference wrapper using SlimGenerator.

    Prompt formatting uses the model's built-in chat template
    via ``tokenizer.hf_chat_template()``. Sampler and stop
    conditions are derived from config / model metadata.
    """

    def __init__(self, config: LLMConfig, hf_token: str = "") -> None:
        self._config = config
        self._hf_token = hf_token
        self._lock = asyncio.Lock()
        self._loaded = False

        # Set by load()
        self._model = None
        self._tokenizer = None
        self._generator = None  # SlimGenerator
        self._cache = None
        self._vision_model = None
        self._prefix_cache: PrefixCache | None = None
        self._stop_conditions: list[int | str] = []

    def download(self) -> str:
        """Download model from HuggingFace Hub. Returns local path."""
        logger.info(
            "Downloading model %s (rev: %s)",
            self._config.model_repo,
            self._config.model_revision,
        )
        path = snapshot_download(
            self._config.model_repo,
            revision=self._config.model_revision,
            token=self._hf_token or None,
        )
        logger.info("Model downloaded to %s", path)
        return path

    def load(self, model_path: str, enable_prefix_cache: bool = True) -> None:
        """Load ExLlamaV3 model, tokenizer, cache, and SlimGenerator."""
        from exllamav3 import Cache, Config, Model, Tokenizer

        from exllamav3_opt.generator import SlimGenerator
        from exllamav3_opt.prefix_cache import PrefixCache

        logger.info("Loading model from %s", model_path)

        config = Config.from_directory(model_path)
        self._model = Model.from_config(config)
        self._cache = Cache(self._model, max_num_tokens=self._config.cache_size)
        self._model.load(progressbar=True)

        self._tokenizer = Tokenizer.from_config(config)

        # Stop conditions: merge eos_token_id from config.json
        # and generation_config.json (which often has extra stop tokens
        # like <|im_end|> that config.json misses)
        self._stop_conditions = list(config.eos_token_id_list)
        gen_cfg_path = Path(model_path) / "generation_config.json"
        if gen_cfg_path.exists():
            import json

            with open(gen_cfg_path) as f:
                gen_cfg = json.load(f)
            gc_eos = gen_cfg.get("eos_token_id", [])
            if isinstance(gc_eos, int):
                gc_eos = [gc_eos]
            for tid in gc_eos:
                if tid not in self._stop_conditions:
                    self._stop_conditions.append(tid)

        # Prefix cache (optional)
        if enable_prefix_cache:
            self._prefix_cache = PrefixCache()

        # SlimGenerator replaces upstream Generator
        self._generator = SlimGenerator(
            self._model,
            self._cache,
            self._tokenizer,
            prefix_cache=self._prefix_cache,
        )

        # Vision model (separate component, fp16)
        try:
            self._vision_model = Model.from_config(config, component="vision")
            self._vision_model.load(progressbar=True)
            logger.info("Vision model loaded")
        except Exception:
            logger.debug("No vision component found, vision features disabled")
            self._vision_model = None

        self._loaded = True
        logger.info("Model loaded successfully (OptimizedLLM)")

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vision_model(self):
        return self._vision_model

    def _make_sampler(self):
        """Create sampler from config."""
        from exllamav3.generator.sampler import ComboSampler

        return ComboSampler(
            temperature=self._config.temperature,
            top_k=self._config.top_k,
            top_p=self._config.top_p,
            min_p=self._config.min_p,
        )

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        messages: list[dict[str, str]],
    ) -> torch.Tensor:
        """Build token IDs using the model's chat template.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.

        Returns:
            Token IDs tensor (1, seq_len) ready for generate/stream.
        """
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded")
        return self._tokenizer.hf_chat_template(
            messages, add_generation_prompt=True
        )

    # ------------------------------------------------------------------
    # Sync internals
    # ------------------------------------------------------------------

    def _generate_sync(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
        embeddings: list | None = None,
    ) -> str:
        """Synchronous full generation (non-streaming)."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        return self._generator.generate(
            input_ids=input_ids,
            stop_conditions=self._stop_conditions,
            sampler=self._make_sampler(),
            max_new_tokens=max_new_tokens or self._config.max_new_tokens,
            embeddings=embeddings,
        )

    def _stream_sync(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
        embeddings: list | None = None,
        cancel_flag: threading.Event | None = None,
    ):
        """Synchronous streaming generation. Yields token strings."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        for chunk in self._generator.stream_tokens(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens or self._config.max_new_tokens,
            sampler=self._make_sampler(),
            stop_conditions=self._stop_conditions,
            embeddings=embeddings,
            cancel_flag=cancel_flag,
        ):
            if chunk.text:
                yield chunk.text

    # ------------------------------------------------------------------
    # Async public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
        embeddings: list | None = None,
    ) -> str:
        """Async full generation with lock."""
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(
                None,
                functools.partial(
                    self._generate_sync,
                    input_ids,
                    max_new_tokens,
                    embeddings,
                ),
            )

    async def stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
        embeddings: list | None = None,
    ) -> AsyncIterator[str]:
        """Async streaming generation with lock.

        Yields token strings as they are generated.
        Lock is held for the entire stream duration.
        On CancelledError: signals the thread to stop via cancel_flag,
        waits for it to finish, then releases the lock.
        """
        cancel_flag = threading.Event()

        async with self._lock:
            loop = asyncio.get_running_loop()
            token_queue: asyncio.Queue[str | None] = asyncio.Queue()

            def _run():
                try:
                    for token in self._stream_sync(
                        input_ids, max_new_tokens, embeddings, cancel_flag
                    ):
                        loop.call_soon_threadsafe(token_queue.put_nowait, token)
                finally:
                    loop.call_soon_threadsafe(token_queue.put_nowait, None)

            future = loop.run_in_executor(None, _run)

            try:
                while True:
                    token = await token_queue.get()
                    if token is None:
                        break
                    yield token
            except GeneratorExit:
                cancel_flag.set()
            finally:
                await future

    # ------------------------------------------------------------------
    # Vision processing
    # ------------------------------------------------------------------

    def process_image(self, image_bytes: bytes) -> tuple[list, list[str]] | None:
        """Process image for vision input. Returns (embeddings, aliases) or None."""
        if self._vision_model is None:
            return None
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((512, 512))
        emb = self._vision_model.get_image_embeddings(self._tokenizer, image)
        return [emb], [emb.text_alias]

    def process_pil_images(self, images) -> tuple[list, list[str]] | None:
        """Process PIL Images for vision input. Returns (embeddings, aliases) or None."""
        if self._vision_model is None or not images:
            return None
        img = images[0].convert("RGB")
        img.thumbnail((512, 512))
        emb = self._vision_model.get_image_embeddings(self._tokenizer, img)
        return [emb], [emb.text_alias]
