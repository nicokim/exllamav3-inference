"""Drop-in OptimizedLLM replacement for kohai-v2's LLM class.

Same API contract: async generate/stream, process_image/video/pil_images,
strategy property. Internally uses SlimGenerator instead of upstream Generator.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from huggingface_hub import snapshot_download

if TYPE_CHECKING:
    from exllamav3_opt.prefix_cache import PrefixCache

logger = logging.getLogger(__name__)


class LLMConfig:
    """Minimal config matching kohai-v2's LLMConfig fields used by LLM."""

    def __init__(
        self,
        model_repo: str = "kohai-channel/kohai-vl-27b-v2-EXL3",
        model_revision: str = "bpw3.0",
        max_new_tokens: int = 256,
        cache_size: int = 2048,
        system_prompt_path: str = "prompts/system.txt",
        **kwargs,
    ) -> None:
        self.model_repo = model_repo
        self.model_revision = model_revision
        self.max_new_tokens = max_new_tokens
        self.cache_size = cache_size
        self.system_prompt_path = system_prompt_path
        # Store extra fields (temperature, top_p, top_k, etc.)
        for k, v in kwargs.items():
            setattr(self, k, v)


class OptimizedLLM:
    """Drop-in replacement for kohai-v2's LLM using SlimGenerator.

    API-compatible: same async generate/stream, process_image/video/pil_images,
    and strategy property.
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
        self._strategy = None
        self._prefix_cache: PrefixCache | None = None

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
            logger.warning("No vision component found, vision features disabled")
            self._vision_model = None

        # Strategy detection (lazy import to match kohai-v2 pattern)
        try:
            from kohai.brain.strategy import detect_strategy

            self._strategy = detect_strategy(self._config.model_repo)
        except ImportError:
            # Running standalone — use a default strategy
            logger.warning("kohai.brain.strategy not found, using built-in Qwen35VL strategy")
            self._strategy = _FallbackQwen35VLStrategy()

        self._loaded = True
        logger.info("Model loaded successfully (OptimizedLLM)")

    @property
    def strategy(self):
        return self._strategy

    # ------------------------------------------------------------------
    # Sync internals
    # ------------------------------------------------------------------

    def _generate_sync(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        embeddings: list | None = None,
    ) -> str:
        """Synchronous full generation (non-streaming)."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        sampler = self._strategy.get_sampler()
        stop = self._strategy.get_stop_conditions(self._tokenizer)

        return self._generator.generate(
            prompt=prompt,
            stop_conditions=stop,
            sampler=sampler,
            max_new_tokens=max_new_tokens or self._config.max_new_tokens,
            add_bos=self._strategy.get_add_bos(),
            encode_special_tokens=True,
            completion_only=True,
            embeddings=embeddings,
        )

    def _stream_sync(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        embeddings: list | None = None,
        cancel_flag: threading.Event | None = None,
    ):
        """Synchronous streaming generation. Yields token strings."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        sampler = self._strategy.get_sampler()
        stop = self._strategy.get_stop_conditions(self._tokenizer)

        for chunk in self._generator.stream_tokens(
            prompt,
            max_new_tokens=max_new_tokens or self._config.max_new_tokens,
            sampler=sampler,
            stop_conditions=stop,
            add_bos=self._strategy.get_add_bos(),
            encode_special_tokens=True,
            embeddings=embeddings,
            cancel_flag=cancel_flag,
        ):
            if chunk.text:
                yield chunk.text

    # ------------------------------------------------------------------
    # Async public API (matches kohai-v2's LLM exactly)
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
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
                    prompt,
                    max_new_tokens,
                    embeddings,
                ),
            )

    async def stream(
        self,
        prompt: str,
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
                        prompt, max_new_tokens, embeddings, cancel_flag
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
    # Vision processing (delegates to strategy, same as kohai-v2)
    # ------------------------------------------------------------------

    def process_image(self, image_bytes: bytes) -> tuple[list, list[str]] | None:
        """Process image for vision input. Returns (embeddings, aliases) or None."""
        if self._vision_model is None or self._strategy is None:
            return None
        return self._strategy.process_image(
            self._vision_model, self._tokenizer, image_bytes
        )

    def process_video(self, video_bytes: bytes) -> tuple[list, list[str]] | None:
        """Process video frames for vision input. Returns (embeddings, aliases) or None."""
        if self._vision_model is None or self._strategy is None:
            return None
        return self._strategy.process_video(
            self._vision_model, self._tokenizer, video_bytes
        )

    def process_pil_images(self, images) -> tuple[list, list[str]] | None:
        """Process PIL Images for vision input (used by screen capture)."""
        if self._vision_model is None or self._strategy is None:
            return None
        return self._strategy.process_pil_images(
            self._vision_model, self._tokenizer, images
        )


class _FallbackQwen35VLStrategy:
    """Minimal Qwen3.5-VL strategy for standalone use (no kohai dependency)."""

    def build_prompt(self, user_text, system_prompt="", history=None, embedding_aliases=None):
        parts = []
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
        if history:
            for msg in history:
                parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        user_content = (
            "".join(embedding_aliases) + user_text if embedding_aliases else user_text
        )
        parts.append(f"<|im_start|>user\n{user_content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def get_sampler(self):
        from exllamav3.generator.sampler import ComboSampler

        return ComboSampler(temperature=0.6, top_p=0.95, top_k=20)

    def get_stop_conditions(self, tokenizer):
        return [tokenizer.eos_token_id, "<|im_end|>"]

    def get_add_bos(self):
        return True

    def get_cache_size(self):
        return 8192

    def process_image(self, vision_model, tokenizer, image_bytes):
        import io

        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((512, 512))
        embeddings = vision_model.get_image_embeddings(tokenizer, image)
        return [embeddings], [embeddings.text_alias]

    def process_video(self, vision_model, tokenizer, video_bytes):
        return None  # Requires cv2 — import from kohai strategy if needed

    def process_pil_images(self, vision_model, tokenizer, images):
        if not images:
            return None
        img = images[0].convert("RGB")
        img.thumbnail((512, 512))
        embeddings = vision_model.get_image_embeddings(tokenizer, img)
        return [embeddings], [embeddings.text_alias]
