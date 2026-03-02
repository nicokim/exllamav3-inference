"""torch.compile wrappers for specific model components.

Targets:
- Vision model (SigLIP): fullgraph=False due to dynamic shapes from grid_thw
- LM head projection: fullgraph=True, fixed shapes during decode (batch=1, seq=1)

Does NOT compile transformer blocks — they use custom CUDA kernels that are
already optimized (ExL3 quantized matmul, FlashInfer attention, etc.).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def compile_vision_model(vision_model) -> None:
    """Apply torch.compile to the vision model forward.

    SigLIP has head_dim=72 which requires sdpa_nc. torch.compile with
    fullgraph=False handles the dynamic grid_thw shapes.
    """
    if vision_model is None:
        return

    try:
        vision_model.forward = torch.compile(
            vision_model.forward,
            mode="max-autotune",
            fullgraph=False,
        )
        logger.info("Vision model compiled with torch.compile (max-autotune)")
    except Exception as e:
        logger.warning("Failed to compile vision model: %s", e)


def compile_lm_head(model) -> None:
    """Apply torch.compile to the LM head projection.

    During decode, shapes are fixed (1, 1, hidden_size) -> (1, 1, vocab_size),
    so dynamic=False is optimal. Uses reduce-overhead mode to avoid the
    30-60s autotuning warmup of max-autotune.

    The head module is at model.modules[model.logit_layer_idx].
    """
    idx = getattr(model, "logit_layer_idx", None)
    if idx is None:
        return

    head = model.modules[idx]

    try:
        head.forward = torch.compile(
            head.forward,
            mode="reduce-overhead",
            dynamic=False,
        )
        logger.info("LM head compiled with torch.compile (reduce-overhead, static)")
    except Exception as e:
        logger.warning("Failed to compile LM head: %s", e)


def compile_components(model, vision_model=None) -> None:
    """Compile all applicable components."""
    compile_lm_head(model)
    if vision_model is not None:
        compile_vision_model(vision_model)
