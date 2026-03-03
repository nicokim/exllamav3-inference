"""ExLlamaV3-Optimized: single-user inference wrapper for ExLlamaV3.

Lazy imports: modules that depend on exllamav3 are imported on first access
so the package can be installed without exllamav3 for development/testing.
"""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy import for modules that depend on exllamav3."""
    _lazy = {
        "CacheLayer_fp8": ("exllamav3_opt.fp8_cache", "CacheLayer_fp8"),
        "LLMConfig": ("exllamav3_opt.integration", "LLMConfig"),
        "OptimizedLLM": ("exllamav3_opt.integration", "OptimizedLLM"),
        "PrefixCache": ("exllamav3_opt.prefix_cache", "PrefixCache"),
        "SlimGenerator": ("exllamav3_opt.generator", "SlimGenerator"),
        "StreamChunk": ("exllamav3_opt.generator", "StreamChunk"),
        "TensorPool": ("exllamav3_opt.tensor_pool", "TensorPool"),
        "compile_components": ("exllamav3_opt.compile", "compile_components"),
    }

    if name in _lazy:
        module_path, attr = _lazy[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr)

    raise AttributeError(f"module 'exllamav3_opt' has no attribute {name!r}")


__all__ = [
    "CacheLayer_fp8",
    "LLMConfig",
    "OptimizedLLM",
    "PrefixCache",
    "SlimGenerator",
    "StreamChunk",
    "TensorPool",
    "compile_components",
]
