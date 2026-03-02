"""CUDA extension bindings.

Imports the compiled extension if available, otherwise stubs raise ImportError
at call time so the rest of the library works without custom kernels.
"""

try:
    from exllamav3_opt._ext.exllamav3_opt_ext import (  # type: ignore[import-not-found]
        dequant_fp8_cache_paged,
        fused_rmsnorm_residual,
        fused_sample,
        quant_fp8_cache_paged,
    )
except ImportError:
    _MSG = (
        "exllamav3_opt CUDA extension not compiled. "
        "Run: pip install --no-build-isolation -e . "
        "from the exllamav3-optimized directory."
    )

    def fused_rmsnorm_residual(*args, **kwargs):
        raise ImportError(_MSG)

    def fused_sample(*args, **kwargs):
        raise ImportError(_MSG)

    def quant_fp8_cache_paged(*args, **kwargs):
        raise ImportError(_MSG)

    def dequant_fp8_cache_paged(*args, **kwargs):
        raise ImportError(_MSG)


__all__ = [
    "dequant_fp8_cache_paged",
    "fused_rmsnorm_residual",
    "fused_sample",
    "quant_fp8_cache_paged",
]
