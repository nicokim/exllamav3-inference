"""Build CUDA extensions for exllamav3-inference.

Compiles two CUDA extensions:
  1. exllamav3_ext — vendored exllamav3 core kernels (~52 sources)
  2. exllamav3_opt._ext.exllamav3_opt_ext — our custom inference kernels (3 sources)

Falls back to a pure-Python install if torch or CUDA are unavailable
(kernels will raise ImportError at runtime).
"""

import os
from pathlib import Path

from setuptools import setup

EXLLAMAV3_EXT_DIR = Path("src/exllamav3/exllamav3_ext")
OPT_EXT_DIR = Path("src/exllamav3_opt/_ext")


def _collect_sources(root: Path) -> list[str]:
    """Recursively collect .c, .cpp, .cu files under *root*."""
    exts = {".c", ".cpp", ".cu"}
    return sorted(
        str(p) for p in root.rglob("*") if p.suffix in exts
    )


def _get_ext_modules():
    """Build CUDAExtensions if torch and CUDA sources exist."""
    try:
        from torch.utils.cpp_extension import CUDAExtension
    except ImportError:
        return []

    if os.environ.get("EXLLAMAV3_OPT_NOCOMPILE"):
        return []

    extensions = []

    # --- exllamav3_ext (vendored upstream kernels) ---
    exl3_sources = _collect_sources(EXLLAMAV3_EXT_DIR)
    if exl3_sources:
        extensions.append(
            CUDAExtension(
                name="exllamav3_ext",
                sources=exl3_sources,
                include_dirs=[str(EXLLAMAV3_EXT_DIR)],
                extra_compile_args={
                    "nvcc": ["-O3", "--use_fast_math", "-lineinfo", "-std=c++17"],
                    "cxx": ["-O3", "-std=c++17"],
                },
            )
        )

    # --- exllamav3_opt_ext (our custom kernels) ---
    opt_sources = _collect_sources(OPT_EXT_DIR)
    if opt_sources:
        extensions.append(
            CUDAExtension(
                name="exllamav3_opt._ext.exllamav3_opt_ext",
                sources=opt_sources,
                include_dirs=[str(OPT_EXT_DIR)],
                extra_compile_args={
                    "nvcc": ["-O3", "--use_fast_math", "-lineinfo", "-std=c++17"],
                    "cxx": ["-O3", "-std=c++17"],
                },
            )
        )

    return extensions


def _get_cmdclass():
    try:
        from torch.utils.cpp_extension import BuildExtension

        return {"build_ext": BuildExtension}
    except ImportError:
        return {}


setup(
    ext_modules=_get_ext_modules(),
    cmdclass=_get_cmdclass(),
)
