"""Build CUDA extensions for exllamav3-optimized.

Auto-discovers .cu/.cpp files in src/exllamav3_opt/_ext/ and compiles them
as a single torch CUDA extension. Falls back to a pure-Python install if
torch or CUDA are unavailable (kernels will raise ImportError at runtime).
"""

import os
from pathlib import Path

from setuptools import setup

EXT_DIR = Path("src/exllamav3_opt/_ext")

# Upstream exllamav3_ext include path for reusing util.cuh, reduction.cuh, etc.
UPSTREAM_EXT = os.environ.get(
    "EXLLAMAV3_EXT_INCLUDE",
    str(Path.home() / "Projects/kohai-v2/exllamav3/exllamav3/exllamav3_ext"),
)


def _get_ext_modules():
    """Build CUDAExtension if torch and CUDA sources exist."""
    try:
        from torch.utils.cpp_extension import CUDAExtension
    except ImportError:
        return []

    if os.environ.get("EXLLAMAV3_OPT_NOCOMPILE"):
        return []

    cuda_sources = sorted(str(p) for p in EXT_DIR.glob("*.cu"))
    cpp_sources = sorted(str(p) for p in EXT_DIR.glob("*.cpp"))
    sources = cpp_sources + cuda_sources

    if not sources:
        return []

    include_dirs = [str(EXT_DIR)]
    if os.path.isdir(UPSTREAM_EXT):
        include_dirs.append(UPSTREAM_EXT)

    nvcc_flags = [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "-std=c++17",
    ]

    cxx_flags = [
        "-O3",
        "-std=c++17",
    ]

    return [
        CUDAExtension(
            name="exllamav3_opt._ext.exllamav3_opt_ext",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "nvcc": nvcc_flags,
                "cxx": cxx_flags,
            },
        )
    ]


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
