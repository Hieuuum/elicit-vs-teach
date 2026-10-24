"""Load the environment's own libstdc++ before torch / transformers import optree.

On the cluster the system libstdc++ lacks CXXABI_1.3.15, which optree's extension needs; the usual fix
is LD_LIBRARY_PATH=$CONDA_PREFIX/lib in the shell. Preloading the environment's copy by its soname
does the same from inside the process: the loader then reuses it for every later extension."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


def preload_libstdcxx() -> str | None:
    for root in (os.environ.get("CONDA_PREFIX"), sys.prefix):
        if root and (lib := Path(root) / "lib" / "libstdc++.so.6").is_file():
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                return str(lib)
            except OSError:
                continue
    return None
