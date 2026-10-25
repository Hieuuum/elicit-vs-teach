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


def em_store() -> Path:
    """$GEODE_STORE/em, or a clear stop when the variable is unset (a relative path would be wrong)."""
    s = os.environ.get("GEODE_STORE")
    if not s:
        raise SystemExit("[em] GEODE_STORE is not set: export GEODE_STORE=<the geode store> first")
    return Path(s) / "em"


def load_causal_lm(model_id: str, dtype, device: str):
    """AutoModelForCausalLM across transformers versions: newer ones take ``dtype``, older ones only
    ``torch_dtype`` (the cluster runs the newer, a laptop may run the older)."""
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    return model.to(device)
