"""biocast — parametric geometry generation and MICP success estimation for
bio-cemented construction-waste casts.

Importing this package preloads libspatialindex, because `rtree` (used by
trimesh for the polygon enclosure tree during plane slicing) resolves its
native library relative to `sys.prefix`, which is not the real environment root
in this sandbox. Without the preload, `mesh.slice_plane` fails with
"Could not load libspatialindex_c library".
"""
from __future__ import annotations

import ctypes as _ctypes
import os as _os
from pathlib import Path as _Path


def _preload_spatialindex() -> bool:
    prefix = _os.environ.get("CONDA_PREFIX")
    if not prefix:
        return False
    libdir = _Path(prefix) / "lib"
    candidates = ["libspatialindex.so.8", "libspatialindex.so", "libspatialindex_c.so.8",
                  "libspatialindex_c.so"]
    loaded = False
    for name in candidates:
        p = libdir / name
        if p.exists():
            try:
                _ctypes.CDLL(str(p), mode=_ctypes.RTLD_GLOBAL)
                loaded = True
            except OSError:
                pass
    c_lib = libdir / "libspatialindex_c.so"
    if c_lib.exists():
        _os.environ.setdefault("SPATIALINDEX_C_LIBRARY", str(c_lib))
    return loaded


_preload_spatialindex()

__all__ = ["params", "constraints", "grammars", "physics", "score"]
