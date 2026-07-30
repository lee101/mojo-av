from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "capi.mojo"
LIB = Path(os.environ.get("MOJOAV_LIB", ROOT / "dist" / "libmojo-av.so"))

I = ctypes.c_int64
_SIGNATURES = {
    "mav_rescale_many": ([I] * 8, I),
    "mav_durations_from_pts": ([I] * 4, I),
    "mav_scan_ivf": ([I] * 7, I),
    "mav_fixed_packets": ([I] * 7, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJOAV_LIB"):
        if LIB.exists():
            return str(LIB)
        raise BuildError(f"MOJOAV_LIB does not exist: {LIB}")
    if not force and LIB.exists() and LIB.stat().st_mtime >= SRC.stat().st_mtime:
        return str(LIB)
    mojo = shutil.which("mojo")
    if not mojo:
        raise BuildError("mojo not found; run inside `pixi run` or set MOJOAV_LIB")
    LIB.parent.mkdir(parents=True, exist_ok=True)
    command = [mojo, "build", "--emit", "shared-lib", str(SRC), "-o", str(LIB)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode or not LIB.exists():
        raise BuildError((result.stderr or result.stdout).strip()[:4000])
    return str(LIB)


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
