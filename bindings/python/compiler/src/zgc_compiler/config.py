from __future__ import annotations

import os
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Literal

from zgc import ABI_VERSION

OptimizeMode = Literal["Debug", "ReleaseSafe", "ReleaseFast", "ReleaseSmall"]
SelectionTarget = Literal["balanced", "memory", "fast", "small"]
ZigSelection = str | PathLike[str]

SUPPORTED_ZIG_VERSION = "0.17.0"
SUPPORTED_ABI_VERSION = ABI_VERSION


def default_cache_dir() -> Path:
    return Path(os.environ.get("ZGC_CACHE_DIR", ".zgc-cache"))


@dataclass(frozen=True)
class CompilerConfig:
    """Configuration for external executable compilation.

    ``zig`` accepts an explicit executable path or ``"auto"``. Automatic
    resolution checks ``ZGC_ZIG``, the managed toolchain cache, and ``PATH``.
    ``zgc_cli`` selects the native CLI. ``zgc_root`` identifies the ZGC
    source checkout used when compiling executables.
    """

    zig: ZigSelection = "auto"
    zgc_cli: ZigSelection = "auto"
    cache_dir: str | PathLike[str] = field(default_factory=default_cache_dir)
    toolchain_dir: str | PathLike[str] | None = None
    zgc_root: str | PathLike[str] | None = None
    optimize: OptimizeMode = "ReleaseFast"
    target: str | None = None
    selection_target: SelectionTarget = "balanced"
    frontier_index: int | None = None
    inspectable: bool = False
    bind_outputs: bool = False
    zig_version: str = SUPPORTED_ZIG_VERSION

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir).expanduser().resolve()
