from __future__ import annotations

import shutil
from dataclasses import dataclass
from os import PathLike
from pathlib import Path


@dataclass(frozen=True)
class CompiledArtifact:
    """A cached, ABI-compatible executable library produced by Zig."""

    path: Path
    structural_fingerprint: int | None = None

    def save(self, destination: str | PathLike[str]) -> Path:
        """Copy the compiled library to a stable user-selected location."""

        path = Path(destination).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.path, path)
        return path
