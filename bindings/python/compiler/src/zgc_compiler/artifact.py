from __future__ import annotations

import shutil
from dataclasses import dataclass
from os import PathLike
from pathlib import Path


@dataclass(frozen=True)
class CompiledArtifact:
    """A cached, ABI-compatible model library produced by Zig."""

    library_path: Path
    manifest_path: Path
    cache_key: str
    cache_hit: bool

    @property
    def path(self) -> Path:
        return self.library_path

    def save(self, destination: str | PathLike[str]) -> Path:
        """Copy the compiled library to a stable user-selected location."""

        path = Path(destination).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.library_path, path)
        return path
