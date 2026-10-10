from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .errors import CompilerError


class CompilerCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.graphs = root / "graphs"
        self.models = root / "models"
        self.temporary = root / "temporary"
        self.locks = root / "locks"
        self.zig = root / "zig"

    def prepare(self) -> None:
        for path in (self.graphs, self.models, self.temporary, self.locks, self.zig):
            path.mkdir(parents=True, exist_ok=True)

    def graph(self, fingerprint: str, generator_version: int) -> Path:
        return self.graphs / fingerprint / f"generator-{generator_version}"

    def model(self, key: str) -> Path:
        return self.models / key

    @contextmanager
    def lock(self, key: str, timeout: float = 120.0) -> Iterator[None]:
        self.prepare()
        path = self.locks / f"{key}.lock"
        deadline = time.monotonic() + timeout
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(descriptor, str(os.getpid()).encode())
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise CompilerError(f"timed out waiting for compilation lock {path}")
                time.sleep(0.05)
        try:
            yield
        finally:
            os.close(descriptor)
            path.unlink(missing_ok=True)
