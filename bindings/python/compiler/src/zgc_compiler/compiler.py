from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

from .artifact import CompiledArtifact
from .config import SUPPORTED_ABI_VERSION, CompilerConfig
from .errors import CompilationError, CompilerError
from .frontend import FrozenGraph, Graph
from .toolchain import ZigToolchain, resolve_zig


def _artifact_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-.")
    if not normalized:
        raise CompilerError(
            "executable name must contain at least one usable character"
        )
    return normalized


class Compiler:
    """Use the native ZGC CLI to compile and inspect executable artifacts."""

    def __init__(self, config: CompilerConfig | None = None) -> None:
        self.config = config or CompilerConfig()

    def toolchain(self) -> ZigToolchain:
        return resolve_zig(
            self.config.zig,
            expected_version=self.config.zig_version,
            toolchain_dir=self.config.toolchain_dir,
        )

    def _native_cli(self) -> Path:
        configured = self.config.zgc_cli or os.environ.get("ZGC_CLI")
        if configured and configured != "auto":
            executable = shutil.which(os.fspath(configured))
            if executable is None:
                raise CompilerError(f"native ZGC CLI does not exist: {configured}")
            return Path(executable).resolve()
        root = self.config.zgc_root or os.environ.get("ZGC_SOURCE_ROOT")
        if root is not None:
            local = Path(root).expanduser().resolve() / "zig-out" / "bin" / "zgc"
            if local.is_file():
                return local
        for parent in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
            local = parent / "zig-out" / "bin" / "zgc"
            if (parent / "src" / "root.zig").is_file() and local.is_file():
                return local
        found = shutil.which("zgc")
        if found is not None:
            return Path(found).resolve()
        raise CompilerError(
            "native ZGC CLI not found; set CompilerConfig(zgc_cli=...), "
            "ZGC_CLI, or build it with 'zig build'"
        )

    def _run(self, *arguments: str, compile_command: bool = False) -> str:
        command = [str(self._native_cli()), *arguments]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError as error:
            raise CompilerError(f"cannot run native ZGC CLI: {error}") from error
        if completed.returncode != 0:
            if compile_command:
                raise CompilationError(
                    "native ZGC compilation failed",
                    command=tuple(command),
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
            raise CompilerError(
                completed.stderr.strip()
                or f"native ZGC exited with {completed.returncode}"
            )
        return completed.stdout

    def _compile_file(
        self,
        definition: str | os.PathLike[str],
        *,
        name: str | None = None,
        imports: Mapping[str, str | os.PathLike[str]] | None = None,
    ) -> CompiledArtifact:
        definition_path = Path(definition).expanduser().resolve()
        if not definition_path.is_file():
            raise CompilerError(f"definition does not exist: {definition_path}")
        if definition_path.suffix not in {".zig", ".zgir"}:
            raise CompilerError("definition must be a .zig or .zgir file")
        toolchain = self.toolchain()
        command = [
            "compile",
            str(definition_path),
            "--cache-output",
            "--cache-dir",
            str(self.config.cache_path),
            "--zig",
            str(toolchain.executable),
            "--optimize",
            self.config.optimize,
        ]
        if name is not None:
            command.extend(("--name", _artifact_name(name)))
        if self.config.inspectable:
            command.append("--inspectable")
        if self.config.zgc_root is not None:
            command.extend(
                ("--zgc-root", str(Path(self.config.zgc_root).expanduser().resolve()))
            )
        if self.config.target is not None:
            command.extend(("--target", self.config.target))
        if definition_path.suffix == ".zgir":
            command.extend(("--selection-target", self.config.selection_target))
            if self.config.frontier_index is not None:
                command.extend(("--frontier-index", str(self.config.frontier_index)))
        elif (
            self.config.frontier_index is not None
            or self.config.selection_target != "balanced"
        ):
            raise CompilerError("executable selection options require a .zgir graph")
        for module_name, path in sorted((imports or {}).items()):
            if definition_path.suffix == ".zgir":
                raise CompilerError(
                    "serialized graphs do not accept Zig module imports"
                )
            if not module_name.isidentifier():
                raise CompilerError(f"invalid Zig module name: {module_name!r}")
            module_path = Path(path).expanduser().resolve()
            if not module_path.is_file():
                raise CompilerError(
                    f"module {module_name!r} does not exist: {module_path}"
                )
            command.extend(("--import", f"{module_name}={module_path}"))
        library = Path(self._run(*command, compile_command=True).strip()).resolve()
        if not library.is_file():
            raise CompilerError(f"native ZGC did not produce a library: {library}")
        fingerprint = (
            self._compiled_fingerprint(library) if self.config.target is None else None
        )
        return CompiledArtifact(path=library, structural_fingerprint=fingerprint)

    def _compile_graph(
        self, graph: Graph | FrozenGraph | str, *, name: str = "executable"
    ) -> CompiledArtifact:
        """Return a cached graph artifact or delegate compilation to native ZGC."""
        if isinstance(graph, (Graph, FrozenGraph)):
            fingerprint = graph.fingerprint
        elif isinstance(graph, str):
            fingerprint = hashlib.sha256(graph.encode("utf-8")).hexdigest()
        else:
            raise TypeError(
                "graph must be Graph, FrozenGraph, or serialized graph text"
            )
        artifact_name = _artifact_name(name)
        index = self._graph_cache_index(fingerprint, artifact_name)
        cached = self._cached_graph_artifact(index)
        if cached is not None:
            return cached
        serialized = graph if isinstance(graph, str) else graph.serialize()
        with tempfile.TemporaryDirectory(prefix="zgc-graph-") as temporary:
            path = Path(temporary) / "graph.zgir"
            path.write_text(serialized, encoding="utf-8")
            artifact = self._compile_file(path, name=artifact_name)
        index.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "library": str(artifact.path),
            "library_stamp": self._file_stamp(artifact.path),
            "structural_fingerprint": artifact.structural_fingerprint,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=index.parent, delete=False
        ) as temporary:
            json.dump(payload, temporary)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, index)
        return artifact

    def _graph_cache_index(self, fingerprint: str, name: str) -> Path:
        """Partition graph entries by every setting that affects compilation."""
        native = self._native_cli()
        zig = self.toolchain()
        root = self._source_root(native)
        settings = {
            "format": 1,
            "abi": SUPPORTED_ABI_VERSION,
            "name": name,
            "optimize": self.config.optimize,
            "target": self.config.target,
            "selection_target": self.config.selection_target,
            "frontier_index": self.config.frontier_index,
            "inspectable": self.config.inspectable,
            "native": str(native),
            "native_stamp": self._file_stamp(native),
            "zig": str(zig.executable),
            "zig_version": zig.version,
            "zig_stamp": self._file_stamp(zig.executable),
            "source_root": str(root) if root is not None else None,
            "source_stamp": self._source_stamp(root) if root is not None else None,
        }
        variant = hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return (
            self.config.cache_path
            / "python"
            / "graphs"
            / fingerprint
            / f"{variant}.json"
        )

    def _source_root(self, native: Path) -> Path | None:
        configured = self.config.zgc_root or os.environ.get("ZGC_SOURCE_ROOT")
        if configured is not None:
            return Path(configured).expanduser().resolve()
        for candidate in (
            native.parents[2],
            Path.cwd().resolve(),
            *Path.cwd().resolve().parents,
        ):
            if (candidate / "src" / "root.zig").is_file():
                return candidate
        return None

    @staticmethod
    def _file_stamp(path: Path) -> tuple[int, int]:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns

    @classmethod
    def _source_stamp(cls, root: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted((root / "src").rglob("*.zig")):
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(repr(cls._file_stamp(path)).encode("ascii"))
        return digest.hexdigest()

    def _cached_graph_artifact(self, index: Path) -> CompiledArtifact | None:
        try:
            entry = json.loads(index.read_text(encoding="utf-8"))
            library = Path(entry["library"]).resolve()
            if (
                not library.is_relative_to(self.config.cache_path)
                or not library.is_file()
            ):
                return None
            if list(self._file_stamp(library)) != entry["library_stamp"]:
                return None
            fingerprint = entry["structural_fingerprint"]
            if self.config.target is None:
                actual = self._compiled_fingerprint(library)
                if actual != fingerprint:
                    return None
            return CompiledArtifact(library, fingerprint)
        except (OSError, ValueError, KeyError, TypeError, CompilerError):
            return None

    def compile(
        self,
        definition: Graph | FrozenGraph | str | os.PathLike[str],
        *,
        name: str | None = None,
        imports: Mapping[str, str | os.PathLike[str]] | None = None,
    ) -> CompiledArtifact:
        """Compile a graph, serialized graph, or definition file."""
        if isinstance(definition, (Graph, FrozenGraph)):
            if imports:
                raise CompilerError("Python graphs do not accept Zig module imports")
            return self._compile_graph(definition, name=name or "executable")
        if isinstance(definition, str) and (
            definition.startswith("zgir ") or "\n" in definition
        ):
            if imports:
                raise CompilerError(
                    "serialized graphs do not accept Zig module imports"
                )
            return self._compile_graph(definition, name=name or "executable")
        return self._compile_file(definition, name=name, imports=imports)

    def inspect(
        self,
        executable: CompiledArtifact | str | os.PathLike[str],
        section: str = "all",
    ) -> str:
        """Return native ZGC's inspection text for a compiled executable."""
        path = (
            executable.path if isinstance(executable, CompiledArtifact) else executable
        )
        return self._run("inspect", str(Path(path).expanduser().resolve()), section)

    def cache_path(self) -> Path:
        """Return the cache path selected by native ZGC."""
        return Path(
            self._run(
                "cache", "path", "--cache-dir", str(self.config.cache_path)
            ).strip()
        )

    def clear_cache(self) -> None:
        """Ask native ZGC to clear the configured cache."""
        self._run("cache", "clear", "--cache-dir", str(self.config.cache_path))

    @staticmethod
    def _compiled_fingerprint(library: Path) -> int:
        try:
            compiled = ctypes.CDLL(str(library))
            compiled.zgc_abi_version.restype = ctypes.c_uint32
            compiled.zgc_executable_structural_fingerprint.restype = ctypes.c_uint64
            version = compiled.zgc_abi_version()
            if version != SUPPORTED_ABI_VERSION:
                raise CompilerError(f"executable has unsupported ABI version {version}")
            return int(compiled.zgc_executable_structural_fingerprint())
        except (OSError, AttributeError) as error:
            raise CompilerError(f"cannot load executable: {library}") from error
