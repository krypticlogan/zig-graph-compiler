from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from .artifact import CompiledArtifact
from .cache import CompilerCache
from .config import CompilerConfig, SUPPORTED_ABI_VERSION
from .errors import CompilationError, CompilerError
from .frontend import FrozenGraph, Graph, parse_graph
from .toolchain import ZigToolchain, resolve_zig

_COMPILER_FORMAT_VERSION = 2
_GRAPH_ADAPTER_VERSION = 3


class _Digest(Protocol):
    def update(self, value: bytes, /) -> object: ...


def _library_suffix(target: str | None) -> str:
    system = target or platform.system().lower()
    if "windows" in system:
        return ".dll"
    if "macos" in system or "darwin" in system:
        return ".dylib"
    return ".so"


def _artifact_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-.")
    if not normalized:
        raise CompilerError("model name must contain at least one usable character")
    return normalized


def _hash_file(digest: _Digest, path: Path, label: str) -> None:
    digest.update(label.encode())
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")


class Compiler:
    """Compile Zig model definitions into cached C-ABI model libraries."""

    def __init__(self, config: CompilerConfig | None = None) -> None:
        self.config = config or CompilerConfig()

    def toolchain(self) -> ZigToolchain:
        return resolve_zig(
            self.config.zig,
            expected_version=self.config.zig_version,
            toolchain_dir=self.config.toolchain_dir,
        )

    def compile_model(
        self,
        model: str | os.PathLike[str],
        *,
        name: str | None = None,
        imports: Mapping[str, str | os.PathLike[str]] | None = None,
        _cache_inputs: tuple[Path, ...] = (),
    ) -> CompiledArtifact:
        model_path = Path(model).expanduser().resolve()
        if not model_path.is_file():
            raise CompilerError(f"model definition does not exist: {model_path}")
        source_root = self._resolve_source_root(model_path)
        toolchain = self.toolchain()
        modules = {
            module_name: Path(path).expanduser().resolve()
            for module_name, path in (imports or {}).items()
        }
        for module_name, path in modules.items():
            if not module_name.isidentifier():
                raise CompilerError(f"invalid Zig module name: {module_name!r}")
            if not path.is_file():
                raise CompilerError(f"module {module_name!r} does not exist: {path}")

        artifact_name = _artifact_name(name or model_path.stem)
        cache = CompilerCache(self.config.cache_path)
        key = self._cache_key(
            model_path,
            source_root,
            toolchain,
            modules,
            artifact_name,
            _cache_inputs,
        )
        filename = f"{artifact_name}{_library_suffix(self.config.target)}"
        cached = self._cached_artifact(cache, key, filename)
        if cached is not None:
            return cached

        with cache.lock(key):
            cached = self._cached_artifact(cache, key, filename)
            if cached is not None:
                return cached
            return self._compile(
                cache,
                key,
                filename,
                model_path,
                source_root,
                toolchain,
                modules,
            )

    def compile_graph(
        self,
        graph: Graph | FrozenGraph | str,
        *,
        name: str = "model",
    ) -> CompiledArtifact:
        """Canonicalize, cache, and compile a Python or serialized semantic graph."""

        if isinstance(graph, Graph):
            frozen = graph.freeze()
        elif isinstance(graph, FrozenGraph):
            frozen = graph
        elif isinstance(graph, str):
            frozen = parse_graph(graph)
        else:
            raise TypeError("graph must be Graph, FrozenGraph, or serialized graph text")
        source_root = self._resolve_source_root(Path.cwd().resolve())
        model = self._graph_definition(frozen, source_root)
        return self.compile_model(
            model,
            name=name,
            _cache_inputs=(model.with_name("graph.zgir"),),
        )

    def compile_graph_file(
        self,
        graph: str | os.PathLike[str],
        *,
        name: str | None = None,
    ) -> CompiledArtifact:
        path = Path(graph).expanduser().resolve()
        if not path.is_file():
            raise CompilerError(f"graph definition does not exist: {path}")
        frozen = parse_graph(path.read_text(encoding="utf-8"))
        return self.compile_graph(frozen, name=name or path.stem)

    def _graph_definition(self, graph: FrozenGraph, source_root: Path) -> Path:
        cache = CompilerCache(self.config.cache_path)
        destination = cache.graph(graph.fingerprint, _GRAPH_ADAPTER_VERSION)
        model = destination / "model.zig"
        serialized = destination / "graph.zgir"
        graph_module = destination / "graph.zig"
        manifest = destination / "manifest.json"
        if all(path.is_file() for path in (model, serialized, graph_module, manifest)):
            return model

        lock_key = f"graph-{graph.fingerprint}-{_GRAPH_ADAPTER_VERSION}"
        with cache.lock(lock_key):
            if all(path.is_file() for path in (model, serialized, graph_module, manifest)):
                return model
            cache.prepare()
            parent = destination.parent
            parent.mkdir(parents=True, exist_ok=True)
            work = Path(tempfile.mkdtemp(prefix="graph-", dir=cache.temporary))
            try:
                (work / "graph.zgir").write_text(graph.serialize(), encoding="utf-8")
                shutil.copyfile(
                    source_root / "src" / "artifact" / "zgir_model.zig",
                    work / "model.zig",
                )
                shutil.copyfile(
                    source_root / "src" / "artifact" / "zgir_source.zig",
                    work / "graph.zig",
                )
                (work / "manifest.json").write_text(
                    json.dumps(
                        {
                            "adapter_format": _GRAPH_ADAPTER_VERSION,
                            "graph_fingerprint": graph.fingerprint,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                if destination.exists():
                    shutil.rmtree(destination)
                os.replace(work, destination)
            finally:
                if work.exists():
                    shutil.rmtree(work)
        return model

    def _resolve_source_root(self, model: Path) -> Path:
        configured = self.config.zgc_root or os.environ.get("ZGC_SOURCE_ROOT")
        if configured is not None:
            root = Path(configured).expanduser().resolve()
            self._validate_source_root(root)
            return root

        for start in (Path.cwd().resolve(), *model.parents):
            for candidate in (start, *start.parents):
                if (candidate / "src" / "root.zig").is_file() and (
                    candidate / "src" / "artifact" / "model_abi.zig"
                ).is_file():
                    return candidate
        raise CompilerError(
            "could not locate ZGC compiler sources; set CompilerConfig(zgc_root=...) "
            "or ZGC_SOURCE_ROOT"
        )

    @staticmethod
    def _validate_source_root(root: Path) -> None:
        required = (
            root / "src" / "root.zig",
            root / "src" / "artifact" / "model_abi.zig",
            root / "src" / "artifact" / "zgir_model.zig",
            root / "src" / "artifact" / "zgir_source.zig",
        )
        if not all(path.is_file() for path in required):
            raise CompilerError(f"not a ZGC source root: {root}")

    def _cache_key(
        self,
        model: Path,
        source_root: Path,
        toolchain: ZigToolchain,
        imports: Mapping[str, Path],
        artifact_name: str,
        cache_inputs: tuple[Path, ...],
    ) -> str:
        digest = hashlib.sha256()
        metadata = {
            "abi": SUPPORTED_ABI_VERSION,
            "artifact_name": artifact_name,
            "format": _COMPILER_FORMAT_VERSION,
            "optimize": self.config.optimize,
            "target": self.config.target,
            "zig": toolchain.version,
        }
        digest.update(json.dumps(metadata, sort_keys=True).encode())
        _hash_file(digest, model, "model")
        for index, path in enumerate(cache_inputs):
            _hash_file(digest, path, f"cache-input:{index}")
        for path in sorted((source_root / "src").rglob("*.zig")):
            _hash_file(digest, path, f"zgc:{path.relative_to(source_root)}")
        for module_name, path in sorted(imports.items()):
            _hash_file(digest, path, f"import:{module_name}")
        return digest.hexdigest()

    @staticmethod
    def _cached_artifact(
        cache: CompilerCache,
        key: str,
        filename: str,
    ) -> CompiledArtifact | None:
        directory = cache.model(key)
        library = directory / filename
        manifest = directory / "manifest.json"
        if library.is_file() and manifest.is_file():
            return CompiledArtifact(library, manifest, key, True)
        return None

    def _compile(
        self,
        cache: CompilerCache,
        key: str,
        filename: str,
        model: Path,
        source_root: Path,
        toolchain: ZigToolchain,
        imports: Mapping[str, Path],
    ) -> CompiledArtifact:
        cache.prepare()
        work = Path(tempfile.mkdtemp(prefix=f"{key[:12]}-", dir=cache.temporary))
        try:
            library = work / filename
            command = self._command(toolchain, source_root, model, imports, library)
            environment = os.environ.copy()
            environment["ZIG_LOCAL_CACHE_DIR"] = str(cache.zig)
            completed = subprocess.run(
                command,
                cwd=source_root,
                env=environment,
                capture_output=True,
                text=True,
            )
            (work / "build.log").write_text(completed.stdout + completed.stderr)
            if completed.returncode != 0:
                raise CompilationError(
                    f"Zig failed to compile {model}",
                    command=tuple(command),
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
            manifest_data = {
                "abi": SUPPORTED_ABI_VERSION,
                "cache_key": key,
                "compiler_format": _COMPILER_FORMAT_VERSION,
                "library": filename,
                "model": str(model),
                "optimize": self.config.optimize,
                "target": self.config.target,
                "zig": toolchain.version,
            }
            (work / "manifest.json").write_text(
                json.dumps(manifest_data, indent=2, sort_keys=True) + "\n"
            )
            destination = cache.model(key)
            os.replace(work, destination)
            return CompiledArtifact(
                destination / filename,
                destination / "manifest.json",
                key,
                False,
            )
        finally:
            if work.exists():
                shutil.rmtree(work)

    def _command(
        self,
        toolchain: ZigToolchain,
        source_root: Path,
        model: Path,
        imports: Mapping[str, Path],
        output: Path,
    ) -> list[str]:
        command = [
            str(toolchain.executable),
            "build-lib",
            "-dynamic",
            f"-O{self.config.optimize}",
        ]
        if self.config.target:
            command.extend(("-target", self.config.target))
        command.extend(
            (
                "--dep",
                "model",
                "--dep",
                "zgc",
                f"-Mroot={source_root / 'src' / 'artifact' / 'model_abi.zig'}",
                "--dep",
                "zgc",
            )
        )
        graph_module = model.with_name("graph.zig")
        if graph_module.is_file():
            command.extend(("--dep", "graph"))
        for module_name in sorted(imports):
            command.extend(("--dep", module_name))
        command.append(f"-Mmodel={model}")
        command.append(f"-Mzgc={source_root / 'src' / 'root.zig'}")
        if graph_module.is_file():
            command.append(f"-Mgraph={graph_module}")
        for module_name, path in sorted(imports.items()):
            command.append(f"-M{module_name}={path}")
        command.append(f"-femit-bin={output}")
        return command
