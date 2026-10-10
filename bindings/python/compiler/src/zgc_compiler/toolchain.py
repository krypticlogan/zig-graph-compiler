from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .config import SUPPORTED_ZIG_VERSION, ZigSelection
from .errors import ToolchainError

_DOWNLOAD_INDEX = "https://ziglang.org/download/index.json"


@dataclass(frozen=True)
class ZigToolchain:
    executable: Path
    version: str
    managed: bool


def default_toolchain_dir() -> Path:
    configured = os.environ.get("ZGC_TOOLCHAIN_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Caches" / "zgc" / "toolchains"
    if system == "Windows":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "zgc" / "Cache" / "toolchains"
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "zgc" / "toolchains"


def platform_key() -> str:
    systems = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}
    machines = {
        "x86_64": "x86_64",
        "AMD64": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
    }
    try:
        return f"{machines[platform.machine()]}-{systems[platform.system()]}"
    except KeyError as error:
        raise ToolchainError(
            f"unsupported Zig host {platform.machine()}-{platform.system()}"
        ) from error


def _managed_executable(root: Path, version: str) -> Path:
    name = "zig.exe" if platform.system() == "Windows" else "zig"
    return root / version / platform_key() / name


def verify_zig(path: Path, expected_version: str) -> ZigToolchain:
    if not path.is_file():
        raise ToolchainError(f"Zig executable does not exist: {path}")
    try:
        completed = subprocess.run(
            [str(path), "version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ToolchainError(f"failed to execute Zig at {path}: {error}") from error
    version = completed.stdout.strip()
    if version != expected_version:
        raise ToolchainError(
            f"ZGC requires Zig {expected_version}; {path} reports {version}"
        )
    return ZigToolchain(path.resolve(), version, False)


def resolve_zig(
    selection: ZigSelection = "auto",
    *,
    expected_version: str = SUPPORTED_ZIG_VERSION,
    toolchain_dir: str | os.PathLike[str] | None = None,
) -> ZigToolchain:
    if os.fspath(selection) != "auto":
        return verify_zig(Path(selection).expanduser().resolve(), expected_version)

    configured = os.environ.get("ZGC_ZIG")
    if configured:
        return verify_zig(Path(configured).expanduser().resolve(), expected_version)

    root = (
        Path(toolchain_dir).expanduser().resolve()
        if toolchain_dir is not None
        else default_toolchain_dir()
    )
    managed = _managed_executable(root, expected_version)
    if managed.is_file():
        found = verify_zig(managed, expected_version)
        return ZigToolchain(found.executable, found.version, True)

    discovered = shutil.which("zig")
    if discovered:
        return verify_zig(Path(discovered), expected_version)

    raise ToolchainError(
        "no compatible Zig compiler was found; configure CompilerConfig(zig=...), "
        "set ZGC_ZIG, add Zig to PATH, or run: zgc toolchain install"
    )


def _archive_entry(version: str) -> tuple[str, str]:
    try:
        with urllib.request.urlopen(_DOWNLOAD_INDEX, timeout=30) as response:
            index = json.load(response)
        release = cast(dict[str, object], index[version])
        entry = cast(dict[str, str], release[platform_key()])
        return entry["tarball"], entry["shasum"]
    except (KeyError, OSError, ValueError, TypeError) as error:
        raise ToolchainError(
            f"could not resolve the Zig {version} download for {platform_key()}"
        ) from error


def _safe_archive_path(root: Path, name: str) -> Path:
    destination = (root / name).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ToolchainError(f"unsafe path in Zig archive: {name}")
    return destination


def _extract_archive(archive: Path, destination: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                _safe_archive_path(destination, member.filename)
            bundle.extractall(destination)
        return
    with tarfile.open(archive, "r:xz") as bundle:
        for member in bundle.getmembers():
            _safe_archive_path(destination, member.name)
            if member.issym() or member.islnk():
                raise ToolchainError("Zig archive contains an unsupported link")
        bundle.extractall(destination)


def install_zig(
    *,
    version: str = SUPPORTED_ZIG_VERSION,
    toolchain_dir: str | os.PathLike[str] | None = None,
) -> ZigToolchain:
    root = (
        Path(toolchain_dir).expanduser().resolve()
        if toolchain_dir is not None
        else default_toolchain_dir()
    )
    executable = _managed_executable(root, version)
    if executable.is_file():
        found = verify_zig(executable, version)
        return ZigToolchain(found.executable, found.version, True)

    url, expected_hash = _archive_entry(version)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="install-", dir=root) as temporary:
        work = Path(temporary)
        archive = work / Path(urllib.parse.urlparse(url).path).name
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
        except OSError as error:
            raise ToolchainError(f"failed to download Zig {version}: {error}") from error
        if digest.hexdigest() != expected_hash:
            raise ToolchainError(f"checksum verification failed for Zig {version}")

        extracted = work / "extracted"
        extracted.mkdir()
        _extract_archive(archive, extracted)
        name = "zig.exe" if platform.system() == "Windows" else "zig"
        candidates = tuple(extracted.rglob(name))
        if len(candidates) != 1:
            raise ToolchainError("downloaded Zig archive has an unexpected layout")
        source_root = candidates[0].parent
        install_root = executable.parent
        install_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source_root, install_root)

    found = verify_zig(executable, version)
    return ZigToolchain(found.executable, found.version, True)
