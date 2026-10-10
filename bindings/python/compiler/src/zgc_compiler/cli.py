from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from collections.abc import Sequence

from zgc.cli import CommandRegistry

from .compiler import Compiler
from .config import CompilerConfig, SUPPORTED_ZIG_VERSION, default_cache_dir
from .errors import CompilerError
from .toolchain import install_zig, resolve_zig


def _imports(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise argparse.ArgumentTypeError("imports must use NAME=PATH")
        result[name] = Path(path)
    return result


def register(commands: CommandRegistry) -> None:
    compile_parser = commands.add_parser(
        "compile",
        help="compile a serialized graph or Zig model definition",
    )
    compile_parser.add_argument("model", type=Path)
    compile_parser.add_argument("--name")
    compile_parser.add_argument("--output", type=Path)
    compile_parser.add_argument("--import", dest="imports", action="append", default=[])
    compile_parser.add_argument("--zig", default="auto")
    compile_parser.add_argument("--zgc-root", type=Path)
    compile_parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    compile_parser.add_argument(
        "--optimize",
        choices=("Debug", "ReleaseSafe", "ReleaseFast", "ReleaseSmall"),
        default="ReleaseFast",
    )
    compile_parser.add_argument("--target")
    compile_parser.set_defaults(_zgc_handler=_execute)

    toolchain = commands.add_parser("toolchain", help="manage the Zig toolchain")
    toolchain_commands = toolchain.add_subparsers(dest="toolchain_command", required=True)
    install = toolchain_commands.add_parser("install", help="install the supported Zig compiler")
    install.add_argument("--version", default=SUPPORTED_ZIG_VERSION)
    install.add_argument("--toolchain-dir", type=Path)
    verify = toolchain_commands.add_parser("verify", help="locate and verify Zig")
    verify.add_argument("--zig", default="auto")
    verify.add_argument("--version", default=SUPPORTED_ZIG_VERSION)
    verify.add_argument("--toolchain-dir", type=Path)
    toolchain.set_defaults(_zgc_handler=_execute)

    cache = commands.add_parser("cache", help="inspect or clear compiled artifacts")
    cache.add_argument("action", choices=("path", "clear"))
    cache.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    cache.set_defaults(_zgc_handler=_execute)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zgc-compiler",
        description="Compile ZGC model artifacts",
    )
    register(parser.add_subparsers(dest="command", required=True))
    return parser


def _run(arguments: argparse.Namespace) -> int:
    if arguments.command == "compile":
        config = CompilerConfig(
            zig=arguments.zig,
            cache_dir=arguments.cache_dir,
            zgc_root=arguments.zgc_root,
            optimize=arguments.optimize,
            target=arguments.target,
        )
        compiler = Compiler(config)
        if arguments.model.suffix == ".zgir":
            if arguments.imports:
                raise CompilerError("serialized graphs do not accept Zig module imports")
            artifact = compiler.compile_graph_file(
                arguments.model,
                name=arguments.name,
            )
        else:
            artifact = compiler.compile_model(
                arguments.model,
                name=arguments.name,
                imports=_imports(arguments.imports),
            )
        path = artifact.save(arguments.output) if arguments.output else artifact.path
        print(path)
        return 0
    if arguments.command == "toolchain":
        if arguments.toolchain_command == "install":
            toolchain = install_zig(
                version=arguments.version,
                toolchain_dir=arguments.toolchain_dir,
            )
        else:
            toolchain = resolve_zig(
                arguments.zig,
                expected_version=arguments.version,
                toolchain_dir=arguments.toolchain_dir,
            )
        print(f"{toolchain.executable} ({toolchain.version})")
        return 0
    cache = arguments.cache_dir.expanduser().resolve()
    if arguments.action == "path":
        print(cache)
        return 0
    if cache in {Path(cache.anchor), Path.home().resolve(), Path.cwd().resolve()}:
        raise CompilerError(f"refusing to clear broad cache path: {cache}")
    if cache.exists():
        shutil.rmtree(cache)
    return 0


def _execute(arguments: argparse.Namespace) -> int:
    try:
        return _run(arguments)
    except (CompilerError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    return _execute(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
