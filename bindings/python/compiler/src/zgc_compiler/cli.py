from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import SUPPORTED_ZIG_VERSION
from .errors import CompilerError
from .toolchain import install_zig, resolve_zig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="zgc-compiler", description="Manage the Zig toolchain used by ZGC"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    toolchain = commands.add_parser("toolchain", help="manage the Zig toolchain")
    actions = toolchain.add_subparsers(dest="action", required=True)
    install = actions.add_parser("install", help="install the supported Zig compiler")
    install.add_argument("--version", default=SUPPORTED_ZIG_VERSION)
    install.add_argument("--toolchain-dir", type=Path)
    verify = actions.add_parser("verify", help="locate and verify Zig")
    verify.add_argument("--zig", default="auto")
    verify.add_argument("--version", default=SUPPORTED_ZIG_VERSION)
    verify.add_argument("--toolchain-dir", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.action == "install":
            result = install_zig(
                version=arguments.version,
                toolchain_dir=arguments.toolchain_dir,
            )
        else:
            result = resolve_zig(
                arguments.zig,
                expected_version=arguments.version,
                toolchain_dir=arguments.toolchain_dir,
            )
    except (CompilerError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"{result.executable} ({result.version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
