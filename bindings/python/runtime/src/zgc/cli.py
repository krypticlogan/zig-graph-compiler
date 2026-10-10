from __future__ import annotations

import argparse
import importlib.metadata
from collections.abc import Callable, Sequence
from typing import Protocol, cast

_EXTENSION_GROUP = "zgc.cli"


class CommandExtension(Protocol):
    """Register commands on the shared ZGC command-line parser."""

    def __call__(self, commands: CommandRegistry) -> None: ...


class CommandRegistry(Protocol):
    """The command registration surface exposed to companion packages."""

    def add_parser(self, name: str, **kwargs: object) -> argparse.ArgumentParser: ...


def _extensions() -> tuple[CommandExtension, ...]:
    discovered = importlib.metadata.entry_points(group=_EXTENSION_GROUP)
    return tuple(
        cast(CommandExtension, entry_point.load())
        for entry_point in sorted(discovered, key=lambda entry: entry.name)
    )


def _parser(extensions: Sequence[CommandExtension]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zgc",
        description="Run and manage ZGC model artifacts",
    )
    commands = parser.add_subparsers(dest="command")
    for extension in extensions:
        extension(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser(_extensions())
    arguments = parser.parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int] | None, getattr(arguments, "_zgc_handler", None))
    if handler is None:
        parser.print_help()
        return 0
    return handler(arguments)
