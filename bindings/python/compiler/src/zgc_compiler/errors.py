class CompilerError(RuntimeError):
    """Base error raised by the external ZGC compiler package."""


class ToolchainError(CompilerError):
    """A Zig toolchain could not be located, installed, or verified."""


class CompilationError(CompilerError):
    """Zig failed to produce a model artifact."""

    def __init__(
        self,
        message: str,
        *,
        command: tuple[str, ...],
        stdout: str,
        stderr: str,
    ) -> None:
        super().__init__(message)
        self.command = command
        self.stdout = stdout
        self.stderr = stderr


class GraphParseError(CompilerError):
    """A serialized graph does not conform to the ZGC graph language."""

    def __init__(self, message: str, *, line: int, column: int) -> None:
        super().__init__(f"{line}:{column}: {message}")
        self.line = line
        self.column = column
