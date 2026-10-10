from __future__ import annotations

import json
import math
from collections.abc import Sequence
from enum import Enum
from typing import Protocol, runtime_checkable

from .types import Redirect, SliceIteration

GRAPH_FORMAT_VERSION = 1


@runtime_checkable
class ValueReference(Protocol):
    @property
    def id(self) -> int: ...


class NodeRepresentation(Protocol):
    @property
    def id(self) -> int: ...

    @property
    def operation(self) -> str: ...

    @property
    def inputs(self) -> tuple[object, ...]: ...

    @property
    def attributes(self) -> tuple[tuple[str, object], ...]: ...


class GraphRepresentation(Protocol):
    @property
    def nodes(self) -> tuple[NodeRepresentation, ...]: ...

    @property
    def outputs(self) -> tuple[ValueReference, ...]: ...


def serialize_graph(graph: GraphRepresentation) -> str:
    lines = [f"zgir {GRAPH_FORMAT_VERSION}", ""]
    for node in graph.nodes:
        arguments = [_literal(value) for value in node.inputs]
        arguments.extend(f"{name}={_literal(value)}" for name, value in node.attributes)
        lines.append(f"%{node.id} = {node.operation}({', '.join(arguments)})")
    lines.extend(("", f"outputs = [{', '.join(_literal(value) for value in graph.outputs)}]"))
    return "\n".join(lines) + "\n"


def _literal(value: object) -> str:
    if isinstance(value, ValueReference):
        return f"%{value.id}"
    if isinstance(value, Enum):
        return value.value if isinstance(value.value, str) else value.name.lower()
    if isinstance(value, Redirect):
        return f"redirect({value.iteration})"
    if isinstance(value, SliceIteration):
        return (
            "iteration(offsets="
            f"{_literal(value.offsets)}, boundary={_literal(value.boundary)})"
        )
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("graph literals must be finite")
        return repr(value)
    if isinstance(value, Sequence):
        return f"[{', '.join(_literal(item) for item in value)}]"
    raise TypeError(f"unsupported graph literal: {type(value).__name__}")
