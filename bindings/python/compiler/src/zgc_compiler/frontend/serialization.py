from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Sequence
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


def _graph_lines(graph: GraphRepresentation) -> Iterator[str]:
    yield f"zgir {GRAPH_FORMAT_VERSION}"
    yield ""
    for node in graph.nodes:
        arguments = [_literal(value) for value in node.inputs]
        arguments.extend(f"{name}={_literal(value)}" for name, value in node.attributes)
        yield f"%{node.id} = {node.operation}({', '.join(arguments)})"
    yield ""
    yield f"outputs = [{', '.join(_literal(value) for value in graph.outputs)}]"


def serialize_graph(graph: GraphRepresentation) -> str:
    return "\n".join(_graph_lines(graph)) + "\n"


def fingerprint_graph(graph: GraphRepresentation) -> str:
    """Hash the canonical graph without constructing its serialized text."""
    digest = hashlib.sha256()
    for line in _graph_lines(graph):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


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
