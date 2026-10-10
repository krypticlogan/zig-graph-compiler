from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from .serialization import serialize_graph
from .types import Binding, Boundary, SliceIteration, dtype as dtype_enum

Scalar: TypeAlias = float | int | bool
Shape: TypeAlias = Sequence[int]
Axes: TypeAlias = int | Sequence[int] | None

OPERATION_NAMES = frozenset(
    {
        "input", "parameter", "constant", "scalar", "full",
        "relu", "exp", "neg", "abs", "sqrt", "log", "reciprocal",
        "add", "sub", "mul", "div", "minimum", "maximum", "clamp",
        "equal", "not_equal", "less_than", "less_equal", "greater_than",
        "greater_equal", "logical_not", "logical_and", "logical_or", "where",
        "copy", "contiguous", "pad", "shift", "slice_loop", "matmul",
        "sum", "mean", "min", "max", "concat", "softmax", "transpose",
        "reshape", "broadcast_to", "flatten", "squeeze", "unsqueeze",
        "permute", "slice", "windows",
    }
)


@dataclass(frozen=True)
class Node:
    id: int
    operation: str
    inputs: tuple[object, ...]
    attributes: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class FrozenGraph:
    nodes: tuple[Node, ...]
    outputs: tuple[Value, ...]

    def serialize(self) -> str:
        return serialize_graph(self)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.serialize().encode("utf-8")).hexdigest()


@dataclass(frozen=True, eq=False)
class Value:
    _graph: Graph
    id: int

    @property
    def graph(self) -> Graph:
        return self._graph

    def relu(self) -> Value: return self._graph.relu(self)
    def exp(self) -> Value: return self._graph.exp(self)
    def neg(self) -> Value: return self._graph.neg(self)
    def abs(self) -> Value: return self._graph.abs(self)
    def sqrt(self) -> Value: return self._graph.sqrt(self)
    def log(self) -> Value: return self._graph.log(self)
    def reciprocal(self) -> Value: return self._graph.reciprocal(self)
    def add(self, rhs: Value) -> Value: return self._graph.add(self, rhs)
    def sub(self, rhs: Value) -> Value: return self._graph.sub(self, rhs)
    def mul(self, rhs: Value) -> Value: return self._graph.mul(self, rhs)
    def div(self, rhs: Value) -> Value: return self._graph.div(self, rhs)
    def minimum(self, rhs: Value) -> Value: return self._graph.minimum(self, rhs)
    def maximum(self, rhs: Value) -> Value: return self._graph.maximum(self, rhs)
    def clamp(self, lower: Value, upper: Value) -> Value: return self._graph.clamp(self, lower, upper)
    def equal(self, rhs: Value) -> Value: return self._graph.equal(self, rhs)
    def not_equal(self, rhs: Value) -> Value: return self._graph.not_equal(self, rhs)
    def less_than(self, rhs: Value) -> Value: return self._graph.less_than(self, rhs)
    def less_equal(self, rhs: Value) -> Value: return self._graph.less_equal(self, rhs)
    def greater_than(self, rhs: Value) -> Value: return self._graph.greater_than(self, rhs)
    def greater_equal(self, rhs: Value) -> Value: return self._graph.greater_equal(self, rhs)
    def logical_not(self) -> Value: return self._graph.logical_not(self)
    def logical_and(self, rhs: Value) -> Value: return self._graph.logical_and(self, rhs)
    def logical_or(self, rhs: Value) -> Value: return self._graph.logical_or(self, rhs)
    def where(self, when_true: Value, when_false: Value) -> Value:
        return self._graph.where(self, when_true, when_false)
    def copy(self) -> Value: return self._graph.copy(self)
    def contiguous(self) -> Value: return self._graph.contiguous(self)
    def pad(self, fill: Value, *, before: Shape, after: Shape) -> Value:
        return self._graph.pad(self, fill, before=before, after=after)
    def shift(self, offsets: Sequence[int], *, boundary: Boundary, fill: Value | None = None) -> Value:
        return self._graph.shift(self, offsets, boundary=boundary, fill=fill)
    def slice_loop(self, *, axis: int, iterations: Sequence[SliceIteration]) -> Value:
        return self._graph.slice_loop(self, axis=axis, iterations=iterations)
    def matmul(self, rhs: Value) -> Value: return self._graph.matmul(self, rhs)
    def sum(self, *, axes: Axes = None, keep_dims: bool = False) -> Value:
        return self._graph.sum(self, axes=axes, keep_dims=keep_dims)
    def mean(self, *, axes: Axes = None, keep_dims: bool = False) -> Value:
        return self._graph.mean(self, axes=axes, keep_dims=keep_dims)
    def min(self, *, axes: Axes = None, keep_dims: bool = False) -> Value:
        return self._graph.min(self, axes=axes, keep_dims=keep_dims)
    def max(self, *, axes: Axes = None, keep_dims: bool = False) -> Value:
        return self._graph.max(self, axes=axes, keep_dims=keep_dims)
    def concat(self, *others: Value, axis: int) -> Value:
        return self._graph.concat((self, *others), axis=axis)
    def softmax(self, *, axis: int) -> Value: return self._graph.softmax(self, axis=axis)
    def transpose(self, axis_a: int, axis_b: int) -> Value:
        return self._graph.transpose(self, axis_a, axis_b)
    def reshape(self, shape: Shape) -> Value: return self._graph.reshape(self, shape)
    def broadcast_to(self, shape: Shape) -> Value: return self._graph.broadcast_to(self, shape)
    def flatten(self, *, start_axis: int = 0, end_axis: int = -1) -> Value:
        return self._graph.flatten(self, start_axis=start_axis, end_axis=end_axis)
    def squeeze(self, axis: int) -> Value: return self._graph.squeeze(self, axis)
    def unsqueeze(self, axis: int) -> Value: return self._graph.unsqueeze(self, axis)
    def permute(self, axes: Sequence[int]) -> Value: return self._graph.permute(self, axes)
    def slice(self, *, axis: int, start: int = 0, end: int | None = None, step: int = 1) -> Value:
        return self._graph.slice(self, axis=axis, start=start, end=end, step=step)
    def windows(
        self,
        sizes: Shape,
        *,
        strides: Shape | None = None,
        dilations: Shape | None = None,
    ) -> Value:
        return self._graph.windows(self, sizes, strides=strides, dilations=dilations)


class Graph:
    """Mutable Python frontend for a ZGC semantic graph."""

    def __init__(self) -> None:
        self._nodes: list[Node] = []
        self._outputs: list[Value] = []
        self._source_names: set[str] = set()

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes)

    def input(self, name: str, dtype: dtype_enum, shape: Shape, *, binding: Binding = Binding.OWNED) -> Value:
        return self._source("input", name, dtype, shape, binding)

    def parameter(self, name: str, dtype: dtype_enum, shape: Shape, *, binding: Binding = Binding.OWNED) -> Value:
        return self._source("parameter", name, dtype, shape, binding)

    def constant(self, name: str, dtype: dtype_enum, shape: Shape, *, binding: Binding = Binding.OWNED) -> Value:
        return self._source("constant", name, dtype, shape, binding)

    def scalar(self, dtype: dtype_enum, value: Scalar) -> Value:
        return self._operation("scalar", dtype=dtype, value=_scalar(dtype, value))

    def full(self, dtype: dtype_enum, shape: Shape, value: Scalar) -> Value:
        return self._operation("full", dtype=dtype, shape=_shape(shape), value=_scalar(dtype, value))

    def relu(self, tensor: Value) -> Value: return self._operation("relu", tensor)
    def exp(self, tensor: Value) -> Value: return self._operation("exp", tensor)
    def neg(self, tensor: Value) -> Value: return self._operation("neg", tensor)
    def abs(self, tensor: Value) -> Value: return self._operation("abs", tensor)
    def sqrt(self, tensor: Value) -> Value: return self._operation("sqrt", tensor)
    def log(self, tensor: Value) -> Value: return self._operation("log", tensor)
    def reciprocal(self, tensor: Value) -> Value: return self._operation("reciprocal", tensor)
    def add(self, lhs: Value, rhs: Value) -> Value: return self._operation("add", lhs, rhs)
    def sub(self, lhs: Value, rhs: Value) -> Value: return self._operation("sub", lhs, rhs)
    def mul(self, lhs: Value, rhs: Value) -> Value: return self._operation("mul", lhs, rhs)
    def div(self, lhs: Value, rhs: Value) -> Value: return self._operation("div", lhs, rhs)
    def minimum(self, lhs: Value, rhs: Value) -> Value: return self._operation("minimum", lhs, rhs)
    def maximum(self, lhs: Value, rhs: Value) -> Value: return self._operation("maximum", lhs, rhs)
    def clamp(self, tensor: Value, lower: Value, upper: Value) -> Value:
        return self._operation("clamp", tensor, lower, upper)
    def equal(self, lhs: Value, rhs: Value) -> Value: return self._operation("equal", lhs, rhs)
    def not_equal(self, lhs: Value, rhs: Value) -> Value: return self._operation("not_equal", lhs, rhs)
    def less_than(self, lhs: Value, rhs: Value) -> Value: return self._operation("less_than", lhs, rhs)
    def less_equal(self, lhs: Value, rhs: Value) -> Value: return self._operation("less_equal", lhs, rhs)
    def greater_than(self, lhs: Value, rhs: Value) -> Value: return self._operation("greater_than", lhs, rhs)
    def greater_equal(self, lhs: Value, rhs: Value) -> Value: return self._operation("greater_equal", lhs, rhs)
    def logical_not(self, tensor: Value) -> Value: return self._operation("logical_not", tensor)
    def logical_and(self, lhs: Value, rhs: Value) -> Value: return self._operation("logical_and", lhs, rhs)
    def logical_or(self, lhs: Value, rhs: Value) -> Value: return self._operation("logical_or", lhs, rhs)
    def where(self, condition: Value, when_true: Value, when_false: Value) -> Value:
        return self._operation("where", condition, when_true, when_false)
    def copy(self, tensor: Value) -> Value: return self._operation("copy", tensor)
    def contiguous(self, tensor: Value) -> Value: return self._operation("contiguous", tensor)

    def pad(self, tensor: Value, fill: Value, *, before: Shape, after: Shape) -> Value:
        return self._operation("pad", tensor, fill, before=_shape(before), after=_shape(after))

    def shift(
        self,
        tensor: Value,
        offsets: Sequence[int],
        *,
        boundary: Boundary,
        fill: Value | None = None,
    ) -> Value:
        if not isinstance(boundary, Boundary):
            raise TypeError("invalid shift boundary")
        normalized = _integers(offsets, "shift offsets")
        if boundary is Boundary.CONSTANT and fill is None:
            raise ValueError("constant shift boundary requires a fill value")
        if boundary is not Boundary.CONSTANT and fill is not None:
            raise ValueError("shift fill is only valid for a constant boundary")
        return self._operation("shift", tensor, offsets=normalized, boundary=boundary, fill=fill)

    def slice_loop(self, tensor: Value, *, axis: int, iterations: Sequence[SliceIteration]) -> Value:
        normalized = tuple(iterations)
        if any(not isinstance(value, SliceIteration) for value in normalized):
            raise TypeError("slice loop iterations must be SliceIteration values")
        return self._operation("slice_loop", tensor, axis=axis, iterations=normalized)

    def matmul(self, lhs: Value, rhs: Value) -> Value: return self._operation("matmul", lhs, rhs)
    def sum(self, tensor: Value, *, axes: Axes = None, keep_dims: bool = False) -> Value:
        return self._reduction("sum", tensor, axes, keep_dims)
    def mean(self, tensor: Value, *, axes: Axes = None, keep_dims: bool = False) -> Value:
        return self._reduction("mean", tensor, axes, keep_dims)
    def min(self, tensor: Value, *, axes: Axes = None, keep_dims: bool = False) -> Value:
        return self._reduction("min", tensor, axes, keep_dims)
    def max(self, tensor: Value, *, axes: Axes = None, keep_dims: bool = False) -> Value:
        return self._reduction("max", tensor, axes, keep_dims)

    def concat(self, tensors: Sequence[Value], *, axis: int) -> Value:
        values = tuple(tensors)
        if not values:
            raise ValueError("concat requires at least one input")
        self._validate_values(values)
        return self._operation("concat", values, axis=axis)

    def softmax(self, tensor: Value, *, axis: int) -> Value:
        return self._operation("softmax", tensor, axis=axis)
    def transpose(self, tensor: Value, axis_a: int, axis_b: int) -> Value:
        return self._operation("transpose", tensor, axis_a=axis_a, axis_b=axis_b)
    def reshape(self, tensor: Value, shape: Shape) -> Value:
        return self._operation("reshape", tensor, shape=_shape(shape))
    def broadcast_to(self, tensor: Value, shape: Shape) -> Value:
        return self._operation("broadcast_to", tensor, shape=_shape(shape))
    def flatten(self, tensor: Value, *, start_axis: int = 0, end_axis: int = -1) -> Value:
        return self._operation("flatten", tensor, start_axis=start_axis, end_axis=end_axis)
    def squeeze(self, tensor: Value, axis: int) -> Value:
        return self._operation("squeeze", tensor, axis=axis)
    def unsqueeze(self, tensor: Value, axis: int) -> Value:
        return self._operation("unsqueeze", tensor, axis=axis)
    def permute(self, tensor: Value, axes: Sequence[int]) -> Value:
        return self._operation("permute", tensor, axes=_integers(axes, "axes"))
    def slice(self, tensor: Value, *, axis: int, start: int = 0, end: int | None = None, step: int = 1) -> Value:
        return self._operation("slice", tensor, axis=axis, start=start, end=end, step=step)
    def windows(
        self,
        tensor: Value,
        sizes: Shape,
        *,
        strides: Shape | None = None,
        dilations: Shape | None = None,
    ) -> Value:
        return self._operation(
            "windows",
            tensor,
            sizes=_shape(sizes),
            strides=None if strides is None else _shape(strides),
            dilations=None if dilations is None else _shape(dilations),
        )

    def outputs(self, *values: Value) -> None:
        if not values:
            raise ValueError("a graph must expose at least one output")
        self._validate_values(values)
        self._outputs = list(values)

    def freeze(self) -> FrozenGraph:
        if not self._outputs:
            raise ValueError("graph outputs have not been defined")
        return FrozenGraph(tuple(self._nodes), tuple(self._outputs))

    def serialize(self) -> str:
        return self.freeze().serialize()

    @property
    def fingerprint(self) -> str:
        return self.freeze().fingerprint

    def _source(self, operation: str, name: str, dtype: dtype_enum, shape: Shape, binding: Binding) -> Value:
        if not name:
            raise ValueError("source name must not be empty")
        if name in self._source_names:
            raise ValueError(f"source name is already defined: {name!r}")
        if not isinstance(dtype, dtype_enum):
            raise TypeError("source dtype must be a dtype")
        if not isinstance(binding, Binding):
            raise TypeError("source binding must be a Binding")
        if binding not in (Binding.OWNED, Binding.BOUND):
            raise ValueError("Python graph sources support owned or bound binding")
        self._source_names.add(name)
        return self._operation(operation, name=name, dtype=dtype, shape=_shape(shape), binding=binding)

    def _reduction(self, operation: str, tensor: Value, axes: Axes, keep_dims: bool) -> Value:
        if axes is None:
            normalized = None
        elif isinstance(axes, int) and not isinstance(axes, bool):
            normalized = (axes,)
        else:
            normalized = _integers(axes, "reduction axes")
            if not normalized:
                raise ValueError("reduction axes cannot be empty")
        return self._operation(operation, tensor, axes=normalized, keep_dims=keep_dims)

    def _operation(self, operation: str, *inputs: object, **attributes: object) -> Value:
        if operation not in OPERATION_NAMES:
            raise ValueError(f"unknown graph operation: {operation}")
        self._validate_nested_values(inputs)
        self._validate_nested_values(attributes.values())
        result = Value(self, len(self._nodes))
        self._nodes.append(Node(result.id, operation, tuple(inputs), tuple(attributes.items())))
        return result

    def _validate_values(self, values: Sequence[Value]) -> None:
        for value in values:
            if not isinstance(value, Value):
                raise TypeError("graph operands must be Value objects")
            if value.graph is not self:
                raise ValueError("values from different graphs cannot be combined")
            if value.id < 0 or value.id >= len(self._nodes):
                raise ValueError("value does not belong to this graph")

    def _validate_nested_values(self, values: object) -> None:
        if isinstance(values, Value):
            self._validate_values((values,))
        elif isinstance(values, dict):
            for value in values.values():
                self._validate_nested_values(value)
        elif isinstance(values, (tuple, list, dict_values)):
            for value in values:
                self._validate_nested_values(value)


# Runtime name used by dict.values() without treating strings as operand sequences.
dict_values = type({}.values())


def _shape(shape: Shape) -> tuple[int, ...]:
    normalized = tuple(shape)
    if any(not isinstance(extent, int) or isinstance(extent, bool) for extent in normalized):
        raise TypeError("shape extents must be integers")
    if any(extent <= 0 for extent in normalized):
        raise ValueError("shape extents must be greater than zero")
    return normalized


def _integers(values: Sequence[int], label: str) -> tuple[int, ...]:
    normalized = tuple(values)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in normalized):
        raise TypeError(f"{label} must be integers")
    return normalized


def _scalar(dtype: dtype_enum, value: Scalar) -> Scalar:
    if not isinstance(dtype, dtype_enum):
        raise TypeError("scalar dtype must be a dtype")
    if dtype is dtype_enum.bool:
        if not isinstance(value, bool):
            raise TypeError("bool scalar requires a bool value")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{dtype.name.lower()} scalar requires a numeric value")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("scalar values must be finite")
    if dtype is dtype_enum.i8:
        if not isinstance(value, int):
            raise TypeError("i8 scalar requires an integer value")
        if value < -128 or value > 127:
            raise ValueError("i8 scalar must be between -128 and 127")
    return value
