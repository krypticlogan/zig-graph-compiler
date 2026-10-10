from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

from ..errors import GraphParseError
from .graph import OPERATION_NAMES, FrozenGraph, Graph, Value
from .serialization import GRAPH_FORMAT_VERSION
from .types import Binding, Boundary, Redirect, SliceIteration, dtype

_HEADER = re.compile(r"^zgir\s+(\d+)\s*(?:#.*)?$")
_ASSIGNMENT = re.compile(r"^%(\d+)\s*=\s*(.+)$")
_VALUE_NAME = re.compile(r"^__zgc_value_(\d+)$")

_ATOMS: dict[str, object] = {
    "f32": dtype.f32,
    "f16": dtype.f16,
    "i8": dtype.i8,
    "bool": dtype.bool,
    "owned": Binding.OWNED,
    "bound": Binding.BOUND,
    "wrap": Boundary.WRAP,
    "edge": Boundary.EDGE,
    "reflect": Boundary.REFLECT,
    "constant": Boundary.CONSTANT,
    "true": True,
    "false": False,
    "null": None,
}


def parse_graph(text: str) -> FrozenGraph:
    lines = text.splitlines()
    header_index = _next_content_line(lines, 0)
    if header_index is None:
        raise GraphParseError("missing graph header", line=1, column=1)
    match = _HEADER.fullmatch(lines[header_index].strip())
    if match is None:
        raise GraphParseError("expected 'zgir <version>'", line=header_index + 1, column=1)
    version = int(match.group(1))
    if version != GRAPH_FORMAT_VERSION:
        raise GraphParseError(
            f"unsupported graph format version {version}",
            line=header_index + 1,
            column=1,
        )

    graph = Graph()
    values: list[Value] = []
    outputs_seen = False
    for index in range(header_index + 1, len(lines)):
        source = lines[index].strip()
        if not source or source.startswith("#"):
            continue
        if outputs_seen:
            raise GraphParseError("outputs must be the final statement", line=index + 1, column=1)
        if source.startswith("outputs"):
            name, separator, expression = source.partition("=")
            if not separator or name.strip() != "outputs":
                raise GraphParseError("invalid outputs statement", line=index + 1, column=1)
            decoded = _expression(expression.strip(), values, index + 1)
            if not isinstance(decoded, list) or not decoded:
                raise GraphParseError("outputs must be a non-empty value list", line=index + 1, column=1)
            if any(not isinstance(value, Value) for value in decoded):
                raise GraphParseError("outputs may only contain value references", line=index + 1, column=1)
            graph.outputs(*decoded)
            outputs_seen = True
            continue

        assignment = _ASSIGNMENT.fullmatch(source)
        if assignment is None:
            raise GraphParseError("expected a value assignment", line=index + 1, column=1)
        value_id = int(assignment.group(1))
        if value_id != len(values):
            raise GraphParseError(
                f"expected value %{len(values)}, received %{value_id}",
                line=index + 1,
                column=1,
            )
        operation, arguments, attributes = _operation(assignment.group(2), values, index + 1)
        try:
            result = getattr(graph, operation)(*arguments, **attributes)
        except (TypeError, ValueError) as error:
            raise GraphParseError(str(error), line=index + 1, column=1) from error
        values.append(result)

    if not outputs_seen:
        raise GraphParseError("missing outputs statement", line=len(lines) or 1, column=1)
    return graph.freeze()


def parse_graph_file(path: str | Path) -> FrozenGraph:
    location = Path(path)
    try:
        return parse_graph(location.read_text(encoding="utf-8"))
    except OSError as error:
        raise GraphParseError(str(error), line=1, column=1) from error


def _operation(source: str, values: list[Value], line: int) -> tuple[str, list[object], dict[str, object]]:
    expression = _syntax_tree(_normalize_references(source, line), line)
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise GraphParseError("assignment must contain an operation call", line=line, column=1)
    name = expression.func.id
    if name not in OPERATION_NAMES:
        raise GraphParseError(f"unknown operation {name!r}", line=line, column=1)
    arguments = [_decode(argument, values, line) for argument in expression.args]
    attributes: dict[str, object] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise GraphParseError("expanded keyword arguments are not supported", line=line, column=1)
        if keyword.arg in attributes:
            raise GraphParseError(f"duplicate argument {keyword.arg!r}", line=line, column=1)
        attributes[keyword.arg] = _decode(keyword.value, values, line)
    return name, arguments, attributes


def _expression(source: str, values: list[Value], line: int) -> object:
    return _decode(_syntax_tree(_normalize_references(source, line), line), values, line)


def _syntax_tree(source: str, line: int) -> ast.expr:
    try:
        return ast.parse(source, mode="eval").body
    except SyntaxError as error:
        raise GraphParseError(
            error.msg,
            line=line,
            column=(error.offset or 1),
        ) from error


def _normalize_references(source: str, line: int) -> str:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError as error:
        raise GraphParseError(str(error.args[0]), line=line, column=1) from error
    result: list[tokenize.TokenInfo] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == tokenize.OP and token.string == "%":
            if index + 1 >= len(tokens) or tokens[index + 1].type != tokenize.NUMBER:
                raise GraphParseError("'%' must introduce a numeric value reference", line=line, column=token.start[1] + 1)
            reference = tokens[index + 1]
            if not reference.string.isdigit():
                raise GraphParseError("value reference must be an integer", line=line, column=reference.start[1] + 1)
            result.append(tokenize.TokenInfo(tokenize.NAME, f"__zgc_value_{reference.string}", token.start, reference.end, token.line))
            index += 2
            continue
        result.append(token)
        index += 1
    return tokenize.untokenize(result)


def _decode(expression: ast.expr, values: list[Value], line: int) -> object:
    if isinstance(expression, ast.Constant):
        if expression.value is None or isinstance(expression.value, (str, bool, int, float)):
            return expression.value
        raise GraphParseError("unsupported literal", line=line, column=expression.col_offset + 1)
    if isinstance(expression, ast.Name):
        reference = _VALUE_NAME.fullmatch(expression.id)
        if reference is not None:
            value_id = int(reference.group(1))
            if value_id >= len(values):
                raise GraphParseError(
                    f"value %{value_id} has not been defined",
                    line=line,
                    column=expression.col_offset + 1,
                )
            return values[value_id]
        try:
            return _ATOMS[expression.id]
        except KeyError as error:
            raise GraphParseError(
                f"unknown literal {expression.id!r}",
                line=line,
                column=expression.col_offset + 1,
            ) from error
    if isinstance(expression, (ast.List, ast.Tuple)):
        return [_decode(item, values, line) for item in expression.elts]
    if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.USub):
        operand = _decode(expression.operand, values, line)
        if isinstance(operand, bool) or not isinstance(operand, (int, float)):
            raise GraphParseError("invalid negative literal", line=line, column=expression.col_offset + 1)
        return -operand
    if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
        if expression.func.id == "redirect":
            if len(expression.args) != 1 or expression.keywords:
                raise GraphParseError("redirect requires one iteration", line=line, column=expression.col_offset + 1)
            iteration = _decode(expression.args[0], values, line)
            if not isinstance(iteration, int) or isinstance(iteration, bool):
                raise GraphParseError("redirect iteration must be an integer", line=line, column=expression.col_offset + 1)
            return Redirect(iteration)
        if expression.func.id == "iteration":
            if expression.args or any(keyword.arg is None for keyword in expression.keywords):
                raise GraphParseError("iteration fields must be named", line=line, column=expression.col_offset + 1)
            fields = {keyword.arg: _decode(keyword.value, values, line) for keyword in expression.keywords if keyword.arg is not None}
            if set(fields) != {"offsets", "boundary"}:
                raise GraphParseError("iteration requires offsets and boundary", line=line, column=expression.col_offset + 1)
            try:
                return SliceIteration(fields["offsets"], fields["boundary"])
            except (TypeError, ValueError) as error:
                raise GraphParseError(str(error), line=line, column=expression.col_offset + 1) from error
    raise GraphParseError("unsupported graph expression", line=line, column=expression.col_offset + 1)


def _next_content_line(lines: list[str], start: int) -> int | None:
    for index in range(start, len(lines)):
        stripped = lines[index].strip()
        if stripped and not stripped.startswith("#"):
            return index
    return None
