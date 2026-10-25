# Serialized graphs (`.zgir`)

`.zgir` is ZGC's versioned, language-neutral representation of a semantic
graph. A frontend can produce it without generating Zig source. The ZGC
compiler parses the graph at compile time, replays it through
`DefinitionBuilder`, validates it, and emits a specialized model library.
The resulting library does not parse `.zgir` at runtime.

## Format

The first line is the format header, `zgir <version-number>`. Each subsequent
assignment creates one value, numbered consecutively from `%0`. Arguments
that refer to values must name an earlier assignment. Positional arguments
come before named attributes. The `outputs` line lists the values exposed by
the compiled model, in order.

```text
zgir 1

%0 = input(name="input", dtype=f32, shape=[128, 64], binding=bound)
%1 = parameter(name="weight", dtype=f32, shape=[64, 10], binding=owned)
%2 = matmul(%0, %1)
%3 = relu(%2)

outputs = [%3]
```

Blank lines and full-line comments beginning with `#` are allowed after the
header. Source declarations may occur anywhere before their use. `input`,
`parameter`, and `constant` declare named sources; their `binding` is `owned`
or `bound`. `owned` sources receive copied values; `bound` sources borrow
external storage. See [model artifacts](executables.md#sources) for the
host-side binding contract.

Literals include quoted strings, integers, finite decimal numbers, `true`,
`false`, `null`, lists, and unquoted enum atoms. Dtypes are `f32`, `f16`, `i8`, `i32`, `u8`,
and `bool`. Shape and axis arguments are integer lists or integers as required
by each operation. The Python graph frontend emits the same syntax through
`Graph.serialize()`.

## Operations

Operation names mirror the semantic graph operations. These are the currently
recognized names:

| Group | Operations |
| --- | --- |
| Sources and values | `input`, `parameter`, `constant`, `scalar`, `full` |
| Elementwise math | `relu`, `exp`, `neg`, `abs`, `sqrt`, `log`, `reciprocal`, `add`, `sub`, `mul`, `div`, `minimum`, `maximum`, `clamp` |
| Predicates and selection | `equal`, `not_equal`, `less_than`, `less_equal`, `greater_than`, `greater_equal`, `logical_not`, `logical_and`, `logical_or`, `where` |
| Movement and structure | `copy`, `contiguous`, `pad`, `shift`, `slice_loop`, `concat`, `transpose`, `reshape`, `broadcast_to`, `flatten`, `squeeze`, `unsqueeze`, `permute`, `slice`, `windows` |
| Contraction and reduction | `matmul`, `sum`, `mean`, `min`, `max`, `softmax` |

Unary and binary operations take their input references as positional
arguments, for example `relu(%0)` or `add(%0, %1)`. Operation-specific options
are named attributes:

```text
%4 = sum(%3, axes=[1], keep_dims=false)
%5 = concat([%4, %4], axis=0)
%6 = shift(%5, offsets=[1], boundary=wrap)
```

For reductions, `axes=null` selects the default reduction axes. `shift`
accepts `wrap`, `edge`, `reflect`, or `constant`; the `constant` boundary also
requires a `fill=%id` reference. `slice_loop` uses an `iterations` list of
`iteration(offsets=[...], boundary=...)` values; a boundary may be
`redirect(n)` to use another iteration's result. Shape, dtype, rank, and
operation constraints are checked during Zig specialization. See
[operation semantics](operations.md) for those contracts.

## Compile and cache

Compile a file directly with the native CLI:

```sh
zgc compile graph.zgir -o graph.dylib
zgc compile graph.zgir --selection-target fast -o graph-fast.dylib
```

The Python compiler can compile a constructed graph, `.zgir` text, or a file:

```python
from zgc_compiler import Compiler

compiler = Compiler()
artifact = compiler.compile("graph.zgir")
# compiler.compile(graph)      # Python Graph or FrozenGraph
# compiler.compile(zgir_text)  # serialized text
```

The Python package serializes graphs and passes text through unchanged; Zig
owns parsing and validation. The Python compiler caches graph adapters by the
SHA-256 digest of the UTF-8 serialization. For supplied text, whitespace and
comments therefore affect cache identity even when graph semantics are the
same. See [generated executable artifacts](executables.md) for the ABI and
execution path.
