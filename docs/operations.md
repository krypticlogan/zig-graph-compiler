# Operation semantics

All model shapes, axes, dtypes, and operation attributes are compile-time
values. Inputs are read-only and compute operations write a distinct output.
View operations alias an existing storage root and execute no runtime kernel.

Tensor dimensions must be positive. Operations preserve their input dtype
unless stated otherwise. Supported dtypes are `f32`, `f16`, `i8`, and `bool`.

## Definition API types

The model-definition surface uses concrete parameter types so editor tooling
can expose accepted fields and scalar types:

| Methods                     | Parameter contract                                                                       |
| --------------------------- | ---------------------------------------------------------------------------------------- |
| `scalar`, `full`            | `value` has the scalar type selected by the compile-time `dtype` argument                |
| `sum`, `mean`, `min`, `max` | `ReductionOptions` with `axes: ?[]const i8` and `keep_dims: bool`                        |
| `flatten`                   | `FlattenOptions`                                                                         |
| `slice`                     | `SliceOptions`                                                                           |
| `pad`                       | `PadOptions`                                                                             |
| `windows`                   | `WindowOptions`                                                                          |
| `modelWith`                 | `[]const SourceOverride`, whose entries contain a source-enum value and `Source.Binding` |

Options may use inferred struct literals because the method signature supplies
their concrete type. For example, a single-axis reduction is
`builder.sum(tensor, .{ .axes = &.{1} })`; omitting `axes` reduces every axis.

## Scalars and filled tensors

`scalar(dtype, value)` creates an immutable, source-free rank-zero tensor whose
value is embedded in the generated program. It reserves no mutable model memory
and executes no runtime kernel.

`full(dtype, shape, value)` creates that scalar and broadcasts it to the target
shape. The result is a read-only zero-stride alias backed by one scalar element,
not a materialized array. `broadcastTo(tensor, shape)` exposes the same
trailing-axis expansion for any broadcast-compatible tensor.

## Elementwise operations

`relu`, `exp`, `neg`, `abs`, `sqrt`, `log`, `reciprocal`, `add`, `sub`, `mul`,
`div`, `minimum`, `maximum`, and `clamp` operate elementwise. `exp`, `neg`,
`abs`, `sqrt`, `log`, `reciprocal`, and `div` require floating-point tensors.
`relu`, arithmetic other than division, minimum, maximum, and clamp support
the implemented numeric dtypes. Boolean tensors are not numeric.

Binary operands must have matching dtypes and use trailing-axis broadcasting:

- equal aligned extents are paired directly;
- an extent of one broadcasts across the other operand;
- absent leading axes behave as singleton dimensions;
- rank-zero tensors broadcast as scalars.

Broadcasting is represented with zero input strides and does not materialize
expanded operands.

`minimum(lhs, rhs)` and `maximum(lhs, rhs)` are elementwise and are distinct
from the `min` and `max` reduction operations. `clamp(value, lower, upper)`
broadcasts all three tensors.

## Comparisons, booleans, and selection

`equal`, `notEqual`, `lessThan`, `lessEqual`, `greaterThan`, and
`greaterEqual` broadcast their operands and produce a `bool` tensor. Equality
and inequality support matching numeric or boolean operands. Ordered
comparisons require numeric operands.

`logicalNot`, `logicalAnd`, and `logicalOr` accept only boolean tensors.
Numeric tensors have no implicit truthiness: use an explicit comparison such
as `notEqual(value, scalar(dtype, 0))` to construct a mask.

`where(condition, when_true, when_false)` requires a boolean condition and
matching value dtypes. The condition and both value tensors use trailing-axis
broadcasting. Its result has the dtype of the value tensors.

## Materialization

`copy(tensor)` writes logical tensor values into distinct storage. Lowering may
retain a useful physical layout such as a batch-oriented layout.

`contiguous(tensor)` also writes into distinct storage and requires the result
to use logical row-major strides. Both operations preserve shape and dtype and
can materialize transposed, sliced, or broadcast views.

## Padding, shifting, and windows

`pad(tensor, fill, options)` materializes constant padding around every input
axis. `fill` must be a rank-zero tensor with the input dtype. `before` and
`after` provide one compile-time width per input axis, and the result uses
logical row-major storage.

`shift(tensor, offsets, boundary)` translates values along every axis while
preserving shape and dtype. `offsets` contains one compile-time signed value per
axis; a positive offset moves an input value toward a higher output coordinate.
Out-of-bounds source coordinates use one of four boundary modes: `wrap` (periodic),
`edge` (repeat the nearest edge value), `reflect` (mirror without repeating the
edge value), or `constant` (read a rank-zero fill tensor with the input dtype).
For example, `shift(x, &.{1}, .wrap)` turns `[a,b,c]` into `[c,a,b]`;
`shift(x, &.{1}, .reflect)` turns it into `[b,a,b]`. A size-one reflected axis
always reads its only element. Shift results use distinct row-major storage.

`windows(tensor, options)` creates one overlapping view across the trailing
axes selected by `sizes`. The corresponding `strides` and `dilations` default
to one. For an input `[H, W]` and window sizes `[KH, KW]`, the output is
`[OH, OW, KH, KW]`. Unwindowed leading axes remain ahead of the output-position
axes, and the window axes are appended.

Windows use valid geometry: every dilated window must fit within its input.
Window boundary behavior can be expressed by padding the input first. The complete window
tensor aliases one storage root; individual windows do not allocate storage.

## Matrix multiplication

`matmul` accepts two rank-two `f32` tensors with logical shapes `[M, K]` and
`[K, N]`, producing `[M, N]`. Lowering selects the traversal strategy and may
pack parameter or constant right-hand operands into an output-major physical
layout.

## Reductions

`sum`, `mean`, `min`, and `max` accept `ReductionOptions`. The `axes` field is
a slice containing one or more axes; its default value of `null` selects every
axis. Negative axes are normalized relative to the input rank; duplicate,
empty, and out-of-range axis sets are rejected. Reduced axes are removed by
default or retained with extent one when `keep_dims` is true.

`mean` requires a floating-point tensor. `sum`, `min`, and `max` support the
implemented floating-point and signed-integer dtypes. Reduction geometry and
axis traversal are fixed in the generated kernel.

`softmax` accepts a floating-point tensor and one axis. It preserves shape and
uses a numerically stable shifted exponential calculation.

## Concatenation

`concat` accepts one or more tensors with matching ranks and dtypes. All
extents outside the selected axis must match. The output extent on that axis is
the sum of the corresponding input extents.

Concatenation semantically produces a new contiguous tensor. Its direct
implementation uses fixed block copies for contiguous inputs and static
strided traversal otherwise. Executable search may instead compose nested
concatenations and compatible shifts into one segmented remap that writes leaf
views directly into the final output, leaving intermediate concatenation
results unmaterialized.

## Structural views

Structural operations alias their source storage:

- `transpose` exchanges two axes and their strides.
- `permute` validates a complete unique axis ordering and lowers it to
  transpose aliases.
- `reshape` preserves element count and requires a logically row-major
  contiguous input.
- `flatten` collapses an inclusive axis range and requires only that range to
  be logically contiguous.
- `squeeze` removes a selected extent-one axis.
- `unsqueeze` inserts an extent-one axis at the selected position.
- `slice` selects a non-empty range with compile-time positive bounds and
  step, producing an offset strided alias.
- `broadcastTo` introduces leading or singleton expansion axes with zero
  strides.

Negative axes are accepted by definition-builder methods and normalized before
graph validation. Structural views receive no independent memory-plan region.
