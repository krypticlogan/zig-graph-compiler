# Zig Graph Compiler

ZGC is an allocation-free ahead-of-time tensor computation graph compiler written in Zig.
A graph architecture is defined at compile time,
lowered and optimized to a fixed execution graph,
assigned an inline memory plan,
and emitted as a specialized Zig type.

The binary is the model: graph traversal, tensor ranks, shapes, dtypes, layouts,
and kernel selection are compile-time-known.

The project targets Zig 0.16.0.

## Capabilities

- Front-facing `DefinitionBuilder` with enum-indexed sources.
- `definition.model()` for semantic graph construction, validation, analysis,
  optimization, and executable-model generation.
- Exact graph capacities derived directly from the model.
- Lifetime-planned, reusable inline model memory with no heap allocation during
  execution.
- Compact optimized semantic graphs with dead-node elimination, provenance,
  structural canonicalization, and scalar folding.
- Multiple graph inputs, parameters, constants, and outputs.
- Source-free scalar literals and storage-efficient zero-stride filled tensors.
- `f32`, `f16`, `i8`, and strict boolean tensor dtypes.
- Unary math, arithmetic, comparison, logical, selection, matmul, reduction,
  softmax, and concatenation operations.
- Explicit copy and row-major contiguous materialization.
- Zero-copy overlapping windows over views and shape-preserving shifts with
  wrap, edge, reflect, or constant boundaries.
- Optional segmented remap lowering for composed shift, slice, and
  concatenation data movement.
- Static-geometry model views and dynamic low-level views for contiguous,
  offset, broadcast, transposed, and generally strided layouts.
- Transpose, permutation, reshape, flatten, squeeze, unsqueeze, and static slicing view operations.
- Trailing-axis broadcasting for binary arithmetic and compile-time single- or multi-axis reductions.
- SIMD fast paths for contiguous kernels and generic strided traversal.
- Core-backed extensions and abstractions like `nn` and `img`.
- Operation and generated-model benchmarks, plus standalone example packages.
- Lean binaries and first-class inspection for any compiled tensor graph.

See [development state](docs/development-state.md) for precise limitations and
[architecture](docs/architecture.md) for the compilation pipeline.

> [!Note]
> ZGC is under active development. Public APIs and compiled-model contracts should be expected to
> change before a stable release.

## Installation & usage

In a Zig project directory:

```bash
zig fetch --save git+https://github.com/krypticlogan/zgc
```

Add to your build.zig:

```zig
const zgc_dep = b.dependency("zgc", .{
    .target = target,
    .optimize = optimize,
});
const zgc_mod = zgc_dep.module("zgc");
```

Then, add the module as an import to your own module.

```zig
exe.root_module.addImport("zgc", zgc_mod);
```

Finally, you may import the zgc module to your own.

## Defining a model

`DefinitionBuilder` is the public model-building surface. Source keys and tensor
values are concrete types. User code does not run separately against counting
and graph builders.

```zig
const std = @import("std");
const zgc = @import("zgc");

const Sources = enum(usize) { input, weights }; // user-defined source keys
const Definition = zgc.DefinitionBuilder(Sources, .{ .max_rank = 2 });

fn define(builder: *Definition) void { // complete graph architecture is defined here
    const input = builder.input(.input, .f32, &.{ 4, 8 });
    const weights = builder.parameter(.weights, .f32, &.{ 8, 16 });
    builder.output(
        builder.relu(
            builder.matmul(input, weights)
        )
    );
}

const definition = blk: {
    var builder = Definition.init();
    define(&builder);
    break :blk builder.finish();
};

const MyModel = definition.model(); // graph is constructed and lowered to a specialized Zig type

pub fn main() !void {
    var model = MyModel.init();

    const input_values: [4 * 8]f32 = @splat(1);
    const weight_values: [8 * 16]f32 = @splat(0.25);
    try model.copyInput(.input, &input_values);
    try model.copySource(.weights, &weight_values);
    model.run();

    const output = model.outputView(0);
    std.debug.print("shape={any} data={any}\n", .{ output.shape, output.storage });
}
```

Definition limits have defaults and may be overridden at compile time. These
are **front-end bounds, not final allocation sizes**. The counting pass derives the
exact node, tensor, reference, output, source, and rank capacities before graph
construction and memory planning.

## Domain extensions

### Neural-network layers

`zgc.ext.nn` composes higher-level layers through `DefinitionBuilder`; it does not
provide a separate tensor runtime.

```zig
const Dense = zgc.ext.nn.Dense(Sources);
const Classifier = zgc.ext.nn.Sequential(&[_]Dense{
    .{ .weights = .w1, .bias = .b1, .output_size = 16, .activation = .relu },
    .{ .weights = .w2, .bias = .b2, .output_size = 10, .activation = .softmax },
});

const input = builder.input(.input, .f32, &.{ batch_size, input_size });
builder.output(Classifier.apply(builder, input));
```

Dense weights use logical `[input, output]` storage by default. Set
`.weight_layout = .output_input` for sources stored as `[output, input]`; the
layer adds an aliasing transpose before matmul.

### Image-processing pipelines

`zgc.ext.img.Dimensions` defines channel-first or channel-last rank-4 shapes,
and `zgc.ext.img.input` declares a core graph input with that convention.

## Source storage

Sources use model-owned storage by default. Runtime inputs can be copied into
that storage with `copyInput`, while owned parameters and constants use the
typed `copySource` API shown above.

Parameters and constants may instead be embedded directly into the program:

```zig
const EmbeddedModel = definition.modelWith(&.{
    .{ .source = .weights, .binding = zgc.memory.Source.embed(@embedFile("weights.bin")) },
});
```

The required byte length is derived from the source tensor's compile-time dtype
and shape and checked during compilation. `Source.embed` accepts raw logical
row-major, native-endian tensor data and packs it at compile time when lowering
selects another physical layout. `Source.embedPacked` accepts bytes already in
the layout reported by `Model.sourceLayout`. Embedded values remain read-only
and do not receive a region in the model's mutable memory plan.

Inputs can also borrow caller-owned runtime storage without a copy:

```zig
const BorrowingModel = definition.modelWith(&.{
    .{ .source = .input, .binding = zgc.memory.Source.bound },
    .{ .source = .weights, .binding = zgc.memory.Source.embed(@embedFile("weights.bin")) },
});

var model = BorrowingModel.init();
try model.bindInput(.input, runtime_values);
model.run();
```

The bound slice must:

- use the physical order reported by `BorrowingModel.sourceLayout(.input)`
- remain alive and unchanged while `run()` is executing.

It may be updated or rebound between runs. `copyInput` accepts logical
row-major values and packs them when lowering selects another layout.

Unspecified sources use model-owned storage, so `definition.model()` is
equivalent to an all-owned source plan.

## Build and test

```sh
zig build test
zig build check
```

The package exports a module named `zgc`. The [examples](examples/) show how
separate Zig packages consume it through `b.dependency("zgc", ...)`.

## Benchmarks

Run the complete suite in `ReleaseFast`:

```sh
zig build benchmark -Doptimize=ReleaseFast
```

Select an individual case with `-Dop`, for example:

```sh
zig build benchmark -Dop=matmul-rhs-strided -Doptimize=ReleaseFast
```

See the [benchmark dashboard](benchmarks/README.md) for selectors, methodology,
and recorded results.

## Examples of ZGC graphs in use

### Conway's Game of Life

https://github.com/user-attachments/assets/4764d198-9650-4db1-b541-27d46fec85e2

### MNIST digit classifier

https://github.com/user-attachments/assets/1c3cbe07-377e-42a5-aab2-9fd6ec340f38

## Repository layout

| Path                    | Purpose                                                                                            |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| `src/zgc/frontend/`     | Public definition types and the typed graph-building surface                                        |
| `src/zgc/compiler/`     | Construction, validation, analysis, optimization, planning, scheduling, and executable search       |
| `src/zgc/core/`         | Graph, tensor/view, model, and inspection structures                                                |
| `src/zgc/execution/`    | Executable programs, kernel plans, and dispatch                                                     |
| `src/zgc/operations/`   | Semantic operation definitions and reusable operation-family descriptors                           |
| `src/zgc/kernels/`      | Elementwise, reduction, contraction, layout, and special kernels                                   |
| `src/zgc/storage/`      | Dtypes, source policies, and memory planning                                                        |
| `src/cli/`              | Model-specific command-line entry points                                                           |
| `src/artifact/`         | Generated-model artifact entry points                                                              |
| `src/extensions/`       | Optional domain abstractions exported through `zgc.ext.nn` and `zgc.ext.img`                       |
| `tests/`                | Compile-time graph, runtime model, validation, view, and kernel coverage                           |
| `benchmarks/`           | Operation and generated-model benchmark harness, with recorded results                             |
| `examples/`             | Standalone model definitions, interactive applications, inspection, and artifact analysis          |
| `playground/`           | Standalone scratch package for graph construction, execution, testing, inspection, and disassembly |
| `docs/`                 | Architecture, design constraints, capabilities, and limitations                                    |

## Documentation

- [Documentation index](docs/README.md)
- [Architecture and compilation pipeline](docs/architecture.md)
- [Design constraints](docs/design-constraints.md)
- [Model inspection](docs/inspection.md)
- [Generated model artifacts](docs/model-artifacts.md)
- [Development state](docs/development-state.md)
- [MNIST digit classifier](examples/mnist-digit-classifier/README.md)
- [Conway's Game of Life](examples/conways-game-of-life/README.md)
- [Graph playground](playground/README.md)
- [Benchmarks](benchmarks/README.md)
