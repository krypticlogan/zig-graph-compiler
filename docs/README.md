# Documentation

These documents describe the public model-construction flow, compiler pipeline,
runtime storage, supported operations, constraints, and tooling.

Run `zig build docs` to generate the Zig API reference at
`zig-out/docs/api/index.html`.

- [Architecture](architecture.md) — definition, semantic graphs, executable
  candidates, validation, memory planning, and generated model execution.
- [Design constraints](design-constraints.md) — compile-time specialization,
  source ownership, execution, and memory invariants.
- [Operation semantics](operations.md) — shape, dtype, broadcasting,
  reduction, concatenation, and structural-view contracts.
- [Executable inspection](inspection.md) — inspect compiled libraries or render
  model representations to a writer.
- [Generated model artifacts](executables.md) — opaque model lifecycle,
  versioned C ABI, libraries, and minimal executables.
- [Serialized graphs](zgir.md) — `.zgir` syntax, operations, and compilation.
- [Python compiler package](executables.md#python-compiler) — external Zig
  toolchains, compiled-model caching, and the runtime/compiler boundary.
- [Development state](development-state.md) — implemented capabilities,
  limitations, and tests.
- [Repository README](../README.md) — public API example and common commands.
- [MNIST digit classifier](../examples/mnist-digit-classifier/README.md) —
  embedded parameters, interactive inference, inspection, and generated-code
  analysis.
- [Conway's Game of Life](../examples/conways-game-of-life/README.md) — static
  window geometry, externally managed state, and interactive rendering.
- [Benchmarks](../benchmarks/README.md) — reference workloads and timing notes.
