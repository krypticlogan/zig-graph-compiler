# Design constraints

ZGC specializes a model from compile-time graph metadata while keeping runtime
data ownership explicit. The following constraints define model construction
and execution.

## Compile-time model structure

- Architecture, tensor ranks, shapes, dtypes, layouts, and operation order are
  known at compile time.
- Tensor dimensions are positive; zero-length dimensions are rejected during
  definition.
- Definition bounds limit front-end construction. The counting pass derives the
  exact capacities used by graph lowering and memory planning.
- Shape and operation compatibility errors are reported during compilation when
  their inputs are statically known.
- Semantic validation checks the constructed graph before optimization and
  analysis. Semantic analysis derives use, output, consumer-edge, and
  topological-order facts. Fusion, layout, and remap analysis only advertise
  legal alternatives; executable search owns selection and lowering.
- Executable search composes compatible analysis regions with early ownership
  checks, retains an unfused generic reference candidate, and keeps a bounded
  Pareto frontier of scheduled executable alternatives.
- Raw and semantic graphs, the active executable, the reference candidate,
  and executable-candidate selection metadata remain available for inspection. The
  selected candidate supplies the executable used for model generation.
- The generated model type contains a fixed executable program and memory plan;
  execution does not interpret or allocate graph nodes.

## Source ownership

Every graph source has one storage policy. `definition.modelWith` accepts a
typed slice of `.source` and `.binding` overrides:

```zig
const Model = definition.modelWith(&.{
    .{ .source = .input, .binding = zgc.memory.Source.bound },
    .{ .source = .weights, .binding = zgc.memory.Source.embed(weights_bytes) },
});
```

- Owned sources receive an aligned region in model memory and are populated
  through `copyInput` or `copySource`. Values use logical row-major order and
  are packed into the compiled physical layout.
- Bound inputs borrow a caller-owned slice through `bindInput`. The slice must
  use the layout reported by `sourceLayout`, and remain valid and unchanged for
  the duration of `run()`.
- Embedded parameters and constants use read-only program data.
  `Source.embed`, commonly used with `@embedFile`, accepts logical row-major
  bytes and compile-time packs them into the selected layout;
  `Source.embedPacked` accepts bytes already in that physical layout.

Unspecified policies are owned. Bound and embedded sources do not consume space
in the model's mutable memory plan.

## Views and execution

- Operations receive read-only input views and write only to their designated
  output storage.
- Layout-changing graph operations such as transpose create aliases rather than
  copying tensor data.
- Reshape, flatten, squeeze, and unsqueeze are aliasing views. Reshape requires
  logical row-major contiguity, while flatten requires contiguity only within
  its collapsed axis range.
- Axis permutation lowers to transpose aliases. Static slices use positive
  bounds and steps and retain the source storage root.
- Generated-model views carry shape, strides, base offset, element count, and
  layout traits in their types. Their runtime state contains storage and any
  cursor offset introduced by runtime-selected subviews.
- Dynamic views retain runtime geometry for explicit low-level use.
- Optimization may choose a first-axis-contiguous physical layout for eligible
  rank-2 matmuls and propagate it through compatible dense operations.
- Matmul parameter and constant right-hand sides retain logical `[K, N]` shape
  while optimization may store them output-major with physical strides `[1, K]`.
- Generated matmuls carry a compile-time contraction plan selected during
  contraction planning from a candidate's concrete layouts. Semantic matmul
  nodes contain no kernel plan.
- Kernel-local axis ordering, vectorization, accumulator lanes, and unrolling
  form a `TraversalPlan`; model-level ordering is the executable's node sequence.
- A candidate `Schedule` is a legal topological ordering of an executable.
  Initial variants preserve existing order, reduce memory pressure, or favor
  critical-path work.
- Every candidate is finalized through lifetime and memory planning before its
  structured runtime, memory, code-size, scratch, and conversion costs are
  compared. Dominated candidates are removed from a bounded Pareto frontier.
- Kernel plans are inert compile-time data. `ExecutableCompute` owns dispatch
  to map, reduction, and contraction kernel families.
- Executable invocations contain flattened input and output references and may
  represent multiple stores. Sibling reduction fusion uses this representation
  to write independent reduction results from one traversal.
- Fused elementwise programs contain compile-time operation descriptors and
  references. Kernels unroll those programs and do not interpret opcodes at
  runtime.
- Concatenation materializes distinct contiguous output storage. Its axis and
  input geometry are validated and specialized before execution.
- Filled tensors alias one scalar storage element through zero strides. They do
  not allocate or initialize storage proportional to their logical shape.
- `copy` and `contiguous` are compute operations with distinct output storage.
  `copy` permits a lowering-selected physical layout; `contiguous` fixes
  logical row-major output strides.
- Constant padding is materialized once per padded tensor. A window operation
  is an overlapping read-only alias with appended window axes and no separate
  allocation for individual windows.
- Shifts materialize one shape-preserving output. Axis offsets and boundary mode
  are compile-time attributes; constant fill may be a runtime scalar input.
- Comparisons produce boolean tensors. Logical operations and selection
  conditions require boolean tensors; numeric values have no implicit
  truthiness conversion.
- `run()` executes the selected fixed schedule sequentially. Compile-time node
  and plan dispatch are inlined into this graph-specific executable; substantial
  kernels remain valid function boundaries. Runtime input values may change
  between runs without rebuilding the model type.

## Memory

- A model instance owns one inline, aligned mutable byte array.
- The compile-time memory plan assigns regions to owned source and result
  tensors; aliases do not receive separate regions.
- Model-owned sources and graph outputs retain persistent regions.
- Intermediate regions may overlap when their validated half-open lifetimes do
  not overlap. Free spans are alignment-aware, split when partially consumed,
  and coalesced when adjacent.
- Heap allocation is not required for model initialization or execution.

See [architecture](architecture.md) for the compilation pipeline and
[development state](development-state.md) for supported operations and current
limitations.
