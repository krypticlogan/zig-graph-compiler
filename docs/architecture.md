# Architecture

ZGC specializes a statically defined tensor program into an executable Zig
model type. Definition bounds make compile-time storage possible; a counting
pass then removes unused capacity before the graph and model are materialized.

```text
typed model function
        │
        ▼
DefinitionBuilder ──► completed definition
                            │
                            ▼
                      Construction
                exact capacities + raw graph
                            │
                            ▼
                  SemanticValidation
                            │
                            ▼
                  SemanticOptimization
             canonicalization + liveness + compaction
                            │
                            ▼
                 Final semantic validation
                            │
                            ▼
                     SemanticAnalysis
             uses + dependency topology
       ┌────────────┬───────────┴──────────┐
       ▼            ▼                      ▼
 FusionAnalysis LayoutAnalysis       RemapAnalysis
 candidate      candidate            data-transfer
 regions        regimes              regions
       └────────────┴───────────┬──────────┘
                      ▼
               ExecutableSearch ◄── generic reference
             combine + lower + schedule
             bounded PlanCandidate frontier
                            │
                            ▼
       FinalValidation + LifetimeAnalysis + MemoryPlan
                  exact candidate costs
                            │
                            ▼
                Pareto pruning and selection
                            │
                            ▼
                    Executable type with Program and Runtime
```

Calling `definition.executable()` performs every stage after definition. Counting
and compiler-analysis steps are internal implementation details and are not
exported as public graph-building APIs.

[Comptime diagnostics](compile-diagnostics.md) provide optional stage logs and
intentional checkpoints for investigating compiler time and memory use.

## Definition

`DefinitionBuilder` is the front end. Its operation
methods consume and return the concrete `Value` type, whose metadata contains
an ID, dtype, and `Shape`. Build shapes retain their literal dimensions
without imposing a maximum rank.

Named sources are introduced through `builder.sources(SourceKey)`. `SourceKey`
must be an enum, giving every input, parameter, or constant a stable
compile-time index without specializing the builder itself. The builder records
that type when the source facade is requested. Definitions without sources do
not need to declare an empty source enum.

The concrete builder appends nodes, tensors, input references, and outputs into
64-entry chunks. Each collection keeps a writable tail and immutable completed
chunks, so appending does not concatenate the graph's existing arrays. Each
collection has a logical maximum of `DefinitionBuilder.max_size` (`maxInt(u32)`);
that maximum does not reserve backing storage. Source-name tables use the same
limit and are supplied once when sources are initialized. `finish()` flattens the
populated entries into exact-sized definition arrays and records the maximum
tensor rank. Builder copies can be extended independently, and later appends
leave previously finished definitions unchanged.

The public definition representation lives in `frontend/`. Completed
definitions cross into `compiler/`, where construction produces the semantic
graph. Shared graph, tensor, generated-model, and inspection representations
live in `core/`.

Optional extensions under `zgc.ext`, such as `zgc.ext.nn` and `zgc.ext.img`,
build on this same front end. Neural-network layers expand into core sources
and operations, while image helpers declare core inputs with explicit rank-4
layout conventions. They do not own a separate runtime or storage
representation.

The definition records:

- source tensors and their source kinds;
- compute and view operations;
- flattened input references;
- inferred dtype and shape metadata;
- graph outputs.

Operation inputs are validated while operations are added. `finish()`
derives the exact node, tensor, input-reference, output, and maximum-rank
capacities and returns the specialized immutable definition value.

## Construction and analysis

The counting step reads the completed definition and derives exact graph
capacities. In particular, the graph's rank capacity is the largest rank
actually used, rather than the definition's rank bound. Source storage uses
direct enum indexing, so its capacity is the highest referenced source index
plus one.

After semantic optimization, finalization rebuilds the graph using its actual
node, tensor, input-reference, output, source-slot, and rank counts. Analysis and
search use those canonical capacities rather than the original definition's
sizes. Source IDs remain stable, so holes below the highest live source index
remain present. Forward provenance maps retain raw graph dimensions; reverse
maps contain exactly the optimized node and tensor counts.

Search candidates share the canonical semantic capacity while they are compared.
After selection, the executable is finalized again, including its exact output
reference count. `Executable.internal_capacity` describes that selected program;
`raw_capacity` and `semantic_capacity` describe the earlier stages. Compiler
internals inspecting `semantic_graph` should use `semantic_capacity`.

Graph construction records a raw semantic graph in definition order. Compute
results initially use canonical dense storage. View operations preserve their
source storage tensor and derive aliasing shape and strides from that semantic
layout. Semantic validation checks operation shapes, dtypes, and view aliases
before analysis relies on them.

Semantic optimization canonicalizes identity and composed views, flattens
compatible nested concatenations, folds scalar expressions and conservative
algebraic identities, and performs backward liveness from declared outputs.
It rebuilds a compact semantic graph with renumbered nodes and tensors while
retaining bidirectional provenance to the raw graph. Source enum indices remain
stable because they are part of the public binding contract.

Semantic analysis records tensor use counts, graph outputs, tensor-to-consumer
edges, and a deterministic topological order and rank for every node. Analysis
is organized under `compiler/analysis/`: `semantic.zig` owns dependency facts,
while `fusion.zig`, `layout.zig`, and `remap.zig` discover their respective
alternatives. Fusion and remap discovery traverse dependency facts rather than
treating node identifiers as dependency adjacency. Fusion analysis emits an
unfused regime plus discovered pointwise map, producer-to-reduction, and
compatible sibling-reduction regions when available. Layout analysis emits
canonical and propagated layout regions without choosing between them. Remap
analysis emits a direct regime plus optional composed regions for compatible
shift and concatenation operations.

Executable search owns the decision boundary. It incrementally composes
non-conflicting fusion, layout, and remap regions into partial whole-program
representations. Node-claim and tensor-layout checks reject incompatible local
choices before lowering. An empty planned baseline and a maximally compatible
anchor are retained, while bounded local pruning favors representations that
cover more nodes and tensors with fewer region boundaries. Each representation
is lowered through the corresponding data-only kernel planners and expanded
into legal schedule variants. Map, reduction, and contraction planning
implementations live under `compiler/planning/`; search owns their composition
and selection. There is no canonical fused/layout/kernel executable before
this search. A rank-2 matmul retains the logical
contract `[M, K] * [K, N]` while eligible parameter and constant right-hand
sides use physical strides `[1, K]`. Batch-oriented layouts propagate through
compatible operations in the propagated regime.

Executable search also preserves an unfused, generic lowering as the legal
reference. It generates semantic-order, memory-pressure, and critical-path
schedules for the reference and for every physical combination. Each lowered
representation is validated once and receives one output binding plan.
Canonical-layout representations reuse semantic validation. Distinct scheduled
programs receive lifetime analysis and memory planning; identical programs
reuse completed plans. Source bindings are validated once from the semantic
graph. Geometry- and layout-dependent cost facts and persistent storage are
computed once per lowered representation and reused across its schedules.

A `PlanCandidate` associates an `Executable` and `Schedule` with its origin and
a structured cost containing estimated runtime work, ordinary read/write
traffic, peak and persistent memory, scratch, code size, and conversion cost.
Representation composition retains at most 24 locally useful partial
representations before lowering. The executable search then retains at most 16
completed candidates on a Pareto frontier and applies deterministic
tie-breaking. When the frontier is full, a stronger incomparable candidate may
replace the weakest retained candidate. Equal-cost physical choices favor
analyzed fusion, propagated layouts, and composed remaps. The selected
candidate becomes the active executable. Capacity compaction preserves its
validated operations and reuses its lifetime, output binding, and storage
plans; it does not validate or allocate the selected program again.

Semantic compute nodes use the operation representation in `operations/`.
Executable compute nodes retain unchanged operations as `direct` semantic
operations. Specialized computation uses a `KernelPlan` classified as map,
reduction, or contraction. Plans contain compile-time data only;
`ExecutableCompute` dispatches them to their kernel family. Matmul lowers to a
contraction plan. Map regions combine single-consumer pointwise
expressions into one traversal and also represent pure transfers over a shared
domain/load/body/store structure. A map plan selects an expression traversal or
a segmented transfer strategy. Reduction regions combine compatible pointwise
producers and sibling accumulators over a shared domain. Segmented map plans
describe rectangular source-to-destination transfers with static offsets,
extents, and strides. Lowering composes compatible concatenation trees and
shifts so intermediate results can remain unmaterialized and leaf views can
write directly into the final destination. A compatible pointwise producer on
a canonical contiguous domain may be substituted into the segment loader, so
its expression is evaluated at the remapped source coordinate and its original
store and alias-only views are omitted. Boundary partitions are resolved during
lowering rather than dispatched per element at runtime. Composed segments use
SIMD on compatible contiguous inner runs. Short tails and incompatible static
strides use the scalar evaluator selected at compile time.

The semantic graph stores fixed arrays of nodes, tensor metadata, flattened
input references, outputs, and sources. An `Executable` stores a fixed
sequence of invocations with flattened input and output references. An
invocation may name multiple outputs, allowing sibling reductions to share one
traversal without representing secondary stores as no-op nodes.

The generated model retains `raw_graph`, the compact `semantic_graph`, semantic
rewrite provenance, the active `executable`, the generic
`reference_executable_candidate`, the
`executable_candidate_frontier`, and `selected_executable_candidate` as
compile-time inspection metadata. Fusion, layout, remap, and generated
executable candidate counts are also retained. The selected program is
validated before compaction and provides mutable and read-only tensor view types
whose shape, strides, base offset, element count, and layout traits are
compile-time properties.

## Memory planning and model generation

`MemoryPlan` assigns one aligned byte region to each storage-owning tensor.
Aliasing views point at their root storage tensor's region. The generated model
contains one inline byte array sized and aligned by that plan. Model-owned
sources and compute results receive regions in this array; embedded parameters,
embedded constants, and runtime-bound inputs remain external to it.

Lifetime analysis records a half-open node interval for each storage root and
propagates alias uses to that root. Model-owned sources and output roots remain
persistent. The planner releases expired intermediate regions, coalesces
adjacent free spans, and places new tensors into the smallest aligned span that
fits. Oversized spans are split around the allocation. If no span fits, the
planner extends the model's storage high-water mark. Execution does not
allocate.

The model API provides:

- `init()` to zero-initialize model memory;
- `copyInput(key, values)` to pack a logical row-major runtime input into owned storage;
- `copySource(key, values)` to pack logical row-major values into any model-owned source;
- `bindInput(key, values)` to borrow input already stored in the compiled physical layout;
- `sourceLayout(key)` to query that source layout;
- `run()` to execute the selected statically scheduled program;
- `outputView(index)` to retrieve a typed read-only view.

The internal inspector consumes the model's compile-time graph and memory-plan metadata
without adding rendering responsibilities to the model, graph, operation,
tensor, or storage types. It also renders bounded mutable memory from a model
instance when requested. Consumers access inspection through `zgc_inspect_cli`
or inspectable executable artifacts.

`zgc_executable_abi` specializes a versioned C ABI around a consumer-provided model
module. It exposes opaque placement initialization, source and output
descriptors, copied or borrowed source loading, execution, and borrowed or
copied output access. `zgc_executable_runner` packages the same interface in a
minimal executable for generated-code inspection. Artifact operations retain
the model's compile-time geometry and selected executable schedule.

View nodes do not execute kernels. Their result layouts are resolved during
graph construction, and downstream compute kernels receive static-geometry
views into the aliased storage.

`definition.executableWith(...)` accepts a typed slice of source-enum tags and
storage bindings to select non-default storage.
`zgc.memory.Source.embed(bytes)` accepts logical row-major parameter or constant bytes
and compile-time packs them into the lowered source layout.
`zgc.memory.Source.embedPacked(bytes)` accepts bytes already in that physical layout.
Both place the resulting storage in read-only program data.
`zgc.memory.Source.bound` makes an input borrow storage supplied to each model
instance. Dtype is enforced by the typed copy/bind APIs, and element or byte
counts are checked before a source is accepted.

## Kernel dispatch

Each compute node resolves prevalidated static input and output view types and
dispatches through its executable operation. Optimization selects physical
layouts, while kernels traverse contiguous axes in target-native SIMD chunks
with scalar tails. Runtime view state contains storage and any cursor offset
introduced by runtime-selected subviews; fixed tensor geometry is carried by
the type.

The generated model is a graph-specific function, not a runtime graph
interpreter. Node selection and `ExecutableCompute` plan dispatch are explicit
inline boundaries so compile-time tags disappear before code generation.
Load, expression, address-resolution, and store helpers used inside an element
loop are also inline where crossing the boundary would hide constants or block
loop optimization. `Executable.Runtime.run`, tensor-view construction, and substantial
kernel bodies remain ordinary function boundaries; unselected implementations
are never instantiated, and selected kernels may remain standalone functions
when the boundary preserves all specialization information.

Executable lowering records a concrete matmul traversal strategy in a data-only
contraction plan. Generated models dispatch directly to that strategy and do not
branch over layout metadata at runtime. Direct semantic matmul execution uses
the general scalar kernel.

Map execution uses a compile-time instruction program built from
pipeline-independent elementwise descriptors. Instruction arity and accepted
dtypes belong to semantic operations; instruction references and traversal are
executable-graph details. The instruction sequence is unrolled at compile time,
so a fused kernel performs one output traversal without runtime opcode dispatch
or storage for instruction results.

Logical optimization regions and physical kernel plans are separate. Map
regions contain a domain, loads, an expression or transfer body, and stores.
Reduction regions add reduction axes and accumulators. Contraction regions add
an optional epilogue. Every region uses the same access-aware load and store
descriptors; store values explicitly identify expression, accumulator,
contraction, or transfer results. Plans select traversal, vectorization,
segmented-transfer, or contraction strategies after layout selection. Neither
representation owns execution behavior.

Shape, dtype, rank, axis, and plan compatibility checks belong to semantic and
final validation. Execution kernels assume those contracts.
Dynamic `Tensor.View` and `Tensor.ConstView` types remain available when a
low-level caller intentionally supplies runtime geometry.

Kernels are grouped by family:

| Family            | Implemented operations                                                                                                             |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Literals          | Rank-zero scalar values and zero-stride filled-tensor expansion                                                                    |
| Elementwise       | Numeric operations, comparisons, strict boolean logic, and conditional selection                                                   |
| Fused elementwise | Compile-time typed instruction programs with contiguous SIMD and static-stride fallback traversal                                  |
| Materialization   | Logical copy, row-major conversion, and constant padding                                                                           |
| Shifting          | Shape-preserving translation with wrap, edge, reflect, or constant boundaries                                                      |
| Contraction       | Rank-2 matmul                                                                                                                      |
| Reduction         | Sum, mean, min, and max over compile-time axis sets                                                                                |
| Special           | Softmax over one axis                                                                                                              |
| Concatenation     | Materialized output with contiguous block-copy and static strided paths                                                            |
| Remapping         | Map regions with statically segmented shift/concatenation transfers and direct final-destination stores                            |
| Layout            | Compile-time transpose, permutation, reshape, flatten, squeeze, unsqueeze, slicing, broadcasting, and overlapping-window inference |

Binary arithmetic aligns shapes from the trailing axis. Equal extents are
paired directly, singleton extents broadcast with zero strides, and absent
leading axes behave as singleton dimensions. Reduction axes are normalized,
deduplicated, and encoded at definition time. `keep_dims` retains reduced axes
as singleton dimensions so reduction outputs can broadcast back over inputs.

Comparisons return boolean tensors. Logical operations accept only boolean
tensors, and `where` requires a boolean condition. Numeric tensors are never
interpreted through implicit truthiness rules.

Scalar literals are immutable, source-free rank-zero tensors embedded in the
generated program. They do not reserve model memory or execute a kernel. A
filled tensor is a scalar literal followed by a zero-stride broadcast view, so
its storage remains one element regardless of logical shape.

Structural operations create aliases and do not execute kernels. Squeeze and
unsqueeze preserve arbitrary source strides. Flatten requires its selected
axis range to be logically contiguous. General reshape requires a
logically row-major contiguous source because it must preserve element order
without copying. Permutation lowers to transpose aliases. Slicing uses
compile-time positive bounds and steps to produce an offset strided alias.
Windows append static neighborhood axes and may overlap within the same
storage root.

Elementwise kernels use SIMD for row-major tensors and matching dense axis
permutations. Trailing-vector binary arithmetic also vectorizes across a contiguous first
axis, covering bias operations on batch-oriented matmul results. Selected
reduction and contraction paths use SIMD. Generic view traversal handles
offsets and positive or negative strides where the relevant kernel supports
them.


### Shared compiler facts and region catalogs

Semantic analysis establishes dependency order and consumer edges, use/output
facts, tensor byte sizes and storage roots, operation classifications, and
fusion anchors once for the immutable semantic graph. Selectors reuse these
facts. They do not substitute for physical use counts, layout-dependent
properties, or schedule-dependent lifetimes after lowering.

Fusion, layout, and remap selections reference immutable region definitions.
Candidate copies carry region references and ownership information instead of
embedding every possible group. Region membership uses packed bitsets with
ascending set-bit iteration. Reduction groups keep exact-sized reduction lists;
discovery links reductions by ID before freezing the lists. Region traffic
contributions are computed once during analysis. Composition checks conflicts
before cloning candidate state, and rejected branches still consume their search
allowance. Contraction admission overlays the three relevant tensor descriptors
instead of constructing a layout-modified graph.

Reference construction, physical schedule application, and graph validation use
ordinary comptime loops instead of explicitly expanding a separate loop body
for every node. These changes preserve static compilation and final candidate
costing.


### Physical program construction and frontier ownership

Reference construction, physical lowering, and non-identity schedule application
append invocations and tensor references through a chunked program builder. A
small producer table records changed origins; immutable tensor metadata stays
shared until final assembly. Completed programs retain the existing concrete
array representation. Identity schedules reuse the existing program directly.

The executable frontier and its eviction pool store immutable candidate
references. Comparisons borrow candidates/programs instead of passing full
program values. Frontier snapshots therefore copy references and keep their
previous candidate records intact after replacement. `selected()` borrows the
chosen candidate; frontier entries are read-only references.

Reference and physical scheduling compute only the policy prefix that fits the
remaining executable allowance. A one-slot prefix uses the known dependency
order without building memory-pressure or critical-path schedules. This avoids
performing full alternate traversals before a capped search can reject them.
The complete three-policy API and ordering remain unchanged.
