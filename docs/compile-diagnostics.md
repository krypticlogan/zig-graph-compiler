# Comptime compilation diagnostics

Use diagnostic builds to determine which compiler stage is expensive. Normal
builds leave diagnostics disabled. No graph arrays or tensor contents are logged.

```zig
const Executable = definition.executableWith(.{
    .diagnostics = .{ .stop_after = .canonical_tensor_order },
});
```

`stop_after` emits pipeline summaries and deliberately raises a compile error
immediately after the requested stage finishes. Use an earlier checkpoint when
the requested stage cannot finish. This prevents later stages from hiding the
results or consuming memory. `executableFromZgirWithOptions` accepts the same
`diagnostics` field, as does the serialized-graph adapter's `executable_options`.

`.log = true` logs pipeline beginnings and completions without stopping early.
`.detailed = true` adds canonicalization substeps, representation lowering and
scheduling, representation validation and output bindings, and each distinct
schedule's lifetimes, storage planning, and estimated costs. Select a detailed stage with
`stop_after` to enable its own log without enabling all detailed logs.

Zig's [`@compileLog`](https://ziglang.org/documentation/master/#compileLog)
prevents artifact emission. These are intentionally failing diagnostic builds,
not ordinary builds with a live progress stream. A killed/OOM compiler may never
print accumulated compile logs. The checkpoints provide a way to obtain output
before that happens. There are no comptime clock readings.

## Probe the free-surface graph

The host-side helper imports a Zig module's public `definition` and requests a
checkpoint without compiling the viewer or depending on Raylib:

```sh
python tools/profile_compile.py examples/fluid-dynamics/src/free_surface_lbm.zig \
  --stage raw_validation --stage canonical_concat_analysis \
  --stage canonical_tensor_order --timeout 60 --max-memory-mib 1536
```

Use `--definition <name>` for a different public definition declaration,
`--detailed` for detailed logs, and `--output-dir <directory>` to preserve separate
runs. The source module must import only `zgc` and its own relative files.

Each probe runs a separate `zig build-obj -fno-emit-bin` invocation. The helper
records elapsed wall time, peak compiler RSS on Windows/Linux, command, status,
and raw diagnostics in `.zig-cache/compile-diagnostics/`. `report.json` is updated
after each probe. A successful checkpoint has status `reached` despite Zig's
intentional nonzero exit. Any other status stops the sequence. The host kills its
own compiler process on timeout or exceeding the memory threshold. RSS sampling
can overshoot that threshold; the memory limit is unavailable on other platforms.

Durations and memory include the builder and all stages preceding a checkpoint;
they are not isolated stage durations or precise allocation accounting. Cache
state and diagnostic instrumentation also affect the measurements. The
`executable_generation` assembles the selected executable from retained
candidate plans and establishes its runtime layout; it does not compile every
lazy runtime method or measure LLVM codegen.

## Checkpoints

Pipeline stages are `definition`, `counting`, `construction`, `raw_validation`,
`semantic_optimization`, `semantic_finalization`, `semantic_validation`, `semantic_analysis`,
`fusion_analysis`, `layout_analysis`, `remap_analysis`, `contraction_analysis`,
`source_planning`, `executable_search`, `selection`, `selected_compaction`,
`selected_lifetimes`, `selected_output_plan`, and `executable_generation`.

Optimizer substeps are `semantic_canonicalization`, `canonical_use_counts`,
`canonical_outputs`, `canonical_concat_analysis`, `canonical_tensor_order`,
`canonical_rewrite`, `semantic_dead_elimination`, and `semantic_provenance`.
Search/candidate checkpoints are listed in
[`diagnostics.zig`](../src/zgc/compiler/diagnostics.zig). A repeated substep
checkpoint stops at its first occurrence.
`reference_cost` and `representation_cost` measure order-invariant cost facts;
`candidate_cost` finalizes schedule-dependent memory costs.

## Initial free-surface investigation

The investigated graph finishes with 1,008 nodes, 1,319 tensors, 1,957 input references,
seven outputs, and rank three. The constructed semantic graph occupies 265,552
bytes as a Zig value. On the development Windows machine, diagnostic probes found:

| Checkpoint | Status | Cumulative time | Peak compiler working memory |
| --- | --- | ---: | ---: |
| `definition` | Reached | 0.77 s | 83 MiB |
| `raw_validation` | Reached | 2.52 s | 167 MiB |
| `canonical_concat_analysis` | Reached | 2.13 s | 172 MiB |
| `canonical_tensor_order` | Memory threshold exceeded | 7.03 s | 1,544 MiB |

These are separate invocations, hence the small variation between earlier
checkpoints. The first isolated bottleneck was `topologicalTensorOrder()` in
`semantic_optimization.zig`, before canonical rewriting, analysis, or search.
Its original implementation restarted its scan at tensor zero for every emitted
tensor, performing quadratic work even on an ordered graph. Subsequent changes
replaced it with iterative DFS and semantic dependency ordering with a ready
queue. These original measurements precede those changes.

With those ordering changes and canonical graph finalization, subsequent probes
reached semantic analysis at 808 MiB peak compiler working memory and fusion
analysis at 822 MiB. Their graph/analysis capacities are now 967 nodes, 1,248
tensors, and 1,882 input references, rather than the raw definition's counts.
These probes did not establish whether later stages can finish.

## Traversal improvements and remaining bottleneck

The compiler now retains an already valid dependency order and shares a consumer
edge index across semantic analysis and schedule variants. Physical programs
build one index for their three schedules. Critical work is computed once in
reverse dependency order. Priority schedules use an indexed heap; readiness and
memory-pressure score changes follow affected edges instead of rescanning every
node. Existing priority and node-ID tie rules are retained.

Storage planning indexes allocation starts and lifetime ends by invocation. It
processes these events in tensor-ID order, preserving allocation behavior without
scanning all tensors and active allocations at every invocation. Free-span
insertion uses binary search and neighboring coalescing; best-fit allocation
still scans available free spans. Layout grouping indexes storage roots directly.
Canonical tensor ordering also returns immediately when dependency order is
already valid.

On the same free-surface graph, cumulative probes after these changes reached
layout analysis in 8.75 s at 801 MiB and contraction analysis in 10.00 s at
838 MiB. These are individual diagnostic runs, not isolated stage timings.

Representation composition is the next measured bottleneck. Before search
representation changes, its checkpoint exceeded the 2,048 MiB host limit after
28.58 s. Branch snapshots now copy pointers to immutable retained representations
instead of copying the entire representation set. Local costs accumulate region
and contraction contributions instead of rescanning and applying layouts to the
whole graph on every insertion. A subsequent composition probe remained below
the memory limit (1,744 MiB peak) but timed out at 90 s without reaching the
checkpoint. This does **not** establish that the complete graph can compile.

Remaining expensive paths include dense region membership scans and candidate
cloning during representation composition, per-reduction group comparisons in
fusion analysis, and best-fit free-span allocation. They require further
measurement before attributing the remaining time to one of these loops.

Regression tests compare the new schedules against an independent scan-based
implementation on branching DAGs, repeated inputs, and multi-output invocations.
Storage tests cover lifetime reuse, release coalescing, and exact offsets.
Incremental representation costs are compared against full graph estimates.

## Search allowances

Every definition and ZGIR executable path accepts a `.search` option:

```zig
const Executable = definition.executableWith(.{ .search = .{
    .max_executables = 128,
    .max_composition_attempts = 1024,
} });
```

These are the defaults. The executable allowance counts completed candidates,
including reference schedules and every optimized schedule variant. It must be
at least one. The composition allowance counts local branch attempts before
cloning or checking their compatibility; duplicates and rejected branches count.
Zero disables additional branching. The existing retention limits (24
representations and 16 frontier executables) are separate from these work limits.

One baseline representation and a greedy combination of compatible analyzed
optimizations are seeded before additional branching. When at least two
executables are allowed, reference generation reserves a slot for this optimized
anchor and planned generation visits it first. Remaining partial alternatives
are visited in local cost order. With one slot the direct reference
is used, except where masked reduction semantics require fusion; those graphs
use a valid fused candidate. An optimized candidate can lose final selection if
the measured cost prefers the reference. This is not a guarantee of globally
optimal selection or a fixed compilation duration.

Search returns its retained frontier when the allowance is consumed. Normal
selection still honors the optimization target or a forced frontier index.
Construction, analyses, greedy seeding, lowering and planning an individual
candidate, and final executable compilation are not bounded in time or memory.

Executable metadata exposes `executable_candidate_count`,
`search_executable_limit_reached`, `search_composition_attempts`, and
`search_composition_limit_reached`. Both `limit_reached` flags report allowance
consumption, not proof that further distinct candidates existed.

For ZGIR CLI compilation use `--max-executables N` and
`--max-composition-attempts N`. For a Zig module, set `.search` where the module
constructs its executable. Both CLI limits participate in artifact cache keys.
The profiling helper accepts the same flags, for example:

```sh
python tools/profile_compile.py examples/fluid-dynamics/src/free_surface_lbm.zig \
  --stage search_composition --max-executables 2 \
  --max-composition-attempts 0 --timeout 90 --max-memory-mib 2048
```

That probe reached composition in 14.52 s at 1,566 MiB. A separate probe of the
next `search_reference` checkpoint exceeded 2 GiB after 19.62 s. Limiting
composition therefore does not resolve the remaining per-candidate compilation
memory cost on this graph. A 1,024-attempt composition probe also exceeded 2 GiB
before reaching its checkpoint.

With 16 composition attempts, the large-graph composition checkpoint reached in
23.19 s at 1,624 MiB (with another build running concurrently). Its diagnostic
reported exactly 16 attempts, verifying that a nonzero allowance also exits the
branching loops. These limits are configurable; larger defaults retain a broader
search for smaller graphs and are not a memory guarantee for this workload.

## Shared facts and compact region implementation

The shared semantic facts now include operation classifications, tensor byte
sizes, storage roots, and fusion anchors, in addition to dependency adjacency,
ordering, and use/output information. Fusion discovery reuses those facts.
Physical layouts, uses, and lifetimes retain their own validity boundaries.

Region selections now hold references to immutable group definitions. Membership
and conflict tests use packed sets with ordered iteration over selected IDs.
Reduction discovery uses linked membership and freezes exact-sized reduction
lists rather than storing a graph-sized reduction list for every possible group.
Per-region traffic savings are computed once. Branch admission checks conflicts
before cloning. Contraction admission reads three overlaid tensor descriptors
instead of creating a modified semantic graph to check strategy legality.

Explicit node-loop unrolling was removed from reference construction, physical
schedule application, and semantic/physical validation where ordinary comptime
loops suffice. `reference_construction` separates program construction from
candidate validation and later planning. Composition logs include state and
retained-set byte sizes (excluding separately referenced immutable catalog data).
The profiling helper also writes `<stage>.memory.json` with approximately
one-second samples of current and peak working memory. Its last-current value
may be sampled during compiler cleanup and is not the stage's live working set.

A 16-attempt free-surface composition probe reached in 16.24 s at 822 MiB, compared
with the earlier 1,624 MiB probe. Its candidate state was 154,080 bytes and its
pointer table plus counters 224 bytes. A subsequent series reached reference
construction, candidate validation, lifetimes, storage, and costing. The costing
checkpoint peaked at 1,864 MiB. These are cumulative, separately compiled probes;
other builds ran during parts of the series, so timings are indicative only.

After validation loop cleanup, the one-executable, zero-branch probe reached
`executable_generation` in 21.70 s at 1,946 MiB. This establishes completion of
the comptime generation pipeline with that budget, not full native codegen or
completion of the default optimized search. The remaining large memory increase
is concentrated between composition and reference program construction; the
resulting reference program itself occupies 398,032 bytes.

Next work remains: intern layout-choice descriptors, reduce full-program copies
in candidate/frontier planning, avoid graph-specialized intermediate types where
possible, and normalize coordinate/view compositions through a shared interface.
The current implementation retains exact final candidate costing and existing
runtime execution semantics.

The final default-allowance composition probe (1,024 attempted branches,
24 retained representations) reached in 23.07 s at 950 MiB. The earlier probe
with that same allowance exceeded 2 GiB before reaching composition. Thus the
structural changes allow the broader composition search to complete under the
probe limit; they do not by themselves establish completion of all default
executable candidates.

The corresponding two-executable, 16-attempt full-generation probe still exceeded
2 GiB (2,056 MiB sampled peak at 25.78 s). The one-candidate result should not be
read as resolution of optimized search's remaining memory cost. Further work
should target reference/full-program construction and intermediate candidate
representations before increasing search allowances.

## Program construction and candidate storage improvements

The compiler now uses chunked invocation/reference storage during reference
construction, lowering, and schedule application. Producer overrides are kept
separately, unchanged tensor arrays are reused in bulk, and complete programs
are assembled once. An identity schedule returns the existing program. The
frontend and compiler builders share the same append-only chunk implementation.

Executable frontier entries and the temporary eviction pool now hold immutable
candidate references, with pointer-based comparisons. The frontier structure is
136 bytes on the development machine, excluding the referenced programs and
candidate records. Snapshot/eviction/duplicate tests verify that replacing an
entry does not mutate earlier snapshots or lose the protected target winners.

A separate issue was exposed while measuring construction: scheduling eagerly
produced all three policies even if only the first could fit the executable
budget. Both semantic-reference and physical scheduling now compute only the
required prefix. One schedule requires no priority heap or critical-work
traversal; two omit critical-path calculation. Tests compare every limited
prefix against full scheduling on branching graphs with repeated inputs.
`reference_scheduling` is a new checkpoint ahead of `reference_construction`.

With two executables and 16 composition attempts, free-surface probes reached:

| Checkpoint | Cumulative time | Peak working memory |
| --- | ---: | ---: |
| `reference_scheduling` | 20.24 s | 816 MiB |
| `reference_construction` | 9.70 s | 830 MiB |
| `executable_generation` | 23.07 s | 1,140 MiB |

These are separate diagnostic invocations, with other builds running during
parts of the series. The differing cumulative times are not stage durations.
The same two-candidate full-generation probe previously exceeded 2 GiB. The
completed generation uses a reference candidate and the compatible greedy
optimized candidate; it does not measure LLVM/native codegen.

The full default allowance still exceeded 2 GiB (47.40 s, 2,049 MiB sampled
peak). These changes establish completion of the small optimized search, not a
fixed-memory guarantee for larger searches. Broader schedule generation and
repeated distinct candidate planning still need profiling.

New regression coverage includes program construction across two 64-entry chunk
boundaries, multi-output invocations, sparse source slots, unused tensors,
identity/reordered schedules, a 130-node runtime chain, and frontier eviction
and immutable snapshots. Exact candidate costing and selection priorities are
preserved.
