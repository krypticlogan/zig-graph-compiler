# Benchmarks

Benchmarks compare complete compiled ZGC graphs with equivalent direct Zig
implementations. Each workload is compiled as a separate executable so its
graph, selected executable, code size, and compiler decisions remain isolated.

## Workloads

| Name | Graph |
| --- | --- |
| `contraction` | One 64 × 64 matrix contraction |
| `reduction` | Sum, mean, minimum, and maximum over the same 256 × 256 domain |
| `fusion` | Elementwise multiply and add feeding a trailing-axis reduction |
| `dense` | A 128 → 64 → 10 dense network at batch 32 |
| `lbm` | The canonical 180 × 320 D2Q9 fluid collision and streaming step |
| `conway` | The complete 120 × 88 Conway update graph used by the example |

The kernel-oriented cases are still complete specialized graphs. They measure
the implementation selected by analysis, executable search, layout planning,
and lowering rather than calling a semantic operation or runtime-geometry
kernel directly.

The dense, LBM, and Conway workloads are complete application graphs. The LBM
and Conway definitions are kept in their workload modules so benchmark builds
are self-contained.

## Run

From the repository root:

```sh
zig build benchmark -Dbenchmark=dense -Doptimize=ReleaseFast
```

Run every workload as an independently compiled process:

```sh
./benchmarks/run-suite.sh
```

The default configuration performs a two-second warmup followed by 30 samples
calibrated to at least 250 ms each. Duration and explicit-iteration overrides
remain available:

```sh
zig build benchmark \
  -Dbenchmark=fusion \
  -Doptimize=ReleaseFast \
  -Dwarmup_ms=500 \
  -Dsample_ms=100 \
  -Druns=10
```

Choose a compiler target or measure one retained candidate directly:

```sh
zig build benchmark -Dbenchmark=dense -Doptimize=ReleaseFast -Doptimization_target=memory
zig build benchmark -Dbenchmark=dense -Doptimize=ReleaseFast -Dshow_frontier=true
zig build benchmark -Dbenchmark=dense -Doptimize=ReleaseFast -Dfrontier_index=3
```

The reported frontier index is zero-based. A forced index overrides the target;
an index outside that workload's retained frontier fails compilation.
Retain results as a JSON array and append later runs with `-Dresults_file`:

```sh
zig build benchmark -Dbenchmark=dense -Doptimize=ReleaseFast \
  -Dshow_frontier=true -Dresults_file=benchmarks/results.json
```

Each record links the selected target, candidate index and strategy, cost and
memory estimates, per-run timing statistics and raw samples, direct comparison,
observed nanoseconds per estimated work unit, implied traffic rate, environment,
and benchmark settings. Reusing the path appends another record. Keep the file
with the machine and target details described above when using it for offline
tuning.

## Cost and memory report

The ZGC result pairs the selected executable's compile-time cost estimate with
its measured average latency. The total work units and arithmetic, access,
register-pressure, and transfer components come from the compiler's cost model;
they are relative scores, not nanoseconds. The `calibration` line divides the
observed latency by that total so runs on the same machine can be compared and
used to tune the model offline. A single run does not identify the time spent
in each component. Register-pressure work is a predicted spill penalty: it is
zero while estimated demand fits the target's register budget, even though the
kernel still uses registers.
The selection line identifies the optimization target and selected frontier
index. Its count includes distinct retained alternatives, including candidates
kept for explicit selection even when another candidate wins on all estimated
costs.

The traffic estimate counts reads and writes predicted for one graph invocation.
The reported GiB/s divides those estimated bytes by measured latency; it is an
**implied rate**, not measured DRAM bandwidth. The existing throughput line uses
the workload's logical bytes per invocation, which is a different quantity.
Timing alone cannot reveal cache hits, cache misses, or physical memory traffic;
those require hardware counters or a profiler.

The storage line reports the compiler's model-owned buffer plan (peak,
persistent, and scratch bytes) and `@sizeOf(Executable)` for the generated model
state. This is static storage accounting, not process RSS.
Scratch is the peak buffer capacity beyond persistent storage. It covers the
space needed for reusable intermediates and any alignment padding, so peak
equals persistent plus scratch. It is not a separate runtime allocation.
Borrowed source buffers and embedded static data are outside the model-owned
buffer, and a workload may hold additional benchmark inputs and reference
outputs. Process RSS would include those, the direct implementation, and the
benchmark runner, so it cannot be attributed to one graph invocation here.
