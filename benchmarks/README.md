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

Each workload initializes and validates both implementations before timing.
The runner uses the same warmup and sampling configuration for each, reports
their timings independently, and prints the direct/ZGC latency ratio. Output
barriers remain inside each invocation loop so repeated stateless calls cannot
be removed. Report the Zig version, target, optimization mode, CPU/OS,
workload, and timing overrides with retained results.
