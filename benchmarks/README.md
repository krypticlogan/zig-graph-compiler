# Benchmark workloads

`workloads/` retains the complete Zig graph definitions and equivalent direct
implementations for contraction, reduction, fusion, dense, LBM, and Conway.
These are reference workloads for compiler and kernel development. The root
build also provides `zig build benchmark -Dbenchmark=<workload>` for direct
comparisons with their Zig implementations.

Use `zgc compile --time` to measure compilation, then `zgc time` to measure
the resulting shared library:

```sh
zgc compile graph.zgir -O ReleaseFast --time -o graph.so
zgc time graph.so --out benchmarks/results.json
zgc time graph.so --runs 30 --warmup-ms 2000 --sample-ms 250
```

The JSON file is a top-level array; each invocation appends one record with
configuration, raw samples, and summary statistics. Compilation time is reported
separately by `zgc compile --time` and may reflect a warm or cold Zig cache.

Non-embedded sources are initialized with zero-filled buffers. To measure a
workload with representative input data or compare it with a direct Zig
implementation, use a dedicated host that initializes and runs both paths.
