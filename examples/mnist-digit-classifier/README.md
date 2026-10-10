# MNIST Digit Classifier

This standalone Zig package consumes ZGC through its public package interface.
It contains a 784→128→64→10 digit-classification network for interactive
inference, inspection, and generated-artifact analysis.

Run commands from this directory unless noted otherwise.

## Interactive demo

```sh
zig build demo -Doptimize=ReleaseFast
```

The `demo` executable opens a 28×28 drawing canvas backed directly by the
model's runtime-bound input. Hold the left mouse button to draw a digit; the
dashboard updates all ten class probabilities as the canvas changes. Press
Enter to clear the canvas. The model parameters remain embedded read-only data.

The six parameter binaries live in `model_params/`. The `model_params` module
exposes them through `@embedFile`, and `src/digit-classifier.zig` assigns them with
`zgc.memory.Source.embed`. They are read-only data in the executable, not members of
the model's inline mutable memory. The 784-element input uses
`zgc.memory.Source.bound` and borrows caller storage at runtime.

The weight files are serialized as `[output, input]`. Three zero-copy transpose
views adapt them to the graph matmul convention of `[input, output]`; the views
continue to alias the embedded bytes.

Consequently, the six parameters (437,544 bytes total) and the input receive no
memory-plan regions. Lifetime-based reuse reserves 1,024 bytes for the
network's nine intermediate/output tensors.

## Model inspection

```sh
zig build inspect
```

The model-specific `zgc-inspect` executable uses the library inspection CLI and
the `Model` exported from `src/digit-classifier.zig`. With no argument it prints
the exact capacity, tensor and operation listing, output-oriented graph tree,
and memory plan. Individual representations can be selected after `--`:

```sh
zig build inspect -- summary
zig build inspect -- graph
zig build inspect -- tree
zig build inspect -- memory-plan
```

See [model inspection](../../docs/inspection.md) for the programmatic API and for
wiring the CLI to another generated model.

## Generated model artifact

Build the binary used for generated-code inspection:

```sh
zig build build-model -Doptimize=ReleaseFast
```

The resulting artifact is `zig-out/bin/zgc-model`. It uses ZGC's
`zgc_model_runner` module and the `Model` exported from
`src/digit-classifier.zig`. Its exported `zgc_model_run` symbol contains model
execution without logging, timing, input generation, or output formatting. The
executable entry point performs no inference because runtime input binding
belongs to the host application.

Build the same model as a shared library exposing the complete C ABI:

```sh
zig build build-model-library -Doptimize=ReleaseFast
```

The host can discover source and output metadata, initialize opaque model
storage, copy or bind sources, execute the model, and borrow or copy outputs.

Disassemble only the stable execution symbol:

```sh
zig build disassemble-model -Doptimize=ReleaseFast
```

On macOS, inspect it interactively with LLDB:

```text
lldb zig-out/bin/zgc-model
(lldb) image lookup --name zgc_model_run
(lldb) disassemble --name zgc_model_run
```

See [generated model artifacts](../../docs/model-artifacts.md) for the host
lifecycle, C header, source policies, and build integration.

## Model definitions

`src/digit-classifier.zig` uses the public workflow expected of a consumer:

1. Instantiate the concrete `DefinitionBuilder` and request a typed source
   facade with `builder.sources(Sources)`.
2. Define the graph once.
3. Finish the definition and call `definition.modelWith(...)` to select bound
   and embedded source storage.
4. Initialize the generated model, bind a runtime input, run, and retrieve an
   output view.

The digit classifier composes its three dense stages with `zgc.ext.nn.Dense`
and `zgc.ext.nn.Sequential`. Its output-major parameter files are selected
through each layer's `.weight_layout = .output_input` setting.

## Benchmarks

Run the benchmark suite from the repository root:

```sh
zig build benchmark -Doptimize=ReleaseFast
```

See [benchmarks/README.md](../../benchmarks/README.md) for individual selectors,
methodology, and recorded results.
