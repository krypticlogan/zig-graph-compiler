# Generated model artifacts

ZGC can specialize a versioned C ABI around any generated model. The ABI keeps
the model opaque to its host: the host discovers its size, alignment, sources,
and outputs through exported functions, and supplies aligned storage for each
model instance.

The installed C declaration is
[`bindings/c/include/zgc_model.h`](../bindings/c/include/zgc_model.h).
`zgc_abi_version()` returns `ZGC_ABI_VERSION` so a host can reject an
incompatible artifact before creating a model.

## Compile a serialized graph

The native `zgc` command specializes a `.zgir` graph and emits the shared
library directly:

```sh
zig build
./zig-out/bin/zgc compile model.zgir -o model.dylib
```

The default optimization mode is `ReleaseFast`. Select another mode, Zig
executable, target, or source checkout explicitly when needed:

```sh
zgc compile model.zgir \
  -o libmodel.so \
  -O ReleaseSafe \
  --target x86_64-linux-gnu \
  --zig /path/to/zig \
  --zgc-root /path/to/zgc
```

The command embeds the serialized graph, parses it at compile time, replays it
through `DefinitionBuilder`, and exports the specialized model through the C
ABI. Parsing, validation, planning, and kernel selection do not occur when the
resulting library is loaded or executed.

Build-system integrations can use the exported `zgc_zgir_model` module. Supply
its required `graph` import with a module exposing `pub const source`, then use
the resulting model module with `zgc_model_abi`:

```zig
const graph_source = b.createModule(.{
    .root_source_file = b.path("src/graph.zig"),
    .target = target,
    .optimize = optimize,
});

const model = zgc_dep.module("zgc_zgir_model");
model.addImport("graph", graph_source);

const abi = zgc_dep.module("zgc_model_abi");
abi.addImport("model", model);
```

The graph source module can embed the serialized file without translating it:

```zig
pub const source = @embedFile("model.zgir");
```

## Build integration

The consumer supplies a module named `model` containing `pub const Model`. Use
`zgc_model_abi` as the root module of a static or shared library:

```zig
const zgc_dep = b.dependency("zgc", .{
    .target = target,
    .optimize = optimize,
});

const model_module = b.createModule(.{
    .root_source_file = b.path("src/model.zig"),
    .target = target,
    .optimize = optimize,
    // Add imports required by the model definition here.
});

const abi_module = zgc_dep.module("zgc_model_abi");
abi_module.addImport("model", model_module);

const artifact = b.addLibrary(.{
    .name = "my-model",
    .linkage = .dynamic, // Use .static for a static archive.
    .root_module = abi_module,
});
artifact.forceUndefinedSymbol(if (target.result.os.tag == .macos)
    "_zgc_model_run"
else
    "zgc_model_run");
b.installArtifact(artifact);
```

For disassembly or standalone binary inspection, use `zgc_model_runner` as an
executable root module instead. It exports the same ABI and has an inert entry
point:

```zig
const runner_module = zgc_dep.module("zgc_model_runner");
runner_module.addImport("model", model_module);

const artifact = b.addExecutable(.{
    .name = "my-model",
    .root_module = runner_module,
});
artifact.forceUndefinedSymbol(if (target.result.os.tag == .macos)
    "_zgc_model_run"
else
    "zgc_model_run");
b.installArtifact(artifact);
```

## Host lifecycle

A host uses a generated artifact in this order:

1. Verify `zgc_abi_version()`.
2. Allocate `zgc_model_size()` bytes at `zgc_model_alignment()` alignment.
3. Initialize the instance with `zgc_model_init()`.
4. Enumerate source descriptors and either copy or bind runtime source data.
5. Call `zgc_model_run()`.
6. Read outputs through a borrowed view or copy them into row-major host memory.
7. Call `zgc_model_deinit()` before releasing the instance storage.

`zgc_model_init()` does not allocate. `zgc_model_mutable_bytes()` reports the
mutable tensor arena inside the opaque model allocation; it is informational
and is already included in `zgc_model_size()`.

## Sources

`zgc_get_source_descriptor()` exposes each source's stable numeric key, name,
kind, binding policy, dtype, shape, and physical strides.

- `ZGC_BINDING_OWNED` accepts logical row-major data through
  `zgc_model_copy_source()`. The adapter packs it into the compiled layout.
- `ZGC_BINDING_BOUND` accepts borrowed physical-layout storage through
  `zgc_model_bind_source()`. The memory must remain alive and unchanged while
  the model runs.
- `ZGC_BINDING_EMBEDDED` is already part of the artifact and accepts neither
  operation.

`zgc_model_run()` reports `ZGC_STATUS_MISSING_BINDING` when a required bound
input has not been supplied.

## Outputs

`zgc_model_output_view()` returns a borrowed physical view into model storage.
The descriptor's element offset and strides define logical indexing. The view
remains valid until the model is run again or destroyed.

`zgc_model_copy_output()` is the portable boundary: it materializes the output
in logical row-major order and therefore does not expose compiler-selected
layouts to the host.

All source and output geometry is generated from the compiled model. The run
path does not rediscover shapes, layouts, or kernel choices at runtime.

## Exported functions

| Group | Functions |
| --- | --- |
| Version and storage | `zgc_abi_version`, `zgc_model_size`, `zgc_model_alignment`, `zgc_model_mutable_bytes` |
| Lifecycle | `zgc_model_init`, `zgc_model_deinit`, `zgc_model_run` |
| Sources | `zgc_source_count`, `zgc_get_source_descriptor`, `zgc_model_copy_source`, `zgc_model_bind_source` |
| Outputs | `zgc_output_count`, `zgc_get_output_descriptor`, `zgc_model_output_view`, `zgc_model_copy_output` |

Every fallible operation returns `zgc_status`. The status values and descriptor
layouts are part of ABI version 1.

## Python runtime

The reusable package under `bindings/python/runtime` loads any ABI-compatible
model; it is not generated per model and has no compiler or Zig dependency.
Install it with:

```sh
pip install ./bindings/python/runtime
```

Then load an artifact and address its sources by their compiled names:

```python
import zgc

with zgc.load("libmy-model.so") as model:
    output = model(input=[1.0, 2.0, 3.0, 4.0])
    print(output.shape)
    print(output.tolist())
```

Sources accept ordinary Python iterables and objects implementing the Python
buffer protocol, including `array.array`, `memoryview`, ctypes arrays, and
compatible arrays from third-party packages. `model.set_source(name, values)`
always copies logical values into model-owned storage. `model.bind(name,
buffer)` is the explicit zero-copy path; it requires an exact, writable
physical layout and retains the buffer until it is rebound or the model is
closed.

Copied outputs own contiguous storage. `Tensor.buffer` exposes that storage as
a shaped `memoryview`, while `Tensor.tolist()` and `Tensor.values` provide
ordinary Python values. The package owns model allocation and lifecycle and
uses only the Python standard library.

The separate `zgc-compiler` package owns Zig discovery, compilation, and the
compiled-model cache. Its output is the same shared-library ABI described in
this document, so the compiler package is not required to load or execute a
saved artifact. See [Python compiler package](python-compiler.md).

## Python compiler

The Python integration is divided into independent runtime and compiler
packages. `zgc-runtime` loads and executes an existing model artifact. It does
not locate Zig, construct graphs, invoke a compiler, or manage compiled-model
caches. `zgc-compiler` depends on the runtime and produces the same versioned
shared-library artifact used by the C ABI and runtime package.

Install the runtime alone for artifact execution, or install the compiler to
receive both packages:

```sh
python -m pip install ./bindings/python/runtime
python -m pip install ./bindings/python/runtime ./bindings/python/compiler
```

The compiler frontend constructs and serializes semantic graphs, generates their
Zig definitions, and compiles them through the model artifact ABI:

```python
import zgc_compiler as zgc

graph = zgc.Graph()
inputs = graph.input("input", zgc.f32, (128, 64), binding=zgc.bound)
weights = graph.parameter("weight", zgc.f32, (64, 10))
result = inputs.matmul(weights).relu()
graph.outputs(result)

print(graph.serialize())
print(graph.fingerprint)

artifact = zgc.Compiler().compile_graph(graph, name="classifier")
```

Direct graph calls and chained value calls produce the same representation.
The serialized form is the language-neutral `.zgir` DAG accepted by the ZGC
frontend. Its value identifiers follow graph construction order:

```text
zgir 1

%0 = input(name="input", dtype=f32, shape=[128, 64], binding=bound)
%1 = parameter(name="weight", dtype=f32, shape=[64, 10], binding=owned)
%2 = matmul(%0, %1)
%3 = relu(%2)

outputs = [%3]
```

Source calls identify source kinds, while dtype, binding, boundary, and similar
tags use the corresponding frontend enums. Operation names mirror the semantic
ZGC definition surface. The graph fingerprint is the SHA-256 digest of this
canonical serialization.

`parse_graph(text)` reconstructs the normalized graph. `compile_graph` accepts
a mutable `Graph`, a `FrozenGraph`, or serialized graph text, and
`compile_graph_file` reads the same representation from a `.zgir` file. Zig
parses the serialized graph at compile time and replays it directly through
`DefinitionBuilder`.

Handwritten Zig files exporting `pub const Model` remain supported:

```python
import zgc
from zgc_compiler import Compiler, CompilerConfig

compiler = Compiler(CompilerConfig(
    zig="auto",
    cache_dir=".zgc-cache",
    zgc_root="/path/to/zgc",
))
artifact = compiler.compile_model("model.zig")

with zgc.load(artifact.path) as model:
    output = model(input=[1.0, 2.0, 3.0, 4.0])
```

`compile_model` returns a path to an ordinary ABI-compatible dynamic library.
The artifact can be copied with `artifact.save(path)` and executed later with
only `zgc-runtime` installed. Generated definitions use this same artifact
compilation path.

`zgc.dtype` and `zgc.SourceBinding` are shared by graph construction and model
execution. The compiler package re-exports those same types, so values from
either namespace are interchangeable.

### Toolchain selection

`CompilerConfig.zig` accepts an executable path or `"auto"`. Automatic
resolution checks, in order:

1. `ZGC_ZIG`;
2. the managed ZGC toolchain cache;
3. `zig` on `PATH`.

Every candidate must report the Zig version supported by the compiler package.
If no compatible compiler exists, install the pinned toolchain explicitly:

```sh
python -m zgc toolchain install
python -m zgc toolchain verify
```

The runtime package owns the `zgc` command. Installed companion packages can
register additional commands through its CLI extension point. The compiler
depends on that registration surface and contributes `compile`, `toolchain`,
and `cache` without adding a compiler dependency to the runtime.
`python -m zgc_compiler` and the `zgc-compiler` console command expose the same
commands directly.

Managed toolchains use the operating system's user cache by default. Set
`ZGC_TOOLCHAIN_DIR` or pass `toolchain_dir` to select another location. The
installer downloads the pinned platform archive from ziglang.org, verifies its
SHA-256 digest from the release index, verifies `zig version`, and installs the
result atomically.

### Model cache

Compiled artifacts default to `./.zgc-cache` and can be redirected with
`CompilerConfig.cache_dir`, `--cache-dir`, or `ZGC_CACHE_DIR`. Cache identities
include the model definition, imported model modules, ZGC Zig sources,
optimization mode, target, compiler format, and Zig version.

```sh
python -m zgc compile model.zgir --zgc-root /path/to/zgc
python -m zgc compile model.zig --zgc-root /path/to/zgc
python -m zgc cache path
python -m zgc cache clear
```

Compilation uses a per-key lock and publishes completed cache entries with an
atomic rename. Graph entries contain the canonical `graph.zgir`, the invariant
model adapter, and a manifest. Artifact entries contain the model library, a
manifest, and the Zig build log.
