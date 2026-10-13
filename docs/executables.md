# Generated executables

ZGC compiles a graph definition into an executable artifact. Its versioned C
ABI exposes the executable's metadata and lets a host allocate, initialize,
and run an opaque runtime instance.

The installed C declaration is
[`bindings/c/zgc_executable.h`](../bindings/c/zgc_executable.h).
`zgc_abi_version()` returns `ZGC_ABI_VERSION` so a host can reject an
incompatible artifact before creating a runtime.

## Compile a serialized graph

The native `zgc` command specializes a `.zgir` graph and emits the shared
library directly. The [serialized graph format](zgir.md) is described
separately:

```sh
zig build
./zig-out/bin/zgc compile graph.zgir -o graph.dylib
```

The default optimization mode is `ReleaseFast`. Select another mode, Zig
executable, target, or source checkout explicitly when needed:

```sh
zgc compile graph.zgir \
  -o libgraph.so \
  -O ReleaseSafe \
  --selection-target fast \
  --target x86_64-linux-gnu \
  --zig /path/to/zig \
  --zgc-root /path/to/zgc
```

`--selection-target` accepts `balanced`, `memory`, `fast`, or `small`;
`--frontier-index` instead forces a retained executable candidate. These
options select the compiled executable, not a runtime dispatch path.

The command embeds the serialized graph, parses it at compile time, replays it
through `DefinitionBuilder`, and exports the specialized executable through the C
ABI. Parsing, validation, planning, and kernel selection do not occur when the
resulting library is loaded or executed.

Build-system integrations can use the exported `zgc_zgir_executable` module. Supply
its required `graph` import with a module exposing `pub const source`, then use
the resulting executable module with `zgc_executable_abi`:

```zig
const graph_source = b.createModule(.{
    .root_source_file = b.path("src/graph.zig"),
    .target = target,
    .optimize = optimize,
});

const executable = zgc_dep.module("zgc_zgir_executable");
executable.addImport("graph", graph_source);

const abi = zgc_dep.module("zgc_executable_abi");
abi.addImport("executable", executable);
```

The graph source module can embed the serialized file without translating it:

```zig
pub const source = @embedFile("graph.zgir");
```

## Build integration

The consumer supplies a module named `executable` containing
`pub const Executable`. Use
`zgc_executable_abi` as the root module of a static or shared library:

```zig
const zgc_dep = b.dependency("zgc", .{
    .target = target,
    .optimize = optimize,
});

const executable_module = b.createModule(.{
    .root_source_file = b.path("src/definition.zig"),
    .target = target,
    .optimize = optimize,
    // Add imports required by the definition here.
});

const abi_module = zgc_dep.module("zgc_executable_abi");
abi_module.addImport("executable", executable_module);

const artifact = b.addLibrary(.{
    .name = "my-executable",
    .linkage = .dynamic, // Use .static for a static archive.
    .root_module = abi_module,
});
artifact.forceUndefinedSymbol(if (target.result.os.tag == .macos)
    "_zgc_runtime_run"
else
    "zgc_runtime_run");
b.installArtifact(artifact);
```

For disassembly or standalone binary inspection, use `zgc_executable_runner` as an
executable root module instead. It exports the same ABI and has an inert entry
point:

```zig
const runner_module = zgc_dep.module("zgc_executable_runner");
runner_module.addImport("executable", executable_module);

const artifact = b.addExecutable(.{
    .name = "my-executable",
    .root_module = runner_module,
});
artifact.forceUndefinedSymbol(if (target.result.os.tag == .macos)
    "_zgc_runtime_run"
else
    "zgc_runtime_run");
b.installArtifact(artifact);
```

## Host lifecycle

A host uses a generated artifact in this order:

1. Verify `zgc_abi_version()` and, when comparing executable structure,
   read `zgc_executable_structural_fingerprint()`.
2. Allocate `zgc_runtime_size()` bytes at `zgc_runtime_alignment()` alignment.
3. Initialize the instance with `zgc_runtime_init()`.
4. Enumerate source descriptors and either copy or bind runtime source data.
5. Call `zgc_runtime_run()`.
6. Read outputs through a borrowed view or copy them into row-major host memory.
7. Call `zgc_runtime_deinit()` before releasing the instance storage.

`zgc_runtime_init()` does not allocate. `zgc_runtime_mutable_bytes()` reports the
mutable tensor arena inside the runtime allocation; it is informational
and is already included in `zgc_runtime_size()`.

## Sources

`zgc_executable_get_source_descriptor()` exposes each source's stable numeric key, name,
kind, binding policy, dtype, shape, and physical strides.

- `ZGC_BINDING_OWNED` accepts logical row-major data through
  `zgc_runtime_copy_source()`. The adapter packs it into the compiled layout.
- `ZGC_BINDING_BOUND` accepts borrowed physical-layout storage through
  `zgc_runtime_bind_source()`. The memory must remain alive and unchanged while
  the runtime runs.
- `ZGC_BINDING_EMBEDDED` is already part of the artifact and accepts neither
  operation.

`zgc_runtime_run()` reports `ZGC_STATUS_MISSING_BINDING` when a required bound
input has not been supplied.

## Outputs

`zgc_runtime_output_view()` returns a borrowed physical view into runtime storage.
The descriptor's element offset and strides define logical indexing. The view
remains valid until the runtime is run again or destroyed.

`zgc_runtime_copy_output()` is the portable boundary: it materializes the output
in logical row-major order and therefore does not expose compiler-selected
layouts to the host.

All source and output geometry is generated from the executable. The run
path does not rediscover shapes, layouts, or kernel choices at runtime.

## Exported functions

| Group | Functions |
| --- | --- |
| Version, identity, and storage | `zgc_abi_version`, `zgc_executable_structural_fingerprint`, `zgc_runtime_size`, `zgc_runtime_alignment`, `zgc_runtime_mutable_bytes` |
| Optional inspection | `zgc_executable_inspect` (only with `--inspectable`) |
| Lifecycle | `zgc_runtime_init`, `zgc_runtime_deinit`, `zgc_runtime_run` |
| Sources | `zgc_executable_source_count`, `zgc_executable_get_source_descriptor`, `zgc_runtime_copy_source`, `zgc_runtime_bind_source` |
| Outputs | `zgc_executable_output_count`, `zgc_executable_get_output_descriptor`, `zgc_runtime_output_view`, `zgc_runtime_copy_output` |

Every fallible operation returns `zgc_status`. The status values and descriptor
layouts are part of ABI version 1. The structural fingerprint is a `uint64_t`
hash of the selected executable graph. It is not a hash of the source text or
the library bytes, and does not identify embedded parameter values.
When enabled, `zgc_executable_inspect` renders the selected executable's graph and memory-plan sections
without initializing it. Query the required byte count with a null destination
before supplying a buffer. See [executable inspection](inspection.md).

## Python runtime

The reusable package under `bindings/python/runtime` loads any ABI-compatible
executable; it is not generated per executable and has no compiler or Zig dependency.
Install it with:

```sh
pip install ./bindings/python/runtime
```

Then load an artifact and address its sources by their compiled names:

```python
import zgc

with zgc.load("libgraph.so") as runtime:
    output, = runtime.run(input=[1.0, 2.0, 3.0, 4.0])
    print(output.shape)
    print(output.tolist())
```

Sources accept ordinary Python iterables and objects implementing the Python
buffer protocol, including `array.array`, `memoryview`, ctypes arrays, and
compatible arrays from third-party packages. `runtime.set_source(name, values)`
always copies logical values into runtime-owned storage. `runtime.bind(name,
buffer)` is the explicit zero-copy path; it requires an exact, writable
physical layout and retains the buffer until it is rebound or the runtime is
closed.

Copied outputs own contiguous storage. `Tensor.buffer` exposes that storage as
a shaped `memoryview`, while `Tensor.tolist()` and `Tensor.values` provide
ordinary Python values. The package owns runtime allocation and lifecycle and
uses only the Python standard library.

The separate `zgc-compiler` package provides Zig discovery and a Python
frontend that delegates compilation to the native `zgc` CLI. Its output is the same shared-library ABI described in
this document, so the compiler package is not required to load or execute a
saved artifact. See [Python compiler](#python-compiler).

## Python compiler

The Python integration is divided into independent runtime and compiler
packages. `zgc-runtime` loads and executes an existing executable artifact. It does
not locate Zig, construct graphs, invoke a compiler, or manage compiled-executable
caches. `zgc-compiler` depends on the runtime and produces the same versioned
shared-library artifact used by the C ABI and runtime package.

Install the runtime alone for artifact execution, or install the compiler to
receive both packages:

```sh
python -m pip install ./bindings/python/runtime
python -m pip install ./bindings/python/runtime ./bindings/python/compiler
```

The compiler frontend constructs and serializes semantic graphs, then uses
Zig's `.zgir` frontend to specialize the executable artifact ABI:

```python
import zgc_compiler as zgc

graph = zgc.Graph()
inputs = graph.input("input", zgc.f32, (128, 64), binding=zgc.bound)
weights = graph.parameter("weight", zgc.f32, (64, 10))
result = inputs.matmul(weights).relu()
graph.outputs(result)

print(graph.serialize())
print(graph.fingerprint)

artifact = zgc.Compiler().compile(graph, name="classifier")
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

`Compiler.compile` accepts a mutable `Graph`, a `FrozenGraph`, serialized graph
text, or a `.zgir` or `.zig` file path. Python passes supplied text
through without parsing it; Zig parses and replays it through
`DefinitionBuilder` at compile time. See [`.zgir` format](zgir.md) for the
syntax and operation vocabulary.

Handwritten Zig files exporting `pub const Executable` are also supported:

```python
import zgc
from zgc_compiler import Compiler, CompilerConfig

compiler = Compiler(CompilerConfig(
    zig="auto",
    cache_dir=".zgc-cache",
    zgc_root="/path/to/zgc",
))
artifact = compiler.compile("definition.zig")

with zgc.load(artifact.path) as runtime:
    output = runtime.run(input=[1.0, 2.0, 3.0, 4.0])
```

`Compiler.compile` returns an ordinary ABI-compatible dynamic library artifact.
The artifact can be copied with `artifact.save(path)` and executed later with
only `zgc-runtime` installed. Generated definitions use this same artifact
compilation path.

`zgc.dtype` and `zgc.SourceBinding` are shared by graph construction and runtime
execution. The compiler package re-exports those same types, so values from
either namespace are interchangeable.

`CompilerConfig.selection_target` and `CompilerConfig.frontier_index` select
the program for serialized graphs. Compiled artifacts expose
`structural_fingerprint` for host-native builds; loaded Python runtimes expose
the same value through `Runtime.structural_fingerprint`.
Set `CompilerConfig(inspectable=True)` to include inspection data and export
`zgc_executable_inspect`; it is omitted by default.

### Toolchain selection

`CompilerConfig.zig` accepts an executable path or `"auto"`. Automatic
resolution checks, in order:

1. `ZGC_ZIG`;
2. the managed ZGC toolchain cache;
3. `zig` on `PATH`.

Every candidate must report the Zig version supported by the compiler package.
If no compatible compiler exists, install the pinned toolchain explicitly:

```sh
zgc-compiler toolchain install
zgc-compiler toolchain verify
```

The native `zgc` command owns compilation, inspection, and cache commands.
The runtime distribution is `zgc-runtime` and installs no command-line tool.
The compiler package exposes only toolchain commands through `zgc-compiler`
or `python -m zgc_compiler`.
`Compiler.compile`, `Compiler.inspect`, `cache_path`, and
`clear_cache` delegate to native `zgc`. Set `CompilerConfig.zgc_cli` or
`ZGC_CLI` when the native executable is not on `PATH`.

Managed toolchains use the operating system's user cache by default. Set
`ZGC_TOOLCHAIN_DIR` or pass `toolchain_dir` to select another location. The
installer downloads the pinned platform archive from ziglang.org, verifies its
SHA-256 digest from the release index, verifies `zig version`, and installs the
result atomically.

### Executable cache

Compiled artifacts default to `./.zgc-cache` and can be redirected with
`CompilerConfig.cache_dir`, `--cache-dir`, or `ZGC_CACHE_DIR`. Native ZGC
manages the cache and uses Zig's local build cache to track source dependencies.
For Python graph compilation, Python hashes the canonical graph before
serializing it and looks up a cached artifact under `python/graphs/`. A valid
match returns immediately without writing `.zgir` or invoking native ZGC.
The lookup also accounts for compilation options, the toolchain, and local
compiler source changes. A miss delegates compilation to native ZGC.

```sh
zgc compile graph.zgir --zgc-root /path/to/zgc
zgc compile definition.zig --zgc-root /path/to/zgc
zgc cache path
zgc cache clear
```

Pass `--cache-output` to write an executable artifact under the native cache.
Serialized graph adapters live under `graphs/`, compiled libraries under
`executables/`, and Zig build data under `zig/`.
