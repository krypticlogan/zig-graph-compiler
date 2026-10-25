# Executable inspection

Inspection is separate from graph construction, memory planning, kernel
dispatch, and runtime execution. Inspection representations are included only
when requested during compilation.

## Inspect a compiled library

Compile with `--inspectable`, then pass the shared library to `zgc inspect`:

```sh
zgc compile graph.zgir --inspectable -o graph.dylib
zgc inspect graph.dylib
zgc inspect graph.dylib executable
zgc inspect graph.dylib memory-plan
```

`all` is the default. Other sections are `summary`, `raw-graph`,
`semantic-graph`, `executable`, `representations`, `tree`, and `memory-plan`.
Python can call `zgc_compiler.Compiler().inspect("graph.dylib", "executable")`
to delegate to the same native command. Inspection does not initialize or run a runtime, but it
loads the shared library; inspect only artifacts you trust. A library built for
a different target cannot be loaded by the host inspector.

For Python compilation, set `CompilerConfig(inspectable=True)` before calling
`Compiler.compile(...)`. The default is `False`; non-inspectable libraries omit
the inspection export and its rendering code. The two variants have separate
cache entries.

For a direct Zig build, use the `zgc_inspectable_executable_abi` module instead
of `zgc_executable_abi` when linking the shared library.

An inspectable library exports `zgc_executable_inspect()`. Call it first with a null
destination and zero capacity to obtain the required byte count, then again
with a caller-owned buffer. It renders the executable's compile-time representations
and does not expose a mutable runtime instance. Inspection support adds code to
the library but is not part of `zgc_runtime_run()`. Artifacts built against an
earlier ABI need recompilation before they can be inspected this way.

Use an inspectable artifact or the executable-specific CLI module below to inspect a compiled executable.

## Executable-specific CLI

ZGC exports a `zgc_inspect_cli` module. A consumer supplies a module named
`executable`. If that module exports one generated executable type, the inspector infers
its declaration name. If it exports several generated executables, select one with
`--executable <declaration>`:

```zig
const zgc_dep = b.dependency("zgc", .{
    .target = target,
    .optimize = optimize,
});

const executable_module = b.createModule(.{
    .root_source_file = b.path("src/executable.zig"),
    .target = target,
    .optimize = optimize,
    // Add the imports required by the executable definition here.
});

const cli_module = zgc_dep.module("zgc_inspect_cli");
cli_module.addImport("executable", executable_module);

const inspector = b.addExecutable(.{
    .name = "zgc-inspect",
    .root_module = cli_module,
});
b.installArtifact(inspector);
```

The executable supports these commands:

```text
zgc-inspect all
zgc-inspect --executable 'declaration-name' executable
zgc-inspect summary
zgc-inspect raw-graph
zgc-inspect semantic-graph
zgc-inspect executable
zgc-inspect representations
zgc-inspect tree
zgc-inspect memory-plan
zgc-inspect help
```

`all` is the default. Each inspector is compile-time specialized for the executable
types exported by its supplied module. Declaration discovery checks generated
executable metadata and does not require a declaration named `Executable`.
