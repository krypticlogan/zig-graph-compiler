# Generated model artifacts

ZGC exports a `zgc_model_runner` module for producing a minimal executable from
any generated model type. The consumer supplies a module named `model` that
contains `pub const Model`:

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

const runner_module = zgc_dep.module("zgc_model_runner");
runner_module.addImport("model", model_module);

const artifact = b.addExecutable(.{
    .name = "my-model",
    .root_module = runner_module,
});
artifact.forceUndefinedSymbol(if (target.result.os.tag == .macos)
    "_zgc_run_model"
else
    "zgc_run_model");
b.installArtifact(artifact);
```

The resulting executable contains no logging, timing, input generation, or
output formatting. It exports:

| Symbol                    | Meaning                                                        |
| ------------------------- | -------------------------------------------------------------- |
| `zgc_run_model`           | Execute the generated model through a caller-provided `*Model` |
| `zgc_model_size`          | Return `@sizeOf(Model)`                                        |
| `zgc_model_alignment`     | Return `@alignOf(Model)`                                       |
| `zgc_model_mutable_bytes` | Return the mutable memory-plan byte count                      |

The executable entry point intentionally performs no inference. Runtime input
ownership and binding are application concerns, so an application or benchmark
harness initializes the model and supplies its inputs before calling the
execution function. The stable symbols support model-code isolation, size
analysis, linkage, and disassembly.

Parameters configured with `zgc.memory.Source.embed` are compile-time packed from
logical row-major bytes into the selected physical layout and remain read-only
program data in the artifact. `zgc.memory.Source.embedPacked` accepts already-packed
bytes. Bound inputs remain external, and neither consumes mutable model memory.
