const generated_executable = @import("executable");
const abi = @import("zgc").artifact.ABI;
const root = @import("root");
const inspectable = if (@hasDecl(root, "inspectable")) root.inspectable else false;

pub const Api = abi.Adapter(generated_executable.Executable);

comptime {
    @export(&Api.abiVersion, .{ .name = "zgc_abi_version" });
    @export(&Api.executableStructuralFingerprint, .{ .name = "zgc_executable_structural_fingerprint" });
    @export(&Api.runtimeSize, .{ .name = "zgc_runtime_size" });
    @export(&Api.runtimeAlignment, .{ .name = "zgc_runtime_alignment" });
    @export(&Api.runtimeMutableBytes, .{ .name = "zgc_runtime_mutable_bytes" });
    if (inspectable) {
        @export(&Api.executableInspect, .{ .name = "zgc_executable_inspect" });
    }
    @export(&Api.runtimeInit, .{ .name = "zgc_runtime_init" });
    @export(&Api.runtimeDeinit, .{ .name = "zgc_runtime_deinit" });
    @export(&Api.runtimeRun, .{ .name = "zgc_runtime_run" });
    @export(&Api.sourceCount, .{ .name = "zgc_executable_source_count" });
    @export(&Api.sourceDescriptor, .{ .name = "zgc_executable_get_source_descriptor" });
    @export(&Api.runtimeCopySource, .{ .name = "zgc_runtime_copy_source" });
    @export(&Api.runtimeBindSource, .{ .name = "zgc_runtime_bind_source" });
    @export(&Api.outputCount, .{ .name = "zgc_executable_output_count" });
    @export(&Api.outputDescriptor, .{ .name = "zgc_executable_get_output_descriptor" });
    @export(&Api.outputBinding, .{ .name = "zgc_executable_get_output_binding" });
    @export(&Api.runtimeBindOutput, .{ .name = "zgc_runtime_bind_output" });
    @export(&Api.runtimeOutputView, .{ .name = "zgc_runtime_output_view" });
    @export(&Api.runtimeCopyOutput, .{ .name = "zgc_runtime_copy_output" });
}

/// Reference one exported function when the ABI is linked into an executable
/// whose entry point otherwise has no dependency on the executable adapter.
pub fn retain() void {
    inline for (.{
        &Api.abiVersion,
        &Api.executableStructuralFingerprint,
        &Api.runtimeSize,
        &Api.runtimeAlignment,
        &Api.runtimeMutableBytes,
        &Api.runtimeInit,
        &Api.runtimeDeinit,
        &Api.runtimeRun,
        &Api.sourceCount,
        &Api.sourceDescriptor,
        &Api.runtimeCopySource,
        &Api.runtimeBindSource,
        &Api.outputCount,
        &Api.outputDescriptor,
        &Api.outputBinding,
        &Api.runtimeBindOutput,
        &Api.runtimeOutputView,
        &Api.runtimeCopyOutput,
    }) |function| {
        @import("std").mem.doNotOptimizeAway(function);
    }
    if (inspectable) @import("std").mem.doNotOptimizeAway(&Api.executableInspect);
}
