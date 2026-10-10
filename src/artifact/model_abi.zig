const generated_model = @import("model");
const abi = @import("zgc").artifact.ABI;

const Api = abi.Adapter(generated_model.Model);

comptime {
    @export(&Api.abiVersion, .{ .name = "zgc_abi_version" });
    @export(&Api.modelSize, .{ .name = "zgc_model_size" });
    @export(&Api.modelAlignment, .{ .name = "zgc_model_alignment" });
    @export(&Api.modelMutableBytes, .{ .name = "zgc_model_mutable_bytes" });
    @export(&Api.modelInit, .{ .name = "zgc_model_init" });
    @export(&Api.modelDeinit, .{ .name = "zgc_model_deinit" });
    @export(&Api.modelRun, .{ .name = "zgc_model_run" });
    @export(&Api.sourceCount, .{ .name = "zgc_source_count" });
    @export(&Api.sourceDescriptor, .{ .name = "zgc_get_source_descriptor" });
    @export(&Api.modelCopySource, .{ .name = "zgc_model_copy_source" });
    @export(&Api.modelBindSource, .{ .name = "zgc_model_bind_source" });
    @export(&Api.outputCount, .{ .name = "zgc_output_count" });
    @export(&Api.outputDescriptor, .{ .name = "zgc_get_output_descriptor" });
    @export(&Api.modelOutputView, .{ .name = "zgc_model_output_view" });
    @export(&Api.modelCopyOutput, .{ .name = "zgc_model_copy_output" });
}

/// Reference one exported function when the ABI is linked into an executable
/// whose entry point otherwise has no dependency on the model adapter.
pub fn retain() void {
    inline for (.{
        &Api.abiVersion,
        &Api.modelSize,
        &Api.modelAlignment,
        &Api.modelMutableBytes,
        &Api.modelInit,
        &Api.modelDeinit,
        &Api.modelRun,
        &Api.sourceCount,
        &Api.sourceDescriptor,
        &Api.modelCopySource,
        &Api.modelBindSource,
        &Api.outputCount,
        &Api.outputDescriptor,
        &Api.modelOutputView,
        &Api.modelCopyOutput,
    }) |function| {
        @import("std").mem.doNotOptimizeAway(function);
    }
}
