const std = @import("std");
const zgc = @import("zgc");
const models = @import("fixtures/models.zig");

test "artifact ABI initializes, describes, executes, and copies a model" {
    const Executable = models.BasicExecutable;
    const Api = zgc.artifact.ABI.Adapter(Executable);
    var storage: [@sizeOf(Executable.Runtime)]u8 align(@alignOf(Executable.Runtime)) = undefined;

    try std.testing.expectEqual(@as(u32, 1), Api.abiVersion());
    try std.testing.expectEqual(Executable.structural_fingerprint, Api.executableStructuralFingerprint());
    try std.testing.expectEqual(@sizeOf(Executable.Runtime), Api.runtimeSize());
    try std.testing.expectEqual(@alignOf(Executable.Runtime), Api.runtimeAlignment());
    try std.testing.expectEqual(.ok, Api.runtimeInit(&storage, storage.len));
    defer Api.runtimeDeinit(&storage);

    try std.testing.expectEqual(@as(usize, 1), Api.sourceCount());
    var source: zgc.artifact.ABI.SourceDescriptor = undefined;
    try std.testing.expectEqual(.ok, Api.sourceDescriptor(0, &source));
    try std.testing.expectEqualStrings("input", std.mem.span(source.name));
    try std.testing.expectEqual(.input, source.kind);
    try std.testing.expectEqual(.owned, source.binding);
    try std.testing.expectEqual(@as(u32, 2), source.tensor.rank);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, source.tensor.shape.?[0..2]);

    const input = [_]f32{ -1, 2, -3, 4, -5, 6 };
    try std.testing.expectEqual(
        .ok,
        Api.runtimeCopySource(&storage, source.key, &input, @sizeOf(@TypeOf(input))),
    );
    try std.testing.expectEqual(.ok, Api.runtimeRun(&storage));

    try std.testing.expectEqual(@as(usize, 1), Api.outputCount());
    var output_descriptor: zgc.artifact.ABI.TensorDescriptor = undefined;
    try std.testing.expectEqual(.ok, Api.outputDescriptor(0, &output_descriptor));
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, output_descriptor.shape.?[0..2]);

    var view: zgc.artifact.ABI.TensorView = undefined;
    try std.testing.expectEqual(.ok, Api.runtimeOutputView(&storage, 0, &view));
    try std.testing.expect(view.data != null);

    var output: [6]f32 = undefined;
    try std.testing.expectEqual(
        .ok,
        Api.runtimeCopyOutput(&storage, 0, &output, @sizeOf(@TypeOf(output))),
    );
    try std.testing.expectEqualSlices(f32, &.{ 0, 4, 2, 0, 0, 6 }, &output);
}

test "artifact ABI requires and accepts bound model sources" {
    const Executable = models.BoundInputExecutable;
    const Api = zgc.artifact.ABI.Adapter(Executable);
    var storage: [@sizeOf(Executable.Runtime)]u8 align(@alignOf(Executable.Runtime)) = undefined;
    try std.testing.expectEqual(.ok, Api.runtimeInit(&storage, storage.len));

    try std.testing.expectEqual(.missing_binding, Api.runtimeRun(&storage));
    const input = [_]f32{ 1, 2, 3, 4 };
    try std.testing.expectEqual(
        .ok,
        Api.runtimeBindSource(&storage, @intFromEnum(models.ParameterSources.input), &input, @sizeOf(@TypeOf(input))),
    );
    try std.testing.expectEqual(.ok, Api.runtimeRun(&storage));

    var output: [4]f32 = undefined;
    try std.testing.expectEqual(.ok, Api.runtimeCopyOutput(&storage, 0, &output, @sizeOf(@TypeOf(output))));
    try std.testing.expectEqualSlices(f32, &.{ 3, 1, 3.5, 7 }, &output);
}

test "artifact ABI renders inspection sections without a model instance" {
    const Api = zgc.artifact.ABI.Adapter(models.BasicExecutable);
    var required: usize = 0;
    try std.testing.expectEqual(.ok, Api.executableInspect(1, null, 0, &required));
    try std.testing.expect(required > 0);
    var bytes: [512]u8 = undefined;
    try std.testing.expectEqual(.buffer_too_small, Api.executableInspect(1, &bytes, required - 1, &required));
    try std.testing.expectEqual(.ok, Api.executableInspect(1, &bytes, bytes.len, &required));
    try std.testing.expect(std.mem.indexOf(u8, bytes[0..required], "Capacity(") != null);
    try std.testing.expectEqual(.invalid_argument, Api.executableInspect(99, null, 0, &required));
}
