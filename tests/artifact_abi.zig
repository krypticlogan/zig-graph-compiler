const std = @import("std");
const zgc = @import("zgc");
const models = @import("fixtures/models.zig");

test "artifact ABI initializes, describes, executes, and copies a model" {
    const Model = models.BasicModel;
    const Api = zgc.artifact.ABI.Adapter(Model);
    var storage: [@sizeOf(Model)]u8 align(@alignOf(Model)) = undefined;

    try std.testing.expectEqual(@as(u32, 1), Api.abiVersion());
    try std.testing.expectEqual(@sizeOf(Model), Api.modelSize());
    try std.testing.expectEqual(@alignOf(Model), Api.modelAlignment());
    try std.testing.expectEqual(.ok, Api.modelInit(&storage, storage.len));
    defer Api.modelDeinit(&storage);

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
        Api.modelCopySource(&storage, source.key, &input, @sizeOf(@TypeOf(input))),
    );
    try std.testing.expectEqual(.ok, Api.modelRun(&storage));

    try std.testing.expectEqual(@as(usize, 1), Api.outputCount());
    var output_descriptor: zgc.artifact.ABI.TensorDescriptor = undefined;
    try std.testing.expectEqual(.ok, Api.outputDescriptor(0, &output_descriptor));
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, output_descriptor.shape.?[0..2]);

    var view: zgc.artifact.ABI.TensorView = undefined;
    try std.testing.expectEqual(.ok, Api.modelOutputView(&storage, 0, &view));
    try std.testing.expect(view.data != null);

    var output: [6]f32 = undefined;
    try std.testing.expectEqual(
        .ok,
        Api.modelCopyOutput(&storage, 0, &output, @sizeOf(@TypeOf(output))),
    );
    try std.testing.expectEqualSlices(f32, &.{ 0, 4, 2, 0, 0, 6 }, &output);
}

test "artifact ABI requires and accepts bound model sources" {
    const Model = models.BoundInputModel;
    const Api = zgc.artifact.ABI.Adapter(Model);
    var storage: [@sizeOf(Model)]u8 align(@alignOf(Model)) = undefined;
    try std.testing.expectEqual(.ok, Api.modelInit(&storage, storage.len));

    try std.testing.expectEqual(.missing_binding, Api.modelRun(&storage));
    const input = [_]f32{ 1, 2, 3, 4 };
    try std.testing.expectEqual(
        .ok,
        Api.modelBindSource(&storage, @intFromEnum(models.ParameterSources.input), &input, @sizeOf(@TypeOf(input))),
    );
    try std.testing.expectEqual(.ok, Api.modelRun(&storage));

    var output: [4]f32 = undefined;
    try std.testing.expectEqual(.ok, Api.modelCopyOutput(&storage, 0, &output, @sizeOf(@TypeOf(output))));
    try std.testing.expectEqualSlices(f32, &.{ 3, 1, 3.5, 7 }, &output);
}
