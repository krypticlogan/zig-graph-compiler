const std = @import("std");
const zgc = @import("zgc_internal").api;

const Sources = enum(usize) { input };
const definition = blk: {
    var builder = zgc.DefinitionBuilder.init();
    const input = builder.sources(Sources).input(.input, .f32, &.{4});
    builder.output(builder.relu(input));
    builder.output(builder.neg(input));
    break :blk builder.finish();
};

const BoundExecutable = definition.executableWith(.{
    .sources = &.{.{ .source = .input, .binding = .bound }},
    .outputs = .bound,
});

test "all bound outputs write caller buffers and can be rebound between runs" {
    try std.testing.expectEqual(@as(usize, 0), BoundExecutable.memory_plan.byte_count);
    var runtime = BoundExecutable.Runtime.init();
    var input = [_]f32{ -2, 3, -4, 5 };
    var positive: [4]f32 = undefined;
    var negative: [4]f32 = undefined;
    try runtime.bindInput(.input, &input);
    try runtime.bindOutput(0, &positive);
    try runtime.bindOutput(1, &negative);
    try std.testing.expect(runtime.outputIsBound(0));
    try std.testing.expect(runtime.outputIsBound(1));
    runtime.run();
    try std.testing.expectEqualSlices(f32, &.{ 0, 3, 0, 5 }, &positive);
    try std.testing.expectEqualSlices(f32, &.{ 2, -3, 4, -5 }, &negative);
    try std.testing.expectEqualSlices(f32, &positive, runtime.outputView(0).storage);

    input = .{ 1, -1, 2, -2 };
    var next_positive: [4]f32 = undefined;
    try runtime.bindOutput(0, &next_positive);
    runtime.run();
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 2, 0 }, &next_positive);
    try std.testing.expectEqualSlices(f32, &.{ -1, 1, -2, 2 }, &negative);
}

test "bound outputs are independent of input binding" {
    const Executable = definition.executableWith(.{ .outputs = .bound });
    var runtime = Executable.Runtime.init();
    try runtime.copyInput(.input, &.{ -2, 3, -4, 5 });
    var positive: [4]f32 = undefined;
    var negative: [4]f32 = undefined;
    try runtime.bindOutput(0, &positive);
    try runtime.bindOutput(1, &negative);
    runtime.run();
    try std.testing.expectEqualSlices(f32, &.{ 0, 3, 0, 5 }, &positive);
    try std.testing.expectEqualSlices(f32, &.{ 2, -3, 4, -5 }, &negative);
    try std.testing.expectEqual(@as(usize, 4 * @sizeOf(f32)), Executable.memory_plan.byte_count);
}

test "ABI checks all output bindings and writes bound storage" {
    const Api = zgc.artifact.ABI.Adapter(BoundExecutable);
    var storage: [@sizeOf(BoundExecutable.Runtime)]u8 align(@alignOf(BoundExecutable.Runtime)) = undefined;
    try std.testing.expectEqual(.ok, Api.runtimeInit(&storage, storage.len));
    var binding: zgc.artifact.ABI.OutputBinding = undefined;
    try std.testing.expectEqual(.ok, Api.outputBinding(0, &binding));
    try std.testing.expectEqual(.bound, binding);
    try std.testing.expectEqual(.invalid_output, Api.outputBinding(2, &binding));
    try std.testing.expectEqual(.missing_binding, Api.runtimeRun(&storage));

    var input = [_]f32{ -2, 3, -4, 5 };
    var first: [4]f32 = undefined;
    var second: [4]f32 = undefined;
    try std.testing.expectEqual(.ok, Api.runtimeBindSource(&storage, 0, &input, @sizeOf(@TypeOf(input))));
    try std.testing.expectEqual(.size_mismatch, Api.runtimeBindOutput(&storage, 0, &first, 4));
    try std.testing.expectEqual(.ok, Api.runtimeBindOutput(&storage, 0, &first, @sizeOf(@TypeOf(first))));
    try std.testing.expectEqual(.missing_binding, Api.runtimeRun(&storage));
    try std.testing.expectEqual(.ok, Api.runtimeBindOutput(&storage, 1, &second, @sizeOf(@TypeOf(second))));
    try std.testing.expectEqual(.ok, Api.runtimeRun(&storage));
    try std.testing.expectEqualSlices(f32, &.{ 0, 3, 0, 5 }, &first);
    try std.testing.expectEqualSlices(f32, &.{ 2, -3, 4, -5 }, &second);

    try std.testing.expectEqual(.ok, Api.runtimeBindOutput(&storage, 1, &first, @sizeOf(@TypeOf(first))));
    try std.testing.expectEqual(.invalid_argument, Api.runtimeRun(&storage));
}
