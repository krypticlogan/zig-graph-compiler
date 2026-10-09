const std = @import("std");
const zgc = @import("zgc");

test "copy materializes logical values from a strided view" {
    var input_storage = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var output_storage: [6]f32 = undefined;
    const transposed: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &input_storage,
        .shape = .{ 3, 2 },
        .strides = .{ 1, 3 },
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &output_storage,
        .shape = .{ 3, 2 },
        .strides = .{ 2, 1 },
        .offset = 0,
    };

    (zgc.Op{ .compute = .contiguous }).execute(.{transposed}, output);

    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, &output_storage);
}

const Sources = enum(usize) { input };
const Definition = zgc.DefinitionBuilder;

const copy_model = model: {
    var builder = Definition.init();
    const builder_sources = builder.sources(Sources);
    const input = builder_sources.input(.input, .f32, &.{ 2, 3 });
    builder.output(builder.copy(builder.transpose(input, 0, 1)));
    break :model builder.finish().model();
};

test "compiled copy materialization owns fresh lowered storage" {
    const graph = copy_model.executable;
    const transposed = graph.tensors[1].?;
    const materialized = graph.tensors[2].?;

    try std.testing.expectEqual(graph.tensors[0].?.storage_tensor, transposed.storage_tensor);
    try std.testing.expectEqual(@as(usize, 2), materialized.storage_tensor);
    try std.testing.expectEqual([2]isize{ 2, 1 }, materialized.layout.strides[0..2].*);

    var model = copy_model.init();
    try model.copyInput(.input, &.{ 1, 2, 3, 4, 5, 6 });
    model.run();
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, model.outputView(0).storage);
}
