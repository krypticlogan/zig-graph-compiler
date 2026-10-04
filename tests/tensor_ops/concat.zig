const std = @import("std");
const zgc = @import("zgc");

test "concat materializes strided inputs along a selected axis" {
    var left_storage = [_]f32{ 1, 2, 3, 4 };
    var right_storage = [_]f32{ 5, 6 };
    var output_storage: [6]f32 = undefined;
    const left: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &left_storage,
        .shape = .{ 2, 2 },
        .strides = .{ 1, 2 },
        .offset = 0,
    };
    const right: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &right_storage,
        .shape = .{ 2, 1 },
        .strides = .{ 1, 1 },
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &output_storage,
        .shape = .{ 2, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };
    const op: zgc.Op = .{ .compute = .{ .concat = .{ .axis = 1 } } };

    op.execute(.{ left, right }, output);
    try std.testing.expectEqualSlices(f32, &.{ 1, 3, 5, 2, 4, 6 }, &output_storage);
}
