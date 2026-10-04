const std = @import("std");
const zgc = @import("zgc");

test "mul and div use trailing-axis broadcasting" {
    var matrix_storage = [_]f32{ 2, 4, 8, 16, 32, 64 };
    var vector_storage = [_]f32{ 2, 4, 8 };
    var product_storage: [6]f32 = undefined;
    var quotient_storage: [6]f32 = undefined;
    const matrix: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &matrix_storage,
        .shape = .{ 2, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };
    const vector: zgc.core.Tensor.ConstView(f32, 1) = .{
        .storage = &vector_storage,
        .shape = .{3},
        .strides = .{1},
        .offset = 0,
    };
    const product: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &product_storage,
        .shape = .{ 2, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };
    const quotient: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &quotient_storage,
        .shape = .{ 2, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };

    const mul_op: zgc.Op = .{ .compute = .mul };
    const div_op: zgc.Op = .{ .compute = .div };
    mul_op.execute(.{ matrix, vector }, product);
    div_op.execute(.{ matrix, vector }, quotient);

    try std.testing.expectEqualSlices(f32, &.{ 4, 16, 64, 32, 128, 512 }, &product_storage);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 8, 8, 8 }, &quotient_storage);
}
