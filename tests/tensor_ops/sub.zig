const std = @import("std");
const zgc = @import("zgc");

test "sub writes elementwise differences" {
    const vector_len = std.simd.suggestVectorLength(f32) orelse 1;
    const len = vector_len + 1;
    var lhs_storage: [len]f32 = undefined;
    var rhs_storage: [len]f32 = undefined;
    var output_storage: [len]f32 = undefined;
    var expected: [len]f32 = undefined;
    for (0..len) |index| {
        lhs_storage[index] = @floatFromInt(index + 3);
        rhs_storage[index] = @floatFromInt(index);
        expected[index] = 3;
    }
    const lhs: zgc.core.Tensor.ConstView(f32, 1) = .{
        .storage = &lhs_storage,
        .shape = .{len},
        .strides = .{1},
        .offset = 0,
    };
    const rhs: zgc.core.Tensor.ConstView(f32, 1) = .{
        .storage = &rhs_storage,
        .shape = .{len},
        .strides = .{1},
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 1) = .{
        .storage = &output_storage,
        .shape = .{len},
        .strides = .{1},
        .offset = 0,
    };

    const op: zgc.Op = .{ .compute = .sub };
    op.execute(.{ lhs, rhs }, output);

    try std.testing.expectEqualSlices(f32, &expected, &output_storage);
}

test "sub supports independently strided integer views" {
    var lhs_storage = [_]i8{ 10, 30, 20, 40 };
    var rhs_storage = [_]i8{ 1, 2, 3, 4 };
    var output_storage: [4]i8 = undefined;
    const lhs: zgc.core.Tensor.ConstView(i8, 2) = .{
        .storage = &lhs_storage,
        .shape = .{ 2, 2 },
        .strides = .{ 1, 2 },
        .offset = 0,
    };
    const rhs: zgc.core.Tensor.ConstView(i8, 2) = .{
        .storage = &rhs_storage,
        .shape = .{ 2, 2 },
        .strides = .{ 2, 1 },
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(i8, 2) = .{
        .storage = &output_storage,
        .shape = .{ 2, 2 },
        .strides = .{ 2, 1 },
        .offset = 0,
    };

    const op: zgc.Op = .{ .compute = .sub };
    op.execute(.{ lhs, rhs }, output);

    try std.testing.expectEqualSlices(i8, &.{ 9, 18, 27, 36 }, &output_storage);
}

test "sub broadcasts a rank-zero scalar" {
    var input_storage = [_]f32{ 4, 5, 6, 7 };
    var scalar_storage = [_]f32{1.5};
    var output_storage: [4]f32 = undefined;
    const input: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &input_storage,
        .shape = .{ 2, 2 },
        .strides = .{ 2, 1 },
        .offset = 0,
    };
    const scalar: zgc.core.Tensor.ConstView(f32, 0) = .{
        .storage = &scalar_storage,
        .shape = .{},
        .strides = .{},
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &output_storage,
        .shape = .{ 2, 2 },
        .strides = .{ 2, 1 },
        .offset = 0,
    };

    const op: zgc.Op = .{ .compute = .sub };
    op.execute(.{ input, scalar }, output);

    try std.testing.expectEqualSlices(f32, &.{ 2.5, 3.5, 4.5, 5.5 }, &output_storage);
}
