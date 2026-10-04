const std = @import("std");
const zgc = @import("zgc");

test "exp applies to contiguous floating-point views" {
    const vector_len = std.simd.suggestVectorLength(f32) orelse 1;
    const len = vector_len + 1;
    var input_storage: [len]f32 = undefined;
    var output_storage: [len]f32 = undefined;
    for (&input_storage, 0..) |*value, index| {
        value.* = @as(f32, @floatFromInt(index)) / 10 - 1;
    }
    const input: zgc.core.Tensor.ConstView(f32, 1) = .{
        .storage = &input_storage,
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

    const op: zgc.Op = .{ .compute = .exp };
    op.execute(.{input}, output);

    for (input_storage, output_storage) |value, actual| {
        try std.testing.expectApproxEqAbs(@exp(value), actual, 1e-6);
    }
}

test "exp traverses strided f16 views" {
    var input_storage = [_]f16{ 0, 2, 1, 3 };
    var output_storage: [4]f16 = undefined;
    const input: zgc.core.Tensor.ConstView(f16, 2) = .{
        .storage = &input_storage,
        .shape = .{ 2, 2 },
        .strides = .{ 1, 2 },
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f16, 2) = .{
        .storage = &output_storage,
        .shape = .{ 2, 2 },
        .strides = .{ 2, 1 },
        .offset = 0,
    };

    const op: zgc.Op = .{ .compute = .exp };
    op.execute(.{input}, output);

    const expected = [_]f16{ @exp(@as(f16, 0)), @exp(@as(f16, 1)), @exp(@as(f16, 2)), @exp(@as(f16, 3)) };
    for (expected, output_storage) |expected_value, actual| {
        try std.testing.expectApproxEqAbs(expected_value, actual, 0.01);
    }
}
