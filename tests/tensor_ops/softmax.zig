const std = @import("std");
const zgc = @import("zgc");

test "softmax is stable and normalizes along its axis" {
    var input_storage = [_]f32{ 1000, 1001, 1002, -1000, -1001, -1002 };
    var output_storage: [6]f32 = undefined;
    const input: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &input_storage,
        .shape = .{ 2, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &output_storage,
        .shape = .{ 2, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };

    const op: zgc.Op = .{ .compute = .{ .softmax = .{ .axis = 1 } } };
    op.execute(.{input}, output);

    for (output_storage) |value| try std.testing.expect(std.math.isFinite(value));
    try std.testing.expectApproxEqAbs(@as(f32, 1), output_storage[0] + output_storage[1] + output_storage[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), output_storage[3] + output_storage[4] + output_storage[5], 1e-6);
    try std.testing.expect(output_storage[0] < output_storage[1]);
    try std.testing.expect(output_storage[1] < output_storage[2]);
    try std.testing.expect(output_storage[3] > output_storage[4]);
    try std.testing.expect(output_storage[4] > output_storage[5]);
}

test "softmax traverses a strided axis" {
    var input_storage = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var output_storage: [6]f32 = undefined;
    const input: zgc.core.Tensor.ConstView(f32, 2) = .{
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

    const op: zgc.Op = .{ .compute = .{ .softmax = .{ .axis = 1 } } };
    op.execute(.{input}, output);

    for (0..3) |row| {
        const row_start = row * 2;
        try std.testing.expectApproxEqAbs(
            @as(f32, 1),
            output_storage[row_start] + output_storage[row_start + 1],
            1e-6,
        );
        try std.testing.expect(output_storage[row_start] < output_storage[row_start + 1]);
    }
}

test "softmax vectorizes a unit-stride axis and handles its tail" {
    const vector_len = std.simd.suggestVectorLength(f32) orelse 1;
    const len = vector_len + 1;
    var input_storage: [len]f32 = @splat(0);
    var output_storage: [len]f32 = undefined;
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

    const op: zgc.Op = .{ .compute = .{ .softmax = .{ .axis = 0 } } };
    op.execute(.{input}, output);

    const expected = 1 / @as(f32, @floatFromInt(len));
    var total: f32 = 0;
    for (output_storage) |value| {
        try std.testing.expectApproxEqAbs(expected, value, 1e-6);
        total += value;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1), total, 1e-6);
}
