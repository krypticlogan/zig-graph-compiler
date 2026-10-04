const std = @import("std");
const zgc = @import("zgc");

test "matmul multiplies contiguous row-major rank-2 tensors" {
    var lhs_data = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    var rhs_data = [_]f32{
        7,  8,
        9,  10,
        11, 12,
    };
    var output_data: [4]f32 = @splat(std.math.nan(f32));

    const lhs: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &lhs_data,
        .shape = .{ 2, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };
    const rhs: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &rhs_data,
        .shape = .{ 3, 2 },
        .strides = .{ 2, 1 },
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &output_data,
        .shape = .{ 2, 2 },
        .strides = .{ 2, 1 },
        .offset = 0,
    };

    (zgc.Op{ .compute = .matmul }).execute(.{ lhs, rhs }, output);

    try std.testing.expectEqualSlices(
        f32,
        &.{ 58, 64, 139, 154 },
        &output_data,
    );
}

test "matmul handles a native SIMD chunk followed by a column tail" {
    const vector_len = std.simd.suggestVectorLength(f32) orelse 1;
    const m = 2;
    const k_len = 3;
    const n = vector_len + 1;

    var lhs_data = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    var rhs_data: [k_len * n]f32 = undefined;
    var output_data: [m * n]f32 = @splat(std.math.nan(f32));
    var expected: [m * n]f32 = undefined;

    for (&rhs_data, 0..) |*value, index| {
        value.* = @floatFromInt(index + 1);
    }

    for (0..m) |row| {
        for (0..n) |col| {
            var accumulator: f32 = 0;
            for (0..k_len) |k| {
                accumulator += lhs_data[row * k_len + k] * rhs_data[k * n + col];
            }
            expected[row * n + col] = accumulator;
        }
    }

    const lhs: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &lhs_data,
        .shape = .{ m, k_len },
        .strides = .{ k_len, 1 },
        .offset = 0,
    };
    const rhs: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &rhs_data,
        .shape = .{ k_len, n },
        .strides = .{ n, 1 },
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &output_data,
        .shape = .{ m, n },
        .strides = .{ n, 1 },
        .offset = 0,
    };

    (zgc.Op{ .compute = .matmul }).execute(.{ lhs, rhs }, output);

    try std.testing.expectEqualSlices(f32, &expected, &output_data);
}

test "matmul supports strided inputs and output" {
    var lhs_storage = [_]f32{ 1, 4, 2, 5, 3, 6 };
    var rhs_storage = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var output_storage: [4]f32 = @splat(std.math.nan(f32));

    const lhs: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &lhs_storage,
        .shape = .{ 2, 3 },
        .strides = .{ 1, 2 },
        .offset = 0,
    };
    const rhs: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &rhs_storage,
        .shape = .{ 3, 2 },
        .strides = .{ 2, 1 },
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &output_storage,
        .shape = .{ 2, 2 },
        .strides = .{ 1, 2 },
        .offset = 0,
    };

    (zgc.Op{ .compute = .matmul }).execute(.{ lhs, rhs }, output);

    try std.testing.expectEqualSlices(
        f32,
        &.{ 58, 139, 64, 154 },
        &output_storage,
    );
}

test "matmul vectorizes output columns with a transposed lhs" {
    const vector_len = std.simd.suggestVectorLength(f32) orelse 1;
    const m = 2;
    const k_len = 3;
    const n = vector_len + 1;

    var lhs_storage = [_]f32{ 1, 4, 2, 5, 3, 6 };
    var rhs_storage: [k_len * n]f32 = undefined;
    var output_storage: [m * n]f32 = @splat(std.math.nan(f32));
    var expected: [m * n]f32 = undefined;

    for (&rhs_storage, 0..) |*value, index| {
        value.* = @floatFromInt(index + 1);
    }
    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f32 = 0;
            for (0..k_len) |k| {
                sum += lhs_storage[k * m + row] * rhs_storage[k * n + col];
            }
            expected[row * n + col] = sum;
        }
    }

    const lhs: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &lhs_storage,
        .shape = .{ m, k_len },
        .strides = .{ 1, m },
        .offset = 0,
    };
    const rhs: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &rhs_storage,
        .shape = .{ k_len, n },
        .strides = .{ n, 1 },
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &output_storage,
        .shape = .{ m, n },
        .strides = .{ n, 1 },
        .offset = 0,
    };

    (zgc.Op{ .compute = .matmul }).execute(.{ lhs, rhs }, output);

    try std.testing.expectEqualSlices(f32, &expected, &output_storage);
}

test "matmul vectorizes the contracted axis for a transposed rhs" {
    const vector_len = std.simd.suggestVectorLength(f32) orelse 1;
    const m = 2;
    const k_len = vector_len + 1;
    const n = 2;

    var lhs_storage: [m * k_len]f32 = undefined;
    var rhs_storage: [n * k_len]f32 = undefined;
    var output_storage: [m * n]f32 = @splat(std.math.nan(f32));
    var expected: [m * n]f32 = undefined;

    for (&lhs_storage, 0..) |*value, index| {
        value.* = @floatFromInt(index + 1);
    }
    for (&rhs_storage, 0..) |*value, index| {
        value.* = @floatFromInt(index + 1);
    }
    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f32 = 0;
            for (0..k_len) |k| {
                sum += lhs_storage[row * k_len + k] *
                    rhs_storage[col * k_len + k];
            }
            expected[row * n + col] = sum;
        }
    }

    const lhs: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &lhs_storage,
        .shape = .{ m, k_len },
        .strides = .{ k_len, 1 },
        .offset = 0,
    };
    const rhs: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &rhs_storage,
        .shape = .{ k_len, n },
        .strides = .{ 1, k_len },
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &output_storage,
        .shape = .{ m, n },
        .strides = .{ n, 1 },
        .offset = 0,
    };

    (zgc.Op{ .compute = .matmul }).execute(.{ lhs, rhs }, output);

    try std.testing.expectEqualSlices(f32, &expected, &output_storage);
}

test "matmul vectorizes output rows for column-major lhs and output" {
    const vector_len = std.simd.suggestVectorLength(f32) orelse 1;
    const m = vector_len + 1;
    const k_len = 2;
    const n = 2;

    var lhs_storage: [k_len * m]f32 = undefined;
    var rhs_storage: [k_len * n]f32 = undefined;
    var output_storage: [n * m]f32 = @splat(std.math.nan(f32));
    var expected: [n * m]f32 = undefined;

    for (&lhs_storage, 0..) |*value, index| {
        value.* = @floatFromInt(index + 1);
    }
    for (&rhs_storage, 0..) |*value, index| {
        value.* = @floatFromInt(index + 1);
    }
    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f32 = 0;
            for (0..k_len) |k| {
                sum += lhs_storage[k * m + row] * rhs_storage[k * n + col];
            }
            expected[col * m + row] = sum;
        }
    }

    const lhs: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &lhs_storage,
        .shape = .{ m, k_len },
        .strides = .{ 1, m },
        .offset = 0,
    };
    const rhs: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &rhs_storage,
        .shape = .{ k_len, n },
        .strides = .{ n, 1 },
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &output_storage,
        .shape = .{ m, n },
        .strides = .{ 1, m },
        .offset = 0,
    };

    (zgc.Op{ .compute = .matmul }).execute(.{ lhs, rhs }, output);

    try std.testing.expectEqualSlices(f32, &expected, &output_storage);
}

test "matmul retains a scalar fallback for incompatible strides" {
    var lhs_storage = [_]f32{ 1, 99, 2, 3, 99, 4 };
    var rhs_storage = [_]f32{ 5, 99, 6, 99, 7, 99, 8 };
    var output_storage: [6]f32 = @splat(99);

    const lhs: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &lhs_storage,
        .shape = .{ 2, 2 },
        .strides = .{ 3, 2 },
        .offset = 0,
    };
    const rhs: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &rhs_storage,
        .shape = .{ 2, 2 },
        .strides = .{ 4, 2 },
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &output_storage,
        .shape = .{ 2, 2 },
        .strides = .{ 3, 2 },
        .offset = 0,
    };

    (zgc.Op{ .compute = .matmul }).execute(.{ lhs, rhs }, output);

    try std.testing.expectEqualSlices(
        f32,
        &.{ 19, 99, 22, 43, 99, 50 },
        &output_storage,
    );
}
