const std = @import("std");
const zgc = @import("zgc");

test "multi-axis reductions preserve unselected dimensions" {
    var input_storage = [_]f32{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    };
    var sum_storage: [3]f32 = undefined;
    var mean_storage: [3]f32 = undefined;
    var min_storage: [3]f32 = undefined;
    var max_storage: [3]f32 = undefined;
    const input: zgc.core.Tensor.ConstView(f32, 3) = .{
        .storage = &input_storage,
        .shape = .{ 2, 3, 2 },
        .strides = .{ 1, 4, 2 },
        .offset = 0,
    };
    const axes = (@as(u64, 1) << 0) | (@as(u64, 1) << 2);
    const attrs: zgc.Op.Compute.ReductionAttrs = .{ .axes = axes };

    inline for (.{
        .{ .op = zgc.Op{ .compute = .{ .sum = attrs } }, .storage = &sum_storage },
        .{ .op = zgc.Op{ .compute = .{ .mean = attrs } }, .storage = &mean_storage },
        .{ .op = zgc.Op{ .compute = .{ .min = attrs } }, .storage = &min_storage },
        .{ .op = zgc.Op{ .compute = .{ .max = attrs } }, .storage = &max_storage },
    }) |case| {
        const output: zgc.core.Tensor.View(f32, 1) = .{
            .storage = case.storage,
            .shape = .{3},
            .strides = .{1},
            .offset = 0,
        };
        case.op.execute(.{input}, output);
    }

    try std.testing.expectEqualSlices(f32, &.{ 10, 26, 42 }, &sum_storage);
    try std.testing.expectEqualSlices(f32, &.{ 2.5, 6.5, 10.5 }, &mean_storage);
    try std.testing.expectEqualSlices(f32, &.{ 1, 5, 9 }, &min_storage);
    try std.testing.expectEqualSlices(f32, &.{ 4, 8, 12 }, &max_storage);
}

test "keep_dims retains reduced axes as singleton dimensions" {
    var input_storage = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var output_storage: [2]f32 = undefined;
    const input: zgc.core.Tensor.ConstView(f32, 3) = .{
        .storage = &input_storage,
        .shape = .{ 2, 1, 3 },
        .strides = .{ 3, 3, 1 },
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 3) = .{
        .storage = &output_storage,
        .shape = .{ 2, 1, 1 },
        .strides = .{ 1, 1, 1 },
        .offset = 0,
    };
    const op: zgc.Op = .{ .compute = .{ .sum = .{
        .axes = (@as(u64, 1) << 1) | (@as(u64, 1) << 2),
        .keep_dims = true,
    } } };
    op.execute(.{input}, output);
    try std.testing.expectEqualSlices(f32, &.{ 6, 15 }, &output_storage);
}

test "integer min and max reductions use finite dtype identities" {
    var input_storage = [_]i8{ -7, 2, 12, -3 };
    var min_storage: [1]i8 = undefined;
    var max_storage: [1]i8 = undefined;
    const input: zgc.core.Tensor.ConstView(i8, 1) = .{
        .storage = &input_storage,
        .shape = .{4},
        .strides = .{1},
        .offset = 0,
    };
    const min_output: zgc.core.Tensor.View(i8, 0) = .{
        .storage = &min_storage,
        .shape = .{},
        .strides = .{},
        .offset = 0,
    };
    const max_output: zgc.core.Tensor.View(i8, 0) = .{
        .storage = &max_storage,
        .shape = .{},
        .strides = .{},
        .offset = 0,
    };
    const attrs: zgc.Op.Compute.ReductionAttrs = .{ .axes = 1 };
    const min_op: zgc.Op = .{ .compute = .{ .min = attrs } };
    const max_op: zgc.Op = .{ .compute = .{ .max = attrs } };
    min_op.execute(.{input}, min_output);
    max_op.execute(.{input}, max_output);

    try std.testing.expectEqual(@as(i8, -7), min_storage[0]);
    try std.testing.expectEqual(@as(i8, 12), max_storage[0]);
}

test "mean min and max vectorize contiguous single-axis reductions with tails" {
    const vector_width = std.simd.suggestVectorLength(f32) orelse 1;
    const len = vector_width + 1;
    var input_storage: [len]f32 = undefined;
    for (&input_storage, 0..) |*value, index| value.* = @floatFromInt(index + 1);
    input_storage[len - 1] = -5;
    var mean_storage: [1]f32 = undefined;
    var min_storage: [1]f32 = undefined;
    var max_storage: [1]f32 = undefined;
    const input: zgc.core.Tensor.ConstView(f32, 1) = .{
        .storage = &input_storage,
        .shape = .{len},
        .strides = .{1},
        .offset = 0,
    };
    const attrs: zgc.Op.Compute.ReductionAttrs = .{ .axes = 1 };

    inline for (.{
        .{ .op = zgc.Op{ .compute = .{ .mean = attrs } }, .storage = &mean_storage },
        .{ .op = zgc.Op{ .compute = .{ .min = attrs } }, .storage = &min_storage },
        .{ .op = zgc.Op{ .compute = .{ .max = attrs } }, .storage = &max_storage },
    }) |case| {
        const output: zgc.core.Tensor.View(f32, 0) = .{
            .storage = case.storage,
            .shape = .{},
            .strides = .{},
            .offset = 0,
        };
        case.op.execute(.{input}, output);
    }

    const expected_mean = (@as(f32, @floatFromInt((len - 1) * len / 2)) - 5) /
        @as(f32, @floatFromInt(len));
    try std.testing.expectApproxEqAbs(expected_mean, mean_storage[0], 0.0001);
    try std.testing.expectEqual(@as(f32, -5), min_storage[0]);
    try std.testing.expectEqual(@as(f32, @floatFromInt(len - 1)), max_storage[0]);
}
