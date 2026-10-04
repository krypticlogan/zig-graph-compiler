const std = @import("std");
const zgc = @import("zgc");

test "comparisons produce boolean tensors with trailing-axis broadcasting" {
    var values_storage = [_]i8{ 1, 2, 3, 4, 5, 6 };
    var threshold_storage = [_]i8{ 1, 3, 5 };
    var output_storage: [6]bool = undefined;
    const values: zgc.core.Tensor.ConstView(i8, 2) = .{ .storage = &values_storage, .shape = .{ 2, 3 }, .strides = .{ 3, 1 }, .offset = 0 };
    const threshold: zgc.core.Tensor.ConstView(i8, 1) = .{ .storage = &threshold_storage, .shape = .{3}, .strides = .{1}, .offset = 0 };
    const output: zgc.core.Tensor.View(bool, 2) = .{ .storage = &output_storage, .shape = .{ 2, 3 }, .strides = .{ 3, 1 }, .offset = 0 };

    inline for (.{
        .{ .op = zgc.Op{ .compute = .equal }, .expected = [6]bool{ true, false, false, false, false, false } },
        .{ .op = zgc.Op{ .compute = .not_equal }, .expected = [6]bool{ false, true, true, true, true, true } },
        .{ .op = zgc.Op{ .compute = .less_than }, .expected = [6]bool{ false, true, true, false, false, false } },
        .{ .op = zgc.Op{ .compute = .less_equal }, .expected = [6]bool{ true, true, true, false, false, false } },
        .{ .op = zgc.Op{ .compute = .greater_than }, .expected = [6]bool{ false, false, false, true, true, true } },
        .{ .op = zgc.Op{ .compute = .greater_equal }, .expected = [6]bool{ true, false, false, true, true, true } },
    }) |case| {
        case.op.execute(.{ values, threshold }, output);
        inline for (case.expected, output_storage) |expected, actual| {
            try std.testing.expectEqual(expected, actual);
        }
    }
}

test "logical operations and where require explicit boolean masks" {
    var row_mask_storage = [_]bool{ true, false };
    var column_mask_storage = [_]bool{ true, false, true };
    var mask_storage: [6]bool = undefined;
    var values_storage = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var fallback_storage = [_]f32{-1};
    var output_storage: [6]f32 = undefined;
    const row_mask: zgc.core.Tensor.ConstView(bool, 2) = .{ .storage = &row_mask_storage, .shape = .{ 2, 1 }, .strides = .{ 1, 1 }, .offset = 0 };
    const column_mask: zgc.core.Tensor.ConstView(bool, 2) = .{ .storage = &column_mask_storage, .shape = .{ 1, 3 }, .strides = .{ 3, 1 }, .offset = 0 };
    const mask: zgc.core.Tensor.View(bool, 2) = .{ .storage = &mask_storage, .shape = .{ 2, 3 }, .strides = .{ 3, 1 }, .offset = 0 };

    (zgc.Op{ .compute = .logical_and }).execute(.{ row_mask, column_mask }, mask);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false, false, false }, &mask_storage);
    (zgc.Op{ .compute = .logical_or }).execute(.{ row_mask, column_mask }, mask);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false, true }, &mask_storage);

    const condition: zgc.core.Tensor.ConstView(bool, 2) = .{ .storage = &mask_storage, .shape = .{ 2, 3 }, .strides = .{ 3, 1 }, .offset = 0 };
    const values: zgc.core.Tensor.ConstView(f32, 2) = .{ .storage = &values_storage, .shape = .{ 2, 3 }, .strides = .{ 3, 1 }, .offset = 0 };
    const fallback: zgc.core.Tensor.ConstView(f32, 0) = .{ .storage = &fallback_storage, .shape = .{}, .strides = .{}, .offset = 0 };
    const output: zgc.core.Tensor.View(f32, 2) = .{ .storage = &output_storage, .shape = .{ 2, 3 }, .strides = .{ 3, 1 }, .offset = 0 };
    (zgc.Op{ .compute = .where }).execute(.{ condition, values, fallback }, output);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, -1, 6 }, &output_storage);

    (zgc.Op{ .compute = .logical_not }).execute(.{condition}, mask);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, true, false }, &mask_storage);
}
