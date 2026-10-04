const std = @import("std");
const zgc = @import("zgc");

test "sum reduces either matrix axis" {
    var input_storage = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var columns_storage: [3]f32 = undefined;
    var rows_storage: [2]f32 = undefined;
    const input: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &input_storage,
        .shape = .{ 2, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };
    const columns: zgc.core.Tensor.View(f32, 1) = .{
        .storage = &columns_storage,
        .shape = .{3},
        .strides = .{1},
        .offset = 0,
    };
    const rows: zgc.core.Tensor.View(f32, 1) = .{
        .storage = &rows_storage,
        .shape = .{2},
        .strides = .{1},
        .offset = 0,
    };

    const sum_columns: zgc.Op = .{ .compute = .{ .sum = .{ .axes = 1 << 0 } } };
    const sum_rows: zgc.Op = .{ .compute = .{ .sum = .{ .axes = 1 << 1 } } };
    sum_columns.execute(.{input}, columns);
    sum_rows.execute(.{input}, rows);

    try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9 }, &columns_storage);
    try std.testing.expectEqualSlices(f32, &.{ 6, 15 }, &rows_storage);
}

test "sum traverses a strided reduction axis" {
    var input_storage = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var output_storage: [3]f32 = undefined;
    const input: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &input_storage,
        .shape = .{ 3, 2 },
        .strides = .{ 1, 3 },
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 1) = .{
        .storage = &output_storage,
        .shape = .{3},
        .strides = .{1},
        .offset = 0,
    };

    const op: zgc.Op = .{ .compute = .{ .sum = .{ .axes = 1 << 1 } } };
    op.execute(.{input}, output);

    try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9 }, &output_storage);
}

test "sum reduces a vector to a rank-zero view" {
    var input_storage = [_]i8{ 1, 2, 3, 4 };
    var output_storage: [1]i8 = undefined;
    const input: zgc.core.Tensor.ConstView(i8, 1) = .{
        .storage = &input_storage,
        .shape = .{4},
        .strides = .{1},
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(i8, 0) = .{
        .storage = &output_storage,
        .shape = .{},
        .strides = .{},
        .offset = 0,
    };

    const op: zgc.Op = .{ .compute = .{ .sum = .{ .axes = 1 << 0 } } };
    op.execute(.{input}, output);

    try std.testing.expectEqual(@as(i8, 10), output_storage[0]);
}

test "sum vectorizes a unit-stride reduction axis and handles its tail" {
    const vector_len = std.simd.suggestVectorLength(f32) orelse 1;
    const len = vector_len + 1;
    var input_storage: [len]f32 = @splat(1);
    var output_storage: [1]f32 = undefined;
    const input: zgc.core.Tensor.ConstView(f32, 1) = .{
        .storage = &input_storage,
        .shape = .{len},
        .strides = .{1},
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 0) = .{
        .storage = &output_storage,
        .shape = .{},
        .strides = .{},
        .offset = 0,
    };

    const op: zgc.Op = .{ .compute = .{ .sum = .{ .axes = 1 << 0 } } };
    op.execute(.{input}, output);

    try std.testing.expectEqual(@as(f32, @floatFromInt(len)), output_storage[0]);
}
