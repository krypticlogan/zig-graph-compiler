const std = @import("std");
const zgc = @import("zgc");

test "add writes elementwise sums to its output view" {
    var a_data = [_]f32{ -3.5, -1.75, 1.0, 1.25, -1.0, 4.5 };
    var b_data = [_]f32{ 6.5, 3.0, 5.25, 2.25, -1.0, 3.5 };
    var output_data: [a_data.len]f32 = @splat(std.math.nan(f32));

    const a: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &a_data,
        .shape = .{ 2, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };
    const b: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &b_data,
        .shape = .{ 2, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &output_data,
        .shape = .{ 2, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };

    const op: zgc.Op = .{ .compute = .add };
    op.execute(.{ a, b }, output);

    try std.testing.expectEqualSlices(
        f32,
        &.{ 3.0, 1.25, 6.25, 3.5, -2.0, 8.0 },
        &output_data,
    );
}

test "add preserves f16 dtype semantics" {
    var a_data = [_]f16{ -3.5, -1.0, 0.0, 2.25, 4.0, 8.0 };
    var b_data = [_]f16{ 1.5, -2.0, 0.5, 1.75, -1.0, 8.0 };
    var output_data: [a_data.len]f16 = undefined;

    const a: zgc.core.Tensor.ConstView(f16, 1) = .{
        .storage = &a_data,
        .shape = .{a_data.len},
        .strides = .{1},
        .offset = 0,
    };
    const b: zgc.core.Tensor.ConstView(f16, 1) = .{
        .storage = &b_data,
        .shape = .{b_data.len},
        .strides = .{1},
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f16, 1) = .{
        .storage = &output_data,
        .shape = .{output_data.len},
        .strides = .{1},
        .offset = 0,
    };

    comptime {
        std.debug.assert(@TypeOf(a).dtype == .f16);
        std.debug.assert(@TypeOf(a).scalar_type == f16);
        std.debug.assert(@TypeOf(a).rank == 1);
    }

    const op: zgc.Op = .{ .compute = .add };
    op.execute(.{ a, b }, output);

    try std.testing.expectEqualSlices(
        f16,
        &.{ -2.0, -3.0, 0.5, 4.0, 3.0, 16.0 },
        &output_data,
    );
}

test "add preserves i8 dtype semantics without overflow" {
    var a_data = [_]i8{ -100, -3, 0, 2, 100 };
    var b_data = [_]i8{ 20, -4, 0, 5, 27 };
    var output_data: [a_data.len]i8 = undefined;

    const a: zgc.core.Tensor.ConstView(i8, 1) = .{
        .storage = &a_data,
        .shape = .{a_data.len},
        .strides = .{1},
        .offset = 0,
    };
    const b: zgc.core.Tensor.ConstView(i8, 1) = .{
        .storage = &b_data,
        .shape = .{b_data.len},
        .strides = .{1},
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(i8, 1) = .{
        .storage = &output_data,
        .shape = .{output_data.len},
        .strides = .{1},
        .offset = 0,
    };

    comptime std.debug.assert(@TypeOf(a).dtype.kind() == .signed_integer);

    const op: zgc.Op = .{ .compute = .add };
    op.execute(.{ a, b }, output);

    try std.testing.expectEqualSlices(i8, &.{ -80, -7, 0, 7, 127 }, &output_data);
}

test "add traverses independently strided inputs and output" {
    var a_storage = [_]f32{ 1, 4, 2, 5, 3, 6 };
    var b_storage = [_]f32{ 10, 20, 30, 40, 50, 60 };
    var output_storage: [6]f32 = @splat(std.math.nan(f32));

    const a: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &a_storage,
        .shape = .{ 2, 3 },
        .strides = .{ 1, 2 },
        .offset = 0,
    };
    const b: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &b_storage,
        .shape = .{ 2, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &output_storage,
        .shape = .{ 2, 3 },
        .strides = .{ 1, 2 },
        .offset = 0,
    };

    const op: zgc.Op = .{ .compute = .add };
    op.execute(.{ a, b }, output);

    try std.testing.expectEqualSlices(
        f32,
        &.{ 11, 44, 22, 55, 33, 66 },
        &output_storage,
    );
}

test "add broadcasts a trailing vector across a matrix" {
    var matrix_storage = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var bias_storage = [_]f32{ 10, 20, 30 };
    var output_storage: [6]f32 = undefined;
    const matrix: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &matrix_storage,
        .shape = .{ 2, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };
    const bias: zgc.core.Tensor.ConstView(f32, 1) = .{
        .storage = &bias_storage,
        .shape = .{3},
        .strides = .{1},
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &output_storage,
        .shape = .{ 2, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };

    const op: zgc.Op = .{ .compute = .add };
    op.execute(.{ matrix, bias }, output);

    try std.testing.expectEqualSlices(
        f32,
        &.{ 11, 22, 33, 14, 25, 36 },
        &output_storage,
    );
}

test "add broadcasts a trailing vector across a first-axis-contiguous matrix" {
    var matrix_storage = [_]f32{ 1, 4, 2, 5, 3, 6 };
    var bias_storage = [_]f32{ 10, 20, 30 };
    var output_storage: [6]f32 = undefined;
    const matrix: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &matrix_storage,
        .shape = .{ 2, 3 },
        .strides = .{ 1, 2 },
        .offset = 0,
    };
    const bias: zgc.core.Tensor.ConstView(f32, 1) = .{
        .storage = &bias_storage,
        .shape = .{3},
        .strides = .{1},
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &output_storage,
        .shape = .{ 2, 3 },
        .strides = .{ 1, 2 },
        .offset = 0,
    };

    const op: zgc.Op = .{ .compute = .add };
    op.execute(.{ matrix, bias }, output);

    try std.testing.expectEqualSlices(
        f32,
        &.{ 11, 14, 22, 25, 33, 36 },
        &output_storage,
    );
}

test "add broadcasts singleton axes from both operands" {
    var lhs_storage = [_]i8{ 1, 2 };
    var rhs_storage = [_]i8{ 10, 20, 30 };
    var output_storage: [6]i8 = undefined;
    const lhs: zgc.core.Tensor.ConstView(i8, 2) = .{
        .storage = &lhs_storage,
        .shape = .{ 2, 1 },
        .strides = .{ 1, 1 },
        .offset = 0,
    };
    const rhs: zgc.core.Tensor.ConstView(i8, 2) = .{
        .storage = &rhs_storage,
        .shape = .{ 1, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(i8, 2) = .{
        .storage = &output_storage,
        .shape = .{ 2, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };

    const op: zgc.Op = .{ .compute = .add };
    op.execute(.{ lhs, rhs }, output);

    try std.testing.expectEqualSlices(i8, &.{ 11, 21, 31, 12, 22, 32 }, &output_storage);
}
