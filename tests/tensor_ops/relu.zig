const std = @import("std");
const zgc = @import("zgc");

test "relu writes clamped values to its output view" {
    var input_data = [_]f32{ -3.5, -0.0, 0.0, 2.25, -1.0, 8.0 };
    var output_data: [input_data.len]f32 = @splat(std.math.nan(f32));

    const input: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &input_data,
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

    const op: zgc.Op = .{ .compute = .relu };
    op.execute(.{input}, output);

    try std.testing.expectEqualSlices(
        f32,
        &.{ 0.0, 0.0, 0.0, 2.25, 0.0, 8.0 },
        &output_data,
    );
}

test "relu preserves f16 dtype semantics" {
    var input_data = [_]f16{ -3.5, -0.0, 0.0, 2.25, -1.0, 8.0 };
    var output_data: [input_data.len]f16 = undefined;

    const input: zgc.core.Tensor.ConstView(f16, 1) = .{
        .storage = &input_data,
        .shape = .{input_data.len},
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
        std.debug.assert(@TypeOf(input).dtype == .f16);
        std.debug.assert(@TypeOf(input).scalar_type == f16);
        std.debug.assert(@TypeOf(input).rank == 1);
    }

    const op: zgc.Op = .{ .compute = .relu };
    op.execute(.{input}, output);

    try std.testing.expectEqualSlices(
        f16,
        &.{ 0.0, 0.0, 0.0, 2.25, 0.0, 8.0 },
        &output_data,
    );
}

test "relu preserves i8 dtype semantics" {
    var input_data = [_]i8{ -128, -3, 0, 2, 127 };
    var output_data: [input_data.len]i8 = undefined;

    const input: zgc.core.Tensor.ConstView(i8, 1) = .{
        .storage = &input_data,
        .shape = .{input_data.len},
        .strides = .{1},
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(i8, 1) = .{
        .storage = &output_data,
        .shape = .{output_data.len},
        .strides = .{1},
        .offset = 0,
    };

    comptime std.debug.assert(@TypeOf(input).dtype.kind() == .signed_integer);

    const op: zgc.Op = .{ .compute = .relu };
    op.execute(.{input}, output);

    try std.testing.expectEqualSlices(i8, &.{ 0, 0, 0, 2, 127 }, &output_data);
}

test "relu traverses a transposed input view" {
    var input_storage = [_]f32{ -1, 2, -3, 4, -5, 6 };
    var output_storage: [6]f32 = @splat(std.math.nan(f32));

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

    const op: zgc.Op = .{ .compute = .relu };
    op.execute(.{input}, output);

    try std.testing.expectEqualSlices(
        f32,
        &.{ 0, 4, 2, 0, 0, 6 },
        &output_storage,
    );
}

test "relu preserves a dense first-axis-contiguous layout" {
    var input_storage = [_]f32{ -1, 4, 2, -5, -3, 6 };
    var output_storage: [6]f32 = @splat(std.math.nan(f32));

    const input: zgc.core.Tensor.ConstView(f32, 2) = .{
        .storage = &input_storage,
        .shape = .{ 2, 3 },
        .strides = .{ 1, 2 },
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &output_storage,
        .shape = .{ 2, 3 },
        .strides = .{ 1, 2 },
        .offset = 0,
    };

    const op: zgc.Op = .{ .compute = .relu };
    op.execute(.{input}, output);

    try std.testing.expectEqualSlices(f32, &.{ 0, 4, 2, 0, 0, 6 }, &output_storage);
}
