const std = @import("std");
const zgc = @import("zgc");

test "unary floating-point math applies to static tensor views" {
    var input_storage = [_]f32{ 0.25, 1, 4, 9 };
    var negative_storage = [_]f32{ -0.25, -1, -4, -9 };
    var output_storage: [4]f32 = undefined;
    const input: zgc.core.Tensor.ConstView(f32, 1) = .{
        .storage = &input_storage,
        .shape = .{4},
        .strides = .{1},
        .offset = 0,
    };
    const negative: zgc.core.Tensor.ConstView(f32, 1) = .{
        .storage = &negative_storage,
        .shape = .{4},
        .strides = .{1},
        .offset = 0,
    };
    const output: zgc.core.Tensor.View(f32, 1) = .{
        .storage = &output_storage,
        .shape = .{4},
        .strides = .{1},
        .offset = 0,
    };

    (zgc.Op{ .compute = .neg }).execute(.{input}, output);
    try std.testing.expectEqualSlices(f32, &negative_storage, &output_storage);
    (zgc.Op{ .compute = .abs }).execute(.{negative}, output);
    try std.testing.expectEqualSlices(f32, &input_storage, &output_storage);
    (zgc.Op{ .compute = .sqrt }).execute(.{input}, output);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 1, 2, 3 }, &output_storage);
    (zgc.Op{ .compute = .log }).execute(.{input}, output);
    for (input_storage, output_storage) |value, actual| {
        try std.testing.expectApproxEqAbs(@log(value), actual, 1e-6);
    }
    (zgc.Op{ .compute = .reciprocal }).execute(.{input}, output);
    try std.testing.expectEqualSlices(f32, &.{ 4, 1, 0.25, 1.0 / 9.0 }, &output_storage);
}

test "minimum maximum and clamp broadcast tensor bounds" {
    var values_storage = [_]f32{ -2, 1, 8, 4, 6, 10 };
    var lower_storage = [_]f32{0};
    var upper_storage = [_]f32{ 3, 5, 9 };
    var output_storage: [6]f32 = undefined;
    const values: zgc.core.Tensor.ConstView(f32, 2) = .{ .storage = &values_storage, .shape = .{ 2, 3 }, .strides = .{ 3, 1 }, .offset = 0 };
    const lower: zgc.core.Tensor.ConstView(f32, 0) = .{ .storage = &lower_storage, .shape = .{}, .strides = .{}, .offset = 0 };
    const upper: zgc.core.Tensor.ConstView(f32, 1) = .{ .storage = &upper_storage, .shape = .{3}, .strides = .{1}, .offset = 0 };
    const output: zgc.core.Tensor.View(f32, 2) = .{ .storage = &output_storage, .shape = .{ 2, 3 }, .strides = .{ 3, 1 }, .offset = 0 };

    (zgc.Op{ .compute = .minimum }).execute(.{ values, upper }, output);
    try std.testing.expectEqualSlices(f32, &.{ -2, 1, 8, 3, 5, 9 }, &output_storage);
    (zgc.Op{ .compute = .maximum }).execute(.{ values, lower }, output);
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 8, 4, 6, 10 }, &output_storage);
    (zgc.Op{ .compute = .clamp }).execute(.{ values, lower, upper }, output);
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 8, 3, 5, 9 }, &output_storage);
}
