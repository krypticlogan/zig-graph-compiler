const std = @import("std");
const models = @import("fixtures/models.zig");

test "model executes a compiled graph and exposes static output geometry" {
    var model = models.BasicExecutable.Runtime.init();
    const input = [_]f32{ -1, 2, -3, 4, -5, 6 };
    try model.copyInput(.input, &input);
    model.run();

    const output = model.outputView(0);
    try std.testing.expectEqual([2]usize{ 3, 2 }, output.shape);
    try std.testing.expectEqual([2]isize{ 2, 1 }, output.strides);
    try std.testing.expectEqualSlices(f32, &.{ 0, 4, 2, 0, 0, 6 }, output.storage);
    try std.testing.expectError(error.SourceSizeMismatch, model.copyInput(.input, &[_]f32{1}));
}

test "filled tensors alias one embedded scalar without model memory" {
    var model = models.FullExecutable.Runtime.init();
    model.run();

    try std.testing.expectEqual(@as(usize, 0), models.FullExecutable.memory_plan.byte_count);
    const output = model.outputView(0);
    try std.testing.expectEqual([2]usize{ 2, 3 }, output.shape);
    try std.testing.expectEqual([2]isize{ 0, 0 }, output.strides);
    for (0..2) |row| {
        for (0..3) |column| {
            try std.testing.expectEqual(@as(f32, 7.5), output.get(.{ row, column }));
        }
    }
}

test "embedded parameters support copied and borrowed runtime inputs" {
    const expected = [_]f32{ 3, 1, 3.5, 7 };

    var copied = models.EmbeddedParameterExecutable.Runtime.init();
    try copied.copyInput(.input, &[_]f32{ 1, 2, 3, 4 });
    copied.run();
    try std.testing.expectEqualSlices(f32, &expected, copied.outputView(0).storage);

    var borrowed = models.BoundInputExecutable.Runtime.init();
    var input = [_]f32{ 1, 2, 3, 4 };
    try borrowed.bindInput(.input, &input);
    borrowed.run();
    try std.testing.expectEqualSlices(f32, &expected, borrowed.outputView(0).storage);
    input[0] = 10;
    borrowed.run();
    try std.testing.expectEqual(@as(f32, 12), borrowed.outputView(0).storage[0]);
}

test "one lowered matmul definition supports copied and embedded storage" {
    var input: [models.matmul_batch * 3]f32 = undefined;
    for (0..models.matmul_batch) |row| {
        input[row * 3 ..][0..3].* = .{ @floatFromInt(row + 1), 2, -1 };
    }
    const weights = [_]f32{ 1, 2, 3, 4, 5, 6 };

    var copied = models.MatmulExecutable.Runtime.init();
    try copied.copyInput(.input, &input);
    try copied.copySource(.weights, &weights);
    copied.run();

    var embedded = models.EmbeddedMatmulExecutable.Runtime.init();
    try embedded.copyInput(.input, &input);
    embedded.run();

    var packed_model = models.PackedMatmulExecutable.Runtime.init();
    try packed_model.copyInput(.input, &input);
    packed_model.run();

    try std.testing.expectEqual(
        [2]isize{ 1, models.matmul_batch },
        models.MatmulExecutable.Runtime.sourceLayout(.input).strides[0..2].*,
    );
    for (0..models.matmul_batch) |row| {
        const expected = [2]f32{ @floatFromInt(row + 2), @floatFromInt(2 * row + 4) };
        inline for (.{ &copied, &embedded, &packed_model }) |current| {
            try std.testing.expectEqual(expected[0], current.outputView(0).get(.{ row, 0 }));
            try std.testing.expectEqual(expected[1], current.outputView(0).get(.{ row, 1 }));
        }
    }
}
