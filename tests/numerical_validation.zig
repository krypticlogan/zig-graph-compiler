const std = @import("std");
const zgc = @import("zgc_internal").api;

const Sources = enum(usize) { input, first_weights, first_bias, second_weights, second_bias };

const definition = blk: {
    var builder = zgc.DefinitionBuilder.init();
    const sources = builder.sources(Sources);
    const input = sources.input(.input, .f32, &.{ 3, 4 });
    const w1 = sources.parameter(.first_weights, .f32, &.{ 4, 5 });
    const b1 = sources.parameter(.first_bias, .f32, &.{5});
    const w2 = sources.parameter(.second_weights, .f32, &.{ 5, 2 });
    const b2 = sources.parameter(.second_bias, .f32, &.{2});

    const shifted = builder.shift(input, &.{ 1, 0 }, .wrap);
    const hidden = builder.relu(builder.add(builder.matmul(shifted, w1), b1));
    const values = builder.add(builder.matmul(hidden, w2), b2);
    builder.output(values);
    builder.output(builder.mean(values, .{ .axes = &.{1} }));
    break :blk builder.finish();
};

const Balanced = definition.executable();
const alternate_index = if (Balanced.selected_executable_candidate_index == 0) 1 else 0;
const Alternate = definition.executableWith(.{ .frontier_index = alternate_index });

const first_weights = [_]f32{
    0.5,   -1,    0.25, 1.5,  -0.5,
    -0.75, 0.5,   1,    0.25, 1.25,
    1,     -0.25, -0.5, 0.75, 0.5,
    0.25,  1,     -1,   0.5,  -0.25,
};
const first_bias = [_]f32{ 0.1, -0.3, 0.2, -0.4, 0.5 };
const second_weights = [_]f32{ 0.5, -1, -0.25, 0.75, 1.25, 0.5, -0.5, -0.25, 0.75, 1 };
const second_bias = [_]f32{ -0.2, 0.4 };
const inputs = [_][12]f32{
    .{ -1, 0.5, 2, -0.5, 3, -2, 1, 0.25, -1.5, 2.5, 0.75, 4 },
    .{ 2, -1, 0.25, 3, -0.5, 1.5, -2, 0.75, 4, -0.25, 1, -1.5 },
};

fn reference(input: []const f32) [3][2]f32 {
    var result: [3][2]f32 = undefined;
    for (0..3) |row| {
        const source_row = (row + 2) % 3;
        var hidden: [5]f32 = undefined;
        for (0..5) |column| {
            var value = first_bias[column];
            for (0..4) |inner| {
                value += input[source_row * 4 + inner] * first_weights[inner * 5 + column];
            }
            hidden[column] = @max(value, 0);
        }
        for (0..2) |column| {
            var value = second_bias[column];
            for (0..5) |inner| {
                value += hidden[inner] * second_weights[inner * 2 + column];
            }
            result[row][column] = value;
        }
    }
    return result;
}

fn validate(comptime Executable: type) !void {
    var runtime = Executable.Runtime.init();
    try runtime.copySource(.first_weights, &first_weights);
    try runtime.copySource(.first_bias, &first_bias);
    try runtime.copySource(.second_weights, &second_weights);
    try runtime.copySource(.second_bias, &second_bias);

    for (inputs) |input| {
        try runtime.copyInput(.input, &input);
        runtime.run();
        const expected = reference(&input);
        const values = runtime.outputView(0);
        const means = runtime.outputView(1);
        try std.testing.expectEqual([2]usize{ 3, 2 }, values.shape);
        try std.testing.expectEqual([1]usize{3}, means.shape);
        for (0..3) |row| {
            for (0..2) |column| {
                const actual = values.get(.{ row, column });
                try std.testing.expect(std.math.isFinite(actual));
                try std.testing.expectApproxEqAbs(expected[row][column], actual, 0.0002);
            }
            const expected_mean = (expected[row][0] + expected[row][1]) / 2;
            const actual_mean = means.get(.{row});
            try std.testing.expect(std.math.isFinite(actual_mean));
            try std.testing.expectApproxEqAbs(expected_mean, actual_mean, 0.0002);
        }
    }
}

test "compiled mixed graph matches an independent scalar reference across runs and plans" {
    try std.testing.expect(Balanced.executable_candidate_frontier.count > 1);
    try std.testing.expect(Balanced.selected_executable_candidate_index != Alternate.selected_executable_candidate_index);
    try validate(Balanced);
    try validate(Alternate);
}
