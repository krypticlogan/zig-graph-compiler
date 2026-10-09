const std = @import("std");
const zgc = @import("zgc");

const Sources = enum(usize) { input, fill };
const Definition = zgc.DefinitionBuilder;

const boundary_model = model: {
    var b = Definition.init();
    const b_sources = b.sources(Sources);
    const input = b_sources.input(.input, .f32, &.{ 2, 3 });
    const fill = b_sources.input(.fill, .f32, &.{});
    b.output(b.shift(input, &.{ 1, -1 }, .wrap));
    b.output(b.shift(input, &.{ 1, -1 }, .edge));
    b.output(b.shift(input, &.{ 1, -1 }, .reflect));
    b.output(b.shift(input, &.{ 1, -1 }, .{ .constant = fill }));
    b.output(b.sliceLoop(input, .{
        .axis = 0,
        .iterations = &.{
            .{ .offsets = &.{1}, .boundary = .{ .redirect = 1 } },
            .{ .offsets = &.{-1}, .boundary = .{ .redirect = 0 } },
        },
    }));
    break :model b.finish().model();
};

test "shift maps positive and negative offsets across every boundary mode" {
    var model = boundary_model.init();
    try model.copyInput(.input, &.{ 1, 2, 3, 4, 5, 6 });
    try model.copyInput(.fill, &.{-9});
    model.run();
    try std.testing.expectEqualSlices(f32, &.{ 5, 6, 4, 2, 3, 1 }, model.outputView(0).contiguousSlice().?);
    try std.testing.expectEqualSlices(f32, &.{ 2, 3, 3, 2, 3, 3 }, model.outputView(1).contiguousSlice().?);
    try std.testing.expectEqualSlices(f32, &.{ 5, 6, 5, 2, 3, 2 }, model.outputView(2).contiguousSlice().?);
    try std.testing.expectEqualSlices(f32, &.{ -9, -9, -9, 2, 3, -9 }, model.outputView(3).contiguousSlice().?);
    try std.testing.expectEqualSlices(f32, &.{ 4, 1, 2, 5, 6, 3 }, model.outputView(4).contiguousSlice().?);

    try model.copyInput(.fill, &.{7});
    model.run();
    try std.testing.expectEqualSlices(f32, &.{ 7, 7, 7, 2, 3, 7 }, model.outputView(3).contiguousSlice().?);
}

const LoopDefinitionSources = enum(usize) { input };
const LoopDefinition = zgc.DefinitionBuilder;

const loop_model = model: {
    var b = LoopDefinition.init();
    const b_sources = b.sources(LoopDefinitionSources);
    const input = b_sources.input(.input, .f32, &.{ 2, 3, 9 });
    const mapped = b.add(input, b.scalar(.f32, 1));
    b.output(b.sliceLoop(mapped, .{
        .axis = 2,
        .iterations = &.{
            .{ .offsets = &.{ 0, 0 }, .boundary = .{ .redirect = 0 } },
            .{ .offsets = &.{ 0, 1 }, .boundary = .{ .redirect = 3 } },
            .{ .offsets = &.{ 1, 0 }, .boundary = .{ .redirect = 4 } },
            .{ .offsets = &.{ 0, -1 }, .boundary = .{ .redirect = 1 } },
            .{ .offsets = &.{ -1, 0 }, .boundary = .{ .redirect = 2 } },
            .{ .offsets = &.{ 1, 1 }, .boundary = .{ .redirect = 7 } },
            .{ .offsets = &.{ 1, -1 }, .boundary = .{ .redirect = 8 } },
            .{ .offsets = &.{ -1, -1 }, .boundary = .{ .redirect = 5 } },
            .{ .offsets = &.{ -1, 1 }, .boundary = .{ .redirect = 6 } },
        },
    }));
    break :model b.finish().model();
};

test "slice loop fuses a vectorized producer and redirects boundary channels" {
    const offsets = [_][2]isize{
        .{ 0, 0 }, .{ 0, 1 },  .{ 1, 0 },   .{ 0, -1 }, .{ -1, 0 },
        .{ 1, 1 }, .{ 1, -1 }, .{ -1, -1 }, .{ -1, 1 },
    };
    const redirects = [_]usize{ 0, 3, 4, 1, 2, 7, 8, 5, 6 };
    var input: [54]f32 = undefined;
    for (&input, 0..) |*value, index| value.* = @floatFromInt(index);

    var expected: [54]f32 = undefined;
    for (0..2) |y| for (0..3) |x| for (0..9) |channel_index| {
        const source_y = @as(isize, @intCast(y)) - offsets[channel_index][0];
        const source_x = @as(isize, @intCast(x)) - offsets[channel_index][1];
        const source_index = if (source_y < 0 or source_y >= 2 or source_x < 0 or source_x >= 3)
            (y * 3 + x) * 9 + redirects[channel_index]
        else
            (@as(usize, @intCast(source_y)) * 3 + @as(usize, @intCast(source_x))) * 9 + channel_index;
        expected[(y * 3 + x) * 9 + channel_index] = input[source_index] + 1;
    };

    var model = loop_model.init();
    try model.copyInput(.input, &input);
    model.run();
    try std.testing.expectEqualSlices(f32, &expected, model.outputView(0).contiguousSlice().?);
}

const InferredLoopDefinitionSources = enum(usize) { input };
const InferredLoopDefinition = zgc.DefinitionBuilder;

const inferred_loop_model = model: {
    var b = InferredLoopDefinition.init();
    const b_sources = b.sources(InferredLoopDefinitionSources);
    const input = b_sources.input(.input, .f32, &.{ 2, 3, 2 });
    const mapped = b.add(input, b.scalar(.f32, 1));
    const left = b.slice(mapped, .{ .axis = 2, .start = 0, .end = 1 });
    const right = b.slice(mapped, .{ .axis = 2, .start = 1, .end = 2 });
    const shifted_left = b.shift(left, &.{ 0, 1, 0 }, .wrap);
    const shifted_right = b.shift(right, &.{ 0, -1, 0 }, .wrap);
    b.output(b.concat(&.{ shifted_left, shifted_right }, 2));
    break :model b.finish().model();
};

test "slice shift concat remaps infer a loop plan" {
    const graph = inferred_loop_model.executable;
    switch (graph.nodes[graph.node_ct - 1].?.op) {
        .compute => |compute| switch (compute) {
            .kernel => |kernel| switch (kernel) {
                .map => |map| switch (map.strategy) {
                    .loop => |loop_plan| try std.testing.expectEqual(@as(usize, 2), loop_plan.iterations.len),
                    .traversal, .segmented => return error.TestUnexpectedResult,
                },
                .reduction, .contraction => return error.TestUnexpectedResult,
            },
            .direct => return error.TestUnexpectedResult,
        },
        .view => return error.TestUnexpectedResult,
    }

    var model = inferred_loop_model.init();
    try model.copyInput(.input, &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 });
    model.run();
    try std.testing.expectEqualSlices(
        f32,
        &.{ 5, 4, 1, 6, 3, 2, 11, 10, 7, 12, 9, 8 },
        model.outputView(0).contiguousSlice().?,
    );
}

const StridedDefinitionSources = enum(usize) { input };
const StridedDefinition = zgc.DefinitionBuilder;

const strided_model = model: {
    var b = StridedDefinition.init();
    const b_sources = b.sources(StridedDefinitionSources);
    const input = b_sources.input(.input, .f32, &.{ 2, 3 });
    const transposed = b.transpose(input, 0, 1);
    b.output(b.shift(transposed, &.{ 0, 1 }, .wrap));
    break :model b.finish().model();
};

test "shift reads a strided source view in logical coordinates" {
    var model = strided_model.init();
    try model.copyInput(.input, &.{ 1, 2, 3, 4, 5, 6 });
    model.run();
    try std.testing.expectEqualSlices(f32, &.{ 4, 1, 5, 2, 6, 3 }, model.outputView(0).contiguousSlice().?);
}

const SingletonDefinitionSources = enum(usize) { input };
const SingletonDefinition = zgc.DefinitionBuilder;

const singleton_model = model: {
    var b = SingletonDefinition.init();
    const b_sources = b.sources(SingletonDefinitionSources);
    const input = b_sources.input(.input, .i8, &.{1});
    b.output(b.shift(input, &.{101}, .reflect));
    break :model b.finish().model();
};

test "reflection of a singleton axis stays on its only element" {
    var model = singleton_model.init();
    try model.copyInput(.input, &.{42});
    model.run();
    try std.testing.expectEqualSlices(i8, &.{42}, model.outputView(0).contiguousSlice().?);
}
