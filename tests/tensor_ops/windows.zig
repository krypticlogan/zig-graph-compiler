const std = @import("std");
const zgc = @import("zgc");

const LifeSources = enum(usize) { world };
const LifeDefinition = zgc.DefinitionBuilder;

const life_step = model: {
    var builder = LifeDefinition.init();
    const builder_sources = builder.sources(LifeSources);
    const world = builder_sources.input(.world, .bool, &.{ 3, 3 });
    const dead = builder.scalar(.bool, false);
    const padded = builder.pad(world, dead, .{
        .before = &.{ 1, 1 },
        .after = &.{ 1, 1 },
    });
    const neighborhoods = builder.windows(padded, .{ .sizes = &.{ 3, 3 } });

    const one = builder.scalar(.i8, 1);
    const zero = builder.scalar(.i8, 0);
    const neighborhood_values = builder.where(neighborhoods, one, zero);
    const neighborhood_total = builder.sum(neighborhood_values, .{ .axes = &.{ -2, -1 } });
    const center_value = builder.where(world, one, zero);
    const neighbor_count = builder.sub(neighborhood_total, center_value);

    const two = builder.scalar(.i8, 2);
    const three = builder.scalar(.i8, 3);
    const has_two_neighbors = builder.equal(neighbor_count, two);
    const has_three_neighbors = builder.equal(neighbor_count, three);
    const survives = builder.logicalAnd(world, has_two_neighbors);
    builder.output(builder.logicalOr(has_three_neighbors, survives));
    break :model builder.finish().model();
};

test "padding and one overlapping window view express a Conway step" {
    const graph = life_step.executable;
    const padded = graph.tensors[2].?;
    const neighborhoods = graph.tensors[3].?;

    try std.testing.expectEqualSlices(usize, &.{ 5, 5 }, padded.shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 3, 3, 3, 3 }, neighborhoods.shape.slice());
    try std.testing.expectEqual([4]isize{ 5, 1, 5, 1 }, neighborhoods.layout.strides[0..4].*);
    try std.testing.expectEqual(padded.storage_tensor, neighborhoods.storage_tensor);

    var model = life_step.init();
    try model.copyInput(.world, &.{
        false, true, false,
        false, true, false,
        false, true, false,
    });
    model.run();
    try std.testing.expectEqualSlices(bool, &.{
        false, false, false,
        true,  true,  true,
        false, false, false,
    }, model.outputView(0).storage);
}

const WindowSources = enum(usize) { input };
const WindowDefinition = zgc.DefinitionBuilder;

const dilated_windows = model: {
    var builder = WindowDefinition.init();
    const builder_sources = builder.sources(WindowSources);
    const input = builder_sources.input(.input, .i8, &.{7});
    const windows = builder.windows(input, .{
        .sizes = &.{3},
        .strides = &.{2},
        .dilations = &.{2},
    });
    builder.output(builder.contiguous(windows));
    break :model builder.finish().model();
};

test "window strides and dilation produce static overlapping geometry" {
    const window = dilated_windows.executable.tensors[1].?;
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, window.shape.slice());
    try std.testing.expectEqual([2]isize{ 2, 2 }, window.layout.strides[0..2].*);

    var model = dilated_windows.init();
    try model.copyInput(.input, &.{ 0, 1, 2, 3, 4, 5, 6 });
    model.run();
    try std.testing.expectEqualSlices(i8, &.{ 0, 2, 4, 2, 4, 6 }, model.outputView(0).storage);
}
