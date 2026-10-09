const std = @import("std");
const zgc = @import("zgc");
pub const width = 120;
pub const height = 88;
pub const cell_count = width * height;

// The world is the model's only external source. Its fixed dimensions make the
// entire neighborhood geometry available while the graph is being defined.
const Sources = enum(usize) { world };
const Definition = zgc.DefinitionBuilder;

pub const Model = model: {
    var b = Definition.init();
    const b_sources = b.sources(Sources);
    const world = b_sources.input(.world, .bool, &.{ height, width });

    // A dead-cell border gives every cell a complete 3x3 neighborhood without
    // requiring boundary checks in the generated computation.
    const dead = b.scalar(.bool, false);
    const padded = b.pad(world, dead, .{
        .before = &.{ 1, 1 },
        .after = &.{ 1, 1 },
    });
    const neighborhoods = b.windows(padded, .{ .sizes = &.{ 3, 3 } });

    // Convert predicates to counts, reduce the two window axes,
    // and remove the center cell so the result contains neighbor counts rather than occupancy.
    const one = b.scalar(.i8, 1);
    const zero = b.scalar(.i8, 0);
    const neighborhood_values = b.where(neighborhoods, one, zero);
    const neighborhood_total = b.sum(neighborhood_values, .{ .axes = &.{ -2, -1 } });
    const center_value = b.where(world, one, zero);
    const neighbor_count = b.sub(neighborhood_total, center_value);

    // A cell is alive next when it is born with three neighbors or survives with two.
    // Expressing both rules as predicates keeps the output boolean.
    const two = b.scalar(.i8, 2);
    const three = b.scalar(.i8, 3);
    b.output(b.logicalOr(b.equal(neighbor_count, three), b.logicalAnd(world, b.equal(neighbor_count, two))));

    // The application owns the evolving world,
    // so the source is bound for each step instead of becoming persistent model storage.
    break :model b.finish().modelWith(&.{
        .{ .source = .world, .binding = zgc.memory.Source.bound },
    });
};

pub fn step(model: *Model, world: *[cell_count]bool) void {
    // State remains outside the graph: bind the current generation, execute,
    // then feed the produced generation back into the application-owned buffer.
    model.bindInput(.world, world) catch unreachable;
    model.run();
    @memcpy(world, model.outputView(0).storage);
}
