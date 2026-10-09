const std = @import("std");
const zgc = @import("zgc");
pub const width = 120;
pub const height = 88;
pub const cell_count = width * height;

// The world is the model's only external source. Its fixed dimensions make the
// entire neighborhood geometry available while the graph is being defined.
const Sources = enum(usize) { world };
const Definition = zgc.DefinitionBuilder;

fn lifeRule(
    comptime neighbor_count: zgc.Expr,
    comptime world: zgc.Expr,
    comptime two: zgc.Expr,
    comptime three: zgc.Expr,
) zgc.Expr {
    const born = neighbor_count.equal(three);
    const survives = world.logicalAnd(neighbor_count.equal(two));
    return born.logicalOr(survives);
}

pub const Model = model: {
    var b = Definition.init();
    const b_sources = b.sources(Sources);
    const world = b.expr(b_sources.input(.world, .bool, &.{ height, width }));

    // A dead-cell border gives every cell a complete 3x3 neighborhood without
    // requiring boundary checks in the generated computation.
    const dead = b.expr(b.scalar(.bool, false));
    const neighborhoods = world
        .pad(dead, .{
            .before = &.{ 1, 1 },
            .after = &.{ 1, 1 },
        })
        .windows(.{ .sizes = &.{ 3, 3 } });

    // Convert predicates to counts, reduce the two window axes,
    // and remove the center cell so the result contains neighbor counts rather than occupancy.
    const one = b.expr(b.scalar(.i8, 1));
    const zero = b.expr(b.scalar(.i8, 0));
    const neighborhood_total = neighborhoods
        .where(one, zero)
        .sum(.{ .axes = &.{ -2, -1 } });
    const center_value = world.where(one, zero);
    const neighbor_count = neighborhood_total.sub(center_value);

    // A cell is alive next when it is born with three neighbors or survives with two.
    // Expressing both rules as predicates keeps the output boolean.
    const two = b.expr(b.scalar(.i8, 2));
    const three = b.expr(b.scalar(.i8, 3));
    b.output(neighbor_count.apply(lifeRule, .{ world, two, three }).value);

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
