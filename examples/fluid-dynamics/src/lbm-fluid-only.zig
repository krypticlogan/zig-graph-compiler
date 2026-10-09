const zgc = @import("zgc");

pub const H = 180;
pub const W = 320;

// D2Q9 channel order:
//
// 6 2 5
// 3 0 1
// 7 4 8
//
// This graph is the direct solver's workload: one BGK fluid collision and
// solid-wall streaming step. It intentionally contains no force field,
// passive scalar, or inspection-only outputs.
const Sources = enum(usize) {
    f,
    omega,
    cx,
    cy,
    weights,
};

const Definition = zgc.DefinitionBuilder;

const Value = zgc.Value;

fn channel(b: *Definition, comptime tensor: Value, comptime index: usize) Value {
    return b.slice(tensor, .{ .axis = 2, .start = index, .end = index + 1 });
}

/// Push one population through the domain and bounce it into its opposite
/// channel at solid outer walls. Slice and concatenation express each static
/// interior and boundary region directly in the graph.
fn streamSolid(
    b: *Definition,
    comptime moving: Value,
    comptime opposite: Value,
    comptime dx: i8,
    comptime dy: i8,
) Value {
    if (dx == 0 and dy == 0) return moving;

    if (dy == 0) {
        if (dx > 0) {
            const wall = b.slice(opposite, .{ .axis = 1, .start = 0, .end = 1 });
            const interior = b.slice(moving, .{ .axis = 1, .start = 0, .end = W - 1 });
            return b.concat(&.{ wall, interior }, 1);
        }
        const interior = b.slice(moving, .{ .axis = 1, .start = 1, .end = W });
        const wall = b.slice(opposite, .{ .axis = 1, .start = W - 1, .end = W });
        return b.concat(&.{ interior, wall }, 1);
    }

    if (dx == 0) {
        if (dy > 0) {
            const wall = b.slice(opposite, .{ .axis = 0, .start = 0, .end = 1 });
            const interior = b.slice(moving, .{ .axis = 0, .start = 0, .end = H - 1 });
            return b.concat(&.{ wall, interior }, 0);
        }
        const interior = b.slice(moving, .{ .axis = 0, .start = 1, .end = H });
        const wall = b.slice(opposite, .{ .axis = 0, .start = H - 1, .end = H });
        return b.concat(&.{ interior, wall }, 0);
    }

    const moving_rows = if (dy > 0)
        b.slice(moving, .{ .axis = 0, .start = 0, .end = H - 1 })
    else
        b.slice(moving, .{ .axis = 0, .start = 1, .end = H });
    const opposite_rows = if (dy > 0)
        b.slice(opposite, .{ .axis = 0, .start = 1, .end = H })
    else
        b.slice(opposite, .{ .axis = 0, .start = 0, .end = H - 1 });

    const body = if (dx > 0) blk: {
        const wall = b.slice(opposite_rows, .{ .axis = 1, .start = 0, .end = 1 });
        const interior = b.slice(moving_rows, .{ .axis = 1, .start = 0, .end = W - 1 });
        break :blk b.concat(&.{ wall, interior }, 1);
    } else blk: {
        const interior = b.slice(moving_rows, .{ .axis = 1, .start = 1, .end = W });
        const wall = b.slice(opposite_rows, .{ .axis = 1, .start = W - 1, .end = W });
        break :blk b.concat(&.{ interior, wall }, 1);
    };

    if (dy > 0) {
        const wall = b.slice(opposite, .{ .axis = 0, .start = 0, .end = 1 });
        return b.concat(&.{ wall, body }, 0);
    }
    const wall = b.slice(opposite, .{ .axis = 0, .start = H - 1, .end = H });
    return b.concat(&.{ body, wall }, 0);
}

fn define(b: *Definition) void {
    const sources = b.sources(Sources);
    const f = sources.input(.f, .f32, &.{ H, W, 9 });
    const omega = sources.input(.omega, .f32, &.{1});
    const cx = sources.constant(.cx, .f32, &.{9});
    const cy = sources.constant(.cy, .f32, &.{9});
    const weights = sources.constant(.weights, .f32, &.{9});

    const one = b.scalar(.f32, 1.0);
    const three = b.scalar(.f32, 3.0);
    const four_point_five = b.scalar(.f32, 4.5);
    const one_point_five = b.scalar(.f32, 1.5);

    const rho = b.sum(f, .{ .axes = &.{2}, .keep_dims = true });
    const momentum_x = b.sum(b.mul(f, cx), .{ .axes = &.{2}, .keep_dims = true });
    const momentum_y = b.sum(b.mul(f, cy), .{ .axes = &.{2}, .keep_dims = true });
    const ux = b.div(momentum_x, rho);
    const uy = b.div(momentum_y, rho);
    const velocity_sq = b.add(b.mul(ux, ux), b.mul(uy, uy));

    const eu = b.add(b.mul(ux, cx), b.mul(uy, cy));
    const equilibrium_poly = b.sub(
        b.add(
            b.add(one, b.mul(three, eu)),
            b.mul(four_point_five, b.mul(eu, eu)),
        ),
        b.mul(one_point_five, velocity_sq),
    );
    const equilibrium = b.mul(b.mul(rho, weights), equilibrium_poly);
    const post_collision = b.add(f, b.mul(omega, b.sub(equilibrium, f)));

    const f0 = channel(b, post_collision, 0);
    const f1 = channel(b, post_collision, 1);
    const f2 = channel(b, post_collision, 2);
    const f3 = channel(b, post_collision, 3);
    const f4 = channel(b, post_collision, 4);
    const f5 = channel(b, post_collision, 5);
    const f6 = channel(b, post_collision, 6);
    const f7 = channel(b, post_collision, 7);
    const f8 = channel(b, post_collision, 8);

    const next_f = b.concat(&.{
        f0,
        streamSolid(b, f1, f3, 1, 0),
        streamSolid(b, f2, f4, 0, 1),
        streamSolid(b, f3, f1, -1, 0),
        streamSolid(b, f4, f2, 0, -1),
        streamSolid(b, f5, f7, 1, 1),
        streamSolid(b, f6, f8, -1, 1),
        streamSolid(b, f7, f5, -1, -1),
        streamSolid(b, f8, f6, 1, -1),
    }, 2);

    b.output(next_f);
}

pub const definition = blk: {
    @setEvalBranchQuota(100_000);
    var builder = Definition.init();
    define(&builder);
    break :blk builder.finish();
};

pub const FluidStep = blk: {
    @setEvalBranchQuota(2_000_000);
    break :blk definition.model();
};
