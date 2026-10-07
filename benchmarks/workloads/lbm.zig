const std = @import("std");
const zgc = @import("zgc");

const H = 180;
const W = 320;

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

const Definition = zgc.DefinitionBuilder(Sources, .{
    .max_rank = 3,
    .max_nodes = 192,
    .max_tensors = 320,
    .max_input_refs = 512,
    .max_outputs = 1,
});

const Value = Definition.TensorValue;

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
    const f = b.input(.f, .f32, &.{ H, W, 9 });
    const omega = b.input(.omega, .f32, &.{1});
    const cx = b.constant(.cx, .f32, &.{9});
    const cy = b.constant(.cy, .f32, &.{9});
    const weights = b.constant(.weights, .f32, &.{9});

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

const definition = blk: {
    @setEvalBranchQuota(100_000);
    var builder = Definition.init();
    define(&builder);
    break :blk builder.finish();
};

const FluidStep = blk: {
    @setEvalBranchQuota(2_000_000);
    break :blk definition.model();
};

fn directLbmStep(
    comptime DirectHeight: usize,
    comptime DirectWidth: usize,
    src: *const [DirectHeight * DirectWidth * 9]f32,
    dst: *[DirectHeight * DirectWidth * 9]f32,
    omega: f32,
) void {
    @setRuntimeSafety(false);

    const f = src.*;
    var out = dst;

    for (0..DirectHeight) |y| {
        for (0..DirectWidth) |x| {
            const base = (y * DirectWidth + x) * 9;

            // ---------------------------------------------------------
            // Load the complete D2Q9 state for this lattice cell.
            // ---------------------------------------------------------

            const f0 = f[base + 0];
            const f1 = f[base + 1];
            const f2 = f[base + 2];
            const f3 = f[base + 3];
            const f4 = f[base + 4];
            const f5 = f[base + 5];
            const f6 = f[base + 6];
            const f7 = f[base + 7];
            const f8 = f[base + 8];

            // ---------------------------------------------------------
            // Macroscopic quantities.
            //
            // Directions:
            //
            //   6 2 5
            //    \|/
            //   3 0 1
            //    /|\
            //   7 4 8
            //
            // cx = { 0, 1, 0,-1, 0, 1,-1,-1, 1 }
            // cy = { 0, 0, 1, 0,-1, 1, 1,-1,-1 }
            // ---------------------------------------------------------

            const rho =
                f0 + f1 + f2 +
                f3 + f4 + f5 +
                f6 + f7 + f8;

            const inv_rho = 1.0 / rho;

            const jx =
                f1 - f3 +
                f5 - f6 -
                f7 + f8;

            const jy =
                f2 - f4 +
                f5 + f6 -
                f7 - f8;

            const ux = jx * inv_rho;
            const uy = jy * inv_rho;

            const u_sq = ux * ux + uy * uy;

            // Common equilibrium component:
            //
            // feq_i = w_i * rho *
            //          (1 + 3 eu + 4.5 eu² - 1.5 u²)
            const common = 1.0 - 1.5 * u_sq;

            // ---------------------------------------------------------
            // Equilibrium + BGK collision.
            //
            // Instead of creating:
            //
            // eu
            // eu²
            // polynomial
            // feq
            // delta
            // post_collision
            //
            // as separate tensors, everything remains scalar/local.
            // ---------------------------------------------------------

            const feq0 =
                (4.0 / 9.0) *
                rho *
                common;

            const eu1 = ux;
            const feq1 =
                (1.0 / 9.0) *
                rho *
                (common + 3.0 * eu1 + 4.5 * eu1 * eu1);

            const eu2 = uy;
            const feq2 =
                (1.0 / 9.0) *
                rho *
                (common + 3.0 * eu2 + 4.5 * eu2 * eu2);

            const eu3 = -ux;
            const feq3 =
                (1.0 / 9.0) *
                rho *
                (common + 3.0 * eu3 + 4.5 * eu3 * eu3);

            const eu4 = -uy;
            const feq4 =
                (1.0 / 9.0) *
                rho *
                (common + 3.0 * eu4 + 4.5 * eu4 * eu4);

            const eu5 = ux + uy;
            const feq5 =
                (1.0 / 36.0) *
                rho *
                (common + 3.0 * eu5 + 4.5 * eu5 * eu5);

            const eu6 = -ux + uy;
            const feq6 =
                (1.0 / 36.0) *
                rho *
                (common + 3.0 * eu6 + 4.5 * eu6 * eu6);

            const eu7 = -ux - uy;
            const feq7 =
                (1.0 / 36.0) *
                rho *
                (common + 3.0 * eu7 + 4.5 * eu7 * eu7);

            const eu8 = ux - uy;
            const feq8 =
                (1.0 / 36.0) *
                rho *
                (common + 3.0 * eu8 + 4.5 * eu8 * eu8);

            const p0 = f0 + omega * (feq0 - f0);
            const p1 = f1 + omega * (feq1 - f1);
            const p2 = f2 + omega * (feq2 - f2);
            const p3 = f3 + omega * (feq3 - f3);
            const p4 = f4 + omega * (feq4 - f4);
            const p5 = f5 + omega * (feq5 - f5);
            const p6 = f6 + omega * (feq6 - f6);
            const p7 = f7 + omega * (feq7 - f7);
            const p8 = f8 + omega * (feq8 - f8);

            // ---------------------------------------------------------
            // Streaming.
            //
            // Push each post-collision population directly into its
            // destination cell in dst.
            //
            // There is no separate shift/roll pass.
            // ---------------------------------------------------------

            out[(y * DirectWidth + x) * 9 + 0] = p0;

            if (x + 1 < DirectWidth) out[(y * DirectWidth + x + 1) * 9 + 1] = p1 else out[base + 3] = p1;
            if (y + 1 < DirectHeight) out[((y + 1) * DirectWidth + x) * 9 + 2] = p2 else out[base + 4] = p2;
            if (x > 0) out[(y * DirectWidth + x - 1) * 9 + 3] = p3 else out[base + 1] = p3;
            if (y > 0) out[((y - 1) * DirectWidth + x) * 9 + 4] = p4 else out[base + 2] = p4;

            if (x + 1 < DirectWidth and y + 1 < DirectHeight) out[((y + 1) * DirectWidth + x + 1) * 9 + 5] = p5 else out[base + 7] = p5;
            if (x > 0 and y + 1 < DirectHeight) out[((y + 1) * DirectWidth + x - 1) * 9 + 6] = p6 else out[base + 8] = p6;
            if (x > 0 and y > 0) out[((y - 1) * DirectWidth + x - 1) * 9 + 7] = p7 else out[base + 5] = p7;
            if (x + 1 < DirectWidth and y > 0) out[((y - 1) * DirectWidth + x + 1) * 9 + 8] = p8 else out[base + 6] = p8;
        }
    }
}

const cell_count = H * W;
const population_count = cell_count * 9;
const fluid_weights = [9]f32{
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
};
const cx_values = [9]f32{ 0, 1, 0, -1, 0, 1, -1, -1, 1 };
const cy_values = [9]f32{ 0, 0, 1, 0, -1, 1, 1, -1, -1 };

var zgc_model = FluidStep.init();
var zgc_populations: [population_count]f32 = undefined;
var direct_populations: [population_count]f32 = undefined;
var direct_output: [population_count]f32 = undefined;

pub const ZgcBenchmark = struct {
    pub const CompiledModel = FluidStep;

    pub const name = "graph/application/lbm-d2q9/180x320/zgc";
    pub const default_iterations = 1;
    pub const default_warmup_iterations = 1;
    pub const work_items_per_invocation: f64 = cell_count;
    pub const work_unit = "cells";
    pub const bytes_per_invocation: f64 = 2 * @sizeOf(f32) * population_count;

    pub fn init() ZgcBenchmark {
        zgc_model = FluidStep.init();
        initializePopulations(&zgc_populations);
        zgc_model.copyInput(.f, &zgc_populations) catch unreachable;
        zgc_model.copyInput(.omega, &.{1.0}) catch unreachable;
        zgc_model.copySource(.cx, &cx_values) catch unreachable;
        zgc_model.copySource(.cy, &cy_values) catch unreachable;
        zgc_model.copySource(.weights, &fluid_weights) catch unreachable;
        return .{};
    }

    pub fn validate(_: *ZgcBenchmark) !void {
        zgc_model.run();
        const rest_population = zgc_model.outputView(0).get(.{ H / 2, W / 2, 0 });
        if (!std.math.isFinite(rest_population) or
            !std.math.approxEqAbs(f32, rest_population, fluid_weights[0], 1e-5))
        {
            return error.IncorrectResult;
        }
    }

    pub fn run(_: *ZgcBenchmark, iterations: usize) void {
        for (0..iterations) |_| {
            zgc_model.run();
            std.mem.doNotOptimizeAway(zgc_model.outputView(0).storage);
        }
    }
};

pub const DirectBenchmark = struct {
    pub const name = "graph/application/lbm-d2q9/180x320/direct";
    pub const default_iterations = ZgcBenchmark.default_iterations;
    pub const default_warmup_iterations = ZgcBenchmark.default_warmup_iterations;
    pub const work_items_per_invocation = ZgcBenchmark.work_items_per_invocation;
    pub const work_unit = ZgcBenchmark.work_unit;
    pub const bytes_per_invocation = ZgcBenchmark.bytes_per_invocation;

    pub fn init() DirectBenchmark {
        initializePopulations(&direct_populations);
        return .{};
    }

    pub fn validate(_: *DirectBenchmark) !void {
        directLbmStep(H, W, &direct_populations, &direct_output, 1.0);
        const rest_population = direct_output[((H / 2) * W + W / 2) * 9];
        if (!std.math.isFinite(rest_population) or
            !std.math.approxEqAbs(f32, rest_population, fluid_weights[0], 1e-5))
        {
            return error.IncorrectResult;
        }
    }

    pub fn run(_: *DirectBenchmark, iterations: usize) void {
        for (0..iterations) |_| {
            directLbmStep(H, W, &direct_populations, &direct_output, 1.0);
            std.mem.doNotOptimizeAway(&direct_output);
        }
    }
};

fn initializePopulations(populations: *[population_count]f32) void {
    for (0..cell_count) |cell| {
        for (fluid_weights, 0..) |weight, direction| {
            populations[cell * 9 + direction] = weight;
        }
    }
}
