const std = @import("std");
const zgc = @import("zgc");

pub const H = 180;
pub const W = 320;

// Fluid D2Q9:
//
// 6 2 5
// 3 0 1
// 7 4 8
//
// x grows rightward.
// y grows downward.
//
// Smoke uses D2Q5:
//   2
// 3 0 1
//   4
//
// The fluid is a standard BGK D2Q9 lattice.
// Smoke is an independent passive-scalar BGK D2Q5 lattice advected by the
// fluid's macroscopic velocity.
//
// Recurrent state:
//   f       [H,W,9]  fluid populations
//   smoke_g [H,W,5]  smoke populations
//
// Per-step controls:
//   smoke_injection [H,W,1] amount of smoke mass added this step
//   force_x/y       [H,W,1] fluid momentum impulse
//   omega           [1]     fluid BGK relaxation
//   smoke_omega     [1]     smoke BGK relaxation / diffusivity control
//   smoke_retention [1]     multiplicative per-step smoke retention
//
// Outputs:
//   0 next_f
//   1 next_smoke_g
//   2 smoke concentration
//   3 rho
//   4 ux
//   5 uy

const Sources = enum(usize) {
    f,
    smoke_g,
    smoke_injection,
    force_x,
    force_y,
    omega,
    smoke_omega,
    smoke_retention,
    cx,
    cy,
    fluid_weights,
    smoke_weights,
};

const Definition = zgc.DefinitionBuilder;

const Value = zgc.Value;

fn equilibriumPolynomial(
    comptime eu: zgc.Expr,
    comptime velocity_sq: zgc.Expr,
    comptime one: zgc.Expr,
    comptime three: zgc.Expr,
    comptime four_point_five: zgc.Expr,
    comptime one_point_five: zgc.Expr,
) zgc.Expr {
    const linear = three.mul(eu);
    const eu_sq = eu.mul(eu);
    const quadratic = four_point_five.mul(eu_sq);
    const speed_correction = one_point_five.mul(velocity_sq);
    return one.add(linear).add(quadratic).sub(speed_correction);
}

fn passiveScalarPolynomial(
    comptime eu: zgc.Expr,
    comptime one: zgc.Expr,
    comptime three: zgc.Expr,
) zgc.Expr {
    return one.add(three.mul(eu));
}

fn bgkCollision(
    comptime populations: zgc.Expr,
    comptime equilibrium: zgc.Expr,
    comptime omega: zgc.Expr,
) zgc.Expr {
    return populations.add(omega.mul(equilibrium.sub(populations)));
}

fn channel(
    b: *Definition,
    comptime x: Value,
    comptime index: usize,
) Value {
    return b.slice(x, .{
        .axis = 2,
        .start = index,
        .end = index + 1,
    });
}

/// Streams one population by an integer lattice offset while applying
/// bounce-back at the outer domain boundary.
///
/// `moving` is the post-collision population travelling in (dx,dy).
/// `opposite` is the post-collision population travelling in (-dx,-dy).
fn streamSolid(
    b: *Definition,
    comptime moving: Value,
    comptime opposite: Value,
    comptime dx: i8,
    comptime dy: i8,
) Value {
    if (dx == 0 and dy == 0) return moving;

    // Horizontal movement.
    if (dy == 0) {
        if (dx > 0) {
            const wall = b.slice(opposite, .{
                .axis = 1,
                .start = 0,
                .end = 1,
            });
            const interior = b.slice(moving, .{
                .axis = 1,
                .start = 0,
                .end = W - 1,
            });
            return b.concat(&.{ wall, interior }, 1);
        }

        const interior = b.slice(moving, .{
            .axis = 1,
            .start = 1,
            .end = W,
        });
        const wall = b.slice(opposite, .{
            .axis = 1,
            .start = W - 1,
            .end = W,
        });
        return b.concat(&.{ interior, wall }, 1);
    }

    // Vertical movement.
    if (dx == 0) {
        if (dy > 0) {
            const wall = b.slice(opposite, .{
                .axis = 0,
                .start = 0,
                .end = 1,
            });
            const interior = b.slice(moving, .{
                .axis = 0,
                .start = 0,
                .end = H - 1,
            });
            return b.concat(&.{ wall, interior }, 0);
        }

        const interior = b.slice(moving, .{
            .axis = 0,
            .start = 1,
            .end = H,
        });
        const wall = b.slice(opposite, .{
            .axis = 0,
            .start = H - 1,
            .end = H,
        });
        return b.concat(&.{ interior, wall }, 0);
    }

    // Diagonal movement. Only the fluid D2Q9 lattice uses this branch.
    const moving_rows = if (dy > 0)
        b.slice(moving, .{ .axis = 0, .start = 0, .end = H - 1 })
    else
        b.slice(moving, .{ .axis = 0, .start = 1, .end = H });

    const opposite_rows = if (dy > 0)
        b.slice(opposite, .{ .axis = 0, .start = 1, .end = H })
    else
        b.slice(opposite, .{ .axis = 0, .start = 0, .end = H - 1 });

    const body = if (dx > 0) blk: {
        const wall = b.slice(opposite_rows, .{
            .axis = 1,
            .start = 0,
            .end = 1,
        });
        const interior = b.slice(moving_rows, .{
            .axis = 1,
            .start = 0,
            .end = W - 1,
        });
        break :blk b.concat(&.{ wall, interior }, 1);
    } else blk: {
        const interior = b.slice(moving_rows, .{
            .axis = 1,
            .start = 1,
            .end = W,
        });
        const wall = b.slice(opposite_rows, .{
            .axis = 1,
            .start = W - 1,
            .end = W,
        });
        break :blk b.concat(&.{ interior, wall }, 1);
    };

    if (dy > 0) {
        const wall = b.slice(opposite, .{
            .axis = 0,
            .start = 0,
            .end = 1,
        });
        return b.concat(&.{ wall, body }, 0);
    }

    const wall = b.slice(opposite, .{
        .axis = 0,
        .start = H - 1,
        .end = H,
    });
    return b.concat(&.{ body, wall }, 0);
}

fn define(b: *Definition) void {
    const sources = b.sources(Sources);
    // ---------------------------------------------------------------------
    // Sources
    // ---------------------------------------------------------------------
    const f = b.expr(sources.input(.f, .f32, &.{ H, W, 9 }));
    const smoke_g = b.expr(sources.input(.smoke_g, .f32, &.{ H, W, 5 }));

    // Amount of smoke concentration added during this step.
    // Usually zero everywhere except at emitter cells.
    const smoke_injection = b.expr(sources.input(
        .smoke_injection,
        .f32,
        &.{ H, W, 1 },
    ));

    const force_x = b.expr(sources.input(.force_x, .f32, &.{ H, W, 1 }));
    const force_y = b.expr(sources.input(.force_y, .f32, &.{ H, W, 1 }));

    // Fluid relaxation:
    //
    //   nu = cs^2 * (1 / omega - 1/2)
    //
    // for cs^2 = 1/3 in standard D2Q9 lattice units.
    const omega = b.expr(sources.input(.omega, .f32, &.{1}));

    // Passive-scalar relaxation:
    //
    //   D = cs^2 * (1 / smoke_omega - 1/2)
    //
    // D2Q5 below also uses cs^2 = 1/3.
    const smoke_omega = b.expr(sources.input(.smoke_omega, .f32, &.{1}));

    // 1.0 => no decay.
    // e.g. 0.997 => retain 99.7% of smoke each step.
    const smoke_retention = b.expr(sources.input(.smoke_retention, .f32, &.{1}));

    // Fluid directions:
    //
    // cx = [ 0, 1, 0,-1, 0, 1,-1,-1, 1 ]
    // cy = [ 0, 0, 1, 0,-1, 1, 1,-1,-1 ]
    const cx = b.expr(sources.constant(.cx, .f32, &.{9}));
    const cy = b.expr(sources.constant(.cy, .f32, &.{9}));

    // Standard D2Q9 weights:
    // [4/9, 1/9,1/9,1/9,1/9, 1/36,1/36,1/36,1/36]
    const fluid_weights = b.expr(sources.constant(.fluid_weights, .f32, &.{9}));

    // D2Q5 passive-scalar weights:
    // [1/3, 1/6, 1/6, 1/6, 1/6]
    const smoke_weights = b.expr(sources.constant(.smoke_weights, .f32, &.{5}));

    // First five fluid directions are exactly the D2Q5 cardinal set.
    const smoke_cx = cx.slice(.{
        .axis = 0,
        .start = 0,
        .end = 5,
    });

    const smoke_cy = cy.slice(.{
        .axis = 0,
        .start = 0,
        .end = 5,
    });

    // ---------------------------------------------------------------------
    // Scalars
    // ---------------------------------------------------------------------

    const one = b.expr(b.scalar(.f32, 1.0));
    const three = b.expr(b.scalar(.f32, 3.0));
    const four_point_five = b.expr(b.scalar(.f32, 4.5));
    const one_point_five = b.expr(b.scalar(.f32, 1.5));

    // =====================================================================
    // FLUID: MACROSCOPIC DENSITY
    // =====================================================================

    // rho = sum_i f_i
    const rho = f.sum(.{
        .axes = &.{2},
        .keep_dims = true,
    });

    // =====================================================================
    // FLUID: MACROSCOPIC VELOCITY
    // =====================================================================

    const momentum_x = f.mul(cx).sum(.{
        .axes = &.{2},
        .keep_dims = true,
    });

    const momentum_y = f.mul(cy).sum(.{
        .axes = &.{2},
        .keep_dims = true,
    });

    const ux = momentum_x.div(rho);
    const uy = momentum_y.div(rho);

    // Apply caller-controlled momentum injection to the equilibrium velocity.
    const collision_ux = ux.add(force_x.div(rho));
    const collision_uy = uy.add(force_y.div(rho));

    const ux_sq = collision_ux.mul(collision_ux);
    const uy_sq = collision_uy.mul(collision_uy);
    const velocity_sq = ux_sq.add(uy_sq);

    // =====================================================================
    // FLUID: D2Q9 EQUILIBRIUM
    // =====================================================================

    const fluid_eu_x = collision_ux.mul(cx);
    const fluid_eu_y = collision_uy.mul(cy);
    const fluid_eu = fluid_eu_x.add(fluid_eu_y);

    // f_eq_i =
    //   w_i rho [1 + 3 e_i.u + 4.5(e_i.u)^2 - 1.5|u|^2]
    const fluid_equilibrium_poly = fluid_eu.apply(equilibriumPolynomial, .{
        velocity_sq,
        one,
        three,
        four_point_five,
        one_point_five,
    });

    const f_eq = rho.mul(fluid_weights).mul(fluid_equilibrium_poly);

    // =====================================================================
    // FLUID: BGK COLLISION
    // =====================================================================

    // f* = f + omega(f_eq - f)
    const post_collision = f.apply(bgkCollision, .{ f_eq, omega });

    // =====================================================================
    // FLUID: STREAMING + DOMAIN-WALL BOUNCE-BACK
    // =====================================================================

    const f0 = channel(b, post_collision.value, 0);
    const f1 = channel(b, post_collision.value, 1);
    const f2 = channel(b, post_collision.value, 2);
    const f3 = channel(b, post_collision.value, 3);
    const f4 = channel(b, post_collision.value, 4);
    const f5 = channel(b, post_collision.value, 5);
    const f6 = channel(b, post_collision.value, 6);
    const f7 = channel(b, post_collision.value, 7);
    const f8 = channel(b, post_collision.value, 8);

    const s0 = f0;
    const s1 = streamSolid(b, f1, f3, 1, 0);
    const s2 = streamSolid(b, f2, f4, 0, 1);
    const s3 = streamSolid(b, f3, f1, -1, 0);
    const s4 = streamSolid(b, f4, f2, 0, -1);
    const s5 = streamSolid(b, f5, f7, 1, 1);
    const s6 = streamSolid(b, f6, f8, -1, 1);
    const s7 = streamSolid(b, f7, f5, -1, -1);
    const s8 = streamSolid(b, f8, f6, 1, -1);

    const next_f = b.concat(
        &.{ s0, s1, s2, s3, s4, s5, s6, s7, s8 },
        2,
    );

    // =====================================================================
    // SMOKE: DECAY + SOURCE INJECTION
    // =====================================================================

    // Decay every population uniformly so total smoke concentration decays
    // by smoke_retention as well.
    const smoke_retained = smoke_g.mul(smoke_retention);

    // Inject source mass into the rest population. This adds smoke exactly,
    // independently of smoke_omega, before collision redistributes it.
    const g0_retained = b.expr(channel(b, smoke_retained.value, 0));
    const g1_retained = channel(b, smoke_retained.value, 1);
    const g2_retained = channel(b, smoke_retained.value, 2);
    const g3_retained = channel(b, smoke_retained.value, 3);
    const g4_retained = channel(b, smoke_retained.value, 4);

    const g0_injected = g0_retained.add(smoke_injection);

    const smoke_pre_collision_g = b.concat(
        &.{
            g0_injected.value,
            g1_retained,
            g2_retained,
            g3_retained,
            g4_retained,
        },
        2,
    );

    // Scalar concentration:
    //
    // smoke = sum_i g_i
    const smoke = b.expr(smoke_pre_collision_g).sum(.{
        .axes = &.{2},
        .keep_dims = true,
    });

    // =====================================================================
    // SMOKE: D2Q5 ADVECTION-DIFFUSION EQUILIBRIUM
    // =====================================================================

    // The passive scalar is advected by the fluid velocity.
    //
    // g_eq_i = w_i * smoke * [1 + 3(e_i.u)]
    //
    // The fluid velocity is not changed by the passive scalar here.
    const smoke_eu_x = collision_ux.mul(smoke_cx);
    const smoke_eu_y = collision_uy.mul(smoke_cy);
    const smoke_eu = smoke_eu_x.add(smoke_eu_y);

    const smoke_equilibrium_poly = smoke_eu.apply(passiveScalarPolynomial, .{ one, three });

    const smoke_eq = smoke.mul(smoke_weights).mul(smoke_equilibrium_poly);

    // =====================================================================
    // SMOKE: BGK COLLISION
    // =====================================================================

    // g* = g + omega_s(g_eq - g)
    const smoke_post_collision = b.expr(smoke_pre_collision_g).apply(bgkCollision, .{
        smoke_eq,
        smoke_omega,
    });

    // =====================================================================
    // SMOKE: STREAMING + NO-FLUX DOMAIN WALLS
    // =====================================================================

    const g0 = channel(b, smoke_post_collision.value, 0);
    const g1 = channel(b, smoke_post_collision.value, 1);
    const g2 = channel(b, smoke_post_collision.value, 2);
    const g3 = channel(b, smoke_post_collision.value, 3);
    const g4 = channel(b, smoke_post_collision.value, 4);

    // Bounce-back at the outer walls creates a simple no-flux boundary for
    // the scalar field.
    const sg0 = g0;
    const sg1 = streamSolid(b, g1, g3, 1, 0);
    const sg2 = streamSolid(b, g2, g4, 0, 1);
    const sg3 = streamSolid(b, g3, g1, -1, 0);
    const sg4 = streamSolid(b, g4, g2, 0, -1);

    const next_smoke_g = b.concat(
        &.{ sg0, sg1, sg2, sg3, sg4 },
        2,
    );

    // Render the post-step concentration rather than the pre-step field.
    const next_smoke = b.expr(next_smoke_g).sum(.{
        .axes = &.{2},
        .keep_dims = true,
    });

    // =====================================================================
    // OUTPUTS
    // =====================================================================

    // Recurrent state.
    b.output(next_f);
    b.output(next_smoke_g);

    // Inspection / rendering.
    b.output(next_smoke.value);
    b.output(rho.value);
    b.output(ux.value);
    b.output(uy.value);
}

pub const definition = blk: {
    @setEvalBranchQuota(2_000_000);
    var builder = Definition.init();
    define(&builder);
    break :blk builder.finish();
};

pub const FluidSmokeStep = blk: {
    @setEvalBranchQuota(2_000_000);
    break :blk definition.model();
};
