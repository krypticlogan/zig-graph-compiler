//! D2Q9 single-phase free-surface LBM (2D, lattice units).
//!
//! Each call advances populations, liquid mass, and cell types in lockstep.
//! Prototype: validate mass conservation and interface topology before use.
//! Gas is not simulated. Atmospheric reconstruction supplies missing incoming
//! populations at liquid/gas interfaces. No surface tension is modeled.
//!
//! Cell types in `state` (f32 codes to avoid casts in the expression API):
//! 0 = gas, 1 = interface, 2 = liquid, 3 = solid.
//!
//! Required invariant: liquid must not directly neighbor gas; maintain at least
//! one interface cell between them. Exterior perimeter cells must be SOLID.
//!
//! Input populations of gas/solid cells are inert placeholder distributions.
//! Initialize active populations to D2Q9 equilibrium at rho=1, u=(0,0).
//! For liquid cells mass=rho; interface cells mass=epsilon*rho; elsewhere 0.
//!
//! Force inputs are accelerations in lattice units; +y is downward.
//! `omega` must be in (0,2). Keep Mach number small (|u| << 1/sqrt(3)).

const zgc = @import("zgc");
const B = zgc.DefinitionBuilder;
const E = zgc.Expr;

pub const H: usize = 180;
pub const W: usize = 320;

pub const Cell = struct {
    pub const gas: f32 = 0;
    pub const interface: f32 = 1;
    pub const liquid: f32 = 2;
    pub const solid: f32 = 3;
};

pub const Sources = enum(usize) {
    f, mass, state, force_x, force_y, omega,
};

const cx = [_]isize{ 0, 1, 0, -1, 0, 1, -1, -1, 1 };
const cy = [_]isize{ 0, 0, 1, 0, -1, 1, 1, -1, -1 };
const opposite = [_]usize{ 0, 3, 4, 1, 2, 7, 8, 5, 6 };
const weights = [_]f32{ 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0 };

fn scalar(b: *B, comptime x: f32) E {
    return b.expr(b.scalar(.f32, x));
}
fn component(comptime a: E, comptime i: usize) E {
    return a.slice(.{ .axis = 2, .start = i, .end = i + 1 });
}
fn direction(comptime a: E, comptime i: usize) E {
    return a.shift(&.{ cy[i], cx[i], 0 }, .edge);
}
fn neighbor(comptime a: E, comptime i: usize) E {
    return a.shift(&.{ -cy[i], -cx[i], 0 }, .edge);
}
fn boolF(comptime a: E, comptime one: E, comptime zero: E) E {
    return a.where(one, zero);
}
/// Shared cell-local moments, reused across all nine lattice directions.
const Moments = struct {
    rho: E,
    ux: E,
    uy: E,
    uv2: E,
};

fn moments(comptime rho: E, comptime ux: E, comptime uy: E) Moments {
    return .{ .rho = rho, .ux = ux, .uy = uy, .uv2 = ux.mul(ux).add(uy.mul(uy)) };
}

fn dotDirection(comptime ux: E, comptime uy: E, comptime i: usize, comptime b: *B) E {
    return ux.mul(scalar(b, @floatFromInt(cx[i])))
        .add(uy.mul(scalar(b, @floatFromInt(cy[i]))));
}

fn equilibrium(comptime m: Moments, comptime i: usize, comptime b: *B) E {
    const eu = dotDirection(m.ux, m.uy, i, b);
    return m.rho.mul(scalar(b, weights[i])).mul(
        scalar(b, 1).add(scalar(b, 3).mul(eu))
            .add(scalar(b, 4.5).mul(eu.mul(eu)))
            .sub(scalar(b, 1.5).mul(m.uv2)),
    );
}

/// Sum of opposite atmospheric equilibrium populations. Linear terms cancel.
fn atmosphericPair(comptime ux: E, comptime uy: E, comptime uv2: E, comptime i: usize, comptime b: *B) E {
    const eu = dotDirection(ux, uy, i, b);
    return scalar(b, 2 * weights[i]).mul(
        scalar(b, 1).add(scalar(b, 4.5).mul(eu.mul(eu)))
            .sub(scalar(b, 1.5).mul(uv2)),
    );
}

fn combine9(comptime a: [9]E) E {
    var result = a[0];
    inline for (1..9) |i| result = result.concat(a[i], 2);
    return result;
}

fn build(b: *B) void {
    const s = b.sources(Sources);
    const f = b.expr(s.input(.f, .f32, &.{ H, W, 9 }));
    const mass = b.expr(s.input(.mass, .f32, &.{ H, W, 1 }));
    const state = b.expr(s.input(.state, .f32, &.{ H, W, 1 }));
    const ax = b.expr(s.input(.force_x, .f32, &.{ H, W, 1 }));
    const ay = b.expr(s.input(.force_y, .f32, &.{ H, W, 1 }));
    const omega = b.expr(s.input(.omega, .f32, &.{1}));

    const zero = scalar(b, 0);
    const one = scalar(b, 1);
    const half = scalar(b, 0.5);
    const eps = scalar(b, 1.0e-6);
    const rho_air = one; // p_atm = c_s^2 * rho_air
    const gas = state.equal(scalar(b, Cell.gas));
    const iface = state.equal(scalar(b, Cell.interface));
    const liquid = state.equal(scalar(b, Cell.liquid));
    const solid = state.equal(scalar(b, Cell.solid));
    const active = liquid.logicalOr(iface);

    // Recover conserved moments. Gas populations must not be used as fluid.
    const rho_raw = f.sum(.{ .axes = &.{2}, .keep_dims = true });
    const rho = active.where(rho_raw, rho_air);
    const safe_rho = rho.maximum(eps);
    var px = zero;
    var py = zero;
    inline for (1..9) |i| {
        const fi = component(f, i);
        px = px.add(fi.mul(scalar(b, @floatFromInt(cx[i]))));
        py = py.add(fi.mul(scalar(b, @floatFromInt(cy[i]))));
    }
    // Guo forcing: half-step velocity correction and explicit source term.
    const ux = px.add(half.mul(rho.mul(ax))).div(safe_rho);
    const uy = py.add(half.mul(rho.mul(ay))).div(safe_rho);
    const local = moments(rho, ux, uy);
    const force_prefactor = one.sub(half.mul(omega));
    const u_dot_a = ux.mul(ax).add(uy.mul(ay));

    var post: [9]E = undefined;
    inline for (0..9) |i| {
        const fi = component(f, i);
        const eu = dotDirection(ux, uy, i, b);
        const ea = dotDirection(ax, ay, i, b);
        // Guo: (1 - omega/2) w_i rho [3(e_i.a - u.a) + 9(e_i.u)(e_i.a)]
        const source = force_prefactor.mul(scalar(b, weights[i])).mul(rho).mul(
            scalar(b, 3).mul(ea.sub(u_dot_a))
                .add(scalar(b, 9).mul(eu.mul(ea))),
        );
        post[i] = active.where(fi.add(omega.mul(equilibrium(local, i, b).sub(fi))).add(source), fi);
    }

    // Pull streaming: at walls, bounce back the local outgoing population.
    // At gas-to-interface links, reconstruct incoming distribution using
    // rho_air and local interface velocity.
    var streamed: [9]E = undefined;
    streamed[0] = post[0];
    inline for (1..9) |i| {
        const upstream_state = direction(state, i);
        const upstream_solid = upstream_state.equal(scalar(b, Cell.solid));
        const upstream_gas = upstream_state.equal(scalar(b, Cell.gas));
        const incoming = direction(post[i], i);
        const bounce = post[opposite[i]];
        const atmosphere = atmosphericPair(ux, uy, local.uv2, i, b)
            .sub(post[opposite[i]]);
        streamed[i] = active.where(
            upstream_solid.where(bounce,
                iface.logicalAnd(upstream_gas).where(atmosphere, incoming)),
            post[i],
        );
    }

    // Conservative link mass exchange. Each undirected link's two cells
    // receive equal-and-opposite contributions (subject to valid state flags).
    // Neighbor at x+c_i sends its opposite distribution back toward x.
    const fill = mass.div(safe_rho).clamp(zero, one);
    var mass_delta = zero;
    inline for (1..9) |i| {
        const neighbor_state = neighbor(state, i);
        const neighbor_active = neighbor_state.equal(scalar(b, Cell.liquid))
            .logicalOr(neighbor_state.equal(scalar(b, Cell.interface)));
        const neighbor_fill = neighbor(fill, i);
        const both_iface = iface.logicalAnd(neighbor_state.equal(scalar(b, Cell.interface)));
        const factor = both_iface.where(half.mul(fill.add(neighbor_fill)), one);
        const flux = neighbor(post[opposite[i]], i).sub(post[i]);
        const usable = active.logicalAnd(neighbor_active);
        mass_delta = mass_delta.add(usable.where(factor.mul(flux), zero));
    }
    const tentative_mass = mass.add(mass_delta);

    // Candidate conversions, with a small hysteresis around 0 and rho.
    const wants_full = iface.logicalAnd(tentative_mass.greaterEqual(rho.sub(eps)));
    const wants_empty = iface.logicalAnd(tentative_mass.lessEqual(eps));

    // Only convert when a recipient is guaranteed to remain an interface.
    // For a filling cell: an adjacent gas cell will become interface, or a
    // stable interface cell remains available. Emptying is symmetric.
    // This is deliberately conservative and may delay difficult transitions.
    const stable_iface = iface.logicalAnd(wants_full.logicalNot())
        .logicalAnd(wants_empty.logicalNot());
    const full_recipient = gas.logicalOr(stable_iface);
    const empty_recipient = liquid.logicalOr(stable_iface);
    var can_fill = zero.greaterThan(one);
    var can_empty = zero.greaterThan(one);
    inline for (1..9) |i| {
        can_fill = can_fill.logicalOr(neighbor(full_recipient, i));
        can_empty = can_empty.logicalOr(neighbor(empty_recipient, i));
    }
    const full = wants_full.logicalAnd(can_fill);
    const empty = wants_empty.logicalAnd(can_empty);
    var near_full = full;
    var near_empty = empty;
    inline for (1..9) |i| {
        near_full = near_full.logicalOr(direction(full, i));
        near_empty = near_empty.logicalOr(direction(empty, i));
    }
    const gas_to_iface = gas.logicalAnd(near_full);
    const liquid_to_iface = liquid.logicalAnd(near_empty);
    const next_gas = gas.logicalAnd(gas_to_iface.logicalNot()).logicalOr(empty);
    const next_liquid = liquid.logicalAnd(liquid_to_iface.logicalNot()).logicalOr(full);
    const next_iface = active.logicalOr(gas_to_iface)
        .logicalAnd(next_liquid.logicalNot())
        .logicalAnd(next_gas.logicalNot())
        .logicalAnd(solid.logicalNot());

    // Count the actual next-interface recipients. The conversion guards
    // above guarantee at least one for every converting cell.
    var recipients = zero;
    inline for (1..9) |i| {
        recipients = recipients.add(boolF(neighbor(next_iface, i), one, zero));
    }
    const excess = full.where(tentative_mass.sub(rho), empty.where(tentative_mass, zero));
    const share = excess.div(recipients.maximum(one));
    var received = zero;
    inline for (1..9) |i| received = received.add(direction(share, i));
    const retained = next_iface.where(
        gas_to_iface.where(zero, liquid_to_iface.where(rho, tentative_mass)),
        next_liquid.where(rho, zero),
    );
    const next_mass = retained.add(next_iface.where(received, zero));

    // Initialize populations in newly activated gas cells using the averaged
    // neighboring active-cell moments. No invalid gas distributions are read.
    var rho_sum = zero;
    var ux_sum = zero;
    var uy_sum = zero;
    var count = zero;
    inline for (1..9) |i| {
        const n_active = neighbor(active, i);
        const weight = boolF(n_active, one, zero);
        rho_sum = rho_sum.add(neighbor(rho, i).mul(weight));
        ux_sum = ux_sum.add(neighbor(ux, i).mul(weight));
        uy_sum = uy_sum.add(neighbor(uy, i).mul(weight));
        count = count.add(weight);
    }
    const n_safe = count.maximum(one);
    const seed_rho = rho_sum.div(n_safe).maximum(eps);
    const seed_ux = ux_sum.div(n_safe);
    const seed_uy = uy_sum.div(n_safe);
    const seed_moments = moments(seed_rho, seed_ux, seed_uy);
    var final_f: [9]E = undefined;
    inline for (0..9) |i| {
        const seed = equilibrium(seed_moments, i, b);
        final_f[i] = gas_to_iface.where(seed, streamed[i]);
    }
    const next_f = combine9(final_f);

    const next_state = solid.where(scalar(b, Cell.solid),
        next_liquid.where(scalar(b, Cell.liquid),
            next_iface.where(scalar(b, Cell.interface), scalar(b, Cell.gas))));
    b.output(next_f.value);
    b.output(next_mass.value);
    b.output(next_state.value);
    const next_rho = next_f.sum(.{ .axes = &.{2}, .keep_dims = true }).maximum(eps);
    b.output(next_mass.div(next_rho).clamp(zero, one).value);
    b.output(rho.value);
    b.output(ux.value);
    b.output(uy.value);
}

pub const definition = blk: {
    @setEvalBranchQuota(5_000_000);
    var builder = B.init();
    build(&builder);
    break :blk builder.finish();
};
// comptime {
//     @compileLog("Definition built: {any}", .{definition});
// }

pub const FreeSurfaceStep = blk: {
    @setEvalBranchQuota(5_000_000);
    break :blk definition.executable();
};

// pub const FreeSurfaceStep = {
//     @setEvalBranchQuota(5_000_000);
// 
//     const graph = definition;
// 
//     if (@TypeOf(graph) == void)
//         @compileError("unexpected graph");
// 
    // @compileError("Graph definition completed");
// };

/// Host-side initializer for a partially filled, stationary container.
/// Arrays are logical row-major. All perimeter cells are solid.
/// `waterline` is a continuous Y coordinate: cells at larger Y are submerged.
/// The one-cell fractional layer is the initial liquid/gas interface.
pub fn initialize(
    f: *[H * W * 9]f32,
    mass: *[H * W]f32,
    state: *[H * W]f32,
    waterline: f32,
) void {
    for (0..H) |y| {
        for (0..W) |x| {
            const idx = y * W + x;
            const wall = x == 0 or y == 0 or x == W - 1 or y == H - 1;
            const cell_top: f32 = @floatFromInt(y);
            const fraction = @max(0.0, @min(1.0, cell_top + 1.0 - waterline));
            state[idx] = if (wall) Cell.solid else if (fraction <= 0.0) Cell.gas else if (fraction >= 1.0) Cell.liquid else Cell.interface;
            mass[idx] = if (wall) 0.0 else fraction;
            for (weights, 0..) |w, i| f[idx * 9 + i] = w;
        }
    }
}
