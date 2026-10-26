//! D2Q9 single-phase free-surface LBM (2D, lattice units).
//!
//! Each call advances populations, liquid mass, and cell types in lockstep.
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
    f,
    mass,
    state,
    force_x,
    force_y,
    omega,
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
fn equilibrium(comptime rho: E, comptime ux: E, comptime uy: E, comptime i: usize, comptime b: *B) E {
    const one = scalar(b, 1.0);
    const three = scalar(b, 3.0);
    const half3 = scalar(b, 1.5);
    const half9 = scalar(b, 4.5);
    const dot = ux.mul(scalar(b, @floatFromInt(cx[i]))).add(uy.mul(scalar(b, @floatFromInt(cy[i]))));
    const uv2 = ux.mul(ux).add(uy.mul(uy));
    return rho.mul(scalar(b, weights[i])).mul(one.add(three.mul(dot)).add(half9.mul(dot.mul(dot))).sub(half3.mul(uv2)));
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
    const force_prefactor = one.sub(half.mul(omega));

    var post: [9]E = undefined;
    inline for (0..9) |i| {
        const fi = component(f, i);
        const feq = equilibrium(rho, ux, uy, i, b);
        const ex = scalar(b, @floatFromInt(cx[i]));
        const ey = scalar(b, @floatFromInt(cy[i]));
        const eu = ux.mul(ex).add(uy.mul(ey));
        // S_i = (1-omega/2) * w_i * rho *
        //       [3(e_i - u) + 9(e_i.u)e_i] . a
        const fx = scalar(b, 3).mul(ex.sub(ux)).add(scalar(b, 9).mul(eu).mul(ex));
        const fy = scalar(b, 3).mul(ey.sub(uy)).add(scalar(b, 9).mul(eu).mul(ey));
        const source = force_prefactor.mul(scalar(b, weights[i])).mul(rho).mul(fx.mul(ax).add(fy.mul(ay)));
        post[i] = active.where(fi.add(omega.mul(feq.sub(fi))).add(source), fi);
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
        const atmosphere = equilibrium(rho_air, ux, uy, i, b)
            .add(equilibrium(rho_air, ux, uy, opposite[i], b))
            .sub(post[opposite[i]]);
        streamed[i] = active.where(
            upstream_solid.where(bounce, iface.logicalAnd(upstream_gas).where(atmosphere, incoming)),
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

    // Mark interface cells that have filled or emptied. A small hysteresis
    // guards against numeric chatter around 0 and rho.
    const full = iface.logicalAnd(tentative_mass.greaterEqual(rho.sub(eps)));
    const empty = iface.logicalAnd(tentative_mass.lessEqual(eps));

    // Neighbors of conversions join the interface. This maintains a one-cell
    // transition layer (assumes valid initial cell topology).
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

    // Excess/deficit mass from converted interface cells is redistributed
    // uniformly to adjacent interface cells, including freshly created ones.
    // All eight neighbors participate. No mass is discarded by clipping.
    const excess = full.where(tentative_mass.sub(rho), empty.where(tentative_mass, zero));
    var recipients = zero;
    inline for (1..9) |i| {
        const eligible = neighbor(next_iface, i);
        recipients = recipients.add(boolF(eligible, one, zero));
    }
    const share = excess.div(recipients.maximum(one));
    // The source divides its excess among all next-step interface neighbors.
    // Each target collects shares from its eight adjacent source cells.
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
    var final_f: [9]E = undefined;
    inline for (0..9) |i| {
        const seed = equilibrium(seed_rho, seed_ux, seed_uy, i, b);
        final_f[i] = gas_to_iface.where(seed, streamed[i]);
    }
    const next_f = combine9(final_f);

    const next_state = solid.where(scalar(b, Cell.solid), next_liquid.where(scalar(b, Cell.liquid), next_iface.where(scalar(b, Cell.interface), scalar(b, Cell.gas))));
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

pub const FreeSurfaceStep = blk: {
    @setEvalBranchQuota(5_000_000);
    break :blk definition.executable();
};

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
