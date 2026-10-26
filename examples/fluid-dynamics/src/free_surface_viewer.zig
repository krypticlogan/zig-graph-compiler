//! Raylib frontend for free_surface_lbm.zig.
//! Build with your existing Raylib example build, mapping @import("fluid_model")
//! to the free-surface graph module.
const std = @import("std");
const rl = @import("raylib");
const fluid = @import("fluid_model");

const cell_size = 3;
const header_height = 108;
const arrow_spacing = 12;
const screen_width: i32 = fluid.W * cell_size;
const screen_height: i32 = fluid.H * cell_size + header_height;
const cell_count = fluid.H * fluid.W;
const population_count = cell_count * 9;
const Runtime = fluid.FreeSurfaceStep.Runtime;
const View = enum { water, speed };
const background = rl.Color.init(10, 15, 23, 255);
const water = rl.Color.init(35, 135, 226, 255);
const wall_color = rl.Color.init(65, 73, 88, 255);

pub fn main() void {
    rl.setTraceLogLevel(.err);
    rl.initWindow(screen_width, screen_height, "ZGC free-surface D2Q9");
    defer rl.closeWindow();
    rl.setTargetFPS(60);

    var runtime = Runtime.init();
    var populations: [population_count]f32 = undefined;
    var mass: [cell_count]f32 = undefined;
    var state: [cell_count]f32 = undefined;
    var force_x: [cell_count]f32 = @splat(0);
    var force_y: [cell_count]f32 = @splat(0);

    var omega: f32 = 1.0;
    var gravity: f32 = 0.0001; // lattice acceleration per step, +Y down
    var gravity_enabled = true;
    var running = true;
    var show_vectors = false;
    var view: View = .water;
    var step_count: u64 = 0;
    var previous_mouse = rl.getMousePosition();
    var was_dragging = false;

    reset(&populations, &mass, &state);
    // No initial `advance`: render the initial condition as seeded.
    // The graph's inspection outputs become available after the first step.
    advance(&runtime, &populations, &mass, &state, &force_x, &force_y, omega);
    step_count += 1;

    while (!rl.windowShouldClose()) {
        if (rl.isKeyPressed(.space)) running = !running;
        if (rl.isKeyPressed(.a)) show_vectors = !show_vectors;
        if (rl.isKeyPressed(.v)) view = if (view == .water) .speed else .water;
        if (rl.isKeyPressed(.g)) gravity_enabled = !gravity_enabled;
        if (rl.isKeyPressed(.left_bracket)) omega = std.math.clamp(omega - 0.05, 0.6, 1.7);
        if (rl.isKeyPressed(.right_bracket)) omega = std.math.clamp(omega + 0.05, 0.6, 1.7);
        if (rl.isKeyPressed(.up)) gravity = @min(gravity * 1.25, 0.001);
        if (rl.isKeyPressed(.down)) gravity = @max(gravity / 1.25, 0.000001);

        updateForces(&force_x, &force_y, &state, &previous_mouse, &was_dragging,
            if (gravity_enabled) gravity else 0.0);

        if (rl.isKeyPressed(.r)) {
            reset(&populations, &mass, &state);
            previous_mouse = rl.getMousePosition();
            was_dragging = false;
            // Recompute the graph outputs from the fresh state before rendering.
            updateForces(&force_x, &force_y, &state, &previous_mouse, &was_dragging,
                if (gravity_enabled) gravity else 0.0);
            advance(&runtime, &populations, &mass, &state, &force_x, &force_y, omega);
            step_count = 1;
        } else if (rl.isKeyPressed(.n)) {
            advance(&runtime, &populations, &mass, &state, &force_x, &force_y, omega);
            step_count += 1;
            running = false;
        } else if (running) {
            advance(&runtime, &populations, &mass, &state, &force_x, &force_y, omega);
            step_count += 1;
        }

        // Output 3 is the UPDATED fill fraction; 5/6 are pre-step velocities.
        // Read fields only after `run()`; copy recurrent outputs before reuse.
        const fill = runtime.outputView(3).contiguousSlice().?;
        const velocity_x = runtime.outputView(5).contiguousSlice().?;
        const velocity_y = runtime.outputView(6).contiguousSlice().?;

        rl.beginDrawing();
        rl.clearBackground(background);
        drawHeader(running, step_count, view, show_vectors, omega, gravity, gravity_enabled);
        drawField(view, fill, &state, velocity_x, velocity_y);
        if (show_vectors) drawVectors(&state, fill, velocity_x, velocity_y);
        rl.endDrawing();
    }
}

fn reset(populations: *[population_count]f32, mass: *[cell_count]f32, state: *[cell_count]f32) void {
    const waterline: f32 = @as(f32, @floatFromInt(fluid.H)) * 0.48 + 0.35;
    fluid.initialize(populations, mass, state, waterline);
}

fn advance(
    runtime: *Runtime,
    populations: *[population_count]f32,
    mass: *[cell_count]f32,
    state: *[cell_count]f32,
    force_x: *const [cell_count]f32,
    force_y: *const [cell_count]f32,
    omega: f32,
) void {
    runtime.copyInput(.f, populations) catch unreachable;
    runtime.copyInput(.mass, mass) catch unreachable;
    runtime.copyInput(.state, state) catch unreachable;
    runtime.copyInput(.force_x, force_x) catch unreachable;
    runtime.copyInput(.force_y, force_y) catch unreachable;
    runtime.copyInput(.omega, &.{omega}) catch unreachable;
    runtime.run();
    @memcpy(populations, runtime.outputView(0).contiguousSlice().?);
    @memcpy(mass, runtime.outputView(1).contiguousSlice().?);
    @memcpy(state, runtime.outputView(2).contiguousSlice().?);
}

fn updateForces(
    force_x: *[cell_count]f32,
    force_y: *[cell_count]f32,
    state: *const [cell_count]f32,
    previous_mouse: *rl.Vector2,
    was_dragging: *bool,
    gravity: f32,
) void {
    force_x.* = @splat(0);
    for (0..cell_count) |i| {
        const active = state[i] == fluid.Cell.liquid or state[i] == fluid.Cell.interface;
        force_y[i] = if (active) gravity else 0.0;
    }

    const mouse = rl.getMousePosition();
    const grid_y = mouse.y - @as(f32, @floatFromInt(header_height));
    const dragging = rl.isMouseButtonDown(.left);
    const inside = mouse.x >= 0 and mouse.x < @as(f32, @floatFromInt(screen_width)) and
        grid_y >= 0 and grid_y < @as(f32, @floatFromInt(fluid.H * cell_size));

    if (dragging and was_dragging.* and inside) {
        const grid_dx = (mouse.x - previous_mouse.x) / cell_size;
        const grid_dy = (mouse.y - previous_mouse.y) / cell_size;
        const impulse_x = std.math.clamp(grid_dx * 0.0008, -0.004, 0.004);
        const impulse_y = std.math.clamp(grid_dy * 0.0008, -0.004, 0.004);
        applyLocalizedForce(force_x, force_y, state, mouse.x / cell_size,
            grid_y / cell_size, impulse_x, impulse_y);
    }
    previous_mouse.* = mouse;
    was_dragging.* = dragging and inside;
}

fn applyLocalizedForce(
    force_x: *[cell_count]f32,
    force_y: *[cell_count]f32,
    state: *const [cell_count]f32,
    center_x: f32,
    center_y: f32,
    impulse_x: f32,
    impulse_y: f32,
) void {
    for (0..fluid.H) |y| {
        for (0..fluid.W) |x| {
        const index = y * fluid.W + x;
        if (state[index] != fluid.Cell.liquid and state[index] != fluid.Cell.interface) continue;
        const dx = @as(f32, @floatFromInt(x)) - center_x;
        const dy = @as(f32, @floatFromInt(y)) - center_y;
        const falloff = @exp(-(dx * dx + dy * dy) / 64.0);
        force_x[index] += impulse_x * falloff;
        force_y[index] += impulse_y * falloff;
        }
    }
}

fn drawHeader(running: bool, steps: u64, view: View, vectors: bool, omega: f32, gravity: f32, gravity_enabled: bool) void {
    rl.drawRectangle(0, 0, screen_width, header_height, rl.Color.init(22, 30, 42, 255));
    rl.drawText("D2Q9 free-surface LBM", 16, 8, 24, .ray_white);
    rl.drawText("Space: pause  N: step  R: reset  V: water/speed  A: arrows", 16, 65, 16, .light_gray);
    rl.drawText("Drag: stir water  G: gravity  Up/Down: gravity  [ / ]: omega", 16, 88, 16, .light_gray);
    var buffer: [192]u8 = undefined;
    const status = std.mem.printSentinel(&buffer,
        "{s}  step {d}  {s}  arrows {s}  omega {d:.2}  gravity {s} ({d:.6})", .{
            if (running) "running" else "paused", steps,
            if (view == .water) "water" else "speed",
            if (vectors) "on" else "off", omega,
            if (gravity_enabled) "on" else "off", gravity,
        }, 0) catch unreachable;
    rl.drawText(status, 16, 39, 15, if (running) .lime else .gold);
}

fn drawField(view: View, fill: []const f32, state: *const [cell_count]f32, ux: []const f32, uy: []const f32) void {
    for (0..cell_count) |i| {
        const x: i32 = @intCast((i % fluid.W) * cell_size);
        const y: i32 = @intCast(header_height + (i / fluid.W) * cell_size);
        const color = if (state[i] == fluid.Cell.solid)
            wall_color
        else if (state[i] == fluid.Cell.gas)
            background
        else blk: {
            const coverage = std.math.clamp(fill[i], 0.0, 1.0);
            const base = if (view == .speed)
                speedColor(@sqrt(ux[i] * ux[i] + uy[i] * uy[i]))
            else
                water;
            break :blk lerpColor(background, base, coverage);
        };
        rl.drawRectangle(x, y, cell_size, cell_size, color);
    }
}

fn drawVectors(state: *const [cell_count]f32, fill: []const f32, ux: []const f32, uy: []const f32) void {
    const color = rl.Color.init(245, 248, 255, 220);
    const size: f32 = @floatFromInt(cell_size);
    const header: f32 = @floatFromInt(header_height);
    for (0..fluid.H / arrow_spacing) |row| {
        for (0..fluid.W / arrow_spacing) |column| {
            const x = column * arrow_spacing + arrow_spacing / 2;
            const y = row * arrow_spacing + arrow_spacing / 2;
            const i = y * fluid.W + x;
            if (state[i] != fluid.Cell.liquid and state[i] != fluid.Cell.interface) continue;
            if (fill[i] < 0.05) continue;
            const vx = ux[i];
            const vy = uy[i];
            const speed = @sqrt(vx * vx + vy * vy);
            if (speed < 0.0005 or !std.math.isFinite(speed)) continue;
            const length = std.math.clamp(speed * 140.0, 2.0, 17.0);
            const dx = vx / speed * length;
            const dy = vy / speed * length;
            const center = rl.Vector2.init((@as(f32, @floatFromInt(x)) + 0.5) * size,
                header + (@as(f32, @floatFromInt(y)) + 0.5) * size);
            const tip = rl.Vector2.init(center.x + dx, center.y + dy);
            const side_x = -dy / length * 4.0;
            const side_y = dx / length * 4.0;
            const back_x = dx / length * 4.0;
            const back_y = dy / length * 4.0;
            rl.drawLineEx(center, tip, 1.5, color);
            rl.drawLineEx(tip, rl.Vector2.init(tip.x - back_x + side_x, tip.y - back_y + side_y), 1.5, color);
            rl.drawLineEx(tip, rl.Vector2.init(tip.x - back_x - side_x, tip.y - back_y - side_y), 1.5, color);
        }
    }
}

fn speedColor(speed: f32) rl.Color {
    const t = std.math.clamp(speed / 0.12, 0.0, 1.0);
    const cold = rl.Color.init(17, 42, 88, 255);
    const middle = rl.Color.init(35, 190, 213, 255);
    const hot = rl.Color.init(255, 224, 97, 255);
    return if (t < 0.5) lerpColor(cold, middle, t * 2.0)
        else lerpColor(middle, hot, (t - 0.5) * 2.0);
}

fn lerpColor(a: rl.Color, b: rl.Color, t: f32) rl.Color {
    return rl.Color.init(mixChannel(a.r, b.r, t), mixChannel(a.g, b.g, t), mixChannel(a.b, b.b, t), 255);
}
fn mixChannel(a: u8, b: u8, t: f32) u8 {
    const start: f32 = @floatFromInt(a);
    const end: f32 = @floatFromInt(b);
    return @intFromFloat(start + (end - start) * std.math.clamp(t, 0.0, 1.0));
}
