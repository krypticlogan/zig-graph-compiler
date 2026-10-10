const std = @import("std");
const rl = @import("raylib");
const fluid = @import("fluid_model");

const cell_size = 3;
const header_height = 108;
const arrow_spacing = 12;
const screen_width: i32 = fluid.W * cell_size;
const screen_height: i32 = fluid.H * cell_size + header_height;
const population_count = fluid.H * fluid.W * 9;
const smoke_population_count = fluid.H * fluid.W * 5;
const cell_count = fluid.H * fluid.W;
const fluid_weights = [9]f32{ 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0 };
const smoke_weights = [5]f32{ 1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0 };
const cx = [9]f32{ 0, 1, 0, -1, 0, 1, -1, -1, 1 };
const cy = [9]f32{ 0, 0, 1, 0, -1, 1, 1, -1, -1 };
const Field = enum { speed, density };
const View = enum { smoke, data };

pub fn main() void {
    rl.setTraceLogLevel(.err);
    rl.initWindow(screen_width, screen_height, "ZGC D2Q9 fluid dynamics");
    defer rl.closeWindow();
    rl.setTargetFPS(60);

    var model = fluid.FluidSmokeStep.init();
    model.copySource(.cx, &cx) catch unreachable;
    model.copySource(.cy, &cy) catch unreachable;
    model.copySource(.fluid_weights, &fluid_weights) catch unreachable;
    model.copySource(.smoke_weights, &smoke_weights) catch unreachable;

    var populations: [population_count]f32 = undefined;
    var smoke_populations: [smoke_population_count]f32 = @splat(0);
    var smoke_injection: [cell_count]f32 = undefined;
    var force_x: [cell_count]f32 = @splat(0);
    var force_y: [cell_count]f32 = @splat(0);
    seedVortices(&populations);
    seedSmokeEmitter(&smoke_injection);
    var omega: f32 = 1.0;
    const smoke_omega: f32 = 1.2;
    const smoke_retention: f32 = 0.997;
    advance(&model, &populations, &smoke_populations, &smoke_injection, &force_x, &force_y, omega, smoke_omega, smoke_retention);

    var running = true;
    var step_count: u64 = 1;
    var field: Field = .speed;
    var view: View = .smoke;
    var show_vectors = false;
    var previous_mouse = rl.getMousePosition();
    var was_dragging = false;

    while (!rl.windowShouldClose()) {
        updateForces(&force_x, &force_y, &previous_mouse, &was_dragging);
        if (rl.isKeyPressed(.space)) running = !running;
        if (rl.isKeyPressed(.v)) field = if (field == .speed) .density else .speed;
        if (rl.isKeyPressed(.p)) view = if (view == .smoke) .data else .smoke;
        if (rl.isKeyPressed(.a)) show_vectors = !show_vectors;
        if (rl.isKeyPressed(.left_bracket)) omega = std.math.clamp(omega - 0.05, 0.6, 1.7);
        if (rl.isKeyPressed(.right_bracket)) omega = std.math.clamp(omega + 0.05, 0.6, 1.7);
        if (rl.isKeyPressed(.r)) {
            seedVortices(&populations);
            smoke_populations = @splat(0);
            force_x = @splat(0);
            force_y = @splat(0);
            advance(&model, &populations, &smoke_populations, &smoke_injection, &force_x, &force_y, omega, smoke_omega, smoke_retention);
            step_count = 1;
        } else if (rl.isKeyPressed(.n)) {
            advance(&model, &populations, &smoke_populations, &smoke_injection, &force_x, &force_y, omega, smoke_omega, smoke_retention);
            step_count += 1;
            running = false;
        } else if (running) {
            advance(&model, &populations, &smoke_populations, &smoke_injection, &force_x, &force_y, omega, smoke_omega, smoke_retention);
            step_count += 1;
        }

        const smoke = model.outputView(2).contiguousSlice().?;
        const density = model.outputView(3).contiguousSlice().?;
        const velocity_x = model.outputView(4).contiguousSlice().?;
        const velocity_y = model.outputView(5).contiguousSlice().?;

        rl.beginDrawing();
        defer rl.endDrawing();
        rl.clearBackground(rl.Color.init(10, 15, 23, 255));
        drawHeader(running, step_count, view, field, show_vectors, omega);
        switch (view) {
            .smoke => drawSmoke(smoke),
            .data => {
                drawField(field, density, velocity_x, velocity_y);
            },
        }
        if (show_vectors) drawVectors(velocity_x, velocity_y);
    }
}

fn advance(
    model: *fluid.FluidSmokeStep,
    populations: *[population_count]f32,
    smoke_populations: *[smoke_population_count]f32,
    smoke_injection: *const [cell_count]f32,
    force_x: *const [cell_count]f32,
    force_y: *const [cell_count]f32,
    omega: f32,
    smoke_omega: f32,
    smoke_retention: f32,
) void {
    model.copyInput(.f, populations) catch unreachable;
    model.copyInput(.smoke_g, smoke_populations) catch unreachable;
    model.copyInput(.smoke_injection, smoke_injection) catch unreachable;
    model.copyInput(.force_x, force_x) catch unreachable;
    model.copyInput(.force_y, force_y) catch unreachable;
    model.copyInput(.omega, &.{omega}) catch unreachable;
    model.copyInput(.smoke_omega, &.{smoke_omega}) catch unreachable;
    model.copyInput(.smoke_retention, &.{smoke_retention}) catch unreachable;
    model.run();
    @memcpy(populations, model.outputView(0).contiguousSlice().?);
    @memcpy(smoke_populations, model.outputView(1).contiguousSlice().?);
}

fn updateForces(
    force_x: *[cell_count]f32,
    force_y: *[cell_count]f32,
    previous_mouse: *rl.Vector2,
    was_dragging: *bool,
) void {
    force_x.* = @splat(0);
    force_y.* = @splat(0);

    const mouse = rl.getMousePosition();
    const dragging = rl.isMouseButtonDown(.left);
    const grid_y = mouse.y - @as(f32, @floatFromInt(header_height));
    const inside = mouse.x >= 0 and mouse.x < @as(f32, @floatFromInt(screen_width)) and
        grid_y >= 0 and grid_y < @as(f32, @floatFromInt(fluid.H * cell_size));

    if (dragging and was_dragging.* and inside) {
        const grid_dx = (mouse.x - previous_mouse.x) / cell_size;
        const grid_dy = (mouse.y - previous_mouse.y) / cell_size;
        const impulse_x = std.math.clamp(grid_dx * 0.012, -0.05, 0.05);
        const impulse_y = std.math.clamp(grid_dy * 0.012, -0.05, 0.05);
        applyLocalizedForce(force_x, force_y, mouse.x / cell_size, grid_y / cell_size, impulse_x, impulse_y);
    }

    previous_mouse.* = mouse;
    was_dragging.* = dragging and inside;
}

fn applyLocalizedForce(
    force_x: *[cell_count]f32,
    force_y: *[cell_count]f32,
    center_x: f32,
    center_y: f32,
    impulse_x: f32,
    impulse_y: f32,
) void {
    for (0..fluid.H) |y| {
        for (0..fluid.W) |x| {
            const xf: f32 = @floatFromInt(x);
            const yf: f32 = @floatFromInt(y);
            const dx = xf - center_x;
            const dy = yf - center_y;
            const falloff = @exp(-(dx * dx + dy * dy) / 64.0);
            const index = y * fluid.W + x;
            force_x[index] = impulse_x * falloff;
            force_y[index] = impulse_y * falloff;
        }
    }
}

fn seedVortices(populations: *[population_count]f32) void {
    const center_y: f32 = @as(f32, @floatFromInt(fluid.H)) * 0.5;
    const left_x: f32 = @as(f32, @floatFromInt(fluid.W)) * 0.3;
    const right_x: f32 = @as(f32, @floatFromInt(fluid.W)) * 0.7;
    const radius_sq: f32 = 18.0 * 18.0;

    for (0..fluid.H) |y| {
        for (0..fluid.W) |x| {
            const xf: f32 = @floatFromInt(x);
            const yf: f32 = @floatFromInt(y);
            const dy = yf - center_y;
            const left_dx = xf - left_x;
            const right_dx = xf - right_x;
            const left_envelope = @exp(-(left_dx * left_dx + dy * dy) / radius_sq);
            const right_envelope = @exp(-(right_dx * right_dx + dy * dy) / radius_sq);
            const ux = 0.008 * dy * (right_envelope - left_envelope);
            const uy = 0.008 * (left_dx * left_envelope - right_dx * right_envelope);
            const speed_sq = ux * ux + uy * uy;

            for (0..9) |direction| {
                const dot = cx[direction] * ux + cy[direction] * uy;
                populations[(y * fluid.W + x) * 9 + direction] = fluid_weights[direction] *
                    (1.0 + 3.0 * dot + 4.5 * dot * dot - 1.5 * speed_sq);
            }
        }
    }
}

fn seedSmokeEmitter(injection: *[cell_count]f32) void {
    const center_x: f32 = @as(f32, @floatFromInt(fluid.W)) * 0.3;
    const center_y: f32 = @as(f32, @floatFromInt(fluid.H)) * 0.5 + 12.0;
    for (0..fluid.H) |y| {
        for (0..fluid.W) |x| {
            const dx = @as(f32, @floatFromInt(x)) - center_x;
            const dy = @as(f32, @floatFromInt(y)) - center_y;
            injection[y * fluid.W + x] = 0.035 * @exp(-(dx * dx + dy * dy) / 18.0);
        }
    }
}

fn drawHeader(running: bool, step_count: u64, view: View, field: Field, show_vectors: bool, omega: f32) void {
    rl.drawRectangle(0, 0, screen_width, header_height, rl.Color.init(22, 30, 42, 255));
    rl.drawText("D2Q9 lattice Boltzmann", 16, 8, 24, .ray_white);
    rl.drawText("Space: pause  N: step  R: reset  P: smoke/data", 16, 65, 16, .light_gray);
    rl.drawText("V: speed/density  A: vectors  Drag: stir  [ / ]: omega", 16, 88, 16, .light_gray);

    var status_buffer: [128]u8 = undefined;
    const status = std.mem.printSentinel(&status_buffer, "{s}  step {d}  {s}  {s}  vectors {s}  omega {d:.2}", .{
        if (running) "running" else "paused",
        step_count,
        if (view == .smoke) "smoke" else "data",
        if (field == .speed) "speed" else "density",
        if (show_vectors) "on" else "off",
        omega,
    }, 0) catch unreachable;
    rl.drawText(status, 16, 39, 15, if (running) .lime else .gold);
}

fn drawSmoke(smoke: []const f32) void {
    for (smoke, 0..) |concentration, index| {
        const intensity = std.math.clamp(concentration / 0.8, 0.0, 1.0);
        const shade: u8 = @intFromFloat(18.0 + 225.0 * @sqrt(intensity));
        const x = index % fluid.W;
        const y = index / fluid.W;
        rl.drawRectangle(
            @intCast(x * cell_size),
            @intCast(header_height + y * cell_size),
            cell_size,
            cell_size,
            rl.Color.init(shade, shade, @min(255, @as(u16, shade) + 8), 255),
        );
    }
}

fn drawField(field: Field, density: []const f32, ux: []const f32, uy: []const f32) void {
    for (0..cell_count) |index| {
        const color = switch (field) {
            .speed => speedColor(@sqrt(ux[index] * ux[index] + uy[index] * uy[index])),
            .density => densityColor(density[index]),
        };
        const x = index % fluid.W;
        const y = index / fluid.W;
        rl.drawRectangle(@intCast(x * cell_size), @intCast(header_height + y * cell_size), cell_size, cell_size, color);
    }
}

fn drawVectors(ux: []const f32, uy: []const f32) void {
    const color = rl.Color.init(245, 248, 255, 220);
    const cell_size_f: f32 = @floatFromInt(cell_size);
    const header_f: f32 = @floatFromInt(header_height);
    for (0..fluid.H / arrow_spacing) |row| {
        for (0..fluid.W / arrow_spacing) |column| {
            const x = column * arrow_spacing + arrow_spacing / 2;
            const y = row * arrow_spacing + arrow_spacing / 2;
            const index = y * fluid.W + x;
            const vx = ux[index];
            const vy = uy[index];
            const speed = @sqrt(vx * vx + vy * vy);
            if (speed < 0.002) continue;

            const length = std.math.clamp(speed * 140.0, 2.0, 17.0);
            const dx = vx / speed * length;
            const dy = vy / speed * length;
            const center = rl.Vector2.init((@as(f32, @floatFromInt(x)) + 0.5) * cell_size_f, header_f + (@as(f32, @floatFromInt(y)) + 0.5) * cell_size_f);
            const tip = rl.Vector2.init(center.x + dx, center.y + dy);
            const head_length: f32 = 4.0;
            const side_x = -dy / length * head_length;
            const side_y = dx / length * head_length;
            const back_x = dx / length * head_length;
            const back_y = dy / length * head_length;

            rl.drawLineEx(center, tip, 1.5, color);
            rl.drawLineEx(tip, rl.Vector2.init(tip.x - back_x + side_x, tip.y - back_y + side_y), 1.5, color);
            rl.drawLineEx(tip, rl.Vector2.init(tip.x - back_x - side_x, tip.y - back_y - side_y), 1.5, color);
        }
    }
}

fn speedColor(speed: f32) rl.Color {
    return colorRamp(std.math.clamp(speed / 0.12, 0.0, 1.0));
}

fn densityColor(density: f32) rl.Color {
    return colorRamp(std.math.clamp(0.5 + (density - 1.0) * 8.0, 0.0, 1.0));
}

fn colorRamp(t: f32) rl.Color {
    const cold = rl.Color.init(17, 42, 88, 255);
    const middle = rl.Color.init(35, 190, 213, 255);
    const hot = rl.Color.init(255, 224, 97, 255);
    return if (t < 0.5) lerpColor(cold, middle, t * 2.0) else lerpColor(middle, hot, (t - 0.5) * 2.0);
}

fn lerpColor(a: rl.Color, b: rl.Color, t: f32) rl.Color {
    return rl.Color.init(mixChannel(a.r, b.r, t), mixChannel(a.g, b.g, t), mixChannel(a.b, b.b, t), 255);
}

fn mixChannel(a: u8, b: u8, t: f32) u8 {
    const start: f32 = @floatFromInt(a);
    const end: f32 = @floatFromInt(b);
    return @intFromFloat(start + (end - start) * t);
}
