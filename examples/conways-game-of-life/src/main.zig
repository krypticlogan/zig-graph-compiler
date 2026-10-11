const std = @import("std");
const rl = @import("raylib");
const life = @import("life-model.zig");

const cell_size: usize = 8;
const header_height: usize = 64;
const screen_width: i32 = @intCast(life.width * cell_size);
const screen_height: i32 = @intCast(life.height * cell_size + header_height);
const step_seconds: f32 = 0.08;

pub fn main() void {
    rl.setTraceLogLevel(.err);
    rl.initWindow(screen_width, screen_height, "ZGC Conway's Game of Life");
    defer rl.closeWindow();
    rl.setTargetFPS(120);

    var model = life.Model.init();
    var world: [life.cell_count]bool = @splat(false);
    seedWorld(&world);

    var running = true;
    var generation: u64 = 0;
    var elapsed: f32 = 0;

    while (!rl.windowShouldClose()) {
        if (rl.isKeyPressed(.space)) running = !running;
        if (rl.isKeyPressed(.c)) {
            world = @splat(false);
            generation = 0;
            running = false;
        }
        if (rl.isKeyPressed(.r)) {
            seedWorld(&world);
            generation = 0;
        }
        if (rl.isKeyPressed(.n)) {
            life.step(&model, &world);
            generation += 1;
            elapsed = 0;
        }

        editWorld(&world, &running);

        if (running) {
            elapsed += rl.getFrameTime();
            while (elapsed >= step_seconds) : (elapsed -= step_seconds) {
                life.step(&model, &world);
                generation += 1;
            }
        } else {
            elapsed = 0;
        }

        rl.beginDrawing();
        defer rl.endDrawing();
        rl.clearBackground(rl.Color.init(13, 17, 23, 255));
        drawHeader(running, generation);
        drawWorld(&world);
    }
}

fn editWorld(world: *[life.cell_count]bool, running: *bool) void {
    if (!rl.isMouseButtonDown(.left) and !rl.isMouseButtonDown(.right)) return;
    const mouse = rl.getMousePosition();
    const grid_y = mouse.y - @as(f32, @floatFromInt(header_height));
    if (mouse.x < 0 or grid_y < 0 or
        mouse.x >= @as(f32, @floatFromInt(screen_width)) or
        grid_y >= @as(f32, @floatFromInt(life.height * cell_size))) return;

    const x: usize = @intFromFloat(mouse.x / @as(f32, @floatFromInt(cell_size)));
    const y: usize = @intFromFloat(grid_y / @as(f32, @floatFromInt(cell_size)));
    world[y * life.width + x] = rl.isMouseButtonDown(.left);
    running.* = false;
}

fn drawHeader(running: bool, generation: u64) void {
    rl.drawRectangle(0, 0, screen_width, @intCast(header_height), rl.Color.init(22, 27, 34, 255));
    rl.drawText("Conway's Game of Life", 16, 10, 24, .ray_white);
    rl.drawText("Space: pause  N: step  R: reset  C: clear  Mouse: draw/erase", 16, 38, 16, .gray);

    var status_buffer: [64]u8 = undefined;
    const status = std.fmt.bufPrintSentinel(
        &status_buffer,
        "{s}  generation {d}",
        .{ if (running) "running" else "paused", generation },
        0
    ) catch unreachable;
    const status_width = rl.measureText(status, 18);
    rl.drawText(status, screen_width - status_width - 16, 14, 18, if (running) .lime else .gold);
}

fn drawWorld(world: *const [life.cell_count]bool) void {
    const live_color = rl.Color.init(103, 232, 144, 255);
    const dead_color = rl.Color.init(18, 24, 31, 255);
    for (0..life.height) |y| {
        for (0..life.width) |x| {
            const color = if (world[y * life.width + x]) live_color else dead_color;
            rl.drawRectangle(
                @intCast(x * cell_size),
                @intCast(header_height + y * cell_size),
                @intCast(cell_size - 1),
                @intCast(cell_size - 1),
                color,
            );
        }
    }
}

fn seedWorld(world: *[life.cell_count]bool) void {
    world.* = @splat(false);
    seedGlider(world, 8, 8);
    seedGlider(world, 34, 20);
    seedGlider(world, 74, 54);
    seedBlinker(world, 58, 36);
    seedBlinker(world, 92, 18);
}

fn seedGlider(world: *[life.cell_count]bool, x: usize, y: usize) void {
    setAlive(world, x + 1, y);
    setAlive(world, x + 2, y + 1);
    setAlive(world, x, y + 2);
    setAlive(world, x + 1, y + 2);
    setAlive(world, x + 2, y + 2);
}

fn seedBlinker(world: *[life.cell_count]bool, x: usize, y: usize) void {
    setAlive(world, x, y);
    setAlive(world, x, y + 1);
    setAlive(world, x, y + 2);
}

fn setAlive(world: *[life.cell_count]bool, x: usize, y: usize) void {
    world[y * life.width + x] = true;
}
