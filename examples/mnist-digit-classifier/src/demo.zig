const std = @import("std");
const rl = @import("raylib");
const classifier = @import("digit-classifier.zig");

const screen_width = 1000;
const screen_height = 600;
const canvas_width = 600;
const resolution = 28;
const cell_size = 21;
const grid_size = resolution * cell_size;
const grid_origin_x = (canvas_width - grid_size) / 2;
const grid_origin_y = (screen_height - grid_size) / 2;

pub fn main() void {
    rl.setTraceLogLevel(.err);
    rl.initWindow(screen_width, screen_height, "ZGC digit classifier");
    defer rl.closeWindow();
    rl.setTargetFPS(120);

    var model = classifier.Model.init();
    var canvas: [classifier.input_size]f32 = @splat(0);
    var predictions: [classifier.output_size]f32 = @splat(0);

    classifier.bindInput(&model, &canvas);
    updatePredictions(&model, &predictions);

    while (!rl.windowShouldClose()) {
        if (rl.isKeyPressed(.enter) or rl.isKeyPressed(.kp_enter)) {
            canvas = @splat(0);
            updatePredictions(&model, &predictions);
        }

        if (rl.isMouseButtonDown(.left)) {
            const mouse = rl.getMousePosition();
            const local_x = mouse.x - @as(f32, @floatFromInt(grid_origin_x));
            const local_y = mouse.y - @as(f32, @floatFromInt(grid_origin_y));
            if (local_x >= 0 and local_y >= 0 and
                local_x < @as(f32, @floatFromInt(grid_size)) and
                local_y < @as(f32, @floatFromInt(grid_size)))
            {
                const x: usize = @intFromFloat(local_x / @as(f32, @floatFromInt(cell_size)));
                const y: usize = @intFromFloat(local_y / @as(f32, @floatFromInt(cell_size)));
                paint(&canvas, x, y, 0.34);
                if (x > 0) paint(&canvas, x - 1, y, 0.10);
                if (x + 1 < resolution) paint(&canvas, x + 1, y, 0.10);
                if (y > 0) paint(&canvas, x, y - 1, 0.10);
                if (y + 1 < resolution) paint(&canvas, x, y + 1, 0.10);
                updatePredictions(&model, &predictions);
            }
        }

        rl.beginDrawing();
        defer rl.endDrawing();
        rl.clearBackground(.ray_white);
        drawCanvas(&canvas);
        drawDashboard(&predictions);
    }
}

fn paint(canvas: *[classifier.input_size]f32, x: usize, y: usize, amount: f32) void {
    const index = y * resolution + x;
    canvas[index] = @min(canvas[index] + amount, 1.0);
}

fn updatePredictions(
    model: *classifier.Model,
    predictions: *[classifier.output_size]f32,
) void {
    model.run();
    const output = model.outputView(0);
    for (predictions, 0..) |*prediction, digit| {
        prediction.* = output.get(.{ 0, digit });
    }
}

fn drawCanvas(canvas: *const [classifier.input_size]f32) void {
    rl.drawRectangle(0, 0, canvas_width, screen_height, .black);
    for (0..resolution) |y| {
        for (0..resolution) |x| {
            const value = canvas[y * resolution + x];
            const channel: u8 = @intFromFloat(value * 255.0);
            const color = rl.Color.init(channel, channel, channel, 255);
            rl.drawRectangle(
                grid_origin_x + @as(i32, @intCast(x * cell_size)),
                grid_origin_y + @as(i32, @intCast(y * cell_size)),
                cell_size,
                cell_size,
                color,
            );
        }
    }
}

fn drawDashboard(predictions: *const [classifier.output_size]f32) void {
    const dashboard_x = canvas_width;
    const dashboard_width = screen_width - canvas_width;
    const bar_x = dashboard_x + 64;
    const bar_width = 230;
    const bar_height = 22;
    const best = bestDigit(predictions);

    rl.drawRectangle(dashboard_x, 0, dashboard_width, screen_height, .dark_brown);
    rl.drawText("Draw a digit", dashboard_x + 32, 24, 32, .white);
    rl.drawText("Enter clears the canvas", dashboard_x + 32, 62, 20, .light_gray);

    var prediction_buffer: [32]u8 = undefined;
    const prediction_text = std.mem.printSentinel(
        &prediction_buffer,
        "Prediction: {d}",
        .{best},
        0,
    ) catch unreachable;
    rl.drawText(prediction_text, dashboard_x + 32, 102, 28, .lime);

    for (predictions, 0..) |probability, digit| {
        const y: i32 = 155 + @as(i32, @intCast(digit)) * 40;
        const clamped = std.math.clamp(probability, 0.0, 1.0);
        const fill_width: i32 = @intFromFloat(clamped * @as(f32, @floatFromInt(bar_width)));

        rl.drawRectangle(bar_x, y, bar_width, bar_height, .dark_gray);
        rl.drawRectangle(bar_x, y, fill_width, bar_height, if (digit == best) .lime else .gray);

        var digit_buffer: [4]u8 = undefined;
        const digit_text = std.mem.printSentinel(&digit_buffer, "{d}", .{digit}, 0) catch unreachable;
        rl.drawText(digit_text, bar_x - 30, y - 4, 26, .white);

        var probability_buffer: [24]u8 = undefined;
        const probability_text = std.mem.printSentinel(
            &probability_buffer,
            "{d:.1}%",
            .{probability * 100.0},
            0,
        ) catch unreachable;
        rl.drawText(probability_text, bar_x + bar_width + 12, y, 18, .white);
    }
}

fn bestDigit(predictions: *const [classifier.output_size]f32) usize {
    var best: usize = 0;
    for (predictions[1..], 1..) |probability, digit| {
        if (probability > predictions[best]) best = digit;
    }
    return best;
}
