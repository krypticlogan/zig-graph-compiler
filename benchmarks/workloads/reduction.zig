const std = @import("std");
const zgc = @import("zgc");

const rows = 256;
const columns = 256;
const element_count = rows * columns;

const Sources = enum(usize) { input };
const Definition = zgc.DefinitionBuilder(Sources, .{
    .max_rank = 2,
    .max_nodes = 4,
    .max_tensors = 5,
    .max_input_refs = 4,
    .max_outputs = 4,
});

const definition = blk: {
    var builder = Definition.init();
    const input = builder.input(.input, .f32, &.{ rows, columns });
    builder.output(builder.sum(input, .{ .axes = &.{1} }));
    builder.output(builder.mean(input, .{ .axes = &.{1} }));
    builder.output(builder.min(input, .{ .axes = &.{1} }));
    builder.output(builder.max(input, .{ .axes = &.{1} }));
    break :blk builder.finish();
};

const Model = definition.modelWith(&.{.{
    .source = .input,
    .binding = zgc.memory.Source.bound,
}});

pub const ZgcBenchmark = struct {
    const Self = @This();
    pub const CompiledModel = Model;

    pub const name = "graph/reduction/sum-mean-min-max/256x256/axis1";
    pub const default_iterations = 1_000;
    pub const default_warmup_iterations = 100;
    pub const work_items_per_invocation: f64 = element_count;
    pub const work_unit = "elements";
    pub const bytes_per_invocation: f64 = @sizeOf(f32) * (element_count + 4 * rows);

    model: Model,
    input: [element_count]f32,

    pub fn init() Self {
        var result: Self = .{ .model = Model.init(), .input = undefined };
        const layout = Model.sourceLayout(.input);
        for (0..rows) |row| {
            for (0..columns) |column| {
                const offset: usize = @intCast(row * @as(usize, @intCast(layout.strides[0])) +
                    column * @as(usize, @intCast(layout.strides[1])));
                result.input[offset] = value(row, column);
            }
        }
        return result;
    }

    pub fn prepare(self: *Self) !void {
        try self.model.bindInput(.input, &self.input);
    }

    pub fn validate(self: *Self) !void {
        self.model.run();
        var expected_sum: f32 = 0;
        var expected_min = std.math.inf(f32);
        var expected_max = -std.math.inf(f32);
        for (0..columns) |column| {
            const item = value(0, column);
            expected_sum += item;
            expected_min = @min(expected_min, item);
            expected_max = @max(expected_max, item);
        }
        if (!std.math.approxEqAbs(f32, expected_sum, self.model.outputView(0).get(.{0}), 1e-4) or
            !std.math.approxEqAbs(f32, expected_sum / @as(f32, @floatFromInt(columns)), self.model.outputView(1).get(.{0}), 1e-5) or
            self.model.outputView(2).get(.{0}) != expected_min or
            self.model.outputView(3).get(.{0}) != expected_max) return error.IncorrectResult;
    }

    pub fn run(self: *Self, iterations: usize) void {
        for (0..iterations) |_| {
            self.model.run();
            inline for (0..4) |output| std.mem.doNotOptimizeAway(self.model.outputView(output).storage);
        }
    }
};

pub const DirectBenchmark = struct {
    const Self = @This();

    pub const name = "direct/reduction/sum-mean-min-max/256x256/axis1";
    pub const default_iterations = ZgcBenchmark.default_iterations;
    pub const default_warmup_iterations = ZgcBenchmark.default_warmup_iterations;
    pub const work_items_per_invocation = ZgcBenchmark.work_items_per_invocation;
    pub const work_unit = ZgcBenchmark.work_unit;
    pub const bytes_per_invocation = ZgcBenchmark.bytes_per_invocation;

    input: [element_count]f32,
    sums: [rows]f32,
    means: [rows]f32,
    minima: [rows]f32,
    maxima: [rows]f32,

    pub fn init() Self {
        var result: Self = undefined;
        for (&result.input, 0..) |*item, index| item.* = value(index / columns, index % columns);
        return result;
    }

    pub fn validate(self: *Self) !void {
        directStep(self);
        var expected_sum: f32 = 0;
        var expected_min = std.math.inf(f32);
        var expected_max = -std.math.inf(f32);
        for (0..columns) |column| {
            const item = value(0, column);
            expected_sum += item;
            expected_min = @min(expected_min, item);
            expected_max = @max(expected_max, item);
        }
        if (!std.math.approxEqAbs(f32, expected_sum, self.sums[0], 1e-4) or
            !std.math.approxEqAbs(f32, expected_sum / @as(f32, @floatFromInt(columns)), self.means[0], 1e-5) or
            self.minima[0] != expected_min or self.maxima[0] != expected_max) return error.IncorrectResult;
    }

    pub fn run(self: *Self, iterations: usize) void {
        for (0..iterations) |_| {
            directStep(self);
            std.mem.doNotOptimizeAway(&self.sums);
            std.mem.doNotOptimizeAway(&self.means);
            std.mem.doNotOptimizeAway(&self.minima);
            std.mem.doNotOptimizeAway(&self.maxima);
        }
    }
};

fn directStep(state: *DirectBenchmark) void {
    @setRuntimeSafety(false);
    for (0..rows) |row| {
        var sum: f32 = 0;
        var minimum = std.math.inf(f32);
        var maximum = -std.math.inf(f32);
        for (0..columns) |column| {
            const item = state.input[row * columns + column];
            sum += item;
            minimum = @min(minimum, item);
            maximum = @max(maximum, item);
        }
        state.sums[row] = sum;
        state.means[row] = sum / @as(f32, @floatFromInt(columns));
        state.minima[row] = minimum;
        state.maxima[row] = maximum;
    }
}

fn value(row: usize, column: usize) f32 {
    return @as(f32, @floatFromInt((row * columns + column) % 31)) / 10.0 - 1.5;
}
