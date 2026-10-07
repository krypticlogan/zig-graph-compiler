const std = @import("std");
const zgc = @import("zgc");

const m = 64;
const k = 64;
const n = 64;

const Sources = enum(usize) { input, weights };
const Definition = zgc.DefinitionBuilder(Sources, .{
    .max_rank = 2,
    .max_nodes = 1,
    .max_tensors = 3,
    .max_input_refs = 2,
    .max_outputs = 1,
});

const definition = blk: {
    var builder = Definition.init();
    const input = builder.input(.input, .f32, &.{ m, k });
    const weights = builder.parameter(.weights, .f32, &.{ k, n });
    builder.output(builder.matmul(input, weights));
    break :blk builder.finish();
};

const Model = definition.modelWith(&.{.{
    .source = .input,
    .binding = zgc.memory.Source.bound,
}});

pub const ZgcBenchmark = struct {
    const Self = @This();
    pub const CompiledModel = Model;

    pub const name = "graph/contraction/64x64x64";
    pub const default_iterations = 100;
    pub const default_warmup_iterations = 10;
    pub const work_items_per_invocation: f64 = 2 * m * n * k;
    pub const work_unit = "FLOP";
    pub const bytes_per_invocation: f64 = @sizeOf(f32) * (m * k + k * n + m * n);

    model: Model,
    input: [m * k]f32,

    pub fn init() Self {
        var result: Self = .{ .model = Model.init(), .input = undefined };
        const layout = Model.sourceLayout(.input);
        for (0..m) |row| {
            for (0..k) |column| {
                const offset: usize = @intCast(row * @as(usize, @intCast(layout.strides[0])) +
                    column * @as(usize, @intCast(layout.strides[1])));
                result.input[offset] = inputValue(row, column);
            }
        }
        var weights: [k * n]f32 = undefined;
        for (0..k) |row| {
            for (0..n) |column| weights[row * n + column] = weightValue(row, column);
        }
        result.model.copySource(.weights, &weights) catch unreachable;
        return result;
    }

    pub fn prepare(self: *Self) !void {
        try self.model.bindInput(.input, &self.input);
    }

    pub fn validate(self: *Self) !void {
        self.model.run();
        var expected: f32 = 0;
        for (0..k) |inner| expected += inputValue(0, inner) * weightValue(inner, 0);
        if (!std.math.approxEqAbs(f32, expected, self.model.outputView(0).get(.{ 0, 0 }), 1e-4)) {
            return error.IncorrectResult;
        }
    }

    pub fn run(self: *Self, iterations: usize) void {
        for (0..iterations) |_| {
            self.model.run();
            std.mem.doNotOptimizeAway(self.model.outputView(0).storage);
        }
    }
};

pub const DirectBenchmark = struct {
    const Self = @This();

    pub const name = "direct/contraction/64x64x64";
    pub const default_iterations = ZgcBenchmark.default_iterations;
    pub const default_warmup_iterations = ZgcBenchmark.default_warmup_iterations;
    pub const work_items_per_invocation = ZgcBenchmark.work_items_per_invocation;
    pub const work_unit = ZgcBenchmark.work_unit;
    pub const bytes_per_invocation = ZgcBenchmark.bytes_per_invocation;

    input: [m * k]f32,
    weights: [k * n]f32,
    output: [m * n]f32,

    pub fn init() Self {
        var result: Self = undefined;
        for (&result.input, 0..) |*item, index| item.* = inputValue(index / k, index % k);
        for (&result.weights, 0..) |*item, index| item.* = weightValue(index / n, index % n);
        return result;
    }

    pub fn validate(self: *Self) !void {
        directStep(&self.input, &self.weights, &self.output);
        var expected: f32 = 0;
        for (0..k) |inner| expected += inputValue(0, inner) * weightValue(inner, 0);
        if (!std.math.approxEqAbs(f32, expected, self.output[0], 1e-4)) return error.IncorrectResult;
    }

    pub fn run(self: *Self, iterations: usize) void {
        for (0..iterations) |_| {
            directStep(&self.input, &self.weights, &self.output);
            std.mem.doNotOptimizeAway(&self.output);
        }
    }
};

fn directStep(input: *const [m * k]f32, weights: *const [k * n]f32, output: *[m * n]f32) void {
    @setRuntimeSafety(false);
    for (0..m) |row| {
        for (0..n) |column| {
            var accumulator: f32 = 0;
            for (0..k) |inner| accumulator += input[row * k + inner] * weights[inner * n + column];
            output[row * n + column] = accumulator;
        }
    }
}

fn inputValue(row: usize, column: usize) f32 {
    return @as(f32, @floatFromInt((row * k + column) % 31)) / 31.0;
}

fn weightValue(row: usize, column: usize) f32 {
    return @as(f32, @floatFromInt((row * n + column) % 29)) / 29.0;
}
