const std = @import("std");
const zgc = @import("zgc_internal").api;

const batch_size = 32;
const input_width = 128;
const hidden_width = 64;
const output_width = 10;
const parameter_total = input_width * hidden_width + hidden_width +
    hidden_width * output_width + output_width;

const Sources = enum(usize) { input, w1, b1, w2, b2 };
const Definition = zgc.DefinitionBuilder;

const definition = blk: {
    var builder = Definition.init();
    const builder_sources = builder.sources(Sources);
    const input = builder_sources.input(.input, .f32, &.{ batch_size, input_width });
    const w1 = builder_sources.parameter(.w1, .f32, &.{ input_width, hidden_width });
    const b1 = builder_sources.parameter(.b1, .f32, &.{hidden_width});
    const w2 = builder_sources.parameter(.w2, .f32, &.{ hidden_width, output_width });
    const b2 = builder_sources.parameter(.b2, .f32, &.{output_width});
    const hidden = builder.relu(builder.add(builder.matmul(input, w1), b1));
    builder.output(builder.softmax(builder.add(builder.matmul(hidden, w2), b2), 1));
    break :blk builder.finish();
};

const Executable = @import("benchmark_selection").executable(definition, &.{.{
    .source = .input,
    .binding = zgc.memory.Source.bound,
}});

pub const ZgcBenchmark = struct {
    const Self = @This();
    pub const CompiledExecutable = Executable;

    pub const name = "graph/dense/128-64-10/batch32";
    pub const default_iterations = 100;
    pub const default_warmup_iterations = 10;
    pub const work_items_per_invocation: f64 = batch;
    pub const work_unit = "inferences";
    pub const bytes_per_invocation: f64 = @sizeOf(f32) *
        (parameter_total + batch_size * input_width + batch_size * output_width);
    pub const latency_divisor: f64 = batch_size;
    pub const latency_unit = "inference";
    pub const parameter_count = parameter_total;
    pub const batch = batch_size;

    runtime: Executable.Runtime,
    input: [batch_size * input_width]f32,

    pub fn init() Self {
        var result: Self = .{ .runtime = Executable.Runtime.init(), .input = undefined };
        const layout = Executable.Runtime.sourceLayout(.input);
        for (0..batch_size) |row| {
            for (0..input_width) |column| {
                const offset: usize = @intCast(row * @as(usize, @intCast(layout.strides[0])) +
                    column * @as(usize, @intCast(layout.strides[1])));
                result.input[offset] = sample(row * input_width + column, 17);
            }
        }

        var w1: [input_width * hidden_width]f32 = undefined;
        var b1: [hidden_width]f32 = undefined;
        var w2: [hidden_width * output_width]f32 = undefined;
        var b2: [output_width]f32 = undefined;
        for (&w1, 0..) |*item, index| item.* = sample(index, 31);
        for (&b1, 0..) |*item, index| item.* = sample(index, 7);
        for (&w2, 0..) |*item, index| item.* = sample(index, 44);
        for (&b2, 0..) |*item, index| item.* = sample(index, 26);
        result.runtime.copySource(.w1, &w1) catch unreachable;
        result.runtime.copySource(.b1, &b1) catch unreachable;
        result.runtime.copySource(.w2, &w2) catch unreachable;
        result.runtime.copySource(.b2, &b2) catch unreachable;
        return result;
    }

    pub fn prepare(self: *Self) !void {
        try self.runtime.bindInput(.input, &self.input);
    }

    pub fn validate(self: *Self) !void {
        self.runtime.run();
        const output = self.runtime.outputView(0);
        var probability_sum: f32 = 0;
        for (0..output_width) |column| {
            const probability = output.get(.{ 0, column });
            if (!std.math.isFinite(probability) or probability < 0) return error.IncorrectResult;
            probability_sum += probability;
        }
        if (!std.math.approxEqAbs(f32, probability_sum, 1, 1e-4)) return error.IncorrectResult;
    }

    pub fn run(self: *Self, iterations: usize) void {
        for (0..iterations) |_| {
            self.runtime.run();
            std.mem.doNotOptimizeAway(self.runtime.outputView(0).storage);
        }
    }
};

pub const DirectBenchmark = struct {
    const Self = @This();

    pub const name = "direct/dense/128-64-10/batch32";
    pub const default_iterations = ZgcBenchmark.default_iterations;
    pub const default_warmup_iterations = ZgcBenchmark.default_warmup_iterations;
    pub const work_items_per_invocation = ZgcBenchmark.work_items_per_invocation;
    pub const work_unit = ZgcBenchmark.work_unit;
    pub const bytes_per_invocation = ZgcBenchmark.bytes_per_invocation;
    pub const latency_divisor = ZgcBenchmark.latency_divisor;
    pub const latency_unit = ZgcBenchmark.latency_unit;
    pub const parameter_count = ZgcBenchmark.parameter_count;
    pub const batch = ZgcBenchmark.batch;

    input: [batch_size * input_width]f32,
    w1: [input_width * hidden_width]f32,
    b1: [hidden_width]f32,
    w2: [hidden_width * output_width]f32,
    b2: [output_width]f32,
    hidden: [batch_size * hidden_width]f32,
    output: [batch_size * output_width]f32,

    pub fn init() Self {
        var result: Self = undefined;
        for (&result.input, 0..) |*item, index| item.* = sample(index, 17);
        for (&result.w1, 0..) |*item, index| item.* = sample(index, 31);
        for (&result.b1, 0..) |*item, index| item.* = sample(index, 7);
        for (&result.w2, 0..) |*item, index| item.* = sample(index, 44);
        for (&result.b2, 0..) |*item, index| item.* = sample(index, 26);
        return result;
    }

    pub fn validate(self: *Self) !void {
        directStep(self);
        var probability_sum: f32 = 0;
        for (self.output[0..output_width]) |probability| {
            if (!std.math.isFinite(probability) or probability < 0) return error.IncorrectResult;
            probability_sum += probability;
        }
        if (!std.math.approxEqAbs(f32, probability_sum, 1, 1e-4)) return error.IncorrectResult;
    }

    pub fn run(self: *Self, iterations: usize) void {
        for (0..iterations) |_| {
            directStep(self);
            std.mem.doNotOptimizeAway(&self.output);
        }
    }
};

fn directStep(state: *DirectBenchmark) void {
    @setRuntimeSafety(false);
    for (0..batch_size) |batch_index| {
        const input_row = state.input[batch_index * input_width ..][0..input_width];
        const hidden_row = state.hidden[batch_index * hidden_width ..][0..hidden_width];
        const output_row = state.output[batch_index * output_width ..][0..output_width];

        @memcpy(hidden_row, &state.b1);
        for (input_row, 0..) |value, inner| {
            const weight_row = state.w1[inner * hidden_width ..][0..hidden_width];
            for (hidden_row, weight_row) |*result, weight| {
                result.* += value * weight;
            }
        }
        for (hidden_row) |*value| value.* = @max(value.*, 0);

        @memcpy(output_row, &state.b2);
        for (hidden_row, 0..) |value, inner| {
            const weight_row = state.w2[inner * output_width ..][0..output_width];
            for (output_row, weight_row) |*result, weight| {
                result.* += value * weight;
            }
        }
        var maximum = -std.math.inf(f32);
        for (output_row) |value| {
            maximum = @max(maximum, value);
        }

        var total: f32 = 0;
        for (output_row) |*value| {
            value.* = @exp(value.* - maximum);
            total += value.*;
        }
        const inverse_total = 1.0 / total;
        for (output_row) |*value| {
            value.* *= inverse_total;
        }
    }
}

fn sample(index: usize, salt: usize) f32 {
    const centered = @as(f32, @floatFromInt((index * 37 + salt) % 257)) - 128.0;
    return centered / 2048.0;
}
