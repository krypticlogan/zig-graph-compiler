const std = @import("std");
const zgc = @import("zgc");

const rows = 256;
const columns = 256;
const element_count = rows * columns;

const Sources = enum(usize) { lhs, rhs, bias };
const Definition = zgc.DefinitionBuilder;

const definition = blk: {
    var builder = Definition.init();
    const builder_sources = builder.sources(Sources);
    const lhs = builder_sources.input(.lhs, .f32, &.{ rows, columns });
    const rhs = builder_sources.input(.rhs, .f32, &.{ rows, columns });
    const bias = builder_sources.input(.bias, .f32, &.{columns});
    const expression = builder.add(builder.mul(lhs, rhs), bias);
    builder.output(builder.sum(expression, .{ .axes = &.{1} }));
    break :blk builder.finish();
};

const Model = definition.modelWith(&.{
    .{ .source = .lhs, .binding = zgc.memory.Source.bound },
    .{ .source = .rhs, .binding = zgc.memory.Source.bound },
    .{ .source = .bias, .binding = zgc.memory.Source.bound },
});

pub const ZgcBenchmark = struct {
    const Self = @This();
    pub const CompiledModel = Model;

    pub const name = "graph/fusion/mul-add-sum/256x256";
    pub const default_iterations = 500;
    pub const default_warmup_iterations = 50;
    pub const work_items_per_invocation: f64 = element_count;
    pub const work_unit = "elements";
    pub const bytes_per_invocation: f64 = @sizeOf(f32) * (2 * element_count + columns + rows);

    model: Model,
    lhs: [element_count]f32,
    rhs: [element_count]f32,
    bias: [columns]f32,

    pub fn init() Self {
        var result: Self = .{
            .model = Model.init(),
            .lhs = undefined,
            .rhs = undefined,
            .bias = undefined,
        };
        fillMatrix(.lhs, &result.lhs, lhsValue);
        fillMatrix(.rhs, &result.rhs, rhsValue);
        for (&result.bias, 0..) |*item, column| item.* = biasValue(column);
        return result;
    }

    pub fn prepare(self: *Self) !void {
        try self.model.bindInput(.lhs, &self.lhs);
        try self.model.bindInput(.rhs, &self.rhs);
        try self.model.bindInput(.bias, &self.bias);
    }

    pub fn validate(self: *Self) !void {
        self.model.run();
        var expected: f32 = 0;
        for (0..columns) |column| {
            expected += lhsValue(0, column) * rhsValue(0, column) + biasValue(column);
        }
        if (!std.math.approxEqAbs(f32, expected, self.model.outputView(0).get(.{0}), 1e-3)) {
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

    pub const name = "direct/fusion/mul-add-sum/256x256";
    pub const default_iterations = ZgcBenchmark.default_iterations;
    pub const default_warmup_iterations = ZgcBenchmark.default_warmup_iterations;
    pub const work_items_per_invocation = ZgcBenchmark.work_items_per_invocation;
    pub const work_unit = ZgcBenchmark.work_unit;
    pub const bytes_per_invocation = ZgcBenchmark.bytes_per_invocation;

    lhs: [element_count]f32,
    rhs: [element_count]f32,
    bias: [columns]f32,
    output: [rows]f32,

    pub fn init() Self {
        var result: Self = undefined;
        for (&result.lhs, 0..) |*item, index| item.* = lhsValue(index / columns, index % columns);
        for (&result.rhs, 0..) |*item, index| item.* = rhsValue(index / columns, index % columns);
        for (&result.bias, 0..) |*item, column| item.* = biasValue(column);
        return result;
    }

    pub fn validate(self: *Self) !void {
        directStep(self);
        var expected: f32 = 0;
        for (0..columns) |column| expected += lhsValue(0, column) * rhsValue(0, column) + biasValue(column);
        if (!std.math.approxEqAbs(f32, expected, self.output[0], 1e-3)) return error.IncorrectResult;
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
    for (0..rows) |row| {
        var accumulator: f32 = 0;
        for (0..columns) |column| {
            const index = row * columns + column;
            accumulator += state.lhs[index] * state.rhs[index] + state.bias[column];
        }
        state.output[row] = accumulator;
    }
}

fn fillMatrix(comptime source: Sources, storage: *[element_count]f32, comptime valueFn: anytype) void {
    const layout = Model.sourceLayout(source);
    for (0..rows) |row| {
        for (0..columns) |column| {
            const offset: usize = @intCast(row * @as(usize, @intCast(layout.strides[0])) +
                column * @as(usize, @intCast(layout.strides[1])));
            storage[offset] = valueFn(row, column);
        }
    }
}

fn lhsValue(row: usize, column: usize) f32 {
    return @as(f32, @floatFromInt((row * columns + column) % 31)) / 31.0;
}

fn rhsValue(row: usize, column: usize) f32 {
    return @as(f32, @floatFromInt((row * columns + column) % 29)) / 29.0;
}

fn biasValue(column: usize) f32 {
    return @as(f32, @floatFromInt(column % 17)) / 17.0;
}
