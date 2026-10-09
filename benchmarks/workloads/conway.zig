const std = @import("std");
const zgc = @import("zgc");

const width = 120;
const height = 88;
const cell_count = width * height;

const Sources = enum(usize) { world };
const Definition = zgc.DefinitionBuilder;

const Model = model: {
    var builder = Definition.init();
    const builder_sources = builder.sources(Sources);
    const world = builder_sources.input(.world, .bool, &.{ height, width });
    const dead = builder.scalar(.bool, false);
    const padded = builder.pad(world, dead, .{
        .before = &.{ 1, 1 },
        .after = &.{ 1, 1 },
    });
    const neighborhoods = builder.windows(padded, .{ .sizes = &.{ 3, 3 } });
    const one = builder.scalar(.i8, 1);
    const zero = builder.scalar(.i8, 0);
    const neighborhood_values = builder.where(neighborhoods, one, zero);
    const neighborhood_total = builder.sum(neighborhood_values, .{ .axes = &.{ -2, -1 } });
    const center_value = builder.where(world, one, zero);
    const neighbor_count = builder.sub(neighborhood_total, center_value);
    const two = builder.scalar(.i8, 2);
    const three = builder.scalar(.i8, 3);
    builder.output(builder.logicalOr(
        builder.equal(neighbor_count, three),
        builder.logicalAnd(world, builder.equal(neighbor_count, two)),
    ));
    break :model builder.finish().modelWith(&.{.{
        .source = .world,
        .binding = zgc.memory.Source.bound,
    }});
};

pub const ZgcBenchmark = struct {
    const Self = @This();
    pub const CompiledModel = Model;

    pub const name = "graph/application/conway/120x88/zgc";
    pub const default_iterations = 100;
    pub const default_warmup_iterations = 10;
    pub const work_items_per_invocation: f64 = cell_count;
    pub const work_unit = "cells";
    pub const bytes_per_invocation: f64 = 2 * cell_count;

    model: Model,
    world: [cell_count]bool,

    pub fn init() Self {
        var result: Self = .{ .model = Model.init(), .world = undefined };
        initializeWorld(&result.world);
        return result;
    }

    pub fn prepare(self: *Self) !void {
        try self.model.bindInput(.world, &self.world);
    }

    pub fn validate(self: *Self) !void {
        self.model.run();
        var expected: [cell_count]bool = undefined;
        directStep(&self.world, &expected);
        if (!std.mem.eql(bool, &expected, self.model.outputView(0).storage)) return error.IncorrectResult;
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

    pub const name = "graph/application/conway/120x88/direct";
    pub const default_iterations = ZgcBenchmark.default_iterations;
    pub const default_warmup_iterations = ZgcBenchmark.default_warmup_iterations;
    pub const work_items_per_invocation = ZgcBenchmark.work_items_per_invocation;
    pub const work_unit = ZgcBenchmark.work_unit;
    pub const bytes_per_invocation = ZgcBenchmark.bytes_per_invocation;

    world: [cell_count]bool,
    output: [cell_count]bool,

    pub fn init() Self {
        var result: Self = undefined;
        initializeWorld(&result.world);
        return result;
    }

    pub fn validate(self: *Self) !void {
        directStep(&self.world, &self.output);
    }

    pub fn run(self: *Self, iterations: usize) void {
        for (0..iterations) |_| {
            directStep(&self.world, &self.output);
            std.mem.doNotOptimizeAway(&self.output);
        }
    }
};

fn initializeWorld(world: *[cell_count]bool) void {
    for (world, 0..) |*cell, index| {
        cell.* = ((index * 17 + index / width * 13) % 23) < 7;
    }
}

fn directStep(world: *const [cell_count]bool, output: *[cell_count]bool) void {
    @setRuntimeSafety(false);
    for (0..height) |row| {
        for (0..width) |column| {
            var neighbors: u8 = 0;
            const row_start = if (row == 0) 0 else row - 1;
            const row_end = @min(row + 1, height - 1);
            const column_start = if (column == 0) 0 else column - 1;
            const column_end = @min(column + 1, width - 1);
            for (row_start..row_end + 1) |neighbor_row| {
                for (column_start..column_end + 1) |neighbor_column| {
                    if ((neighbor_row != row or neighbor_column != column) and
                        world[neighbor_row * width + neighbor_column]) neighbors += 1;
                }
            }
            const alive = world[row * width + column];
            output[row * width + column] = neighbors == 3 or (alive and neighbors == 2);
        }
    }
}
