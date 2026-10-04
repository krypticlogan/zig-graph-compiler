const std = @import("std");
const zgc = @import("zgc");

const rows = 128;
const columns = 128;
const element_count = rows * columns;
const Layout = enum { contiguous, transposed };

fn AddBenchmark(comptime layout: Layout, comptime benchmark_name: []const u8) type {
    return struct {
        const Self = @This();

        pub const name = benchmark_name;
        pub const default_iterations = 5_000;
        pub const default_warmup_iterations = 500;
        pub const work_items_per_invocation: f64 = element_count;
        pub const work_unit = "elements";
        pub const bytes_per_invocation: f64 = element_count * @sizeOf(f32) * 3;

        lhs_storage: [element_count]f32,
        rhs_storage: [element_count]f32,
        output_storage: [element_count]f32,

        pub fn init() Self {
            var result: Self = undefined;
            for (&result.lhs_storage, 0..) |*value, index| {
                value.* = @floatFromInt(index % 127);
            }
            for (&result.rhs_storage, 0..) |*value, index| {
                value.* = @floatFromInt(index % 61);
            }
            result.output_storage = @splat(0);
            return result;
        }

        pub fn run(self: *Self, iterations: usize) void {
            const strides: [2]isize = if (layout == .contiguous)
                .{ columns, 1 }
            else
                .{ 1, rows };
            const lhs: zgc.core.Tensor.ConstView(f32, 2) = .{
                .storage = &self.lhs_storage,
                .shape = .{ rows, columns },
                .strides = strides,
                .offset = 0,
            };
            const rhs: zgc.core.Tensor.ConstView(f32, 2) = .{
                .storage = &self.rhs_storage,
                .shape = .{ rows, columns },
                .strides = strides,
                .offset = 0,
            };
            const output: zgc.core.Tensor.View(f32, 2) = .{
                .storage = &self.output_storage,
                .shape = .{ rows, columns },
                .strides = strides,
                .offset = 0,
            };
            const op: zgc.Op = .{ .compute = .add };
            for (0..iterations) |_| {
                op.execute(.{ lhs, rhs }, output);
                std.mem.doNotOptimizeAway(&self.output_storage);
            }
        }
    };
}

pub const AddContiguous = AddBenchmark(.contiguous, "add/f32/128x128/contiguous");
pub const AddStrided = AddBenchmark(.transposed, "add/f32/128x128/transposed-views");

pub const AddBroadcast = struct {
    const Self = @This();

    pub const name = "add/f32/128x128+bias128/broadcast";
    pub const default_iterations = 5_000;
    pub const default_warmup_iterations = 500;
    pub const work_items_per_invocation: f64 = element_count;
    pub const work_unit = "elements";
    pub const bytes_per_invocation: f64 = element_count * @sizeOf(f32) * 3;

    matrix_storage: [element_count]f32,
    bias_storage: [columns]f32,
    output_storage: [element_count]f32,

    pub fn init() Self {
        var result: Self = undefined;
        for (&result.matrix_storage, 0..) |*value, index| value.* = @floatFromInt(index % 127);
        for (&result.bias_storage, 0..) |*value, index| value.* = @floatFromInt(index % 31);
        result.output_storage = @splat(0);
        return result;
    }

    pub fn run(self: *Self, iterations: usize) void {
        const matrix: zgc.core.Tensor.ConstView(f32, 2) = .{
            .storage = &self.matrix_storage,
            .shape = .{ rows, columns },
            .strides = .{ columns, 1 },
            .offset = 0,
        };
        const bias: zgc.core.Tensor.ConstView(f32, 1) = .{
            .storage = &self.bias_storage,
            .shape = .{columns},
            .strides = .{1},
            .offset = 0,
        };
        const output: zgc.core.Tensor.View(f32, 2) = .{
            .storage = &self.output_storage,
            .shape = .{ rows, columns },
            .strides = .{ columns, 1 },
            .offset = 0,
        };
        const op: zgc.Op = .{ .compute = .add };
        for (0..iterations) |_| {
            op.execute(.{ matrix, bias }, output);
            std.mem.doNotOptimizeAway(&self.output_storage);
        }
    }
};

pub const ExpContiguous = struct {
    const Self = @This();

    pub const name = "exp/f32/16k/contiguous";
    pub const default_iterations = 1_000;
    pub const default_warmup_iterations = 100;
    pub const work_items_per_invocation: f64 = element_count;
    pub const work_unit = "elements";
    pub const bytes_per_invocation: f64 = element_count * @sizeOf(f32) * 2;

    input_storage: [element_count]f32,
    output_storage: [element_count]f32,

    pub fn init() Self {
        var result: Self = undefined;
        for (&result.input_storage, 0..) |*value, index| {
            value.* = @as(f32, @floatFromInt(index % 21)) / 10.0 - 1.0;
        }
        result.output_storage = @splat(0);
        return result;
    }

    pub fn run(self: *Self, iterations: usize) void {
        const input: zgc.core.Tensor.ConstView(f32, 1) = .{
            .storage = &self.input_storage,
            .shape = .{element_count},
            .strides = .{1},
            .offset = 0,
        };
        const output: zgc.core.Tensor.View(f32, 1) = .{
            .storage = &self.output_storage,
            .shape = .{element_count},
            .strides = .{1},
            .offset = 0,
        };
        const op: zgc.Op = .{ .compute = .exp };
        for (0..iterations) |_| {
            op.execute(.{input}, output);
            std.mem.doNotOptimizeAway(&self.output_storage);
        }
    }
};
