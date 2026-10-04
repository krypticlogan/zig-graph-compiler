const std = @import("std");
const zgc = @import("zgc");

fn ReluBenchmark(
    comptime element_count: usize,
    comptime benchmark_name: []const u8,
    comptime iterations: usize,
) type {
    return struct {
        const Self = @This();

        pub const name = benchmark_name;
        pub const default_iterations = iterations;
        pub const default_warmup_iterations = @max(iterations / 10, 1);
        pub const work_items_per_invocation: f64 = element_count;
        pub const work_unit = "elements";
        pub const bytes_per_invocation: f64 = element_count * @sizeOf(f32) * 2;

        input_data: [element_count]f32,
        output_data: [element_count]f32,

        pub fn init() Self {
            var input_data: [element_count]f32 = undefined;
            for (&input_data, 0..) |*value, index| {
                const magnitude: f32 = @floatFromInt(index % 127);
                value.* = if (index % 2 == 0) magnitude else -magnitude;
            }
            return .{
                .input_data = input_data,
                .output_data = @splat(0),
            };
        }

        pub fn run(self: *Self, run_iterations: usize) void {
            const input: zgc.core.Tensor.ConstView(f32, 1) = .{
                .storage = &self.input_data,
                .shape = .{element_count},
                .strides = .{1},
                .offset = 0,
            };
            const output: zgc.core.Tensor.View(f32, 1) = .{
                .storage = &self.output_data,
                .shape = .{element_count},
                .strides = .{1},
                .offset = 0,
            };
            const op: zgc.Op = .{ .compute = .relu };
            for (0..run_iterations) |_| {
                op.execute(.{input}, output);
                std.mem.doNotOptimizeAway(&self.output_data);
            }
        }
    };
}

pub const Benchmark = ReluBenchmark(16 * 1024, "relu/f32/16k", 5_000);
pub const Small16 = ReluBenchmark(16, "relu/f32/16", 1_000_000);
pub const Medium64 = ReluBenchmark(64, "relu/f32/64", 500_000);
pub const Large256 = ReluBenchmark(256, "relu/f32/256", 100_000);
