const std = @import("std");
const zgc = @import("zgc");

const Layout = enum { contiguous, transposed };

fn MatmulBenchmark(
    comptime m: usize,
    comptime k_len: usize,
    comptime n: usize,
    comptime lhs_layout: Layout,
    comptime rhs_layout: Layout,
    comptime output_layout: Layout,
    comptime benchmark_name: []const u8,
    comptime iterations: usize,
) type {
    return struct {
        const Self = @This();

        pub const name = benchmark_name;
        pub const default_iterations = iterations;
        pub const default_warmup_iterations = @max(iterations / 10, 1);
        pub const work_items_per_invocation: f64 = 2 * m * n * k_len;
        pub const work_unit = "FLOP";
        pub const bytes_per_invocation: f64 = @sizeOf(f32) * (m * k_len + k_len * n + m * n);

        lhs_storage: [m * k_len]f32,
        rhs_storage: [k_len * n]f32,
        output_storage: [m * n]f32,

        pub fn init() Self {
            var result: Self = undefined;
            for (&result.lhs_storage, 0..) |*value, index| {
                value.* = @as(f32, @floatFromInt(index % 31)) / 31.0;
            }
            for (&result.rhs_storage, 0..) |*value, index| {
                value.* = @as(f32, @floatFromInt(index % 29)) / 29.0;
            }
            result.output_storage = @splat(0);
            return result;
        }

        pub fn run(self: *Self, run_iterations: usize) void {
            const lhs: zgc.core.Tensor.ConstView(f32, 2) = .{
                .storage = &self.lhs_storage,
                .shape = .{ m, k_len },
                .strides = if (lhs_layout == .contiguous) .{ k_len, 1 } else .{ 1, m },
                .offset = 0,
            };
            const rhs: zgc.core.Tensor.ConstView(f32, 2) = .{
                .storage = &self.rhs_storage,
                .shape = .{ k_len, n },
                .strides = if (rhs_layout == .contiguous) .{ n, 1 } else .{ 1, k_len },
                .offset = 0,
            };
            const output: zgc.core.Tensor.View(f32, 2) = .{
                .storage = &self.output_storage,
                .shape = .{ m, n },
                .strides = if (output_layout == .contiguous) .{ n, 1 } else .{ 1, m },
                .offset = 0,
            };
            const op: zgc.Op = .{ .compute = .matmul };
            for (0..run_iterations) |_| {
                op.execute(.{ lhs, rhs }, output);
                std.mem.doNotOptimizeAway(&self.output_storage);
            }
        }
    };
}

pub const Square32 = MatmulBenchmark(32, 32, 32, .contiguous, .contiguous, .contiguous, "matmul/f32/32x32x32/contiguous", 500);
pub const Square64 = MatmulBenchmark(64, 64, 64, .contiguous, .contiguous, .contiguous, "matmul/f32/64x64x64/contiguous", 100);
pub const Square128 = MatmulBenchmark(128, 128, 128, .contiguous, .contiguous, .contiguous, "matmul/f32/128x128x128/contiguous", 10);
pub const Rectangular = MatmulBenchmark(32, 128, 64, .contiguous, .contiguous, .contiguous, "matmul/f32/32x128x64/contiguous", 100);
pub const LhsStrided = MatmulBenchmark(64, 64, 64, .transposed, .contiguous, .contiguous, "matmul/f32/64x64x64/lhs-strided", 100);
pub const RhsStrided = MatmulBenchmark(64, 64, 64, .contiguous, .transposed, .contiguous, "matmul/f32/64x64x64/rhs-strided", 100);
pub const OutputStrided = MatmulBenchmark(64, 64, 64, .contiguous, .contiguous, .transposed, "matmul/f32/64x64x64/output-strided", 100);
pub const BatchContiguous = MatmulBenchmark(32, 128, 64, .transposed, .contiguous, .transposed, "matmul/f32/32x128x64/batch-contiguous", 100);
pub const Reference16x8x24 = MatmulBenchmark(16, 8, 24, .contiguous, .contiguous, .contiguous, "matmul/f32/16x8x24/reference", 5_000);
pub const Reference16x32x64 = MatmulBenchmark(16, 32, 64, .contiguous, .contiguous, .contiguous, "matmul/f32/16x32x64/reference", 1_000);
