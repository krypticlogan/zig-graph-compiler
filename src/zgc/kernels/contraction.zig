const std = @import("std");
const accumulation = @import("accumulation.zig");
const ContractionPlan = @import("../execution/execution.zig").ContractionPlan;
pub const Strategy = ContractionPlan.Strategy;

/// Execute one compile-time-selected traversal.
///
/// lhs:    [M, K]
/// rhs:    [K, N]
/// output: [M, N]
pub fn matmulWithPlan(
    comptime strategy: Strategy,
    lhs: anytype,
    rhs: anytype,
    output: anytype,
) void {
    const m = lhs.shape[0];
    const k_len = lhs.shape[1];
    const n = rhs.shape[1];

    switch (comptime strategy) {
        .output_columns => matmulOutputColumns(lhs, rhs, output, m, k_len, n),
        .contracted_axis => matmulContractedAxis(lhs, rhs, output, m, k_len, n),
        .output_rows => matmulOutputRows(lhs, rhs, output, m, k_len, n),
        .scalar => matmulScalar(lhs, rhs, output, m, k_len, n),
    }
}

fn matmulOutputColumns(
    lhs: anytype,
    rhs: anytype,
    output: anytype,
    m: usize,
    k_len: usize,
    n: usize,
) void {
    const Output = @TypeOf(output);
    const dtype = Output.dtype;
    const vector_len = std.simd.suggestVectorLength(Output.scalar_type) orelse 1;
    const InputVector = dtype.Vector(vector_len);
    const Accumulator = accumulation.AccumulatorVector(dtype, vector_len);
    const AccumulatorScalar = accumulation.AccumulatorScalar(dtype);

    for (0..m) |row| {
        var col: usize = 0;
        while (col + vector_len <= n) : (col += vector_len) {
            var accumulator: Accumulator = @splat(0);
            for (0..k_len) |k| {
                const lhs_value: Accumulator = @splat(
                    accumulation.widenScalar(dtype, lhs.get(.{ row, k })),
                );
                const rhs_offset = rhs.elementOffset(.{ k, col });
                const rhs_values: InputVector =
                    rhs.storage[rhs_offset..][0..vector_len].*;
                accumulator += lhs_value *
                    accumulation.widenVector(dtype, vector_len, rhs_values);
            }

            const output_offset = output.elementOffset(.{ row, col });
            output.storage[output_offset..][0..vector_len].* = accumulator;
        }

        while (col < n) : (col += 1) {
            var accumulator: AccumulatorScalar = 0;
            for (0..k_len) |k| {
                accumulator +=
                    accumulation.widenScalar(dtype, lhs.get(.{ row, k })) *
                    accumulation.widenScalar(dtype, rhs.get(.{ k, col }));
            }
            output.set(.{ row, col }, accumulator);
        }
    }
}

fn matmulContractedAxis(
    lhs: anytype,
    rhs: anytype,
    output: anytype,
    m: usize,
    k_len: usize,
    n: usize,
) void {
    const Output = @TypeOf(output);
    const dtype = Output.dtype;
    const vector_len = std.simd.suggestVectorLength(Output.scalar_type) orelse 1;
    const InputVector = dtype.Vector(vector_len);
    const Accumulator = accumulation.AccumulatorVector(dtype, vector_len);
    const AccumulatorScalar = accumulation.AccumulatorScalar(dtype);

    for (0..m) |row| {
        for (0..n) |col| {
            var accumulator: Accumulator = @splat(0);
            var k: usize = 0;
            while (k + vector_len <= k_len) : (k += vector_len) {
                const lhs_offset = lhs.elementOffset(.{ row, k });
                const rhs_offset = rhs.elementOffset(.{ k, col });
                const lhs_values: InputVector =
                    lhs.storage[lhs_offset..][0..vector_len].*;
                const rhs_values: InputVector =
                    rhs.storage[rhs_offset..][0..vector_len].*;
                accumulator +=
                    accumulation.widenVector(dtype, vector_len, lhs_values) *
                    accumulation.widenVector(dtype, vector_len, rhs_values);
            }

            var result: AccumulatorScalar = @reduce(.Add, accumulator);
            while (k < k_len) : (k += 1) {
                result +=
                    accumulation.widenScalar(dtype, lhs.get(.{ row, k })) *
                    accumulation.widenScalar(dtype, rhs.get(.{ k, col }));
            }
            output.set(.{ row, col }, result);
        }
    }
}

fn matmulOutputRows(
    lhs: anytype,
    rhs: anytype,
    output: anytype,
    m: usize,
    k_len: usize,
    n: usize,
) void {
    const Output = @TypeOf(output);
    const dtype = Output.dtype;
    const vector_len = std.simd.suggestVectorLength(Output.scalar_type) orelse 1;
    const InputVector = dtype.Vector(vector_len);
    const Accumulator = accumulation.AccumulatorVector(dtype, vector_len);
    const AccumulatorScalar = accumulation.AccumulatorScalar(dtype);

    for (0..n) |col| {
        var row: usize = 0;
        while (row + vector_len <= m) : (row += vector_len) {
            var accumulator: Accumulator = @splat(0);
            for (0..k_len) |k| {
                const lhs_offset = lhs.elementOffset(.{ row, k });
                const lhs_values: InputVector =
                    lhs.storage[lhs_offset..][0..vector_len].*;
                const rhs_value: Accumulator = @splat(
                    accumulation.widenScalar(dtype, rhs.get(.{ k, col })),
                );
                accumulator +=
                    accumulation.widenVector(dtype, vector_len, lhs_values) *
                    rhs_value;
            }

            const output_offset = output.elementOffset(.{ row, col });
            output.storage[output_offset..][0..vector_len].* = accumulator;
        }

        while (row < m) : (row += 1) {
            var accumulator: AccumulatorScalar = 0;
            for (0..k_len) |k| {
                accumulator +=
                    accumulation.widenScalar(dtype, lhs.get(.{ row, k })) *
                    accumulation.widenScalar(dtype, rhs.get(.{ k, col }));
            }
            output.set(.{ row, col }, accumulator);
        }
    }
}

fn matmulScalar(
    lhs: anytype,
    rhs: anytype,
    output: anytype,
    m: usize,
    k_len: usize,
    n: usize,
) void {
    const dtype = @TypeOf(output).dtype;
    const AccumulatorScalar = accumulation.AccumulatorScalar(dtype);

    for (0..m) |row| {
        for (0..n) |col| {
            var accumulator: AccumulatorScalar = 0;
            for (0..k_len) |k| {
                accumulator +=
                    accumulation.widenScalar(dtype, lhs.get(.{ row, k })) *
                    accumulation.widenScalar(dtype, rhs.get(.{ k, col }));
            }
            output.set(.{ row, col }, accumulator);
        }
    }
}
