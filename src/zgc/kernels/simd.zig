const std = @import("std");
const Dtype = @import("../storage/dtype.zig").Dtype;
const accumulation = @import("accumulation.zig");

/// Reduce a contiguous slice. The operator defines the scalar and vector
/// implementations while this helper owns vector chunking and scalar tails.
pub fn reduce(
    comptime dtype: Dtype,
    values: []const dtype.Scalar(),
    context: anytype,
    comptime Operator: type,
) accumulation.AccumulatorScalar(dtype) {
    const vector_len = std.simd.suggestVectorLength(dtype.Scalar()) orelse 1;
    const InputVector = dtype.Vector(vector_len);
    const AccumulatorVector = accumulation.AccumulatorVector(dtype, vector_len);

    var vector_accumulator: AccumulatorVector =
        @splat(Operator.identity(dtype, context));
    var index: usize = 0;
    while (index + vector_len <= values.len) : (index += vector_len) {
        const vector: InputVector = values[index..][0..vector_len].*;
        vector_accumulator = Operator.vector(
            dtype,
            vector_len,
            vector_accumulator,
            vector,
            context,
        );
    }

    var result = Operator.horizontal(dtype, vector_len, vector_accumulator);
    while (index < values.len) : (index += 1) {
        result = Operator.scalar(dtype, result, values[index], context);
    }
    return result;
}

/// Map between equally-sized contiguous slices. The operator supplies only its
/// scalar and vector math; this helper owns chunking and tail dispatch.
pub fn map(
    comptime dtype: Dtype,
    input: []const dtype.Scalar(),
    output: []dtype.Scalar(),
    context: anytype,
    comptime Operator: type,
) void {
    const vector_len = std.simd.suggestVectorLength(dtype.Scalar()) orelse 1;
    const Vector = dtype.Vector(vector_len);

    var index: usize = 0;
    while (index + vector_len <= input.len) : (index += vector_len) {
        const values: Vector = input[index..][0..vector_len].*;
        output[index..][0..vector_len].* =
            Operator.vector(dtype, vector_len, values, context);
    }
    while (index < input.len) : (index += 1) {
        output[index] = Operator.scalar(dtype, input[index], context);
    }
}
