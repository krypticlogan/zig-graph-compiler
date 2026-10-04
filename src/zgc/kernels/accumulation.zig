const std = @import("std");
const Dtype = @import("../storage/dtype.zig").Dtype;
const Reduction = @import("../operations/reduction.zig");

pub fn AccumulatorScalar(comptime dtype: Dtype) type {
    return switch (dtype) { // dtypes with smaller bit withs accumulate into larger windows
        .f16 => f32,
        .f32 => f32,
        .i8 => i32,
        .bool => @compileError("boolean tensors cannot be accumulated"),
    };
}

pub fn AccumulatorVector(comptime dtype: Dtype, comptime len: usize) type {
    return @Vector(len, AccumulatorScalar(dtype));
}

pub fn widenVector(
    comptime dtype: Dtype,
    comptime len: usize,
    values: dtype.Vector(len),
) AccumulatorVector(dtype, len) {
    const AccVector = AccumulatorVector(dtype, len);

    return switch (comptime dtype.kind()) {
        .float => @as(AccVector, @floatCast(values)),
        .signed_integer => @as(AccVector, @intCast(values)),
        .boolean => @compileError("boolean tensors cannot be accumulated"),
    };
}

pub fn widenScalar(
    comptime dtype: Dtype,
    value: dtype.Scalar(),
) AccumulatorScalar(dtype) {
    const AccT = AccumulatorScalar(dtype);

    return switch (comptime dtype.kind()) {
        .float => @as(AccT, @floatCast(value)),
        .signed_integer => @as(AccT, @intCast(value)),
        .boolean => @compileError("boolean tensors cannot be accumulated"),
    };
}

pub fn narrowVector(
    comptime dtype: Dtype,
    comptime len: usize,
    values: AccumulatorVector(dtype, len),
) dtype.Vector(len) {
    return switch (comptime dtype.kind()) {
        .float => @floatCast(values),
        .signed_integer => @intCast(values),
        .boolean => @compileError("boolean tensors cannot be accumulated"),
    };
}

pub fn narrowScalar(
    comptime dtype: Dtype,
    value: AccumulatorScalar(dtype),
) dtype.Scalar() {
    return switch (comptime dtype.kind()) {
        .float => @floatCast(value),
        .signed_integer => @intCast(value),
        .boolean => @compileError("boolean tensors cannot be accumulated"),
    };
}

pub fn identity(comptime dtype: Dtype, comptime kind: Reduction.Combine) AccumulatorScalar(dtype) {
    return switch (kind) {
        .sum => 0,
        .minimum => switch (comptime dtype.kind()) {
            .float => std.math.inf(AccumulatorScalar(dtype)),
            .signed_integer => std.math.maxInt(AccumulatorScalar(dtype)),
            .boolean => @compileError("boolean minimum reduction is unsupported"),
        },
        .maximum => switch (comptime dtype.kind()) {
            .float => -std.math.inf(AccumulatorScalar(dtype)),
            .signed_integer => std.math.minInt(AccumulatorScalar(dtype)),
            .boolean => @compileError("boolean maximum reduction is unsupported"),
        },
    };
}

pub fn combine(
    comptime dtype: Dtype,
    comptime kind: Reduction.Combine,
    accumulator: AccumulatorScalar(dtype),
    value: dtype.Scalar(),
) AccumulatorScalar(dtype) {
    return combineAccumulator(dtype, kind, accumulator, widenScalar(dtype, value));
}

pub fn combineAccumulator(
    comptime dtype: Dtype,
    comptime kind: Reduction.Combine,
    accumulator: AccumulatorScalar(dtype),
    value: AccumulatorScalar(dtype),
) AccumulatorScalar(dtype) {
    return switch (kind) {
        .sum => accumulator + value,
        .minimum => @min(accumulator, value),
        .maximum => @max(accumulator, value),
    };
}

pub fn combineVector(
    comptime dtype: Dtype,
    comptime len: usize,
    comptime kind: Reduction.Combine,
    accumulator: AccumulatorVector(dtype, len),
    values: dtype.Vector(len),
) AccumulatorVector(dtype, len) {
    const widened = widenVector(dtype, len, values);
    return switch (kind) {
        .sum => accumulator + widened,
        .minimum => @min(accumulator, widened),
        .maximum => @max(accumulator, widened),
    };
}

pub fn horizontal(
    comptime dtype: Dtype,
    comptime len: usize,
    comptime kind: Reduction.Combine,
    accumulator: AccumulatorVector(dtype, len),
) AccumulatorScalar(dtype) {
    return switch (kind) {
        .sum => @reduce(.Add, accumulator),
        .minimum => @reduce(.Min, accumulator),
        .maximum => @reduce(.Max, accumulator),
    };
}

pub fn finish(
    comptime dtype: Dtype,
    comptime finalizer: Reduction.Finalize,
    accumulator: AccumulatorScalar(dtype),
    reduction_count: usize,
) dtype.Scalar() {
    return switch (finalizer) {
        .identity => narrowScalar(dtype, accumulator),
        .mean => blk: {
            if (comptime dtype.kind() != .float) @compileError("mean requires a floating-point dtype");
            const divisor: AccumulatorScalar(dtype) = @floatFromInt(reduction_count);
            break :blk narrowScalar(dtype, accumulator / divisor);
        },
    };
}
