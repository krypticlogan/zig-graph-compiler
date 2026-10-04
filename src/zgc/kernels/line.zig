const Dtype = @import("../storage/dtype.zig").Dtype;
const accumulation = @import("accumulation.zig");
const simd = @import("simd.zig");

/// Reduce one logical rank-one view. Contiguous lines use the operator's SIMD
/// implementation; strided lines use the same operator's scalar implementation.
fn reduce(input: anytype, context: anytype, comptime Operator: type) accumulation.AccumulatorScalar(@TypeOf(input).dtype) {
    const Input = @TypeOf(input);
    if (comptime hasStaticGeometry(Input)) {
        if (comptime Input.static_is_contiguous) {
            return simd.reduce(Input.dtype, input.contiguousSlice().?, context, Operator);
        }
    } else {
        if (input.contiguousSlice()) |values| {
            return simd.reduce(Input.dtype, values, context, Operator);
        }
    }

    var result = Operator.identity(Input.dtype, context);
    for (0..input.len()) |index| {
        result = Operator.scalar(
            Input.dtype,
            result,
            input.get(.{index}),
            context,
        );
    }
    return result;
}

/// Map between equally-sized logical lines with contiguous SIMD and generic
/// strided paths driven by one scalar/vector operator definition.
fn map(input: anytype, output: anytype, context: anytype, comptime Operator: type) void {
    const Input = @TypeOf(input);
    const Output = @TypeOf(output);
    if (comptime hasStaticGeometry(Input) and hasStaticGeometry(Output)) {
        if (comptime Input.static_is_contiguous and Output.static_is_contiguous) {
            simd.map(
                Input.dtype,
                input.contiguousSlice().?,
                output.contiguousSlice().?,
                context,
                Operator,
            );
            return;
        }
    } else {
        if (input.contiguousSlice()) |input_values| {
            if (output.contiguousSlice()) |output_values| {
                simd.map(Input.dtype, input_values, output_values, context, Operator);
                return;
            }
        }
    }

    for (0..input.len()) |index| {
        output.set(
            .{index},
            Operator.scalar(Input.dtype, input.get(.{index}), context),
        );
    }
}

fn hasStaticGeometry(comptime View: type) bool {
    return @hasDecl(View, "geometry_is_static") and View.geometry_is_static;
}

pub fn sum(input: anytype) accumulation.AccumulatorScalar(@TypeOf(input).dtype) {
    return reduce(input, {}, struct {
        pub fn identity(comptime dtype: Dtype, _: void) accumulation.AccumulatorScalar(dtype) {
            return 0;
        }

        pub fn scalar(
            comptime dtype: Dtype,
            accumulator: accumulation.AccumulatorScalar(dtype),
            value: dtype.Scalar(),
            _: void,
        ) accumulation.AccumulatorScalar(dtype) {
            return accumulator + accumulation.widenScalar(dtype, value);
        }

        pub fn vector(
            comptime dtype: Dtype,
            comptime len: usize,
            accumulator: accumulation.AccumulatorVector(dtype, len),
            values: dtype.Vector(len),
            _: void,
        ) accumulation.AccumulatorVector(dtype, len) {
            return accumulator + accumulation.widenVector(dtype, len, values);
        }

        pub fn horizontal(
            comptime dtype: Dtype,
            comptime len: usize,
            accumulator: accumulation.AccumulatorVector(dtype, len),
        ) accumulation.AccumulatorScalar(dtype) {
            return @reduce(.Add, accumulator);
        }
    });
}

pub fn max(input: anytype) accumulation.AccumulatorScalar(@TypeOf(input).dtype) {
    return reduce(input, {}, struct {
        pub fn identity(comptime dtype: Dtype, _: void) accumulation.AccumulatorScalar(dtype) {
            return accumulation.identity(dtype, .maximum);
        }

        pub fn scalar(
            comptime dtype: Dtype,
            accumulator: accumulation.AccumulatorScalar(dtype),
            value: dtype.Scalar(),
            _: void,
        ) accumulation.AccumulatorScalar(dtype) {
            return @max(accumulator, accumulation.widenScalar(dtype, value));
        }

        pub fn vector(
            comptime dtype: Dtype,
            comptime len: usize,
            accumulator: accumulation.AccumulatorVector(dtype, len),
            values: dtype.Vector(len),
            _: void,
        ) accumulation.AccumulatorVector(dtype, len) {
            return @max(
                accumulator,
                accumulation.widenVector(dtype, len, values),
            );
        }

        pub fn horizontal(
            comptime dtype: Dtype,
            comptime len: usize,
            accumulator: accumulation.AccumulatorVector(dtype, len),
        ) accumulation.AccumulatorScalar(dtype) {
            return @reduce(.Max, accumulator);
        }
    });
}

pub fn min(input: anytype) accumulation.AccumulatorScalar(@TypeOf(input).dtype) {
    return reduce(input, {}, struct {
        pub fn identity(comptime dtype: Dtype, _: void) accumulation.AccumulatorScalar(dtype) {
            return accumulation.identity(dtype, .minimum);
        }

        pub fn scalar(
            comptime dtype: Dtype,
            accumulator: accumulation.AccumulatorScalar(dtype),
            value: dtype.Scalar(),
            _: void,
        ) accumulation.AccumulatorScalar(dtype) {
            return @min(accumulator, accumulation.widenScalar(dtype, value));
        }

        pub fn vector(
            comptime dtype: Dtype,
            comptime len: usize,
            accumulator: accumulation.AccumulatorVector(dtype, len),
            values: dtype.Vector(len),
            _: void,
        ) accumulation.AccumulatorVector(dtype, len) {
            return @min(
                accumulator,
                accumulation.widenVector(dtype, len, values),
            );
        }

        pub fn horizontal(
            comptime dtype: Dtype,
            comptime len: usize,
            accumulator: accumulation.AccumulatorVector(dtype, len),
        ) accumulation.AccumulatorScalar(dtype) {
            return @reduce(.Min, accumulator);
        }
    });
}

pub fn shiftedExpSum(
    input: anytype,
    shift: accumulation.AccumulatorScalar(@TypeOf(input).dtype),
) accumulation.AccumulatorScalar(@TypeOf(input).dtype) {
    return reduce(input, shift, struct {
        pub fn identity(comptime dtype: Dtype, _: accumulation.AccumulatorScalar(dtype)) accumulation.AccumulatorScalar(dtype) {
            return 0;
        }

        pub fn scalar(
            comptime dtype: Dtype,
            accumulator: accumulation.AccumulatorScalar(dtype),
            value: dtype.Scalar(),
            scalar_shift: accumulation.AccumulatorScalar(dtype),
        ) accumulation.AccumulatorScalar(dtype) {
            return accumulator + @exp(
                accumulation.widenScalar(dtype, value) - scalar_shift,
            );
        }

        pub fn vector(
            comptime dtype: Dtype,
            comptime len: usize,
            accumulator: accumulation.AccumulatorVector(dtype, len),
            values: dtype.Vector(len),
            scalar_shift: accumulation.AccumulatorScalar(dtype),
        ) accumulation.AccumulatorVector(dtype, len) {
            const shift_vector: accumulation.AccumulatorVector(dtype, len) =
                @splat(scalar_shift);
            return accumulator + @exp(
                accumulation.widenVector(dtype, len, values) - shift_vector,
            );
        }

        pub fn horizontal(
            comptime dtype: Dtype,
            comptime len: usize,
            accumulator: accumulation.AccumulatorVector(dtype, len),
        ) accumulation.AccumulatorScalar(dtype) {
            return @reduce(.Add, accumulator);
        }
    });
}

pub fn normalizedExp(
    input: anytype,
    output: anytype,
    shift: accumulation.AccumulatorScalar(@TypeOf(input).dtype),
    denominator: accumulation.AccumulatorScalar(@TypeOf(input).dtype),
) void {
    const Input = @TypeOf(input);
    const Context = struct {
        shift: accumulation.AccumulatorScalar(Input.dtype),
        denominator: accumulation.AccumulatorScalar(Input.dtype),
    };
    map(input, output, Context{
        .shift = shift,
        .denominator = denominator,
    }, struct {
        pub fn scalar(
            comptime dtype: Dtype,
            value: dtype.Scalar(),
            context: Context,
        ) dtype.Scalar() {
            const normalized = @exp(
                accumulation.widenScalar(dtype, value) - context.shift,
            ) / context.denominator;
            return accumulation.narrowScalar(dtype, normalized);
        }

        pub fn vector(
            comptime dtype: Dtype,
            comptime len: usize,
            values: dtype.Vector(len),
            context: Context,
        ) dtype.Vector(len) {
            const shift_vector: accumulation.AccumulatorVector(dtype, len) =
                @splat(context.shift);
            const denominator_vector: accumulation.AccumulatorVector(dtype, len) =
                @splat(context.denominator);
            const normalized = @exp(
                accumulation.widenVector(dtype, len, values) - shift_vector,
            ) / denominator_vector;
            return accumulation.narrowVector(dtype, len, normalized);
        }
    });
}
