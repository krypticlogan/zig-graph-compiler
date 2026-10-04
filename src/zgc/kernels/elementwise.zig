const std = @import("std");
const Dtype = @import("../storage/dtype.zig").Dtype;
const Operation = @import("../operations/elementwise.zig").Operation;
const operation = @import("elementwise_operation.zig");

/// Apply an element-wise operator through a contiguous SIMD fast path or a generic strided
/// traversal when any participating view is non-contiguous.
fn unary(input: anytype, output: anytype, comptime op: Operation) void {
    const Input = @TypeOf(input);
    const Output = @TypeOf(output);
    if (comptime hasStaticGeometry(Input) and hasStaticGeometry(Output)) {
        if (comptime std.mem.eql(isize, &Input.static_strides, &Output.static_strides) and
            Input.static_is_dense_positive and Output.static_is_dense_positive)
        {
            applyUnaryDense(input.denseSlice().?, output.denseSlice().?, op);
            return;
        }
    } else {
        if (std.mem.eql(isize, &input.strides, &output.strides)) {
            if (input.denseSlice()) |input_storage| {
                if (output.denseSlice()) |output_storage| {
                    applyUnaryDense(input_storage, output_storage, op);
                    return;
                }
            }
        }
    }

    for (0..input.len()) |linear_index| {
        const input_index = input.elementOffsetFromLinear(linear_index);
        const output_index = output.elementOffsetFromLinear(linear_index);
        const value = input.storage[input_index];
        output.storage[output_index] = operation.evaluateScalar(
            Output.dtype,
            op,
            .{value},
        );
    }
}

fn applyUnaryDense(
    input_storage: anytype,
    output_storage: anytype,
    comptime op: Operation,
) void {
    const InputScalar = @typeInfo(@TypeOf(input_storage)).pointer.child;
    const OutputScalar = @typeInfo(@TypeOf(output_storage)).pointer.child;
    const input_dtype = comptime Dtype.fromScalar(InputScalar);
    const output_dtype = comptime Dtype.fromScalar(OutputScalar);
    const vector_len = comptime vectorLength(InputScalar);

    var index: usize = 0;
    while (index + vector_len <= input_storage.len) : (index += vector_len) {
        const values: input_dtype.Vector(vector_len) = input_storage[index..][0..vector_len].*;
        output_storage[index..][0..vector_len].* = operation.evaluate(output_dtype, vector_len, op, .{values});
    }
    while (index < input_storage.len) : (index += 1) {
        const value = input_storage[index];
        output_storage[index] = operation.evaluateScalar(output_dtype, op, .{value});
    }
}

/// Apply a element-wise binary operator through a contiguous SIMD fast path or a generic
/// strided traversal.
fn binary(a: anytype, b: anytype, output: anytype, comptime op: Operation) void {
    const Output = @TypeOf(output);

    const output_shape = if (comptime hasStaticGeometry(Output))
        Output.static_shape
    else
        output.shape;
    const a_view = a.broadcastTo(Output.rank, output_shape);
    const b_view = b.broadcastTo(Output.rank, output_shape);
    const AView = @TypeOf(a_view);
    const BView = @TypeOf(b_view);

    if (comptime hasStaticGeometry(AView) and
        hasStaticGeometry(BView) and
        hasStaticGeometry(Output))
    {
        if (comptime std.mem.eql(isize, &AView.static_strides, &BView.static_strides) and
            std.mem.eql(isize, &AView.static_strides, &Output.static_strides) and
            AView.static_is_dense_positive and
            BView.static_is_dense_positive and
            Output.static_is_dense_positive)
        {
            applyBinaryDense(
                a_view.denseSlice().?,
                b_view.denseSlice().?,
                output.denseSlice().?,
                op,
            );
            return;
        }
    } else {
        if (std.mem.eql(isize, &a_view.strides, &b_view.strides) and
            std.mem.eql(isize, &a_view.strides, &output.strides))
        {
            if (a_view.denseSlice()) |a_storage| {
                if (b_view.denseSlice()) |b_storage| {
                    if (output.denseSlice()) |output_storage| {
                        applyBinaryDense(a_storage, b_storage, output_storage, op);
                        return;
                    }
                }
            }
        }
    }

    if (comptime Output.rank == 2) {
        if (hasSameLayout(a_view, output) and isTrailingVectorBroadcast(b_view) and output.denseSlice() != null) {
            applyFirstAxisBroadcast(a_view, b_view, output, op, true);
            return;
        }
        if (isTrailingVectorBroadcast(a_view) and hasSameLayout(b_view, output) and output.denseSlice() != null) {
            applyFirstAxisBroadcast(b_view, a_view, output, op, false);
            return;
        }
    }

    if (a_view.contiguousSlice()) |a_storage| {
        if (b_view.contiguousSlice()) |b_storage| {
            if (output.contiguousSlice()) |output_storage| {
                applyBinaryDense(a_storage, b_storage, output_storage, op);
                return;
            }
        }
    }

    for (0..output.len()) |linear_index| {
        const a_index = a_view.elementOffsetFromLinear(linear_index);
        const b_index = b_view.elementOffsetFromLinear(linear_index);
        const output_index = output.elementOffsetFromLinear(linear_index);
        output.storage[output_index] = operation.evaluateScalar(
            Output.dtype,
            op,
            .{ a_view.storage[a_index], b_view.storage[b_index] },
        );
    }
}

fn hasSameLayout(input: anytype, output: anytype) bool {
    const Input = @TypeOf(input);
    const Output = @TypeOf(output);
    if (comptime hasStaticGeometry(Input) and hasStaticGeometry(Output)) {
        return comptime std.mem.eql(isize, &Input.static_strides, &Output.static_strides) and
            Input.static_is_dense_positive;
    }
    return std.mem.eql(isize, &input.strides, &output.strides) and
        input.denseSlice() != null;
}

fn isTrailingVectorBroadcast(view: anytype) bool {
    const View = @TypeOf(view);
    if (comptime hasStaticGeometry(View)) {
        return comptime View.static_strides[0] == 0 and View.static_strides[1] == 1;
    }
    return view.strides[0] == 0 and view.strides[1] == 1;
}

fn hasStaticGeometry(comptime View: type) bool {
    return @hasDecl(View, "geometry_is_static") and View.geometry_is_static;
}

/// Apply a trailing vector across a dense [batch, width] tensor whose first
/// axis is contiguous. SIMD lanes traverse independent batch rows while the
/// broadcast value is splatted once per output column.
fn applyFirstAxisBroadcast(
    dense: anytype,
    broadcast: anytype,
    output: anytype,
    comptime op: Operation,
    comptime dense_is_a: bool,
) void {
    const DenseScalar = @TypeOf(dense).scalar_type;
    const Output = @TypeOf(output);
    const dense_dtype = @TypeOf(dense).dtype;
    const broadcast_dtype = @TypeOf(broadcast).dtype;
    const vector_len = comptime vectorLength(DenseScalar);
    const batch = output.shape[0];
    const width = output.shape[1];

    for (0..width) |column| {
        const broadcast_vector: broadcast_dtype.Vector(vector_len) = @splat(broadcast.get(.{ 0, column }));
        var row: usize = 0;
        while (row + vector_len <= batch) : (row += vector_len) {
            const dense_offset = dense.elementOffset(.{ row, column });
            const output_offset = output.elementOffset(.{ row, column });
            const dense_values: dense_dtype.Vector(vector_len) = dense.storage[dense_offset..][0..vector_len].*;
            output.storage[output_offset..][0..vector_len].* = if (dense_is_a)
                operation.evaluate(Output.dtype, vector_len, op, .{ dense_values, broadcast_vector })
            else
                operation.evaluate(Output.dtype, vector_len, op, .{ broadcast_vector, dense_values });
        }
        while (row < batch) : (row += 1) {
            const dense_value = dense.get(.{ row, column });
            const broadcast_value = broadcast.get(.{ 0, column });
            output.set(
                .{ row, column },
                if (dense_is_a)
                    operation.evaluateScalar(Output.dtype, op, .{ dense_value, broadcast_value })
                else
                    operation.evaluateScalar(Output.dtype, op, .{ broadcast_value, dense_value }),
            );
        }
    }
}

fn applyBinaryDense(
    a_storage: anytype,
    b_storage: anytype,
    output_storage: anytype,
    comptime op: Operation,
) void {
    const AScalar = @typeInfo(@TypeOf(a_storage)).pointer.child;
    const BScalar = @typeInfo(@TypeOf(b_storage)).pointer.child;
    const OutputScalar = @typeInfo(@TypeOf(output_storage)).pointer.child;
    const a_dtype = comptime Dtype.fromScalar(AScalar);
    const b_dtype = comptime Dtype.fromScalar(BScalar);
    const output_dtype = comptime Dtype.fromScalar(OutputScalar);
    const vector_len = comptime vectorLength(AScalar);

    var index: usize = 0;
    while (index + vector_len <= a_storage.len) : (index += vector_len) {
        const a_values: a_dtype.Vector(vector_len) = a_storage[index..][0..vector_len].*;
        const b_values: b_dtype.Vector(vector_len) = b_storage[index..][0..vector_len].*;
        output_storage[index..][0..vector_len].* = operation.evaluate(
            output_dtype,
            vector_len,
            op,
            .{ a_values, b_values },
        );
    }
    while (index < a_storage.len) : (index += 1) {
        output_storage[index] = operation.evaluateScalar(output_dtype, op, .{ a_storage[index], b_storage[index] });
    }
}

fn ternary(a: anytype, b: anytype, c: anytype, output: anytype, comptime op: Operation) void {
    const Output = @TypeOf(output);
    const output_shape = if (comptime hasStaticGeometry(Output)) Output.static_shape else output.shape;
    const a_view = a.broadcastTo(Output.rank, output_shape);
    const b_view = b.broadcastTo(Output.rank, output_shape);
    const c_view = c.broadcastTo(Output.rank, output_shape);
    const AView = @TypeOf(a_view);
    const BView = @TypeOf(b_view);
    const CView = @TypeOf(c_view);

    if (comptime hasStaticGeometry(AView) and
        hasStaticGeometry(BView) and
        hasStaticGeometry(CView) and
        hasStaticGeometry(Output))
    {
        if (comptime std.mem.eql(isize, &AView.static_strides, &BView.static_strides) and
            std.mem.eql(isize, &AView.static_strides, &CView.static_strides) and
            std.mem.eql(isize, &AView.static_strides, &Output.static_strides) and
            AView.static_is_dense_positive and
            BView.static_is_dense_positive and
            CView.static_is_dense_positive and
            Output.static_is_dense_positive)
        {
            applyTernaryDense(
                a_view.denseSlice().?,
                b_view.denseSlice().?,
                c_view.denseSlice().?,
                output.denseSlice().?,
                op,
            );
            return;
        }
    } else if (std.mem.eql(isize, &a_view.strides, &b_view.strides) and
        std.mem.eql(isize, &a_view.strides, &c_view.strides) and
        std.mem.eql(isize, &a_view.strides, &output.strides))
    {
        if (a_view.denseSlice()) |a_storage| {
            if (b_view.denseSlice()) |b_storage| {
                if (c_view.denseSlice()) |c_storage| {
                    if (output.denseSlice()) |output_storage| {
                        applyTernaryDense(a_storage, b_storage, c_storage, output_storage, op);
                        return;
                    }
                }
            }
        }
    }

    for (0..output.len()) |linear_index| {
        const a_index = a_view.elementOffsetFromLinear(linear_index);
        const b_index = b_view.elementOffsetFromLinear(linear_index);
        const c_index = c_view.elementOffsetFromLinear(linear_index);
        const output_index = output.elementOffsetFromLinear(linear_index);
        output.storage[output_index] = operation.evaluateScalar(
            Output.dtype,
            op,
            .{ a_view.storage[a_index], b_view.storage[b_index], c_view.storage[c_index] },
        );
    }
}

fn applyTernaryDense(
    a_storage: anytype,
    b_storage: anytype,
    c_storage: anytype,
    output_storage: anytype,
    comptime op: Operation,
) void {
    const AScalar = @typeInfo(@TypeOf(a_storage)).pointer.child;
    const BScalar = @typeInfo(@TypeOf(b_storage)).pointer.child;
    const CScalar = @typeInfo(@TypeOf(c_storage)).pointer.child;
    const OutputScalar = @typeInfo(@TypeOf(output_storage)).pointer.child;
    const a_dtype = comptime Dtype.fromScalar(AScalar);
    const b_dtype = comptime Dtype.fromScalar(BScalar);
    const c_dtype = comptime Dtype.fromScalar(CScalar);
    const output_dtype = comptime Dtype.fromScalar(OutputScalar);
    const vector_len = comptime vectorLength(OutputScalar);

    var index: usize = 0;
    while (index + vector_len <= output_storage.len) : (index += vector_len) {
        const a_values: a_dtype.Vector(vector_len) = a_storage[index..][0..vector_len].*;
        const b_values: b_dtype.Vector(vector_len) = b_storage[index..][0..vector_len].*;
        const c_values: c_dtype.Vector(vector_len) = c_storage[index..][0..vector_len].*;
        output_storage[index..][0..vector_len].* = operation.evaluate(
            output_dtype,
            vector_len,
            op,
            .{ a_values, b_values, c_values },
        );
    }
    while (index < output_storage.len) : (index += 1) {
        output_storage[index] = operation.evaluateScalar(
            output_dtype,
            op,
            .{ a_storage[index], b_storage[index], c_storage[index] },
        );
    }
}

fn vectorLength(comptime Scalar: type) usize {
    const NativeScalar = if (Scalar == bool) u8 else Scalar;
    return std.simd.suggestVectorLength(NativeScalar) orelse 1;
}

pub fn execute(comptime op: Operation, inputs: anytype, output: anytype) void {
    switch (comptime op.arity()) {
        1 => unary(inputs[0], output, op),
        2 => binary(inputs[0], inputs[1], output, op),
        3 => ternary(inputs[0], inputs[1], inputs[2], output, op),
        else => @compileError("unsupported pointwise operation arity"),
    }
}
