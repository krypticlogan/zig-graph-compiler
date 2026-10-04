const std = @import("std");
const MapPlan = @import("../../execution/kernel_plan.zig").MapPlan;
const ElementwiseProgram = @import("../../optimization/fusion/expression.zig").Program;
const Tensor = @import("../../tensor.zig");
const elementwise = @import("../elementwise_operation.zig");

/// Execute a compile-time elementwise program in one output traversal. The
/// instruction sequence and every value reference are unrolled at compile
/// time; no operation dispatch occurs at runtime.
pub fn execute(
    comptime plan: MapPlan,
    inputs: anytype,
    outputs: anytype,
) void {
    if (outputs.len != 1 or plan.region.stores.len != 1) {
        @compileError("map kernel currently requires exactly one output store");
    }
    executeProgram(plan.region.expressions, plan.traversal_plan.vector_width, inputs, outputs[0]);
}

fn executeProgram(
    comptime program: ElementwiseProgram,
    comptime vector_len: usize,
    inputs: anytype,
    output: anytype,
) void {
    const Output = @TypeOf(output);

    if (comptime canUseContiguousVectors(@TypeOf(inputs), Output)) {
        var index: usize = 0;
        while (index + vector_len <= output.len()) : (index += vector_len) {
            var values: VectorValues(program, vector_len) = undefined;
            inline for (program.instructions, 0..) |instruction, instruction_index| {
                var params: VectorParams(program, @TypeOf(inputs), instruction, vector_len) = undefined;
                inline for (instruction.args[0..params.len], 0..) |reference, param_index| {
                    params[param_index] = resolveVector(program, reference, inputs, &values, index, Output, vector_len);
                }
                values[instruction_index] = elementwise.evaluate(
                    instruction.dtype,
                    vector_len,
                    instruction.operation,
                    params,
                );
            }
            output.contiguousSlice().?[index..][0..vector_len].* = values[program.instructions.len - 1];
        }
        while (index < output.len()) : (index += 1) executeScalar(program, inputs, output, index);
        return;
    }

    for (0..output.len()) |linear_index| executeScalar(program, inputs, output, linear_index);
}

fn executeScalar(comptime program: ElementwiseProgram, inputs: anytype, output: anytype, linear_index: usize) void {
    const Output = @TypeOf(output);
    var values: ScalarValues(program) = undefined;
    inline for (program.instructions, 0..) |instruction, instruction_index| {
        var params: ScalarParams(program, @TypeOf(inputs), instruction) = undefined;
        inline for (instruction.args[0..params.len], 0..) |reference, param_index| {
            params[param_index] = resolveScalar(program, reference, inputs, &values, linear_index, Output);
        }
        values[instruction_index] = elementwise.evaluateScalar(instruction.dtype, instruction.operation, params);
    }
    output.storage[output.elementOffsetFromLinear(linear_index)] = values[program.instructions.len - 1];
}

fn resolveScalar(
    comptime program: ElementwiseProgram,
    comptime reference: ElementwiseProgram.ValueRef,
    inputs: anytype,
    values: anytype,
    linear_index: usize,
    comptime Output: type,
) ScalarReferenceType(program, @TypeOf(inputs), reference) {
    return switch (reference) {
        .input => |input_index| blk: {
            const view = inputs[input_index].broadcastTo(Output.rank, Output.static_shape);
            break :blk view.storage[view.elementOffsetFromLinear(linear_index)];
        },
        .instruction => |instruction_index| values[instruction_index],
        .accumulator => @compileError("map expressions cannot reference accumulators"),
    };
}

fn resolveVector(
    comptime program: ElementwiseProgram,
    comptime reference: ElementwiseProgram.ValueRef,
    inputs: anytype,
    values: anytype,
    index: usize,
    comptime Output: type,
    comptime vector_len: usize,
) VectorReferenceType(program, @TypeOf(inputs), reference, vector_len) {
    return switch (reference) {
        .input => |input_index| blk: {
            const view = inputs[input_index].broadcastTo(Output.rank, Output.static_shape);
            break :blk view.contiguousSlice().?[index..][0..vector_len].*;
        },
        .instruction => |instruction_index| values[instruction_index],
        .accumulator => @compileError("map expressions cannot reference accumulators"),
    };
}

fn ScalarValues(comptime program: ElementwiseProgram) type {
    var types: [program.instructions.len]type = undefined;
    for (program.instructions, 0..) |instruction, index| types[index] = instruction.dtype.Scalar();
    return @Tuple(&types);
}

fn VectorValues(comptime program: ElementwiseProgram, comptime vector_len: usize) type {
    var types: [program.instructions.len]type = undefined;
    for (program.instructions, 0..) |instruction, index| types[index] = instruction.dtype.Vector(vector_len);
    return @Tuple(&types);
}

fn ScalarParams(comptime program: ElementwiseProgram, comptime Inputs: type, comptime instruction: ElementwiseProgram.Instruction) type {
    var types: [instruction.operation.arity()]type = undefined;
    for (instruction.args[0..types.len], 0..) |reference, index| {
        types[index] = ScalarReferenceType(program, Inputs, reference);
    }
    return @Tuple(&types);
}

fn VectorParams(
    comptime program: ElementwiseProgram,
    comptime Inputs: type,
    comptime instruction: ElementwiseProgram.Instruction,
    comptime vector_len: usize,
) type {
    var types: [instruction.operation.arity()]type = undefined;
    for (instruction.args[0..types.len], 0..) |reference, index| {
        types[index] = VectorReferenceType(program, Inputs, reference, vector_len);
    }
    return @Tuple(&types);
}

fn ScalarReferenceType(
    comptime program: ElementwiseProgram,
    comptime Inputs: type,
    comptime reference: ElementwiseProgram.ValueRef,
) type {
    return switch (reference) {
        .input => |input_index| @typeInfo(Inputs).@"struct".field_types[input_index].scalar_type,
        .instruction => |instruction_index| program.instructions[instruction_index].dtype.Scalar(),
        .accumulator => @compileError("map expressions cannot reference accumulators"),
    };
}

fn VectorReferenceType(
    comptime program: ElementwiseProgram,
    comptime Inputs: type,
    comptime reference: ElementwiseProgram.ValueRef,
    comptime vector_len: usize,
) type {
    return switch (reference) {
        .input => |input_index| @typeInfo(Inputs).@"struct".field_types[input_index].dtype.Vector(vector_len),
        .instruction => |instruction_index| program.instructions[instruction_index].dtype.Vector(vector_len),
        .accumulator => @compileError("map expressions cannot reference accumulators"),
    };
}

fn canUseContiguousVectors(comptime Inputs: type, comptime Output: type) bool {
    if (!Output.static_is_contiguous) return false;
    inline for (@typeInfo(Inputs).@"struct".field_types) |Input| {
        const View = @TypeOf(@as(Input, undefined).broadcastTo(Output.rank, Output.static_shape));
        if (!View.static_is_contiguous) return false;
    }
    return true;
}

test "fused elementwise program evaluates mul followed by add" {
    const program: ElementwiseProgram = .{ .instructions = &.{
        .{ .operation = .mul, .dtype = .f32, .args = .{ .{ .input = 0 }, .{ .input = 1 }, .{ .input = 0 } } },
        .{ .operation = .add, .dtype = .f32, .args = .{ .{ .instruction = 0 }, .{ .input = 2 }, .{ .input = 0 } } },
    } };
    var a_values = [_]f32{ 1, 2, 3, 4 };
    var b_values = [_]f32{ 2, 3, 4, 5 };
    var c_values = [_]f32{ 10, 20, 30, 40 };
    var output_values: [4]f32 = undefined;
    const View = Tensor.StaticConstView(f32, .{4}, .{1}, 0);
    const Output = Tensor.StaticView(f32, .{4}, .{1}, 0);
    const a: View = .{ .storage = &a_values };
    const b: View = .{ .storage = &b_values };
    const c: View = .{ .storage = &c_values };
    const output: Output = .{ .storage = &output_values };

    executeProgram(program, std.simd.suggestVectorLength(f32) orelse 1, .{ a, b, c }, output);
    try std.testing.expectEqualSlices(f32, &.{ 12, 26, 42, 60 }, &output_values);
}

test "fused elementwise program preserves broadcast semantics" {
    const program: ElementwiseProgram = .{ .instructions = &.{
        .{ .operation = .mul, .dtype = .f32, .args = .{ .{ .input = 0 }, .{ .input = 1 }, .{ .input = 0 } } },
        .{ .operation = .add, .dtype = .f32, .args = .{ .{ .instruction = 0 }, .{ .input = 2 }, .{ .input = 0 } } },
    } };
    var matrix_values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var row_values = [_]f32{ 2, 3, 4 };
    var scalar_value = [_]f32{10};
    var output_values: [6]f32 = undefined;
    const Matrix = Tensor.StaticConstView(f32, .{ 2, 3 }, .{ 3, 1 }, 0);
    const Row = Tensor.StaticConstView(f32, .{3}, .{1}, 0);
    const Scalar = Tensor.StaticConstView(f32, .{}, .{}, 0);
    const Output = Tensor.StaticView(f32, .{ 2, 3 }, .{ 3, 1 }, 0);
    const matrix: Matrix = .{ .storage = &matrix_values };
    const row: Row = .{ .storage = &row_values };
    const scalar: Scalar = .{ .storage = &scalar_value };
    const output: Output = .{ .storage = &output_values };

    executeProgram(program, std.simd.suggestVectorLength(f32) orelse 1, .{ matrix, row, scalar }, output);
    try std.testing.expectEqualSlices(f32, &.{ 12, 16, 22, 18, 25, 34 }, &output_values);
}

test "fused elementwise execution ignores operands beyond operation arity" {
    const program: ElementwiseProgram = .{ .instructions = &.{.{
        .operation = .relu,
        .dtype = .f32,
        .args = .{ .{ .input = 0 }, .{ .input = 99 }, .{ .instruction = 99 } },
    }} };
    var input_values = [_]f32{ -2, 0, 3, -4 };
    var output_values: [4]f32 = undefined;
    const Input = Tensor.StaticConstView(f32, .{4}, .{1}, 0);
    const Output = Tensor.StaticView(f32, .{4}, .{1}, 0);

    executeProgram(
        program,
        std.simd.suggestVectorLength(f32) orelse 1,
        .{Input{ .storage = &input_values }},
        Output{ .storage = &output_values },
    );
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 3, 0 }, &output_values);
}
