const std = @import("std");
const MapPlan = @import("../../execution/execution.zig").MapPlan;
const ElementwiseProgram = @import("../../compiler/optimization/fusion/expression.zig").Program;
const Tensor = @import("../../core/tensor.zig");
const elementwise = @import("../elementwise_operation.zig");
const segmented = @import("segmented.zig");
const loop = @import("loop.zig");

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
    switch (plan.strategy) {
        .traversal => |traversal| switch (plan.region.body) {
            .expression => |program| executeProgram(program, traversal, inputs, outputs[0]),
            .transfer, .expression_transfer => @compileError("transfer map regions require a segmented strategy"),
        },
        .segmented => |transfer| switch (plan.region.body) {
            .transfer => segmented.execute(transfer, inputs, outputs[0]),
            .expression_transfer => |program| segmented.executeExpression(program, transfer, inputs, outputs[0]),
            .expression => @compileError("expression map regions require a traversal strategy"),
        },
        .loop => |loop_plan| switch (plan.region.body) {
            .transfer => loop.execute(null, loop_plan, inputs, outputs[0]),
            .expression_transfer => |program| loop.execute(program, loop_plan, inputs, outputs[0]),
            .expression => @compileError("loop maps require a transfer body"),
        },
    }
}

fn executeProgram(
    comptime program: ElementwiseProgram,
    comptime traversal: MapPlan.TraversalPlan,
    inputs: anytype,
    output: anytype,
) void {
    const Output = @TypeOf(output);
    const broadcast_inputs = broadcastInputs(inputs, Output);
    var input_offsets: [inputs.len]isize = undefined;
    inline for (std.meta.fields(@TypeOf(broadcast_inputs)), 0..) |field, index| {
        const view = broadcast_inputs[index];
        input_offsets[index] = @intCast(field.type.base_offset + view.runtime_offset);
    }
    executeAxes(
        program,
        traversal,
        0,
        broadcast_inputs,
        output,
        input_offsets,
        @intCast(Output.base_offset + output.runtime_offset),
    );
}

fn executeAxes(
    comptime program: ElementwiseProgram,
    comptime traversal: MapPlan.TraversalPlan,
    comptime level: usize,
    inputs: anytype,
    output: anytype,
    input_offsets: [inputs.len]isize,
    output_offset: isize,
) void {
    const Output = @TypeOf(output);
    if (comptime level == traversal.axis_order.len) {
        output.storage[@intCast(output_offset)] = evaluateScalarAtOffsets(program, inputs, input_offsets);
        return;
    }

    const axis: usize = traversal.axis_order[level];
    if (comptime traversal.vector_axis != null and traversal.vector_axis.? == axis) {
        var current_inputs = input_offsets;
        var current_output = output_offset;
        var index: usize = 0;
        while (index + traversal.vector_width <= Output.static_shape[axis]) : (index += traversal.vector_width) {
            output.storage[@intCast(current_output)..][0..traversal.vector_width].* = evaluateVectorAtOffsets(
                program,
                inputs,
                current_inputs,
                axis,
                traversal.vector_width,
            );
            advanceInputOffsets(@TypeOf(inputs), &current_inputs, axis, traversal.vector_width);
            current_output += Output.static_strides[axis] * @as(isize, @intCast(traversal.vector_width));
        }
        while (index < Output.static_shape[axis]) : (index += 1) {
            output.storage[@intCast(current_output)] = evaluateScalarAtOffsets(program, inputs, current_inputs);
            advanceInputOffsets(@TypeOf(inputs), &current_inputs, axis, 1);
            current_output += Output.static_strides[axis];
        }
        return;
    }

    var current_inputs = input_offsets;
    var current_output = output_offset;
    for (0..Output.static_shape[axis]) |_| {
        executeAxes(program, traversal, level + 1, inputs, output, current_inputs, current_output);
        advanceInputOffsets(@TypeOf(inputs), &current_inputs, axis, 1);
        current_output += Output.static_strides[axis];
    }
}

fn evaluateScalarAtOffsets(
    comptime program: ElementwiseProgram,
    inputs: anytype,
    input_offsets: [inputs.len]isize,
) ScalarReferenceType(program, @TypeOf(inputs), .{ .instruction = program.instructions.len - 1 }) {
    var values: ScalarValues(program) = undefined;
    inline for (program.instructions, 0..) |instruction, instruction_index| {
        var params: ScalarParams(program, @TypeOf(inputs), instruction) = undefined;
        inline for (instruction.args[0..params.len], 0..) |reference, param_index| {
            params[param_index] = resolveScalarOffset(program, reference, inputs, &values, input_offsets);
        }
        values[instruction_index] = elementwise.evaluateScalar(instruction.dtype, instruction.operation, params);
    }
    return values[program.instructions.len - 1];
}

fn evaluateVectorAtOffsets(
    comptime program: ElementwiseProgram,
    inputs: anytype,
    input_offsets: [inputs.len]isize,
    comptime vector_axis: usize,
    comptime vector_len: usize,
) VectorReferenceType(program, @TypeOf(inputs), .{ .instruction = program.instructions.len - 1 }, vector_len) {
    var values: VectorValues(program, vector_len) = undefined;
    inline for (program.instructions, 0..) |instruction, instruction_index| {
        var params: VectorParams(program, @TypeOf(inputs), instruction, vector_len) = undefined;
        inline for (instruction.args[0..params.len], 0..) |reference, param_index| {
            params[param_index] = resolveVectorOffset(
                program,
                reference,
                inputs,
                &values,
                input_offsets,
                vector_axis,
                vector_len,
            );
        }
        values[instruction_index] = elementwise.evaluate(
            instruction.dtype,
            vector_len,
            instruction.operation,
            params,
        );
    }
    return values[program.instructions.len - 1];
}

fn resolveScalarOffset(
    comptime program: ElementwiseProgram,
    comptime reference: ElementwiseProgram.ValueRef,
    inputs: anytype,
    values: anytype,
    input_offsets: [inputs.len]isize,
) ScalarReferenceType(program, @TypeOf(inputs), reference) {
    return switch (reference) {
        .input => |input_index| inputs[input_index].storage[@intCast(input_offsets[input_index])],
        .instruction => |instruction_index| values[instruction_index],
        .accumulator => @compileError("map expressions cannot reference accumulators"),
    };
}

fn resolveVectorOffset(
    comptime program: ElementwiseProgram,
    comptime reference: ElementwiseProgram.ValueRef,
    inputs: anytype,
    values: anytype,
    input_offsets: [inputs.len]isize,
    comptime vector_axis: usize,
    comptime vector_len: usize,
) VectorReferenceType(program, @TypeOf(inputs), reference, vector_len) {
    return switch (reference) {
        .input => |input_index| blk: {
            const View = std.meta.fields(@TypeOf(inputs))[input_index].type;
            const offset: usize = @intCast(input_offsets[input_index]);
            if (comptime View.static_strides[vector_axis] == 0) {
                break :blk @splat(inputs[input_index].storage[offset]);
            }
            break :blk inputs[input_index].storage[offset..][0..vector_len].*;
        },
        .instruction => |instruction_index| values[instruction_index],
        .accumulator => @compileError("map expressions cannot reference accumulators"),
    };
}

fn ScalarValues(comptime program: ElementwiseProgram) type {
    var types: [program.instructions.len]type = undefined;
    for (program.instructions, 0..) |instruction, index| types[index] = instruction.dtype.Scalar();
    return std.meta.Tuple(&types);
}

fn VectorValues(comptime program: ElementwiseProgram, comptime vector_len: usize) type {
    var types: [program.instructions.len]type = undefined;
    for (program.instructions, 0..) |instruction, index| types[index] = instruction.dtype.Vector(vector_len);
    return std.meta.Tuple(&types);
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
    return std.meta.Tuple(&types);
}

fn ScalarParams(
    comptime program: ElementwiseProgram,
    comptime Inputs: type,
    comptime instruction: ElementwiseProgram.Instruction,
) type {
    var types: [instruction.operation.arity()]type = undefined;
    for (instruction.args[0..types.len], 0..) |reference, index| {
        types[index] = ScalarReferenceType(program, Inputs, reference);
    }
    return std.meta.Tuple(&types);
}

fn ScalarReferenceType(
    comptime program: ElementwiseProgram,
    comptime Inputs: type,
    comptime reference: ElementwiseProgram.ValueRef,
) type {
    return switch (reference) {
        .input => |input_index| std.meta.fields(Inputs)[input_index].type.scalar_type,
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
        .input => |input_index| std.meta.fields(Inputs)[input_index].type.dtype.Vector(vector_len),
        .instruction => |instruction_index| program.instructions[instruction_index].dtype.Vector(vector_len),
        .accumulator => @compileError("map expressions cannot reference accumulators"),
    };
}

fn BroadcastInputs(comptime Inputs: type, comptime Output: type) type {
    var types: [std.meta.fields(Inputs).len]type = undefined;
    inline for (std.meta.fields(Inputs), 0..) |field, index| {
        types[index] = @TypeOf(@as(field.type, undefined).broadcastTo(Output.rank, Output.static_shape));
    }
    return std.meta.Tuple(&types);
}

fn broadcastInputs(inputs: anytype, comptime Output: type) BroadcastInputs(@TypeOf(inputs), Output) {
    var result: BroadcastInputs(@TypeOf(inputs), Output) = undefined;
    inline for (std.meta.fields(@TypeOf(inputs)), 0..) |_, index| {
        result[index] = inputs[index].broadcastTo(Output.rank, Output.static_shape);
    }
    return result;
}

fn advanceInputOffsets(
    comptime Inputs: type,
    offsets: *[std.meta.fields(Inputs).len]isize,
    comptime axis: usize,
    amount: usize,
) void {
    inline for (std.meta.fields(Inputs), 0..) |field, index| {
        offsets[index] += field.type.static_strides[axis] * @as(isize, @intCast(amount));
    }
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

    const vector_len = std.simd.suggestVectorLength(f32) orelse 1;
    const planned_vector_len = if (output_values.len >= vector_len) vector_len else 1;
    executeProgram(program, .{
        .axis_order = &.{0},
        .traversal = .contiguous,
        .vector_axis = if (planned_vector_len > 1) 0 else null,
        .vector_width = planned_vector_len,
    }, .{ a, b, c }, output);
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

    executeProgram(program, .{
        .axis_order = &.{ 0, 1 },
        .traversal = .contiguous,
        .vector_axis = null,
        .vector_width = 1,
    }, .{ matrix, row, scalar }, output);
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

    const vector_len = std.simd.suggestVectorLength(f32) orelse 1;
    const planned_vector_len = if (output_values.len >= vector_len) vector_len else 1;
    executeProgram(program, .{
        .axis_order = &.{0},
        .traversal = .contiguous,
        .vector_axis = if (planned_vector_len > 1) 0 else null,
        .vector_width = planned_vector_len,
    }, .{Input{ .storage = &input_values }}, Output{ .storage = &output_values });
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 3, 0 }, &output_values);
}
