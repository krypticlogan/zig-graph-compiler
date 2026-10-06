const std = @import("std");
const Program = @import("../../compiler/optimization/fusion/expression.zig").Program;
const elementwise = @import("../elementwise_operation.zig");

/// Evaluate one value from a compile-time expression program at logical domain
/// coordinates. Access composition uses this without materializing the
/// expression's original output tensor.
pub inline fn evaluateAt(
    comptime program: Program,
    comptime result: Program.ValueRef,
    inputs: anytype,
    comptime shape: anytype,
    coordinates: [shape.len]usize,
) ReferenceType(program, @TypeOf(inputs), result) {
    var values: Values(program) = undefined;
    inline for (program.instructions, 0..) |instruction, instruction_index| {
        var params: Params(program, @TypeOf(inputs), instruction) = undefined;
        inline for (instruction.args[0..params.len], 0..) |reference, param_index| {
            params[param_index] = resolve(program, reference, inputs, &values, shape, coordinates);
        }
        values[instruction_index] = elementwise.evaluateScalar(instruction.dtype, instruction.operation, params);
    }
    return resolve(program, result, inputs, &values, shape, coordinates);
}

/// Evaluate a vector of adjacent values along one logical domain axis. Input
/// access is either contiguous on that axis or broadcast from one scalar.
pub inline fn evaluateVectorAt(
    comptime program: Program,
    comptime result: Program.ValueRef,
    inputs: anytype,
    comptime shape: anytype,
    coordinates: [shape.len]usize,
    comptime vector_axis: usize,
    comptime vector_width: usize,
) VectorReferenceType(program, @TypeOf(inputs), result, vector_width) {
    var values: VectorValues(program, vector_width) = undefined;
    inline for (program.instructions, 0..) |instruction, instruction_index| {
        var params: VectorParams(program, @TypeOf(inputs), instruction, vector_width) = undefined;
        inline for (instruction.args[0..params.len], 0..) |reference, param_index| {
            params[param_index] = resolveVector(
                program,
                reference,
                inputs,
                &values,
                shape,
                coordinates,
                vector_axis,
                vector_width,
            );
        }
        values[instruction_index] = elementwise.evaluate(
            instruction.dtype,
            vector_width,
            instruction.operation,
            params,
        );
    }
    return resolveVector(
        program,
        result,
        inputs,
        &values,
        shape,
        coordinates,
        vector_axis,
        vector_width,
    );
}

pub inline fn coordinatesFromLinear(comptime shape: anytype, linear_index: usize) [shape.len]usize {
    var coordinates: [shape.len]usize = @splat(0);
    var remaining = linear_index;
    var axis = shape.len;
    while (axis > 0) {
        axis -= 1;
        coordinates[axis] = remaining % shape[axis];
        remaining /= shape[axis];
    }
    return coordinates;
}

inline fn resolve(
    comptime program: Program,
    comptime reference: Program.ValueRef,
    inputs: anytype,
    values: anytype,
    comptime shape: anytype,
    coordinates: [shape.len]usize,
) ReferenceType(program, @TypeOf(inputs), reference) {
    return switch (reference) {
        .input => |input_index| blk: {
            const view = inputs[input_index].broadcastTo(shape.len, shape);
            break :blk view.storage[view.elementOffset(coordinates)];
        },
        .instruction => |instruction_index| values[instruction_index],
        .accumulator => @compileError("map expressions cannot reference accumulators"),
    };
}

fn Values(comptime program: Program) type {
    var types: [program.instructions.len]type = undefined;
    for (program.instructions, 0..) |instruction, index| types[index] = instruction.dtype.Scalar();
    return std.meta.Tuple(&types);
}

inline fn resolveVector(
    comptime program: Program,
    comptime reference: Program.ValueRef,
    inputs: anytype,
    values: anytype,
    comptime shape: anytype,
    coordinates: [shape.len]usize,
    comptime vector_axis: usize,
    comptime vector_width: usize,
) VectorReferenceType(program, @TypeOf(inputs), reference, vector_width) {
    return switch (reference) {
        .input => |input_index| blk: {
            const view = inputs[input_index].broadcastTo(shape.len, shape);
            const offset = view.elementOffset(coordinates);
            if (comptime @TypeOf(view).static_strides[vector_axis] == 0) {
                break :blk @splat(view.storage[offset]);
            }
            if (comptime @TypeOf(view).static_strides[vector_axis] != 1) {
                @compileError("vector map input must be contiguous or broadcast on its vector axis");
            }
            break :blk view.storage[offset..][0..vector_width].*;
        },
        .instruction => |instruction_index| values[instruction_index],
        .accumulator => @compileError("map expressions cannot reference accumulators"),
    };
}

fn VectorValues(comptime program: Program, comptime vector_width: usize) type {
    var types: [program.instructions.len]type = undefined;
    for (program.instructions, 0..) |instruction, index| {
        types[index] = instruction.dtype.Vector(vector_width);
    }
    return std.meta.Tuple(&types);
}

fn Params(comptime program: Program, comptime Inputs: type, comptime instruction: Program.Instruction) type {
    var types: [instruction.operation.arity()]type = undefined;
    for (instruction.args[0..types.len], 0..) |reference, index| {
        types[index] = ReferenceType(program, Inputs, reference);
    }
    return std.meta.Tuple(&types);
}

fn VectorParams(
    comptime program: Program,
    comptime Inputs: type,
    comptime instruction: Program.Instruction,
    comptime vector_width: usize,
) type {
    var types: [instruction.operation.arity()]type = undefined;
    for (instruction.args[0..types.len], 0..) |reference, index| {
        types[index] = VectorReferenceType(program, Inputs, reference, vector_width);
    }
    return std.meta.Tuple(&types);
}

fn ReferenceType(comptime program: Program, comptime Inputs: type, comptime reference: Program.ValueRef) type {
    return switch (reference) {
        .input => |input_index| std.meta.fields(Inputs)[input_index].type.scalar_type,
        .instruction => |instruction_index| program.instructions[instruction_index].dtype.Scalar(),
        .accumulator => @compileError("map expressions cannot reference accumulators"),
    };
}

fn VectorReferenceType(
    comptime program: Program,
    comptime Inputs: type,
    comptime reference: Program.ValueRef,
    comptime vector_width: usize,
) type {
    return switch (reference) {
        .input => |input_index| std.meta.fields(Inputs)[input_index].type.dtype.Vector(vector_width),
        .instruction => |instruction_index| program.instructions[instruction_index].dtype.Vector(vector_width),
        .accumulator => @compileError("map expressions cannot reference accumulators"),
    };
}
