const std = @import("std");
const accumulation = @import("../accumulation.zig");
const Dtype = @import("../../storage/dtype.zig").Dtype;
const ReductionPlan = @import("../../execution/execution.zig").ReductionPlan;
const Expression = @import("../../compiler/optimization/fusion/expression.zig").Program;
const elementwise = @import("../elementwise_operation.zig");

/// Execute one statically planned multi-accumulator reduction region.
pub fn execute(comptime plan: ReductionPlan, inputs: anytype, outputs: anytype) void {
    if (outputs.len == 0) @compileError("reduction plan requires an output");
    if (comptime plan.traversal_plan.vector_axis != null and plan.traversal_plan.vector_width > 1) {
        executeVectorized(plan, inputs, outputs);
    } else {
        executeScalar(plan, inputs, outputs);
    }
}

fn executeScalar(comptime plan: ReductionPlan, inputs: anytype, outputs: anytype) void {
    const Output = @TypeOf(outputs[0]);
    const dtype = Output.dtype;
    const Accumulator = accumulation.AccumulatorScalar(dtype);
    const rank = plan.region.domain.shape.len;
    const domain_shape: [rank]usize = plan.region.domain.shape[0..rank].*;
    const reduction_count = comptime reducedElementCount(domain_shape, plan.region.reduction_axes);

    for (0..outputs[0].len()) |output_linear| {
        var accumulators: [plan.region.accumulators.len]Accumulator = undefined;
        inline for (plan.region.accumulators, 0..) |accumulator, index| {
            accumulators[index] = accumulation.identity(dtype, accumulator.combine);
        }

        for (0..reduction_count) |reduction_linear| {
            const coordinates = domainCoordinates(
                domain_shape,
                plan.region.reduction_axes,
                output_linear,
                reduction_linear,
            );
            const values = evaluateScalar(plan, inputs, coordinates);
            inline for (plan.region.accumulators, 0..) |accumulator, accumulator_index| {
                const value = resolveScalar(
                    dtype,
                    rank,
                    domain_shape,
                    accumulator.update,
                    inputs,
                    &values,
                    coordinates,
                );
                accumulators[accumulator_index] = accumulation.combine(
                    dtype,
                    accumulator.combine,
                    accumulators[accumulator_index],
                    value,
                );
            }
        }

        storeResults(plan, outputs, output_linear, accumulators, reduction_count);
    }
}

fn executeVectorized(comptime plan: ReductionPlan, inputs: anytype, outputs: anytype) void {
    const Output = @TypeOf(outputs[0]);
    const dtype = Output.dtype;
    const vector_width = plan.traversal_plan.vector_width;
    const vector_axis: usize = plan.traversal_plan.vector_axis.?;
    const Accumulator = accumulation.AccumulatorScalar(dtype);
    const VectorAccumulator = accumulation.AccumulatorVector(dtype, vector_width);
    const rank = plan.region.domain.shape.len;
    const domain_shape: [rank]usize = plan.region.domain.shape[0..rank].*;
    const reduction_count = comptime reducedElementCount(domain_shape, plan.region.reduction_axes);
    const vector_axis_extent = domain_shape[vector_axis];
    const reduction_outer_count = reduction_count / vector_axis_extent;

    for (0..outputs[0].len()) |output_linear| {
        var scalar_accumulators: [plan.region.accumulators.len]Accumulator = undefined;
        var vector_accumulators: [plan.region.accumulators.len]VectorAccumulator = undefined;
        inline for (plan.region.accumulators, 0..) |accumulator, index| {
            const identity = accumulation.identity(dtype, accumulator.combine);
            scalar_accumulators[index] = identity;
            vector_accumulators[index] = @splat(identity);
        }

        for (0..reduction_outer_count) |reduction_outer| {
            var coordinates = domainCoordinatesExcludingAxis(
                domain_shape,
                plan.region.reduction_axes,
                vector_axis,
                output_linear,
                reduction_outer,
            );
            var vector_index: usize = 0;
            while (vector_index + vector_width <= vector_axis_extent) : (vector_index += vector_width) {
                coordinates[vector_axis] = vector_index;
                const values = evaluateVector(plan, inputs, coordinates);
                inline for (plan.region.accumulators, 0..) |accumulator, accumulator_index| {
                    const value = resolveVector(
                        dtype,
                        rank,
                        domain_shape,
                        vector_axis,
                        vector_width,
                        accumulator.update,
                        inputs,
                        &values,
                        coordinates,
                    );
                    vector_accumulators[accumulator_index] = accumulation.combineVector(
                        dtype,
                        vector_width,
                        accumulator.combine,
                        vector_accumulators[accumulator_index],
                        value,
                    );
                }
            }

            while (vector_index < vector_axis_extent) : (vector_index += 1) {
                coordinates[vector_axis] = vector_index;
                const values = evaluateScalar(plan, inputs, coordinates);
                inline for (plan.region.accumulators, 0..) |accumulator, accumulator_index| {
                    const value = resolveScalar(
                        dtype,
                        rank,
                        domain_shape,
                        accumulator.update,
                        inputs,
                        &values,
                        coordinates,
                    );
                    scalar_accumulators[accumulator_index] = accumulation.combine(
                        dtype,
                        accumulator.combine,
                        scalar_accumulators[accumulator_index],
                        value,
                    );
                }
            }
        }

        inline for (plan.region.accumulators, 0..) |accumulator, accumulator_index| {
            scalar_accumulators[accumulator_index] = accumulation.combineAccumulator(
                dtype,
                accumulator.combine,
                scalar_accumulators[accumulator_index],
                accumulation.horizontal(
                    dtype,
                    vector_width,
                    accumulator.combine,
                    vector_accumulators[accumulator_index],
                ),
            );
        }
        storeResults(plan, outputs, output_linear, scalar_accumulators, reduction_count);
    }
}

fn evaluateScalar(
    comptime plan: ReductionPlan,
    inputs: anytype,
    coordinates: anytype,
) [plan.region.expressions.instructions.len]@TypeOf(inputs[0]).dtype.Scalar() {
    const dtype = @TypeOf(inputs[0]).dtype;
    const rank = plan.region.domain.shape.len;
    const domain_shape: [rank]usize = plan.region.domain.shape[0..rank].*;
    var values: [plan.region.expressions.instructions.len]dtype.Scalar() = undefined;
    inline for (plan.region.expressions.instructions, 0..) |instruction, instruction_index| {
        var params: [instruction.operation.arity()]dtype.Scalar() = undefined;
        inline for (instruction.args[0..params.len], 0..) |reference, param_index| {
            params[param_index] = resolveScalar(dtype, rank, domain_shape, reference, inputs, &values, coordinates);
        }
        values[instruction_index] = elementwise.evaluateScalar(dtype, instruction.operation, params);
    }
    return values;
}

fn evaluateVector(
    comptime plan: ReductionPlan,
    inputs: anytype,
    coordinates: anytype,
) [plan.region.expressions.instructions.len]@TypeOf(inputs[0]).dtype.Vector(plan.traversal_plan.vector_width) {
    const dtype = @TypeOf(inputs[0]).dtype;
    const rank = plan.region.domain.shape.len;
    const domain_shape: [rank]usize = plan.region.domain.shape[0..rank].*;
    const vector_axis: usize = plan.traversal_plan.vector_axis.?;
    const vector_width = plan.traversal_plan.vector_width;
    var values: [plan.region.expressions.instructions.len]dtype.Vector(vector_width) = undefined;
    inline for (plan.region.expressions.instructions, 0..) |instruction, instruction_index| {
        var params: [instruction.operation.arity()]dtype.Vector(vector_width) = undefined;
        inline for (instruction.args[0..params.len], 0..) |reference, param_index| {
            params[param_index] = resolveVector(
                dtype,
                rank,
                domain_shape,
                vector_axis,
                vector_width,
                reference,
                inputs,
                &values,
                coordinates,
            );
        }
        values[instruction_index] = elementwise.evaluate(dtype, vector_width, instruction.operation, params);
    }
    return values;
}

fn storeResults(
    comptime plan: ReductionPlan,
    outputs: anytype,
    output_linear: usize,
    accumulators: anytype,
    reduction_count: usize,
) void {
    const dtype = @TypeOf(outputs[0]).dtype;
    inline for (plan.region.stores) |store| {
        const accumulator_index = switch (store.value) {
            .accumulator => |index| index,
            .expression, .contraction, .transfer => @compileError("reduction stores must reference an accumulator"),
        };
        const accumulator = plan.region.accumulators[accumulator_index];
        const value = accumulation.finish(
            dtype,
            accumulator.finalize,
            accumulators[accumulator_index],
            reduction_count,
        );
        outputs[store.output].storage[outputs[store.output].elementOffsetFromLinear(output_linear)] = value;
    }
}

fn resolveScalar(
    comptime dtype: Dtype,
    comptime rank: usize,
    comptime domain_shape: [rank]usize,
    comptime reference: Expression.ValueRef,
    inputs: anytype,
    values: anytype,
    coordinates: [rank]usize,
) dtype.Scalar() {
    return switch (reference) {
        .input => |input_index| blk: {
            const view = inputs[input_index].broadcastTo(rank, domain_shape);
            break :blk view.storage[view.elementOffset(coordinates)];
        },
        .instruction => |instruction_index| values[instruction_index],
        .accumulator => @compileError("reduction body expressions cannot read accumulators"),
    };
}

fn resolveVector(
    comptime dtype: Dtype,
    comptime rank: usize,
    comptime domain_shape: [rank]usize,
    comptime vector_axis: usize,
    comptime vector_width: usize,
    comptime reference: Expression.ValueRef,
    inputs: anytype,
    values: anytype,
    coordinates: [rank]usize,
) dtype.Vector(vector_width) {
    return switch (reference) {
        .input => |input_index| blk: {
            const view = inputs[input_index].broadcastTo(rank, domain_shape);
            const offset = view.elementOffset(coordinates);
            if (comptime @TypeOf(view).static_strides[vector_axis] == 0) {
                break :blk @splat(view.storage[offset]);
            }
            if (comptime @TypeOf(view).static_strides[vector_axis] != 1) {
                @compileError("reduction SIMD input must be contiguous or broadcast on its vector axis");
            }
            break :blk view.storage[offset..][0..vector_width].*;
        },
        .instruction => |instruction_index| values[instruction_index],
        .accumulator => @compileError("reduction body expressions cannot read accumulators"),
    };
}

fn reducedElementCount(comptime shape: anytype, comptime axes: u64) usize {
    var count: usize = 1;
    for (shape, 0..) |extent, axis| {
        if (axes & (@as(u64, 1) << @intCast(axis)) != 0) count *= extent;
    }
    return count;
}

fn domainCoordinates(
    comptime shape: anytype,
    comptime reduction_axes: u64,
    output_linear: usize,
    reduction_linear: usize,
) [shape.len]usize {
    return domainCoordinatesExcludingAxis(shape, reduction_axes, null, output_linear, reduction_linear);
}

fn domainCoordinatesExcludingAxis(
    comptime shape: anytype,
    comptime reduction_axes: u64,
    comptime excluded_axis: ?usize,
    output_linear: usize,
    reduction_linear: usize,
) [shape.len]usize {
    var coordinates: [shape.len]usize = @splat(0);
    var outer_remaining = output_linear;
    var reduction_remaining = reduction_linear;
    var axis = shape.len;
    while (axis > 0) {
        axis -= 1;
        if (reduction_axes & (@as(u64, 1) << @intCast(axis)) != 0) {
            if (excluded_axis == null or axis != excluded_axis.?) {
                coordinates[axis] = reduction_remaining % shape[axis];
                reduction_remaining /= shape[axis];
            }
        } else {
            coordinates[axis] = outer_remaining % shape[axis];
            outer_remaining /= shape[axis];
        }
    }
    return coordinates;
}

test "SIMD reductions support multiple combine operations and scalar tails" {
    const Tensor = @import("../../core/tensor.zig");
    const vector_width = std.simd.suggestVectorLength(f32) orelse 1;
    const len = vector_width + 1;
    const Input = Tensor.StaticConstView(f32, .{len}, .{1}, 0);
    const Output = Tensor.StaticView(f32, .{}, .{}, 0);
    const plan: ReductionPlan = .{
        .region = .{
            .domain = .{ .shape = &.{len} },
            .loads = &.{.{ .input = 0 }},
            .expressions = .{ .instructions = &.{} },
            .reduction_axes = 1,
            .keep_dims = false,
            .accumulators = &.{
                .{ .combine = .minimum, .update = .{ .input = 0 }, .finalize = .identity },
                .{ .combine = .maximum, .update = .{ .input = 0 }, .finalize = .identity },
            },
            .stores = &.{
                .{ .output = 0, .value = .{ .accumulator = 0 } },
                .{ .output = 1, .value = .{ .accumulator = 1 } },
            },
        },
        .traversal_plan = .{
            .outer_axis_order = &.{},
            .reduction_axis_order = &.{0},
            .vector_axis = if (vector_width > 1) 0 else null,
            .vector_width = vector_width,
            .accumulator_lanes = 1,
        },
    };
    var input_storage: [len]f32 = undefined;
    for (&input_storage, 0..) |*value, index| value.* = @floatFromInt(index + 1);
    input_storage[len - 1] = -5;
    var minimum_storage: [1]f32 = undefined;
    var maximum_storage: [1]f32 = undefined;

    execute(
        plan,
        .{Input{ .storage = &input_storage }},
        .{
            Output{ .storage = &minimum_storage },
            Output{ .storage = &maximum_storage },
        },
    );

    try std.testing.expectEqual(@as(f32, -5), minimum_storage[0]);
    try std.testing.expectEqual(@as(f32, @floatFromInt(len - 1)), maximum_storage[0]);
}
