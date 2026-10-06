const Plan = @import("../../execution/execution.zig");
const Program = @import("../../compiler/optimization/fusion/expression.zig").Program;
const evaluation = @import("evaluation.zig");

pub fn execute(
    comptime program: ?Program,
    comptime plan: Plan.MapPlan.LoopPlan,
    inputs: anytype,
    output: anytype,
) void {
    const Output = @TypeOf(output);
    const rank = Output.rank;
    const loop_axis: usize = plan.axis;
    const spatial_elements = output.len() / plan.iterations.len;

    for (0..spatial_elements) |linear_index| {
        var coordinates: [rank]usize = @splat(0);
        var remaining = linear_index;
        var axis = rank;
        while (axis > 0) {
            axis -= 1;
            if (axis == loop_axis) continue;
            coordinates[axis] = remaining % Output.static_shape[axis];
            remaining /= Output.static_shape[axis];
        }

        if (comptime plan.vector_width > 1 and loop_axis == rank - 1) {
            comptime var channel: usize = 0;
            inline while (channel + plan.vector_width <= plan.iterations.len) : (channel += plan.vector_width) {
                coordinates[loop_axis] = channel;
                const values = evaluateVector(Output.scalar_type, program, plan, inputs, Output.static_shape, coordinates);
                inline for (0..plan.vector_width) |lane| {
                    store(plan.axis, plan.iterations[channel + lane], channel + lane, coordinates, values[lane], output);
                }
            }
            inline while (channel < plan.iterations.len) : (channel += 1) {
                coordinates[loop_axis] = channel;
                store(
                    plan.axis,
                    plan.iterations[channel],
                    channel,
                    coordinates,
                    evaluateScalar(Output.scalar_type, program, plan, inputs, Output.static_shape, coordinates),
                    output,
                );
            }
            continue;
        }

        inline for (plan.iterations, 0..) |iteration, channel| {
            coordinates[loop_axis] = channel;
            store(
                plan.axis,
                iteration,
                channel,
                coordinates,
                evaluateScalar(Output.scalar_type, program, plan, inputs, Output.static_shape, coordinates),
                output,
            );
        }
    }
}

inline fn evaluateScalar(
    comptime Scalar: type,
    comptime program: ?Program,
    comptime plan: Plan.MapPlan.LoopPlan,
    inputs: anytype,
    comptime shape: anytype,
    coordinates: [shape.len]usize,
) Scalar {
    if (program) |expression| {
        return evaluation.evaluateAt(expression, plan.value, inputs, shape, coordinates);
    }
    return inputs[plan.value.input].storage[inputs[plan.value.input].elementOffset(coordinates)];
}

inline fn evaluateVector(
    comptime Scalar: type,
    comptime program: ?Program,
    comptime plan: Plan.MapPlan.LoopPlan,
    inputs: anytype,
    comptime shape: anytype,
    coordinates: [shape.len]usize,
) @Vector(plan.vector_width, Scalar) {
    if (program) |expression| {
        return evaluation.evaluateVectorAt(
            expression,
            plan.value,
            inputs,
            shape,
            coordinates,
            plan.axis,
            plan.vector_width,
        );
    }
    const input = inputs[plan.value.input].broadcastTo(shape.len, shape);
    const offset = input.elementOffset(coordinates);
    return input.storage[offset..][0..plan.vector_width].*;
}

inline fn store(
    comptime loop_axis: usize,
    comptime iteration: Plan.MapPlan.LoopPlan.Iteration,
    comptime channel: usize,
    source_coordinates: anytype,
    value: anytype,
    output: anytype,
) void {
    var destination = source_coordinates;
    var outside = false;
    var offset_axis: usize = 0;
    inline for (0..destination.len) |axis| {
        if (axis == loop_axis) continue;
        const extent = @TypeOf(output).static_shape[axis];
        const shifted = @as(isize, @intCast(source_coordinates[axis])) + iteration.offsets[offset_axis];
        offset_axis += 1;
        switch (iteration.boundary) {
            .wrap => destination[axis] = @intCast(@mod(shifted, @as(isize, @intCast(extent)))),
            .redirect => {
                if (shifted < 0 or shifted >= extent) {
                    outside = true;
                } else {
                    destination[axis] = @intCast(shifted);
                }
            },
        }
    }
    destination[loop_axis] = if (outside) switch (iteration.boundary) {
        .redirect => |redirect| blk: {
            inline for (0..destination.len) |axis| {
                if (axis != loop_axis) destination[axis] = source_coordinates[axis];
            }
            break :blk redirect;
        },
        .wrap => unreachable,
    } else channel;
    output.storage[output.elementOffset(destination)] = value;
}
