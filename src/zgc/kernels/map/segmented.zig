const std = @import("std");
const Plan = @import("../../execution/execution.zig");
const Program = @import("../../compiler/optimization/fusion/expression.zig").Program;
const evaluation = @import("evaluation.zig");

/// Execute a statically validated segmented map. Segment geometry carries
/// every source and destination decision made during lowering.
pub fn execute(comptime plan: Plan.MapPlan.SegmentedPlan, inputs: anytype, output: anytype) void {
    executeWithProgram(null, plan, inputs, output);
}

pub fn executeExpression(
    comptime program: Program,
    comptime plan: Plan.MapPlan.SegmentedPlan,
    inputs: anytype,
    output: anytype,
) void {
    executeWithProgram(program, plan, inputs, output);
}

fn executeWithProgram(
    comptime program: ?Program,
    comptime plan: Plan.MapPlan.SegmentedPlan,
    inputs: anytype,
    output: anytype,
) void {
    inline for (plan.segments) |segment| {
        const element_count = comptime segment.elementCount();
        if (comptime segment.expression_value != null) {
            const expression = program orelse @compileError("expression segment requires a map expression program");
            if (comptime canVectorizeExpressionSegment(
                expression,
                segment,
                @TypeOf(inputs),
                plan.vector_width,
            )) {
                executeVectorExpressionSegment(expression, segment, plan.vector_width, inputs, output);
            } else {
                executeScalarExpressionSegment(expression, segment, inputs, output);
            }
            continue;
        }

        const input = inputs[segment.input];
        if (comptime isContiguous(segment.source_strides, segment.extents, segment.rank) and
            isContiguous(segment.destination_strides, segment.extents, segment.rank))
        {
            @memcpy(
                output.storage[segment.destination_offset..][0..element_count],
                input.storage[segment.source_offset..][0..element_count],
            );
            continue;
        }

        for (0..element_count) |linear_index| {
            var remaining = linear_index;
            var source_offset: isize = @intCast(segment.source_offset);
            var destination_offset: isize = @intCast(segment.destination_offset);
            comptime var axis: usize = segment.rank;
            inline while (axis > 0) {
                axis -= 1;
                const coordinate = remaining % segment.extents[axis];
                remaining /= segment.extents[axis];
                source_offset += @as(isize, @intCast(coordinate)) * segment.source_strides[axis];
                destination_offset += @as(isize, @intCast(coordinate)) * segment.destination_strides[axis];
            }
            output.storage[@intCast(destination_offset)] = input.storage[@intCast(source_offset)];
        }
    }
}

fn executeScalarExpressionSegment(
    comptime program: Program,
    comptime segment: Plan.MapPlan.SegmentedPlan.Segment,
    inputs: anytype,
    output: anytype,
) void {
    const rank = segment.expression_rank;
    const shape: [rank]usize = segment.expression_shape[0..rank].*;
    for (0..segment.elementCount()) |linear_index| {
        const offsets = segmentOffsets(segment, linear_index);
        const coordinates = evaluation.coordinatesFromLinear(shape, @intCast(offsets.expression));
        output.storage[@intCast(offsets.destination)] = evaluation.evaluateAt(
            program,
            segment.expression_value.?,
            inputs,
            shape,
            coordinates,
        );
    }
}

fn executeVectorExpressionSegment(
    comptime program: Program,
    comptime segment: Plan.MapPlan.SegmentedPlan.Segment,
    comptime vector_width: usize,
    inputs: anytype,
    output: anytype,
) void {
    const rank = segment.expression_rank;
    const shape: [rank]usize = segment.expression_shape[0..rank].*;
    const inner_axis = segment.rank - 1;
    const inner_extent = segment.extents[inner_axis];
    const outer_count = segment.elementCount() / inner_extent;

    for (0..outer_count) |outer_index| {
        const row_offsets = segmentOffsets(segment, outer_index * inner_extent);
        var inner_index: usize = 0;
        while (inner_index + vector_width <= inner_extent) : (inner_index += vector_width) {
            const expression_offset = row_offsets.expression + @as(isize, @intCast(inner_index));
            const destination_offset = row_offsets.destination + @as(isize, @intCast(inner_index));
            const coordinates = evaluation.coordinatesFromLinear(shape, @intCast(expression_offset));
            output.storage[@intCast(destination_offset)..][0..vector_width].* = evaluation.evaluateVectorAt(
                program,
                segment.expression_value.?,
                inputs,
                shape,
                coordinates,
                shape.len - 1,
                vector_width,
            );
        }
        while (inner_index < inner_extent) : (inner_index += 1) {
            const expression_offset = row_offsets.expression + @as(isize, @intCast(inner_index));
            const destination_offset = row_offsets.destination + @as(isize, @intCast(inner_index));
            const coordinates = evaluation.coordinatesFromLinear(shape, @intCast(expression_offset));
            output.storage[@intCast(destination_offset)] = evaluation.evaluateAt(
                program,
                segment.expression_value.?,
                inputs,
                shape,
                coordinates,
            );
        }
    }
}

fn canVectorizeExpressionSegment(
    comptime program: Program,
    comptime segment: Plan.MapPlan.SegmentedPlan.Segment,
    comptime Inputs: type,
    comptime vector_width: usize,
) bool {
    if (vector_width <= 1 or segment.rank == 0 or segment.expression_rank == 0) return false;
    const inner_axis = segment.rank - 1;
    const expression_axis = segment.expression_rank - 1;
    const inner_extent = segment.extents[inner_axis];
    const expression_axis_extent = segment.expression_shape[expression_axis];
    if (inner_extent < vector_width or
        segment.expression_strides[inner_axis] != 1 or
        segment.destination_strides[inner_axis] != 1 or
        segment.expression_offset % expression_axis_extent + inner_extent > expression_axis_extent)
    {
        return false;
    }
    for (segment.expression_strides[0..inner_axis]) |stride| {
        if (@mod(stride, @as(isize, @intCast(expression_axis_extent))) != 0) return false;
    }

    const shape = segment.expression_shape[0..segment.expression_rank].*;
    inline for (std.meta.fields(Inputs), 0..) |field, input_index| {
        if (!programUsesInput(program, input_index)) continue;
        const View = @TypeOf(@as(field.type, undefined).broadcastTo(segment.expression_rank, shape));
        const stride = View.static_strides[expression_axis];
        if (stride != 0 and stride != 1) return false;
    }
    return true;
}

fn programUsesInput(comptime program: Program, comptime input_index: usize) bool {
    for (program.instructions) |instruction| {
        for (instruction.args[0..instruction.operation.arity()]) |reference| {
            switch (reference) {
                .input => |index| if (index == input_index) return true,
                .instruction, .accumulator => {},
            }
        }
    }
    return false;
}

fn segmentOffsets(
    comptime segment: Plan.MapPlan.SegmentedPlan.Segment,
    linear_index: usize,
) struct { expression: isize, destination: isize } {
    var remaining = linear_index;
    var expression_offset: isize = @intCast(segment.expression_offset);
    var destination_offset: isize = @intCast(segment.destination_offset);
    comptime var axis: usize = segment.rank;
    inline while (axis > 0) {
        axis -= 1;
        const coordinate = remaining % segment.extents[axis];
        remaining /= segment.extents[axis];
        expression_offset += @as(isize, @intCast(coordinate)) * segment.expression_strides[axis];
        destination_offset += @as(isize, @intCast(coordinate)) * segment.destination_strides[axis];
    }
    return .{ .expression = expression_offset, .destination = destination_offset };
}

fn isContiguous(
    comptime strides: [Plan.MapPlan.SegmentedPlan.max_rank]isize,
    comptime extents: [Plan.MapPlan.SegmentedPlan.max_rank]usize,
    comptime rank: u8,
) bool {
    var expected: isize = 1;
    var axis: usize = rank;
    while (axis > 0) {
        axis -= 1;
        if (extents[axis] > 1 and strides[axis] != expected) return false;
        expected *= @intCast(extents[axis]);
    }
    return true;
}
