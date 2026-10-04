const std = @import("std");
const op_module = @import("../operations/semantic.zig");
const Op = op_module.Op;
const Tensor = @import("../core/tensor.zig");
const validation = @import("../validation.zig");

pub fn Result(comptime max_rank: usize) type {
    return struct {
        shape: Tensor.Shape(max_rank),
        layout: Tensor.Layout(max_rank),
        storage_tensor: Tensor.Id,
    };
}

/// Infer the graph metadata produced by a view operation. This switch is
/// intentionally exhaustive so every new `Op.View` requires an implementation.
pub fn infer(
    comptime op: Op.View,
    comptime inputs: anytype,
    comptime output_shape: anytype,
    comptime max_rank: usize,
) Result(max_rank) {
    return switch (op) {
        .transpose => |attrs| transpose(inputs, attrs, max_rank),
        .reshape => reshape(inputs, output_shape, max_rank),
        .flatten => |attrs| flatten(inputs, attrs, output_shape, max_rank),
        .squeeze => |attrs| squeeze(inputs, attrs, output_shape, max_rank),
        .unsqueeze => |attrs| unsqueeze(inputs, attrs, output_shape, max_rank),
        .slice => |attrs| slice(inputs, attrs, output_shape, max_rank),
        .broadcast => broadcast(inputs, output_shape, max_rank),
        .windows => |attrs| windows(inputs, attrs, output_shape, max_rank),
    };
}

fn windows(
    comptime inputs: anytype,
    comptime attrs: Op.View.WindowAttrs,
    comptime output_shape: anytype,
    comptime max_rank: usize,
) Result(max_rank) {
    const expected_shape = op_module.inferWindowsShape(inputs, attrs, max_rank);
    if (!std.mem.eql(usize, expected_shape.slice(), output_shape.slice())) {
        @compileError("windows output shape does not match its inferred geometry");
    }

    const input = inputs[0];
    const input_rank = input.shape.rank;
    const window_rank = attrs.sizes.len;
    const first_window_axis = input_rank - window_rank;
    var output_layout = Tensor.Layout(max_rank){
        .offset = input.layout.offset,
        .strides = @splat(0),
    };

    for (0..first_window_axis) |axis| {
        output_layout.strides[axis] = input.layout.strides[axis];
    }
    for (0..window_rank) |window_axis| {
        const input_axis = first_window_axis + window_axis;
        output_layout.strides[input_axis] = input.layout.strides[input_axis] *
            @as(isize, @intCast(op_module.windowStride(attrs, window_axis)));
        output_layout.strides[input_rank + window_axis] = input.layout.strides[input_axis] *
            @as(isize, @intCast(op_module.windowDilation(attrs, window_axis)));
    }
    return aliasResult(input, expected_shape, output_layout, max_rank);
}

fn reshape(
    comptime inputs: anytype,
    comptime output_shape: anytype,
    comptime max_rank: usize,
) Result(max_rank) {
    validation.requireInputCount("reshape", inputs, 1);
    if (inputs[0].shape.elementCount() != output_shape.elementCount()) {
        @compileError("reshape must preserve the tensor element count");
    }
    if (!isLogicallyContiguous(inputs[0])) {
        @compileError("reshape requires a logically contiguous input layout");
    }
    var output_layout = Tensor.Layout(max_rank).contiguous(output_shape);
    output_layout.offset = inputs[0].layout.offset;
    return aliasResult(inputs[0], output_shape, output_layout, max_rank);
}

fn broadcast(
    comptime inputs: anytype,
    comptime output_shape: anytype,
    comptime max_rank: usize,
) Result(max_rank) {
    validation.requireInputCount("broadcast", inputs, 1);
    const input = inputs[0];
    if (input.shape.rank > output_shape.rank) {
        @compileError("broadcast target rank cannot be smaller than its source rank");
    }
    var output_layout = Tensor.Layout(max_rank){
        .offset = input.layout.offset,
        .strides = @splat(0),
    };
    const rank_offset = output_shape.rank - input.shape.rank;
    for (0..input.shape.rank) |input_axis| {
        const output_axis = rank_offset + input_axis;
        const source_extent = input.shape.at(input_axis);
        const target_extent = output_shape.at(output_axis);
        if (source_extent != 1 and source_extent != target_extent) {
            @compileError("broadcast source extent must equal its target or be one");
        }
        if (source_extent == target_extent) {
            output_layout.strides[output_axis] = input.layout.strides[input_axis];
        }
    }
    return aliasResult(input, output_shape, output_layout, max_rank);
}

fn flatten(
    comptime inputs: anytype,
    comptime attrs: Op.View.FlattenAttrs,
    comptime output_shape: anytype,
    comptime max_rank: usize,
) Result(max_rank) {
    validation.requireInputCount("flatten", inputs, 1);
    const start: usize = @intCast(attrs.start_axis);
    const end: usize = @intCast(attrs.end_axis);
    const input = inputs[0];
    if (start > end or end >= input.shape.rank) @compileError("flatten axes are outside the input rank");
    if (output_shape.rank != input.shape.rank - (end - start)) {
        @compileError("flatten output rank does not match its collapsed axes");
    }
    var expected_output_axis: usize = 0;
    for (0..start) |input_axis| {
        if (output_shape.at(expected_output_axis) != input.shape.at(input_axis)) {
            @compileError("flatten output shape does not preserve leading axes");
        }
        expected_output_axis += 1;
    }
    var flattened_extent: usize = 1;
    for (start..end + 1) |input_axis| flattened_extent *= input.shape.at(input_axis);
    if (output_shape.at(expected_output_axis) != flattened_extent) {
        @compileError("flatten output shape has an invalid collapsed extent");
    }
    expected_output_axis += 1;
    for (end + 1..input.shape.rank) |input_axis| {
        if (output_shape.at(expected_output_axis) != input.shape.at(input_axis)) {
            @compileError("flatten output shape does not preserve trailing axes");
        }
        expected_output_axis += 1;
    }

    var flattened_stride: ?isize = null;
    var expected: isize = 0;
    var axis = end + 1;
    while (axis > start) {
        axis -= 1;
        const extent = input.shape.at(axis);
        if (extent <= 1) continue;
        if (flattened_stride == null) {
            flattened_stride = input.layout.strides[axis];
            expected = flattened_stride.? * @as(isize, @intCast(extent));
        } else {
            if (input.layout.strides[axis] != expected) {
                @compileError("flatten axes are not logically contiguous");
            }
            expected *= @intCast(extent);
        }
    }
    const unit_stride = flattened_stride orelse if (end + 1 < input.shape.rank)
        input.layout.strides[end + 1] * @as(isize, @intCast(input.shape.at(end + 1)))
    else
        1;

    var output_layout = Tensor.Layout(max_rank){
        .offset = input.layout.offset,
        .strides = @splat(0),
    };
    for (0..start) |input_axis| output_layout.strides[input_axis] = input.layout.strides[input_axis];
    output_layout.strides[start] = unit_stride;
    for (end + 1..input.shape.rank) |input_axis| {
        output_layout.strides[start + 1 + input_axis - (end + 1)] = input.layout.strides[input_axis];
    }
    return aliasResult(input, output_shape, output_layout, max_rank);
}

fn squeeze(
    comptime inputs: anytype,
    comptime attrs: Op.View.AxisAttrs,
    comptime output_shape: anytype,
    comptime max_rank: usize,
) Result(max_rank) {
    validation.requireInputCount("squeeze", inputs, 1);
    const axis: usize = @intCast(attrs.axis);
    const input = inputs[0];
    if (axis >= input.shape.rank or input.shape.at(axis) != 1) {
        @compileError("squeeze axis must have extent one");
    }
    if (output_shape.rank + 1 != input.shape.rank) @compileError("squeeze output rank is invalid");
    var output_layout = Tensor.Layout(max_rank){ .offset = input.layout.offset, .strides = @splat(0) };
    var output_axis: usize = 0;
    for (0..input.shape.rank) |input_axis| {
        if (input_axis == axis) continue;
        if (output_shape.at(output_axis) != input.shape.at(input_axis)) {
            @compileError("squeeze output shape does not match its input");
        }
        output_layout.strides[output_axis] = input.layout.strides[input_axis];
        output_axis += 1;
    }
    return aliasResult(input, output_shape, output_layout, max_rank);
}

fn unsqueeze(
    comptime inputs: anytype,
    comptime attrs: Op.View.AxisAttrs,
    comptime output_shape: anytype,
    comptime max_rank: usize,
) Result(max_rank) {
    validation.requireInputCount("unsqueeze", inputs, 1);
    const axis: usize = @intCast(attrs.axis);
    const input = inputs[0];
    if (axis > input.shape.rank) @compileError("unsqueeze axis is outside the output rank");
    if (output_shape.rank != input.shape.rank + 1 or output_shape.at(axis) != 1) {
        @compileError("unsqueeze output shape is invalid");
    }
    var output_layout = Tensor.Layout(max_rank){ .offset = input.layout.offset, .strides = @splat(0) };
    for (0..output_shape.rank) |output_axis| {
        if (output_axis < axis) {
            if (output_shape.at(output_axis) != input.shape.at(output_axis)) {
                @compileError("unsqueeze output shape does not match its input");
            }
            output_layout.strides[output_axis] = input.layout.strides[output_axis];
        } else if (output_axis == axis) {
            output_layout.strides[output_axis] = if (axis < input.shape.rank)
                input.layout.strides[axis] * @as(isize, @intCast(input.shape.at(axis)))
            else
                1;
        } else {
            if (output_shape.at(output_axis) != input.shape.at(output_axis - 1)) {
                @compileError("unsqueeze output shape does not match its input");
            }
            output_layout.strides[output_axis] = input.layout.strides[output_axis - 1];
        }
    }
    return aliasResult(input, output_shape, output_layout, max_rank);
}

fn slice(
    comptime inputs: anytype,
    comptime attrs: Op.View.SliceAttrs,
    comptime output_shape: anytype,
    comptime max_rank: usize,
) Result(max_rank) {
    validation.requireInputCount("slice", inputs, 1);
    const input = inputs[0];
    const axis: usize = @intCast(attrs.axis);
    if (axis >= input.shape.rank or attrs.step == 0 or attrs.length == 0) {
        @compileError("slice contains invalid static geometry");
    }
    const final_index = attrs.start + (attrs.length - 1) * attrs.step;
    if (final_index >= input.shape.at(axis) or output_shape.rank != input.shape.rank) {
        @compileError("slice range is outside the input shape");
    }

    var output_layout = input.layout;
    output_layout.offset = @intCast(
        @as(isize, @intCast(input.layout.offset)) +
            @as(isize, @intCast(attrs.start)) * input.layout.strides[axis],
    );
    output_layout.strides[axis] *= @intCast(attrs.step);
    for (0..input.shape.rank) |current_axis| {
        const expected = if (current_axis == axis) attrs.length else input.shape.at(current_axis);
        if (output_shape.at(current_axis) != expected) @compileError("slice output shape is invalid");
    }
    return aliasResult(input, output_shape, output_layout, max_rank);
}

fn aliasResult(
    comptime input: anytype,
    comptime shape: anytype,
    comptime tensor_layout: anytype,
    comptime max_rank: usize,
) Result(max_rank) {
    return .{
        .shape = shape,
        .layout = tensor_layout,
        .storage_tensor = input.storage_tensor,
    };
}

fn isLogicallyContiguous(comptime input: anytype) bool {
    var expected: isize = 1;
    var axis = input.shape.rank;
    while (axis > 0) {
        axis -= 1;
        if (input.shape.at(axis) > 1 and input.layout.strides[axis] != expected) return false;
        expected *= @intCast(input.shape.at(axis));
    }
    return true;
}

fn transpose(
    comptime inputs: anytype,
    comptime attrs: Op.View.TransposeAttrs,
    comptime max_rank: usize,
) Result(max_rank) {
    validation.requireInputCount("transpose", inputs, 1);
    validation.requireAxis("transpose", inputs[0], attrs.axis_a);
    validation.requireAxis("transpose", inputs[0], attrs.axis_b);

    const axis_a: usize = @intCast(attrs.axis_a);
    const axis_b: usize = @intCast(attrs.axis_b);
    var shape = inputs[0].shape;
    var tensor_layout = inputs[0].layout;

    std.mem.swap(usize, &shape.dims[axis_a], &shape.dims[axis_b]);
    std.mem.swap(
        isize,
        &tensor_layout.strides[axis_a],
        &tensor_layout.strides[axis_b],
    );

    return .{
        .shape = shape,
        .layout = tensor_layout,
        .storage_tensor = inputs[0].storage_tensor,
    };
}
