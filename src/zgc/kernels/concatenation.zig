const std = @import("std");

/// Materialize inputs consecutively along one statically selected axis.
pub fn concat(inputs: anytype, output: anytype, comptime concat_axis: i8) void {
    const axis: usize = @intCast(concat_axis);
    if (comptime usesStaticContiguousCopies(@TypeOf(inputs), @TypeOf(output))) {
        concatContiguous(inputs, output, axis);
    } else {
        concatStrided(
            inputs,
            output,
            axis,
            allGeometryIsStatic(@TypeOf(inputs), @TypeOf(output)),
        );
    }
}

fn concatContiguous(inputs: anytype, output: anytype, comptime axis: usize) void {
    const Output = @TypeOf(output);
    const inner_count = comptime blk: {
        var count: usize = 1;
        for (Output.static_shape[axis + 1 ..]) |extent| count *= extent;
        break :blk count;
    };
    const outer_count = comptime blk: {
        var count: usize = 1;
        for (Output.static_shape[0..axis]) |extent| count *= extent;
        break :blk count;
    };
    const output_values = output.contiguousSlice().?;

    for (0..outer_count) |outer| {
        var axis_offset: usize = 0;
        inline for (inputs) |input| {
            const Input = @TypeOf(input);
            const copy_count = Input.static_shape[axis] * inner_count;
            const input_values = input.contiguousSlice().?;
            const input_start = outer * copy_count;
            const output_start = outer * Output.static_shape[axis] * inner_count + axis_offset * inner_count;
            @memcpy(
                output_values[output_start..][0..copy_count],
                input_values[input_start..][0..copy_count],
            );
            axis_offset += Input.static_shape[axis];
        }
    }
}

fn concatStrided(
    inputs: anytype,
    output: anytype,
    comptime axis: usize,
    comptime static_geometry: bool,
) void {
    comptime var static_axis_offset: usize = 0;
    var dynamic_axis_offset: usize = 0;
    inline for (inputs) |input| {
        const Input = @TypeOf(input);
        const axis_offset = if (comptime static_geometry)
            static_axis_offset
        else
            dynamic_axis_offset;
        for (0..input.len()) |linear_index| {
            var remaining = linear_index;
            var input_offset: isize = @intCast(input.elementOffsetFromLinear(0));
            var output_offset: isize = @intCast(output.elementOffsetFromLinear(0));
            comptime var current_axis = Input.rank;
            inline while (current_axis > 0) {
                current_axis -= 1;
                var coordinate = remaining % input.shape[current_axis];
                remaining /= input.shape[current_axis];
                input_offset += @as(isize, @intCast(coordinate)) * input.strides[current_axis];
                if (comptime current_axis == axis) coordinate += axis_offset;
                output_offset += @as(isize, @intCast(coordinate)) * output.strides[current_axis];
            }
            output.storage[@intCast(output_offset)] = input.storage[@intCast(input_offset)];
        }
        if (comptime static_geometry) {
            static_axis_offset += Input.static_shape[axis];
        } else {
            dynamic_axis_offset += input.shape[axis];
        }
    }
}

fn usesStaticContiguousCopies(comptime Inputs: type, comptime Output: type) bool {
    if (!allGeometryIsStatic(Inputs, Output) or !Output.static_is_contiguous) return false;
    inline for (@typeInfo(Inputs).@"struct".field_types) |Input| {
        if (!Input.static_is_contiguous) return false;
    }
    return true;
}

fn allGeometryIsStatic(comptime Inputs: type, comptime Output: type) bool {
    if (!hasStaticGeometry(Output)) return false;
    inline for (@typeInfo(Inputs).@"struct".field_types) |Input| {
        if (!hasStaticGeometry(Input)) return false;
    }
    return true;
}

fn hasStaticGeometry(comptime View: type) bool {
    return @hasDecl(View, "geometry_is_static") and View.geometry_is_static;
}
