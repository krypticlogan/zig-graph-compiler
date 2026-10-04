const Op = @import("../operations/semantic.zig").Op;

/// Reference execution for a fixed slice loop. Iteration descriptors are
/// unrolled at compile time; the optimized path lowers the same operation to a
/// segmented map plan.
pub fn execute(comptime attrs: Op.Compute.SliceLoopAttrs, input: anytype, output: anytype) void {
    const rank = @TypeOf(output).rank;
    const loop_axis: usize = @intCast(attrs.axis);
    const slice_elements = output.len() / attrs.iterations.len;

    inline for (attrs.iterations, 0..) |iteration, iteration_index| {
        for (0..slice_elements) |slice_linear| {
            var output_coordinates: [rank]usize = @splat(0);
            output_coordinates[loop_axis] = iteration_index;
            var remaining = slice_linear;
            var axis = rank;
            while (axis > 0) {
                axis -= 1;
                if (axis == loop_axis) continue;
                output_coordinates[axis] = remaining % output.shape[axis];
                remaining /= output.shape[axis];
            }

            var source_coordinates = output_coordinates;
            var outside = false;
            var offset_axis: usize = 0;
            for (0..rank) |current_axis| {
                if (current_axis == loop_axis) continue;
                const mapped = remapCoordinate(
                    output_coordinates[current_axis],
                    output.shape[current_axis],
                    iteration.offsets[offset_axis],
                    iteration.boundary,
                );
                offset_axis += 1;
                if (mapped) |coordinate| {
                    source_coordinates[current_axis] = coordinate;
                } else {
                    outside = true;
                }
            }
            if (outside) {
                source_coordinates = output_coordinates;
                source_coordinates[loop_axis] = switch (iteration.boundary) {
                    .redirect => |redirect| redirect,
                    .wrap, .edge, .reflect => unreachable,
                };
            }
            output.set(output_coordinates, input.get(source_coordinates));
        }
    }
}

fn remapCoordinate(
    coordinate: usize,
    extent: usize,
    offset: isize,
    comptime boundary: Op.Compute.SliceLoopAttrs.Boundary,
) ?usize {
    return switch (boundary) {
        .wrap => blk: {
            const displacement: usize = @intCast(@mod(@as(i128, offset), @as(i128, @intCast(extent))));
            break :blk if (coordinate >= displacement)
                coordinate - displacement
            else
                extent - (displacement - coordinate);
        },
        .edge => blk: {
            if (offset >= 0) {
                const displacement: usize = @intCast(offset);
                break :blk if (displacement >= extent or coordinate < displacement) 0 else coordinate - displacement;
            }
            const displacement: usize = @intCast(@abs(offset));
            break :blk if (displacement >= extent or coordinate >= extent - displacement) extent - 1 else coordinate + displacement;
        },
        .reflect => blk: {
            if (extent == 1) break :blk 0;
            const period = @as(u128, extent - 1) * 2;
            const displacement: u128 = @intCast(@mod(@as(i128, offset), @as(i128, @intCast(period))));
            const phase = if (@as(u128, coordinate) >= displacement)
                @as(u128, coordinate) - displacement
            else
                period - (displacement - coordinate);
            break :blk @intCast(if (phase < extent) phase else period - phase);
        },
        .redirect => blk: {
            if (offset >= 0) {
                const displacement: usize = @intCast(offset);
                if (displacement > coordinate) break :blk null;
                break :blk coordinate - displacement;
            }
            const displacement: usize = @intCast(@abs(offset));
            if (displacement >= extent or coordinate >= extent - displacement) break :blk null;
            break :blk coordinate + displacement;
        },
    };
}
