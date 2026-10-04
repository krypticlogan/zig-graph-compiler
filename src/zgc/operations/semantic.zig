const std = @import("std");
const Dtype = @import("../storage/dtype.zig").Dtype;
const Elementwise = @import("elementwise.zig");
const Tensor = @import("../core/tensor.zig");
const kernels = @import("../kernels/root.zig");
const validation = @import("../validation.zig");

/// Tensor operation classified by whether it computes new storage or creates
/// another view of existing storage.
pub const Op = union(enum) {
    compute: Compute,
    view: View,

    pub const Kind = enum { compute, view };

    pub const Compute = union(enum) {
        relu,
        exp,
        neg,
        abs,
        sqrt,
        log,
        reciprocal,
        add,
        sub,
        mul,
        div,
        minimum,
        maximum,
        clamp,
        equal,
        not_equal,
        less_than,
        less_equal,
        greater_than,
        greater_equal,
        logical_not,
        logical_and,
        logical_or,
        where,
        copy,
        contiguous,
        pad: PadAttrs,
        shift: ShiftAttrs,
        slice_loop: SliceLoopAttrs,
        matmul,
        sum: ReductionAttrs,
        mean: ReductionAttrs,
        min: ReductionAttrs,
        max: ReductionAttrs,
        concat: ConcatAttrs,
        softmax: SoftmaxAttrs,

        pub const ReductionAttrs = struct {
            axes: u64,
            keep_dims: bool = false,
        };
        pub const SoftmaxAttrs = struct { axis: i8 };
        pub const ConcatAttrs = struct { axis: i8 };
        pub const PadAttrs = struct {
            before: []const usize,
            after: []const usize,
        };
        pub const ShiftAttrs = struct {
            offsets: []const isize,
            boundary: Boundary,

            pub const Boundary = enum { wrap, edge, reflect, constant };
        };
        pub const SliceLoopAttrs = struct {
            axis: i8,
            iterations: []const Iteration,

            pub const Iteration = struct {
                offsets: []const isize,
                boundary: Boundary,
            };

            pub const Boundary = union(enum) {
                wrap,
                edge,
                reflect,
                redirect: usize,
            };
        };
        pub fn execute(
            comptime op: Compute,
            inputs: anytype,
            output: anytype,
        ) void {
            kernels.execute(op, inputs, output);
        }

        pub fn inferShape(
            comptime op: Compute,
            comptime inputs: anytype,
            comptime max_rank: usize,
        ) Tensor.Shape(max_rank) {
            return switch (op) {
                .relu => blk: {
                    _ = inferNumericUnaryRank("relu", inputs);
                    break :blk inferUnaryShape("relu", inputs, max_rank);
                },
                .exp, .neg, .abs, .sqrt, .log, .reciprocal => blk: {
                    _ = inferFloatUnaryRank(@tagName(op), inputs);
                    break :blk inferUnaryShape(@tagName(op), inputs, max_rank);
                },
                .add, .sub, .mul, .minimum, .maximum => inferNumericBinaryShape(@tagName(op), inputs, max_rank),
                .div => blk: {
                    validation.requireDtypeKind("div", inputs[0], .float);
                    break :blk inferNumericBinaryShape("div", inputs, max_rank);
                },
                .clamp => inferNumericTernaryShape("clamp", inputs, max_rank),
                .equal, .not_equal => inferBinaryElementwiseShape(@tagName(op), inputs, max_rank),
                .less_than, .less_equal, .greater_than, .greater_equal => inferNumericBinaryShape(@tagName(op), inputs, max_rank),
                .logical_not => blk: {
                    _ = inferBooleanUnaryRank("logical_not", inputs);
                    break :blk inferUnaryShape("logical_not", inputs, max_rank);
                },
                .logical_and, .logical_or => inferBooleanBinaryShape(@tagName(op), inputs, max_rank),
                .where => inferWhereShape(inputs, max_rank),
                .copy, .contiguous => inferUnaryShape(@tagName(op), inputs, max_rank),
                .pad => |attrs| inferPadShape(inputs, attrs, max_rank),
                .shift => |attrs| blk: {
                    _ = inferShiftRank(inputs, attrs);
                    break :blk inputs[0].shape;
                },
                .slice_loop => |attrs| inferSliceLoopShape(inputs, attrs, max_rank),
                .matmul => inferMatmulShape(inputs, max_rank),
                .sum => |attrs| inferNumericReductionShape("sum", inputs, attrs, max_rank),
                .mean => |attrs| blk: {
                    _ = inferReductionRank("mean", inputs, attrs);
                    validation.requireDtypeKind("mean", inputs[0], .float);
                    break :blk inferReductionShape("mean", inputs, attrs, max_rank);
                },
                .min => |attrs| inferNumericReductionShape("min", inputs, attrs, max_rank),
                .max => |attrs| inferNumericReductionShape("max", inputs, attrs, max_rank),
                .concat => |attrs| inferConcatShape(inputs, attrs.axis, max_rank),
                .softmax => |attrs| blk: {
                    _ = inferFloatUnaryRank("softmax", inputs);
                    const shape = inferUnaryShape("softmax", inputs, max_rank);
                    validation.requireAxis("softmax", inputs[0], attrs.axis);
                    break :blk shape;
                },
            };
        }

        pub fn inferDtype(comptime op: Compute, comptime inputs: anytype) Dtype {
            if (comptime Elementwise.fromCompute(op)) |pointwise| {
                var dtypes: [inputs.len]Dtype = undefined;
                inline for (inputs, 0..) |input, index| dtypes[index] = input.dtype;
                return pointwise.inferDtype(&dtypes);
            }
            return inputs[0].dtype;
        }
    };

    pub const View = union(enum) {
        transpose: TransposeAttrs,
        reshape,
        flatten: FlattenAttrs,
        squeeze: AxisAttrs,
        unsqueeze: AxisAttrs,
        slice: SliceAttrs,
        broadcast,
        windows: WindowAttrs,

        pub const TransposeAttrs = struct { axis_a: i8, axis_b: i8 };
        pub const FlattenAttrs = struct { start_axis: i8, end_axis: i8 };
        pub const AxisAttrs = struct { axis: i8 };
        pub const SliceAttrs = struct {
            axis: i8,
            start: usize,
            length: usize,
            step: usize,
        };
        pub const WindowAttrs = struct {
            sizes: []const usize,
            strides: ?[]const usize = null,
            dilations: ?[]const usize = null,
        };
    };

    pub fn kind(op: Op) Kind {
        return switch (op) {
            .compute => .compute,
            .view => .view,
        };
    }

    pub fn execute(comptime op: Op, inputs: anytype, output: anytype) void {
        switch (op) {
            .compute => |compute| compute.execute(inputs, output),
            .view => @compileError("view operations do not execute a runtime kernel"),
        }
    }
};

pub fn inferWindowsShape(
    comptime inputs: anytype,
    comptime attrs: Op.View.WindowAttrs,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    validation.requireInputCount("windows", inputs, 1);
    const input_rank = validation.rankOf(inputs[0]);
    const window_rank = attrs.sizes.len;
    if (window_rank == 0) @compileError("windows requires at least one window axis");
    if (window_rank > input_rank) @compileError("windows cannot cover more axes than the input rank");
    if (input_rank + window_rank > max_rank) @compileError("windows output exceeds the definition max_rank");
    if (attrs.strides) |strides| {
        if (strides.len != window_rank) @compileError("window strides must match the number of window axes");
    }
    if (attrs.dilations) |dilations| {
        if (dilations.len != window_rank) @compileError("window dilations must match the number of window axes");
    }

    var result = Tensor.Shape(max_rank){
        .rank = input_rank + window_rank,
        .dims = @splat(0),
    };
    for (inputs[0].shape.slice(), 0..) |extent, axis| result.dims[axis] = extent;

    const first_window_axis = input_rank - window_rank;
    for (0..window_rank) |window_axis| {
        const size = attrs.sizes[window_axis];
        const stride = windowStride(attrs, window_axis);
        const dilation = windowDilation(attrs, window_axis);
        if (size == 0 or stride == 0 or dilation == 0) {
            @compileError("window sizes, strides, and dilations must be greater than zero");
        }
        const span = std.math.mul(usize, size - 1, dilation) catch
            @compileError("dilated window span exceeds usize");
        const effective_size = std.math.add(usize, span, 1) catch
            @compileError("dilated window size exceeds usize");
        const input_axis = first_window_axis + window_axis;
        const input_extent = inputs[0].shape.at(input_axis);
        if (effective_size > input_extent) @compileError("window does not fit within its input extent");
        result.dims[input_axis] = (input_extent - effective_size) / stride + 1;
        result.dims[input_rank + window_axis] = size;
    }
    return result;
}

pub fn windowStride(comptime attrs: Op.View.WindowAttrs, comptime axis: usize) usize {
    return if (attrs.strides) |strides| strides[axis] else 1;
}

pub fn windowDilation(comptime attrs: Op.View.WindowAttrs, comptime axis: usize) usize {
    return if (attrs.dilations) |dilations| dilations[axis] else 1;
}

fn inferUnaryRank(comptime operation: []const u8, inputs: anytype) usize {
    validation.requireInputCount(operation, inputs, 1);
    return validation.rankOf(inputs[0]);
}

fn inferUnaryShape(
    comptime operation: []const u8,
    comptime inputs: anytype,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    validation.requireInputCount(operation, inputs, 1);
    return inputs[0].shape;
}

fn inferFloatUnaryRank(comptime operation: []const u8, inputs: anytype) usize {
    const rank = inferUnaryRank(operation, inputs);
    validation.requireDtypeKind(operation, inputs[0], .float);
    return rank;
}

fn inferNumericUnaryRank(comptime operation: []const u8, inputs: anytype) usize {
    const rank = inferUnaryRank(operation, inputs);
    validation.requireNumericDtype(operation, inputs[0]);
    return rank;
}

fn inferBooleanUnaryRank(comptime operation: []const u8, inputs: anytype) usize {
    const rank = inferUnaryRank(operation, inputs);
    validation.requireDtype(operation, inputs[0], .bool);
    return rank;
}

fn inferReductionRank(
    comptime operation: []const u8,
    inputs: anytype,
    comptime attrs: Op.Compute.ReductionAttrs,
) usize {
    const rank = inferUnaryRank(operation, inputs);
    validation.requireReductionAxes(operation, inputs[0], attrs.axes);
    return if (attrs.keep_dims) rank else rank - @popCount(attrs.axes);
}

fn inferReductionShape(
    comptime operation: []const u8,
    comptime inputs: anytype,
    comptime attrs: Op.Compute.ReductionAttrs,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    _ = inferReductionRank(operation, inputs, attrs);
    var shape = Tensor.Shape(max_rank){ .rank = 0, .dims = @splat(0) };
    for (inputs[0].shape.slice(), 0..) |extent, axis| {
        const reduced = attrs.axes & (@as(u64, 1) << @intCast(axis)) != 0;
        if (reduced and !attrs.keep_dims) continue;
        shape.dims[shape.rank] = if (reduced) 1 else extent;
        shape.rank += 1;
    }
    return shape;
}

fn inferNumericReductionRank(
    comptime operation: []const u8,
    inputs: anytype,
    comptime attrs: Op.Compute.ReductionAttrs,
) usize {
    const rank = inferReductionRank(operation, inputs, attrs);
    validation.requireNumericDtype(operation, inputs[0]);
    return rank;
}

fn inferNumericReductionShape(
    comptime operation: []const u8,
    comptime inputs: anytype,
    comptime attrs: Op.Compute.ReductionAttrs,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    _ = inferNumericReductionRank(operation, inputs, attrs);
    return inferReductionShape(operation, inputs, attrs, max_rank);
}

fn inferBinaryElementwiseRank(
    comptime operation: []const u8,
    inputs: anytype,
) usize {
    validation.requireInputCount(operation, inputs, 2);
    validation.requireMatchingDtypes(operation, inputs);
    return @max(validation.rankOf(inputs[0]), validation.rankOf(inputs[1]));
}

fn inferNumericBinaryRank(comptime operation: []const u8, inputs: anytype) usize {
    const rank = inferBinaryElementwiseRank(operation, inputs);
    validation.requireNumericDtype(operation, inputs[0]);
    return rank;
}

fn inferBooleanBinaryRank(comptime operation: []const u8, inputs: anytype) usize {
    const rank = inferBinaryElementwiseRank(operation, inputs);
    validation.requireDtype(operation, inputs[0], .bool);
    return rank;
}

fn inferNumericTernaryRank(comptime operation: []const u8, inputs: anytype) usize {
    validation.requireInputCount(operation, inputs, 3);
    validation.requireMatchingDtypes(operation, inputs);
    validation.requireNumericDtype(operation, inputs[0]);
    return broadcastRank(inputs);
}

fn inferWhereRank(inputs: anytype) usize {
    validation.requireInputCount("where", inputs, 3);
    validation.requireDtype("where", inputs[0], .bool);
    if (inputs[1].dtype != inputs[2].dtype) {
        @compileError("where value dtypes must match");
    }
    return broadcastRank(inputs);
}

fn inferPadRank(inputs: anytype, comptime attrs: Op.Compute.PadAttrs) usize {
    validation.requireInputCount("pad", inputs, 2);
    validation.requireMatchingDtypes("pad", inputs);
    if (validation.rankOf(inputs[1]) != 0) @compileError("pad fill value must be rank zero");
    const rank = validation.rankOf(inputs[0]);
    if (attrs.before.len != rank or attrs.after.len != rank) {
        @compileError("pad requires one before and after width per input axis");
    }
    return rank;
}

fn inferShiftRank(inputs: anytype, comptime attrs: Op.Compute.ShiftAttrs) usize {
    const constant = attrs.boundary == .constant;
    validation.requireInputCount("shift", inputs, if (constant) 2 else 1);
    const rank = validation.rankOf(inputs[0]);
    if (attrs.offsets.len != rank) {
        @compileError("shift requires one offset per input axis");
    }
    if (constant) {
        validation.requireMatchingDtypes("shift", inputs);
        if (validation.rankOf(inputs[1]) != 0) {
            @compileError("shift constant fill value must be rank zero");
        }
    }
    return rank;
}

fn inferSliceLoopShape(
    comptime inputs: anytype,
    comptime attrs: Op.Compute.SliceLoopAttrs,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    validation.requireInputCount("sliceLoop", inputs, 1);
    const rank = validation.rankOf(inputs[0]);
    validation.requireAxis("sliceLoop", inputs[0], attrs.axis);
    const axis: usize = @intCast(attrs.axis);
    if (attrs.iterations.len != inputs[0].shape.at(axis)) {
        @compileError("sliceLoop requires one iteration descriptor per selected slice");
    }
    for (attrs.iterations) |iteration| {
        if (iteration.offsets.len != rank - 1) {
            @compileError("sliceLoop iteration offsets must cover every non-loop axis");
        }
        switch (iteration.boundary) {
            .redirect => |redirect| if (redirect >= attrs.iterations.len) {
                @compileError("sliceLoop redirect is outside the loop axis");
            },
            .wrap, .edge, .reflect => {},
        }
    }
    return inputs[0].shape;
}

fn inferPadShape(
    comptime inputs: anytype,
    comptime attrs: Op.Compute.PadAttrs,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    const rank = inferPadRank(inputs, attrs);
    var result = inputs[0].shape;
    for (0..rank) |axis| {
        const with_before = std.math.add(usize, result.dims[axis], attrs.before[axis]) catch
            @compileError("pad output extent exceeds usize");
        result.dims[axis] = std.math.add(usize, with_before, attrs.after[axis]) catch
            @compileError("pad output extent exceeds usize");
    }
    return result;
}

fn inferConcatRank(inputs: anytype, comptime axis: i8) usize {
    if (inputs.len == 0) @compileError("concat requires at least one input");
    validation.requireMatchingRanks("concat", inputs);
    validation.requireMatchingDtypes("concat", inputs);
    validation.requireAxis("concat", inputs[0], axis);
    return validation.rankOf(inputs[0]);
}

fn inferConcatShape(
    comptime inputs: anytype,
    comptime axis: i8,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    const rank = inferConcatRank(inputs, axis);
    const concat_axis: usize = @intCast(axis);
    var result = inputs[0].shape;
    var concat_extent: usize = 0;
    for (inputs) |input| {
        for (0..rank) |current_axis| {
            if (current_axis == concat_axis) continue;
            if (input.shape.at(current_axis) != inputs[0].shape.at(current_axis)) {
                @compileError("concat input extents must match outside the concatenation axis");
            }
        }
        concat_extent = std.math.add(usize, concat_extent, input.shape.at(concat_axis)) catch
            @compileError("concat axis extent exceeds usize");
    }
    result.dims[concat_axis] = concat_extent;
    return result;
}

fn inferBinaryElementwiseShape(
    comptime operation: []const u8,
    comptime inputs: anytype,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    _ = inferBinaryElementwiseRank(operation, inputs);
    return inferBroadcastShape(operation, inputs, max_rank);
}

fn inferNumericBinaryShape(
    comptime operation: []const u8,
    comptime inputs: anytype,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    _ = inferNumericBinaryRank(operation, inputs);
    return inferBroadcastShape(operation, inputs, max_rank);
}

fn inferBooleanBinaryShape(
    comptime operation: []const u8,
    comptime inputs: anytype,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    _ = inferBooleanBinaryRank(operation, inputs);
    return inferBroadcastShape(operation, inputs, max_rank);
}

fn inferNumericTernaryShape(
    comptime operation: []const u8,
    comptime inputs: anytype,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    _ = inferNumericTernaryRank(operation, inputs);
    return inferBroadcastShape(operation, inputs, max_rank);
}

fn inferWhereShape(comptime inputs: anytype, comptime max_rank: usize) Tensor.Shape(max_rank) {
    _ = inferWhereRank(inputs);
    return inferBroadcastShape("where", inputs, max_rank);
}

fn broadcastRank(inputs: anytype) usize {
    var result: usize = 0;
    for (inputs) |input| result = @max(result, validation.rankOf(input));
    return result;
}

fn inferBroadcastShape(
    comptime operation: []const u8,
    comptime inputs: anytype,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    const result_rank = broadcastRank(inputs);
    var result = Tensor.Shape(max_rank){
        .rank = result_rank,
        .dims = @splat(0),
    };

    for (0..result_rank) |axis_from_end| {
        var extent: usize = 1;
        for (inputs) |input| {
            const shape = input.shape.slice();
            const candidate = if (axis_from_end < shape.len)
                shape[shape.len - 1 - axis_from_end]
            else
                1;
            if (!validation.extentsBroadcast(extent, candidate)) {
                @compileError(std.fmt.comptimePrint(
                    "{s} cannot broadcast extents {d} and {d} at aligned axis {d}",
                    .{ operation, extent, candidate, result_rank - 1 - axis_from_end },
                ));
            }
            if (extent == 1) extent = candidate;
        }
        result.dims[result_rank - 1 - axis_from_end] = extent;
    }
    return result;
}

fn inferMatmulRank(inputs: anytype) usize {
    validation.requireInputCount("matmul", inputs, 2);
    validation.requireRanks("matmul", inputs, &.{ 2, 2 });
    validation.requireMatchingDtypes("matmul", inputs);
    validation.requireDtype("matmul", inputs[0], .f32);
    return 2;
}

fn inferMatmulShape(
    comptime inputs: anytype,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    _ = inferMatmulRank(inputs);
    validation.requireMatchingExtents("matmul", inputs[0], 1, inputs[1], 0);

    return Tensor.Shape(max_rank).init(&.{
        inputs[0].shape.at(0),
        inputs[1].shape.at(1),
    });
}
