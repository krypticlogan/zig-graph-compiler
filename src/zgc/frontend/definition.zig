const std = @import("std");
const Dtype = @import("../storage/dtype.zig").Dtype;
const ScalarValue = @import("../storage/dtype.zig").ScalarValue;
const op_module = @import("../operations/semantic.zig");
const Op = op_module.Op;
const SourceStorage = @import("../storage/source.zig");
const Tensor = @import("../core/tensor.zig");
const Capacity = @import("../core/graph.zig").Capacity;
pub const Limits = struct {
    max_rank: usize = 8,
    max_nodes: usize = 64,
    max_tensors: usize = 128,
    max_input_refs: usize = 192,
    max_outputs: usize = 8,
};

/// Axes omitted with `null` reduce the entire tensor. Reduced dimensions are
/// removed unless `keep_dims` retains them as singleton dimensions.
pub const ReductionOptions = struct {
    /// Axes to reduce. `null` reduces every axis.
    axes: ?[]const i8 = null,
    /// Preserve reduced axes as extent-one dimensions.
    keep_dims: bool = false,
};

pub const FlattenOptions = struct {
    /// First axis in the inclusive flattened range.
    start_axis: i8 = 0,
    /// Last axis in the inclusive flattened range.
    end_axis: i8 = -1,
};

pub const SliceOptions = struct {
    /// Axis containing the selected range.
    axis: i8,
    /// Inclusive range start.
    start: usize = 0,
    /// Exclusive range end. `null` selects through the axis extent.
    end: ?usize = null,
    /// Positive distance between selected elements.
    step: usize = 1,
};

pub const PadOptions = struct {
    /// Padding width before each input axis.
    before: []const usize,
    /// Padding width after each input axis.
    after: []const usize,
};

pub const WindowOptions = struct {
    /// Window extent over each selected trailing axis.
    sizes: []const usize,
    /// Step between adjacent window origins. Defaults to one per axis.
    strides: ?[]const usize = null,
    /// Spacing between elements within each window. Defaults to one per axis.
    dilations: ?[]const usize = null,
};

pub fn Value(comptime max_rank: usize) type {
    return struct {
        id: Tensor.Id,
        dtype: Dtype,
        shape: Tensor.Shape(max_rank),
    };
}

pub fn Definition(comptime SourceKey: type, comptime limits: Limits) type {
    return struct {
        const Self = @This();
        pub const max_rank = limits.max_rank;
        pub const Source = SourceKey;
        pub const ValueType = Value(max_rank);
        pub const SourceOverride = struct {
            /// Source enum tag whose default ownership policy is replaced.
            source: SourceKey,
            /// Storage policy applied to the selected source.
            binding: SourceStorage.Binding,
        };

        const Node = struct {
            op: Op,
            input_start: usize,
            input_count: usize,
            result: Tensor.Id,
        };

        const TensorRecord = struct {
            value: ValueType,
            origin: Tensor.Origin,
            source_kind: ?Tensor.Source.Kind = null,
        };

        nodes: [limits.max_nodes]Node = undefined,
        tensors: [limits.max_tensors]TensorRecord = undefined,
        input_refs: [limits.max_input_refs]Tensor.Id = undefined,
        outputs: [limits.max_outputs]Tensor.Id = undefined,
        node_count: usize = 0,
        tensor_count: usize = 0,
        input_ref_count: usize = 0,
        output_count: usize = 0,

        /// Run capacity counting, graph lowering, memory planning, and model
        /// generation for this completed definition.
        pub fn model(comptime definition: Self) type {
            return @import("../compiler/root.zig").model(
                Self,
                definition,
                @as([]const SourceOverride, &.{}),
            );
        }

        /// Compile this definition with typed source-storage overrides.
        pub fn modelWith(comptime definition: Self, comptime sources: []const SourceOverride) type {
            return @import("../compiler/root.zig").model(Self, definition, sources);
        }

        pub fn counts(comptime definition: Definition) Capacity {
            var capacity: Capacity = .{
                .max_nodes = definition.node_count,
                .max_input_refs = definition.input_ref_count,
                .max_tensors = definition.tensor_count,
                .max_outputs = definition.output_count,
            };

            for (definition.tensors[0..definition.tensor_count]) |record| {
                capacity.max_rank = @max(capacity.max_rank, record.value.shape.rank);
                switch (record.origin) {
                    .source => |source_index| {
                        capacity.max_sources = @max(capacity.max_sources, source_index + 1);
                    },
                    .node, .literal => {},
                }
            }
            return capacity;
        }
    };
}

/// The typed, front-facing model-definition builder.
pub fn DefinitionBuilder(comptime SourceKey: type, comptime limits: Limits) type {
    const source_capacity = enumCapacity(SourceKey);
    const DefinitionType = Definition(SourceKey, limits);
    const ValueType = DefinitionType.ValueType;

    return struct {
        const Self = @This();
        pub const Source = SourceKey;
        pub const definition_limits = limits;
        pub const DefinitionOutput = DefinitionType;
        pub const TensorValue = ValueType;
        pub const SourceOverride = DefinitionType.SourceOverride;
        pub const ShiftBoundary = union(enum) {
            wrap,
            edge,
            reflect,
            constant: ValueType,
        };
        pub const SliceLoopIteration = Op.Compute.SliceLoopAttrs.Iteration;
        pub const SliceLoopBoundary = Op.Compute.SliceLoopAttrs.Boundary;
        pub const SliceLoopOptions = struct {
            axis: i8,
            iterations: []const SliceLoopIteration,
        };

        definition: DefinitionType = .{},
        used_sources: [source_capacity]bool = @splat(false),

        pub fn init() Self {
            return .{};
        }

        pub fn input(self: *Self, comptime source_key: SourceKey, comptime dtype: Dtype, comptime shape: []const usize) ValueType {
            return self.addSource(source_key, .input, dtype, shape);
        }

        pub fn parameter(
            self: *Self,
            comptime source_key: SourceKey,
            comptime dtype: Dtype,
            comptime shape: []const usize,
        ) ValueType {
            return self.addSource(source_key, .parameter, dtype, shape);
        }

        pub fn constant(
            self: *Self,
            comptime source_key: SourceKey,
            comptime dtype: Dtype,
            comptime shape: []const usize,
        ) ValueType {
            return self.addSource(source_key, .constant, dtype, shape);
        }

        pub fn scalar(self: *Self, comptime dtype: Dtype, comptime value: dtype.Scalar()) ValueType {
            if (self.definition.tensor_count == limits.max_tensors) @compileError("definition exceeds max_tensors");
            const tensor_id = self.definition.tensor_count;
            const tensor_value: ValueType = .{
                .id = tensor_id,
                .dtype = dtype,
                .shape = .init(&.{}),
            };
            self.definition.tensors[tensor_id] = .{
                .value = tensor_value,
                .origin = .{ .literal = ScalarValue.init(dtype, value) },
            };
            self.definition.tensor_count += 1;
            return tensor_value;
        }

        pub fn full(
            self: *Self,
            comptime dtype: Dtype,
            comptime extents: []const usize,
            comptime value: dtype.Scalar(),
        ) ValueType {
            const scalar_value = self.scalar(dtype, value);
            if (extents.len == 0) return scalar_value;
            return self.broadcastTo(scalar_value, extents);
        }

        pub fn relu(self: *Self, comptime tensor: ValueType) ValueType {
            return self.addCompute(.relu, &.{tensor});
        }

        pub fn exp(self: *Self, comptime tensor: ValueType) ValueType {
            return self.addCompute(.exp, &.{tensor});
        }

        pub fn neg(self: *Self, comptime tensor: ValueType) ValueType {
            return self.addCompute(.neg, &.{tensor});
        }

        pub fn abs(self: *Self, comptime tensor: ValueType) ValueType {
            return self.addCompute(.abs, &.{tensor});
        }

        pub fn sqrt(self: *Self, comptime tensor: ValueType) ValueType {
            return self.addCompute(.sqrt, &.{tensor});
        }

        pub fn log(self: *Self, comptime tensor: ValueType) ValueType {
            return self.addCompute(.log, &.{tensor});
        }

        pub fn reciprocal(self: *Self, comptime tensor: ValueType) ValueType {
            return self.addCompute(.reciprocal, &.{tensor});
        }

        pub fn add(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.addCompute(.add, &.{ lhs, rhs });
        }

        pub fn sub(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.addCompute(.sub, &.{ lhs, rhs });
        }

        pub fn mul(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.addCompute(.mul, &.{ lhs, rhs });
        }

        pub fn div(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.addCompute(.div, &.{ lhs, rhs });
        }

        pub fn minimum(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.addCompute(.minimum, &.{ lhs, rhs });
        }

        pub fn maximum(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.addCompute(.maximum, &.{ lhs, rhs });
        }

        pub fn clamp(
            self: *Self,
            comptime tensor: ValueType,
            comptime lower: ValueType,
            comptime upper: ValueType,
        ) ValueType {
            return self.addCompute(.clamp, &.{ tensor, lower, upper });
        }

        pub fn equal(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.addCompute(.equal, &.{ lhs, rhs });
        }

        pub fn notEqual(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.addCompute(.not_equal, &.{ lhs, rhs });
        }

        pub fn lessThan(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.addCompute(.less_than, &.{ lhs, rhs });
        }

        pub fn lessEqual(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.addCompute(.less_equal, &.{ lhs, rhs });
        }

        pub fn greaterThan(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.addCompute(.greater_than, &.{ lhs, rhs });
        }

        pub fn greaterEqual(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.addCompute(.greater_equal, &.{ lhs, rhs });
        }

        pub fn logicalNot(self: *Self, comptime tensor: ValueType) ValueType {
            return self.addCompute(.logical_not, &.{tensor});
        }

        pub fn logicalAnd(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.addCompute(.logical_and, &.{ lhs, rhs });
        }

        pub fn logicalOr(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.addCompute(.logical_or, &.{ lhs, rhs });
        }

        pub fn where(
            self: *Self,
            comptime condition: ValueType,
            comptime when_true: ValueType,
            comptime when_false: ValueType,
        ) ValueType {
            return self.addCompute(.where, &.{ condition, when_true, when_false });
        }

        /// Materialize a tensor into fresh storage. Lowering may preserve a
        /// useful physical layout while retaining the tensor's logical shape.
        pub fn copy(self: *Self, comptime tensor: ValueType) ValueType {
            return self.addCompute(.copy, &.{tensor});
        }

        /// Materialize a tensor into fresh logical row-major storage.
        pub fn contiguous(self: *Self, comptime tensor: ValueType) ValueType {
            return self.addCompute(.contiguous, &.{tensor});
        }

        /// Materialize constant padding around every input axis.
        pub fn pad(
            self: *Self,
            comptime tensor: ValueType,
            comptime fill: ValueType,
            comptime options: PadOptions,
        ) ValueType {
            return self.addCompute(.{ .pad = .{
                .before = options.before,
                .after = options.after,
            } }, &.{ tensor, fill });
        }

        /// Materialize a translated tensor with one signed offset per axis.
        /// Positive offsets move input values toward higher output coordinates.
        pub fn shift(
            self: *Self,
            comptime tensor: ValueType,
            comptime offsets: []const isize,
            comptime boundary: ShiftBoundary,
        ) ValueType {
            const mode: Op.Compute.ShiftAttrs.Boundary = switch (boundary) {
                .wrap => .wrap,
                .edge => .edge,
                .reflect => .reflect,
                .constant => .constant,
            };
            const attrs: Op.Compute.ShiftAttrs = .{
                .offsets = offsets,
                .boundary = mode,
            };
            return switch (boundary) {
                .constant => |fill| self.addCompute(.{ .shift = attrs }, &.{ tensor, fill }),
                else => self.addCompute(.{ .shift = attrs }, &.{tensor}),
            };
        }

        /// Apply one statically described transform per slice without
        /// expanding the iterations into separate graph nodes.
        pub fn sliceLoop(
            self: *Self,
            comptime tensor: ValueType,
            comptime options: SliceLoopOptions,
        ) ValueType {
            const axis = normalizeAxis(tensor.shape.rank, options.axis);
            return self.addCompute(.{ .slice_loop = .{
                .axis = @intCast(axis),
                .iterations = options.iterations,
            } }, &.{tensor});
        }

        pub fn matmul(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.addCompute(.matmul, &.{ lhs, rhs });
        }

        pub fn sum(self: *Self, comptime tensor: ValueType, comptime options: ReductionOptions) ValueType {
            return self.addCompute(.{ .sum = reductionAttrs(tensor, options) }, &.{tensor});
        }

        pub fn mean(self: *Self, comptime tensor: ValueType, comptime options: ReductionOptions) ValueType {
            return self.addCompute(.{ .mean = reductionAttrs(tensor, options) }, &.{tensor});
        }

        pub fn min(self: *Self, comptime tensor: ValueType, comptime options: ReductionOptions) ValueType {
            return self.addCompute(.{ .min = reductionAttrs(tensor, options) }, &.{tensor});
        }

        pub fn max(self: *Self, comptime tensor: ValueType, comptime options: ReductionOptions) ValueType {
            return self.addCompute(.{ .max = reductionAttrs(tensor, options) }, &.{tensor});
        }

        pub fn concat(
            self: *Self,
            comptime inputs: []const ValueType,
            comptime axis: i8,
        ) ValueType {
            if (inputs.len == 0) @compileError("concat requires at least one input");
            const normalized = normalizeAxis(inputs[0].shape.rank, axis);
            return self.addCompute(.{ .concat = .{ .axis = @intCast(normalized) } }, inputs);
        }

        pub fn softmax(self: *Self, comptime tensor: ValueType, comptime axis: i8) ValueType {
            return self.addCompute(.{ .softmax = .{ .axis = @intCast(normalizeAxis(tensor.shape.rank, axis)) } }, &.{tensor});
        }

        pub fn transpose(
            self: *Self,
            comptime tensor: ValueType,
            comptime axis_a: i8,
            comptime axis_b: i8,
        ) ValueType {
            const normalized_a = normalizeAxis(tensor.shape.rank, axis_a);
            const normalized_b = normalizeAxis(tensor.shape.rank, axis_b);
            var shape = tensor.shape;
            std.mem.swap(usize, &shape.dims[normalized_a], &shape.dims[normalized_b]);
            return self.addNode(
                .{ .view = .{ .transpose = .{
                    .axis_a = @intCast(normalized_a),
                    .axis_b = @intCast(normalized_b),
                } } },
                &.{tensor},
                tensor.dtype,
                shape,
            );
        }

        pub fn reshape(
            self: *Self,
            comptime tensor: ValueType,
            comptime extents: []const usize,
        ) ValueType {
            if (extents.len > limits.max_rank) @compileError("reshape exceeds definition max_rank");
            for (extents) |extent| {
                if (extent == 0) @compileError("tensor dimensions must be greater than zero");
            }
            const shape = Tensor.Shape(limits.max_rank).init(extents);
            if (shape.elementCount() != tensor.shape.elementCount()) {
                @compileError("reshape must preserve the tensor element count");
            }
            return self.addNode(.{ .view = .reshape }, &.{tensor}, tensor.dtype, shape);
        }

        pub fn broadcastTo(
            self: *Self,
            comptime tensor: ValueType,
            comptime extents: []const usize,
        ) ValueType {
            if (extents.len > limits.max_rank) @compileError("broadcast target exceeds definition max_rank");
            if (tensor.shape.rank > extents.len) @compileError("broadcast target rank cannot be smaller than its source rank");
            for (extents) |extent| {
                if (extent == 0) @compileError("tensor dimensions must be greater than zero");
            }
            const rank_offset = extents.len - tensor.shape.rank;
            for (tensor.shape.slice(), 0..) |source_extent, source_axis| {
                const target_extent = extents[rank_offset + source_axis];
                if (source_extent != 1 and source_extent != target_extent) {
                    @compileError("broadcast source extent must equal its target or be one");
                }
            }
            const shape = Tensor.Shape(limits.max_rank).init(extents);
            return self.addNode(.{ .view = .broadcast }, &.{tensor}, tensor.dtype, shape);
        }

        pub fn flatten(
            self: *Self,
            comptime tensor: ValueType,
            comptime options: FlattenOptions,
        ) ValueType {
            const start_axis = normalizeAxis(tensor.shape.rank, options.start_axis);
            const end_axis = normalizeAxis(tensor.shape.rank, options.end_axis);
            if (start_axis > end_axis) @compileError("flatten start_axis must not follow end_axis");

            var shape = Tensor.Shape(limits.max_rank){ .rank = 0, .dims = @splat(0) };
            for (tensor.shape.slice()[0..start_axis]) |extent| {
                shape.dims[shape.rank] = extent;
                shape.rank += 1;
            }
            var flattened_extent: usize = 1;
            for (tensor.shape.slice()[start_axis .. end_axis + 1]) |extent| flattened_extent *= extent;
            shape.dims[shape.rank] = flattened_extent;
            shape.rank += 1;
            for (tensor.shape.slice()[end_axis + 1 ..]) |extent| {
                shape.dims[shape.rank] = extent;
                shape.rank += 1;
            }

            return self.addNode(.{ .view = .{ .flatten = .{
                .start_axis = @intCast(start_axis),
                .end_axis = @intCast(end_axis),
            } } }, &.{tensor}, tensor.dtype, shape);
        }

        pub fn squeeze(self: *Self, comptime tensor: ValueType, comptime axis: i8) ValueType {
            const normalized = normalizeAxis(tensor.shape.rank, axis);
            if (tensor.shape.at(normalized) != 1) @compileError("squeeze axis must have extent one");

            var shape = tensor.shape;
            var current = normalized;
            while (current + 1 < shape.rank) : (current += 1) {
                shape.dims[current] = shape.dims[current + 1];
            }
            shape.rank -= 1;
            shape.dims[shape.rank] = 0;
            return self.addNode(.{ .view = .{ .squeeze = .{ .axis = @intCast(normalized) } } }, &.{tensor}, tensor.dtype, shape);
        }

        pub fn unsqueeze(self: *Self, comptime tensor: ValueType, comptime axis: i8) ValueType {
            if (tensor.shape.rank == limits.max_rank) @compileError("unsqueeze exceeds definition max_rank");
            const normalized = normalizeInsertionAxis(tensor.shape.rank, axis);
            var shape = tensor.shape;
            var current = shape.rank;
            while (current > normalized) : (current -= 1) {
                shape.dims[current] = shape.dims[current - 1];
            }
            shape.dims[normalized] = 1;
            shape.rank += 1;
            return self.addNode(.{ .view = .{ .unsqueeze = .{ .axis = @intCast(normalized) } } }, &.{tensor}, tensor.dtype, shape);
        }

        pub fn permute(
            self: *Self,
            comptime tensor: ValueType,
            comptime axes: []const i8,
        ) ValueType {
            if (axes.len != tensor.shape.rank) @compileError("permute requires one axis for every input dimension");
            var current_axes: [limits.max_rank]usize = undefined;
            var target_axes: [limits.max_rank]usize = undefined;
            for (0..tensor.shape.rank) |axis| current_axes[axis] = axis;
            for (axes, 0..) |axis, index| {
                const normalized = normalizeAxis(tensor.shape.rank, axis);
                for (target_axes[0..index]) |previous| {
                    if (previous == normalized) @compileError("permute axes must be unique");
                }
                target_axes[index] = normalized;
            }

            var result = tensor;
            for (target_axes[0..tensor.shape.rank], 0..) |target_axis, output_axis| {
                var current_position = output_axis;
                while (current_axes[current_position] != target_axis) : (current_position += 1) {}
                if (current_position == output_axis) continue;
                result = self.transpose(result, @intCast(output_axis), @intCast(current_position));
                std.mem.swap(usize, &current_axes[output_axis], &current_axes[current_position]);
            }
            return result;
        }

        pub fn slice(
            self: *Self,
            comptime tensor: ValueType,
            comptime options: SliceOptions,
        ) ValueType {
            const axis = normalizeAxis(tensor.shape.rank, options.axis);
            const extent = tensor.shape.at(axis);
            const end = options.end orelse extent;
            if (options.step == 0) @compileError("slice step must be greater than zero");
            if (options.start >= end or end > extent) {
                @compileError("slice bounds must select a non-empty range within the axis");
            }
            const length = (end - options.start + options.step - 1) / options.step;
            var shape = tensor.shape;
            shape.dims[axis] = length;
            return self.addNode(.{ .view = .{ .slice = .{
                .axis = @intCast(axis),
                .start = options.start,
                .length = length,
                .step = options.step,
            } } }, &.{tensor}, tensor.dtype, shape);
        }

        /// Expose overlapping windows over the trailing input axes. Output
        /// position axes retain their input positions and window axes append
        /// to the result.
        pub fn windows(
            self: *Self,
            comptime tensor: ValueType,
            comptime options: WindowOptions,
        ) ValueType {
            const attrs: Op.View.WindowAttrs = .{
                .sizes = options.sizes,
                .strides = options.strides,
                .dilations = options.dilations,
            };
            const shape = op_module.inferWindowsShape(&.{tensor}, attrs, limits.max_rank);
            return self.addNode(.{ .view = .{ .windows = attrs } }, &.{tensor}, tensor.dtype, shape);
        }

        pub fn output(self: *Self, comptime value: ValueType) void {
            if (self.definition.output_count == limits.max_outputs) {
                @compileError("definition exceeds max_outputs");
            }
            self.definition.outputs[self.definition.output_count] = value.id;
            self.definition.output_count += 1;
        }

        pub fn finish(self: *const Self) DefinitionType {
            return self.definition;
        }

        fn addSource(
            self: *Self,
            comptime source_key: SourceKey,
            comptime kind: Tensor.Source.Kind,
            comptime dtype: Dtype,
            comptime shape_extents: []const usize,
        ) ValueType {
            if (shape_extents.len > limits.max_rank) @compileError("source shape exceeds definition max_rank");
            for (shape_extents) |extent| {
                if (extent == 0) @compileError("tensor dimensions must be greater than zero");
            }
            if (self.definition.tensor_count == limits.max_tensors) @compileError("definition exceeds max_tensors");

            const source_index: usize = @intCast(@intFromEnum(source_key));
            if (self.used_sources[source_index]) @compileError("a source key may only be defined once");
            self.used_sources[source_index] = true;

            const id = self.definition.tensor_count;
            const value: ValueType = .{
                .id = id,
                .dtype = dtype,
                .shape = .init(shape_extents),
            };
            self.definition.tensors[id] = .{
                .value = value,
                .origin = .{ .source = source_index },
                .source_kind = kind,
            };
            self.definition.tensor_count += 1;
            return value;
        }

        fn addCompute(
            self: *Self,
            comptime compute: Op.Compute,
            comptime inputs: []const ValueType,
        ) ValueType {
            const shape = compute.inferShape(inputs, limits.max_rank);
            const dtype = compute.inferDtype(inputs);
            return self.addNode(.{ .compute = compute }, inputs, dtype, shape);
        }

        fn addNode(
            self: *Self,
            comptime op: Op,
            comptime inputs: []const ValueType,
            comptime dtype: Dtype,
            comptime shape: Tensor.Shape(limits.max_rank),
        ) ValueType {
            if (self.definition.node_count == limits.max_nodes) @compileError("definition exceeds max_nodes");
            if (self.definition.tensor_count == limits.max_tensors) @compileError("definition exceeds max_tensors");
            if (self.definition.input_ref_count + inputs.len > limits.max_input_refs) @compileError("definition exceeds max_input_refs");

            const node_id = self.definition.node_count;
            const tensor_id = self.definition.tensor_count;
            const input_start = self.definition.input_ref_count;
            for (inputs) |input_value| {
                self.definition.input_refs[self.definition.input_ref_count] = input_value.id;
                self.definition.input_ref_count += 1;
            }
            self.definition.nodes[node_id] = .{
                .op = op,
                .input_start = input_start,
                .input_count = inputs.len,
                .result = tensor_id,
            };
            const value: ValueType = .{ .id = tensor_id, .dtype = dtype, .shape = shape };
            self.definition.tensors[tensor_id] = .{
                .value = value,
                .origin = .{ .node = node_id },
            };
            self.definition.node_count += 1;
            self.definition.tensor_count += 1;
            return value;
        }

        fn reductionAttrs(comptime tensor: ValueType, comptime options: ReductionOptions) Op.Compute.ReductionAttrs {
            return .{
                .axes = reductionAxesMask(tensor.shape.rank, options.axes),
                .keep_dims = options.keep_dims,
            };
        }
    };
}

fn reductionAxesMask(comptime rank: usize, comptime axes: ?[]const i8) u64 {
    return if (axes) |explicit| explicitAxesMask(rank, explicit) else allAxesMask(rank);
}

fn explicitAxesMask(comptime rank: usize, comptime axes: []const i8) u64 {
    if (axes.len == 0) @compileError("reduction axes cannot be empty");
    var result: u64 = 0;
    for (axes) |axis| {
        const mask = axisMask(rank, axis);
        if (result & mask != 0) @compileError("reduction axes must be unique");
        result |= mask;
    }
    return result;
}

fn allAxesMask(comptime rank: usize) u64 {
    if (rank == 0 or rank > 64) @compileError("reductions support tensor ranks from 1 through 64");
    return if (rank == 64) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(rank)) - 1;
}

fn axisMask(comptime rank: usize, comptime requested_axis: i8) u64 {
    if (rank == 0 or rank > 64) @compileError("reductions support tensor ranks from 1 through 64");
    return @as(u64, 1) << @intCast(normalizeAxis(rank, requested_axis));
}

fn normalizeAxis(comptime rank: usize, comptime requested_axis: i8) usize {
    if (rank == 0) @compileError("cannot select an axis from a rank-zero tensor");
    const axis: isize = @intCast(requested_axis);
    const normalized = if (axis < 0) axis + @as(isize, @intCast(rank)) else axis;
    if (normalized < 0 or normalized >= rank) @compileError("axis is outside the input rank");
    return @intCast(normalized);
}

fn normalizeInsertionAxis(comptime rank: usize, comptime requested_axis: i8) usize {
    const axis: isize = @intCast(requested_axis);
    const normalized = if (axis < 0) axis + @as(isize, @intCast(rank + 1)) else axis;
    if (normalized < 0 or normalized > rank) @compileError("insertion axis is outside the output rank");
    return @intCast(normalized);
}

fn enumCapacity(comptime Enum: type) usize {
    const info = @typeInfo(Enum);
    if (info != .@"enum") @compileError("DefinitionBuilder source keys must be an enum type");
    var capacity: usize = 0;
    for (info.@"enum".fields) |field| {
        if (field.value < 0) @compileError("source enum values must be non-negative");
        capacity = @max(capacity, @as(usize, @intCast(field.value)) + 1);
    }
    return capacity;
}
