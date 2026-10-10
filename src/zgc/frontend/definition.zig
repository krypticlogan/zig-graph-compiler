const std = @import("std");
const Dtype = @import("../storage/dtype.zig").Dtype;
const ScalarValue = @import("../storage/dtype.zig").ScalarValue;
const op_module = @import("../operations/semantic.zig");
const Op = op_module.Op;
const SourceStorage = @import("../storage/source.zig");
const Tensor = @import("../core/tensor.zig");
const Expr = @import("expressive.zig");
const EmptySourceKey = enum(usize) {};
pub const SerializedSourceKey = enum(usize) { _ };

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

/// Tensor shape description for graph construction. The shape is immutable.
pub const BuildShape = struct {
    rank: usize,
    dims: []const usize,

    pub fn init(comptime dims: []const usize) BuildShape {
        for (dims) |extent| {
            if (extent == 0) @compileError("tensor dimensions must be greater than zero");
        }
        const owned_dims = dims[0..dims.len].*;
        return .{ .rank = dims.len, .dims = &owned_dims };
    }

    pub fn slice(self: BuildShape) []const usize {
        return self.dims;
    }

    pub fn at(self: BuildShape, axis: usize) usize {
        return self.dims[axis];
    }

    pub fn elementCount(self: BuildShape) usize {
        var count: usize = 1;
        for (self.dims) |extent| count *= extent;
        return count;
    }
};

/// Tensor value type for graph construction. The value is immutable.
pub const Value = struct {
    id: Tensor.Id,
    dtype: Dtype,
    shape: BuildShape,
};

pub const Node = struct {
    op: Op,
    input_start: usize,
    input_count: usize,
    result: Tensor.Id,
};

pub const TensorRecord = struct {
    value: Value,
    origin: Tensor.Origin,
    source_kind: ?Tensor.Source.Kind = null,
};

/// Compile-time graph definition. Lowered to a `zgc.Execution.Program` with `model()` or `modelWith()`.
pub fn Definition(
    comptime SourceKey: type,
    comptime node_count: usize,
    comptime tensor_count: usize,
    comptime input_ref_count: usize,
    comptime output_count: usize,
    comptime rank_capacity: usize,
    comptime source_name_count: usize,
) type {
    return struct {
        const Self = @This();
        pub const max_rank = rank_capacity;
        pub const Source = SourceKey;
        pub const SourceOverride = struct {
            /// Source enum tag whose default ownership policy is replaced.
            source: SourceKey,
            /// Storage policy applied to the selected source.
            binding: SourceStorage.Binding,
        };

        nodes: [node_count]Node,
        tensors: [tensor_count]TensorRecord,
        input_refs: [input_ref_count]Tensor.Id,
        outputs: [output_count]Tensor.Id,
        source_names: [source_name_count][:0]const u8,
        node_count: usize = node_count,
        tensor_count: usize = tensor_count,
        input_ref_count: usize = input_ref_count,
        output_count: usize = output_count,

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
    };
}

/// Capacity-free graph construction interface.
pub const DefinitionBuilder = struct {
    const Self = @This();
    pub const ShiftBoundary = union(enum) {
        wrap,
        edge,
        reflect,
        constant: Value,
    };
    pub const SliceLoopIteration = Op.Compute.SliceLoopAttrs.Iteration;
    pub const SliceLoopBoundary = Op.Compute.SliceLoopAttrs.Boundary;
    pub const SliceLoopOptions = struct {
        axis: i8,
        iterations: []const SliceLoopIteration,
    };

    nodes: []const Node = &.{},
    tensors: []const TensorRecord = &.{},
    input_refs: []const Tensor.Id = &.{},
    outputs: []const Tensor.Id = &.{},
    source_key_type: ?[]const type = null,
    source_names: []const [:0]const u8 = &.{},

    pub fn init() Self {
        @setEvalBranchQuota(100_000);
        return .{};
    }

    pub fn sources(comptime self: *Self, comptime SourceKey: type) Sources(SourceKey) {
        validateSourceKey(SourceKey);
        if (self.source_key_type) |existing| {
            if (existing[0] != SourceKey) {
                @compileError("a definition may only use one source key type");
            }
        } else {
            self.source_key_type = &[_]type{SourceKey};
            const fields = @typeInfo(SourceKey).@"enum".fields;
            var names: [sourceNameCapacity(SourceKey)][:0]const u8 = @splat("");
            for (fields) |field| names[@intCast(field.value)] = field.name;
            self.source_names = &names;
        }
        return .{ .builder = self };
    }

    /// Attach source names supplied by a serialized frontend. Numeric source
    /// keys remain internal to the replayed definition.
    pub fn serializedSources(
        comptime self: *Self,
        comptime names: []const [:0]const u8,
    ) Sources(SerializedSourceKey) {
        if (self.source_key_type != null) @compileError("definition sources are already initialized");
        self.source_key_type = &[_]type{SerializedSourceKey};
        self.source_names = names;
        return .{ .builder = self };
    }

    pub fn expr(comptime self: *Self, comptime value: Value) Expr {
        return .{
            .b = self,
            .value = value,
        };
    }

    pub fn scalar(comptime self: *Self, comptime dtype: Dtype, comptime value: dtype.Scalar()) Value {
        const tensor_id = self.tensors.len;
        const tensor_value: Value = .{
            .id = tensor_id,
            .dtype = dtype,
            .shape = BuildShape.init(&.{}),
        };
        self.appendTensor(.{
            .value = tensor_value,
            .origin = .{ .literal = ScalarValue.init(dtype, value) },
        });
        return tensor_value;
    }

    pub fn full(
        self: *Self,
        comptime dtype: Dtype,
        comptime extents: []const usize,
        comptime value: dtype.Scalar(),
    ) Value {
        const scalar_value = self.scalar(dtype, value);
        if (extents.len == 0) return scalar_value;
        return self.broadcastTo(scalar_value, extents);
    }

    pub fn relu(self: *Self, comptime tensor: Value) Value {
        return self.addCompute(.relu, &.{tensor});
    }

    pub fn exp(self: *Self, comptime tensor: Value) Value {
        return self.addCompute(.exp, &.{tensor});
    }

    pub fn neg(self: *Self, comptime tensor: Value) Value {
        return self.addCompute(.neg, &.{tensor});
    }

    pub fn abs(self: *Self, comptime tensor: Value) Value {
        return self.addCompute(.abs, &.{tensor});
    }

    pub fn sqrt(self: *Self, comptime tensor: Value) Value {
        return self.addCompute(.sqrt, &.{tensor});
    }

    pub fn log(self: *Self, comptime tensor: Value) Value {
        return self.addCompute(.log, &.{tensor});
    }

    pub fn reciprocal(self: *Self, comptime tensor: Value) Value {
        return self.addCompute(.reciprocal, &.{tensor});
    }

    pub fn add(self: *Self, comptime lhs: Value, comptime rhs: Value) Value {
        return self.addCompute(.add, &.{ lhs, rhs });
    }

    pub fn sub(self: *Self, comptime lhs: Value, comptime rhs: Value) Value {
        return self.addCompute(.sub, &.{ lhs, rhs });
    }

    pub fn mul(self: *Self, comptime lhs: Value, comptime rhs: Value) Value {
        return self.addCompute(.mul, &.{ lhs, rhs });
    }

    pub fn div(self: *Self, comptime lhs: Value, comptime rhs: Value) Value {
        return self.addCompute(.div, &.{ lhs, rhs });
    }

    pub fn minimum(self: *Self, comptime lhs: Value, comptime rhs: Value) Value {
        return self.addCompute(.minimum, &.{ lhs, rhs });
    }

    pub fn maximum(self: *Self, comptime lhs: Value, comptime rhs: Value) Value {
        return self.addCompute(.maximum, &.{ lhs, rhs });
    }

    pub fn clamp(
        self: *Self,
        comptime tensor: Value,
        comptime lower: Value,
        comptime upper: Value,
    ) Value {
        return self.addCompute(.clamp, &.{ tensor, lower, upper });
    }

    pub fn equal(self: *Self, comptime lhs: Value, comptime rhs: Value) Value {
        return self.addCompute(.equal, &.{ lhs, rhs });
    }

    pub fn notEqual(self: *Self, comptime lhs: Value, comptime rhs: Value) Value {
        return self.addCompute(.not_equal, &.{ lhs, rhs });
    }

    pub fn lessThan(self: *Self, comptime lhs: Value, comptime rhs: Value) Value {
        return self.addCompute(.less_than, &.{ lhs, rhs });
    }

    pub fn lessEqual(self: *Self, comptime lhs: Value, comptime rhs: Value) Value {
        return self.addCompute(.less_equal, &.{ lhs, rhs });
    }

    pub fn greaterThan(self: *Self, comptime lhs: Value, comptime rhs: Value) Value {
        return self.addCompute(.greater_than, &.{ lhs, rhs });
    }

    pub fn greaterEqual(self: *Self, comptime lhs: Value, comptime rhs: Value) Value {
        return self.addCompute(.greater_equal, &.{ lhs, rhs });
    }

    pub fn logicalNot(self: *Self, comptime tensor: Value) Value {
        return self.addCompute(.logical_not, &.{tensor});
    }

    pub fn logicalAnd(self: *Self, comptime lhs: Value, comptime rhs: Value) Value {
        return self.addCompute(.logical_and, &.{ lhs, rhs });
    }

    pub fn logicalOr(self: *Self, comptime lhs: Value, comptime rhs: Value) Value {
        return self.addCompute(.logical_or, &.{ lhs, rhs });
    }

    pub fn where(
        self: *Self,
        comptime condition: Value,
        comptime when_true: Value,
        comptime when_false: Value,
    ) Value {
        return self.addCompute(.where, &.{ condition, when_true, when_false });
    }

    /// Materialize a tensor into fresh storage. Lowering may preserve a
    /// useful physical layout while retaining the tensor's logical shape.
    pub fn copy(self: *Self, comptime tensor: Value) Value {
        return self.addCompute(.copy, &.{tensor});
    }

    /// Materialize a tensor into fresh logical row-major storage.
    pub fn contiguous(self: *Self, comptime tensor: Value) Value {
        return self.addCompute(.contiguous, &.{tensor});
    }

    /// Materialize constant padding around every input axis.
    pub fn pad(
        self: *Self,
        comptime tensor: Value,
        comptime fill: Value,
        comptime options: PadOptions,
    ) Value {
        return self.addCompute(.{ .pad = .{
            .before = options.before,
            .after = options.after,
        } }, &.{ tensor, fill });
    }

    /// Materialize a translated tensor with one signed offset per axis.
    /// Positive offsets move input values toward higher output coordinates.
    pub fn shift(
        self: *Self,
        comptime tensor: Value,
        comptime offsets: []const isize,
        comptime boundary: ShiftBoundary,
    ) Value {
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
        comptime tensor: Value,
        comptime options: SliceLoopOptions,
    ) Value {
        const axis = normalizeAxis(tensor.shape.rank, options.axis);
        return self.addCompute(.{ .slice_loop = .{
            .axis = @intCast(axis),
            .iterations = options.iterations,
        } }, &.{tensor});
    }

    pub fn matmul(self: *Self, comptime lhs: Value, comptime rhs: Value) Value {
        return self.addCompute(.matmul, &.{ lhs, rhs });
    }

    pub fn sum(self: *Self, comptime tensor: Value, comptime options: ReductionOptions) Value {
        return self.addCompute(.{ .sum = reductionAttrs(tensor, options) }, &.{tensor});
    }

    pub fn mean(self: *Self, comptime tensor: Value, comptime options: ReductionOptions) Value {
        return self.addCompute(.{ .mean = reductionAttrs(tensor, options) }, &.{tensor});
    }

    pub fn min(self: *Self, comptime tensor: Value, comptime options: ReductionOptions) Value {
        return self.addCompute(.{ .min = reductionAttrs(tensor, options) }, &.{tensor});
    }

    pub fn max(self: *Self, comptime tensor: Value, comptime options: ReductionOptions) Value {
        return self.addCompute(.{ .max = reductionAttrs(tensor, options) }, &.{tensor});
    }

    pub fn concat(
        self: *Self,
        comptime inputs: []const Value,
        comptime axis: i8,
    ) Value {
        if (inputs.len == 0) @compileError("concat requires at least one input");
        const normalized = normalizeAxis(inputs[0].shape.rank, axis);
        return self.addCompute(.{ .concat = .{ .axis = @intCast(normalized) } }, inputs);
    }

    pub fn softmax(self: *Self, comptime tensor: Value, comptime axis: i8) Value {
        return self.addCompute(.{ .softmax = .{ .axis = @intCast(normalizeAxis(tensor.shape.rank, axis)) } }, &.{tensor});
    }

    pub fn transpose(
        self: *Self,
        comptime tensor: Value,
        comptime axis_a: i8,
        comptime axis_b: i8,
    ) Value {
        const normalized_a = normalizeAxis(tensor.shape.rank, axis_a);
        const normalized_b = normalizeAxis(tensor.shape.rank, axis_b);
        var dims: [tensor.shape.rank]usize = tensor.shape.slice()[0..tensor.shape.rank].*;
        std.mem.swap(usize, &dims[normalized_a], &dims[normalized_b]);
        const shape = BuildShape.init(&dims);
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
        comptime tensor: Value,
        comptime extents: []const usize,
    ) Value {
        const shape = BuildShape.init(extents);
        if (shape.elementCount() != tensor.shape.elementCount()) {
            @compileError("reshape must preserve the tensor element count");
        }
        return self.addNode(.{ .view = .reshape }, &.{tensor}, tensor.dtype, shape);
    }

    pub fn broadcastTo(
        self: *Self,
        comptime tensor: Value,
        comptime extents: []const usize,
    ) Value {
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
        const shape = BuildShape.init(extents);
        return self.addNode(.{ .view = .broadcast }, &.{tensor}, tensor.dtype, shape);
    }

    pub fn flatten(
        self: *Self,
        comptime tensor: Value,
        comptime options: FlattenOptions,
    ) Value {
        const start_axis = normalizeAxis(tensor.shape.rank, options.start_axis);
        const end_axis = normalizeAxis(tensor.shape.rank, options.end_axis);
        if (start_axis > end_axis) @compileError("flatten start_axis must not follow end_axis");

        const output_rank = tensor.shape.rank - (end_axis - start_axis);
        var dims: [output_rank]usize = undefined;
        var output_axis: usize = 0;
        for (tensor.shape.slice()[0..start_axis]) |extent| {
            dims[output_axis] = extent;
            output_axis += 1;
        }
        var flattened_extent: usize = 1;
        for (tensor.shape.slice()[start_axis .. end_axis + 1]) |extent| flattened_extent *= extent;
        dims[output_axis] = flattened_extent;
        output_axis += 1;
        for (tensor.shape.slice()[end_axis + 1 ..]) |extent| {
            dims[output_axis] = extent;
            output_axis += 1;
        }
        const shape = BuildShape.init(&dims);

        return self.addNode(.{ .view = .{ .flatten = .{
            .start_axis = @intCast(start_axis),
            .end_axis = @intCast(end_axis),
        } } }, &.{tensor}, tensor.dtype, shape);
    }

    pub fn squeeze(self: *Self, comptime tensor: Value, comptime axis: i8) Value {
        const normalized = normalizeAxis(tensor.shape.rank, axis);
        if (tensor.shape.at(normalized) != 1) @compileError("squeeze axis must have extent one");

        var dims: [tensor.shape.rank - 1]usize = undefined;
        var output_axis: usize = 0;
        for (tensor.shape.slice(), 0..) |extent, input_axis| {
            if (input_axis == normalized) continue;
            dims[output_axis] = extent;
            output_axis += 1;
        }
        const shape = BuildShape.init(&dims);
        return self.addNode(.{ .view = .{ .squeeze = .{ .axis = @intCast(normalized) } } }, &.{tensor}, tensor.dtype, shape);
    }

    pub fn unsqueeze(self: *Self, comptime tensor: Value, comptime axis: i8) Value {
        const normalized = normalizeInsertionAxis(tensor.shape.rank, axis);
        var dims: [tensor.shape.rank + 1]usize = undefined;
        for (0..dims.len) |output_axis| {
            dims[output_axis] = if (output_axis < normalized)
                tensor.shape.at(output_axis)
            else if (output_axis == normalized)
                1
            else
                tensor.shape.at(output_axis - 1);
        }
        const shape = BuildShape.init(&dims);
        return self.addNode(.{ .view = .{ .unsqueeze = .{ .axis = @intCast(normalized) } } }, &.{tensor}, tensor.dtype, shape);
    }

    pub fn permute(
        self: *Self,
        comptime tensor: Value,
        comptime axes: []const i8,
    ) Value {
        if (axes.len != tensor.shape.rank) @compileError("permute requires one axis for every input dimension");
        var current_axes: [tensor.shape.rank]usize = undefined;
        var target_axes: [tensor.shape.rank]usize = undefined;
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
        comptime tensor: Value,
        comptime options: SliceOptions,
    ) Value {
        const axis = normalizeAxis(tensor.shape.rank, options.axis);
        const extent = tensor.shape.at(axis);
        const end = options.end orelse extent;
        if (options.step == 0) @compileError("slice step must be greater than zero");
        if (options.start >= end or end > extent) {
            @compileError("slice bounds must select a non-empty range within the axis");
        }
        const length = (end - options.start + options.step - 1) / options.step;
        var dims: [tensor.shape.rank]usize = tensor.shape.slice()[0..tensor.shape.rank].*;
        dims[axis] = length;
        const shape = BuildShape.init(&dims);
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
        comptime tensor: Value,
        comptime options: WindowOptions,
    ) Value {
        const attrs: Op.View.WindowAttrs = .{
            .sizes = options.sizes,
            .strides = options.strides,
            .dilations = options.dilations,
        };
        const inferred = op_module.inferWindowsShape(&.{tensor}, attrs, tensor.shape.rank + attrs.sizes.len);
        return self.addNode(.{ .view = .{ .windows = attrs } }, &.{tensor}, tensor.dtype, BuildShape.init(inferred.slice()));
    }

    pub fn output(comptime self: *Self, comptime value: Value) void {
        self.outputs = self.outputs ++ &[_]Tensor.Id{value.id};
    }

    pub fn finish(comptime self: *const Self) Definition(
        self.sourceKey(),
        self.nodes.len,
        self.tensors.len,
        self.input_refs.len,
        self.outputs.len,
        maximumRank(self.tensors),
        self.source_names.len,
    ) {
        const SourceKey = self.sourceKey();
        validateSources(self.tensors, SourceKey, self.source_names);
        const Result = Definition(
            SourceKey,
            self.nodes.len,
            self.tensors.len,
            self.input_refs.len,
            self.outputs.len,
            maximumRank(self.tensors),
            self.source_names.len,
        );
        return Result{
            .nodes = self.nodes[0..self.nodes.len].*,
            .tensors = self.tensors[0..self.tensors.len].*,
            .input_refs = self.input_refs[0..self.input_refs.len].*,
            .outputs = self.outputs[0..self.outputs.len].*,
            .source_names = self.source_names[0..self.source_names.len].*,
        };
    }

    fn sourceKey(comptime self: *const Self) type {
        return if (self.source_key_type) |source_key| source_key[0] else EmptySourceKey;
    }

    fn addSource(
        comptime self: *Self,
        comptime source_index: usize,
        comptime kind: Tensor.Source.Kind,
        comptime dtype: Dtype,
        comptime shape_extents: []const usize,
    ) Value {
        for (self.tensors) |tensor| switch (tensor.origin) {
            .source => |existing| if (existing == source_index) @compileError("a source key may only be defined once"),
            .node, .literal => {},
        };

        const id = self.tensors.len;
        const value: Value = .{
            .id = id,
            .dtype = dtype,
            .shape = BuildShape.init(shape_extents),
        };
        self.appendTensor(.{
            .value = value,
            .origin = .{ .source = source_index },
            .source_kind = kind,
        });
        return value;
    }

    fn addCompute(
        self: *Self,
        comptime compute: Op.Compute,
        comptime inputs: []const Value,
    ) Value {
        const shape_capacity = maximumInputRank(inputs);
        const InferenceValue = struct {
            dtype: Dtype,
            shape: Tensor.Shape(shape_capacity),
        };
        var inference_inputs: [inputs.len]InferenceValue = undefined;
        inline for (inputs, 0..) |input, index| {
            inference_inputs[index] = .{
                .dtype = input.dtype,
                .shape = .init(input.shape.slice()),
            };
        }
        const inferred = compute.inferShape(&inference_inputs, shape_capacity);
        const dtype = compute.inferDtype(inputs);
        return self.addNode(.{ .compute = compute }, inputs, dtype, BuildShape.init(inferred.slice()));
    }

    fn addNode(
        self: *Self,
        comptime op: Op,
        comptime inputs: []const Value,
        comptime dtype: Dtype,
        comptime shape: BuildShape,
    ) Value {
        const node_id = self.nodes.len;
        const tensor_id = self.tensors.len;
        const input_start = self.input_refs.len;
        for (inputs) |input_value| {
            self.input_refs = self.input_refs ++ &[_]Tensor.Id{input_value.id};
        }
        self.appendNode(.{
            .op = op,
            .input_start = input_start,
            .input_count = inputs.len,
            .result = tensor_id,
        });
        const value: Value = .{ .id = tensor_id, .dtype = dtype, .shape = shape };
        self.appendTensor(.{
            .value = value,
            .origin = .{ .node = node_id },
        });
        return value;
    }

    fn appendNode(comptime self: *Self, comptime node: Node) void {
        self.nodes = self.nodes ++ &[_]Node{node};
    }

    fn appendTensor(comptime self: *Self, comptime tensor: TensorRecord) void {
        self.tensors = self.tensors ++ &[_]TensorRecord{tensor};
    }

    fn reductionAttrs(comptime tensor: Value, comptime options: ReductionOptions) Op.Compute.ReductionAttrs {
        return .{
            .axes = reductionAxesMask(tensor.shape.rank, options.axes),
            .keep_dims = options.keep_dims,
        };
    }
};

pub fn Sources(comptime SourceKey: type) type {
    return struct {
        builder: *DefinitionBuilder,

        pub fn input(
            comptime self: @This(),
            comptime key: SourceKey,
            comptime dtype: Dtype,
            comptime shape: []const usize,
        ) Value {
            return self.builder.addSource(@intCast(@intFromEnum(key)), .input, dtype, shape);
        }

        pub fn parameter(
            comptime self: @This(),
            comptime key: SourceKey,
            comptime dtype: Dtype,
            comptime shape: []const usize,
        ) Value {
            return self.builder.addSource(@intCast(@intFromEnum(key)), .parameter, dtype, shape);
        }

        pub fn constant(
            comptime self: @This(),
            comptime key: SourceKey,
            comptime dtype: Dtype,
            comptime shape: []const usize,
        ) Value {
            return self.builder.addSource(@intCast(@intFromEnum(key)), .constant, dtype, shape);
        }
    };
}

fn maximumInputRank(comptime inputs: []const Value) usize {
    var result: usize = 0;
    for (inputs) |input| result = @max(result, input.shape.rank);
    return result;
}

fn maximumRank(comptime tensors: []const TensorRecord) usize {
    var result: usize = 0;
    for (tensors) |tensor| result = @max(result, tensor.value.shape.rank);
    return result;
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

fn validateSourceKey(comptime Enum: type) void {
    const info = @typeInfo(Enum);
    if (info != .@"enum") @compileError("DefinitionBuilder source keys must be an enum type");
    for (info.@"enum".fields) |field| {
        if (field.value < 0) @compileError("source enum values must be non-negative");
    }
}

fn sourceNameCapacity(comptime Enum: type) usize {
    var capacity: usize = 0;
    for (@typeInfo(Enum).@"enum".fields) |field| {
        capacity = @max(capacity, @as(usize, @intCast(field.value)) + 1);
    }
    return capacity;
}

fn validateSources(
    comptime tensors: []const TensorRecord,
    comptime SourceKey: type,
    comptime source_names: []const [:0]const u8,
) void {
    const fields = @typeInfo(SourceKey).@"enum".fields;
    for (tensors) |tensor| switch (tensor.origin) {
        .source => |source_index| {
            var found = false;
            for (fields) |field| {
                if (field.value == source_index) {
                    found = true;
                    break;
                }
            }
            if (!found and @typeInfo(SourceKey).@"enum".is_exhaustive) {
                @compileError("definition contains a source outside its SourceKey enum");
            }
            if (source_index >= source_names.len) {
                @compileError("definition source does not have a corresponding name");
            }
        },
        .node, .literal => {},
    };
}
