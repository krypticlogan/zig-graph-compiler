const std = @import("std");
const Dtype = @import("../storage/dtype.zig").Dtype;
const ScalarValue = @import("../storage/dtype.zig").ScalarValue;

pub const Id = usize;
pub const Shape_T = []const usize;

/// View into memory for a given tensor.
pub fn View(comptime T: type, comptime tensor_rank: usize) type {
    return struct {
        const Self = @This();
        pub const scalar_type = T;
        pub const dtype = Dtype.fromScalar(T);
        pub const rank = tensor_rank;
        // pub const len = Shape(tensor_rank).init().elementCount();

        storage: []T,
        shape: [tensor_rank]usize,
        strides: [tensor_rank]isize,
        offset: usize,

        pub fn len(self: *const Self) usize {
            return Shape(rank).init(&self.shape).elementCount();
        }
        pub fn elementOffset(
            self: *const Self,
            indices: [rank]usize,
        ) usize {
            var offset: isize = @intCast(self.offset);
            for (self.strides, indices, 0..) |stride, index, axis| {
                std.debug.assert(index < self.shape[axis]);
                offset += @as(isize, @intCast(index)) * stride;
            }
            std.debug.assert(offset >= 0);
            const storage_index: usize = @intCast(offset);
            std.debug.assert(storage_index < self.storage.len);
            return storage_index;
        }

        pub fn elementOffsetFromLinear(
            self: *const Self,
            linear_index: usize,
        ) usize {
            std.debug.assert(linear_index < self.len());
            if (comptime rank == 0) {
                std.debug.assert(self.offset < self.storage.len);
                return self.offset;
            }
            var remaining = linear_index;
            var offset: isize = @intCast(self.offset);
            var axis = rank;
            while (axis > 0) {
                axis -= 1;
                const extent = self.shape[axis];
                const index = remaining % extent;
                remaining /= extent;
                offset += @as(isize, @intCast(index)) * self.strides[axis];
            }
            std.debug.assert(offset >= 0);
            const storage_index: usize = @intCast(offset);
            std.debug.assert(storage_index < self.storage.len);
            return storage_index;
        }

        pub fn get(
            self: *const Self,
            indices: [rank]usize,
        ) T {
            const elem_index = self.elementOffset(indices);
            return self.storage[elem_index];
        }

        pub fn set(
            self: *const Self,
            indices: [rank]usize,
            value: T,
        ) void {
            const elem_index = self.elementOffset(indices);
            self.storage[elem_index] = value;
        }

        pub fn isContiguous(self: *const Self) bool {
            var expected: isize = 1;
            var axis = rank;
            while (axis > 0) {
                axis -= 1;
                if (self.shape[axis] > 1 and self.strides[axis] != expected)
                    return false;
                expected *= @intCast(self.shape[axis]);
            }
            return true;
        }

        pub fn contiguousSlice(self: *const Self) ?[]T {
            if (!self.isContiguous()) return null;
            return self.storage[self.offset..][0..self.len()];
        }

        /// Return the physical storage span for any dense positive-stride
        /// layout, including axis permutations such as strides [1, M]. The
        /// slice is in physical rather than logical row-major order.
        pub fn denseSlice(self: *const Self) ?[]T {
            if (!isDensePositive(self)) return null;
            return self.storage[self.offset..][0..self.len()];
        }

        /// Select one logical line along `axis`. `slice_index` addresses the
        /// remaining axes in row-major logical order. The returned view aliases
        /// the same storage and preserves the selected axis stride.
        pub fn axisSlice(
            self: *const Self,
            axis: usize,
            slice_index: usize,
        ) View(T, 1) {
            const slice_offset = axisSliceOffset(self, axis, slice_index);
            return .{
                .storage = self.storage,
                .shape = .{self.shape[axis]},
                .strides = .{self.strides[axis]},
                .offset = slice_offset,
            };
        }

        /// Expand this view to a compile-time-known target rank. Broadcast
        /// compatibility is established by graph validation; singleton and
        /// newly introduced leading axes are represented with zero strides.
        pub fn broadcastTo(
            self: *const Self,
            comptime target_rank: usize,
            target_shape: [target_rank]usize,
        ) View(T, target_rank) {
            if (comptime rank > target_rank) {
                @compileError("broadcast target rank cannot be smaller than its source rank");
            }

            var result = View(T, target_rank){
                .storage = self.storage,
                .shape = target_shape,
                .strides = @splat(0),
                .offset = self.offset,
            };

            if (comptime rank > 0) {
                var source_axis = rank;
                var target_axis = target_rank;
                while (source_axis > 0) {
                    source_axis -= 1;
                    target_axis -= 1;
                    if (self.shape[source_axis] == target_shape[target_axis]) {
                        result.strides[target_axis] = self.strides[source_axis];
                    }
                }
            }
            return result;
        }
    };
}

/// Read-only view into memory for a given tensor.
pub fn ConstView(comptime T: type, comptime tensor_rank: usize) type {
    return struct {
        const Self = @This();
        pub const scalar_type = T;
        pub const dtype = Dtype.fromScalar(T);
        pub const rank = tensor_rank;

        storage: []const T,
        shape: [tensor_rank]usize,
        strides: [tensor_rank]isize,
        offset: usize,

        pub fn len(self: *const Self) usize {
            return Shape(rank).init(&self.shape).elementCount();
        }
        pub fn elementOffset(
            self: *const Self,
            indices: [rank]usize,
        ) usize {
            var offset: isize = @intCast(self.offset);
            for (self.strides, indices, 0..) |stride, index, axis| {
                std.debug.assert(index < self.shape[axis]);
                offset += @as(isize, @intCast(index)) * stride;
            }
            std.debug.assert(offset >= 0);
            const storage_index: usize = @intCast(offset);
            std.debug.assert(storage_index < self.storage.len);
            return storage_index;
        }

        pub fn elementOffsetFromLinear(
            self: *const Self,
            linear_index: usize,
        ) usize {
            std.debug.assert(linear_index < self.len());
            if (comptime rank == 0) {
                std.debug.assert(self.offset < self.storage.len);
                return self.offset;
            }
            var remaining = linear_index;
            var offset: isize = @intCast(self.offset);
            var axis = rank;
            while (axis > 0) {
                axis -= 1;
                const extent = self.shape[axis];
                const index = remaining % extent;
                remaining /= extent;
                offset += @as(isize, @intCast(index)) * self.strides[axis];
            }
            std.debug.assert(offset >= 0);
            const storage_index: usize = @intCast(offset);
            std.debug.assert(storage_index < self.storage.len);
            return storage_index;
        }

        pub fn get(
            self: *const Self,
            indices: [rank]usize,
        ) T {
            const elem_index = self.elementOffset(indices);
            return self.storage[elem_index];
        }

        pub fn isContiguous(self: *const Self) bool {
            var expected: isize = 1;
            var axis = rank;
            while (axis > 0) {
                axis -= 1;
                if (self.shape[axis] > 1 and self.strides[axis] != expected)
                    return false;
                expected *= @intCast(self.shape[axis]);
            }
            return true;
        }

        pub fn contiguousSlice(self: *const Self) ?[]const T {
            if (!self.isContiguous()) return null;
            return self.storage[self.offset..][0..self.len()];
        }

        /// Read-only counterpart to `View.denseSlice`.
        pub fn denseSlice(self: *const Self) ?[]const T {
            if (!isDensePositive(self)) return null;
            return self.storage[self.offset..][0..self.len()];
        }

        /// Read-only counterpart to `View.axisSlice`.
        pub fn axisSlice(
            self: *const Self,
            axis: usize,
            slice_index: usize,
        ) ConstView(T, 1) {
            const slice_offset = axisSliceOffset(self, axis, slice_index);
            return .{
                .storage = self.storage,
                .shape = .{self.shape[axis]},
                .strides = .{self.strides[axis]},
                .offset = slice_offset,
            };
        }

        /// Read-only counterpart to `View.broadcastTo`.
        pub fn broadcastTo(
            self: *const Self,
            comptime target_rank: usize,
            target_shape: [target_rank]usize,
        ) ConstView(T, target_rank) {
            if (comptime rank > target_rank) {
                @compileError("broadcast target rank cannot be smaller than its source rank");
            }

            var result = ConstView(T, target_rank){
                .storage = self.storage,
                .shape = target_shape,
                .strides = @splat(0),
                .offset = self.offset,
            };

            if (comptime rank > 0) {
                var source_axis = rank;
                var target_axis = target_rank;
                while (source_axis > 0) {
                    source_axis -= 1;
                    target_axis -= 1;
                    if (self.shape[source_axis] == target_shape[target_axis]) {
                        result.strides[target_axis] = self.strides[source_axis];
                    }
                }
            }
            return result;
        }
    };
}

/// Mutable view whose tensor geometry is part of its type. Compiled models use
/// this form so shape, strides, base offset, element count, and layout
/// properties are available to kernels at compile time.
pub fn StaticView(
    comptime T: type,
    comptime tensor_shape: anytype,
    comptime tensor_strides: anytype,
    comptime tensor_offset: usize,
) type {
    return StaticViewImpl(T, tensor_shape, tensor_strides, tensor_offset, false);
}

/// Read-only counterpart to `StaticView`.
pub fn StaticConstView(
    comptime T: type,
    comptime tensor_shape: anytype,
    comptime tensor_strides: anytype,
    comptime tensor_offset: usize,
) type {
    return StaticViewImpl(T, tensor_shape, tensor_strides, tensor_offset, true);
}

fn StaticViewImpl(
    comptime T: type,
    comptime tensor_shape: anytype,
    comptime tensor_strides: anytype,
    comptime tensor_offset: usize,
    comptime read_only: bool,
) type {
    const tensor_rank = tensor_shape.len;
    if (tensor_strides.len != tensor_rank) {
        @compileError("static tensor shape and stride ranks must match");
    }
    const shape_value: [tensor_rank]usize = tensor_shape;
    const strides_value: [tensor_rank]isize = tensor_strides;
    const Storage = if (read_only) []const T else []T;
    const element_count = comptime elementCount(shape_value);
    const contiguous = comptime isContiguousLayout(shape_value, strides_value);
    const dense_positive = comptime isDensePositiveLayout(shape_value, strides_value);

    return struct {
        const Self = @This();
        pub const scalar_type = T;
        pub const dtype = Dtype.fromScalar(T);
        pub const rank = tensor_rank;
        pub const geometry_is_static = true;
        pub const base_offset = tensor_offset;
        pub const static_shape = shape_value;
        pub const static_strides = strides_value;
        pub const static_element_count = element_count;
        pub const static_is_contiguous = contiguous;
        pub const static_is_dense_positive = dense_positive;

        comptime shape: [rank]usize = shape_value,
        comptime strides: [rank]isize = strides_value,
        storage: Storage,
        /// Additional offset introduced by a runtime-selected subview.
        runtime_offset: usize = 0,

        pub fn len(_: *const Self) usize {
            return element_count;
        }

        pub fn elementOffset(self: *const Self, indices: [rank]usize) usize {
            var offset: isize = @intCast(base_offset + self.runtime_offset);
            inline for (strides_value, indices) |stride, index| {
                offset += @as(isize, @intCast(index)) * stride;
            }
            return @intCast(offset);
        }

        pub fn elementOffsetFromLinear(self: *const Self, linear_index: usize) usize {
            if (comptime rank == 0) {
                return base_offset + self.runtime_offset;
            }
            var remaining = linear_index;
            var offset: isize = @intCast(base_offset + self.runtime_offset);
            comptime var axis = rank;
            inline while (axis > 0) {
                axis -= 1;
                const extent = shape_value[axis];
                const index = remaining % extent;
                remaining /= extent;
                offset += @as(isize, @intCast(index)) * strides_value[axis];
            }
            return @intCast(offset);
        }

        pub fn get(self: *const Self, indices: [rank]usize) T {
            return self.storage[self.elementOffset(indices)];
        }

        pub fn set(self: *const Self, indices: [rank]usize, value: T) void {
            if (comptime read_only) {
                @compileError("cannot mutate a static const tensor view");
            }
            self.storage[self.elementOffset(indices)] = value;
        }

        pub fn isContiguous(_: *const Self) bool {
            return contiguous;
        }

        pub fn contiguousSlice(self: *const Self) ?Storage {
            if (comptime !contiguous) return null;
            const offset = base_offset + self.runtime_offset;
            return self.storage[offset..][0..element_count];
        }

        pub fn denseSlice(self: *const Self) ?Storage {
            if (comptime !dense_positive) return null;
            const offset = base_offset + self.runtime_offset;
            return self.storage[offset..][0..element_count];
        }

        pub fn axisSlice(
            self: *const Self,
            comptime axis: usize,
            slice_index: usize,
        ) StaticViewImpl(
            T,
            .{shape_value[axis]},
            .{strides_value[axis]},
            0,
            read_only,
        ) {
            if (comptime rank == 0 or axis >= rank) {
                @compileError("slice axis is outside the tensor rank");
            }

            var remaining = slice_index;
            var offset: isize = @intCast(base_offset + self.runtime_offset);
            comptime var current_axis = rank;
            inline while (current_axis > 0) {
                current_axis -= 1;
                if (current_axis == axis) continue;
                const extent = shape_value[current_axis];
                const index = remaining % extent;
                remaining /= extent;
                offset += @as(isize, @intCast(index)) * strides_value[current_axis];
            }

            return .{
                .storage = self.storage,
                .runtime_offset = @intCast(offset),
            };
        }

        pub fn broadcastTo(
            self: *const Self,
            comptime target_rank: usize,
            comptime target_shape: [target_rank]usize,
        ) StaticViewImpl(
            T,
            target_shape,
            broadcastStrides(shape_value, strides_value, target_shape),
            base_offset,
            read_only,
        ) {
            if (comptime rank > target_rank) {
                @compileError("broadcast target rank cannot be smaller than its source rank");
            }
            return .{
                .storage = self.storage,
                .runtime_offset = self.runtime_offset,
            };
        }
    };
}

fn elementCount(comptime shape: anytype) usize {
    var count: usize = 1;
    for (shape) |extent| count *= extent;
    return count;
}

fn isContiguousLayout(comptime shape: anytype, comptime strides: anytype) bool {
    var expected: isize = 1;
    var axis = shape.len;
    while (axis > 0) {
        axis -= 1;
        if (shape[axis] > 1 and strides[axis] != expected) return false;
        expected *= @intCast(shape[axis]);
    }
    return true;
}

fn isDensePositiveLayout(comptime shape: anytype, comptime strides: anytype) bool {
    var visited: [shape.len]bool = @splat(false);
    var expected_stride: isize = 1;
    var remaining_axes: usize = 0;
    for (shape) |extent| {
        if (extent > 1) remaining_axes += 1;
    }
    while (remaining_axes > 0) {
        var matched = false;
        for (shape, strides, 0..) |extent, stride, axis| {
            if (extent <= 1 or visited[axis] or stride != expected_stride) continue;
            visited[axis] = true;
            expected_stride *= @intCast(extent);
            remaining_axes -= 1;
            matched = true;
            break;
        }
        if (!matched) return false;
    }
    return true;
}

fn broadcastStrides(
    comptime source_shape: anytype,
    comptime source_strides: anytype,
    comptime target_shape: anytype,
) [target_shape.len]isize {
    if (source_shape.len > target_shape.len) {
        @compileError("broadcast target rank cannot be smaller than its source rank");
    }
    var result: [target_shape.len]isize = @splat(0);
    var source_axis = source_shape.len;
    var target_axis = target_shape.len;
    while (source_axis > 0) {
        source_axis -= 1;
        target_axis -= 1;
        if (source_shape[source_axis] == target_shape[target_axis]) {
            result[target_axis] = source_strides[source_axis];
        }
    }
    return result;
}

fn isDensePositive(view: anytype) bool {
    const rank = @TypeOf(view.*).rank;
    var visited: [rank]bool = @splat(false);
    var expected_stride: isize = 1;
    var remaining_axes: usize = 0;

    for (view.shape) |extent| {
        if (extent > 1) remaining_axes += 1;
    }

    while (remaining_axes > 0) {
        var matched = false;
        for (view.shape, view.strides, 0..) |extent, stride, axis| {
            if (extent <= 1 or visited[axis] or stride != expected_stride) continue;
            visited[axis] = true;
            expected_stride *= @intCast(extent);
            remaining_axes -= 1;
            matched = true;
            break;
        }
        if (!matched) return false;
    }

    return view.offset + view.len() <= view.storage.len;
}

fn axisSliceOffset(view: anytype, axis: usize, slice_index: usize) usize {
    const rank = @TypeOf(view.*).rank;
    comptime std.debug.assert(rank > 0);
    std.debug.assert(axis < rank);

    var slice_count: usize = 1;
    for (view.shape, 0..) |extent, current_axis| {
        if (current_axis != axis) slice_count *= extent;
    }
    std.debug.assert(slice_index < slice_count);

    var remaining = slice_index;
    var result: isize = @intCast(view.offset);
    var current_axis = rank;
    while (current_axis > 0) {
        current_axis -= 1;
        if (current_axis == axis) continue;
        const extent = view.shape[current_axis];
        const index = remaining % extent;
        remaining /= extent;
        result += @as(isize, @intCast(index)) * view.strides[current_axis];
    }

    std.debug.assert(result >= 0);
    const storage_index: usize = @intCast(result);
    std.debug.assert(storage_index <= view.storage.len);
    return storage_index;
}

pub fn Layout(comptime max_rank: usize) type {
    return struct {
        const This = @This();
        offset: usize,
        strides: [max_rank]isize,

        pub fn contiguous(shape: Shape(max_rank)) This {
            var result: This = .{
                .offset = 0,
                .strides = @splat(0),
            };

            var stride: isize = 1;
            var axis = shape.rank;

            while (axis > 0) {
                axis -= 1;
                result.strides[axis] = stride;
                stride *= @intCast(shape.at(axis));
            }

            return result;
        }

        /// Dense rank-2 storage with the first logical axis contiguous. This
        /// layout is useful when the leading axis represents independent work
        /// such as a batch: [M, N] uses strides [1, M].
        pub fn firstAxisContiguous(shape: Shape(max_rank)) This {
            if (shape.rank != 2) {
                @compileError("first-axis-contiguous layout requires rank 2");
            }
            var result: This = .{
                .offset = 0,
                .strides = @splat(0),
            };
            result.strides[0] = 1;
            result.strides[1] = @intCast(shape.at(0));
            return result;
        }
    };
}

/// Owned shape metadata with a graph-configurable maximum rank.
pub fn Shape(comptime max_rank: usize) type {
    return struct {
        pub const rank_capacity = max_rank;
        const Self = @This();

        rank: usize,
        dims: [max_rank]usize,

        pub fn init(extents: Shape_T) Self {
            if (extents.len > max_rank) {
                @panic("shape rank exceeds graph capacity");
            }

            var shape = Self{
                .rank = extents.len,
                .dims = @splat(0),
            };
            @memcpy(shape.dims[0..extents.len], extents);
            return shape;
        }

        pub fn elementCount(shape: *const Self) usize {
            var count: usize = 1;
            for (shape.slice()) |extent| count *= extent;
            return count;
        }

        pub fn slice(shape: *const Self) []const usize {
            return shape.dims[0..shape.rank];
        }

        pub fn at(shape: Self, axis: usize) usize {
            return shape.dims[axis];
        }
    };
}

// Represents an input source
pub const Source = struct {
    kind: Kind,
    tensor: Id,

    pub const Kind = enum { input, parameter, constant, state };

    pub const Binding = enum {
        embed,
    };
};

/// Originators (producers) of tensors.
pub const Origin = union(enum) {
    source: Id,
    node: Id,
    literal: ScalarValue,
};

/// Tensor metadata specialized for one graph's maximum rank.
pub fn Info(comptime max_rank: usize) type {
    return struct {
        dtype: Dtype,
        shape: Shape(max_rank),
        layout: Layout(max_rank),

        // root tensor whose allocation backs this value *multiple values may refer to one tensor*
        storage_tensor: Id,

        origin: Origin,
    };
}

pub const Ref = struct { id: Id };
