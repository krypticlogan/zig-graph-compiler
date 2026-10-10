const std = @import("std");

pub const version: u32 = 1;

pub const Status = enum(c_int) {
    ok = 0,
    invalid_argument = 1,
    invalid_model_storage = 2,
    invalid_source = 3,
    invalid_output = 4,
    size_mismatch = 5,
    alignment_mismatch = 6,
    incompatible_binding = 7,
    missing_binding = 8,
    buffer_too_small = 9,
};

pub const Dtype = enum(u32) {
    f32 = 1,
    f16 = 2,
    i8 = 3,
    boolean = 4,
};

pub const SourceKind = enum(u32) {
    input = 1,
    parameter = 2,
    constant = 3,
    state = 4,
};

pub const SourceBinding = enum(u32) {
    owned = 1,
    bound = 2,
    embedded = 3,
};

/// Static logical geometry and physical layout for one tensor.
pub const TensorDescriptor = extern struct {
    dtype: Dtype,
    rank: u32,
    element_count: usize,
    logical_byte_count: usize,
    offset_elements: usize,
    shape: ?[*]const usize,
    strides: ?[*]const isize,
};

/// One named source in the compiled model contract.
pub const SourceDescriptor = extern struct {
    key: u32,
    name: [*:0]const u8,
    kind: SourceKind,
    binding: SourceBinding,
    tensor: TensorDescriptor,
};

/// A borrowed physical output view. `data` points at the backing storage;
/// `tensor.offset_elements` and `tensor.strides` describe logical indexing.
pub const TensorView = extern struct {
    data: ?*const anyopaque,
    storage_byte_count: usize,
    tensor: TensorDescriptor,
};

/// C-compatible operations specialized around one generated model type.
pub fn Adapter(comptime Model: type) type {
    return struct {
        const Self = @This();
        const SourceKey = Model.SourceKeyType;
        const graph = Model.executable;

        pub fn abiVersion() callconv(.c) u32 {
            return version;
        }

        pub fn modelSize() callconv(.c) usize {
            return @sizeOf(Model);
        }

        pub fn modelAlignment() callconv(.c) usize {
            return @alignOf(Model);
        }

        pub fn modelMutableBytes() callconv(.c) usize {
            return Model.memory_plan.byte_count;
        }

        pub fn modelInit(storage: ?*anyopaque, storage_size: usize) callconv(.c) Status {
            const model = mutableModel(storage, storage_size) orelse return storageStatus(storage, storage_size);
            model.* = Model.init();
            return .ok;
        }

        pub fn modelDeinit(storage: ?*anyopaque) callconv(.c) void {
            _ = storage;
        }

        pub fn modelRun(storage: ?*anyopaque) callconv(.c) Status {
            const model = mutableModel(storage, @sizeOf(Model)) orelse return .invalid_model_storage;
            inline for (0..graph.sources.len) |source_id| {
                if (comptime graph.sources[source_id] == null) continue;
                if (comptime Model.source_plan.source_bindings[source_id] != .bound) continue;
                const key: SourceKey = @enumFromInt(source_id);
                if (!model.inputIsBound(key)) return .missing_binding;
            }
            model.run();
            return .ok;
        }

        pub fn sourceCount() callconv(.c) usize {
            return comptime sourceCountValue();
        }

        pub fn sourceDescriptor(source_index: usize, out: ?*SourceDescriptor) callconv(.c) Status {
            const destination = out orelse return .invalid_argument;
            var ordinal: usize = 0;
            inline for (0..graph.sources.len) |source_id| {
                const source = comptime graph.sources[source_id] orelse continue;
                if (source_index == ordinal) {
                    const info = comptime graph.tensors[source.tensor].?;
                    destination.* = .{
                        .key = @intCast(source_id),
                        .name = comptime sourceName(source_id),
                        .kind = sourceKind(source.kind),
                        .binding = comptime sourceBinding(Model.source_plan.source_bindings[source_id]),
                        .tensor = tensorDescriptor(info),
                    };
                    return .ok;
                }
                ordinal += 1;
            }
            return .invalid_source;
        }

        /// Copy logical row-major bytes into an owned source. The generated
        /// model packs values into its selected physical layout.
        pub fn modelCopySource(
            storage: ?*anyopaque,
            source_key: u32,
            data: ?*const anyopaque,
            byte_count: usize,
        ) callconv(.c) Status {
            const model = mutableModel(storage, @sizeOf(Model)) orelse return .invalid_model_storage;
            const source_data = data orelse return .invalid_argument;

            inline for (0..graph.sources.len) |source_id| {
                const source = comptime graph.sources[source_id] orelse continue;
                if (source_key == source_id) {
                    if (comptime Model.source_plan.source_bindings[source_id] != .owned) {
                        return .incompatible_binding;
                    }
                    const info = comptime graph.tensors[source.tensor].?;
                    if (byte_count != tensorLogicalBytes(info)) return .size_mismatch;
                    if (@intFromPtr(source_data) % info.dtype.alignment() != 0) return .alignment_mismatch;
                    const T = info.dtype.Scalar();
                    const raw: [*]const u8 = @ptrCast(source_data);
                    const bytes = raw[0..byte_count];
                    const aligned: []align(@alignOf(T)) const u8 = @alignCast(bytes);
                    const values = std.mem.bytesAsSlice(T, aligned);
                    const key: SourceKey = @enumFromInt(source_id);
                    model.copySource(key, values) catch return .size_mismatch;
                    return .ok;
                }
            }
            return .invalid_source;
        }

        /// Bind physical source storage without copying. The supplied bytes
        /// must follow the layout returned by `sourceDescriptor`.
        pub fn modelBindSource(
            storage: ?*anyopaque,
            source_key: u32,
            data: ?*const anyopaque,
            byte_count: usize,
        ) callconv(.c) Status {
            const model = mutableModel(storage, @sizeOf(Model)) orelse return .invalid_model_storage;
            const source_data = data orelse return .invalid_argument;

            inline for (0..graph.sources.len) |source_id| {
                const source = comptime graph.sources[source_id] orelse continue;
                if (source_key == source_id) {
                    if (comptime Model.source_plan.source_bindings[source_id] != .bound) {
                        return .incompatible_binding;
                    }
                    const info = comptime graph.tensors[source.tensor].?;
                    if (byte_count != tensorLogicalBytes(info)) return .size_mismatch;
                    if (@intFromPtr(source_data) % info.dtype.alignment() != 0) return .alignment_mismatch;
                    const T = info.dtype.Scalar();
                    const raw: [*]const u8 = @ptrCast(source_data);
                    const bytes = raw[0..byte_count];
                    const aligned: []align(@alignOf(T)) const u8 = @alignCast(bytes);
                    const values = std.mem.bytesAsSlice(T, aligned);
                    const key: SourceKey = @enumFromInt(source_id);
                    model.bindInput(key, values) catch return .size_mismatch;
                    return .ok;
                }
            }
            return .invalid_source;
        }

        pub fn outputCount() callconv(.c) usize {
            return graph.output_ct;
        }

        pub fn outputDescriptor(output_index: usize, out: ?*TensorDescriptor) callconv(.c) Status {
            const destination = out orelse return .invalid_argument;
            inline for (0..graph.output_ct) |index| {
                if (output_index == index) {
                    const tensor_id = comptime graph.outputs[index].?;
                    destination.* = tensorDescriptor(graph.tensors[tensor_id].?);
                    return .ok;
                }
            }
            return .invalid_output;
        }

        pub fn modelOutputView(
            storage: ?*const anyopaque,
            output_index: usize,
            out: ?*TensorView,
        ) callconv(.c) Status {
            const model = constModel(storage) orelse return .invalid_model_storage;
            const destination = out orelse return .invalid_argument;
            inline for (0..graph.output_ct) |index| {
                if (output_index == index) {
                    const tensor_id = comptime graph.outputs[index].?;
                    const info = comptime graph.tensors[tensor_id].?;
                    const view = model.outputView(index);
                    destination.* = .{
                        .data = @ptrCast(view.storage.ptr),
                        .storage_byte_count = view.storage.len * info.dtype.byteSize(),
                        .tensor = tensorDescriptor(info),
                    };
                    return .ok;
                }
            }
            return .invalid_output;
        }

        /// Copy an output into logical row-major storage.
        pub fn modelCopyOutput(
            storage: ?*const anyopaque,
            output_index: usize,
            destination: ?*anyopaque,
            destination_size: usize,
        ) callconv(.c) Status {
            const model = constModel(storage) orelse return .invalid_model_storage;
            const destination_ptr = destination orelse return .invalid_argument;
            inline for (0..graph.output_ct) |index| {
                if (output_index == index) {
                    const tensor_id = comptime graph.outputs[index].?;
                    const info = comptime graph.tensors[tensor_id].?;
                    const required = comptime tensorLogicalBytes(info);
                    const element_count = comptime info.shape.elementCount();
                    if (destination_size < required) return .buffer_too_small;
                    const output = model.outputView(index);
                    const bytes: [*]u8 = @ptrCast(destination_ptr);
                    for (0..element_count) |logical_index| {
                        const value = output.storage[output.elementOffsetFromLinear(logical_index)];
                        const start = logical_index * info.dtype.byteSize();
                        @memcpy(bytes[start..][0..info.dtype.byteSize()], std.mem.asBytes(&value));
                    }
                    return .ok;
                }
            }
            return .invalid_output;
        }

        fn mutableModel(storage: ?*anyopaque, storage_size: usize) ?*Model {
            const pointer = storage orelse return null;
            if (storage_size < @sizeOf(Model)) return null;
            if (@intFromPtr(pointer) % @alignOf(Model) != 0) return null;
            return @ptrCast(@alignCast(pointer));
        }

        fn constModel(storage: ?*const anyopaque) ?*const Model {
            const pointer = storage orelse return null;
            if (@intFromPtr(pointer) % @alignOf(Model) != 0) return null;
            return @ptrCast(@alignCast(pointer));
        }

        fn storageStatus(storage: ?*anyopaque, storage_size: usize) Status {
            const pointer = storage orelse return .invalid_argument;
            if (storage_size < @sizeOf(Model)) return .size_mismatch;
            if (@intFromPtr(pointer) % @alignOf(Model) != 0) return .alignment_mismatch;
            return .invalid_model_storage;
        }

        fn sourceCountValue() usize {
            var count: usize = 0;
            for (graph.sources) |source| count += @intFromBool(source != null);
            return count;
        }

        fn sourceName(comptime source_id: usize) [*:0]const u8 {
            return Model.source_names[source_id].ptr;
        }

        fn tensorDescriptor(comptime info: @TypeOf(graph).TensorInfo) TensorDescriptor {
            return .{
                .dtype = dtype(info.dtype),
                .rank = @intCast(info.shape.rank),
                .element_count = info.shape.elementCount(),
                .logical_byte_count = tensorLogicalBytes(info),
                .offset_elements = info.layout.offset,
                .shape = if (info.shape.rank == 0) null else info.shape.slice().ptr,
                .strides = if (info.shape.rank == 0) null else info.layout.strides[0..info.shape.rank].ptr,
            };
        }

        fn tensorLogicalBytes(comptime info: @TypeOf(graph).TensorInfo) usize {
            return info.shape.elementCount() * info.dtype.byteSize();
        }

        fn dtype(comptime value: anytype) Dtype {
            return switch (value) {
                .f32 => .f32,
                .f16 => .f16,
                .i8 => .i8,
                .bool => .boolean,
            };
        }

        fn sourceKind(comptime value: anytype) SourceKind {
            return switch (value) {
                .input => .input,
                .parameter => .parameter,
                .constant => .constant,
                .state => .state,
            };
        }

        fn sourceBinding(comptime value: anytype) SourceBinding {
            return switch (value) {
                .owned => .owned,
                .bound => .bound,
                .embedded => .embedded,
                .literal => unreachable,
            };
        }
    };
}
