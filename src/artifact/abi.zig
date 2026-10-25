//! Runtime ABI for pre-compiled graphs
const std = @import("std");

pub const version: u32 = 2;

/// Error codes returned by ABI functions. These are stable across versions.
pub const Status = enum(c_int) {
    ok = 0,
    invalid_argument = 1,
    invalid_runtime_storage = 2,
    invalid_source = 3,
    invalid_output = 4,
    size_mismatch = 5,
    alignment_mismatch = 6,
    incompatible_binding = 7,
    missing_binding = 8,
    buffer_too_small = 9,
    out_of_memory = 10,
};

/// Backing type for a tensor. Casting is explicit.
pub const Dtype = enum(u32) {
    f32 = 1,
    f16 = 2,
    i8 = 3,
    boolean = 4,
    i32 = 5,
    u8 = 6,
};

/// Roles of a source tensor.
pub const SourceKind = enum(u32) {
    input = 1,
    parameter = 2,
    constant = 3,
    state = 4,
};

/// Defines how a source is provided to the runtime.
pub const SourceBinding = enum(u32) {
    /// The runtime owns the source and copies values into its own storage.
    owned = 1,
    /// The runtime borrows a caller-owned source view.
    bound = 2,
    /// The source is provided at compile time and embedded in the executable. Remains constant at runtime.
    embedded = 3,
};

/// Defines how a group of outputs will be accepted by a caller.
pub const OutputBinding = enum(u32) {
   /// The runtime allocates and owns the output storage. Caller must copy values out. 
    owned = 1, 
    /// The runtime borrows a caller-owned output view. 
    /// Caller must provide storage before run().
    /// Storage may be rebound between runs.
    bound = 2 
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

/// One named source in the compiled runtime contract.
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

pub fn adapter(executable: anytype) Adapter(@TypeOf(executable)) {
    return .{};
}

/// C-compatible operations specialized around one compiled executable.
pub fn Adapter(comptime Executable: type) type {
    return struct {
        const Self = @This();
        const Runtime = Executable.Runtime;
        const SourceKey = Executable.SourceKeyType;
        const graph = Executable.program;

        pub fn abiVersion() callconv(.c) u32 {
            return version;
        }
        /// Return a stable fingerprint of the executable's structure. 
        /// This is not a hash of the program's bytes, 
        /// but rather a fingerprint of the graph's topology, tensor shapes, and metadata.
        pub fn executableStructuralFingerprint() callconv(.c) u64 {
            return Executable.structural_fingerprint;
        }

        /// Return the size of the runtime storage required to run the executable.
        pub fn runtimeSize() callconv(.c) usize {
            return @sizeOf(Runtime);
        }

        /// Return the alignment of the runtime storage required to run the executable.
        pub fn runtimeAlignment() callconv(.c) usize {
            return @alignOf(Runtime);
        }

        /// Return the number of bytes of mutable storage used by the runtime.
        pub fn runtimeMutableBytes() callconv(.c) usize {
            return Executable.memory_plan.byte_count;
        }

        /// Render a compile-time executable section into caller-owned storage.
        /// A null destination with zero capacity queries the required size.
        pub fn executableInspect(section: u32, destination: ?[*]u8, capacity: usize, required: ?*usize) callconv(.c) Status {
            const count = required orelse return .invalid_argument;
            if (destination == null and capacity != 0) return .invalid_argument;
            const command: []const u8 = switch (section) {
                0 => "all",
                1 => "summary",
                2 => "raw-graph",
                3 => "semantic-graph",
                4 => "executable",
                5 => "representations",
                6 => "tree",
                7 => "memory-plan",
                else => return .invalid_argument,
            };
            var output = std.Io.Writer.Allocating.init(std.heap.page_allocator);
            defer output.deinit();
            _ = @import("zgc_inspect_impl").runCli(Executable, &.{command}, &output.writer) catch return .out_of_memory;
            const bytes = output.written();
            count.* = bytes.len;
            const buffer = destination orelse return .ok;
            if (capacity < bytes.len) return .buffer_too_small;
            @memcpy(buffer[0..bytes.len], bytes);
            return .ok;
        }

        pub fn runtimeInit(storage: ?*anyopaque, storage_size: usize) callconv(.c) Status {
            const runtime = mutableRuntime(storage, storage_size) orelse return storageStatus(storage, storage_size);
            runtime.* = Runtime.init();
            return .ok;
        }

        pub fn runtimeDeinit(storage: ?*anyopaque) callconv(.c) void {
            _ = storage;
        }

        pub fn runtimeRun(storage: ?*anyopaque) callconv(.c) Status {
            const runtime = mutableRuntime(storage, @sizeOf(Runtime)) orelse return .invalid_runtime_storage;
            inline for (0..graph.sources.len) |source_id| {
                if (comptime graph.sources[source_id] == null) continue;
                if (comptime Executable.source_plan.source_bindings[source_id] != .bound) continue;
                const key: SourceKey = @fromBackingInt(@intCast(source_id));
                if (!runtime.inputIsBound(key)) return .missing_binding;
            }
            if (comptime Executable.output_plan.output_binding == .bound) {
                inline for (0..graph.output_ct) |output_index| {
                    if (!runtime.outputIsBound(output_index)) return .missing_binding;
                }
                if (runtime.outputBindingsOverlap()) return .invalid_argument;
            }
            runtime.run();
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
                        .binding = comptime sourceBinding(Executable.source_plan.source_bindings[source_id]),
                        .tensor = tensorDescriptor(info),
                    };
                    return .ok;
                }
                ordinal += 1;
            }
            return .invalid_source;
        }

        /// Copy logical row-major bytes into an owned source. The generated
        /// runtime packs values into its selected physical layout.
        pub fn runtimeCopySource(
            storage: ?*anyopaque,
            source_key: u32,
            data: ?*const anyopaque,
            byte_count: usize,
        ) callconv(.c) Status {
            const runtime = mutableRuntime(storage, @sizeOf(Runtime)) orelse return .invalid_runtime_storage;
            const source_data = data orelse return .invalid_argument;

            inline for (0..graph.sources.len) |source_id| {
                const source = comptime graph.sources[source_id] orelse continue;
                if (source_key == source_id) {
                    if (comptime Executable.source_plan.source_bindings[source_id] != .owned) {
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
                    const key: SourceKey = @fromBackingInt(@intCast(source_id));
                    runtime.copySource(key, values) catch return .size_mismatch;
                    return .ok;
                }
            }
            return .invalid_source;
        }

        /// Bind physical source storage without copying. The supplied bytes
        /// must follow the layout returned by `sourceDescriptor`.
        pub fn runtimeBindSource(
            storage: ?*anyopaque,
            source_key: u32,
            data: ?*const anyopaque,
            byte_count: usize,
        ) callconv(.c) Status {
            const runtime = mutableRuntime(storage, @sizeOf(Runtime)) orelse return .invalid_runtime_storage;
            const source_data = data orelse return .invalid_argument;

            inline for (0..graph.sources.len) |source_id| {
                const source = comptime graph.sources[source_id] orelse continue;
                if (source_key == source_id) {
                    if (comptime Executable.source_plan.source_bindings[source_id] != .bound) {
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
                    const key: SourceKey = @fromBackingInt(@intCast(source_id));
                    runtime.bindInput(key, values) catch return .size_mismatch;
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

        pub fn outputBinding(output_index: usize, out: ?*OutputBinding) callconv(.c) Status {
            const destination = out orelse return .invalid_argument;
            if (output_index >= graph.output_ct) return .invalid_output;
            destination.* = switch (Executable.output_plan.output_binding) {
                .owned => .owned,
                .bound => .bound,
            };
            return .ok;
        }

        /// Bind mutable physical output storage for direct writes by run().
        pub fn runtimeBindOutput(
            storage: ?*anyopaque,
            output_index: usize,
            data: ?*anyopaque,
            byte_count: usize,
        ) callconv(.c) Status {
            const runtime = mutableRuntime(storage, @sizeOf(Runtime)) orelse return .invalid_runtime_storage;
            const output_data = data orelse return .invalid_argument;
            if (output_index >= graph.output_ct) return .invalid_output;
            if (comptime Executable.output_plan.output_binding != .bound) return .incompatible_binding;
            inline for (0..graph.output_ct) |index| {
                if (output_index == index) {
                    const info = comptime graph.tensors[graph.outputs[index].?].?;
                    if (byte_count != tensorLogicalBytes(info)) return .size_mismatch;
                    if (@intFromPtr(output_data) % info.dtype.alignment() != 0) return .alignment_mismatch;
                    const T = info.dtype.Scalar();
                    const raw: [*]u8 = @ptrCast(output_data);
                    const bytes = raw[0..byte_count];
                    const aligned: []align(@alignOf(T)) u8 = @alignCast(bytes);
                    const values = std.mem.bytesAsSlice(T, aligned);
                    runtime.bindOutput(index, values) catch return .size_mismatch;
                    return .ok;
                }
            }
            unreachable;
        }

        pub fn runtimeOutputView(
            storage: ?*const anyopaque,
            output_index: usize,
            out: ?*TensorView,
        ) callconv(.c) Status {
            const runtime = constRuntime(storage) orelse return .invalid_runtime_storage;
            const destination = out orelse return .invalid_argument;
            inline for (0..graph.output_ct) |index| {
                if (output_index == index) {
                    if (comptime Executable.output_plan.output_binding == .bound) {
                        if (!runtime.outputIsBound(index)) return .missing_binding;
                    }
                    const tensor_id = comptime graph.outputs[index].?;
                    const info = comptime graph.tensors[tensor_id].?;
                    const view = runtime.outputView(index);
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
        pub fn runtimeCopyOutput(
            storage: ?*const anyopaque,
            output_index: usize,
            destination: ?*anyopaque,
            destination_size: usize,
        ) callconv(.c) Status {
            const runtime = constRuntime(storage) orelse return .invalid_runtime_storage;
            const destination_ptr = destination orelse return .invalid_argument;
            inline for (0..graph.output_ct) |index| {
                if (output_index == index) {
                    if (comptime Executable.output_plan.output_binding == .bound) {
                        if (!runtime.outputIsBound(index)) return .missing_binding;
                    }
                    const tensor_id = comptime graph.outputs[index].?;
                    const info = comptime graph.tensors[tensor_id].?;
                    const required = comptime tensorLogicalBytes(info);
                    const element_count = comptime info.shape.elementCount();
                    if (destination_size < required) return .buffer_too_small;
                    const output = runtime.outputView(index);
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

        fn mutableRuntime(storage: ?*anyopaque, storage_size: usize) ?*Runtime {
            const pointer = storage orelse return null;
            if (storage_size < @sizeOf(Runtime)) return null;
            if (@intFromPtr(pointer) % @alignOf(Runtime) != 0) return null;
            return @ptrCast(@alignCast(pointer));
        }

        fn constRuntime(storage: ?*const anyopaque) ?*const Runtime {
            const pointer = storage orelse return null;
            if (@intFromPtr(pointer) % @alignOf(Runtime) != 0) return null;
            return @ptrCast(@alignCast(pointer));
        }

        fn storageStatus(storage: ?*anyopaque, storage_size: usize) Status {
            const pointer = storage orelse return .invalid_argument;
            if (storage_size < @sizeOf(Runtime)) return .size_mismatch;
            if (@intFromPtr(pointer) % @alignOf(Runtime) != 0) return .alignment_mismatch;
            return .invalid_runtime_storage;
        }

        fn sourceCountValue() usize {
            var count: usize = 0;
            for (graph.sources) |source| count += @intFromBool(source != null);
            return count;
        }

        fn sourceName(comptime source_id: usize) [*:0]const u8 {
            return Executable.source_names[source_id].ptr;
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
                .i32 => .i32,
                .u8 => .u8,
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
