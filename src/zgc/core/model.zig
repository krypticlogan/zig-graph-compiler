const std = @import("std");
const Graph = @import("graph.zig");
const Storage = @import("../storage/storage.zig");
const Tensor = @import("tensor.zig");
const ScalarValue = @import("../storage/dtype.zig").ScalarValue;

fn EmbeddedStorage(comptime bytes: []const u8, comptime alignment: usize) type {
    const contents = bytes;
    return struct {
        const storage: [contents.len]u8 align(alignment) = contents[0..].*;
    };
}

fn LiteralStorage(comptime value: ScalarValue) type {
    const T = value.dtype().Scalar();
    return struct {
        const storage: [1]T = .{value.get(value.dtype())};
    };
}

fn PackedEmbeddedStorage(
    comptime bytes: []const u8,
    comptime info: anytype,
    comptime alignment: usize,
) type {
    const element_bytes = info.dtype.byteSize();
    const element_count = info.shape.elementCount();
    const contents = comptime blk: {
        // Packing work scales with both the element count and rank because
        // each logical index is decomposed into coordinates. Large embedded
        // parameter tensors legitimately exceed Zig's default comptime quota.
        @setEvalBranchQuota(element_count * (info.shape.rank + 8) + 1000);
        var packed_bytes: [bytes.len]u8 = undefined;
        for (0..element_count) |logical_index| {
            var remaining = logical_index;
            var physical_index = info.layout.offset;
            var axis = info.shape.rank;
            while (axis > 0) {
                axis -= 1;
                const extent = info.shape.at(axis);
                const coordinate = remaining % extent;
                remaining /= extent;
                if (info.layout.strides[axis] < 0) {
                    @compileError("embedded source packing does not support negative strides");
                }
                physical_index += coordinate *
                    @as(usize, @intCast(info.layout.strides[axis]));
            }

            const source_start = logical_index * element_bytes;
            const destination_start = physical_index * element_bytes;
            @memcpy(
                packed_bytes[destination_start..][0..element_bytes],
                bytes[source_start..][0..element_bytes],
            );
        }
        break :blk packed_bytes;
    };

    return struct {
        const storage: [contents.len]u8 align(alignment) = contents;
    };
}

fn hasLogicalRowMajorLayout(comptime info: anytype) bool {
    const expected = @TypeOf(info.layout).contiguous(info.shape);
    if (info.layout.offset != expected.offset) return false;
    for (0..info.shape.rank) |axis| {
        if (info.layout.strides[axis] != expected.strides[axis]) return false;
    }
    return true;
}

pub fn Model(
    comptime SourceKey: type,
    comptime capacities: Graph.Capacity,
    comptime raw: anytype,
    comptime semantic_optimization: anytype,
    comptime SemanticValidated: type,
    comptime semantic_analysis: anytype,
    comptime executable_search: anytype,
    comptime Validated: type,
    comptime lifetimes: anytype,
    comptime SourcePlan: type,
) type {
    const graph = Validated.graph;
    const plan = Storage.MemoryPlan(capacities, graph, lifetimes, SourcePlan);
    return struct {
        const Self = @This();
        pub const SourceKeyType = SourceKey;
        pub const SourceError = error{SourceSizeMismatch};
        pub const executable = graph;
        pub const memory_plan = plan;
        pub const internal_capacity = capacities;
        pub const raw_graph = raw;
        pub const semantic_graph = SemanticValidated.graph;
        pub const semantic_optimization_result = semantic_optimization;
        pub const semantic_provenance = semantic_optimization.provenance;
        pub const semantic_analysis_result = semantic_analysis;
        pub const reference_executable_candidate = executable_search.reference;
        pub const executable_candidate_frontier = executable_search.frontier;
        pub const executable_candidate_count = executable_search.generated_count;
        pub const fusion_candidate_count = executable_search.fusion_candidate_count;
        pub const layout_candidate_count = executable_search.layout_candidate_count;
        pub const remap_candidate_count = executable_search.remap_candidate_count;
        pub const selected_executable_candidate = executable_search.selected();
        pub const lifetime_analysis = lifetimes;
        pub const source_plan = SourcePlan;

        memory: [plan.byte_count]u8 align(plan.alignment) = undefined,
        bound_sources: [capacities.max_sources]?[]const u8 = @splat(null),

        pub fn init() Self {
            return .{
                .memory = @splat(0),
                .bound_sources = @splat(null),
            };
        }

        pub fn run(model: *Self) void {
            @setEvalBranchQuota(1_000 + graph.node_ct * 64);
            inline for (0..graph.node_ct) |node_id| {
                model.executeNode(node_id);
            }
        }

        pub fn outputView(model: *const Self, comptime output_index: usize) blk: {
            if (output_index >= graph.output_ct) {
                @compileError("output index is outside the graph's outputs");
            }
            const tensor_id = graph.outputs[output_index].?;
            break :blk Validated.ConstView(tensor_id);
        } {
            const tensor_id = graph.outputs[output_index].?;
            return model.constTensorView(tensor_id);
        }

        /// Copy logical row-major values into a model-owned source. Lowering
        /// may select another physical layout; this method performs the
        /// corresponding one-time packing.
        pub fn copySource(
            model: *Self,
            comptime source_key: SourceKey,
            values: []const sourceScalar(source_key),
        ) SourceError!void {
            const source_id = comptime sourceIndex(source_key);
            if (comptime SourcePlan.source_bindings[source_id] != .owned) {
                @compileError("copySource requires model-owned source storage");
            }
            const source = comptime sourceFor(source_key);
            const bytes = std.mem.sliceAsBytes(values);
            const expected_bytes = comptime sourceByteCount(source_key);
            if (bytes.len != expected_bytes) return error.SourceSizeMismatch;
            const destination = model.tensorView(source.tensor);
            for (values, 0..) |value, logical_index| {
                destination.storage[destination.elementOffsetFromLinear(logical_index)] = value;
            }
        }

        /// Copy one runtime input into its model-owned region.
        pub fn copyInput(
            model: *Self,
            comptime source_key: SourceKey,
            values: []const sourceScalar(source_key),
        ) SourceError!void {
            const source = comptime sourceFor(source_key);
            comptime requireInput(source);
            try model.copySource(source_key, values);
        }

        /// Borrow runtime input storage without copying it. Values must use
        /// the physical order returned by sourceLayout(). The caller must keep
        /// the slice alive and unchanged for the duration of run().
        pub fn bindInput(
            model: *Self,
            comptime source_key: SourceKey,
            values: []const sourceScalar(source_key),
        ) SourceError!void {
            const source = comptime sourceFor(source_key);
            comptime requireInput(source);
            const source_id = comptime sourceIndex(source_key);
            if (comptime SourcePlan.source_bindings[source_id] != .bound) {
                @compileError("bindInput requires the source to be configured as zgc.memory.Source.bound");
            }
            const bytes = std.mem.sliceAsBytes(values);
            const expected_bytes = comptime sourceByteCount(source_key);
            if (bytes.len != expected_bytes) return error.SourceSizeMismatch;
            model.bound_sources[source_id] = bytes;
        }

        pub fn inputIsBound(model: *const Self, comptime source_key: SourceKey) bool {
            const source = comptime sourceFor(source_key);
            comptime requireInput(source);
            const source_id = comptime sourceIndex(source_key);
            if (comptime SourcePlan.source_bindings[source_id] != .bound) {
                @compileError("inputIsBound requires a runtime-bound input source");
            }
            return model.bound_sources[source_id] != null;
        }

        pub fn sourceLayout(
            comptime source_key: SourceKey,
        ) Tensor.Layout(capacities.max_rank) {
            return sourceInfo(source_key).layout;
        }

        fn executeNode(model: *Self, comptime node_id: usize) void {
            const node = graph.nodes[node_id].?;
            const compute = switch (node.op) {
                .compute => |op| op,
                .view => return,
            };
            const InputViews = comptime blk: {
                var input_types: [node.input_count]type = undefined;
                for (0..node.input_count) |input_index| {
                    const tensor_id = graph.input_refs[node.input_start + input_index].?;
                    input_types[input_index] = Validated.ConstView(tensor_id);
                }
                break :blk std.meta.Tuple(&input_types);
            };

            var inputs: InputViews = undefined;
            inline for (0..node.input_count) |input_index| {
                const tensor_id = comptime graph.input_refs[node.input_start + input_index].?;
                inputs[input_index] = model.constTensorView(tensor_id);
            }

            const OutputViews = comptime blk: {
                var output_types: [node.output_count]type = undefined;
                for (0..node.output_count) |output_index| {
                    const tensor_id = graph.output_refs[node.output_start + output_index].?;
                    output_types[output_index] = Validated.View(tensor_id);
                }
                break :blk std.meta.Tuple(&output_types);
            };
            var outputs: OutputViews = undefined;
            inline for (0..node.output_count) |output_index| {
                const tensor_id = comptime graph.output_refs[node.output_start + output_index].?;
                outputs[output_index] = model.tensorView(tensor_id);
            }

            compute.execute(inputs, outputs);
        }

        fn tensorView(model: *Self, comptime tensor_id: Tensor.Id) blk: {
            break :blk Validated.View(tensor_id);
        } {
            const info = graph.tensors[tensor_id].?;
            const T = info.dtype.Scalar();
            const bytes = model.tensorBytes(info.storage_tensor);
            const aligned_bytes: []align(@alignOf(T)) u8 = @alignCast(bytes);

            return .{
                .storage = std.mem.bytesAsSlice(T, aligned_bytes),
            };
        }

        fn constTensorView(model: *const Self, comptime tensor_id: Tensor.Id) blk: {
            break :blk Validated.ConstView(tensor_id);
        } {
            const info = graph.tensors[tensor_id].?;
            const T = info.dtype.Scalar();
            const aligned_bytes = model.constTensorBytes(info.storage_tensor, T);

            return .{
                .storage = std.mem.bytesAsSlice(T, aligned_bytes),
            };
        }

        fn tensorBytes(model: *Self, comptime tensor_id: Tensor.Id) []u8 {
            const region = plan.tensor_regions[tensor_id] orelse
                @compileError("tensor does not have model-owned mutable storage");
            return model.memory[region.offset .. region.offset + region.len_bytes];
        }

        fn constTensorBytes(
            model: *const Self,
            comptime tensor_id: Tensor.Id,
            comptime T: type,
        ) []align(@alignOf(T)) const u8 {
            const info = graph.tensors[tensor_id].?;
            return switch (comptime SourcePlan.bindingForTensor(info)) {
                .owned => blk: {
                    const region = plan.tensor_regions[tensor_id].?;
                    const bytes = model.memory[region.offset .. region.offset + region.len_bytes];
                    break :blk @alignCast(bytes);
                },
                .bound => blk: {
                    const source_id = switch (info.origin) {
                        .source => |id| id,
                        .node, .literal => unreachable,
                    };
                    const bytes = model.bound_sources[source_id] orelse
                        @panic("runtime input has not been bound");
                    break :blk @alignCast(bytes);
                },
                .embedded => |embedded| blk: {
                    const Embedded = switch (embedded.order) {
                        .logical => if (comptime hasLogicalRowMajorLayout(info))
                            EmbeddedStorage(embedded.bytes, @alignOf(T))
                        else
                            PackedEmbeddedStorage(
                                embedded.bytes,
                                info,
                                @alignOf(T),
                            ),
                        .physical => EmbeddedStorage(embedded.bytes, @alignOf(T)),
                    };
                    break :blk &Embedded.storage;
                },
                .literal => |value| blk: {
                    const Literal = LiteralStorage(value);
                    break :blk std.mem.asBytes(&Literal.storage);
                },
            };
        }

        fn sourceIndex(comptime source_key: SourceKey) usize {
            const source_id: usize = @intCast(@intFromEnum(source_key));
            if (source_id >= graph.sources.len or graph.sources[source_id] == null) {
                @compileError("the graph does not contain the provided source key");
            }
            return source_id;
        }

        fn sourceFor(comptime source_key: SourceKey) Tensor.Source {
            return graph.sources[sourceIndex(source_key)].?;
        }

        fn sourceInfo(comptime source_key: SourceKey) @TypeOf(graph).TensorInfo {
            return graph.tensors[sourceFor(source_key).tensor].?;
        }

        fn sourceScalar(comptime source_key: SourceKey) type {
            return sourceInfo(source_key).dtype.Scalar();
        }

        fn sourceByteCount(comptime source_key: SourceKey) usize {
            const info = sourceInfo(source_key);
            return info.shape.elementCount() * info.dtype.byteSize();
        }

        fn requireInput(comptime source: Tensor.Source) void {
            if (source.kind != .input) @compileError("source is not a runtime input");
        }
    };
}
