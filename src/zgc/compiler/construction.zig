const Graph = @import("../core/graph.zig");
const Op = @import("../operations/semantic.zig").Op;
const Tensor = @import("../core/tensor.zig");
const layout_ops = @import("../kernels/layout.zig");

pub fn count(comptime Definition: type, comptime definition: Definition) Graph.Capacity {
    var capacity: Graph.Capacity = .{
        .max_nodes = definition.node_count,
        .max_input_refs = definition.input_ref_count,
        .max_tensors = definition.tensor_count,
        .max_outputs = definition.output_count,
    };

    for (definition.tensors[0..definition.tensor_count]) |record| {
        capacity.max_rank = @max(capacity.max_rank, record.value.shape.rank);
        switch (record.origin) {
            .source => |source_index| capacity.max_sources = @max(capacity.max_sources, source_index + 1),
            .node, .literal => {},
        }
    }
    return capacity;
}

/// Lowers a completed definition into the concrete graph sized by the counting
/// step.
pub fn GraphConstruction(
    comptime Definition: type,
    comptime capacity: Graph.Capacity,
) type {
    return struct {
        pub fn build(comptime definition: Definition) Graph.Graph(capacity, Op) {
            var graph: Graph.Graph(capacity, Op) = .init();
            const TensorInfo = Tensor.Info(capacity.max_rank);

            inline for (0..definition.tensor_count) |tensor_id| {
                const record = definition.tensors[tensor_id];
                const shape: Tensor.Shape(capacity.max_rank) = .init(record.value.shape.slice());

                switch (record.origin) {
                    .source => |source_index| {
                        const info: TensorInfo = .{
                            .dtype = record.value.dtype,
                            .shape = shape,
                            .origin = .{ .source = source_index },
                            .layout = .contiguous(shape),
                            .storage_tensor = tensor_id,
                        };
                        graph.insertSource(source_index, .{
                            .kind = record.source_kind.?,
                            .tensor = tensor_id,
                        });
                        const inserted = graph.insertTensor(info);
                        if (inserted != tensor_id) @compileError("definition tensor order was not preserved");
                    },
                    .literal => |value| {
                        const info: TensorInfo = .{
                            .dtype = record.value.dtype,
                            .shape = shape,
                            .origin = .{ .literal = value },
                            .layout = .contiguous(shape),
                            .storage_tensor = tensor_id,
                        };
                        const inserted = graph.insertTensor(info);
                        if (inserted != tensor_id) @compileError("definition tensor order was not preserved");
                    },
                    .node => |node_id| {
                        const node = definition.nodes[node_id];
                        const InputInfos = [node.input_count]TensorInfo;
                        var input_infos: InputInfos = undefined;
                        inline for (0..node.input_count) |input_index| {
                            const input_id = definition.input_refs[node.input_start + input_index];
                            input_infos[input_index] = graph.tensors[input_id].?;
                        }

                        const info: TensorInfo = switch (node.op) {
                            .compute => .{
                                .dtype = record.value.dtype,
                                .shape = shape,
                                .origin = .{ .node = node_id },
                                .layout = .contiguous(shape),
                                .storage_tensor = tensor_id,
                            },
                            .view => |view| blk: {
                                const result = layout_ops.infer(view, &input_infos, shape, capacity.max_rank);
                                break :blk .{
                                    .dtype = record.value.dtype,
                                    .shape = result.shape,
                                    .origin = .{ .node = node_id },
                                    .layout = result.layout,
                                    .storage_tensor = result.storage_tensor,
                                };
                            },
                        };
                        const inserted = graph.insertTensor(info);
                        if (inserted != tensor_id) @compileError("definition tensor order was not preserved");

                        inline for (0..node.input_count) |input_index| {
                            graph.insertRef(definition.input_refs[node.input_start + input_index]);
                        }
                        graph.insertNode(.{
                            .op = node.op,
                            .input_start = graph.input_ref_ct - node.input_count,
                            .input_count = node.input_count,
                            .result = tensor_id,
                        });
                    },
                }
            }

            inline for (0..definition.output_count) |output_index| {
                graph.insertOutput(definition.outputs[output_index]);
            }
            return graph;
        }
    };
}
