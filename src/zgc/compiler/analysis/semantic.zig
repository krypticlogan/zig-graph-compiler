const Elementwise = @import("../../operations/elementwise.zig");
const Reduction = @import("../../operations/reduction.zig");
const Graph = @import("../../core/graph.zig");
const Traversal = @import("../traversal.zig");

pub fn SemanticAnalysis(comptime capacity: Graph.Capacity) type {
    return struct {
        pub fn analyze(comptime Validated: type) Facts(capacity) {
            const graph = Validated.graph;
            var result: Facts(capacity) = .{};
            result.dependencies = DependencyGraph(capacity).build(graph);
            for (0..graph.tensor_ct) |id| {
                result.use_counts[id] = result.dependencies.consumer_offsets[id + 1] -
                    result.dependencies.consumer_offsets[id];
                const tensor = graph.tensors[id].?;
                result.tensor_bytes[id] = @as(u128, tensor.shape.elementCount()) * tensor.dtype.byteSize();
                result.storage_root[id] = tensor.storage_tensor;
                result.fusion_anchor[id] = id;
            }
            for (graph.outputs[0..graph.output_ct]) |output| result.is_output[output.?] = true;
            for (result.dependencies.topological_order[0..graph.node_ct]) |node_id| {
                const node = graph.nodes[node_id].?;
                if (node.op != .compute) continue;
                result.pointwise[node_id] = Elementwise.fromCompute(node.op.compute);
                result.reductions[node_id] = Reduction.fromCompute(node.op.compute);
                const operation = result.pointwise[node_id] orelse continue;
                if (operation.isReductionCompatible() and node.input_count > 0 and
                    result.use_counts[node.result] == 1 and !result.is_output[node.result])
                {
                    const input = graph.input_refs[node.input_start].?;
                    result.fusion_anchor[node.result] = result.fusion_anchor[input];
                }
            }
            return result;
        }
    };
}

pub const Consumer = Traversal.Consumer;
pub const DependencyGraph = Traversal.Dependencies;

pub fn Facts(comptime capacity: Graph.Capacity) type {
    return struct {
        // Valid only for this immutable semantic graph. Physical uses,
        // layouts and schedule-dependent lifetimes must be analyzed separately.
        tensor_bytes: [capacity.max_tensors]u128 = @splat(0),
        storage_root: [capacity.max_tensors]usize = @splat(0),
        fusion_anchor: [capacity.max_tensors]usize = @splat(0),
        pointwise: [capacity.max_nodes]?Elementwise.Operation = @splat(null),
        reductions: [capacity.max_nodes]?Reduction.Descriptor = @splat(null),
        use_counts: [capacity.max_tensors]usize = @splat(0),
        is_output: [capacity.max_tensors]bool = @splat(false),
        dependencies: DependencyGraph(capacity) = .{},
    };
}
