const Graph = @import("../../core/graph.zig");
const Semantic = @import("../../operations/semantic.zig");

pub fn SemanticAnalysis(comptime capacity: Graph.Capacity) type {
    return struct {
        pub fn analyze(comptime Validated: type) Facts(capacity) {
            const graph = Validated.graph;
            var result: Facts(capacity) = .{};

            for (0..graph.node_ct) |node_id| {
                const node = graph.nodes[node_id].?;
                for (0..node.input_count) |input_index| {
                    const ref_index = node.input_start + input_index;
                    const tensor_id = graph.input_refs[ref_index].?;
                    result.use_counts[tensor_id] += 1;
                    result.dependencies.consumer_offsets[tensor_id + 1] += 1;
                }
            }
            for (0..graph.output_ct) |output_index| {
                result.is_output[graph.outputs[output_index].?] = true;
            }

            for (0..graph.tensor_ct) |tensor_id| {
                result.dependencies.consumer_offsets[tensor_id + 1] +=
                    result.dependencies.consumer_offsets[tensor_id];
            }
            var consumer_cursors = result.dependencies.consumer_offsets;
            for (0..graph.node_ct) |node_id| {
                const node = graph.nodes[node_id].?;
                for (0..node.input_count) |input_index| {
                    const tensor_id = graph.input_refs[node.input_start + input_index].?;
                    const edge_index = consumer_cursors[tensor_id];
                    result.dependencies.consumers[edge_index] = .{
                        .node = node_id,
                        .input_index = input_index,
                    };
                    consumer_cursors[tensor_id] += 1;
                }
            }
            result.dependencies.edge_count = graph.input_ref_ct;
            result.dependencies.buildTopologicalOrder(graph);

            return result;
        }
    };
}

pub const Consumer = struct {
    node: usize,
    input_index: usize,
};

/// Dependency edges and a canonical topological order for the semantic DAG.
/// Node identifiers remain stable graph identities; they do not define
/// dependency adjacency or region membership.
pub fn DependencyGraph(comptime capacity: Graph.Capacity) type {
    return struct {
        const Self = @This();
        const SemanticGraph = Graph.Graph(capacity, Semantic.Op);

        consumer_offsets: [capacity.max_tensors + 1]usize = @splat(0),
        consumers: [capacity.max_input_refs]Consumer = undefined,
        edge_count: usize = 0,
        topological_order: [capacity.max_nodes]usize = @splat(0),
        topological_rank: [capacity.max_nodes]usize = @splat(0),
        node_count: usize = 0,

        pub fn consumersOf(comptime self: Self, comptime tensor_id: usize) []const Consumer {
            return self.consumers[self.consumer_offsets[tensor_id]..self.consumer_offsets[tensor_id + 1]];
        }

        pub fn dependsOn(
            comptime self: Self,
            comptime graph: SemanticGraph,
            comptime node_id: usize,
            comptime predecessor_id: usize,
        ) bool {
            if (node_id == predecessor_id) return true;
            if (self.topological_rank[node_id] <= self.topological_rank[predecessor_id]) return false;

            var visited: [capacity.max_nodes]bool = @splat(false);
            var pending: [capacity.max_nodes]usize = @splat(0);
            var pending_count: usize = 1;
            pending[0] = predecessor_id;
            visited[predecessor_id] = true;

            var cursor: usize = 0;
            while (cursor < pending_count) : (cursor += 1) {
                const producer_id = pending[cursor];
                const result_id = graph.nodes[producer_id].?.result;
                for (self.consumersOf(result_id)) |consumer| {
                    if (consumer.node == node_id) return true;
                    if (visited[consumer.node] or
                        self.topological_rank[consumer.node] >= self.topological_rank[node_id]) continue;
                    visited[consumer.node] = true;
                    pending[pending_count] = consumer.node;
                    pending_count += 1;
                }
            }
            return false;
        }

        fn buildTopologicalOrder(self: *Self, comptime graph: SemanticGraph) void {
            var incoming: [capacity.max_nodes]usize = @splat(0);
            var emitted: [capacity.max_nodes]bool = @splat(false);

            for (0..graph.node_ct) |node_id| {
                const node = graph.nodes[node_id].?;
                for (0..node.input_count) |input_index| {
                    const tensor_id = graph.input_refs[node.input_start + input_index].?;
                    if (graph.tensors[tensor_id].?.origin == .node) incoming[node_id] += 1;
                }
            }

            while (self.node_count < graph.node_ct) {
                var ready: ?usize = null;
                for (0..graph.node_ct) |node_id| {
                    if (!emitted[node_id] and incoming[node_id] == 0) {
                        ready = node_id;
                        break;
                    }
                }
                const node_id = ready orelse @compileError("semantic graph contains a dependency cycle");
                emitted[node_id] = true;
                self.topological_order[self.node_count] = node_id;
                self.topological_rank[node_id] = self.node_count;
                self.node_count += 1;

                const result_id = graph.nodes[node_id].?.result;
                for (self.consumersOf(result_id)) |consumer| incoming[consumer.node] -= 1;
            }
        }
    };
}

pub fn Facts(comptime capacity: Graph.Capacity) type {
    return struct {
        use_counts: [capacity.max_tensors]usize = @splat(0),
        is_output: [capacity.max_tensors]bool = @splat(false),
        dependencies: DependencyGraph(capacity) = .{},
    };
}
