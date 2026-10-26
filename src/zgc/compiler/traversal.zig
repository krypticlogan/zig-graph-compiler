//! Shared sparse dependency traversal for semantic graphs and physical programs.
const Graph = @import("../core/graph.zig");

pub const Consumer = struct { node: usize, input_index: usize };

pub fn outputCount(node: anytype) usize {
    return if (@hasField(@TypeOf(node), "result")) 1 else node.output_count;
}

pub fn outputTensor(graph: anytype, node: anytype, index: usize) usize {
    return if (@hasField(@TypeOf(node), "result")) node.result else graph.output_refs[node.output_start + index].?;
}

pub fn Dependencies(comptime capacity: Graph.Capacity) type {
    return struct {
        const Self = @This();
        consumer_offsets: [capacity.max_tensors + 1]usize = @splat(0),
        consumers: [capacity.max_input_refs]Consumer = undefined,
        incoming: [capacity.max_nodes]usize = @splat(0),
        edge_count: usize = 0,
        topological_order: [capacity.max_nodes]usize = @splat(0),
        topological_rank: [capacity.max_nodes]usize = @splat(0),
        node_count: usize = 0,

        pub fn consumersOf(self: *const Self, tensor_id: usize) []const Consumer {
            return self.consumers[self.consumer_offsets[tensor_id]..self.consumer_offsets[tensor_id + 1]];
        }

        pub fn build(comptime graph: anytype) Self {
            var result: Self = .{};
            var already_ordered = true;
            for (graph.nodes[0..graph.node_ct], 0..) |entry, node_id| {
                const node = entry.?;
                for (0..node.input_count) |index| {
                    const tensor_id = graph.input_refs[node.input_start + index].?;
                    result.consumer_offsets[tensor_id + 1] += 1;
                    if (graph.tensors[tensor_id].?.origin == .node) {
                        const producer = graph.tensors[tensor_id].?.origin.node;
                        result.incoming[node_id] += 1;
                        already_ordered = already_ordered and producer < node_id;
                    }
                }
            }
            for (0..graph.tensor_ct) |id| result.consumer_offsets[id + 1] += result.consumer_offsets[id];
            var cursors = result.consumer_offsets;
            for (graph.nodes[0..graph.node_ct], 0..) |entry, node_id| {
                const node = entry.?;
                for (0..node.input_count) |index| {
                    const tensor_id = graph.input_refs[node.input_start + index].?;
                    result.consumers[cursors[tensor_id]] = .{ .node = node_id, .input_index = index };
                    cursors[tensor_id] += 1;
                }
            }
            result.edge_count = graph.input_ref_ct;

            // Canonical graphs and physical programs commonly already have a
            // valid order. Validate it while indexing edges and retain it.
            if (already_ordered) {
                for (0..graph.node_ct) |id| {
                    result.topological_order[id] = id;
                    result.topological_rank[id] = id;
                }
                result.node_count = graph.node_ct;
                return result;
            }

            var incoming = result.incoming;
            var ready: [capacity.max_nodes]usize = undefined;
            var head: usize = 0;
            var tail: usize = 0;
            for (0..graph.node_ct) |id| {
                if (incoming[id] == 0) {
                    ready[tail] = id;
                    tail += 1;
                }
            }
            while (head < tail) : (head += 1) {
                const id = ready[head];
                result.topological_order[result.node_count] = id;
                result.topological_rank[id] = result.node_count;
                result.node_count += 1;
                const node = graph.nodes[id].?;
                for (0..outputCount(node)) |output| {
                    for (result.consumersOf(outputTensor(&graph, node, output))) |consumer| {
                        incoming[consumer.node] -= 1;
                        if (incoming[consumer.node] == 0) {
                            ready[tail] = consumer.node;
                            tail += 1;
                        }
                    }
                }
            }
            if (result.node_count != graph.node_ct) @compileError("graph contains a dependency cycle");
            return result;
        }

        pub fn dependsOn(self: *const Self, comptime graph: anytype, node_id: usize, predecessor_id: usize) bool {
            if (node_id == predecessor_id) return true;
            if (self.topological_rank[node_id] <= self.topological_rank[predecessor_id]) return false;
            var visited: [capacity.max_nodes]bool = @splat(false);
            var pending: [capacity.max_nodes]usize = undefined;
            pending[0] = predecessor_id;
            visited[predecessor_id] = true;
            var count: usize = 1;
            var cursor: usize = 0;
            while (cursor < count) : (cursor += 1) {
                const producer = graph.nodes[pending[cursor]].?;
                for (0..outputCount(producer)) |output| {
                    for (self.consumersOf(outputTensor(&graph, producer, output))) |consumer| {
                        if (consumer.node == node_id) return true;
                        if (visited[consumer.node] or self.topological_rank[consumer.node] >= self.topological_rank[node_id]) continue;
                        visited[consumer.node] = true;
                        pending[count] = consumer.node;
                        count += 1;
                    }
                }
            }
            return false;
        }
    };
}
