const Set = @import("../node_set.zig").Set;
const Elementwise = @import("../../operations/elementwise.zig");
const Reduction = @import("../../operations/reduction.zig");
const Graph = @import("../../core/graph.zig");
const Semantic = @import("../../operations/semantic.zig");
const Facts = @import("semantic.zig").Facts;

pub fn FusionAnalysis(comptime capacity: Graph.Capacity) type {
    return struct {
        const SemanticGraph = Graph.Graph(capacity, Semantic.Op);
        const GraphFacts = Facts(capacity);
        const Result = FusionCandidates(capacity.max_nodes);

        pub fn analyze(comptime semantic_graph: SemanticGraph, comptime analysis: GraphFacts) Result {
            var candidates: Result = .{};
            candidates.add(.{
                .regime = .unfused,
                .regions = .{},
            });

            const discovered = select(semantic_graph, analysis);
            if (hasSelectedRegion(semantic_graph, discovered)) {
                candidates.add(.{
                    .regime = .discovered,
                    .regions = discovered,
                });
            }
            const independent = selectIndependentReductions(semantic_graph, analysis);
            if (hasSelectedRegion(semantic_graph, independent)) {
                candidates.add(.{
                    .regime = .independent,
                    .regions = independent,
                });
            }
            return candidates;
        }

        /// Keep reductions as separate specialized kernels. Search can combine
        /// these with discovered map groups or choose the shared traversal.
        fn selectIndependentReductions(comptime graph: SemanticGraph, comptime analysis: GraphFacts) FusionSelection(capacity.max_nodes) {
            var result: FusionSelection(capacity.max_nodes) = .{};
            for (analysis.dependencies.topological_order[0..graph.node_ct]) |node_id| {
                const descriptor = analysis.reductions[node_id] orelse continue;
                const input_id = graph.input_refs[graph.nodes[node_id].?.input_start].?;
                const region_id = result.region_count;
                var group: Group(capacity.max_nodes) = .{
                    .descriptor = descriptor,
                    .domain_tensor = input_id,
                    .anchor_tensor = input_id,
                    .emit_node = node_id,
                };
                const reduction_nodes = [_]?usize{node_id};
                group.reduction_nodes = &reduction_nodes;
                group.reduction_count = 1;
                group.members.set(node_id);
                const frozen = group;
                result.reduction_storage[region_id] = &frozen;
                result.node_region[node_id] = .{ .reduction = region_id };
                result.region_count += 1;
            }
            return result;
        }

        fn hasSelectedRegion(comptime graph: SemanticGraph, selection: FusionSelection(capacity.max_nodes)) bool {
            for (selection.node_region[0..graph.node_ct]) |region| {
                if (region != null) return true;
            }
            return false;
        }

        fn select(comptime graph: SemanticGraph, comptime analysis: GraphFacts) FusionSelection(capacity.max_nodes) {
            var result: FusionSelection(capacity.max_nodes) = .{};
            var groups: [capacity.max_nodes]?Group(capacity.max_nodes) = @splat(null);
            var first: [capacity.max_nodes]?usize = @splat(null);
            var last: [capacity.max_nodes]?usize = @splat(null);
            var next: [capacity.max_nodes]?usize = @splat(null);
            // Open-addressed signatures keep unrelated reduction domains out of
            // the group search. A signature can still own multiple groups when
            // dependencies prevent their outputs from being delayed together.
            const bucket_count = capacity.max_nodes * 2 + 1;
            var bucket_first: [bucket_count]?usize = @splat(null);
            var next_group: [capacity.max_nodes]?usize = @splat(null);
            var reduction_group: [capacity.max_nodes]?usize = @splat(null);
            var visited: [capacity.max_nodes]usize = @splat(0);
            var blocked_group: [capacity.max_nodes]usize = @splat(0);
            var pending: [capacity.max_nodes]usize = undefined;

            for (analysis.dependencies.topological_order[0..graph.node_ct]) |node_id| {
                const descriptor = analysis.reductions[node_id] orelse continue;
                const input_id = graph.input_refs[graph.nodes[node_id].?.input_start].?;
                const anchor = analysis.fusion_anchor[input_id];

                var bucket: usize = @intCast(signatureHash(graph, descriptor, input_id, anchor) % bucket_count);
                while (bucket_first[bucket]) |head| {
                    if (compatible(graph, &groups[head].?, descriptor, input_id, anchor)) break;
                    bucket = (bucket + 1) % bucket_count;
                }
                var selected_group: ?usize = null;
                var last_matching: ?usize = null;
                const stamp = node_id + 1;
                if (bucket_first[bucket] != null) {
                    // Walk the candidate's ancestors once rather than asking
                    // reachability separately for every reduction in a group.
                    var pending_count: usize = 1;
                    var pending_index: usize = 0;
                    pending[0] = node_id;
                    visited[node_id] = stamp;
                    while (pending_index < pending_count) : (pending_index += 1) {
                        const ancestor = graph.nodes[pending[pending_index]].?;
                        for (0..ancestor.input_count) |input_index| {
                            const tensor_id = graph.input_refs[ancestor.input_start + input_index].?;
                            const producer_id = switch (graph.tensors[tensor_id].?.origin) {
                                .node => |id| id,
                                .source, .literal => continue,
                            };
                            if (reduction_group[producer_id]) |id| blocked_group[id] = stamp;
                            if (visited[producer_id] == stamp) continue;
                            visited[producer_id] = stamp;
                            pending[pending_count] = producer_id;
                            pending_count += 1;
                        }
                    }
                }
                var cursor = bucket_first[bucket];
                while (cursor) |region_id| {
                    if (blocked_group[region_id] != stamp) {
                        selected_group = region_id;
                        break;
                    }
                    last_matching = region_id;
                    cursor = next_group[region_id];
                }

                const region_id = selected_group orelse blk: {
                    const id = result.region_count;
                    groups[id] = .{
                        .descriptor = descriptor,
                        .domain_tensor = input_id,
                        .anchor_tensor = anchor,
                    };
                    if (last_matching) |previous| {
                        next_group[previous] = id;
                    } else {
                        bucket_first[bucket] = id;
                    }
                    result.region_count += 1;
                    break :blk id;
                };
                var group = &groups[region_id].?;
                if (last[region_id]) |previous| next[previous] = node_id else first[region_id] = node_id;
                last[region_id] = node_id;
                group.reduction_count += 1;
                group.emit_node = node_id;
                reduction_group[node_id] = region_id;
            }

            for (0..result.region_count) |region_id| {
                var group = &groups[region_id].?;
                var ids: [capacity.max_nodes]?usize = @splat(null);
                var cursor = first[region_id];
                var count: usize = 0;
                while (cursor) |node_id| {
                    ids[count] = node_id;
                    count += 1;
                    cursor = next[node_id];
                }
                const reduction_nodes = ids[0..count].*;
                group.reduction_nodes = &reduction_nodes;
                var has_producer = false;
                for (group.reduction_nodes[0..group.reduction_count]) |maybe_node_id| {
                    const node_id = maybe_node_id.?;
                    const node = graph.nodes[node_id].?;
                    const input_id = graph.input_refs[node.input_start].?;
                    has_producer = markFoldedProducers(graph, analysis, input_id, &group.nodes) or has_producer;
                }

                if (group.reduction_count > 1 or has_producer) {
                    group.members = group.nodes;
                    for (group.reduction_nodes[0..group.reduction_count]) |entry| group.members.set(entry.?);
                }
                var folded = group.nodes.iterator();
                while (folded.next()) |node_id| group.traffic_saved_bytes += analysis.tensor_bytes[graph.nodes[node_id].?.result] * 2;
                const frozen = group.*;
                result.reduction_storage[region_id] = &frozen;
                if (group.reduction_count == 1 and !has_producer) continue;
                for (group.reduction_nodes[0..group.reduction_count]) |maybe_node_id| {
                    result.node_region[maybe_node_id.?] = .{ .reduction = region_id };
                }
                var folded_nodes = group.nodes.iterator();
                while (folded_nodes.next()) |node_id| result.node_region[node_id] = .{ .reduction = region_id };
            }

            var topo_index = graph.node_ct;
            while (topo_index > 0) {
                topo_index -= 1;
                const node_id = analysis.dependencies.topological_order[topo_index];
                if (result.node_region[node_id] != null or analysis.pointwise[node_id] == null) continue;
                if (hasUnassignedPointwiseConsumer(graph, analysis, result, node_id)) continue;

                var group: MapGroup(capacity.max_nodes) = .{ .root_node = node_id };
                group.nodes.set(node_id);
                group.node_count = 1;
                const root = graph.nodes[node_id].?;
                for (0..root.input_count) |input_index| {
                    markMapProducers(
                        graph,
                        analysis,
                        result,
                        graph.input_refs[root.input_start + input_index].?,
                        &group,
                    );
                }
                if (group.node_count < 2) continue;

                var internal_nodes = group.nodes.iterator();
                while (internal_nodes.next()) |included| {
                    if (included != group.root_node) group.traffic_saved_bytes += analysis.tensor_bytes[graph.nodes[included].?.result] * 2;
                }
                const map_id = result.map_count;
                const frozen = group;
                result.map_storage[map_id] = &frozen;
                result.map_count += 1;
                var members = group.nodes.iterator();
                while (members.next()) |included_node| result.node_region[included_node] = .{ .map = map_id };
            }
            return result;
        }

        fn hasUnassignedPointwiseConsumer(
            comptime graph: SemanticGraph,
            comptime analysis: GraphFacts,
            comptime selection: FusionSelection(capacity.max_nodes),
            comptime producer_id: usize,
        ) bool {
            const result_id = graph.nodes[producer_id].?.result;
            if (analysis.use_counts[result_id] != 1 or analysis.is_output[result_id]) return false;
            for (analysis.dependencies.consumersOf(result_id)) |edge| {
                const consumer_id = edge.node;
                if (selection.node_region[consumer_id] != null or analysis.pointwise[consumer_id] == null) continue;
                return true;
            }
            return false;
        }

        fn markMapProducers(
            comptime graph: SemanticGraph,
            comptime analysis: GraphFacts,
            comptime selection: FusionSelection(capacity.max_nodes),
            comptime tensor_id: usize,
            group: *MapGroup(capacity.max_nodes),
        ) void {
            if (analysis.use_counts[tensor_id] != 1 or analysis.is_output[tensor_id]) return;
            const producer_id = switch (graph.tensors[tensor_id].?.origin) {
                .node => |id| id,
                .source, .literal => return,
            };
            if (selection.node_region[producer_id] != null or analysis.pointwise[producer_id] == null) return;
            if (group.nodes.isSet(producer_id)) return;

            group.nodes.set(producer_id);
            group.node_count += 1;
            const producer = graph.nodes[producer_id].?;
            for (0..producer.input_count) |input_index| {
                markMapProducers(
                    graph,
                    analysis,
                    selection,
                    graph.input_refs[producer.input_start + input_index].?,
                    group,
                );
            }
        }

        fn compatible(
            comptime graph: SemanticGraph,
            group: *const Group(capacity.max_nodes),
            comptime descriptor: Reduction.Descriptor,
            comptime input_id: usize,
            comptime anchor: usize,
        ) bool {
            if (group.descriptor.axes != descriptor.axes or
                group.descriptor.keep_dims != descriptor.keep_dims or
                group.anchor_tensor != anchor)
            {
                return false;
            }
            const expected = graph.tensors[group.domain_tensor].?;
            const candidate = graph.tensors[input_id].?;
            if (expected.dtype != candidate.dtype or expected.shape.rank != candidate.shape.rank) return false;
            for (0..expected.shape.rank) |axis| {
                if (expected.shape.at(axis) != candidate.shape.at(axis)) return false;
            }
            return true;
        }

        fn signatureHash(
            comptime graph: SemanticGraph,
            descriptor: Reduction.Descriptor,
            input_id: usize,
            anchor: usize,
        ) u64 {
            const tensor = graph.tensors[input_id].?;
            var hash: u64 = 14695981039346656037;
            hash = (hash ^ @as(u64, @intCast(anchor))) *% 1099511628211;
            hash = (hash ^ descriptor.axes) *% 1099511628211;
            hash = (hash ^ @as(u64, @intFromBool(descriptor.keep_dims))) *% 1099511628211;
            hash = (hash ^ @as(u64, @backingInt(tensor.dtype))) *% 1099511628211;
            hash = (hash ^ @as(u64, @intCast(tensor.shape.rank))) *% 1099511628211;
            for (0..tensor.shape.rank) |axis| {
                hash = (hash ^ @as(u64, @intCast(tensor.shape.at(axis)))) *% 1099511628211;
            }
            return hash;
        }

        fn markFoldedProducers(
            comptime graph: SemanticGraph,
            comptime analysis: GraphFacts,
            comptime tensor_id: usize,
            folded: *Set(capacity.max_nodes),
        ) bool {
            const producer_id = switch (graph.tensors[tensor_id].?.origin) {
                .node => |id| id,
                .source, .literal => return false,
            };
            if (analysis.use_counts[tensor_id] != 1 or analysis.is_output[tensor_id]) return false;
            const producer = graph.nodes[producer_id].?;
            const compute = switch (producer.op) {
                .compute => |value| value,
                .view => return false,
            };
            if (Elementwise.fromCompute(compute) == null) return false;

            folded.set(producer_id);
            for (0..producer.input_count) |input_index| {
                _ = markFoldedProducers(
                    graph,
                    analysis,
                    graph.input_refs[producer.input_start + input_index].?,
                    folded,
                );
            }
            return true;
        }
    };
}

pub const FusionRegime = enum {
    unfused,
    discovered,
    independent,
};

pub fn FusionCandidate(comptime node_count: usize) type {
    return struct {
        regime: FusionRegime,
        regions: FusionSelection(node_count),
    };
}

pub fn FusionCandidates(comptime node_count: usize) type {
    return struct {
        const Self = @This();
        pub const max_count = 3;

        values: [max_count]?FusionCandidate(node_count) = @splat(null),
        count: usize = 0,

        fn add(candidates: *Self, candidate: FusionCandidate(node_count)) void {
            candidates.values[candidates.count] = candidate;
            candidates.count += 1;
        }
    };
}

pub fn FusionSelection(comptime node_count: usize) type {
    return struct {
        reduction_storage: [node_count]?*const Group(node_count) = @splat(null),
        map_storage: [node_count]?*const MapGroup(node_count) = @splat(null),
        node_region: [node_count]?FusionRegionRef = @splat(null),
        region_count: usize = 0,
        map_count: usize = 0,
    };
}

pub const FusionRegionRef = union(enum) { reduction: usize, map: usize };

pub fn MapGroup(comptime node_count: usize) type {
    return struct {
        root_node: usize,
        traffic_saved_bytes: u128 = 0,
        nodes: Set(node_count) = .{},
        node_count: usize = 0,
    };
}

pub fn Group(comptime node_count: usize) type {
    return struct {
        descriptor: Reduction.Descriptor,
        members: Set(node_count) = .{},
        traffic_saved_bytes: u128 = 0,
        domain_tensor: usize,
        anchor_tensor: usize,
        reduction_nodes: []const ?usize = &.{},
        reduction_count: usize = 0,
        nodes: Set(node_count) = .{},
        emit_node: usize = 0,
    };
}
