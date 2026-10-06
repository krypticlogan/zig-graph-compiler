const std = @import("std");
const Graph = @import("../../core/graph.zig");
const Semantic = @import("../../operations/semantic.zig");
const Tensor = @import("../../core/tensor.zig");
const semantic = @import("semantic.zig");
const Facts = semantic.Facts;

pub const RemapRegime = enum {
    direct,
    composed,
};

pub fn RemapCandidate(comptime node_count: usize) type {
    return struct {
        regime: RemapRegime,
        regions: RemapSelection(node_count),
    };
}

pub fn RemapCandidates(comptime node_count: usize) type {
    return struct {
        values: [2]?RemapCandidate(node_count) = .{ null, null },
        count: usize = 0,

        fn add(candidates: *@This(), candidate: RemapCandidate(node_count)) void {
            candidates.values[candidates.count] = candidate;
            candidates.count += 1;
        }
    };
}

pub fn RemapSelection(comptime node_count: usize) type {
    return struct {
        groups: [node_count]?RemapGroup(node_count) = @splat(null),
        node_region: [node_count]?usize = @splat(null),
        region_count: usize = 0,
    };
}

pub fn RemapGroup(comptime node_count: usize) type {
    return struct {
        root_node: usize,
        nodes: [node_count]bool = @splat(false),
        node_count: usize = 0,
    };
}

/// Discover optional regions of shifts and concatenations whose static index
/// transforms can be lowered into one segmented transfer program.
pub fn RemapAnalysis(comptime capacity: Graph.Capacity) type {
    return struct {
        const SemanticGraph = Graph.Graph(capacity, Semantic.Op);
        const GraphFacts = Facts(capacity);
        const Result = RemapCandidates(capacity.max_nodes);

        pub fn analyze(comptime graph: SemanticGraph, comptime facts: GraphFacts) Result {
            var candidates: Result = .{};
            candidates.add(.{ .regime = .direct, .regions = .{} });
            const selection = discover(graph, facts);
            if (selection.region_count != 0) {
                candidates.add(.{ .regime = .composed, .regions = selection });
            }
            return candidates;
        }

        fn discover(comptime graph: SemanticGraph, comptime facts: GraphFacts) RemapSelection(capacity.max_nodes) {
            var selection: RemapSelection(capacity.max_nodes) = .{};
            var topo_index = graph.node_ct;
            while (topo_index > 0) {
                topo_index -= 1;
                const node_id = facts.dependencies.topological_order[topo_index];
                if (!isRemapNode(graph, node_id) or hasComposableConsumer(graph, facts, node_id)) continue;
                const region_id = selection.region_count;
                selection.groups[region_id] = .{ .root_node = node_id };
                includeNode(graph, facts, node_id, &selection.groups[region_id].?);
                for (selection.groups[region_id].?.nodes, 0..) |included, included_node| {
                    if (included) selection.node_region[included_node] = region_id;
                }
                selection.region_count += 1;
            }
            return selection;
        }

        fn includeNode(
            comptime graph: SemanticGraph,
            comptime facts: GraphFacts,
            comptime node_id: usize,
            group: *RemapGroup(capacity.max_nodes),
        ) void {
            if (group.nodes[node_id]) return;
            group.nodes[node_id] = true;
            group.node_count += 1;
            const node = graph.nodes[node_id].?;
            const absorbs_inputs = switch (node.op.compute) {
                .concat => true,
                .shift, .slice_loop => false,
                else => unreachable,
            };
            if (!absorbs_inputs) return;
            for (0..node.input_count) |input_index| {
                const tensor_id = graph.input_refs[node.input_start + input_index].?;
                if (facts.use_counts[tensor_id] != 1 or facts.is_output[tensor_id]) continue;
                const producer_id = switch (graph.tensors[tensor_id].?.origin) {
                    .node => |id| id,
                    .source, .literal => continue,
                };
                if (isRemapNode(graph, producer_id)) includeNode(graph, facts, producer_id, group);
            }
        }

        fn hasComposableConsumer(comptime graph: SemanticGraph, comptime facts: GraphFacts, comptime node_id: usize) bool {
            const tensor_id = graph.nodes[node_id].?.result;
            if (facts.use_counts[tensor_id] != 1 or facts.is_output[tensor_id]) return false;
            for (facts.dependencies.consumersOf(tensor_id)) |edge| {
                const consumer_id = edge.node;
                if (!isRemapNode(graph, consumer_id)) continue;
                const consumer = graph.nodes[consumer_id].?;
                switch (consumer.op.compute) {
                    .concat => {},
                    .shift, .slice_loop => continue,
                    else => unreachable,
                }
                return true;
            }
            return false;
        }

        fn isRemapNode(comptime graph: SemanticGraph, comptime node_id: usize) bool {
            return switch (graph.nodes[node_id].?.op) {
                .compute => |compute| switch (compute) {
                    .shift, .slice_loop, .concat => true,
                    else => false,
                },
                .view => false,
            };
        }
    };
}

test "dependency analysis drives remap discovery independently of node identifiers" {
    const capacity: Graph.Capacity = .{
        .max_nodes = 2,
        .max_input_refs = 3,
        .max_tensors = 4,
        .max_outputs = 1,
        .max_sources = 2,
        .max_rank = 1,
    };
    const SemanticGraph = Graph.Graph(capacity, Semantic.Op);
    const shape = Tensor.Shape(1).init(&.{4});
    const layout = Tensor.Layout(1).contiguous(shape);
    const semantic_graph: SemanticGraph = comptime blk: {
        var value: SemanticGraph = .init();
        _ = value.insertTensor(.{
            .dtype = .f32,
            .shape = shape,
            .layout = layout,
            .storage_tensor = 0,
            .origin = .{ .source = 0 },
        });
        _ = value.insertTensor(.{
            .dtype = .f32,
            .shape = shape,
            .layout = layout,
            .storage_tensor = 1,
            .origin = .{ .source = 1 },
        });
        _ = value.insertTensor(.{
            .dtype = .f32,
            .shape = shape,
            .layout = layout,
            .storage_tensor = 2,
            .origin = .{ .node = 1 },
        });
        _ = value.insertTensor(.{
            .dtype = .f32,
            .shape = shape,
            .layout = layout,
            .storage_tensor = 3,
            .origin = .{ .node = 0 },
        });

        // The concat consumer deliberately precedes its shift producer by ID.
        value.insertRef(2);
        value.insertRef(1);
        value.insertNode(.{
            .op = .{ .compute = .{ .concat = .{ .axis = 0 } } },
            .input_start = 0,
            .input_count = 2,
            .result = 3,
        });
        value.insertRef(0);
        value.insertNode(.{
            .op = .{ .compute = .{ .shift = .{ .offsets = &.{1}, .boundary = .wrap } } },
            .input_start = 2,
            .input_count = 1,
            .result = 2,
        });
        value.insertOutput(3);
        break :blk value;
    };
    const Validated = struct {
        pub const graph = semantic_graph;
    };
    const facts = semantic.SemanticAnalysis(capacity).analyze(Validated);

    try std.testing.expectEqualSlices(usize, &.{ 1, 0 }, facts.dependencies.topological_order[0..2]);
    try std.testing.expectEqual(@as(usize, 1), facts.dependencies.consumersOf(2).len);
    try std.testing.expectEqual(@as(usize, 0), facts.dependencies.consumersOf(2)[0].node);
    try std.testing.expect(facts.dependencies.dependsOn(semantic_graph, 0, 1));

    const candidates = RemapAnalysis(capacity).analyze(semantic_graph, facts);
    try std.testing.expectEqual(@as(usize, 2), candidates.count);
    const selection = candidates.values[1].?.regions;
    try std.testing.expectEqual(@as(usize, 1), selection.region_count);
    try std.testing.expectEqual(@as(usize, 0), selection.groups[0].?.root_node);
    try std.testing.expect(selection.groups[0].?.nodes[0]);
    try std.testing.expect(selection.groups[0].?.nodes[1]);
}
