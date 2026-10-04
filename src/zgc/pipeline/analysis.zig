const Elementwise = @import("../operations/elementwise.zig");
const Reduction = @import("../operations/reduction.zig");
const Graph = @import("../graph.zig");
const Semantic = @import("../operations/semantic.zig");
const Tensor = @import("../tensor.zig");
const layout_ops = @import("../kernels/layout.zig");
const matmul = @import("../optimization/matmul.zig");

pub fn SemanticAnalysis() type {
    return struct {
        pub fn analyze(comptime Validated: type) Facts(Validated.graph.tensor_ct) {
            const graph = Validated.graph;
            var result: Facts(graph.tensor_ct) = .{};

            for (0..graph.node_ct) |node_id| {
                const node = graph.nodes[node_id].?;
                for (0..node.input_count) |input_index| {
                    const ref_index = node.input_start + input_index;
                    result.use_counts[graph.input_refs[ref_index].?] += 1;
                }
            }
            for (0..graph.output_ct) |output_index| {
                result.is_output[graph.outputs[output_index].?] = true;
            }

            return result;
        }
    };
}

pub fn Facts(comptime tensor_count: usize) type {
    return struct {
        use_counts: [tensor_count]usize = @splat(0),
        is_output: [tensor_count]bool = @splat(false),
    };
}

/// Discover legal fusion-region alternatives without selecting one.
pub fn FusionAnalysis(comptime capacity: Graph.Capacity) type {
    return struct {
        const SemanticGraph = Graph.Graph(capacity, Semantic.Op);
        const GraphFacts = Facts(capacity.max_tensors);
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
            return candidates;
        }

        fn hasSelectedRegion(comptime graph: SemanticGraph, selection: FusionSelection(capacity.max_nodes)) bool {
            for (selection.node_region[0..graph.node_ct]) |region| {
                if (region != null) return true;
            }
            return false;
        }

        fn select(comptime graph: SemanticGraph, comptime analysis: GraphFacts) FusionSelection(capacity.max_nodes) {
            var result: FusionSelection(capacity.max_nodes) = .{};

            for (0..graph.node_ct) |node_id| {
                const descriptor = reductionForNode(graph, node_id) orelse continue;
                const input_id = graph.input_refs[graph.nodes[node_id].?.input_start].?;
                const anchor = fusionAnchor(graph, analysis, input_id);

                var selected_group: ?usize = null;
                for (0..result.region_count) |region_id| {
                    const group = result.reduction_storage[region_id].?;
                    if (!compatible(graph, group, descriptor, input_id, anchor)) continue;
                    if (!canDelayExistingOutputs(graph, group, node_id)) continue;
                    selected_group = region_id;
                    break;
                }

                const region_id = selected_group orelse blk: {
                    const id = result.region_count;
                    result.reduction_storage[id] = .{
                        .descriptor = descriptor,
                        .domain_tensor = input_id,
                        .anchor_tensor = anchor,
                    };
                    result.region_count += 1;
                    break :blk id;
                };
                var group = &result.reduction_storage[region_id].?;
                group.reduction_nodes[group.reduction_count] = node_id;
                group.reduction_count += 1;
                group.emit_node = node_id;
            }

            for (0..result.region_count) |region_id| {
                var group = &result.reduction_storage[region_id].?;
                var has_producer = false;
                for (group.reduction_nodes[0..group.reduction_count]) |maybe_node_id| {
                    const node_id = maybe_node_id.?;
                    const node = graph.nodes[node_id].?;
                    const input_id = graph.input_refs[node.input_start].?;
                    has_producer = markFoldedProducers(graph, analysis, input_id, &group.nodes) or has_producer;
                }

                if (group.reduction_count == 1 and !has_producer) continue;
                for (group.reduction_nodes[0..group.reduction_count]) |maybe_node_id| {
                    result.node_region[maybe_node_id.?] = .{ .reduction = region_id };
                }
                for (group.nodes, 0..) |folded, node_id| {
                    if (folded) result.node_region[node_id] = .{ .reduction = region_id };
                }
            }

            for (0..graph.node_ct) |node_id| {
                if (result.node_region[node_id] != null or pointwiseForNode(graph, node_id) == null) continue;
                if (hasUnassignedPointwiseConsumer(graph, analysis, result, node_id)) continue;

                var group: MapGroup(capacity.max_nodes) = .{ .root_node = node_id };
                group.nodes[node_id] = true;
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

                const map_id = result.map_count;
                result.map_storage[map_id] = group;
                result.map_count += 1;
                for (group.nodes, 0..) |included, included_node| {
                    if (included) result.node_region[included_node] = .{ .map = map_id };
                }
            }
            return result;
        }

        fn pointwiseForNode(comptime graph: SemanticGraph, comptime node_id: usize) ?Elementwise.Operation {
            return switch (graph.nodes[node_id].?.op) {
                .compute => |compute| Elementwise.fromCompute(compute),
                .view => null,
            };
        }

        fn hasUnassignedPointwiseConsumer(
            comptime graph: SemanticGraph,
            comptime analysis: GraphFacts,
            comptime selection: FusionSelection(capacity.max_nodes),
            comptime producer_id: usize,
        ) bool {
            const result_id = graph.nodes[producer_id].?.result;
            if (analysis.use_counts[result_id] != 1 or analysis.is_output[result_id]) return false;
            for (producer_id + 1..graph.node_ct) |consumer_id| {
                if (selection.node_region[consumer_id] != null or pointwiseForNode(graph, consumer_id) == null) continue;
                const consumer = graph.nodes[consumer_id].?;
                for (0..consumer.input_count) |input_index| {
                    if (graph.input_refs[consumer.input_start + input_index].? == result_id) return true;
                }
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
            if (selection.node_region[producer_id] != null or pointwiseForNode(graph, producer_id) == null) return;
            if (group.nodes[producer_id]) return;

            group.nodes[producer_id] = true;
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

        fn reductionForNode(comptime graph: SemanticGraph, comptime node_id: usize) ?Reduction.Descriptor {
            return switch (graph.nodes[node_id].?.op) {
                .compute => |compute| Reduction.fromCompute(compute),
                .view => null,
            };
        }

        fn compatible(
            comptime graph: SemanticGraph,
            comptime group: Group(capacity.max_nodes),
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

        fn canDelayExistingOutputs(comptime graph: SemanticGraph, comptime group: Group(capacity.max_nodes), comptime candidate_node: usize) bool {
            for (group.reduction_nodes[0..group.reduction_count]) |maybe_node_id| {
                const reduction_node = maybe_node_id.?;
                const result_id = graph.nodes[reduction_node].?.result;
                for (reduction_node + 1..candidate_node + 1) |consumer_id| {
                    const consumer = graph.nodes[consumer_id].?;
                    for (0..consumer.input_count) |input_index| {
                        if (graph.input_refs[consumer.input_start + input_index].? == result_id) return false;
                    }
                }
            }
            return true;
        }

        fn fusionAnchor(comptime graph: SemanticGraph, comptime analysis: GraphFacts, comptime tensor_id: usize) usize {
            const producer_id = switch (graph.tensors[tensor_id].?.origin) {
                .node => |id| id,
                .source, .literal => return tensor_id,
            };
            if (analysis.use_counts[tensor_id] != 1 or analysis.is_output[tensor_id]) return tensor_id;
            const producer = graph.nodes[producer_id].?;
            const compute = switch (producer.op) {
                .compute => |value| value,
                .view => return tensor_id,
            };
            const operation = Elementwise.fromCompute(compute) orelse return tensor_id;
            if (!operation.isReductionCompatible() or producer.input_count == 0) return tensor_id;
            return fusionAnchor(graph, analysis, graph.input_refs[producer.input_start].?);
        }

        fn markFoldedProducers(
            comptime graph: SemanticGraph,
            comptime analysis: GraphFacts,
            comptime tensor_id: usize,
            folded: *[capacity.max_nodes]bool,
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
            const operation = Elementwise.fromCompute(compute) orelse return false;
            if (!operation.isReductionCompatible()) return false;

            folded[producer_id] = true;
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
        pub const max_count = 2;

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
        reduction_storage: [node_count]?Group(node_count) = @splat(null),
        map_storage: [node_count]?MapGroup(node_count) = @splat(null),
        node_region: [node_count]?FusionRegionRef = @splat(null),
        region_count: usize = 0,
        map_count: usize = 0,
    };
}

pub const FusionRegionRef = union(enum) { reduction: usize, map: usize };

pub fn MapGroup(comptime node_count: usize) type {
    return struct {
        root_node: usize,
        nodes: [node_count]bool = @splat(false),
        node_count: usize = 0,
    };
}

pub fn Group(comptime node_count: usize) type {
    return struct {
        descriptor: Reduction.Descriptor,
        domain_tensor: usize,
        anchor_tensor: usize,
        reduction_nodes: [node_count]?usize = @splat(null),
        reduction_count: usize = 0,
        nodes: [node_count]bool = @splat(false),
        emit_node: usize = 0,
    };
}

pub const LayoutRegime = enum {
    canonical,
    propagated,
};

pub const LayoutCandidates = struct {
    values: [2]LayoutRegime = .{ .canonical, .propagated },
    count: usize = 2,
};

pub fn LayoutAnalysis(comptime capacity: Graph.Capacity) type {
    return struct {
        const SemanticGraph = Graph.Graph(capacity, Semantic.Op);
        const Analysis = Facts(capacity.max_tensors);

        pub fn analyze(comptime semantic_graph: SemanticGraph, comptime analysis: Analysis) LayoutCandidates {
            _ = semantic_graph;
            _ = analysis;
            return .{};
        }

        pub fn apply(
            comptime semantic_graph: SemanticGraph,
            comptime analysis: Analysis,
            comptime regime: LayoutRegime,
        ) SemanticGraph {
            return switch (regime) {
                .canonical => semantic_graph,
                .propagated => propagate(semantic_graph, analysis),
            };
        }

        fn propagate(comptime raw: SemanticGraph, comptime analysis: Analysis) SemanticGraph {
            var graph: SemanticGraph = .init();

            for (0..raw.tensor_ct) |tensor_id| {
                const inserted = graph.insertTensor(raw.tensors[tensor_id].?);
                if (inserted != tensor_id) @compileError("optimization changed tensor order");
            }
            for (0..raw.sources.len) |source_index| {
                if (raw.sources[source_index]) |source| graph.insertSource(source_index, source);
            }

            for (0..raw.node_ct) |node_id| {
                const node = raw.nodes[node_id].?;
                const input_ids = raw.input_refs[node.input_start..][0..node.input_count];
                var result = graph.tensors[node.result].?;

                const planned: Semantic.Op = switch (node.op) {
                    .view => |view| blk: {
                        const InputInfos = [node.input_count]SemanticGraph.TensorInfo;
                        var inputs: InputInfos = undefined;
                        inline for (0..node.input_count) |input_index| {
                            inputs[input_index] = graph.tensors[input_ids[input_index].?].?;
                        }
                        const inferred = layout_ops.infer(view, &inputs, result.shape, capacity.max_rank);
                        result.shape = inferred.shape;
                        result.layout = inferred.layout;
                        result.storage_tensor = inferred.storage_tensor;
                        break :blk .{ .view = view };
                    },
                    .compute => |compute| blk: {
                        result.layout = computeLayout(compute, &graph, input_ids, result.shape, analysis);
                        break :blk .{ .compute = compute };
                    },
                };
                graph.tensors[node.result] = result;

                inline for (0..node.input_count) |input_index| graph.insertRef(input_ids[input_index].?);
                graph.insertNode(.{
                    .op = planned,
                    .input_start = graph.input_ref_ct - node.input_count,
                    .input_count = node.input_count,
                    .result = node.result,
                });
            }

            for (0..raw.output_ct) |output_index| graph.insertOutput(raw.outputs[output_index].?);
            return graph;
        }

        fn computeLayout(
            comptime op: Semantic.Op.Compute,
            graph: *SemanticGraph,
            comptime input_ids: []const ?Tensor.Id,
            shape: Tensor.Shape(capacity.max_rank),
            comptime analysis: Analysis,
        ) Tensor.Layout(capacity.max_rank) {
            return switch (op) {
                .matmul => matmul.selectOutputLayout(capacity, graph, input_ids[0].?, input_ids[1].?, shape, analysis),
                .where => preserveBatchLayout(graph, input_ids[1].?, shape),
                .contiguous, .pad, .shift, .sum, .mean, .min, .max, .concat => .contiguous(shape),
                else => preserveBatchLayout(graph, input_ids[0].?, shape),
            };
        }

        fn preserveBatchLayout(
            graph: *const SemanticGraph,
            comptime input_id: Tensor.Id,
            output_shape: Tensor.Shape(capacity.max_rank),
        ) Tensor.Layout(capacity.max_rank) {
            const input = graph.tensors[input_id].?;
            if (output_shape.rank == 2 and
                input.shape.rank == 2 and
                input.storage_tensor == input_id and
                input.shape.at(0) == output_shape.at(0) and
                input.shape.at(1) == output_shape.at(1) and
                isBatchLayout(input))
            {
                return .firstAxisContiguous(output_shape);
            }
            return .contiguous(output_shape);
        }

        fn isBatchLayout(info: Tensor.Info(capacity.max_rank)) bool {
            return info.shape.rank == 2 and
                info.layout.offset == 0 and
                info.layout.strides[0] == 1 and
                info.layout.strides[1] == @as(isize, @intCast(info.shape.at(0)));
        }
    };
}
