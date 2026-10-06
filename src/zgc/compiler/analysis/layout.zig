const std = @import("std");
const Graph = @import("../../core/graph.zig");
const Semantic = @import("../../operations/semantic.zig");
const Tensor = @import("../../core/tensor.zig");
const layout_ops = @import("../../kernels/layout.zig");
const matmul = @import("../planning/contraction.zig");
const Facts = @import("semantic.zig").Facts;

pub const LayoutRegime = enum {
    canonical,
    mixed,
    propagated,
};

/// A physical layout condition advertised by an implementation candidate.
pub fn LayoutRequirement(comptime max_rank: usize) type {
    return struct {
        tensor: Tensor.Id,
        layout: ?Tensor.Layout(max_rank) = null,
        contiguous: bool = false,
    };
}

/// The physical representation produced by an implementation candidate.
pub fn LayoutResult(comptime max_rank: usize) type {
    return struct {
        tensor: Tensor.Id,
        layout: Tensor.Layout(max_rank),
    };
}

pub fn LayoutGroup(comptime capacity: Graph.Capacity) type {
    return struct {
        anchor_tensor: Tensor.Id,
        tensors: [capacity.max_tensors]bool = @splat(false),
        tensor_count: usize = 0,
    };
}

pub fn LayoutSelection(comptime capacity: Graph.Capacity) type {
    return struct {
        results: [capacity.max_tensors]?LayoutResult(capacity.max_rank) = @splat(null),
        tensor_region: [capacity.max_tensors]?usize = @splat(null),
        groups: [capacity.max_tensors]?LayoutGroup(capacity) = @splat(null),
        region_count: usize = 0,
    };
}

pub fn LayoutCandidate(comptime capacity: Graph.Capacity) type {
    return struct {
        regime: LayoutRegime,
        regions: LayoutSelection(capacity),
    };
}

pub fn LayoutCandidates(comptime capacity: Graph.Capacity) type {
    return struct {
        values: [2]?LayoutCandidate(capacity) = @splat(null),
        count: usize = 0,

        fn add(candidates: *@This(), candidate: LayoutCandidate(capacity)) void {
            candidates.values[candidates.count] = candidate;
            candidates.count += 1;
        }
    };
}

pub fn LayoutAnalysis(comptime capacity: Graph.Capacity) type {
    return struct {
        const SemanticGraph = Graph.Graph(capacity, Semantic.Op);
        const Analysis = Facts(capacity);

        pub fn analyze(comptime semantic_graph: SemanticGraph, comptime analysis: Analysis) LayoutCandidates(capacity) {
            var candidates: LayoutCandidates(capacity) = .{};
            candidates.add(.{ .regime = .canonical, .regions = .{} });
            const propagated = propagate(semantic_graph, analysis);
            const selection = partitionChanges(semantic_graph, propagated);
            if (selection.region_count != 0) {
                candidates.add(.{ .regime = .propagated, .regions = selection });
            }
            return candidates;
        }

        pub fn applySelection(
            comptime semantic_graph: SemanticGraph,
            comptime selection: LayoutSelection(capacity),
        ) SemanticGraph {
            var graph = semantic_graph;
            for (selection.results[0..semantic_graph.tensor_ct], 0..) |maybe_result, tensor_id| {
                if (maybe_result) |result| {
                    if (result.tensor != tensor_id) @compileError("layout result is stored under the wrong tensor identity");
                    graph.tensors[tensor_id].?.layout = result.layout;
                }
            }
            return graph;
        }

        fn partitionChanges(comptime canonical: SemanticGraph, comptime propagated: SemanticGraph) LayoutSelection(capacity) {
            var selection: LayoutSelection(capacity) = .{};
            var changed: [capacity.max_tensors]bool = @splat(false);
            for (0..canonical.tensor_ct) |tensor_id| {
                const canonical_info = canonical.tensors[tensor_id].?;
                const propagated_info = propagated.tensors[tensor_id].?;
                if (!sameLayout(canonical_info, propagated_info)) {
                    changed[tensor_id] = true;
                    selection.results[tensor_id] = .{
                        .tensor = tensor_id,
                        .layout = propagated_info.layout,
                    };
                }
            }

            var assigned: [capacity.max_tensors]bool = @splat(false);
            for (0..canonical.tensor_ct) |anchor| {
                if (!changed[anchor] or assigned[anchor]) continue;
                const region_id = selection.region_count;
                var group: LayoutGroup(capacity) = .{ .anchor_tensor = anchor };
                const storage_tensor = propagated.tensors[anchor].?.storage_tensor;

                // A layout belongs to one physical storage object. Alias
                // views must move with their storage root, while distinct
                // owned tensors remain independently selectable even when a
                // compute operation connects them.
                for (0..canonical.tensor_ct) |tensor_id| {
                    if (!changed[tensor_id] or assigned[tensor_id]) continue;
                    if (propagated.tensors[tensor_id].?.storage_tensor != storage_tensor) continue;
                    assigned[tensor_id] = true;
                    group.tensors[tensor_id] = true;
                    group.tensor_count += 1;
                    selection.tensor_region[tensor_id] = region_id;
                }
                selection.groups[region_id] = group;
                selection.region_count += 1;
            }
            return selection;
        }

        fn sameLayout(lhs: SemanticGraph.TensorInfo, rhs: SemanticGraph.TensorInfo) bool {
            return lhs.layout.offset == rhs.layout.offset and
                std.mem.eql(isize, lhs.layout.strides[0..lhs.shape.rank], rhs.layout.strides[0..rhs.shape.rank]);
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

            // Resolve layouts along dependencies. Tensor and node identifiers
            // remain stable identities and do not define evaluation order.
            for (analysis.dependencies.topological_order[0..raw.node_ct]) |node_id| {
                const node = raw.nodes[node_id].?;
                const input_ids = raw.input_refs[node.input_start..][0..node.input_count];
                var result = graph.tensors[node.result].?;

                switch (node.op) {
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
                        break :blk;
                    },
                    .compute => |compute| blk: {
                        result.layout = computeLayout(compute, &graph, input_ids, result.shape, analysis);
                        break :blk;
                    },
                }
                graph.tensors[node.result] = result;
            }

            // Preserve semantic node identities after all dependency-ordered
            // metadata has been resolved.
            for (0..raw.node_ct) |node_id| {
                const node = raw.nodes[node_id].?;
                const input_ids = raw.input_refs[node.input_start..][0..node.input_count];
                inline for (0..node.input_count) |input_index| graph.insertRef(input_ids[input_index].?);
                graph.insertNode(.{
                    .op = node.op,
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
                .contiguous, .pad, .shift, .slice_loop, .sum, .mean, .min, .max, .concat => .contiguous(shape),
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
