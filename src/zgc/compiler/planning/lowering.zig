const ProgramBuilder = @import("../program_builder.zig").Builder;
const Execution = @import("../../execution/execution.zig");
const Program = @import("../../execution/program.zig").Program;
const Graph = @import("../../core/graph.zig");
const Semantic = @import("../../operations/semantic.zig");
const Analysis = @import("../analysis/root.zig");
const ContractionPlanner = @import("contraction.zig");
const Map = @import("map.zig");
const Reduction = @import("reduction.zig");

/// Lower one selected fusion and remap combination into an executable program.
pub fn Lowering(comptime capacity: Graph.Capacity) type {
    return struct {
        const SemanticGraph = Graph.Graph(capacity, Semantic.Op);
        const FusionRegions = Analysis.FusionSelection(capacity.max_nodes);
        const RemapRegions = Analysis.RemapSelection(capacity.max_nodes);
        const MapPlanner = Map.Planner(capacity);
        const ReductionPlanner = Reduction.Planner(capacity);

        pub fn lower(
            comptime source_graph: SemanticGraph,
            comptime topological_order: [capacity.max_nodes]usize,
            comptime fusion_regions: FusionRegions,
            comptime remap_regions: RemapRegions,
            comptime contraction_strategies: [capacity.max_nodes]?ContractionPlanner.Plan.Strategy,
        ) Program(capacity, Execution.Op) {
            return comptime blk: {
                var program: ProgramBuilder(capacity, Execution.Op) = .{};

                for (topological_order[0..source_graph.node_ct]) |node_id| {
                    if (remap_regions.node_region[node_id]) |region_id| {
                        const group = remap_regions.groups[region_id].?;
                        if (node_id != group.root_node) continue;
                        const Planned = MapPlanner.planRemap(source_graph, group, fusion_regions, remap_regions);
                        insertInvocation(
                            &program,
                            .{ .compute = .{ .kernel = .{ .map = Planned.kernel_plan } } },
                            &Planned.input_ids,
                            &Planned.output_ids,
                        );
                        continue;
                    }
                    if (fusion_regions.node_region[node_id]) |region| {
                        switch (region) {
                            .reduction => |region_id| {
                                const group = fusion_regions.reduction_storage[region_id].?;
                                if (node_id != group.emit_node) continue;
                                const Planned = ReductionPlanner.plan(source_graph, group);
                                insertInvocation(
                                    &program,
                                    .{ .compute = .{ .kernel = .{ .reduction = Planned.kernel_plan } } },
                                    &Planned.input_ids,
                                    &Planned.output_ids,
                                );
                            },
                            .map => |map_id| {
                                const group = fusion_regions.map_storage[map_id].?;
                                if (node_id != group.root_node) continue;
                                if (MapPlanner.canSink(source_graph, group, remap_regions)) continue;
                                const Planned = MapPlanner.planMap(source_graph, group);
                                insertInvocation(
                                    &program,
                                    .{ .compute = .{ .kernel = .{ .map = Planned.kernel_plan } } },
                                    &Planned.input_ids,
                                    &Planned.output_ids,
                                );
                            },
                        }
                        continue;
                    }

                    const node = source_graph.nodes[node_id].?;
                    if (node.op == .view and
                        MapPlanner.sunkMapGroupForTensor(source_graph, node.result, fusion_regions, remap_regions) != null)
                    {
                        continue;
                    }
                    const executable: Execution.Op = switch (node.op) {
                        .view => |view| .{ .view = view },
                        .compute => |compute| .{ .compute = planCompute(compute, &source_graph, node, contraction_strategies[node_id]) },
                    };
                    var inputs: [node.input_count]usize = undefined;
                    for (0..node.input_count) |input_index| {
                        inputs[input_index] = source_graph.input_refs[node.input_start + input_index].?;
                    }
                    insertInvocation(&program, executable, &inputs, &.{node.result});
                }

                break :blk program.finish(source_graph);
            };
        }

        fn insertInvocation(
            comptime program: *ProgramBuilder(capacity, Execution.Op),
            comptime op: Execution.Op,
            comptime inputs: []const usize,
            comptime outputs: []const usize,
        ) void {
            program.append(op, inputs, outputs);
        }

        fn planCompute(
            comptime compute: Semantic.Op.Compute,
            comptime graph: anytype,
            comptime node: anytype,
            comptime strategy: ?ContractionPlanner.Plan.Strategy,
        ) Execution.ExecutableCompute {
            return switch (compute) {
                .matmul => blk: {
                    const lhs_id = graph.input_refs[node.input_start].?;
                    const rhs_id = graph.input_refs[node.input_start + 1].?;
                    const Planned = ContractionPlanner.planned(capacity, graph, lhs_id, rhs_id, node.result, strategy);
                    break :blk .{ .kernel = .{ .contraction = Planned.kernel_plan } };
                },
                else => .{ .direct = compute },
            };
        }
    };
}
