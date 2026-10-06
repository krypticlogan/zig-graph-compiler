const Execution = @import("../../execution/execution.zig");
const Executable = @import("../../execution/program.zig").Executable;
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
        ) Executable(capacity, Execution.Op) {
            var program: Executable(capacity, Execution.Op) = .init();

            for (0..source_graph.tensor_ct) |tensor_id| {
                const info = source_graph.tensors[tensor_id].?;
                const inserted = program.insertTensor(info);
                if (inserted != tensor_id) @compileError("executable lowering changed tensor order");
                program.materialized[tensor_id] = switch (info.origin) {
                    .source, .literal => true,
                    .node => false,
                };
            }
            for (0..source_graph.sources.len) |source_index| {
                if (source_graph.sources[source_index]) |source| program.insertSource(source_index, source);
            }

            inline for (topological_order[0..source_graph.node_ct]) |node_id| {
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
                    .compute => |compute| .{ .compute = planCompute(compute, &source_graph, node) },
                };
                var inputs: [node.input_count]usize = undefined;
                inline for (0..node.input_count) |input_index| {
                    inputs[input_index] = source_graph.input_refs[node.input_start + input_index].?;
                }
                insertInvocation(&program, executable, &inputs, &.{node.result});
            }

            for (0..source_graph.output_ct) |output_index| {
                program.insertOutput(source_graph.outputs[output_index].?);
            }
            return program;
        }

        fn insertInvocation(
            program: *Executable(capacity, Execution.Op),
            comptime op: Execution.Op,
            comptime inputs: []const usize,
            comptime outputs: []const usize,
        ) void {
            for (inputs) |tensor_id| program.insertInputRef(tensor_id);
            for (outputs) |tensor_id| {
                program.insertOutputRef(tensor_id);
                program.materialized[tensor_id] = true;
                program.tensors[tensor_id].?.origin = .{ .node = program.node_ct };
            }
            program.insertInvocation(.{
                .op = op,
                .input_start = program.input_ref_ct - inputs.len,
                .input_count = inputs.len,
                .output_start = program.output_ref_ct - outputs.len,
                .output_count = outputs.len,
            });
        }

        fn planCompute(
            comptime compute: Semantic.Op.Compute,
            comptime graph: anytype,
            comptime node: anytype,
        ) Execution.ExecutableCompute {
            return switch (compute) {
                .matmul => blk: {
                    const lhs_id = graph.input_refs[node.input_start].?;
                    const rhs_id = graph.input_refs[node.input_start + 1].?;
                    const Planned = ContractionPlanner.planned(capacity, graph, lhs_id, rhs_id, node.result);
                    break :blk .{ .kernel = .{ .contraction = Planned.kernel_plan } };
                },
                else => .{ .direct = compute },
            };
        }
    };
}
