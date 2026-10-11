const std = @import("std");
const zgc = @import("zgc");
const models = @import("fixtures/models.zig");

test "executable search combines analysis alternatives and schedule variants" {
    const Model = models.MatmulModel;

    try std.testing.expectEqual(@as(usize, 1), Model.fusion_candidate_count);
    try std.testing.expect(Model.layout_candidate_count >= 2);
    try std.testing.expectEqual(@as(usize, 4), Model.contraction_candidate_count);
    try std.testing.expect(Model.representation_candidate_count > 8);
    try std.testing.expect(Model.generated_contraction_strategies[3]); // scalar
    var strategy_count: usize = 0;
    for (Model.generated_contraction_strategies) |generated| {
        if (generated) strategy_count += 1;
    }
    try std.testing.expect(strategy_count >= 2);
    try std.testing.expectEqual(
        3 * (Model.representation_candidate_count + 1),
        Model.executable_candidate_count,
    );

    const reference = Model.reference_executable_candidate;
    try std.testing.expectEqual(.reference, reference.origin);
    try std.testing.expectEqual(.semantic, reference.schedule);
    try std.testing.expectEqual(@as(usize, 1), reference.executable.node_ct);
    switch (reference.executable.nodes[0].?.op) {
        .compute => |compute| switch (compute) {
            .direct => |semantic| try std.testing.expectEqual(.matmul, semantic),
            .kernel => return error.TestUnexpectedResult,
        },
        .view => return error.TestUnexpectedResult,
    }

    try std.testing.expectEqual(.planned, Model.selected_executable_candidate.origin);
    try std.testing.expectEqual(.memory_pressure, Model.selected_executable_candidate.schedule);
    try std.testing.expect(Model.executable_candidate_frontier.count > 0);
    try std.testing.expect(Model.executable_candidate_frontier.count <= 16);
}

test "contraction analysis choices lower to distinct specialized plans" {
    const Model = models.MatmulModel;
    const capacity = Model.internal_capacity;
    const candidates = comptime zgc.compiler.analysis.ContractionAnalysis(capacity).analyze(Model.semantic_graph);
    try std.testing.expectEqual(@as(usize, 4), candidates.count);

    const Strategy = @TypeOf(candidates.values[0].?.strategy);
    const baseline_strategies: [capacity.max_nodes]?Strategy = @splat(null);
    const scalar_strategies = comptime blk: {
        var strategies = baseline_strategies;
        strategies[0] = .scalar;
        break :blk strategies;
    };
    const baseline = comptime zgc.compiler.planning.Lowering(capacity).lower(
        Model.semantic_graph,
        Model.semantic_analysis_result.dependencies.topological_order,
        .{},
        .{},
        baseline_strategies,
    );
    const scalar = comptime zgc.compiler.planning.Lowering(capacity).lower(
        Model.semantic_graph,
        Model.semantic_analysis_result.dependencies.topological_order,
        .{},
        .{},
        scalar_strategies,
    );
    try std.testing.expectEqual(.output_columns, baseline.nodes[0].?.op.compute.kernel.contraction.strategy);
    try std.testing.expectEqual(.scalar, scalar.nodes[0].?.op.compute.kernel.contraction.strategy);
}

test "layout analysis offers propagated and unpacked contraction layouts" {
    const Model = models.MatmulModel;
    const capacity = Model.internal_capacity;
    const analysis = zgc.compiler.analysis.LayoutAnalysis(capacity);
    const candidates = comptime analysis.analyze(Model.semantic_graph, Model.semantic_analysis_result);
    try std.testing.expectEqual(@as(usize, 3), candidates.count);
    try std.testing.expectEqual(.propagated, candidates.values[1].?.regime);
    try std.testing.expectEqual(.unpacked, candidates.values[2].?.regime);

    const propagated = comptime analysis.applySelection(Model.semantic_graph, candidates.values[1].?.regions);
    const unpacked = comptime analysis.applySelection(Model.semantic_graph, candidates.values[2].?.regions);
    try std.testing.expect(!std.meta.eql(propagated.tensors[1].?.layout, unpacked.tensors[1].?.layout));
    try std.testing.expect(std.meta.eql(propagated.tensors[2].?.layout, unpacked.tensors[2].?.layout));
}

test "structured costs support Pareto dominance" {
    const baseline: zgc.compiler.PlanCost = .{
        .estimated_runtime_work = 100,
        .peak_memory_bytes = 64,
        .persistent_memory_bytes = 32,
        .scratch_memory_bytes = 8,
        .code_size_units = 4,
    };
    const improvement: zgc.compiler.PlanCost = .{
        .estimated_runtime_work = 90,
        .peak_memory_bytes = 64,
        .persistent_memory_bytes = 32,
        .scratch_memory_bytes = 8,
        .code_size_units = 4,
    };
    const tradeoff: zgc.compiler.PlanCost = .{
        .estimated_runtime_work = 80,
        .peak_memory_bytes = 128,
        .persistent_memory_bytes = 32,
        .scratch_memory_bytes = 8,
        .code_size_units = 4,
    };

    try std.testing.expect(improvement.dominates(baseline));
    try std.testing.expect(!baseline.dominates(improvement));
    try std.testing.expect(!tradeoff.dominates(improvement));
    try std.testing.expect(!improvement.dominates(tradeoff));
}
