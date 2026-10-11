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
    const fingerprint = zgc.compiler.analysis.fingerprint;
    try std.testing.expectEqual(fingerprint.executable(Model.executable), Model.structural_fingerprint);
    try std.testing.expect(fingerprint.executable(reference.executable) != Model.structural_fingerprint);
    const cost = Model.selected_executable_candidate.cost;
    try std.testing.expect(cost.access_work > 0);
    try std.testing.expectEqual(
        cost.arithmetic_work + cost.access_work + cost.register_pressure_work + cost.transfer_work,
        cost.estimated_runtime_work,
    );
}

test "model selection honors optimization targets and a forced frontier index" {
    const definition = models.matmul_definition;
    const Default = models.MatmulModel;
    const Fast = definition.modelWithOptions(.{ .target = .fast });
    const Memory = definition.modelWithOptions(.{ .target = .memory });
    const Small = definition.modelWithOptions(.{ .target = .small });
    const frontier = Default.executable_candidate_frontier;
    try std.testing.expect(frontier.count > 1);

    for (frontier.candidates[0..frontier.count]) |entry| {
        try std.testing.expect(Fast.selected_executable_candidate.cost.estimated_runtime_work <=
            entry.?.cost.estimated_runtime_work);
        try std.testing.expect(Memory.selected_executable_candidate.cost.peak_memory_bytes <=
            entry.?.cost.peak_memory_bytes);
        try std.testing.expect(Small.selected_executable_candidate.cost.code_size_units <=
            entry.?.cost.code_size_units);
    }
    try std.testing.expectEqual(.balanced, Default.selection_options.target);

    const forced_index = frontier.count - 1;
    const Forced = definition.modelWithOptions(.{ .target = .fast, .frontier_index = forced_index });
    try std.testing.expectEqual(forced_index, Forced.selected_executable_candidate_index);
    try std.testing.expect(std.meta.eql(
        frontier.candidates[forced_index].?.executable,
        Forced.executable,
    ));

    var regular = Default.init();
    var forced = Forced.init();
    const input: [models.matmul_batch * 3]f32 = @splat(1);
    const weights = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try regular.copyInput(.input, &input);
    try regular.copySource(.weights, &weights);
    try forced.copyInput(.input, &input);
    try forced.copySource(.weights, &weights);
    regular.run();
    forced.run();
    for (0..models.matmul_batch) |row| {
        for (0..2) |column| {
            try std.testing.expectApproxEqAbs(
                regular.outputView(0).get(.{ row, column }),
                forced.outputView(0).get(.{ row, column }),
                0.0001,
            );
        }
    }
}

test "expression liveness distinguishes a chain from a wide reduction tree" {
    const Cost = zgc.compiler.analysis.Cost;
    const Instruction = Cost.Expression.Instruction;
    const chain = comptime blk: {
        var instructions: [16]Instruction = undefined;
        for (0..instructions.len) |index| {
            instructions[index] = .{
                .operation = .add,
                .dtype = .f32,
                .args = .{
                    if (index == 0) .{ .input = 0 } else .{ .instruction = index - 1 },
                    .{ .input = 1 },
                    .{ .input = 0 },
                },
            };
        }
        break :blk instructions;
    };
    const wide = comptime blk: {
        var instructions: [63]Instruction = undefined;
        for (0..32) |index| {
            instructions[index] = .{
                .operation = .add,
                .dtype = .f32,
                .args = .{ .{ .input = index }, .{ .input = 32 + index }, .{ .input = 0 } },
            };
        }
        for (32..63) |index| {
            instructions[index] = .{
                .operation = .add,
                .dtype = .f32,
                .args = .{
                    .{ .instruction = if (index == 32) 0 else index - 1 },
                    .{ .instruction = index - 31 },
                    .{ .input = 0 },
                },
            };
        }
        break :blk instructions;
    };
    const chain_live = comptime Cost.peakLiveValues(.{ .instructions = &chain });
    const wide_live = comptime Cost.peakLiveValues(.{ .instructions = &wide });
    try std.testing.expect(chain_live < wide_live);
    try std.testing.expectEqual(@as(u128, 0), Cost.registerPressureWork(100, 8, chain_live + 5));
    try std.testing.expect(Cost.registerPressureWork(100, 8, wide_live + 5) > 0);
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

test "scratch cost accounts for reusable intermediate storage" {
    const cost = models.ReuseModel.selected_executable_candidate.cost;
    try std.testing.expectEqual(models.ReuseModel.memory_plan.byte_count, cost.peak_memory_bytes);
    try std.testing.expect(cost.scratch_memory_bytes > 0);
    try std.testing.expectEqual(
        cost.peak_memory_bytes,
        cost.persistent_memory_bytes + cost.scratch_memory_bytes,
    );
}
