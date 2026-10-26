const std = @import("std");
const internal = @import("zgc_internal");
const zgc = internal.api;
const models = @import("fixtures/models.zig");

test "executable search combines analysis alternatives and schedule variants" {
    const Executable = models.MatmulExecutable;

    try std.testing.expectEqual(@as(usize, 1), Executable.fusion_candidate_count);
    try std.testing.expect(Executable.layout_candidate_count >= 2);
    try std.testing.expectEqual(@as(usize, 4), Executable.contraction_candidate_count);
    try std.testing.expect(Executable.representation_candidate_count > 8);
    try std.testing.expect(Executable.generated_contraction_strategies[3]); // scalar
    var strategy_count: usize = 0;
    for (Executable.generated_contraction_strategies) |generated| {
        if (generated) strategy_count += 1;
    }
    try std.testing.expect(strategy_count >= 2);
    try std.testing.expectEqual(
        3 * (Executable.representation_candidate_count + 1),
        Executable.executable_candidate_count,
    );

    const reference = Executable.reference_executable_candidate;
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

    try std.testing.expectEqual(.planned, Executable.selected_executable_candidate.origin);
    try std.testing.expectEqual(.memory_pressure, Executable.selected_executable_candidate.schedule);
    try std.testing.expect(Executable.executable_candidate_frontier.count > 0);
    try std.testing.expect(Executable.executable_candidate_frontier.count <= 16);
    const fingerprint = internal.analysis.fingerprint;
    try std.testing.expectEqual(fingerprint.executable(Executable.program), Executable.structural_fingerprint);
    try std.testing.expect(fingerprint.executable(reference.executable) != Executable.structural_fingerprint);
    const cost = Executable.selected_executable_candidate.cost;
    try std.testing.expect(cost.access_work > 0);
    try std.testing.expectEqual(
        cost.arithmetic_work + cost.access_work + cost.register_pressure_work + cost.transfer_work,
        cost.estimated_runtime_work,
    );
}

test "model selection honors optimization targets and a forced frontier index" {
    const definition = models.matmul_definition;
    const Default = models.MatmulExecutable;
    const Fast = definition.executableWith(.{ .target = .fast });
    const Memory = definition.executableWith(.{ .target = .memory });
    const Small = definition.executableWith(.{ .target = .small });
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
    const Forced = definition.executableWith(.{ .target = .fast, .frontier_index = forced_index });
    try std.testing.expectEqual(forced_index, Forced.selected_executable_candidate_index);
    try std.testing.expect(std.meta.eql(
        comptime internal.finalization.executable(frontier.candidates[forced_index].?.executable),
        Forced.program,
    ));

    var regular = Default.Runtime.init();
    var forced = Forced.Runtime.init();
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
    const Cost = internal.analysis.Cost;
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
    const Executable = models.MatmulExecutable;
    const capacity = Executable.semantic_capacity;
    const candidates = comptime internal.analysis.ContractionAnalysis(capacity).analyze(Executable.semantic_graph);
    try std.testing.expectEqual(@as(usize, 4), candidates.count);

    const Strategy = @TypeOf(candidates.values[0].?.strategy);
    const baseline_strategies: [capacity.max_nodes]?Strategy = @splat(null);
    const scalar_strategies = comptime blk: {
        var strategies = baseline_strategies;
        strategies[0] = .scalar;
        break :blk strategies;
    };
    const baseline = comptime internal.planning.Lowering(capacity).lower(
        Executable.semantic_graph,
        Executable.semantic_analysis_result.dependencies.topological_order,
        .{},
        .{},
        baseline_strategies,
    );
    const scalar = comptime internal.planning.Lowering(capacity).lower(
        Executable.semantic_graph,
        Executable.semantic_analysis_result.dependencies.topological_order,
        .{},
        .{},
        scalar_strategies,
    );
    try std.testing.expectEqual(.output_columns, baseline.nodes[0].?.op.compute.kernel.contraction.strategy);
    try std.testing.expectEqual(.scalar, scalar.nodes[0].?.op.compute.kernel.contraction.strategy);
}

test "layout analysis offers propagated and unpacked contraction layouts" {
    const Executable = models.MatmulExecutable;
    const capacity = Executable.semantic_capacity;
    const analysis = internal.analysis.LayoutAnalysis(capacity);
    const candidates = comptime analysis.analyze(Executable.semantic_graph, Executable.semantic_analysis_result);
    try std.testing.expectEqual(@as(usize, 3), candidates.count);
    try std.testing.expectEqual(.propagated, candidates.values[1].?.regime);
    try std.testing.expectEqual(.unpacked, candidates.values[2].?.regime);

    const propagated = comptime analysis.applySelection(Executable.semantic_graph, candidates.values[1].?.regions);
    const unpacked = comptime analysis.applySelection(Executable.semantic_graph, candidates.values[2].?.regions);
    try std.testing.expect(!std.meta.eql(propagated.tensors[1].?.layout, unpacked.tensors[1].?.layout));
    try std.testing.expect(std.meta.eql(propagated.tensors[2].?.layout, unpacked.tensors[2].?.layout));
}

test "structured costs support Pareto dominance" {
    const baseline: internal.search.PlanCost = .{
        .estimated_runtime_work = 100,
        .peak_memory_bytes = 64,
        .persistent_memory_bytes = 32,
        .scratch_memory_bytes = 8,
        .code_size_units = 4,
    };
    const improvement: internal.search.PlanCost = .{
        .estimated_runtime_work = 90,
        .peak_memory_bytes = 64,
        .persistent_memory_bytes = 32,
        .scratch_memory_bytes = 8,
        .code_size_units = 4,
    };
    const tradeoff: internal.search.PlanCost = .{
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
    const cost = models.ReuseExecutable.selected_executable_candidate.cost;
    try std.testing.expectEqual(models.ReuseExecutable.memory_plan.byte_count, cost.peak_memory_bytes);
    try std.testing.expect(cost.scratch_memory_bytes > 0);
    try std.testing.expectEqual(
        cost.peak_memory_bytes,
        cost.persistent_memory_bytes + cost.scratch_memory_bytes,
    );
}

test "search limits count completed schedules and attempted branches" {
    inline for (.{ 1, 2, 4, 7, 20 }) |limit| {
        const Executable = models.matmul_definition.executableWith(.{ .search = .{
            .max_executables = limit,
            .max_composition_attempts = 5,
        } });
        try std.testing.expect(Executable.executable_candidate_count <= limit);
        if (limit <= 7) try std.testing.expectEqual(@as(usize, limit), Executable.executable_candidate_count);
        try std.testing.expect(Executable.executable_candidate_count > 0);
        try std.testing.expect(Executable.search_composition_attempts <= 5);
        try std.testing.expect(Executable.executable_candidate_frontier.count > 0);
        if (limit == 1) {
            try std.testing.expectEqual(@as(usize, 1), Executable.executable_candidate_count);
            try std.testing.expectEqual(@as(usize, 0), Executable.search_composition_attempts);
            try std.testing.expectEqual(.reference, Executable.selected_executable_candidate.origin);
        }
        if (limit >= 4) try std.testing.expect(Executable.search_composition_limit_reached);
    }
}

test "zero branch allowance retains an optimized compatible anchor" {
    const definition = comptime blk: {
        var builder = zgc.DefinitionBuilder.init();
        const input = builder.sources(enum { input }).input(.input, .f32, &.{8});
        builder.output(builder.relu(builder.neg(input)));
        break :blk builder.finish();
    };
    const Executable = definition.executableWith(.{ .search = .{
        .max_executables = 2,
        .max_composition_attempts = 0,
    } });
    try std.testing.expectEqual(@as(usize, 2), Executable.executable_candidate_count);
    try std.testing.expectEqual(@as(usize, 0), Executable.search_composition_attempts);
    try std.testing.expect(Executable.program.node_ct < Executable.reference_executable_candidate.executable.node_ct);
    var runtime = Executable.Runtime.init();
    try runtime.copyInput(.input, &.{ -1, 2, -3, 4, -5, 6, -7, 8 });
    runtime.run();
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 3, 0, 5, 0, 7, 0 }, runtime.outputView(0).contiguousSlice().?);
}

test "single executable allowance preserves mandatory masked fusion" {
    const Executable = comptime blk: {
        var builder = zgc.DefinitionBuilder.init();
        const sources = builder.sources(enum { condition, input });
        const condition = sources.input(.condition, .bool, &.{4});
        const input = sources.input(.input, .f32, &.{4});
        builder.output(builder.sum(builder.where(condition, input, builder.scalar(.f32, 0)), .{ .axes = &.{0} }));
        break :blk builder.finish().executableWith(.{ .search = .{
            .max_executables = 1,
            .max_composition_attempts = 0,
        } });
    };
    try std.testing.expectEqual(@as(usize, 1), Executable.executable_candidate_count);
    try std.testing.expectEqual(.planned, Executable.selected_executable_candidate.origin);
    var runtime = Executable.Runtime.init();
    try runtime.copyInput(.condition, &.{ true, false, true, false });
    try runtime.copyInput(.input, &.{ 1, std.math.nan(f32), 3, std.math.nan(f32) });
    runtime.run();
    try std.testing.expectEqual(@as(f32, 4), runtime.outputView(0).get(.{}));
}
