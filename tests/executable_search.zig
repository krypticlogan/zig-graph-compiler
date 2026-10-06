const std = @import("std");
const zgc = @import("zgc");
const models = @import("fixtures/models.zig");

test "executable search combines analysis alternatives and schedule variants" {
    const Model = models.MatmulModel;

    try std.testing.expectEqual(@as(usize, 1), Model.fusion_candidate_count);
    try std.testing.expectEqual(@as(usize, 2), Model.layout_candidate_count);
    try std.testing.expectEqual(@as(usize, 8), Model.representation_candidate_count);
    try std.testing.expectEqual(@as(usize, 27), Model.executable_candidate_count);

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
