const std = @import("std");
const zgc = @import("zgc");

const Sources = enum(usize) { input, tail, fill, dead };
const Definition = zgc.DefinitionBuilder(Sources, .{
    .max_rank = 1,
    .max_nodes = 12,
    .max_tensors = 20,
    .max_input_refs = 24,
    .max_outputs = 4,
});

const OptimizationModel = model: {
    var builder = Definition.init();
    const input = builder.input(.input, .f32, &.{10});
    const tail = builder.input(.tail, .f32, &.{1});
    const fill = builder.input(.fill, .f32, &.{});
    const dead = builder.input(.dead, .f32, &.{10});

    _ = builder.mul(input, dead);
    const folded = builder.add(builder.scalar(.f32, 2), builder.scalar(.f32, 3));
    const identity = builder.mul(input, builder.scalar(.f32, 1));
    const first_slice = builder.slice(input, .{ .axis = 0, .start = 1, .end = 8, .step = 2 });
    const composed_slice = builder.slice(first_slice, .{ .axis = 0, .start = 1, .end = 4, .step = 2 });
    const mapped = builder.add(
        builder.mul(input, builder.scalar(.f32, 2)),
        builder.scalar(.f32, 3),
    );
    const shifted = builder.shift(mapped, &.{1}, .{ .constant = fill });
    const concatenated = builder.concat(&.{ shifted, tail }, 0);
    const flattened_concat = builder.concat(&.{ concatenated, tail }, 0);

    builder.output(identity);
    builder.output(folded);
    builder.output(composed_slice);
    builder.output(flattened_concat);
    break :model builder.finish().model();
};

test "semantic optimization compacts and canonicalizes the live graph" {
    const graph = OptimizationModel.semantic_graph;

    try std.testing.expect(OptimizationModel.raw_graph.node_ct > graph.node_ct);
    try std.testing.expectEqual(@as(?usize, null), OptimizationModel.semantic_provenance.raw_to_optimized_tensor[3]);
    try std.testing.expectEqual(@as(?usize, null), OptimizationModel.semantic_provenance.raw_to_optimized_tensor[4]);
    try std.testing.expectEqual(@as(usize, 5), graph.node_ct);

    const folded = graph.tensors[graph.outputs[1].?].?.origin.literal;
    try std.testing.expectEqual(@as(f32, 5), folded.get(.f32));
    try std.testing.expectEqual(graph.outputs[0].?, graph.sources[@intFromEnum(Sources.input)].?.tensor);

    const slice = graph.nodes[0].?.op.view.slice;
    try std.testing.expectEqual(@as(usize, 3), slice.start);
    try std.testing.expectEqual(@as(usize, 2), slice.length);
    try std.testing.expectEqual(@as(usize, 4), slice.step);
    try std.testing.expectEqual(@as(usize, 3), graph.nodes[graph.node_ct - 1].?.input_count);
}

test "planner composes shift and concatenation into one remap" {
    try std.testing.expectEqual(.composed, OptimizationModel.selected_executable_candidate.remap_regime.?);
    const map = OptimizationModel.executable.nodes[1].?.op.compute.kernel.map;
    try std.testing.expectEqual(@as(usize, 4), map.strategy.segmented.segments.len);
    try std.testing.expectEqual(
        std.simd.suggestVectorLength(f32) orelse 1,
        map.strategy.segmented.vector_width,
    );
    try std.testing.expectEqual(@as(usize, 2), map.region.body.expression_transfer.instructions.len);

    var sunk_pointwise_nodes: usize = 0;
    for (OptimizationModel.semantic_graph.nodes[0..OptimizationModel.semantic_graph.node_ct]) |maybe_node| {
        const node = maybe_node.?;
        const is_sunk = switch (node.op) {
            .compute => |compute| switch (compute) {
                .mul, .add => OptimizationModel.semantic_graph.tensors[node.result].?.shape.rank == 1,
                else => false,
            },
            .view => false,
        };
        if (!is_sunk) continue;
        sunk_pointwise_nodes += 1;
        try std.testing.expect(!OptimizationModel.executable.materialized[node.result]);
        try std.testing.expect(OptimizationModel.memory_plan.tensor_regions[node.result] == null);
    }
    try std.testing.expectEqual(@as(usize, 2), sunk_pointwise_nodes);

    var model = OptimizationModel.init();
    try model.copyInput(.input, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
    try model.copyInput(.tail, &.{9});
    try model.copyInput(.fill, &.{-1});
    model.run();

    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, model.outputView(0).contiguousSlice().?);
    try std.testing.expectEqual(@as(f32, 5), model.outputView(1).get(.{}));
    try std.testing.expectEqual(@as(f32, 4), model.outputView(2).get(.{0}));
    try std.testing.expectEqual(@as(f32, 8), model.outputView(2).get(.{1}));
    try std.testing.expectEqualSlices(f32, &.{ -1, 5, 7, 9, 11, 13, 15, 17, 19, 21, 9, 9 }, model.outputView(3).contiguousSlice().?);
}
