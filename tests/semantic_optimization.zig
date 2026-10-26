const std = @import("std");
const zgc = @import("zgc_internal").api;

const Sources = enum(usize) { input, tail, fill, dead };
const Definition = zgc.DefinitionBuilder;

const OptimizationExecutable = model: {
    var builder = Definition.init();
    const builder_sources = builder.sources(Sources);
    const input = builder_sources.input(.input, .f32, &.{10});
    const tail = builder_sources.input(.tail, .f32, &.{1});
    const fill = builder_sources.input(.fill, .f32, &.{});
    const dead = builder_sources.input(.dead, .f32, &.{10});

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
    break :model builder.finish().executable();
};

test "semantic optimization compacts and canonicalizes the live graph" {
    const graph = OptimizationExecutable.semantic_graph;

    try std.testing.expect(OptimizationExecutable.raw_graph.node_ct > graph.node_ct);
    try std.testing.expectEqual(@as(?usize, null), OptimizationExecutable.semantic_provenance.raw_to_optimized_tensor[3]);
    try std.testing.expectEqual(@as(?usize, null), OptimizationExecutable.semantic_provenance.raw_to_optimized_tensor[4]);
    try std.testing.expectEqual(@as(usize, 5), graph.node_ct);
    try std.testing.expectEqual(graph.node_ct, graph.nodes.len);
    try std.testing.expectEqual(graph.tensor_ct, graph.tensors.len);
    try std.testing.expectEqual(graph.input_ref_ct, graph.input_refs.len);
    try std.testing.expectEqual(graph.output_ct, graph.outputs.len);
    try std.testing.expectEqual(graph.node_ct, OptimizationExecutable.semantic_analysis_result.dependencies.topological_order.len);
    try std.testing.expectEqual(graph.tensor_ct, OptimizationExecutable.semantic_analysis_result.use_counts.len);
    try std.testing.expectEqual(OptimizationExecutable.raw_graph.tensor_ct, OptimizationExecutable.semantic_provenance.raw_to_optimized_tensor.len);
    try std.testing.expectEqual(graph.tensor_ct, OptimizationExecutable.semantic_provenance.optimized_to_raw_tensor.len);
    try std.testing.expectEqual(graph.node_ct, OptimizationExecutable.semantic_provenance.optimized_to_raw_node.len);

    const folded = graph.tensors[graph.outputs[1].?].?.origin.literal;
    try std.testing.expectEqual(@as(f32, 5), folded.get(.f32));
    try std.testing.expectEqual(graph.outputs[0].?, graph.sources[@backingInt(Sources.input)].?.tensor);

    const slice = graph.nodes[0].?.op.view.slice;
    try std.testing.expectEqual(@as(usize, 3), slice.start);
    try std.testing.expectEqual(@as(usize, 2), slice.length);
    try std.testing.expectEqual(@as(usize, 4), slice.step);
    try std.testing.expectEqual(@as(usize, 3), graph.nodes[graph.node_ct - 1].?.input_count);
}

test "finalization shrinks source slots and ranks while preserving sparse source IDs" {
    const SparseSources = enum(usize) { input = 7, dead = 19 };
    const Executable = comptime blk: {
        var builder = Definition.init();
        const sources = builder.sources(SparseSources);
        const input = sources.input(.input, .f32, &.{4});
        const dead = sources.input(.dead, .f32, &.{ 2, 2, 2, 2 });
        _ = builder.neg(dead);
        builder.output(input);
        break :blk builder.finish().executable();
    };
    try std.testing.expectEqual(@as(usize, 4), Executable.raw_capacity.max_rank);
    try std.testing.expectEqual(@as(usize, 20), Executable.raw_capacity.max_sources);
    try std.testing.expectEqual(@as(usize, 1), Executable.semantic_capacity.max_rank);
    try std.testing.expectEqual(@as(usize, 8), Executable.semantic_capacity.max_sources);
    try std.testing.expectEqual(@as(usize, 1), Executable.semantic_graph.source_ct);
    try std.testing.expectEqual(@as(usize, 8), Executable.semantic_graph.sources.len);
    try std.testing.expectEqual(@as(usize, 0), Executable.program.node_ct);
    try std.testing.expectEqual(@as(usize, 0), Executable.program.nodes.len);
    try std.testing.expectEqual(@as(usize, 0), Executable.program.output_refs.len);
    try std.testing.expectEqual(@as(usize, 1), Executable.program.tensors[0].?.shape.dims.len);
    try std.testing.expectEqual(@as(usize, 1), Executable.program.tensors[0].?.layout.strides.len);
    try std.testing.expectEqual(@as(usize, 7), Executable.program.tensors[0].?.origin.source);
    try std.testing.expectEqual(@as(usize, 0), Executable.program.sources[7].?.tensor);
    var runtime = Executable.Runtime.init();
    try runtime.copyInput(.input, &.{ 1, 2, 3, 4 });
    runtime.run();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, runtime.outputView(0).contiguousSlice().?);
}

test "selected fused programs use exact invocation and reference capacities" {
    const Executable = comptime blk: {
        var builder = Definition.init();
        const input = builder.sources(enum { input }).input(.input, .f32, &.{4});
        builder.output(builder.relu(builder.neg(input)));
        break :blk builder.finish().executable();
    };
    const program = Executable.program;
    try std.testing.expect(program.node_ct < Executable.semantic_graph.node_ct);
    try std.testing.expectEqual(program.node_ct, program.nodes.len);
    try std.testing.expectEqual(program.tensor_ct, program.tensors.len);
    try std.testing.expectEqual(program.input_ref_ct, program.input_refs.len);
    try std.testing.expectEqual(program.output_ref_ct, program.output_refs.len);
    try std.testing.expectEqual(program.output_ct, program.outputs.len);
    try std.testing.expectEqual(program.node_ct, Executable.internal_capacity.max_nodes);
    try std.testing.expectEqual(program.output_ref_ct, Executable.internal_capacity.max_output_refs.?);
    try std.testing.expect(std.meta.eql(program, Executable.selected_executable_candidate.executable));
    var runtime = Executable.Runtime.init();
    try runtime.copyInput(.input, &.{ -2, -1, 1, 2 });
    runtime.run();
    try std.testing.expectEqualSlices(f32, &.{ 2, 1, 0, 0 }, runtime.outputView(0).contiguousSlice().?);
}

test "planner composes structural remaps and preserves optimized graph semantics" {
    try std.testing.expectEqual(.composed, OptimizationExecutable.selected_executable_candidate.remap_regime.?);
    var found_remap = false;
    for (OptimizationExecutable.program.nodes[0..OptimizationExecutable.program.node_ct]) |maybe_node| {
        const node = maybe_node.?;
        found_remap = found_remap or switch (node.op) {
            .compute => |compute| switch (compute) {
                .kernel => |kernel| kernel == .map and kernel.map.strategy == .segmented,
                .direct => false,
            },
            .view => false,
        };
    }
    try std.testing.expect(found_remap);

    var model = OptimizationExecutable.Runtime.init();
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
