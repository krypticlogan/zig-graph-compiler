const std = @import("std");
const zgc = @import("zgc");

const Sources = enum(usize) { lhs, rhs, auxiliary };
const Definition = zgc.DefinitionBuilder(Sources, .{
    .max_rank = 4,
    .max_nodes = 6,
    .max_tensors = 8,
    .max_input_refs = 8,
    .max_outputs = 2,
});

const matmul_model = model: {
    var builder = Definition.init();
    const lhs = builder.parameter(.lhs, .f32, &.{ 3, 4 });
    const rhs = builder.parameter(.rhs, .f32, &.{ 4, 7 });
    builder.output(builder.relu(builder.matmul(lhs, rhs)));
    break :model builder.finish().model();
};

test "definition counting and lowering preserve exact graph contracts" {
    const counts = matmul_model.internal_capacity;
    const graph = matmul_model.executable;

    try std.testing.expectEqual(@as(usize, 2), counts.max_nodes);
    try std.testing.expectEqual(@as(usize, 4), counts.max_tensors);
    try std.testing.expectEqual(@as(usize, 3), counts.max_input_refs);
    try std.testing.expectEqual(@as(usize, 1), counts.max_outputs);
    try std.testing.expectEqual(@as(usize, 2), counts.max_sources);
    try std.testing.expectEqual(@as(usize, 2), counts.max_rank);
    try std.testing.expectEqualSlices(usize, &.{ 3, 7 }, graph.tensors[2].?.shape.slice());
    try std.testing.expectEqual([2]isize{ 7, 1 }, graph.tensors[2].?.layout.strides);
    try std.testing.expectEqualStrings("contracted_axis", @tagName(graph.nodes[0].?.op.compute.kernel.contraction.strategy));
    try std.testing.expectEqual(@as(usize, 1), graph.nodes[0].?.output_count);
    try std.testing.expectEqual(@as(usize, 2), graph.output_refs[graph.nodes[0].?.output_start].?);
    try std.testing.expectEqual(@as(usize, 3), graph.outputs[0].?);
}

const batch_model = model: {
    const batch = std.simd.suggestVectorLength(f32) orelse 4;
    var builder = Definition.init();
    const input = builder.input(.lhs, .f32, &.{ batch, 4 });
    const weights = builder.parameter(.rhs, .f32, &.{ 4, 3 });
    const bias = builder.parameter(.auxiliary, .f32, &.{3});
    builder.output(builder.relu(builder.add(builder.matmul(input, weights), bias)));
    break :model builder.finish().model();
};

test "lowering fixes batch-oriented layouts and matmul strategy" {
    const graph = batch_model.executable;
    const batch = std.simd.suggestVectorLength(f32) orelse 4;
    const expected = [2]isize{ 1, batch };

    try std.testing.expectEqual(expected, graph.tensors[0].?.layout.strides);
    try std.testing.expectEqual(expected, graph.tensors[3].?.layout.strides);
    try std.testing.expectEqual(expected, graph.tensors[4].?.layout.strides);
    try std.testing.expectEqual(expected, graph.tensors[5].?.layout.strides);
    try std.testing.expectEqualStrings("output_rows", @tagName(graph.nodes[0].?.op.compute.kernel.contraction.strategy));
}

const fusion_candidate_model = model: {
    var builder = Definition.init();
    const a = builder.input(.lhs, .f32, &.{4});
    const b = builder.input(.rhs, .f32, &.{4});
    const c = builder.input(.auxiliary, .f32, &.{4});
    builder.output(builder.add(builder.mul(a, b), c));
    break :model builder.finish().model();
};

test "semantic analysis records tensor uses and outputs" {
    const raw = fusion_candidate_model.raw_graph;
    const analysis = fusion_candidate_model.semantic_analysis_result;

    try std.testing.expectEqual(@as(usize, 2), raw.node_ct);
    try std.testing.expectEqual(@as(usize, 1), analysis.use_counts[3]);
    try std.testing.expect(!analysis.is_output[3]);
    try std.testing.expect(analysis.is_output[4]);
}

const reduction_model = model: {
    var builder = Definition.init();
    const input = builder.input(.lhs, .f32, &.{ 2, 3, 4, 5 });
    const bias = builder.parameter(.rhs, .f32, &.{ 1, 4, 1 });
    const biased = builder.add(input, bias);
    builder.output(builder.mean(biased, .{ .axes = &.{ -3, -2 }, .keep_dims = true }));
    builder.output(builder.max(input, zgc.ReductionOptions{}));
    break :model builder.finish().model();
};

test "broadcasting and reductions normalize compile-time geometry" {
    const graph = reduction_model.semantic_graph;

    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 4, 5 }, graph.tensors[2].?.shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 1, 5 }, graph.tensors[3].?.shape.slice());
    try std.testing.expectEqualSlices(usize, &.{}, graph.tensors[4].?.shape.slice());
    try std.testing.expectEqual(@as(u64, (1 << 1) | (1 << 2)), graph.nodes[1].?.op.compute.mean.axes);
    try std.testing.expect(graph.nodes[1].?.op.compute.mean.keep_dims);
    try std.testing.expectEqual(@as(u64, 0b1111), graph.nodes[2].?.op.compute.max.axes);
}

const structural_model = model: {
    var builder = Definition.init();
    const input = builder.input(.lhs, .f32, &.{ 2, 1, 3, 4 });
    const squeezed = builder.squeeze(input, 1);
    const flattened = builder.flatten(squeezed, .{ .start_axis = 1 });
    const expanded = builder.unsqueeze(flattened, -1);
    const reshaped = builder.reshape(expanded, &.{ 4, 6 });
    const permuted = builder.permute(reshaped, &.{ 1, 0 });
    builder.output(builder.slice(permuted, .{ .axis = 0, .start = 1, .end = 6, .step = 2 }));
    break :model builder.finish().model();
};

test "structural operations lower to one static alias chain" {
    const graph = structural_model.executable;
    const output = graph.tensors[6].?;

    try std.testing.expectEqualSlices(usize, &.{ 3, 4 }, output.shape.slice());
    try std.testing.expectEqual([2]isize{ 2, 6 }, output.layout.strides[0..2].*);
    try std.testing.expectEqual(@as(usize, 1), output.layout.offset);
    for (1..7) |tensor_id| {
        try std.testing.expectEqual(graph.tensors[0].?.storage_tensor, graph.tensors[tensor_id].?.storage_tensor);
    }
}

test "scalar and full definitions retain immutable literal geometry" {
    const definition = comptime blk: {
        var builder = Definition.init();
        builder.output(builder.scalar(.f32, 2.5));
        builder.output(builder.full(.i8, &.{ 2, 3 }, -4));
        break :blk builder.finish();
    };

    try std.testing.expectEqualSlices(usize, &.{}, definition.tensors[0].value.shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, definition.tensors[2].value.shape.slice());
    switch (definition.tensors[0].origin) {
        .literal => |value| try std.testing.expectEqual(@as(f32, 2.5), value.get(.f32)),
        else => return error.TestUnexpectedResult,
    }
    switch (definition.tensors[1].origin) {
        .literal => |value| try std.testing.expectEqual(@as(i8, -4), value.get(.i8)),
        else => return error.TestUnexpectedResult,
    }
    try std.testing.expectEqual(@as(usize, 1), definition.node_count);
}

const predicate_model = model: {
    var builder = Definition.init();
    const input = builder.input(.lhs, .f32, &.{ 2, 3 });
    const threshold = builder.scalar(.f32, 0);
    const condition = builder.greaterThan(input, threshold);
    const fallback = builder.full(.f32, &.{ 1, 3 }, -1);
    builder.output(condition);
    builder.output(builder.where(condition, input, fallback));
    break :model builder.finish().model();
};

test "comparisons and selection carry explicit boolean dtype through lowering" {
    const graph = predicate_model.executable;
    const condition = graph.tensors[2].?;
    const selected = graph.tensors[5].?;

    try std.testing.expectEqual(zgc.memory.Dtype.bool, condition.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, condition.shape.slice());
    try std.testing.expectEqual(zgc.memory.Dtype.f32, selected.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, selected.shape.slice());
    try std.testing.expectEqual(@as(bool, true), zgc.memory.ScalarValue.init(.bool, true).get(.bool));

    var model = predicate_model.init();
    try model.copyInput(.lhs, &[_]f32{ -2, 0, 3, -4, 5, 6 });
    model.run();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, true, true }, model.outputView(0).storage);
    try std.testing.expectEqualSlices(f32, &.{ -1, -1, 3, -1, 5, 6 }, model.outputView(1).storage);
}

test "primitive builders infer every math and predicate operation without model generation" {
    const PrimitiveDefinition = zgc.DefinitionBuilder(enum(usize) { input }, .{
        .max_rank = 2,
        .max_nodes = 24,
        .max_tensors = 28,
        .max_input_refs = 40,
        .max_outputs = 1,
    });
    const definition = comptime blk: {
        var builder = PrimitiveDefinition.init();
        const input = builder.input(.input, .f32, &.{ 2, 3 });
        const lower = builder.scalar(.f32, -1);
        const upper = builder.scalar(.f32, 1);
        const negated = builder.neg(input);
        const magnitude = builder.abs(negated);
        const rooted = builder.sqrt(magnitude);
        const logged = builder.log(rooted);
        const inverted = builder.reciprocal(logged);
        const bounded_low = builder.minimum(inverted, upper);
        const bounded_high = builder.maximum(bounded_low, lower);
        const clamped = builder.clamp(bounded_high, lower, upper);
        const equal = builder.equal(input, lower);
        const not_equal = builder.notEqual(input, upper);
        _ = builder.lessThan(input, upper);
        _ = builder.lessEqual(input, upper);
        _ = builder.greaterThan(input, lower);
        _ = builder.greaterEqual(input, lower);
        const not = builder.logicalNot(equal);
        const and_mask = builder.logicalAnd(not, not_equal);
        const mask = builder.logicalOr(and_mask, equal);
        builder.output(builder.where(mask, clamped, lower));
        break :blk builder.finish();
    };

    try std.testing.expectEqual(zgc.memory.Dtype.f32, definition.tensors[definition.tensor_count - 1].value.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, definition.tensors[definition.tensor_count - 1].value.shape.slice());
}
