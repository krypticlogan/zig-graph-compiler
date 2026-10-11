const std = @import("std");
const zgc = @import("zgc");

const ProducerSources = enum(usize) { lhs, rhs };
const ProducerDefinition = zgc.DefinitionBuilder;
const ProducerReductionModel = model: {
    var builder = ProducerDefinition.init();
    const builder_sources = builder.sources(ProducerSources);
    const lhs = builder_sources.input(.lhs, .f32, &.{ 2, 3 });
    const rhs = builder_sources.input(.rhs, .f32, &.{ 2, 3 });
    builder.output(builder.sum(builder.mul(lhs, rhs), .{ .axes = &.{1} }));
    break :model builder.finish().model();
};

const SiblingSources = enum(usize) { input, weights };
const SiblingDefinition = zgc.DefinitionBuilder;
const SiblingReductionModel = model: {
    var builder = SiblingDefinition.init();
    const builder_sources = builder.sources(SiblingSources);
    const input = builder_sources.input(.input, .f32, &.{ 2, 3 });
    const weights = builder_sources.parameter(.weights, .f32, &.{3});
    builder.output(builder.sum(input, .{ .axes = &.{1} }));
    builder.output(builder.max(builder.mul(input, weights), .{ .axes = &.{1} }));
    break :model builder.finish().model();
};

const TypedPointwiseSources = enum(usize) { input, threshold };
const TypedPointwiseDefinition = zgc.DefinitionBuilder;
const TypedPointwiseModel = model: {
    var builder = TypedPointwiseDefinition.init();
    const builder_sources = builder.sources(TypedPointwiseSources);
    const input = builder_sources.input(.input, .f32, &.{8});
    const threshold = builder_sources.input(.threshold, .f32, &.{8});
    const positive = builder.greaterThan(input, threshold);
    builder.output(builder.where(positive, input, builder.neg(input)));
    break :model builder.finish().model();
};

const reduction_vector_width = std.simd.suggestVectorLength(f32) orelse 1;
const reduction_vector_length = reduction_vector_width + 1;
const VectorReductionSources = enum(usize) { input, weights };
const VectorReductionDefinition = zgc.DefinitionBuilder;
const VectorReductionModel = model: {
    var builder = VectorReductionDefinition.init();
    const builder_sources = builder.sources(VectorReductionSources);
    const input = builder_sources.input(.input, .f32, &.{ 2, reduction_vector_length });
    const weights = builder_sources.input(.weights, .f32, &.{reduction_vector_length});
    builder.output(builder.sum(builder.mul(input, weights), .{ .axes = &.{1} }));
    break :model builder.finish().model();
};

const map_vector_width = std.simd.suggestVectorLength(f32) orelse 1;
const map_vector_length = map_vector_width + 1;
const BroadcastMapDefinitionSources = enum(usize) { cells, channels };
const BroadcastMapDefinition = zgc.DefinitionBuilder;
const BroadcastMapModel = model: {
    var builder = BroadcastMapDefinition.init();
    const builder_sources = builder.sources(BroadcastMapDefinitionSources);
    const cells = builder_sources.input(.cells, .f32, &.{ 2, 3, 1 });
    const channels = builder_sources.input(.channels, .f32, &.{map_vector_length});
    builder.output(builder.add(builder.mul(cells, channels), builder.scalar(.f32, 1)));
    break :model builder.finish().model();
};

const WhereReductionSources = enum(usize) { condition, input };
const WhereReductionKind = enum { sum, min, max, mean, conditional };

fn whereReductionModel(comptime kind: WhereReductionKind, comptime frontier_index: ?usize) type {
    var builder = zgc.DefinitionBuilder.init();
    const sources = builder.sources(WhereReductionSources);
    const condition = sources.input(.condition, .bool, &.{reduction_vector_length});
    const input = sources.input(.input, .f32, &.{reduction_vector_length});
    const selected = if (kind == .conditional)
        builder.where(condition, input, builder.neg(input))
    else
        builder.where(condition, input, builder.scalar(.f32, 0));
    const reduced = switch (kind) {
        .sum, .conditional => builder.sum(selected, .{ .axes = &.{0} }),
        .min => builder.min(selected, .{ .axes = &.{0} }),
        .max => builder.max(selected, .{ .axes = &.{0} }),
        .mean => builder.mean(selected, .{ .axes = &.{0} }),
    };
    builder.output(reduced);
    return builder.finish().modelWithOptions(.{ .frontier_index = frontier_index });
}

test "where lowers into masked and conditional reductions" {
    inline for ([_]WhereReductionKind{ .sum, .min, .max, .mean, .conditional }) |kind| {
        const Model = whereReductionModel(kind, null);
        var model = Model.init();
        var condition: [reduction_vector_length]bool = @splat(false);
        var input: [reduction_vector_length]f32 = @splat(-2);
        condition[0] = true;
        input[0] = 4;
        try model.copyInput(.condition, &condition);
        try model.copyInput(.input, &input);
        model.run();
        const expected: f32 = switch (kind) {
            .sum => 4,
            .min => 4,
            .max => 4,
            .mean => 4,
            .conditional => 4 + @as(f32, @floatFromInt(reduction_vector_length - 1)) * 2,
        };
        try std.testing.expectEqual(expected, model.outputView(0).contiguousSlice().?[0]);
        if (kind != .conditional) {
            for (Model.executable_candidate_frontier.candidates[0..Model.executable_candidate_frontier.count]) |candidate| {
                // A forced frontier choice must retain the same masked meaning.
                try std.testing.expect(candidate.?.origin != .reference);
            }
        }
    }
}

test "forced masked reduction candidates preserve the selected count" {
    const Default = whereReductionModel(.mean, null);
    const Forced = whereReductionModel(.mean, Default.executable_candidate_frontier.count - 1);
    var model = Forced.init();
    var condition: [reduction_vector_length]bool = @splat(false);
    condition[0] = true;
    var input: [reduction_vector_length]f32 = @splat(8);
    input[0] = 4;
    try model.copyInput(.condition, &condition);
    try model.copyInput(.input, &input);
    model.run();
    try std.testing.expectEqual(@as(f32, 4), model.outputView(0).contiguousSlice().?[0]);
}

test "nested where and comparison predicates fold into a reduction" {
    const Sources = enum(usize) { input, mask };
    const Model = comptime model_type: {
        var builder = zgc.DefinitionBuilder.init();
        const sources = builder.sources(Sources);
        const input = sources.input(.input, .f32, &.{reduction_vector_length});
        const mask = sources.input(.mask, .bool, &.{reduction_vector_length});
        const positive = builder.greaterThan(input, builder.scalar(.f32, 0));
        const signed = builder.where(positive, input, builder.neg(input));
        const selected = builder.where(mask, signed, builder.scalar(.f32, 2));
        builder.output(builder.sum(selected, .{ .axes = &.{0} }));
        break :model_type builder.finish().model();
    };
    try std.testing.expectEqual(@as(usize, 1), Model.executable.node_ct);
    const plan = Model.executable.nodes[0].?.op.compute.kernel.reduction;
    var selections: usize = 0;
    for (plan.region.expressions.instructions) |instruction| {
        if (instruction.operation == .select) selections += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), selections);

    var model = Model.init();
    var input: [reduction_vector_length]f32 = @splat(-3);
    var mask: [reduction_vector_length]bool = @splat(false);
    input[0] = 4;
    mask[0] = true;
    try model.copyInput(.input, &input);
    try model.copyInput(.mask, &mask);
    model.run();
    try std.testing.expectEqual(@as(f32, 4 + 2 * (reduction_vector_length - 1)), model.outputView(0).contiguousSlice().?[0]);
}

test "zero true branch reverses a reduction mask and empty mean returns zero" {
    const Model = comptime model_type: {
        var builder = zgc.DefinitionBuilder.init();
        const sources = builder.sources(WhereReductionSources);
        const condition = sources.input(.condition, .bool, &.{reduction_vector_length});
        const input = sources.input(.input, .f32, &.{reduction_vector_length});
        builder.output(builder.mean(builder.where(condition, builder.scalar(.f32, 0), input), .{ .axes = &.{0} }));
        break :model_type builder.finish().model();
    };

    var model = Model.init();
    var condition_storage: [reduction_vector_length]bool = @splat(true);
    const input_storage: [reduction_vector_length]f32 = @splat(3);
    try model.copyInput(.condition, &condition_storage);
    try model.copyInput(.input, &input_storage);
    model.run();
    try std.testing.expectEqual(@as(f32, 0), model.outputView(0).contiguousSlice().?[0]);

    condition_storage[reduction_vector_length - 1] = false;
    try model.copyInput(.condition, &condition_storage);
    model.run();
    try std.testing.expectEqual(@as(f32, 3), model.outputView(0).contiguousSlice().?[0]);
}

test "elementwise producer folds into its reduction" {
    const Model = ProducerReductionModel;

    try std.testing.expectEqual(.discovered, Model.selected_executable_candidate.fusion_regime.?);
    try std.testing.expectEqual(@as(usize, 2), Model.semantic_graph.node_ct);
    try std.testing.expectEqual(@as(usize, 1), Model.executable.node_ct);
    try std.testing.expect(!Model.executable.materialized[2]);
    try std.testing.expect(Model.memory_plan.tensor_regions[2] == null);
    switch (Model.executable.nodes[0].?.op.compute.kernel) {
        .reduction => |plan| {
            try std.testing.expectEqual(@as(usize, 1), plan.region.expressions.instructions.len);
            try std.testing.expectEqual(@as(usize, 1), plan.region.accumulators.len);
        },
        else => return error.TestUnexpectedResult,
    }

    var model = Model.init();
    try model.copyInput(.lhs, &.{ 1, 2, 3, 4, 5, 6 });
    try model.copyInput(.rhs, &.{ 2, 3, 4, 5, 6, 7 });
    model.run();
    try std.testing.expectEqualSlices(f32, &.{ 20, 92 }, model.outputView(0).contiguousSlice().?);
}

test "reductions over a shared domain execute as one multi-output region" {
    const Model = SiblingReductionModel;

    try std.testing.expectEqual(@as(usize, 3), Model.semantic_graph.node_ct);
    try std.testing.expectEqual(@as(usize, 1), Model.executable.node_ct);
    const invocation = Model.executable.nodes[0].?;
    try std.testing.expectEqual(@as(usize, 2), invocation.output_count);
    switch (invocation.op.compute.kernel) {
        .reduction => |plan| {
            try std.testing.expectEqual(@as(usize, 1), plan.region.expressions.instructions.len);
            try std.testing.expectEqual(@as(usize, 2), plan.region.accumulators.len);
            try std.testing.expectEqual(@as(usize, 2), plan.region.stores.len);
        },
        else => return error.TestUnexpectedResult,
    }

    var model = Model.init();
    try model.copyInput(.input, &.{ 1, 2, 3, 4, 5, 6 });
    try model.copySource(.weights, &.{ 1, 2, 3 });
    model.run();
    try std.testing.expectEqualSlices(f32, &.{ 6, 15 }, model.outputView(0).contiguousSlice().?);
    try std.testing.expectEqualSlices(f32, &.{ 9, 18 }, model.outputView(1).contiguousSlice().?);
}

test "fused reductions vectorize a contiguous reduced axis and handle its tail" {
    const Model = VectorReductionModel;
    const invocation = Model.executable.nodes[0].?;
    switch (invocation.op.compute.kernel) {
        .reduction => |plan| {
            if (reduction_vector_width > 1) {
                try std.testing.expectEqual(@as(?u8, 1), plan.traversal_plan.vector_axis);
                try std.testing.expectEqual(reduction_vector_width, plan.traversal_plan.vector_width);
            } else {
                try std.testing.expectEqual(@as(?u8, null), plan.traversal_plan.vector_axis);
                try std.testing.expectEqual(@as(usize, 1), plan.traversal_plan.vector_width);
            }
        },
        else => return error.TestUnexpectedResult,
    }

    var input: [2 * reduction_vector_length]f32 = undefined;
    for (0..reduction_vector_length) |index| {
        input[index] = 1;
        input[reduction_vector_length + index] = 2;
    }
    var weights: [reduction_vector_length]f32 = @splat(1);
    var model = Model.init();
    try model.copyInput(.input, &input);
    try model.copyInput(.weights, &weights);
    model.run();
    try std.testing.expectEqualSlices(
        f32,
        &.{ reduction_vector_length, 2 * reduction_vector_length },
        model.outputView(0).contiguousSlice().?,
    );
}

test "map traversal vectorizes a contiguous axis with broadcast inputs" {
    const invocation = BroadcastMapModel.executable.nodes[0].?;
    switch (invocation.op.compute.kernel) {
        .map => |plan| switch (plan.strategy) {
            .traversal => |traversal| {
                if (map_vector_width > 1) {
                    try std.testing.expectEqual(@as(?u8, 2), traversal.vector_axis);
                    try std.testing.expectEqual(map_vector_width, traversal.vector_width);
                }
            },
            else => return error.TestUnexpectedResult,
        },
        else => return error.TestUnexpectedResult,
    }

    var cells: [6]f32 = @splat(2);
    var channels: [map_vector_length]f32 = @splat(1);
    var expected: [6 * map_vector_length]f32 = @splat(3);
    var model = BroadcastMapModel.init();
    try model.copyInput(.cells, &cells);
    try model.copyInput(.channels, &channels);
    model.run();
    try std.testing.expectEqualSlices(f32, &expected, model.outputView(0).contiguousSlice().?);
}

test "mixed boolean and numeric pointwise expressions fuse without predicate storage" {
    const Model = TypedPointwiseModel;

    try std.testing.expectEqual(@as(usize, 3), Model.semantic_graph.node_ct);
    try std.testing.expectEqual(@as(usize, 1), Model.executable.node_ct);
    try std.testing.expect(!Model.executable.materialized[2]);
    try std.testing.expect(!Model.executable.materialized[3]);
    try std.testing.expect(Model.memory_plan.tensor_regions[2] == null);
    try std.testing.expect(Model.memory_plan.tensor_regions[3] == null);
    switch (Model.executable.nodes[0].?.op.compute.kernel) {
        .map => |plan| {
            const expression = plan.region.body.expression;
            try std.testing.expectEqual(@as(usize, 3), expression.instructions.len);
            try std.testing.expectEqual(zgc.memory.Dtype.bool, expression.instructions[0].dtype);
            try std.testing.expectEqual(zgc.memory.Dtype.f32, expression.instructions[1].dtype);
            try std.testing.expectEqual(zgc.memory.Dtype.f32, expression.instructions[2].dtype);
        },
        else => return error.TestUnexpectedResult,
    }

    var model = Model.init();
    try model.copyInput(.input, &.{ -1, 2, -3, 4, -5, 6, -7, 8 });
    try model.copyInput(.threshold, &.{ 0, 0, 0, 0, 0, 0, 0, 0 });
    model.run();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, model.outputView(0).contiguousSlice().?);
}
