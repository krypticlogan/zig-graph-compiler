const std = @import("std");
const zgc = @import("zgc");
const models = @import("fixtures/models.zig");

test "memory plan reuses an expired intermediate region" {
    const regions = models.ReuseModel.memory_plan.tensor_regions;

    try std.testing.expectEqual(regions[1], regions[3]);
    try std.testing.expect(regions[1].?.offset != regions[2].?.offset);
    try std.testing.expectEqual(@as(usize, 3 * 4 * @sizeOf(f32)), models.ReuseModel.memory_plan.byte_count);
}

test "persistent outputs retain distinct regions" {
    const Sources = enum(usize) { input };
    const Definition = zgc.DefinitionBuilder(Sources, .{
        .max_rank = 1,
        .max_nodes = 2,
        .max_tensors = 3,
        .max_input_refs = 2,
        .max_outputs = 2,
    });
    const definition = comptime blk: {
        var builder = Definition.init();
        const input = builder.input(.input, .f32, &.{4});
        builder.output(builder.relu(input));
        builder.output(builder.relu(input));
        break :blk builder.finish();
    };
    const Model = definition.modelWith(&.{
        .{ .source = .input, .binding = zgc.memory.Source.bound },
    });
    const regions = Model.memory_plan.tensor_regions;

    try std.testing.expect(regions[1].?.offset != regions[2].?.offset);
    try std.testing.expectEqual(@as(usize, 2 * 4 * @sizeOf(f32)), Model.memory_plan.byte_count);
}

test "memory plan splits and coalesces free spans" {
    const Sources = enum(usize) { input };
    const Definition = zgc.DefinitionBuilder(Sources, .{
        .max_rank = 1,
        .max_nodes = 4,
        .max_tensors = 5,
        .max_input_refs = 4,
        .max_outputs = 2,
    });
    const definition = comptime blk: {
        var builder = Definition.init();
        const input = builder.input(.input, .f32, &.{8});
        const wide_temporary = builder.copy(input);
        const scalar_temporary = builder.sum(wide_temporary, .{ .axes = &.{0} });
        builder.output(builder.relu(scalar_temporary));
        builder.output(builder.relu(input));
        break :blk builder.finish();
    };
    const Model = definition.modelWith(&.{
        .{ .source = .input, .binding = zgc.memory.Source.bound },
    });
    const regions = Model.memory_plan.tensor_regions;

    try std.testing.expectEqual(@as(usize, 0), regions[1].?.offset);
    try std.testing.expectEqual(@as(usize, 32), regions[2].?.offset);
    try std.testing.expectEqual(@as(usize, 0), regions[3].?.offset);
    try std.testing.expectEqual(@as(usize, 4), regions[4].?.offset);
    try std.testing.expectEqual(@as(usize, 36), Model.memory_plan.byte_count);
}

test "memory plan grows when no free span fits" {
    const Sources = enum(usize) { small_input, large_input };
    const Definition = zgc.DefinitionBuilder(Sources, .{
        .max_rank = 1,
        .max_nodes = 4,
        .max_tensors = 6,
        .max_input_refs = 4,
        .max_outputs = 2,
    });
    const definition = comptime blk: {
        var builder = Definition.init();
        const small_input = builder.input(.small_input, .f32, &.{8});
        const large_input = builder.input(.large_input, .f32, &.{10});
        const wide_temporary = builder.copy(small_input);
        const scalar_temporary = builder.sum(wide_temporary, .{ .axes = &.{0} });
        builder.output(builder.relu(scalar_temporary));
        builder.output(builder.relu(large_input));
        break :blk builder.finish();
    };
    const Model = definition.modelWith(&.{
        .{ .source = .small_input, .binding = zgc.memory.Source.bound },
        .{ .source = .large_input, .binding = zgc.memory.Source.bound },
    });
    const regions = Model.memory_plan.tensor_regions;

    try std.testing.expectEqual(@as(usize, 0), regions[4].?.offset);
    try std.testing.expectEqual(@as(usize, 36), regions[5].?.offset);
    try std.testing.expectEqual(@as(usize, 76), Model.memory_plan.byte_count);
}
