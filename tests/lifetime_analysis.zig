const std = @import("std");
const zgc = @import("zgc");
const models = @import("fixtures/models.zig");

const Sources = enum(usize) { input };
const Definition = zgc.DefinitionBuilder;

test "lifetime analysis produces half-open intermediate intervals" {
    const lifetimes = models.ReuseModel.lifetime_analysis.tensor_lifetimes;

    try std.testing.expect(lifetimes[0].isPersistent());
    try std.testing.expectEqual(@as(usize, 0), lifetimes[1].begin_node);
    try std.testing.expectEqual(@as(?usize, 2), lifetimes[1].end_node_exclusive);
    try std.testing.expectEqual(@as(usize, 1), lifetimes[2].begin_node);
    try std.testing.expectEqual(@as(?usize, 3), lifetimes[2].end_node_exclusive);
    try std.testing.expect(lifetimes[3].isPersistent());
}

test "alias uses extend the lifetime of root storage" {
    const definition = comptime blk: {
        var builder = Definition.init();
        const builder_sources = builder.sources(Sources);
        const input = builder_sources.input(.input, .f32, &.{ 2, 2 });
        const root = builder.relu(input);
        const alias = builder.transpose(root, 0, 1);
        builder.output(builder.relu(alias));
        break :blk builder.finish();
    };
    const Model = definition.model();
    const graph = Model.executable;
    const lifetimes = Model.lifetime_analysis.tensor_lifetimes;

    try std.testing.expectEqual(graph.tensors[1].?.storage_tensor, graph.tensors[2].?.storage_tensor);
    try std.testing.expectEqual(lifetimes[1], lifetimes[2]);
    try std.testing.expectEqual(@as(usize, 0), lifetimes[1].begin_node);
    try std.testing.expectEqual(@as(?usize, 3), lifetimes[1].end_node_exclusive);
    try std.testing.expect(lifetimes[3].isPersistent());
}

test "output aliases make their root storage persistent" {
    const definition = comptime blk: {
        var builder = Definition.init();
        const builder_sources = builder.sources(Sources);
        const input = builder_sources.input(.input, .f32, &.{ 2, 2 });
        const root = builder.relu(input);
        builder.output(builder.transpose(root, 0, 1));
        break :blk builder.finish();
    };
    const Model = definition.model();
    const lifetimes = Model.lifetime_analysis.tensor_lifetimes;

    try std.testing.expect(lifetimes[1].isPersistent());
    try std.testing.expect(lifetimes[2].isPersistent());
    try std.testing.expectEqual(lifetimes[1], lifetimes[2]);
}
