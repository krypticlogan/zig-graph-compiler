const std = @import("std");
const internal = @import("zgc_internal");
const zgc = internal.api;
const traversal = internal.traversal;
const scheduling = internal.scheduling;

// Independent, deliberately simple scan implementation of the scheduling
// policy. Small DAGs compare optimized heap/edge traversal against these rules.
fn scanSchedule(comptime graph: anytype, comptime capacity: anytype, comptime kind: scheduling.Kind) scheduling.Schedule(capacity) {
    var result: scheduling.Schedule(capacity) = .{ .kind = kind };
    var emitted: [capacity.max_nodes]bool = @splat(false);
    var uses: [capacity.max_tensors]usize = @splat(0);
    var critical: [capacity.max_nodes]u128 = @splat(0);
    for (graph.input_refs[0..graph.input_ref_ct]) |input| uses[input.?] += 1;
    if (kind == .critical_path) {
        const order = scanSchedule(graph, capacity, .semantic);
        var reverse = order.node_count;
        while (reverse > 0) {
            reverse -= 1;
            const id = order.node_ids[reverse];
            const producer = graph.nodes[id].?;
            var own: u128 = 0;
            var successor: u128 = 0;
            for (0..traversal.outputCount(producer)) |output| {
                const tensor = traversal.outputTensor(&graph, producer, output);
                own += graph.tensors[tensor].?.shape.elementCount();
                for (graph.nodes[0..graph.node_ct], 0..) |entry, consumer_id| {
                    const consumer = entry.?;
                    for (0..consumer.input_count) |input| {
                        if (graph.input_refs[consumer.input_start + input].? == tensor) successor = @max(successor, critical[consumer_id]);
                    }
                }
            }
            critical[id] = own + successor;
        }
    }
    while (result.node_count < graph.node_ct) {
        var best: ?usize = null;
        var best_pressure: i128 = 0;
        for (graph.nodes[0..graph.node_ct], 0..) |entry, id| {
            if (emitted[id]) continue;
            const node = entry.?;
            var ready = true;
            var pressure: i128 = 0;
            for (0..node.input_count) |input| {
                const tensor = graph.input_refs[node.input_start + input].?;
                const info = graph.tensors[tensor].?;
                if (info.origin == .node and !emitted[info.origin.node]) ready = false;
                var output = false;
                for (graph.outputs[0..graph.output_ct]) |value| output = output or value.? == tensor;
                if (uses[tensor] == 1 and info.storage_tensor == tensor and !output) pressure += @intCast(info.shape.elementCount() * info.dtype.byteSize());
            }
            if (!ready) continue;
            for (0..traversal.outputCount(node)) |output| {
                const tensor = traversal.outputTensor(&graph, node, output);
                const info = graph.tensors[tensor].?;
                if (info.storage_tensor == tensor) pressure -= @intCast(info.shape.elementCount() * info.dtype.byteSize());
            }
            const better = if (best) |current| switch (kind) {
                .memory_pressure => pressure > best_pressure or (pressure == best_pressure and id < current),
                .critical_path => critical[id] > critical[current] or (critical[id] == critical[current] and id < current),
                .semantic, .existing => id < current,
            } else true;
            if (better) {
                best = id;
                best_pressure = pressure;
            }
        }
        const id = best orelse @compileError("test graph contains a cycle");
        result.node_ids[result.node_count] = id;
        result.node_count += 1;
        emitted[id] = true;
        const node = graph.nodes[id].?;
        for (0..node.input_count) |input| uses[graph.input_refs[node.input_start + input].?] -= 1;
    }
    return result;
}

test "heap policies match scan policies on branching DAGs with repeated inputs" {
    inline for (0..6) |seed| {
        const Executable = comptime blk: {
            var builder = zgc.DefinitionBuilder.init();
            const input = builder.sources(enum { input }).input(.input, .f32, &.{8});
            var values: [13]zgc.Value = undefined;
            values[0] = input;
            for (0..12) |index| {
                const lhs = values[(index * 3 + seed) % (index + 1)];
                const rhs = values[(index * 5 + seed + 1) % (index + 1)];
                values[index + 1] = if (index % 3 == 0) builder.neg(lhs) else builder.add(lhs, rhs);
                if (index % 4 == 0 or index == 11) builder.output(values[index + 1]);
            }
            break :blk builder.finish().executable();
        };
        const capacity = Executable.semantic_capacity;
        const graph = Executable.semantic_graph;
        const schedules = comptime scheduling.Scheduling(capacity).enumerateWithDependencies(graph, Executable.semantic_analysis_result.dependencies);
        inline for (1..4) |count| {
            const prefix = comptime scheduling.Scheduling(capacity).enumerateWithDependenciesLimited(graph, Executable.semantic_analysis_result.dependencies, count);
            inline for (prefix, 0..) |schedule, index| try std.testing.expectEqualDeep(schedules[index], schedule);
        }
        inline for (schedules) |schedule| {
            const expected = comptime scanSchedule(graph, capacity, schedule.kind);
            try std.testing.expectEqualSlices(usize, expected.node_ids[0..expected.node_count], schedule.node_ids[0..schedule.node_count]);
        }
        const program = Executable.reference_executable_candidate.executable;
        const physical_schedules = comptime scheduling.ExecutableScheduling(capacity).enumerate(program);
        inline for (1..4) |count| {
            const prefix = comptime scheduling.ExecutableScheduling(capacity).enumerateLimited(program, count);
            inline for (prefix, 0..) |schedule, index| try std.testing.expectEqualDeep(physical_schedules[index], schedule);
        }
        inline for (physical_schedules) |schedule| {
            const expected = comptime scanSchedule(program, capacity, schedule.kind);
            try std.testing.expectEqualSlices(usize, expected.node_ids[0..expected.node_count], schedule.node_ids[0..schedule.node_count]);
        }
    }
}

test "physical edge traversal handles multi-output invocations" {
    const Executable = comptime blk: {
        var builder = zgc.DefinitionBuilder.init();
        const input = builder.sources(enum { input }).input(.input, .f32, &.{8});
        builder.output(builder.sum(input, .{ .axes = &.{0} }));
        builder.output(builder.mean(input, .{ .axes = &.{0} }));
        break :blk builder.finish().executable();
    };
    const program = Executable.program;
    var multi_output = false;
    for (program.nodes[0..program.node_ct]) |entry| multi_output = multi_output or entry.?.output_count > 1;
    try std.testing.expect(multi_output);
    const schedules = comptime scheduling.ExecutableScheduling(Executable.internal_capacity).enumerate(program);
    inline for (schedules) |schedule| {
        const expected = comptime scanSchedule(program, Executable.internal_capacity, schedule.kind);
        try std.testing.expectEqualSlices(usize, expected.node_ids[0..expected.node_count], schedule.node_ids[0..schedule.node_count]);
    }
}

test "ready queue keeps tie order and updates waiting and ready nodes" {
    var queue: internal.ready_queue.ReadyQueue(5, i128) = .{ .scores = .{ -8, -8, -2, -20, -8 } };
    queue.push(4);
    queue.push(1);
    queue.push(0);
    queue.raise(3, 30); // A node whose dependencies are not ready yet.
    queue.push(3);
    try std.testing.expectEqual(@as(usize, 3), queue.pop());
    queue.raise(4, 20);
    try std.testing.expectEqual(@as(usize, 4), queue.pop());
    try std.testing.expectEqual(@as(usize, 0), queue.pop());
    try std.testing.expectEqual(@as(usize, 1), queue.pop());
    queue.push(2);
    try std.testing.expectEqual(@as(usize, 2), queue.pop());
    try std.testing.expectEqual(@as(usize, 0), queue.count);
}

fn checkRepresentationCosts(comptime Executable: type) !void {
    @setEvalBranchQuota(1_000_000);
    const capacity = Executable.semantic_capacity;
    const graph = Executable.semantic_graph;
    const facts = Executable.semantic_analysis_result;
    const analysis = internal.analysis;
    const representations = comptime internal.search.ExecutableSearch(capacity).composeRepresentations(
        graph,
        analysis.FusionAnalysis(capacity).analyze(graph, facts),
        analysis.LayoutAnalysis(capacity).analyze(graph, facts),
        analysis.RemapAnalysis(capacity).analyze(graph, facts),
        analysis.ContractionAnalysis(capacity).analyze(graph),
    );
    inline for (representations.values[0..representations.count]) |entry| {
        const representation = entry.?;
        const selected_graph = comptime analysis.LayoutAnalysis(capacity).applySelection(graph, representation.layout_regions);
        const expected = comptime analysis.LocalCost.estimate(capacity, selected_graph, representation.*);
        try std.testing.expectEqualDeep(expected, representation.local_cost);
    }
}

test "incremental representation costs match full graph estimates" {
    try checkRepresentationCosts(@import("fixtures/models.zig").MatmulExecutable);
    const Executable = comptime blk: {
        var builder = zgc.DefinitionBuilder.init();
        const input = builder.sources(enum { input }).input(.input, .f32, &.{ 4, 8 });
        const product = builder.mul(builder.neg(input), input);
        builder.output(builder.sum(product, .{ .axes = &.{1} }));
        builder.output(builder.mean(product, .{ .axes = &.{1} }));
        builder.output(builder.relu(builder.transpose(input, 0, 1)));
        break :blk builder.finish().executable();
    };
    try checkRepresentationCosts(Executable);
}

test "packed membership handles word boundaries and sparse ordered traversal" {
    const Set = internal.node_set.Set(130);
    const a = Set.init(&.{ 0, 63, 64, 129 });
    const b = Set.init(&.{ 1, 64, 128 });
    try std.testing.expect(a.intersects(&b));
    try std.testing.expect(!a.containsAll(&b));
    var combined = a;
    combined.include(&b);
    try std.testing.expect(combined.containsAll(&a));
    try std.testing.expect(combined.containsAll(&b));
    try std.testing.expect(a.eql(&a));
    try std.testing.expect(!a.eql(&combined));
    const expected = [_]usize{ 0, 1, 63, 64, 128, 129 };
    var iterator = combined.iterator();
    for (expected) |id| try std.testing.expectEqual(@as(?usize, id), iterator.next());
    try std.testing.expectEqual(@as(?usize, null), iterator.next());
    try std.testing.expect(!a.isSet(1)); // Union did not modify immutable input.
    const empty: internal.node_set.Set(0) = .{};
    var empty_iterator = empty.iterator();
    try std.testing.expectEqual(@as(?usize, null), empty_iterator.next());
}

test "region selection storage is linear and reduction lists are sparse" {
    const analysis = internal.analysis;
    try std.testing.expect(@sizeOf(analysis.FusionSelection(1024)) < 1024 * 64);
    try std.testing.expect(@sizeOf(analysis.RemapSelection(1024)) < 1024 * 32);
    try std.testing.expect(@sizeOf(analysis.Group(1024)) < 1024);
}

test "shared fusion anchors match a producer-chain reference" {
    const Executable = comptime blk: {
        var builder = zgc.DefinitionBuilder.init();
        const input = builder.sources(enum { input }).input(.input, .f32, &.{8});
        const shared = builder.neg(input);
        builder.output(builder.sum(builder.relu(shared), .{ .axes = &.{0} }));
        builder.output(builder.mean(shared, .{ .axes = &.{0} }));
        break :blk builder.finish().executable();
    };
    const graph = Executable.semantic_graph;
    const facts = Executable.semantic_analysis_result;
    inline for (0..graph.tensor_ct) |tensor| {
        var anchor = tensor;
        while (facts.use_counts[anchor] == 1 and !facts.is_output[anchor]) {
            const producer = switch (graph.tensors[anchor].?.origin) {
                .node => |id| id,
                else => break,
            };
            const operation = facts.pointwise[producer] orelse break;
            const node = graph.nodes[producer].?;
            if (!operation.isReductionCompatible() or node.input_count == 0) break;
            anchor = graph.input_refs[node.input_start].?;
        }
        try std.testing.expectEqual(anchor, facts.fusion_anchor[tensor]);
        const info = graph.tensors[tensor].?;
        try std.testing.expectEqual(@as(u128, info.shape.elementCount()) * info.dtype.byteSize(), facts.tensor_bytes[tensor]);
    }
}
