const ProgramBuilder = @import("program_builder.zig").Builder;
const Graph = @import("../core/graph.zig");
const Tensor = @import("../core/tensor.zig");
const Semantic = @import("../operations/semantic.zig");
const Execution = @import("../execution/execution.zig");
const Traversal = @import("traversal.zig");
const ReadyQueue = @import("ready_queue.zig").ReadyQueue;
const ProgramFactory = @import("../execution/program.zig").Program;

/// The policy used to choose a legal topological schedule.
pub const Kind = enum {
    semantic,
    memory_pressure,
    critical_path,
    existing,
};

pub fn Schedule(comptime capacity: Graph.Capacity) type {
    return struct {
        kind: Kind,
        node_ids: [capacity.max_nodes]usize = @splat(0),
        node_count: usize = 0,
    };
}

/// Produce schedules from the already analyzed dependency graph when available.
pub fn Scheduling(comptime capacity: Graph.Capacity) type {
    return struct {
        const SemanticGraph = Graph.Graph(capacity, Semantic.Op);
        pub const variant_count = 3;

        pub fn enumerate(comptime graph: SemanticGraph) [variant_count]Schedule(capacity) {
            return enumerateWithDependencies(graph, Traversal.Dependencies(capacity).build(graph));
        }

        pub fn enumerateWithDependencies(comptime graph: SemanticGraph, comptime dependencies: Traversal.Dependencies(capacity)) [variant_count]Schedule(capacity) {
            return enumerateWithDependenciesLimited(graph, dependencies, variant_count);
        }

        pub fn enumerateWithDependenciesLimited(comptime graph: SemanticGraph, comptime dependencies: Traversal.Dependencies(capacity), comptime count: usize) [count]Schedule(capacity) {
            return Scheduler(capacity, SemanticGraph).enumerate(graph, dependencies, count);
        }
    };
}

/// Reorder a completed physical plan without changing its invocations or
/// physical tensor choices.
pub fn ExecutableScheduling(comptime capacity: Graph.Capacity) type {
    return struct {
        const Program = ProgramFactory(capacity, Execution.Op);
        const ScheduleType = Schedule(capacity);

        pub const variant_count = 3;

        pub fn enumerate(comptime program: Program) [variant_count]ScheduleType {
            return enumerateLimited(program, variant_count);
        }

        pub fn enumerateLimited(comptime program: Program, comptime count: usize) [count]ScheduleType {
            const dependencies = Traversal.Dependencies(capacity).build(program);
            return Scheduler(capacity, Program).enumerate(program, dependencies, count);
        }

        pub fn apply(comptime program: Program, comptime schedule: ScheduleType) Program {
            return comptime blk: {
                var identity = schedule.node_count == program.node_ct;
                for (schedule.node_ids[0..schedule.node_count], 0..) |id, index| identity = identity and id == index;
                if (identity) break :blk program;
                var builder: ProgramBuilder(capacity, Execution.Op) = .{};
                for (schedule.node_ids[0..schedule.node_count]) |id| {
                    const node = program.nodes[id].?;
                    builder.append(
                        node.op,
                        program.input_refs[node.input_start..][0..node.input_count],
                        program.output_refs[node.output_start..][0..node.output_count],
                    );
                }
                break :blk builder.finish(program);
            };
        }
    };
}

/// One edge index and one reverse critical-work traversal serve all policies.
fn Scheduler(comptime capacity: Graph.Capacity, comptime Representation: type) type {
    return struct {
        const Dependencies = Traversal.Dependencies(capacity);
        const ScheduleType = Schedule(capacity);

        fn enumerate(comptime graph: Representation, comptime dependencies: Dependencies, comptime count: usize) [count]ScheduleType {
            if (count < 1 or count > 3) @compileError("schedule allowance must be between one and three");
            const baseline: ScheduleType = .{
                .kind = if (@hasField(Representation, "output_refs")) .existing else .semantic,
                .node_ids = dependencies.topological_order,
                .node_count = dependencies.node_count,
            };
            if (count == 1) return .{baseline};

            var uses: [capacity.max_tensors]usize = @splat(0);
            var is_output: [capacity.max_tensors]bool = @splat(false);
            for (graph.outputs[0..graph.output_ct]) |output| is_output[output.?] = true;
            var released: [capacity.max_tensors]i128 = @splat(0);
            for (0..graph.tensor_ct) |id| {
                uses[id] = dependencies.consumer_offsets[id + 1] - dependencies.consumer_offsets[id];
                const info = graph.tensors[id].?;
                if (info.storage_tensor == id and !is_output[id]) released[id] = @intCast(info.shape.elementCount() * info.dtype.byteSize());
            }
            var pressure: [capacity.max_nodes]i128 = @splat(0);
            for (graph.nodes[0..graph.node_ct], 0..) |entry, id| {
                const node = entry.?;
                for (0..node.input_count) |input| {
                    const tensor_id = graph.input_refs[node.input_start + input].?;
                    if (uses[tensor_id] == 1) pressure[id] += released[tensor_id];
                }
                for (0..Traversal.outputCount(node)) |output| {
                    const tensor_id = Traversal.outputTensor(&graph, node, output);
                    const info = graph.tensors[tensor_id].?;
                    if (info.storage_tensor == tensor_id) pressure[id] -= @intCast(info.shape.elementCount() * info.dtype.byteSize());
                }
            }
            const memory = prioritized(graph, &dependencies, .memory_pressure, pressure, uses, &released);
            if (count == 2) return .{ baseline, memory };
            var critical: [capacity.max_nodes]u128 = @splat(0);
            var reverse = dependencies.node_count;
            while (reverse > 0) {
                reverse -= 1;
                const id = dependencies.topological_order[reverse];
                const node = graph.nodes[id].?;
                var own: u128 = 0;
                var successor: u128 = 0;
                for (0..Traversal.outputCount(node)) |output| {
                    const tensor_id = Traversal.outputTensor(&graph, node, output);
                    own += graph.tensors[tensor_id].?.shape.elementCount();
                    for (dependencies.consumersOf(tensor_id)) |consumer| successor = @max(successor, critical[consumer.node]);
                }
                critical[id] = own + successor;
            }

            return .{ baseline, memory, prioritized(graph, &dependencies, .critical_path, critical, uses, &released) };
        }

        fn prioritized(
            comptime graph: Representation,
            dependencies: *const Dependencies,
            comptime kind: Kind,
            scores: anytype,
            initial_uses: [capacity.max_tensors]usize,
            released: *const [capacity.max_tensors]i128,
        ) ScheduleType {
            const Score = if (kind == .critical_path) u128 else i128;
            var ready: ReadyQueue(capacity.max_nodes, Score) = .{ .scores = scores };
            var incoming = dependencies.incoming;
            var uses = initial_uses;
            for (0..graph.node_ct) |id| if (incoming[id] == 0) {
                ready.push(id);
            };
            var schedule: ScheduleType = .{ .kind = kind };
            while (schedule.node_count < graph.node_ct) {
                if (ready.count == 0) @compileError("graph contains a dependency cycle");
                const id = ready.pop();
                schedule.node_ids[schedule.node_count] = id;
                schedule.node_count += 1;
                const node = graph.nodes[id].?;
                if (kind == .memory_pressure) {
                    for (0..node.input_count) |input| {
                        const tensor_id = graph.input_refs[node.input_start + input].?;
                        uses[tensor_id] -= 1;
                        // Only the transition to one remaining use changes the
                        // pressure score. Touch those consumers, not all nodes.
                        if (uses[tensor_id] == 1 and released[tensor_id] != 0) {
                            for (dependencies.consumersOf(tensor_id)) |consumer| ready.raise(consumer.node, released[tensor_id]);
                        }
                    }
                }
                for (0..Traversal.outputCount(node)) |output| {
                    const tensor_id = Traversal.outputTensor(&graph, node, output);
                    for (dependencies.consumersOf(tensor_id)) |consumer| {
                        incoming[consumer.node] -= 1;
                        if (incoming[consumer.node] == 0) ready.push(consumer.node);
                    }
                }
            }
            return schedule;
        }
    };
}

pub fn Reference(comptime capacity: Graph.Capacity) type {
    return struct {
        const SemanticGraph = Graph.Graph(capacity, Semantic.Op);
        const Program = ProgramFactory(capacity, Execution.Op);
        const ScheduleType = Schedule(capacity);

        pub fn plan(comptime graph: SemanticGraph, comptime schedule: ScheduleType) Program {
            return comptime blk: {
                var builder: ProgramBuilder(capacity, Execution.Op) = .{};
                for (schedule.node_ids[0..schedule.node_count]) |semantic_node_id| {
                    const node = graph.nodes[semantic_node_id].?;
                    const op: Execution.Op = switch (node.op) {
                        .view => |view| .{ .view = view },
                        .compute => |compute| .{ .compute = .{ .direct = compute } },
                    };
                    builder.append(op, graph.input_refs[node.input_start..][0..node.input_count], &.{node.result});
                }
                break :blk builder.finish(graph);
            };
        }
    };
}

test "semantic schedules follow dependencies rather than node identifiers" {
    const std = @import("std");
    const capacity: Graph.Capacity = .{
        .max_nodes = 2,
        .max_input_refs = 2,
        .max_tensors = 3,
        .max_outputs = 1,
        .max_sources = 1,
        .max_rank = 1,
    };
    const SemanticGraph = Graph.Graph(capacity, Semantic.Op);
    const shape = Tensor.Shape(1).init(&.{4});
    const layout = Tensor.Layout(1).contiguous(shape);
    const graph: SemanticGraph = comptime blk: {
        var value: SemanticGraph = .init();
        _ = value.insertTensor(.{
            .dtype = .f32,
            .shape = shape,
            .layout = layout,
            .storage_tensor = 0,
            .origin = .{ .source = 0 },
        });
        _ = value.insertTensor(.{
            .dtype = .f32,
            .shape = shape,
            .layout = layout,
            .storage_tensor = 1,
            .origin = .{ .node = 1 },
        });
        _ = value.insertTensor(.{
            .dtype = .f32,
            .shape = shape,
            .layout = layout,
            .storage_tensor = 2,
            .origin = .{ .node = 0 },
        });
        value.insertRef(1);
        value.insertNode(.{
            .op = .{ .compute = .relu },
            .input_start = 0,
            .input_count = 1,
            .result = 2,
        });
        value.insertRef(0);
        value.insertNode(.{
            .op = .{ .compute = .relu },
            .input_start = 1,
            .input_count = 1,
            .result = 1,
        });
        value.insertOutput(2);
        break :blk value;
    };

    const schedules = Scheduling(capacity).enumerate(graph);
    inline for (schedules) |schedule| {
        try std.testing.expectEqualSlices(usize, &.{ 1, 0 }, schedule.node_ids[0..schedule.node_count]);
    }
}
