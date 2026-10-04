const Graph = @import("../core/graph.zig");
const Semantic = @import("../operations/semantic.zig");
const Execution = @import("../execution/execution.zig");
const Executable = @import("../execution/program.zig").Executable;

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

/// Produce the initial legal schedules explored for an unfused physical plan.
pub fn Scheduling(comptime capacity: Graph.Capacity) type {
    return struct {
        const SemanticGraph = Graph.Graph(capacity, Semantic.Op);
        const ScheduleType = Schedule(capacity);

        pub const variant_count = 3;

        pub fn enumerate(comptime graph: SemanticGraph) [variant_count]ScheduleType {
            return .{
                semanticSchedule(graph),
                prioritizedSchedule(graph, .memory_pressure),
                prioritizedSchedule(graph, .critical_path),
            };
        }

        fn semanticSchedule(comptime graph: SemanticGraph) ScheduleType {
            var schedule: ScheduleType = .{ .kind = .semantic };
            for (0..graph.node_ct) |node_id| schedule.node_ids[node_id] = node_id;
            schedule.node_count = graph.node_ct;
            return schedule;
        }

        fn prioritizedSchedule(comptime graph: SemanticGraph, comptime kind: Kind) ScheduleType {
            var schedule: ScheduleType = .{ .kind = kind };
            var emitted: [capacity.max_nodes]bool = @splat(false);
            var remaining_uses: [capacity.max_tensors]usize = @splat(0);
            var critical_work: [capacity.max_nodes]u128 = @splat(0);

            for (0..graph.input_ref_ct) |input_index| remaining_uses[graph.input_refs[input_index].?] += 1;
            computeCriticalWork(graph, &critical_work);

            while (schedule.node_count < graph.node_ct) {
                var best: ?usize = null;
                for (0..graph.node_ct) |node_id| {
                    if (emitted[node_id] or !isReady(graph, emitted, node_id)) continue;
                    const current = best orelse {
                        best = node_id;
                        continue;
                    };
                    if (prefer(graph, remaining_uses, critical_work, kind, node_id, current)) best = node_id;
                }

                const node_id = best orelse @compileError("semantic graph contains a dependency cycle");
                const node = graph.nodes[node_id].?;
                schedule.node_ids[schedule.node_count] = node_id;
                schedule.node_count += 1;
                emitted[node_id] = true;
                for (0..node.input_count) |input_index| {
                    remaining_uses[graph.input_refs[node.input_start + input_index].?] -= 1;
                }
            }
            return schedule;
        }

        fn isReady(comptime graph: SemanticGraph, emitted: [capacity.max_nodes]bool, node_id: usize) bool {
            const node = graph.nodes[node_id].?;
            for (0..node.input_count) |input_index| {
                const tensor_id = graph.input_refs[node.input_start + input_index].?;
                switch (graph.tensors[tensor_id].?.origin) {
                    .node => |producer| if (!emitted[producer]) return false,
                    .source, .literal => {},
                }
            }
            return true;
        }

        fn prefer(
            comptime graph: SemanticGraph,
            remaining_uses: [capacity.max_tensors]usize,
            critical_work: [capacity.max_nodes]u128,
            kind: Kind,
            candidate: usize,
            current: usize,
        ) bool {
            return switch (kind) {
                .memory_pressure => blk: {
                    const candidate_score = memoryPressureScore(graph, remaining_uses, candidate);
                    const current_score = memoryPressureScore(graph, remaining_uses, current);
                    break :blk candidate_score > current_score or
                        (candidate_score == current_score and candidate < current);
                },
                .critical_path => critical_work[candidate] > critical_work[current] or
                    (critical_work[candidate] == critical_work[current] and candidate < current),
                .semantic, .existing => unreachable,
            };
        }

        fn memoryPressureScore(graph: SemanticGraph, remaining_uses: [capacity.max_tensors]usize, node_id: usize) i128 {
            const node = graph.nodes[node_id].?;
            var released: i128 = 0;
            for (0..node.input_count) |input_index| {
                const tensor_id = graph.input_refs[node.input_start + input_index].?;
                const info = graph.tensors[tensor_id].?;
                if (remaining_uses[tensor_id] == 1 and info.storage_tensor == tensor_id and !isOutput(graph, tensor_id)) {
                    released += @intCast(byteCount(info));
                }
            }
            const result = graph.tensors[node.result].?;
            const created: i128 = if (result.storage_tensor == node.result) @intCast(byteCount(result)) else 0;
            return released - created;
        }

        fn computeCriticalWork(comptime graph: SemanticGraph, work: *[capacity.max_nodes]u128) void {
            var node_id = graph.node_ct;
            while (node_id > 0) {
                node_id -= 1;
                const node = graph.nodes[node_id].?;
                var successor_work: u128 = 0;
                for (node_id + 1..graph.node_ct) |consumer_id| {
                    const consumer = graph.nodes[consumer_id].?;
                    for (0..consumer.input_count) |input_index| {
                        if (graph.input_refs[consumer.input_start + input_index].? == node.result) {
                            successor_work = @max(successor_work, work[consumer_id]);
                        }
                    }
                }
                work[node_id] = @as(u128, @intCast(graph.tensors[node.result].?.shape.elementCount())) + successor_work;
            }
        }

        fn isOutput(graph: SemanticGraph, tensor_id: usize) bool {
            for (0..graph.output_ct) |output_index| if (graph.outputs[output_index].? == tensor_id) return true;
            return false;
        }

        fn byteCount(info: SemanticGraph.TensorInfo) usize {
            return info.shape.elementCount() * info.dtype.byteSize();
        }
    };
}

/// Reorder a completed physical plan without changing its invocations or
/// physical tensor choices.
pub fn ExecutableScheduling(comptime capacity: Graph.Capacity) type {
    return struct {
        const Program = Executable(capacity, Execution.Op);
        const ScheduleType = Schedule(capacity);

        pub const variant_count = 3;

        pub fn enumerate(comptime program: Program) [variant_count]ScheduleType {
            return .{
                existingSchedule(program),
                prioritizedSchedule(program, .memory_pressure),
                prioritizedSchedule(program, .critical_path),
            };
        }

        pub fn apply(comptime program: Program, comptime schedule: ScheduleType) Program {
            var reordered: Program = .init();
            for (0..program.tensor_ct) |tensor_id| {
                _ = reordered.insertTensor(program.tensors[tensor_id].?);
                reordered.materialized[tensor_id] = program.materialized[tensor_id];
            }
            for (0..program.sources.len) |source_index| {
                if (program.sources[source_index]) |source| reordered.insertSource(source_index, source);
            }

            inline for (0..schedule.node_count) |physical_node_id| {
                const original_node_id = schedule.node_ids[physical_node_id];
                const node = program.nodes[original_node_id].?;
                inline for (0..node.input_count) |input_index| {
                    reordered.insertInputRef(program.input_refs[node.input_start + input_index].?);
                }
                inline for (0..node.output_count) |output_index| {
                    const tensor_id = program.output_refs[node.output_start + output_index].?;
                    reordered.insertOutputRef(tensor_id);
                    reordered.tensors[tensor_id].?.origin = .{ .node = physical_node_id };
                }
                reordered.insertInvocation(.{
                    .op = node.op,
                    .input_start = reordered.input_ref_ct - node.input_count,
                    .input_count = node.input_count,
                    .output_start = reordered.output_ref_ct - node.output_count,
                    .output_count = node.output_count,
                });
            }
            for (0..program.output_ct) |output_index| reordered.insertOutput(program.outputs[output_index].?);
            return reordered;
        }

        fn existingSchedule(comptime program: Program) ScheduleType {
            var schedule: ScheduleType = .{ .kind = .existing };
            for (0..program.node_ct) |node_id| schedule.node_ids[node_id] = node_id;
            schedule.node_count = program.node_ct;
            return schedule;
        }

        fn prioritizedSchedule(comptime program: Program, comptime kind: Kind) ScheduleType {
            var schedule: ScheduleType = .{ .kind = kind };
            var emitted: [capacity.max_nodes]bool = @splat(false);
            var remaining_uses: [capacity.max_tensors]usize = @splat(0);
            var critical_work: [capacity.max_nodes]u128 = @splat(0);
            for (0..program.input_ref_ct) |input_index| remaining_uses[program.input_refs[input_index].?] += 1;
            computeCriticalWork(program, &critical_work);

            while (schedule.node_count < program.node_ct) {
                var best: ?usize = null;
                for (0..program.node_ct) |node_id| {
                    if (emitted[node_id] or !isReady(program, emitted, node_id)) continue;
                    const current = best orelse {
                        best = node_id;
                        continue;
                    };
                    if (prefer(program, remaining_uses, critical_work, kind, node_id, current)) best = node_id;
                }
                const node_id = best orelse @compileError("executable contains a dependency cycle");
                const node = program.nodes[node_id].?;
                schedule.node_ids[schedule.node_count] = node_id;
                schedule.node_count += 1;
                emitted[node_id] = true;
                for (0..node.input_count) |input_index| remaining_uses[program.input_refs[node.input_start + input_index].?] -= 1;
            }
            return schedule;
        }

        fn isReady(program: Program, emitted: [capacity.max_nodes]bool, node_id: usize) bool {
            const node = program.nodes[node_id].?;
            for (0..node.input_count) |input_index| {
                const tensor_id = program.input_refs[node.input_start + input_index].?;
                switch (program.tensors[tensor_id].?.origin) {
                    .node => |producer| if (!emitted[producer]) return false,
                    .source, .literal => {},
                }
            }
            return true;
        }

        fn prefer(program: Program, remaining_uses: [capacity.max_tensors]usize, critical_work: [capacity.max_nodes]u128, kind: Kind, candidate: usize, current: usize) bool {
            return switch (kind) {
                .memory_pressure => blk: {
                    const candidate_score = memoryPressureScore(program, remaining_uses, candidate);
                    const current_score = memoryPressureScore(program, remaining_uses, current);
                    break :blk candidate_score > current_score or
                        (candidate_score == current_score and candidate < current);
                },
                .critical_path => critical_work[candidate] > critical_work[current] or
                    (critical_work[candidate] == critical_work[current] and candidate < current),
                .semantic, .existing => unreachable,
            };
        }

        fn memoryPressureScore(program: Program, remaining_uses: [capacity.max_tensors]usize, node_id: usize) i128 {
            const node = program.nodes[node_id].?;
            var released: i128 = 0;
            for (0..node.input_count) |input_index| {
                const tensor_id = program.input_refs[node.input_start + input_index].?;
                const info = program.tensors[tensor_id].?;
                if (remaining_uses[tensor_id] == 1 and info.storage_tensor == tensor_id and !isOutput(program, tensor_id)) released += @intCast(byteCount(info));
            }
            var created: i128 = 0;
            for (0..node.output_count) |output_index| {
                const tensor_id = program.output_refs[node.output_start + output_index].?;
                const info = program.tensors[tensor_id].?;
                if (info.storage_tensor == tensor_id) created += @intCast(byteCount(info));
            }
            return released - created;
        }

        fn computeCriticalWork(program: Program, work: *[capacity.max_nodes]u128) void {
            var node_id = program.node_ct;
            while (node_id > 0) {
                node_id -= 1;
                const node = program.nodes[node_id].?;
                var own: u128 = 0;
                for (0..node.output_count) |output_index| own += program.tensors[program.output_refs[node.output_start + output_index].?].?.shape.elementCount();
                var successor_work: u128 = 0;
                for (node_id + 1..program.node_ct) |consumer_id| {
                    const consumer = program.nodes[consumer_id].?;
                    for (0..consumer.input_count) |input_index| {
                        for (0..node.output_count) |output_index| {
                            if (program.input_refs[consumer.input_start + input_index].? == program.output_refs[node.output_start + output_index].?) successor_work = @max(successor_work, work[consumer_id]);
                        }
                    }
                }
                work[node_id] = own + successor_work;
            }
        }

        fn isOutput(program: Program, tensor_id: usize) bool {
            for (0..program.output_ct) |output_index| if (program.outputs[output_index].? == tensor_id) return true;
            return false;
        }

        fn byteCount(info: Program.TensorInfo) usize {
            return info.shape.elementCount() * info.dtype.byteSize();
        }
    };
}

pub fn Reference(comptime capacity: Graph.Capacity) type {
    return struct {
        const SemanticGraph = Graph.Graph(capacity, Semantic.Op);
        const Program = Executable(capacity, Execution.Op);
        const ScheduleType = Schedule(capacity);

        pub fn plan(comptime graph: SemanticGraph, comptime schedule: ScheduleType) Program {
            var program: Program = .init();

            for (0..graph.tensor_ct) |tensor_id| {
                const info = graph.tensors[tensor_id].?;
                const inserted = program.insertTensor(info);
                if (inserted != tensor_id) @compileError("reference planning changed tensor order");
                program.materialized[tensor_id] = switch (info.origin) {
                    .source, .literal => true,
                    .node => false,
                };
            }
            for (0..graph.sources.len) |source_index| {
                if (graph.sources[source_index]) |source| program.insertSource(source_index, source);
            }

            inline for (0..schedule.node_count) |physical_node_id| {
                const semantic_node_id = schedule.node_ids[physical_node_id];
                const node = graph.nodes[semantic_node_id].?;
                inline for (0..node.input_count) |input_index| {
                    program.insertInputRef(graph.input_refs[node.input_start + input_index].?);
                }
                program.insertOutputRef(node.result);
                program.materialized[node.result] = true;
                program.tensors[node.result].?.origin = .{ .node = physical_node_id };
                program.insertInvocation(.{
                    .op = switch (node.op) {
                        .view => |view| .{ .view = view },
                        .compute => |compute| .{ .compute = .{ .direct = compute } },
                    },
                    .input_start = program.input_ref_ct - node.input_count,
                    .input_count = node.input_count,
                    .output_start = program.output_ref_ct - 1,
                    .output_count = 1,
                });
            }

            for (0..graph.output_ct) |output_index| program.insertOutput(graph.outputs[output_index].?);
            return program;
        }
    };
}
