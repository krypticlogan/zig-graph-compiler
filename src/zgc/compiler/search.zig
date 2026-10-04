const Execution = @import("../execution/execution.zig");
const Executable = @import("../execution/program.zig").Executable;
const Graph = @import("../core/graph.zig");
const Source = @import("../storage/source.zig");
const Storage = @import("../storage/storage.zig");
const Tensor = @import("../core/tensor.zig");
const Analysis = @import("analysis.zig");
const Scheduling = @import("scheduling.zig");
const validation = @import("validation.zig");
const Planning = @import("planning/root.zig");

pub fn SearchResult(comptime capacity: Graph.Capacity) type {
    return struct {
        reference: PlanCandidate(capacity),
        frontier: CandidateSet(capacity),
        generated_count: usize,
        fusion_candidate_count: usize,
        layout_candidate_count: usize,
        remap_candidate_count: usize,

        pub fn selected(comptime result: @This()) PlanCandidate(capacity) {
            return result.frontier.best();
        }
    };
}

/// Half-open execution interval for one tensor's backing storage.
/// A null end marks storage that must remain reserved for the model lifetime.
pub const Lifetime = struct {
    begin_node: usize,
    end_node_exclusive: ?usize,

    pub fn isPersistent(lifetime: Lifetime) bool {
        return lifetime.end_node_exclusive == null;
    }
};

pub fn LifetimeSet(comptime tensor_count: usize) type {
    return struct {
        tensor_lifetimes: [tensor_count]Lifetime,
    };
}

/// Computes storage lifetimes for a validated, sequential executable program.
/// Aliasing tensors receive the consolidated lifetime of their storage root.
pub fn LifetimeAnalysis() type {
    return struct {
        pub fn analyze(
            comptime Validated: type,
        ) LifetimeSet(Validated.graph.tensor_ct) {
            const graph = Validated.graph;
            var storage_lifetimes: [graph.tensor_ct]?Lifetime = @splat(null);

            for (0..graph.tensor_ct) |tensor_id| {
                if (comptime @hasField(@TypeOf(graph), "materialized")) {
                    if (!graph.materialized[tensor_id]) continue;
                }
                const info = graph.tensors[tensor_id].?;
                if (info.storage_tensor != tensor_id) continue;

                storage_lifetimes[tensor_id] = switch (info.origin) {
                    .source => .{
                        .begin_node = 0,
                        .end_node_exclusive = null,
                    },
                    .node => |node_id| .{
                        .begin_node = node_id,
                        .end_node_exclusive = node_id + 1,
                    },
                    .literal => .{
                        .begin_node = 0,
                        .end_node_exclusive = null,
                    },
                };
            }

            for (0..graph.node_ct) |node_id| {
                const node = graph.nodes[node_id].?;
                for (0..node.input_count) |input_index| {
                    const tensor_id = graph.input_refs[node.input_start + input_index].?;
                    const storage_tensor = graph.tensors[tensor_id].?.storage_tensor;
                    if (storage_lifetimes[storage_tensor].?.end_node_exclusive) |end_node| {
                        storage_lifetimes[storage_tensor].?.end_node_exclusive = @max(
                            end_node,
                            node_id + 1,
                        );
                    }
                }
            }

            for (0..graph.output_ct) |output_index| {
                const tensor_id = graph.outputs[output_index].?;
                const storage_tensor = graph.tensors[tensor_id].?.storage_tensor;
                storage_lifetimes[storage_tensor].?.end_node_exclusive = null;
            }

            var result: LifetimeSet(graph.tensor_ct) = undefined;
            for (0..graph.tensor_ct) |tensor_id| {
                if (comptime @hasField(@TypeOf(graph), "materialized")) {
                    if (!graph.materialized[tensor_id]) {
                        result.tensor_lifetimes[tensor_id] = .{
                            .begin_node = 0,
                            .end_node_exclusive = 0,
                        };
                        continue;
                    }
                }
                const storage_tensor = graph.tensors[tensor_id].?.storage_tensor;
                result.tensor_lifetimes[tensor_id] = storage_lifetimes[storage_tensor].?;
            }
            return result;
        }
    };
}

/// A physical layout condition advertised by an implementation candidate.
pub fn LayoutRequirement(comptime max_rank: usize) type {
    return struct {
        tensor: usize,
        layout: ?Tensor.Layout(max_rank) = null,
        contiguous: bool = false,
    };
}

/// The physical representation produced by an implementation candidate.
pub fn LayoutResult(comptime max_rank: usize) type {
    return struct {
        tensor: usize,
        layout: Tensor.Layout(max_rank),
    };
}

pub const ConversionCost = struct {
    bytes_read: usize = 0,
    bytes_written: usize = 0,
    estimated_work: u128 = 0,
};

pub const MemoryTraffic = struct {
    bytes_read: u128 = 0,
    bytes_written: u128 = 0,
};

pub const PlanCost = struct {
    estimated_runtime_work: u128,
    memory_traffic: MemoryTraffic = .{},
    peak_memory_bytes: usize,
    persistent_memory_bytes: usize,
    scratch_memory_bytes: usize,
    code_size_units: usize,
    conversion: ConversionCost = .{},

    pub fn dominates(lhs: PlanCost, rhs: PlanCost) bool {
        const no_worse = lhs.estimated_runtime_work <= rhs.estimated_runtime_work and
            lhs.memory_traffic.bytes_read <= rhs.memory_traffic.bytes_read and
            lhs.memory_traffic.bytes_written <= rhs.memory_traffic.bytes_written and
            lhs.peak_memory_bytes <= rhs.peak_memory_bytes and
            lhs.persistent_memory_bytes <= rhs.persistent_memory_bytes and
            lhs.scratch_memory_bytes <= rhs.scratch_memory_bytes and
            lhs.code_size_units <= rhs.code_size_units and
            lhs.conversion.estimated_work <= rhs.conversion.estimated_work and
            lhs.conversion.bytes_read <= rhs.conversion.bytes_read and
            lhs.conversion.bytes_written <= rhs.conversion.bytes_written;
        const strictly_better = lhs.estimated_runtime_work < rhs.estimated_runtime_work or
            lhs.memory_traffic.bytes_read < rhs.memory_traffic.bytes_read or
            lhs.memory_traffic.bytes_written < rhs.memory_traffic.bytes_written or
            lhs.peak_memory_bytes < rhs.peak_memory_bytes or
            lhs.persistent_memory_bytes < rhs.persistent_memory_bytes or
            lhs.scratch_memory_bytes < rhs.scratch_memory_bytes or
            lhs.code_size_units < rhs.code_size_units or
            lhs.conversion.estimated_work < rhs.conversion.estimated_work or
            lhs.conversion.bytes_read < rhs.conversion.bytes_read or
            lhs.conversion.bytes_written < rhs.conversion.bytes_written;
        return no_worse and strictly_better;
    }

    pub fn preferredTo(lhs: PlanCost, rhs: PlanCost) bool {
        if (lhs.estimated_runtime_work != rhs.estimated_runtime_work) return lhs.estimated_runtime_work < rhs.estimated_runtime_work;
        const lhs_traffic = lhs.memory_traffic.bytes_read + lhs.memory_traffic.bytes_written;
        const rhs_traffic = rhs.memory_traffic.bytes_read + rhs.memory_traffic.bytes_written;
        if (lhs_traffic != rhs_traffic) return lhs_traffic < rhs_traffic;
        if (lhs.memory_traffic.bytes_written != rhs.memory_traffic.bytes_written) return lhs.memory_traffic.bytes_written < rhs.memory_traffic.bytes_written;
        if (lhs.peak_memory_bytes != rhs.peak_memory_bytes) return lhs.peak_memory_bytes < rhs.peak_memory_bytes;
        if (lhs.persistent_memory_bytes != rhs.persistent_memory_bytes) return lhs.persistent_memory_bytes < rhs.persistent_memory_bytes;
        if (lhs.scratch_memory_bytes != rhs.scratch_memory_bytes) return lhs.scratch_memory_bytes < rhs.scratch_memory_bytes;
        if (lhs.conversion.estimated_work != rhs.conversion.estimated_work) return lhs.conversion.estimated_work < rhs.conversion.estimated_work;
        if (lhs.code_size_units != rhs.code_size_units) return lhs.code_size_units < rhs.code_size_units;
        if (lhs.conversion.bytes_read != rhs.conversion.bytes_read) return lhs.conversion.bytes_read < rhs.conversion.bytes_read;
        return lhs.conversion.bytes_written < rhs.conversion.bytes_written;
    }
};

pub const Origin = enum { reference, planned };
pub const max_candidates = 16;

pub fn PlanCandidate(comptime capacity: Graph.Capacity) type {
    return struct {
        executable: Executable(capacity, Execution.Op),
        origin: Origin,
        fusion_regime: ?Analysis.FusionRegime = null,
        layout_regime: ?Analysis.LayoutRegime = null,
        remap_regime: ?Analysis.RemapRegime = null,
        schedule: Scheduling.Kind,
        cost: PlanCost,
    };
}

pub fn CandidateSet(comptime capacity: Graph.Capacity) type {
    return struct {
        const Self = @This();
        pub const Candidate = PlanCandidate(capacity);

        candidates: [max_candidates]?Candidate = @splat(null),
        count: usize = 0,

        /// Insert one completed candidate while retaining only the Pareto
        /// frontier. Equal-cost candidates remain available for deterministic
        /// final selection.
        pub fn insert(set: *Self, candidate: Candidate) void {
            for (set.candidates[0..set.count]) |existing| {
                if (existing.?.cost.dominates(candidate.cost)) return;
            }

            var index: usize = 0;
            while (index < set.count) {
                if (candidate.cost.dominates(set.candidates[index].?.cost)) {
                    var move = index;
                    while (move + 1 < set.count) : (move += 1) set.candidates[move] = set.candidates[move + 1];
                    set.count -= 1;
                    set.candidates[set.count] = null;
                } else {
                    index += 1;
                }
            }

            if (set.count == max_candidates) @compileError("executable candidate capacity exceeded");
            set.candidates[set.count] = candidate;
            set.count += 1;
        }

        pub fn best(comptime set: Self) Candidate {
            if (set.count == 0) @compileError("executable search produced no candidates");
            var best_index: usize = 0;
            for (1..set.count) |index| {
                const candidate = set.candidates[index].?;
                const current = set.candidates[best_index].?;
                if (candidate.cost.preferredTo(current.cost) or
                    (costsEqual(candidate.cost, current.cost) and tieBreak(candidate, current)))
                {
                    best_index = index;
                }
            }
            return set.candidates[best_index].?;
        }

        fn costsEqual(lhs: PlanCost, rhs: PlanCost) bool {
            return !lhs.preferredTo(rhs) and !rhs.preferredTo(lhs);
        }

        fn tieBreak(candidate: Candidate, current: Candidate) bool {
            if (@intFromEnum(candidate.schedule) != @intFromEnum(current.schedule)) {
                return @intFromEnum(candidate.schedule) < @intFromEnum(current.schedule);
            }
            if (layoutOrder(candidate.layout_regime) != layoutOrder(current.layout_regime)) {
                return layoutOrder(candidate.layout_regime) < layoutOrder(current.layout_regime);
            }
            if (fusionOrder(candidate.fusion_regime) != fusionOrder(current.fusion_regime)) {
                return fusionOrder(candidate.fusion_regime) < fusionOrder(current.fusion_regime);
            }
            if (remapOrder(candidate.remap_regime) != remapOrder(current.remap_regime)) {
                return remapOrder(candidate.remap_regime) < remapOrder(current.remap_regime);
            }
            return @intFromEnum(candidate.origin) < @intFromEnum(current.origin);
        }

        fn layoutOrder(value: ?Analysis.LayoutRegime) usize {
            return if (value) |present| switch (present) {
                .propagated => 1,
                .canonical => 2,
            } else 0;
        }

        fn fusionOrder(value: ?Analysis.FusionRegime) usize {
            return if (value) |present| switch (present) {
                .discovered => 1,
                .unfused => 2,
            } else 0;
        }

        fn remapOrder(value: ?Analysis.RemapRegime) usize {
            return if (value) |present| switch (present) {
                .composed => 1,
                .direct => 2,
            } else 0;
        }
    };
}

/// Build a bounded set of complete executable programs, perform exact lifetime
/// and storage planning for each, and retain their Pareto frontier.
pub fn ExecutableSearch(comptime capacity: Graph.Capacity) type {
    return struct {
        const Program = Executable(capacity, Execution.Op);
        const SemanticFacts = Analysis.Facts(capacity);
        const FusionCandidates = Analysis.FusionCandidates(capacity.max_nodes);
        const RemapCandidates = Analysis.RemapCandidates(capacity.max_nodes);
        const CandidateType = PlanCandidate(capacity);
        const Result = SearchResult(capacity);

        pub fn search(
            comptime SourceKey: type,
            comptime SemanticValidated: type,
            comptime semantic_analysis: SemanticFacts,
            comptime fusion_candidates: FusionCandidates,
            comptime layout_candidates: Analysis.LayoutCandidates,
            comptime remap_candidates: RemapCandidates,
            comptime source_configuration: anytype,
        ) Result {
            var result: Result = undefined;
            result.frontier = .{};
            result.generated_count = 0;
            result.fusion_candidate_count = fusion_candidates.count;
            result.layout_candidate_count = layout_candidates.count;
            result.remap_candidate_count = remap_candidates.count;

            const reference_schedules = Scheduling.Scheduling(capacity).enumerate(SemanticValidated.graph);
            inline for (reference_schedules, 0..) |schedule, schedule_index| {
                const executable = Scheduling.Reference(capacity).plan(SemanticValidated.graph, schedule);
                const candidate = complete(
                    SourceKey,
                    executable,
                    .reference,
                    null,
                    null,
                    null,
                    schedule.kind,
                    source_configuration,
                );
                if (schedule_index == 0) result.reference = candidate;
                result.frontier.insert(candidate);
                result.generated_count += 1;
            }

            inline for (0..layout_candidates.count) |layout_index| {
                const layout_regime = layout_candidates.values[layout_index];
                const layout_graph = Analysis.LayoutAnalysis(capacity).apply(
                    SemanticValidated.graph,
                    semantic_analysis,
                    layout_regime,
                );
                const LayoutValidated = switch (layout_regime) {
                    .canonical => SemanticValidated,
                    .propagated => validation.Validation(capacity).validate(layout_graph),
                };

                inline for (0..fusion_candidates.count) |fusion_index| {
                    const fusion_candidate = fusion_candidates.values[fusion_index].?;
                    inline for (0..remap_candidates.count) |remap_index| {
                        const remap_candidate = remap_candidates.values[remap_index].?;
                        const base_executable = Planning.Lowering(capacity).lower(
                            LayoutValidated.graph,
                            fusion_candidate.regions,
                            remap_candidate.regions,
                        );
                        const schedules = Scheduling.ExecutableScheduling(capacity).enumerate(base_executable);
                        inline for (schedules) |schedule| {
                            const executable = Scheduling.ExecutableScheduling(capacity).apply(base_executable, schedule);
                            result.frontier.insert(complete(
                                SourceKey,
                                executable,
                                .planned,
                                fusion_candidate.regime,
                                layout_regime,
                                remap_candidate.regime,
                                schedule.kind,
                                source_configuration,
                            ));
                            result.generated_count += 1;
                        }
                    }
                }
            }

            return result;
        }

        fn complete(
            comptime SourceKey: type,
            comptime executable: Program,
            comptime origin: Origin,
            comptime fusion_regime: ?Analysis.FusionRegime,
            comptime layout_regime: ?Analysis.LayoutRegime,
            comptime remap_regime: ?Analysis.RemapRegime,
            comptime schedule: Scheduling.Kind,
            comptime source_configuration: anytype,
        ) CandidateType {
            const Validated = validation.FinalValidation(capacity).validate(executable);
            const lifetimes = LifetimeAnalysis().analyze(Validated);
            const SourcePlan = Source.Plan(SourceKey, capacity, executable, source_configuration);
            const MemoryPlan = Storage.MemoryPlan(capacity, executable, lifetimes, SourcePlan);

            return .{
                .executable = executable,
                .origin = origin,
                .fusion_regime = fusion_regime,
                .layout_regime = layout_regime,
                .remap_regime = remap_regime,
                .schedule = schedule,
                .cost = .{
                    .estimated_runtime_work = estimateRuntime(executable),
                    .memory_traffic = estimateMemoryTraffic(executable),
                    .peak_memory_bytes = MemoryPlan.byte_count,
                    .persistent_memory_bytes = persistentBytes(executable, lifetimes, SourcePlan),
                    .scratch_memory_bytes = 0,
                    .code_size_units = codeSize(executable),
                },
            };
        }

        fn estimateRuntime(comptime program: Program) u128 {
            var work: u128 = 0;
            for (0..program.node_ct) |node_id| {
                const node = program.nodes[node_id].?;
                switch (node.op) {
                    .view => continue,
                    .compute => |compute| {
                        work += invocationArithmetic(program, node, compute);
                        for (0..node.input_count) |input_index| {
                            const tensor_id = program.input_refs[node.input_start + input_index].?;
                            work += program.tensors[tensor_id].?.shape.elementCount();
                        }
                        for (0..node.output_count) |output_index| {
                            const tensor_id = program.output_refs[node.output_start + output_index].?;
                            work += program.tensors[tensor_id].?.shape.elementCount();
                        }
                    },
                }
            }
            return work;
        }

        fn estimateMemoryTraffic(comptime program: Program) MemoryTraffic {
            var traffic: MemoryTraffic = .{};
            for (0..program.node_ct) |node_id| {
                const node = program.nodes[node_id].?;
                switch (node.op) {
                    .view => continue,
                    .compute => {
                        for (0..node.input_count) |input_index| {
                            const tensor_id = program.input_refs[node.input_start + input_index].?;
                            traffic.bytes_read += tensorBytes(program.tensors[tensor_id].?);
                        }
                        for (0..node.output_count) |output_index| {
                            const tensor_id = program.output_refs[node.output_start + output_index].?;
                            traffic.bytes_written += tensorBytes(program.tensors[tensor_id].?);
                        }
                    },
                }
            }
            return traffic;
        }

        fn tensorBytes(comptime info: Program.TensorInfo) u128 {
            return @as(u128, info.shape.elementCount()) * info.dtype.byteSize();
        }

        fn invocationArithmetic(comptime program: Program, comptime node: Program.Invocation, comptime compute: Execution.ExecutableCompute) u128 {
            return switch (compute) {
                .direct => |semantic| switch (semantic) {
                    .matmul => contractionWork(program, node) * 4,
                    .sum, .mean, .min, .max => inputElements(program, node, 0),
                    .softmax => inputElements(program, node, 0) * 3,
                    else => outputElements(program, node, 0),
                },
                .kernel => |kernel| switch (kernel) {
                    .map => |plan| switch (plan.strategy) {
                        .traversal => switch (plan.region.body) {
                            .expression => |expression| outputElements(program, node, 0) *
                                @max(expression.instructions.len, 1),
                            .transfer, .expression_transfer => unreachable,
                        },
                        .segmented => |segmented| blk: {
                            var elements: u128 = 0;
                            for (segmented.segments) |segment| elements += segment.elementCount();
                            if (plan.region.body == .expression_transfer) {
                                elements *= @max(plan.region.body.expression_transfer.instructions.len, 1);
                            }
                            break :blk elements;
                        },
                        .loop => |loop_plan| blk: {
                            var elements = outputElements(program, node, 0);
                            if (plan.region.body == .expression_transfer) {
                                elements *= @max(plan.region.body.expression_transfer.instructions.len, 1);
                            }
                            _ = loop_plan;
                            break :blk elements;
                        },
                    },
                    .reduction => |plan| product(plan.region.domain.shape) *
                        @max(plan.region.expressions.instructions.len + plan.region.accumulators.len, 1),
                    .contraction => |plan| contractionWork(program, node) *
                        (if (plan.strategy == .scalar) @as(u128, 4) else 1),
                },
            };
        }

        fn contractionWork(comptime program: Program, comptime node: Program.Invocation) u128 {
            const lhs = program.tensors[program.input_refs[node.input_start].?].?;
            const rhs = program.tensors[program.input_refs[node.input_start + 1].?].?;
            if (lhs.shape.rank == 2 and rhs.shape.rank == 2) {
                return @as(u128, lhs.shape.at(0)) * lhs.shape.at(1) * rhs.shape.at(1) * 2;
            }
            return outputElements(program, node, 0);
        }

        fn inputElements(comptime program: Program, comptime node: Program.Invocation, comptime index: usize) u128 {
            return program.tensors[program.input_refs[node.input_start + index].?].?.shape.elementCount();
        }

        fn outputElements(comptime program: Program, comptime node: Program.Invocation, comptime index: usize) u128 {
            return program.tensors[program.output_refs[node.output_start + index].?].?.shape.elementCount();
        }

        fn product(comptime dimensions: []const usize) u128 {
            var result: u128 = 1;
            for (dimensions) |dimension| result *= dimension;
            return result;
        }

        fn persistentBytes(comptime program: Program, comptime lifetimes: anytype, comptime SourcePlan: type) usize {
            var bytes: usize = 0;
            for (0..program.tensor_ct) |tensor_id| {
                if (!program.materialized[tensor_id]) continue;
                const info = program.tensors[tensor_id].?;
                if (info.storage_tensor != tensor_id or !SourcePlan.isOwned(info)) continue;
                if (lifetimes.tensor_lifetimes[tensor_id].isPersistent()) {
                    bytes += info.shape.elementCount() * info.dtype.byteSize();
                }
            }
            return bytes;
        }

        fn codeSize(comptime program: Program) usize {
            var units = program.node_ct;
            for (0..program.node_ct) |node_id| {
                const node = program.nodes[node_id].?;
                switch (node.op) {
                    .view => {},
                    .compute => |compute| switch (compute) {
                        .direct => units += 1,
                        .kernel => |kernel| switch (kernel) {
                            .map => |plan| switch (plan.strategy) {
                                .traversal => switch (plan.region.body) {
                                    .expression => |expression| units += expression.instructions.len,
                                    .transfer, .expression_transfer => unreachable,
                                },
                                .segmented => |segmented| {
                                    units += segmented.segments.len;
                                    if (plan.region.body == .expression_transfer) {
                                        units += plan.region.body.expression_transfer.instructions.len;
                                    }
                                },
                                .loop => |loop_plan| {
                                    units += loop_plan.iterations.len;
                                    if (plan.region.body == .expression_transfer) {
                                        units += plan.region.body.expression_transfer.instructions.len;
                                    }
                                },
                            },
                            .reduction => |plan| units += plan.region.expressions.instructions.len + plan.region.accumulators.len,
                            .contraction => units += 1,
                        },
                    },
                }
            }
            return units;
        }
    };
}
