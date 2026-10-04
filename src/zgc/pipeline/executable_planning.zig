const Execution = @import("../execution.zig");
const Executable = @import("../execution/program.zig").Executable;
const Graph = @import("../graph.zig");
const Source = @import("../source.zig");
const Storage = @import("../storage.zig");
const Tensor = @import("../tensor.zig");
const Analysis = @import("analysis.zig");
const Scheduling = @import("scheduling.zig");
const validation = @import("validation.zig");

pub fn SearchResult(comptime capacity: Graph.Capacity) type {
    return struct {
        reference: PlanCandidate(capacity),
        frontier: CandidateSet(capacity),
        generated_count: usize,
        fusion_candidate_count: usize,
        layout_candidate_count: usize,

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

pub const PlanCost = struct {
    estimated_runtime_work: u128,
    peak_memory_bytes: usize,
    persistent_memory_bytes: usize,
    scratch_memory_bytes: usize,
    code_size_units: usize,
    conversion: ConversionCost = .{},

    pub fn dominates(lhs: PlanCost, rhs: PlanCost) bool {
        const no_worse = lhs.estimated_runtime_work <= rhs.estimated_runtime_work and
            lhs.peak_memory_bytes <= rhs.peak_memory_bytes and
            lhs.persistent_memory_bytes <= rhs.persistent_memory_bytes and
            lhs.scratch_memory_bytes <= rhs.scratch_memory_bytes and
            lhs.code_size_units <= rhs.code_size_units and
            lhs.conversion.estimated_work <= rhs.conversion.estimated_work and
            lhs.conversion.bytes_read <= rhs.conversion.bytes_read and
            lhs.conversion.bytes_written <= rhs.conversion.bytes_written;
        const strictly_better = lhs.estimated_runtime_work < rhs.estimated_runtime_work or
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
            if (set.count == 0) @compileError("executable planning produced no candidates");
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
    };
}

/// Build a bounded set of complete executable programs, perform exact lifetime
/// and storage planning for each, and retain their Pareto frontier.
pub fn ExecutablePlanning(comptime capacity: Graph.Capacity) type {
    return struct {
        const Program = Executable(capacity, Execution.Op);
        const SemanticFacts = Analysis.Facts(capacity.max_tensors);
        const FusionCandidates = Analysis.FusionCandidates(capacity.max_nodes);
        const CandidateType = PlanCandidate(capacity);
        const Result = SearchResult(capacity);

        pub fn search(
            comptime SourceKey: type,
            comptime SemanticValidated: type,
            comptime semantic_analysis: SemanticFacts,
            comptime fusion_candidates: FusionCandidates,
            comptime layout_candidates: Analysis.LayoutCandidates,
            comptime source_configuration: anytype,
        ) Result {
            var result: Result = undefined;
            result.frontier = .{};
            result.generated_count = 0;
            result.fusion_candidate_count = fusion_candidates.count;
            result.layout_candidate_count = layout_candidates.count;

            const reference_schedules = Scheduling.Scheduling(capacity).enumerate(SemanticValidated.graph);
            inline for (reference_schedules, 0..) |schedule, schedule_index| {
                const executable = Scheduling.Reference(capacity).plan(SemanticValidated.graph, schedule);
                const candidate = complete(
                    SourceKey,
                    executable,
                    .reference,
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
                    const base_executable = ExecutableLowering(capacity).lower(
                        LayoutValidated.graph,
                        fusion_candidate.regions,
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
                            schedule.kind,
                            source_configuration,
                        ));
                        result.generated_count += 1;
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
                .schedule = schedule,
                .cost = .{
                    .estimated_runtime_work = estimateRuntime(executable),
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

        fn invocationArithmetic(comptime program: Program, comptime node: Program.Invocation, comptime compute: Execution.ExecutableCompute) u128 {
            return switch (compute) {
                .direct => |semantic| switch (semantic) {
                    .matmul => contractionWork(program, node) * 4,
                    .sum, .mean, .min, .max => inputElements(program, node, 0),
                    .softmax => inputElements(program, node, 0) * 3,
                    else => outputElements(program, node, 0),
                },
                .kernel => |kernel| switch (kernel) {
                    .map => |plan| outputElements(program, node, 0) *
                        @max(plan.region.expressions.instructions.len, 1),
                    .reduction => |plan| product(plan.region.domain_shape) *
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
                            .map => |plan| units += plan.region.expressions.instructions.len,
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

const std = @import("std");
const Semantic = @import("../operations/semantic.zig");
const Plan = @import("../execution/kernel_plan.zig");
const matmul = @import("../optimization/matmul.zig");
const Expression = @import("../optimization/fusion/expression.zig");
const Region = @import("../optimization/fusion/region.zig");
const Elementwise = @import("../operations/elementwise.zig");
const Reduction = @import("../operations/reduction.zig");
/// Lower one selected fusion/layout combination into an executable. Search
/// policy belongs to ExecutablePlanning; this type only realizes a choice.
pub fn ExecutableLowering(comptime capacity: Graph.Capacity) type {
    return struct {
        const SemanticGraph = Graph.Graph(capacity, Semantic.Op);
        const FusionRegions = Analysis.FusionSelection(capacity.max_nodes);

        pub fn lower(comptime source_graph: SemanticGraph, comptime fusion_regions: FusionRegions) Executable(capacity, Execution.Op) {
            var program: Executable(capacity, Execution.Op) = .init();

            for (0..source_graph.tensor_ct) |tensor_id| {
                const info = source_graph.tensors[tensor_id].?;
                const inserted = program.insertTensor(info);
                if (inserted != tensor_id) @compileError("executable lowering changed tensor order");
                program.materialized[tensor_id] = switch (info.origin) {
                    .source, .literal => true,
                    .node => false,
                };
            }
            for (0..source_graph.sources.len) |source_index| {
                if (source_graph.sources[source_index]) |source| program.insertSource(source_index, source);
            }

            inline for (0..source_graph.node_ct) |node_id| {
                if (fusion_regions.node_region[node_id]) |region| {
                    switch (region) {
                        .reduction => |region_id| {
                            const group = fusion_regions.reduction_storage[region_id].?;
                            if (node_id != group.emit_node) continue;
                            const Planned = PlannedReduction(source_graph, group);
                            insertInvocation(
                                &program,
                                .{ .compute = .{ .kernel = .{ .reduction = Planned.plan } } },
                                &Planned.input_ids,
                                &Planned.output_ids,
                            );
                        },
                        .map => |map_id| {
                            const group = fusion_regions.map_storage[map_id].?;
                            if (node_id != group.root_node) continue;
                            const Planned = PlannedMap(source_graph, group);
                            insertInvocation(
                                &program,
                                .{ .compute = .{ .kernel = .{ .map = Planned.plan } } },
                                &Planned.input_ids,
                                &Planned.output_ids,
                            );
                        },
                    }
                    continue;
                }

                const node = source_graph.nodes[node_id].?;
                const executable: Execution.Op = switch (node.op) {
                    .view => |view| .{ .view = view },
                    .compute => |compute| .{ .compute = planCompute(compute, &source_graph, node) },
                };
                var inputs: [node.input_count]usize = undefined;
                inline for (0..node.input_count) |input_index| {
                    inputs[input_index] = source_graph.input_refs[node.input_start + input_index].?;
                }
                insertInvocation(&program, executable, &inputs, &.{node.result});
            }

            for (0..source_graph.output_ct) |output_index| {
                program.insertOutput(source_graph.outputs[output_index].?);
            }
            return program;
        }

        fn insertInvocation(
            program: *Executable(capacity, Execution.Op),
            comptime op: Execution.Op,
            comptime inputs: []const usize,
            comptime outputs: []const usize,
        ) void {
            for (inputs) |tensor_id| program.insertInputRef(tensor_id);
            for (outputs) |tensor_id| {
                program.insertOutputRef(tensor_id);
                program.materialized[tensor_id] = true;
                program.tensors[tensor_id].?.origin = .{ .node = program.node_ct };
            }
            program.insertInvocation(.{
                .op = op,
                .input_start = program.input_ref_ct - inputs.len,
                .input_count = inputs.len,
                .output_start = program.output_ref_ct - outputs.len,
                .output_count = outputs.len,
            });
        }

        fn planCompute(
            comptime compute: Semantic.Op.Compute,
            comptime graph: anytype,
            comptime node: anytype,
        ) Execution.ExecutableCompute {
            return switch (compute) {
                .matmul => blk: {
                    const lhs_id = graph.input_refs[node.input_start].?;
                    const rhs_id = graph.input_refs[node.input_start + 1].?;
                    break :blk .{ .kernel = .{ .contraction = matmul.plan(
                        capacity,
                        graph.tensors[lhs_id].?,
                        graph.tensors[rhs_id].?,
                        graph.tensors[node.result].?,
                    ) } };
                },
                else => .{ .direct = compute },
            };
        }

        fn PlannedReduction(comptime graph: anytype, comptime group: anytype) type {
            const built = comptime buildReduction(graph, group);
            return struct {
                const input_ids = built.inputs[0..built.input_count].*;
                const output_ids = built.outputs[0..built.output_count].*;
                const instructions = built.instructions[0..built.instruction_count].*;
                const accumulators = built.accumulators[0..built.accumulator_count].*;
                const stores = built.stores[0..built.store_count].*;
                const domain_shape = built.domain_shape[0..built.domain_rank].*;
                const outer_axes = built.outer_axes[0..built.outer_axis_count].*;
                const reduction_axes = built.reduction_axis_order[0..built.reduction_axis_count].*;

                pub const plan: Plan.ReductionPlan = .{
                    .region = .{
                        .expressions = .{ .instructions = &instructions },
                        .domain_shape = &domain_shape,
                        .reduction_axes = group.descriptor.axes,
                        .keep_dims = group.descriptor.keep_dims,
                        .accumulators = &accumulators,
                        .stores = &stores,
                    },
                    .traversal_plan = .{
                        .outer_axis_order = &outer_axes,
                        .reduction_axis_order = &reduction_axes,
                        .vector_axis = built.vector_axis,
                        .vector_width = built.vector_width,
                        .accumulator_lanes = 1,
                    },
                };
            };
        }

        fn PlannedMap(comptime graph: anytype, comptime group: anytype) type {
            const built = comptime buildMap(graph, group);
            return struct {
                const input_ids = built.inputs[0..built.input_count].*;
                const output_ids = built.outputs;
                const instructions = built.instructions[0..built.instruction_count].*;
                const stores = built.stores;
                const axis_order = built.axis_order[0..built.axis_count].*;

                pub const plan: Plan.MapPlan = .{
                    .region = .{
                        .expressions = .{ .instructions = &instructions },
                        .stores = &stores,
                    },
                    .traversal_plan = .{
                        .axis_order = &axis_order,
                        .traversal = built.traversal,
                        .vector_axis = if (built.vector_width > 1 and built.axis_count > 0)
                            @intCast(built.axis_count - 1)
                        else
                            null,
                        .vector_width = built.vector_width,
                    },
                };
            };
        }

        fn MapBuildResult(comptime graph: anytype) type {
            return struct {
                inputs: [graph.tensor_ct]usize = undefined,
                input_count: usize = 0,
                outputs: [1]usize = undefined,
                instructions: [graph.node_ct]Expression.Program.Instruction = undefined,
                instruction_count: usize = 0,
                stores: [1]Region.Store = undefined,
                values: [graph.tensor_ct]?Expression.Program.ValueRef = @splat(null),
                axis_order: [capacity.max_rank]u8 = @splat(0),
                axis_count: usize = 0,
                traversal: Plan.MapPlan.Traversal = .strided,
                vector_width: usize = 1,
            };
        }

        fn buildMap(comptime graph: anytype, comptime group: anytype) MapBuildResult(graph) {
            var built: MapBuildResult(graph) = .{};
            const root = graph.nodes[group.root_node].?;
            const root_value = buildExpressionValue(graph, group.nodes, root.result, &built);
            built.outputs[0] = root.result;
            built.stores[0] = .{ .output = 0, .value = root_value };

            const output = graph.tensors[root.result].?;
            built.axis_count = output.shape.rank;
            for (0..output.shape.rank) |axis| built.axis_order[axis] = @intCast(axis);
            built.traversal = if (isContiguous(output)) .contiguous else .strided;
            var vector_dtype = output.dtype;
            for (built.inputs[0..built.input_count]) |input_id| {
                const dtype = graph.tensors[input_id].?.dtype;
                if (dtype.kind() != .boolean) {
                    vector_dtype = dtype;
                    break;
                }
            }
            built.vector_width = if (output.shape.rank == 0)
                1
            else
                std.simd.suggestVectorLength(vector_dtype.Scalar()) orelse 1;
            return built;
        }

        fn isContiguous(comptime info: anytype) bool {
            var expected: isize = 1;
            var axis = info.shape.rank;
            while (axis > 0) {
                axis -= 1;
                if (info.shape.at(axis) > 1 and info.layout.strides[axis] != expected) return false;
                expected *= @intCast(info.shape.at(axis));
            }
            return true;
        }

        fn BuildResult(comptime graph: anytype) type {
            return struct {
                inputs: [graph.tensor_ct]usize = undefined,
                input_count: usize = 0,
                outputs: [graph.node_ct]usize = undefined,
                output_count: usize = 0,
                instructions: [graph.node_ct]Expression.Program.Instruction = undefined,
                instruction_count: usize = 0,
                accumulators: [graph.node_ct]Region.Reduction.Accumulator = undefined,
                accumulator_count: usize = 0,
                stores: [graph.node_ct]Region.Store = undefined,
                store_count: usize = 0,
                values: [graph.tensor_ct]?Expression.Program.ValueRef = @splat(null),
                domain_shape: [capacity.max_rank]usize = @splat(0),
                domain_rank: usize = 0,
                outer_axes: [capacity.max_rank]u8 = @splat(0),
                outer_axis_count: usize = 0,
                reduction_axis_order: [capacity.max_rank]u8 = @splat(0),
                reduction_axis_count: usize = 0,
                vector_axis: ?u8 = null,
                vector_width: usize = 1,
            };
        }

        fn buildReduction(comptime graph: anytype, comptime group: anytype) BuildResult(graph) {
            var built: BuildResult(graph) = .{};
            const domain = graph.tensors[group.domain_tensor].?;
            built.domain_rank = domain.shape.rank;
            for (0..domain.shape.rank) |axis| {
                built.domain_shape[axis] = domain.shape.at(axis);
                if (group.descriptor.axes & (@as(u64, 1) << @intCast(axis)) != 0) {
                    built.reduction_axis_order[built.reduction_axis_count] = @intCast(axis);
                    built.reduction_axis_count += 1;
                } else {
                    built.outer_axes[built.outer_axis_count] = @intCast(axis);
                    built.outer_axis_count += 1;
                }
            }

            for (group.reduction_nodes[0..group.reduction_count]) |maybe_node_id| {
                const node_id = maybe_node_id.?;
                const node = graph.nodes[node_id].?;
                const descriptor = Reduction.fromCompute(node.op.compute).?;
                const input_id = graph.input_refs[node.input_start].?;
                const accumulator_index = built.accumulator_count;
                built.accumulators[accumulator_index] = .{
                    .combine = switch (descriptor.kind) {
                        .sum, .mean => .sum,
                        .minimum => .minimum,
                        .maximum => .maximum,
                    },
                    .update = buildExpressionValue(graph, group.nodes, input_id, &built),
                    .finalize = if (descriptor.kind == .mean) .mean else .identity,
                };
                built.accumulator_count += 1;
                built.outputs[built.output_count] = node.result;
                built.output_count += 1;
                built.stores[built.store_count] = .{
                    .output = built.store_count,
                    .value = .{ .accumulator = accumulator_index },
                };
                built.store_count += 1;
            }
            chooseReductionVectorization(graph, domain, &built);
            return built;
        }

        fn chooseReductionVectorization(comptime graph: anytype, comptime domain: anytype, built: anytype) void {
            const vector_width = std.simd.suggestVectorLength(domain.dtype.Scalar()) orelse 1;
            if (vector_width == 1) return;

            var axis = domain.shape.rank;
            while (axis > 0) {
                axis -= 1;
                if (groupAxisIsReduced(built, axis) and
                    domain.shape.at(axis) >= vector_width and
                    reductionInputsSupportVectorAxis(graph, domain, built, axis))
                {
                    built.vector_axis = @intCast(axis);
                    built.vector_width = vector_width;
                    return;
                }
            }
        }

        fn groupAxisIsReduced(built: anytype, comptime axis: usize) bool {
            for (built.reduction_axis_order[0..built.reduction_axis_count]) |reduction_axis| {
                if (reduction_axis == axis) return true;
            }
            return false;
        }

        fn reductionInputsSupportVectorAxis(
            comptime graph: anytype,
            comptime domain: anytype,
            built: anytype,
            comptime axis: usize,
        ) bool {
            for (built.inputs[0..built.input_count]) |input_id| {
                const input = graph.tensors[input_id].?;
                const leading_axes = domain.shape.rank - input.shape.rank;
                if (axis < leading_axes) continue;
                const input_axis = axis - leading_axes;
                if (input.shape.at(input_axis) == 1 and domain.shape.at(axis) != 1) continue;
                if (input.layout.strides[input_axis] != 1) return false;
            }
            return true;
        }

        fn buildExpressionValue(
            comptime graph: anytype,
            comptime included_nodes: anytype,
            comptime tensor_id: usize,
            built: anytype,
        ) Expression.Program.ValueRef {
            if (built.values[tensor_id]) |value| return value;
            const producer_id = switch (graph.tensors[tensor_id].?.origin) {
                .node => |id| id,
                .source, .literal => return addExpressionInput(tensor_id, built),
            };
            if (!included_nodes[producer_id]) return addExpressionInput(tensor_id, built);

            const producer = graph.nodes[producer_id].?;
            const operation = Elementwise.fromCompute(producer.op.compute).?;
            var args: [3]Expression.Program.ValueRef = @splat(.{ .input = 0 });
            for (0..producer.input_count) |input_index| {
                args[input_index] = buildExpressionValue(
                    graph,
                    included_nodes,
                    graph.input_refs[producer.input_start + input_index].?,
                    built,
                );
            }
            const value: Expression.Program.ValueRef = .{ .instruction = built.instruction_count };
            built.instructions[built.instruction_count] = .{
                .operation = operation,
                .dtype = graph.tensors[tensor_id].?.dtype,
                .args = args,
            };
            built.instruction_count += 1;
            built.values[tensor_id] = value;
            return value;
        }

        fn addExpressionInput(comptime tensor_id: usize, built: anytype) Expression.Program.ValueRef {
            for (built.inputs[0..built.input_count], 0..) |existing, index| {
                if (existing == tensor_id) return .{ .input = index };
            }
            const index = built.input_count;
            built.inputs[index] = tensor_id;
            built.input_count += 1;
            const value: Expression.Program.ValueRef = .{ .input = index };
            built.values[tensor_id] = value;
            return value;
        }
    };
}
