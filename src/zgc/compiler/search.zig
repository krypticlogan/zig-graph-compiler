const std = @import("std");
const Execution = @import("../execution/execution.zig");
const ProgramFactory = @import("../execution/program.zig").Program;
const Graph = @import("../core/graph.zig");
const Source = @import("../storage/source.zig");
const Output = @import("../storage/output.zig");
const Storage = @import("../storage/storage.zig");
const Tensor = @import("../core/tensor.zig");
const Elementwise = @import("../operations/elementwise.zig");
const Analysis = @import("analysis/root.zig");
const Scheduling = @import("scheduling.zig");
const validation = @import("validation.zig");
const Planning = @import("planning/root.zig");
const Contraction = @import("planning/contraction.zig");

pub fn SearchResult(comptime capacity: Graph.Capacity) type {
    return struct {
        reference: PlanCandidate(capacity),
        frontier: CandidateSet(capacity),
        generated_count: usize,
        representation_candidate_count: usize,
        fusion_candidate_count: usize,
        layout_candidate_count: usize,
        remap_candidate_count: usize,
        contraction_candidate_count: usize,
        generated_contraction_strategies: [4]bool,

        pub fn selected(comptime result: @This(), comptime options: SelectionOptions) PlanCandidate(capacity) {
            return result.frontier.candidates[result.selectedIndex(options)].?;
        }

        pub fn selectedIndex(comptime result: @This(), comptime options: SelectionOptions) usize {
            if (options.frontier_index) |index| {
                if (index >= result.frontier.count) @compileError("frontier_index is outside the retained executable frontier");
                return index;
            }
            return result.frontier.bestIndex(options.target);
        }
    };
}

/// Half-open execution interval for one tensor's backing storage.
/// A null end marks storage that must remain reserved for the runtime lifetime.
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
    arithmetic_work: u128 = 0,
    access_work: u128 = 0,
    register_pressure_work: u128 = 0,
    transfer_work: u128 = 0,
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
pub const OptimizationTarget = enum { 
        /// Balance runtime work, memory traffic, and code size.
        balanced, 
        /// Minimize peak memory usage, including scratch and persistent allocations.
        memory, 
        /// Minimize estimated runtime work, including arithmetic, access, and transfer.
        fast, 
        /// Minimize code size, including instruction bytes and conversion overhead.
        small 
};
pub const SelectionOptions = struct {
    target: OptimizationTarget = .balanced,
    /// Index into the retained frontier; overrides the optimization target.
    frontier_index: ?usize = null,
};

pub fn PlanCandidate(comptime capacity: Graph.Capacity) type {
    return struct {
        executable: ProgramFactory(capacity, Execution.Op),
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

        /// Retain distinct executables, including dominated ones that users
        /// may force for measurement. Once full, preserve each target's best
        /// candidate and evict the weakest remaining balanced candidate.
        pub fn insert(set: *Self, candidate: Candidate) void {
            for (set.candidates[0..set.count], 0..) |existing, index| {
                if (sameExecutable(existing.?.executable, candidate.executable)) {
                    if (candidate.cost.preferredTo(existing.?.cost) or
                        (costsEqual(candidate.cost, existing.?.cost) and tieBreak(candidate, existing.?)))
                    {
                        set.candidates[index] = candidate;
                    }
                    return;
                }
            }

            if (set.count == max_candidates) {
                var pool: [max_candidates + 1]?Candidate = undefined;
                @memcpy(pool[0..max_candidates], &set.candidates);
                pool[max_candidates] = candidate;
                var protected: [max_candidates + 1]bool = @splat(false);
                inline for (std.meta.tags(OptimizationTarget)) |target| {
                    protected[bestOf(&pool, target)] = true;
                }
                const bounds = minima(&pool);
                var worst_index: ?usize = null;
                for (pool, 0..) |maybe_entry, candidate_index| {
                    if (protected[candidate_index]) continue;
                    if (worst_index == null or better(pool[worst_index.?].?, maybe_entry.?, .balanced, bounds)) {
                        worst_index = candidate_index;
                    }
                }
                if (worst_index.? != max_candidates) set.candidates[worst_index.?] = candidate;
                return;
            }
            set.candidates[set.count] = candidate;
            set.count += 1;
        }

        pub fn bestIndex(comptime set: Self, comptime target: OptimizationTarget) usize {
            if (set.count == 0) @compileError("executable search produced no candidates");
            return bestOf(set.candidates[0..set.count], target);
        }

        fn bestOf(comptime entries: []const ?Candidate, comptime target: OptimizationTarget) usize {
            const bounds = minima(entries);
            var best_index: usize = 0;
            for (1..entries.len) |index| {
                if (better(entries[index].?, entries[best_index].?, target, bounds)) best_index = index;
            }
            return best_index;
        }

        const Minima = struct { work: u128, peak: usize, code: usize };

        fn minima(comptime entries: []const ?Candidate) Minima {
            var result: Minima = .{
                .work = entries[0].?.cost.estimated_runtime_work,
                .peak = entries[0].?.cost.peak_memory_bytes,
                .code = entries[0].?.cost.code_size_units,
            };
            for (entries[1..]) |entry| {
                result.work = @min(result.work, entry.?.cost.estimated_runtime_work);
                result.peak = @min(result.peak, entry.?.cost.peak_memory_bytes);
                result.code = @min(result.code, entry.?.cost.code_size_units);
            }
            return result;
        }

        fn balancedScore(cost: PlanCost, bounds: Minima) f64 {
            return 3 * ratio(cost.estimated_runtime_work, bounds.work) +
                ratio(cost.peak_memory_bytes, bounds.peak) +
                ratio(cost.code_size_units, bounds.code);
        }

        fn ratio(value: anytype, baseline: @TypeOf(value)) f64 {
            return @as(f64, @floatFromInt(value)) /
                @as(f64, @floatFromInt(@max(baseline, 1)));
        }

        fn better(candidate: Candidate, current: Candidate, target: OptimizationTarget, bounds: Minima) bool {
            switch (target) {
                .balanced => {
                    const candidate_score = balancedScore(candidate.cost, bounds);
                    const current_score = balancedScore(current.cost, bounds);
                    if (candidate_score != current_score) return candidate_score < current_score;
                },
                .memory => {
                    if (candidate.cost.peak_memory_bytes != current.cost.peak_memory_bytes)
                        return candidate.cost.peak_memory_bytes < current.cost.peak_memory_bytes;
                    if (candidate.cost.scratch_memory_bytes != current.cost.scratch_memory_bytes)
                        return candidate.cost.scratch_memory_bytes < current.cost.scratch_memory_bytes;
                },
                .fast => {},
                .small => {
                    if (candidate.cost.code_size_units != current.cost.code_size_units)
                        return candidate.cost.code_size_units < current.cost.code_size_units;
                },
            }
            if (candidate.cost.preferredTo(current.cost)) return true;
            return costsEqual(candidate.cost, current.cost) and tieBreak(candidate, current);
        }

        fn costsEqual(lhs: PlanCost, rhs: PlanCost) bool {
            return !lhs.preferredTo(rhs) and !rhs.preferredTo(lhs);
        }

        fn sameExecutable(lhs: ProgramFactory(capacity, Execution.Op), rhs: ProgramFactory(capacity, Execution.Op)) bool {
            if (lhs.node_ct != rhs.node_ct or lhs.tensor_ct != rhs.tensor_ct or
                lhs.input_ref_ct != rhs.input_ref_ct or lhs.output_ref_ct != rhs.output_ref_ct or
                lhs.output_ct != rhs.output_ct) return false;
            for (lhs.tensors[0..lhs.tensor_ct], rhs.tensors[0..rhs.tensor_ct]) |left, right| {
                if (!std.meta.eql(left, right)) return false;
            }
            for (lhs.nodes[0..lhs.node_ct], rhs.nodes[0..rhs.node_ct]) |left, right| {
                if (!std.meta.eql(left, right)) return false;
            }
            if (!std.mem.eql(?usize, lhs.input_refs[0..lhs.input_ref_ct], rhs.input_refs[0..rhs.input_ref_ct]) or
                !std.mem.eql(?usize, lhs.output_refs[0..lhs.output_ref_ct], rhs.output_refs[0..rhs.output_ref_ct]) or
                !std.mem.eql(?usize, lhs.outputs[0..lhs.output_ct], rhs.outputs[0..rhs.output_ct])) return false;
            return std.mem.eql(bool, lhs.materialized[0..lhs.tensor_ct], rhs.materialized[0..rhs.tensor_ct]);
        }

        fn tieBreak(candidate: Candidate, current: Candidate) bool {
            if (@backingInt(candidate.schedule) != @backingInt(current.schedule)) {
                return @backingInt(candidate.schedule) < @backingInt(current.schedule);
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
            return @backingInt(candidate.origin) < @backingInt(current.origin);
        }

        fn layoutOrder(value: ?Analysis.LayoutRegime) usize {
            return if (value) |present| switch (present) {
                .propagated => 1,
                .unpacked => 2,
                .mixed => 3,
                .canonical => 4,
            } else 0;
        }

        fn fusionOrder(value: ?Analysis.FusionRegime) usize {
            return if (value) |present| switch (present) {
                .discovered => 1,
                .independent => 2,
                .unfused => 3,
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

/// Build a bounded set of distinct executable programs and perform exact
/// lifetime and storage planning for each.
pub fn ExecutableSearch(comptime capacity: Graph.Capacity) type {
    return struct {
        const Program = ProgramFactory(capacity, Execution.Op);
        const SemanticFacts = Analysis.Facts(capacity);
        const FusionCandidates = Analysis.FusionCandidates(capacity.max_nodes);
        const LayoutCandidates = Analysis.LayoutCandidates(capacity);
        const RemapCandidates = Analysis.RemapCandidates(capacity.max_nodes);
        const CandidateType = PlanCandidate(capacity);
        const Result = SearchResult(capacity);
        const max_representation_candidates = 24;

        const Representation = struct {
            fusion_regions: Analysis.FusionSelection(capacity.max_nodes) = .{},
            layout_regions: Analysis.LayoutSelection(capacity) = .{},
            remap_regions: Analysis.RemapSelection(capacity.max_nodes) = .{},
            contraction_strategies: [capacity.max_nodes]?Contraction.Plan.Strategy = @splat(null),
            claimed_nodes: [capacity.max_nodes]bool = @splat(false),
            selected_nodes: usize = 0,
            selected_tensors: usize = 0,
            selected_regions: usize = 0,
            selected_policies: usize = 0,
            local_cost: Analysis.LocalCost.Estimate = .{},

            fn fusionRegime(representation: @This()) Analysis.FusionRegime {
                if (representation.fusion_regions.map_count != 0) return .discovered;
                for (representation.fusion_regions.reduction_storage[0..representation.fusion_regions.region_count]) |maybe_group| {
                    const group = maybe_group.?;
                    if (group.reduction_count > 1) return .discovered;
                    for (group.nodes) |folded| {
                        if (folded) return .discovered;
                    }
                }
                return if (representation.fusion_regions.region_count == 0) .unfused else .independent;
            }

            fn remapRegime(representation: @This()) Analysis.RemapRegime {
                return if (representation.remap_regions.region_count != 0) .composed else .direct;
            }

            fn layoutRegime(representation: @This(), comptime candidates: LayoutCandidates) Analysis.LayoutRegime {
                if (representation.layout_regions.region_count == 0) return .canonical;
                for (1..candidates.count) |index| {
                    const candidate = candidates.values[index].?;
                    var equal = true;
                    for (representation.layout_regions.results, candidate.regions.results) |lhs, rhs| {
                        if (!std.meta.eql(lhs, rhs)) {
                            equal = false;
                            break;
                        }
                    }
                    if (equal) return candidate.regime;
                }
                return .mixed;
            }
        };

        const RepresentationSet = struct {
            values: [max_representation_candidates]?Representation = @splat(null),
            count: usize = 0,
            protected_count: usize = 1,

            fn init() @This() {
                var result: @This() = .{};
                result.values[0] = .{};
                result.count = 1;
                return result;
            }

            fn insert(set: *@This(), comptime graph: anytype, proposed: Representation) void {
                var candidate = proposed;
                const layout_graph = Analysis.LayoutAnalysis(capacity).applySelection(graph, candidate.layout_regions);
                candidate.local_cost = Analysis.LocalCost.estimate(capacity, layout_graph, candidate);
                for (set.values[0..set.count]) |existing| {
                    if (sameRepresentation(existing.?, candidate)) return;
                }
                if (set.count < max_representation_candidates) {
                    set.values[set.count] = candidate;
                    set.count += 1;
                    return;
                }

                // Keep the baseline and the broad anchor. Compare partial
                // states within a policy class first, so a scalar contraction
                // or an unfused layout is not displaced solely by coverage.
                const candidate_class = policyClass(candidate);
                for (set.values[set.protected_count..set.count], set.protected_count..) |value, index| {
                    if (policyClass(value.?) != candidate_class or !sameBoundary(value.?, candidate)) continue;
                    if (locallyBetter(candidate, value.?)) set.values[index] = candidate;
                    return;
                }
                var worst_index: ?usize = null;
                for (set.protected_count..set.count) |index| {
                    if (policyClass(set.values[index].?) != candidate_class) continue;
                    if (worst_index == null or locallyBetter(set.values[worst_index.?].?, set.values[index].?)) {
                        worst_index = index;
                    }
                }
                if (worst_index == null) {
                    for (set.protected_count..set.count) |index| {
                        var peers: usize = 0;
                        for (set.values[set.protected_count..set.count]) |value| {
                            if (policyClass(value.?) == policyClass(set.values[index].?)) peers += 1;
                        }
                        if (peers < 2) continue;
                        if (worst_index == null or locallyBetter(set.values[worst_index.?].?, set.values[index].?)) {
                            worst_index = index;
                        }
                    }
                }
                if (worst_index) |index| {
                    if (policyClass(set.values[index].?) != candidate_class or locallyBetter(candidate, set.values[index].?)) {
                        set.values[index] = candidate;
                    }
                }
            }
        };

        pub fn search(
            comptime SourceKey: type,
            comptime SemanticValidated: type,
            comptime semantic_analysis: SemanticFacts,
            comptime fusion_candidates: FusionCandidates,
            comptime layout_candidates: LayoutCandidates,
            comptime remap_candidates: RemapCandidates,
            comptime contraction_candidates: Analysis.ContractionCandidates(capacity),
            comptime source_configuration: anytype,
            comptime output_binding: Output.Binding,
        ) Result {
            const representations = composeRepresentations(SemanticValidated.graph, fusion_candidates, layout_candidates, remap_candidates, contraction_candidates);
            var result: Result = undefined;
            result.frontier = .{};
            result.generated_count = 0;
            result.representation_candidate_count = representations.count;
            result.fusion_candidate_count = fusion_candidates.count;
            result.layout_candidate_count = layout_candidates.count;
            result.remap_candidate_count = remap_candidates.count;
            result.contraction_candidate_count = contraction_candidates.count;
            result.generated_contraction_strategies = @splat(false);

            const masked_fusion_required = comptime requiresMaskedFusion(SemanticValidated.graph, semantic_analysis);

            const reference_schedules = Scheduling.Scheduling(capacity).enumerate(SemanticValidated.graph);
            if (!masked_fusion_required) {
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
                        output_binding,
                    );
                    if (schedule_index == 0) result.reference = candidate;
                    result.frontier.insert(candidate);
                    result.generated_count += 1;
                }
            }

            inline for (0..representations.count) |representation_index| {
                const representation = representations.values[representation_index].?;
                if (masked_fusion_required and !respectsMaskedFusion(SemanticValidated.graph, semantic_analysis, representation.fusion_regions)) continue;
                const layout_graph = Analysis.LayoutAnalysis(capacity).applySelection(
                    SemanticValidated.graph,
                    representation.layout_regions,
                );
                const LayoutValidated = validation.Validation(capacity).validate(layout_graph);
                const base_executable = Planning.Lowering(capacity).lower(
                    LayoutValidated.graph,
                    semantic_analysis.dependencies.topological_order,
                    representation.fusion_regions,
                    representation.remap_regions,
                    representation.contraction_strategies,
                );
                for (base_executable.nodes[0..base_executable.node_ct]) |maybe_node| {
                    const node = maybe_node.?;
                    const compute = switch (node.op) {
                        .compute => |value| value,
                        .view => continue,
                    };
                    const kernel = switch (compute) {
                        .kernel => |value| value,
                        .direct => continue,
                    };
                    switch (kernel) {
                        .contraction => |plan| result.generated_contraction_strategies[@backingInt(plan.strategy)] = true,
                        else => {},
                    }
                }
                const schedules = Scheduling.ExecutableScheduling(capacity).enumerate(base_executable);
                inline for (schedules) |schedule| {
                    const executable = Scheduling.ExecutableScheduling(capacity).apply(base_executable, schedule);
                    const candidate = complete(
                        SourceKey,
                        executable,
                        .planned,
                        representation.fusionRegime(),
                        representation.layoutRegime(layout_candidates),
                        representation.remapRegime(),
                        schedule.kind,
                        source_configuration,
                        output_binding,
                    );
                    if (result.generated_count == 0) result.reference = candidate;
                    result.frontier.insert(candidate);
                    result.generated_count += 1;
                }
            }

            return result;
        }

        fn maskedWhereProducer(comptime graph: anytype, comptime analysis: SemanticFacts, comptime reduction_node_id: usize) ?usize {
            const reduction_node = graph.nodes[reduction_node_id].?;
            const reduction_compute = switch (reduction_node.op) {
                .compute => |compute| compute,
                .view => return null,
            };
            if (@import("../operations/reduction.zig").fromCompute(reduction_compute) == null) return null;
            const input_id = graph.input_refs[reduction_node.input_start].?;
            if (analysis.use_counts[input_id] != 1 or analysis.is_output[input_id]) return null;
            const producer_id = switch (graph.tensors[input_id].?.origin) {
                .node => |id| id,
                else => return null,
            };
            const producer = graph.nodes[producer_id].?;
            if (producer.op != .compute or producer.op.compute != .where) return null;
            inline for (1..3) |branch| {
                const branch_id = graph.input_refs[producer.input_start + branch].?;
                switch (graph.tensors[branch_id].?.origin) {
                    .literal => |literal| if (literal.bits == 0) return producer_id,
                    else => {},
                }
            }
            return null;
        }

        fn requiresMaskedFusion(comptime graph: anytype, comptime analysis: SemanticFacts) bool {
            for (0..graph.node_ct) |node_id| {
                if (maskedWhereProducer(graph, analysis, node_id) != null) return true;
            }
            return false;
        }

        fn respectsMaskedFusion(comptime graph: anytype, comptime analysis: SemanticFacts, comptime selection: Analysis.FusionSelection(capacity.max_nodes)) bool {
            for (0..graph.node_ct) |node_id| {
                const producer_id = maskedWhereProducer(graph, analysis, node_id) orelse continue;
                const reduction_region = selection.node_region[node_id] orelse return false;
                const where_region = selection.node_region[producer_id] orelse return false;
                if (!std.meta.eql(reduction_region, where_region)) return false;
            }
            return true;
        }

        fn composeRepresentations(
            comptime semantic_graph: Graph.Graph(capacity, @import("../operations/semantic.zig").Op),
            comptime fusion_candidates: FusionCandidates,
            comptime layout_candidates: LayoutCandidates,
            comptime remap_candidates: RemapCandidates,
            comptime contraction_candidates: Analysis.ContractionCandidates(capacity),
        ) RepresentationSet {
            var representations = RepresentationSet.init();

            // Preserve one maximally compatible anchor before bounded local
            // branching. It represents the coherent set discovered by the
            // analyses and prevents a collection of small partial choices
            // from displacing a useful whole-program composition.
            var anchor: Representation = .{};
            inline for (1..layout_candidates.count) |candidate_index| {
                const selection = layout_candidates.values[candidate_index].?.regions;
                inline for (0..selection.region_count) |region_id| {
                    _ = claimLayoutRegion(&anchor, selection, region_id);
                }
            }
            inline for (1..fusion_candidates.count) |candidate_index| {
                const selection = fusion_candidates.values[candidate_index].?.regions;
                inline for (0..selection.region_count) |region_id| {
                    if (hasReductionRegion(selection, region_id)) {
                        _ = claimFusionRegion(&anchor, selection, .{ .reduction = region_id });
                    }
                }
                inline for (0..selection.map_count) |map_id| {
                    _ = claimFusionRegion(&anchor, selection, .{ .map = map_id });
                }
            }
            inline for (1..remap_candidates.count) |candidate_index| {
                const selection = remap_candidates.values[candidate_index].?.regions;
                inline for (0..selection.region_count) |region_id| {
                    _ = claimRemapRegion(&anchor, selection, region_id);
                }
            }
            if (anchor.selected_regions != 0) {
                representations.insert(semantic_graph, anchor);
                representations.protected_count = representations.count;
            }

            inline for (1..layout_candidates.count) |candidate_index| {
                const selection = layout_candidates.values[candidate_index].?.regions;
                inline for (0..selection.region_count) |region_id| {
                    branchLayout(&representations, semantic_graph, selection, region_id);
                }
            }

            inline for (0..contraction_candidates.count) |candidate_index| {
                branchContraction(&representations, semantic_graph, contraction_candidates.values[candidate_index].?);
            }

            inline for (1..fusion_candidates.count) |candidate_index| {
                const selection = fusion_candidates.values[candidate_index].?.regions;
                inline for (0..selection.region_count) |region_id| {
                    if (!hasReductionRegion(selection, region_id)) continue;
                    branchReduction(&representations, semantic_graph, selection, region_id);
                }
                inline for (0..selection.map_count) |map_id| {
                    branchMap(&representations, semantic_graph, selection, map_id);
                }
            }

            inline for (1..remap_candidates.count) |candidate_index| {
                const selection = remap_candidates.values[candidate_index].?.regions;
                inline for (0..selection.region_count) |region_id| {
                    branchRemap(&representations, semantic_graph, selection, region_id);
                }
            }
            return representations;
        }

        fn branchContraction(
            representations: *RepresentationSet,
            comptime semantic_graph: Graph.Graph(capacity, @import("../operations/semantic.zig").Op),
            comptime policy: Analysis.ContractionPolicy,
        ) void {
            const existing = representations.*;
            for (existing.values[0..existing.count]) |value| {
                var candidate = value.?;
                if (candidate.contraction_strategies[policy.node] != null) continue;
                const graph = Analysis.LayoutAnalysis(capacity).applySelection(semantic_graph, candidate.layout_regions);
                const node = graph.nodes[policy.node].?;
                const lhs = graph.tensors[graph.input_refs[node.input_start].?].?;
                const rhs = graph.tensors[graph.input_refs[node.input_start + 1].?].?;
                const output = graph.tensors[node.result].?;
                if (!Contraction.supportsStrategy(lhs, rhs, output, policy.strategy)) continue;
                if (Contraction.selectStrategy(capacity, lhs, rhs, output) == policy.strategy) continue;
                candidate.contraction_strategies[policy.node] = policy.strategy;
                candidate.selected_policies += 1;
                representations.insert(semantic_graph, candidate);
            }
        }

        fn branchLayout(
            representations: *RepresentationSet,
            comptime semantic_graph: anytype,
            comptime source: Analysis.LayoutSelection(capacity),
            comptime source_region: usize,
        ) void {
            const existing = representations.*;
            var index: usize = 0;
            while (index < existing.count) : (index += 1) {
                var candidate = existing.values[index].?;
                if (!claimLayoutRegion(&candidate, source, source_region)) continue;
                representations.insert(semantic_graph, candidate);
            }
        }

        fn branchReduction(
            representations: *RepresentationSet,
            comptime semantic_graph: anytype,
            comptime source: Analysis.FusionSelection(capacity.max_nodes),
            comptime source_region: usize,
        ) void {
            const existing = representations.*;
            var index: usize = 0;
            while (index < existing.count) : (index += 1) {
                var candidate = existing.values[index].?;
                if (!claimFusionRegion(&candidate, source, .{ .reduction = source_region })) continue;
                representations.insert(semantic_graph, candidate);
            }
        }

        fn branchMap(
            representations: *RepresentationSet,
            comptime semantic_graph: anytype,
            comptime source: Analysis.FusionSelection(capacity.max_nodes),
            comptime source_region: usize,
        ) void {
            const existing = representations.*;
            var index: usize = 0;
            while (index < existing.count) : (index += 1) {
                var candidate = existing.values[index].?;
                if (!claimFusionRegion(&candidate, source, .{ .map = source_region })) continue;
                representations.insert(semantic_graph, candidate);
            }
        }

        fn branchRemap(
            representations: *RepresentationSet,
            comptime semantic_graph: anytype,
            comptime source: Analysis.RemapSelection(capacity.max_nodes),
            comptime source_region: usize,
        ) void {
            const existing = representations.*;
            var index: usize = 0;
            while (index < existing.count) : (index += 1) {
                var candidate = existing.values[index].?;
                if (!claimRemapRegion(&candidate, source, source_region)) continue;
                representations.insert(semantic_graph, candidate);
            }
        }

        fn claimFusionRegion(
            candidate: *Representation,
            comptime source: Analysis.FusionSelection(capacity.max_nodes),
            comptime source_region: Analysis.FusionRegionRef,
        ) bool {
            var nodes: [capacity.max_nodes]bool = @splat(false);
            var node_count: usize = 0;
            for (source.node_region, 0..) |maybe_region, node_id| {
                const region = maybe_region orelse continue;
                if (!std.meta.eql(region, source_region)) continue;
                if (candidate.claimed_nodes[node_id]) return false;
                nodes[node_id] = true;
                node_count += 1;
            }
            if (node_count == 0) return false;

            switch (source_region) {
                .reduction => |region_id| {
                    const group = source.reduction_storage[region_id] orelse return false;
                    if (!nodes[group.emit_node]) return false;
                    for (group.reduction_nodes[0..group.reduction_count]) |maybe_node_id| {
                        if (!nodes[maybe_node_id orelse return false]) return false;
                    }
                    for (group.nodes, 0..) |included, node_id| {
                        if (included and !nodes[node_id]) return false;
                    }
                    const destination = candidate.fusion_regions.region_count;
                    candidate.fusion_regions.reduction_storage[destination] = group;
                    for (nodes, 0..) |selected, node_id| {
                        if (selected) candidate.fusion_regions.node_region[node_id] = .{ .reduction = destination };
                    }
                    candidate.fusion_regions.region_count += 1;
                },
                .map => |region_id| {
                    const group = source.map_storage[region_id] orelse return false;
                    if (!nodes[group.root_node]) return false;
                    for (group.nodes, 0..) |included, node_id| {
                        if (included != nodes[node_id]) return false;
                    }
                    const destination = candidate.fusion_regions.map_count;
                    candidate.fusion_regions.map_storage[destination] = group;
                    for (nodes, 0..) |selected, node_id| {
                        if (selected) candidate.fusion_regions.node_region[node_id] = .{ .map = destination };
                    }
                    candidate.fusion_regions.map_count += 1;
                },
            }
            for (nodes, 0..) |selected, node_id| {
                if (!selected) continue;
                candidate.claimed_nodes[node_id] = true;
                candidate.selected_nodes += 1;
            }
            candidate.selected_regions += 1;
            return true;
        }

        fn claimLayoutRegion(
            candidate: *Representation,
            comptime source: Analysis.LayoutSelection(capacity),
            comptime source_region: usize,
        ) bool {
            const group = source.groups[source_region] orelse return false;
            if (group.tensor_count == 0 or !group.tensors[group.anchor_tensor]) return false;
            const destination = candidate.layout_regions.region_count;

            for (group.tensors, 0..) |included, tensor_id| {
                if (!included) continue;
                const result = source.results[tensor_id] orelse return false;
                if (result.tensor != tensor_id) return false;
                if (candidate.layout_regions.results[tensor_id] != null) return false;
                candidate.layout_regions.results[tensor_id] = result;
                candidate.layout_regions.tensor_region[tensor_id] = destination;
                candidate.selected_tensors += 1;
            }
            candidate.layout_regions.groups[destination] = group;
            candidate.layout_regions.region_count += 1;
            candidate.selected_regions += 1;
            return true;
        }

        fn claimRemapRegion(
            candidate: *Representation,
            comptime source: Analysis.RemapSelection(capacity.max_nodes),
            comptime source_region: usize,
        ) bool {
            var nodes: [capacity.max_nodes]bool = @splat(false);
            var node_count: usize = 0;
            for (source.node_region, 0..) |maybe_region, node_id| {
                if (maybe_region != source_region) continue;
                if (candidate.claimed_nodes[node_id]) return false;
                nodes[node_id] = true;
                node_count += 1;
            }
            if (node_count == 0) return false;
            const group = source.groups[source_region] orelse return false;
            if (!nodes[group.root_node]) return false;
            for (group.nodes, 0..) |included, node_id| {
                if (included != nodes[node_id]) return false;
            }
            const destination = candidate.remap_regions.region_count;
            candidate.remap_regions.groups[destination] = group;
            for (nodes, 0..) |selected, node_id| {
                if (!selected) continue;
                candidate.remap_regions.node_region[node_id] = destination;
                candidate.claimed_nodes[node_id] = true;
                candidate.selected_nodes += 1;
            }
            candidate.remap_regions.region_count += 1;
            candidate.selected_regions += 1;
            return true;
        }

        fn hasReductionRegion(
            comptime selection: Analysis.FusionSelection(capacity.max_nodes),
            comptime region_id: usize,
        ) bool {
            for (selection.node_region) |maybe_region| {
                const region = maybe_region orelse continue;
                if (region == .reduction and region.reduction == region_id) return true;
            }
            return false;
        }

        fn sameRepresentation(lhs: Representation, rhs: Representation) bool {
            if (lhs.selected_nodes != rhs.selected_nodes or
                lhs.selected_tensors != rhs.selected_tensors or
                lhs.selected_policies != rhs.selected_policies or
                lhs.fusion_regions.region_count != rhs.fusion_regions.region_count or
                lhs.fusion_regions.map_count != rhs.fusion_regions.map_count or
                lhs.remap_regions.region_count != rhs.remap_regions.region_count) return false;
            for (lhs.fusion_regions.node_region, rhs.fusion_regions.node_region) |left, right| {
                if (!std.meta.eql(left, right)) return false;
            }
            for (lhs.layout_regions.results, rhs.layout_regions.results) |left, right| {
                if (!std.meta.eql(left, right)) return false;
            }
            for (lhs.remap_regions.node_region, rhs.remap_regions.node_region) |left, right| {
                if (left != right) return false;
            }
            return std.mem.eql(?Contraction.Plan.Strategy, &lhs.contraction_strategies, &rhs.contraction_strategies);
        }

        fn policyClass(candidate: Representation) u16 {
            var result: u16 = 0;
            if (candidate.layout_regions.region_count != 0) result |= 1;
            if (candidate.fusion_regions.region_count != 0) result |= 2;
            if (candidate.fusion_regions.map_count != 0) result |= 4;
            if (candidate.remap_regions.region_count != 0) result |= 8;
            for (candidate.contraction_strategies) |maybe_strategy| {
                const strategy = maybe_strategy orelse continue;
                result |= @as(u16, 1) << @as(u4, @intCast(@as(usize, 4) + @backingInt(strategy)));
            }
            return result;
        }

        fn sameBoundary(lhs: Representation, rhs: Representation) bool {
            if (!std.mem.eql(bool, &lhs.claimed_nodes, &rhs.claimed_nodes)) return false;
            for (lhs.layout_regions.results, rhs.layout_regions.results) |left, right| {
                if (!std.meta.eql(left, right)) return false;
            }
            return std.mem.eql(?Contraction.Plan.Strategy, &lhs.contraction_strategies, &rhs.contraction_strategies);
        }

        fn locallyBetter(lhs: Representation, rhs: Representation) bool {
            if (lhs.local_cost.estimated_runtime_delta != rhs.local_cost.estimated_runtime_delta) {
                return lhs.local_cost.estimated_runtime_delta < rhs.local_cost.estimated_runtime_delta;
            }
            if (lhs.local_cost.traffic_saved_bytes != rhs.local_cost.traffic_saved_bytes) {
                return lhs.local_cost.traffic_saved_bytes > rhs.local_cost.traffic_saved_bytes;
            }
            if (lhs.local_cost.contraction_work_delta != rhs.local_cost.contraction_work_delta) {
                return lhs.local_cost.contraction_work_delta < rhs.local_cost.contraction_work_delta;
            }
            if (lhs.selected_nodes != rhs.selected_nodes) return lhs.selected_nodes > rhs.selected_nodes;
            if (lhs.selected_tensors != rhs.selected_tensors) return lhs.selected_tensors > rhs.selected_tensors;
            if (lhs.selected_policies != rhs.selected_policies) return lhs.selected_policies > rhs.selected_policies;
            return lhs.selected_regions < rhs.selected_regions;
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
            comptime output_binding: Output.Binding,
        ) CandidateType {
            const Validated = validation.FinalValidation(capacity).validate(executable);
            const lifetimes = LifetimeAnalysis().analyze(Validated);
            const SourcePlan = Source.Plan(SourceKey, capacity, executable, source_configuration);
            const OutputPlan = Output.Plan(executable, output_binding);
            const MemoryPlan = Storage.MemoryPlan(capacity, executable, lifetimes, SourcePlan, OutputPlan);
            const memory_traffic = estimateMemoryTraffic(executable);
            const arithmetic_work = estimateRuntime(executable);
            const access_work = estimateAccess(executable);
            const register_pressure_work = estimateRegisterPressure(executable);
            const transfer_work = memoryTransferWork(memory_traffic);
            const persistent_memory_bytes = persistentBytes(executable, lifetimes, SourcePlan, OutputPlan);

            return .{
                .executable = executable,
                .origin = origin,
                .fusion_regime = fusion_regime,
                .layout_regime = layout_regime,
                .remap_regime = remap_regime,
                .schedule = schedule,
                .cost = .{
                    .estimated_runtime_work = arithmetic_work + access_work + register_pressure_work + transfer_work,
                    .arithmetic_work = arithmetic_work,
                    .access_work = access_work,
                    .register_pressure_work = register_pressure_work,
                    .transfer_work = transfer_work,
                    .memory_traffic = memory_traffic,
                    .peak_memory_bytes = MemoryPlan.byte_count,
                    .persistent_memory_bytes = persistent_memory_bytes,
                    .scratch_memory_bytes = MemoryPlan.byte_count - persistent_memory_bytes,
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
                    .compute => |compute| work += invocationArithmetic(program, node, compute),
                }
            }
            return work;
        }

        fn estimateAccess(comptime program: Program) u128 {
            var work: u128 = 0;
            for (0..program.node_ct) |node_id| {
                const node = program.nodes[node_id].?;
                const compute = switch (node.op) {
                    .view => continue,
                    .compute => |value| value,
                };
                work += switch (compute) {
                    .direct => |semantic| if (semantic == .matmul)
                        contractionAccessWork(program, node, .scalar)
                    else
                        directAccessWork(program, node),
                    .kernel => |kernel| switch (kernel) {
                        .map => |plan| mapAccessWork(program, node, plan),
                        .reduction => |plan| reductionAccessWork(program, node, plan),
                        .contraction => |plan| contractionAccessWork(program, node, plan.strategy),
                    },
                };
            }
            return work;
        }

        fn directAccessWork(comptime program: Program, comptime node: Program.Invocation) u128 {
            if (node.output_count == 0) return 0;
            const output = program.tensors[program.output_refs[node.output_start].?].?;
            const width = if (directElementwiseVectorizes(program, node, output))
                (std.simd.suggestVectorLength(output.dtype.Scalar()) orelse 1)
            else
                1;
            const steps = vectorSteps(output.shape.elementCount(), width);
            const axis = if (output.shape.rank == 0) 0 else output.shape.rank - 1;
            var work = steps * strideFactor(if (output.shape.rank == 0) 1 else output.layout.strides[axis]);
            for (0..node.input_count) |index| {
                const input = program.tensors[program.input_refs[node.input_start + index].?].?;
                work += steps * strideFactor(broadcastStrideForAxis(input, output.shape.slice(), axis));
            }
            return work;
        }

        fn mapAccessWork(comptime program: Program, comptime node: Program.Invocation, comptime plan: Execution.MapPlan) u128 {
            return switch (plan.strategy) {
                .traversal => |traversal| blk: {
                    const shape = plan.region.domain.shape;
                    const axis = if (traversal.axis_order.len == 0) 0 else traversal.axis_order[traversal.axis_order.len - 1];
                    const width = if (traversal.vector_axis != null) traversal.vector_width else 1;
                    const steps = domainSteps(shape, traversal.vector_axis, width);
                    const output = program.tensors[program.output_refs[node.output_start].?].?;
                    var work = steps * strideFactor(if (shape.len == 0) 1 else output.layout.strides[axis]);
                    for (0..node.input_count) |index| {
                        const input = program.tensors[program.input_refs[node.input_start + index].?].?;
                        work += steps * strideFactor(broadcastStrideForAxis(input, shape, axis));
                    }
                    break :blk work;
                },
                .segmented => |segmented| blk: {
                    var work: u128 = segmented.segments.len * 4;
                    for (segmented.segments) |segment| {
                        const width = switch (plan.region.body) {
                            .expression_transfer => |expression| if (expressionSegmentVectorizes(program, node, expression, segment, segmented.vector_width))
                                segmented.vector_width
                            else
                                1,
                            .transfer => if (segmentIsContiguous(segment)) segmented.vector_width else 1,
                            .expression => unreachable,
                        };
                        const steps = vectorSteps(segment.elementCount(), width);
                        const axis = if (segment.rank == 0) 0 else segment.rank - 1;
                        work += steps * strideFactor(if (segment.rank == 0) 1 else segment.source_strides[axis]);
                        work += steps * strideFactor(if (segment.rank == 0) 1 else segment.destination_strides[axis]);
                    }
                    break :blk work;
                },
                .loop => |loop_plan| blk: {
                    const elements = product(plan.region.domain.shape);
                    const steps = vectorSteps(elements, loop_plan.vector_width);
                    break :blk steps * (node.input_count + 1) + @as(u128, elements) * 2;
                },
            };
        }

        fn reductionAccessWork(comptime program: Program, comptime node: Program.Invocation, comptime plan: Execution.ReductionPlan) u128 {
            const shape = plan.region.domain.shape;
            const axis = plan.traversal_plan.vector_axis orelse if (plan.traversal_plan.reduction_axis_order.len != 0)
                plan.traversal_plan.reduction_axis_order[plan.traversal_plan.reduction_axis_order.len - 1]
            else
                @as(u8, @intCast(shape.len - 1));
            const steps = domainSteps(shape, plan.traversal_plan.vector_axis, plan.traversal_plan.vector_width);
            var work: u128 = 0;
            for (0..node.input_count) |index| {
                const input = program.tensors[program.input_refs[node.input_start + index].?].?;
                work += steps * strideFactor(broadcastStrideForAxis(input, shape, axis));
            }
            for (0..node.output_count) |index| {
                const output = program.tensors[program.output_refs[node.output_start + index].?].?;
                work += output.shape.elementCount();
            }
            return work;
        }

        fn contractionAccessWork(comptime program: Program, comptime node: Program.Invocation, comptime strategy: Execution.ContractionPlan.Strategy) u128 {
            const lhs = program.tensors[program.input_refs[node.input_start].?].?;
            const rhs = program.tensors[program.input_refs[node.input_start + 1].?].?;
            const output = program.tensors[program.output_refs[node.output_start].?].?;
            const m: u128 = lhs.shape.at(0);
            const k: u128 = lhs.shape.at(1);
            const n: u128 = rhs.shape.at(1);
            const width = std.simd.suggestVectorLength(lhs.dtype.Scalar()) orelse 1;
            return switch (strategy) {
                .output_columns => blk: {
                    const columns = vectorSteps(rhs.shape.at(1), width);
                    break :blk m * k * columns * (strideFactor(lhs.layout.strides[1]) + 1) +
                        m * columns * strideFactor(output.layout.strides[1]);
                },
                .contracted_axis => blk: {
                    const contracted = vectorSteps(lhs.shape.at(1), width);
                    break :blk m * n * contracted * 2 +
                        m * n * strideFactor(output.layout.strides[1]);
                },
                .output_rows => blk: {
                    const rows = vectorSteps(lhs.shape.at(0), width);
                    break :blk n * k * rows * (1 + strideFactor(rhs.layout.strides[0])) +
                        n * rows * strideFactor(output.layout.strides[0]);
                },
                .scalar => m * n * k * (strideFactor(lhs.layout.strides[1]) + strideFactor(rhs.layout.strides[0])) +
                    m * n * strideFactor(output.layout.strides[1]),
            };
        }

        fn broadcastStrideForAxis(comptime input: Program.TensorInfo, comptime shape: []const usize, comptime axis: usize) isize {
            if (input.shape.rank == 0) return 0;
            if (input.shape.rank > shape.len) return input.layout.strides[input.shape.rank - 1];
            if (shape.len == 0 or axis < shape.len - input.shape.rank) return 0;
            const input_axis = axis - (shape.len - input.shape.rank);
            if (input.shape.at(input_axis) == 1 and shape[axis] != 1) return 0;
            return input.layout.strides[input_axis];
        }

        fn strideFactor(comptime stride: isize) u128 {
            return if (stride == 0 or stride == 1) 1 else 3;
        }

        fn estimateRegisterPressure(comptime program: Program) u128 {
            var work: u128 = 0;
            for (0..program.node_ct) |node_id| {
                const node = program.nodes[node_id].?;
                const compute = switch (node.op) {
                    .view => continue,
                    .compute => |value| value,
                };
                const kernel = switch (compute) {
                    .direct => continue,
                    .kernel => |value| value,
                };
                switch (kernel) {
                    .map => |plan| {
                        const expression = switch (plan.region.body) {
                            .expression, .expression_transfer => |value| value,
                            .transfer => continue,
                        };
                        const demand = Analysis.Cost.peakLiveValues(expression) +
                            Analysis.Cost.maxArity(expression) + @min(node.input_count, 3) + 1;
                        switch (plan.strategy) {
                            .traversal => |traversal| {
                                const width = if (traversal.vector_axis != null) traversal.vector_width else 1;
                                const steps = domainSteps(plan.region.domain.shape, traversal.vector_axis, width);
                                work += Analysis.Cost.registerPressureWork(steps, width, demand);
                            },
                            .segmented => |segmented| {
                                for (segmented.segments) |segment| {
                                    const width = if (expressionSegmentVectorizes(program, node, expression, segment, segmented.vector_width))
                                        segmented.vector_width
                                    else
                                        1;
                                    const steps = vectorSteps(segment.elementCount(), width);
                                    work += Analysis.Cost.registerPressureWork(steps, width, demand);
                                }
                            },
                            .loop => |loop_plan| {
                                const steps = vectorSteps(product(plan.region.domain.shape), loop_plan.vector_width);
                                work += Analysis.Cost.registerPressureWork(steps, loop_plan.vector_width, demand);
                            },
                        }
                    },
                    .reduction => |plan| {
                        const expression = plan.region.expressions;
                        const width = plan.traversal_plan.vector_width;
                        const accumulator_demand = plan.region.accumulators.len * (if (width > 1) @as(usize, 2) else 1);
                        var selection_demand: usize = 0;
                        for (plan.region.accumulators) |accumulator| {
                            selection_demand += switch (accumulator.selection) {
                                .none => 0,
                                .masked => 1 + @as(usize, @intFromBool(accumulator.finalize == .mean)),
                            };
                        }
                        const demand = Analysis.Cost.peakLiveValues(expression) +
                            Analysis.Cost.maxArity(expression) + @min(node.input_count, 3) + accumulator_demand + selection_demand;
                        const steps = domainSteps(plan.region.domain.shape, plan.traversal_plan.vector_axis, width);
                        work += Analysis.Cost.registerPressureWork(steps, width, demand);
                    },
                    .contraction => {},
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
                .direct => |semantic| directWork(program, node, semantic),
                .kernel => |kernel| switch (kernel) {
                    .map => |plan| mapWork(program, node, plan),
                    .reduction => |plan| reductionWork(plan),
                    .contraction => |plan| contractionWork(program, node, plan.strategy),
                },
            };
        }

        fn directWork(
            comptime program: Program,
            comptime node: Program.Invocation,
            comptime semantic: @import("../operations/semantic.zig").Op.Compute,
        ) u128 {
            if (Elementwise.fromCompute(semantic)) |operation| {
                const elements = outputElements(program, node, 0);
                return vectorizedElementwiseWork(program, node, operation, elements);
            }
            return switch (semantic) {
                .matmul => contractionWork(program, node, .scalar),
                .sum, .mean, .min, .max => |attrs| directReductionWork(program, node, attrs.axes),
                .softmax => inputElements(program, node, 0) * 3,
                .copy, .contiguous => directCopyWork(program, node),
                .concat => directConcatWork(program, node),
                .pad, .shift, .slice_loop => outputElements(program, node, 0) *
                    @max(program.tensors[program.output_refs[node.output_start].?].?.shape.rank, 1),
                else => outputElements(program, node, 0),
            };
        }

        fn mapWork(
            comptime program: Program,
            comptime node: Program.Invocation,
            comptime plan: Execution.MapPlan,
        ) u128 {
            return switch (plan.strategy) {
                .traversal => |traversal| switch (plan.region.body) {
                    .expression => |expression| blk: {
                        const steps = domainSteps(
                            plan.region.domain.shape,
                            traversal.vector_axis,
                            traversal.vector_width,
                        );
                        break :blk steps * expressionWork(expression);
                    },
                    .transfer, .expression_transfer => unreachable,
                },
                .segmented => |segmented| blk: {
                    var work: u128 = 0;
                    for (segmented.segments) |segment| {
                        const elements = segment.elementCount();
                        switch (plan.region.body) {
                            .transfer => {
                                work += if (segmentIsContiguous(segment))
                                    transferWork(elements, segmented.vector_width)
                                else
                                    @as(u128, elements) * @max(segment.rank, 1);
                            },
                            .expression_transfer => |expression| {
                                const steps = expressionSegmentSteps(
                                    program,
                                    node,
                                    expression,
                                    segment,
                                    segmented.vector_width,
                                );
                                work += steps * expressionWork(expression) +
                                    steps * @max(segment.rank, 1);
                            },
                            .expression => unreachable,
                        }
                    }
                    break :blk work;
                },
                .loop => |loop_plan| blk: {
                    const elements = product(plan.region.domain.shape);
                    const iterations = loop_plan.iterations.len;
                    const spatial = if (iterations == 0) 0 else elements / iterations;
                    const steps_per_spatial = vectorSteps(iterations, loop_plan.vector_width);
                    const expression_count = switch (plan.region.body) {
                        .expression_transfer => |expression| expressionWork(expression),
                        .transfer => 1,
                        .expression => unreachable,
                    };
                    break :blk spatial * steps_per_spatial * expression_count +
                        spatial * @max(plan.region.domain.shape.len - 1, 1);
                },
            };
        }

        fn reductionWork(comptime plan: Execution.ReductionPlan) u128 {
            var operation_count = expressionWork(plan.region.expressions);
            for (plan.region.accumulators) |accumulator| {
                operation_count += 1;
                operation_count += switch (accumulator.selection) {
                    .none => 0,
                    .masked => 2 + @as(u128, @intFromBool(accumulator.finalize == .mean)),
                };
            }
            const steps = domainSteps(
                plan.region.domain.shape,
                plan.traversal_plan.vector_axis,
                plan.traversal_plan.vector_width,
            );
            return steps * operation_count;
        }

        fn contractionWork(
            comptime program: Program,
            comptime node: Program.Invocation,
            comptime strategy: Execution.ContractionPlan.Strategy,
        ) u128 {
            const lhs = program.tensors[program.input_refs[node.input_start].?].?;
            const rhs = program.tensors[program.input_refs[node.input_start + 1].?].?;
            if (lhs.shape.rank == 2 and rhs.shape.rank == 2) {
                return Analysis.contraction.estimateWork(lhs, rhs, strategy);
            }
            return outputElements(program, node, 0);
        }

        fn vectorizedElementwiseWork(
            comptime program: Program,
            comptime node: Program.Invocation,
            comptime operation: Elementwise.Operation,
            comptime elements: usize,
        ) u128 {
            const output_id = program.output_refs[node.output_start].?;
            const output = program.tensors[output_id].?;
            if (!directElementwiseVectorizes(program, node, output)) {
                const indexing = @as(u128, elements) * @max(output.shape.rank, 1) * (node.input_count + 1);
                return indexing + @as(u128, elements) * operationWork(operation);
            }
            const width = std.simd.suggestVectorLength(output.dtype.Scalar()) orelse 1;
            return vectorSteps(elements, width) * operationWork(operation);
        }

        fn directReductionWork(
            comptime program: Program,
            comptime node: Program.Invocation,
            comptime axes: u64,
        ) u128 {
            const input = program.tensors[program.input_refs[node.input_start].?].?;
            if (@popCount(axes) == 1) {
                const axis: usize = @intCast(@ctz(axes));
                if (input.layout.strides[axis] == 1) {
                    const extent = input.shape.at(axis);
                    const outer = input.shape.elementCount() / extent;
                    const width = std.simd.suggestVectorLength(input.dtype.Scalar()) orelse 1;
                    return outer * vectorSteps(extent, width);
                }
            }
            return input.shape.elementCount() * @max(input.shape.rank, 1);
        }

        fn domainSteps(comptime shape: []const usize, comptime vector_axis: ?u8, comptime vector_width: usize) u128 {
            const elements = product(shape);
            const axis = vector_axis orelse return elements;
            const extent = shape[axis];
            return (elements / extent) * vectorSteps(extent, vector_width);
        }

        fn expressionSegmentSteps(
            comptime program: Program,
            comptime node: Program.Invocation,
            comptime expression: anytype,
            comptime segment: Execution.MapPlan.SegmentedPlan.Segment,
            comptime vector_width: usize,
        ) u128 {
            const elements = segment.elementCount();
            if (!expressionSegmentVectorizes(program, node, expression, segment, vector_width)) return elements;
            const extent = segment.extents[segment.rank - 1];
            return (elements / extent) * vectorSteps(extent, vector_width);
        }

        fn expressionSegmentVectorizes(
            comptime program: Program,
            comptime node: Program.Invocation,
            comptime expression: anytype,
            comptime segment: Execution.MapPlan.SegmentedPlan.Segment,
            comptime vector_width: usize,
        ) bool {
            if (vector_width <= 1 or segment.rank == 0 or segment.expression_rank == 0) return false;
            const inner_axis = segment.rank - 1;
            const expression_axis = segment.expression_rank - 1;
            const inner_extent = segment.extents[inner_axis];
            const expression_extent = segment.expression_shape[expression_axis];
            if (inner_extent < vector_width or
                segment.expression_strides[inner_axis] != 1 or
                segment.destination_strides[inner_axis] != 1 or
                segment.expression_offset % expression_extent + inner_extent > expression_extent)
            {
                return false;
            }
            for (segment.expression_strides[0..inner_axis]) |stride| {
                if (@mod(stride, @as(isize, @intCast(expression_extent))) != 0) return false;
            }
            for (0..node.input_count) |input_index| {
                if (!expressionUsesInput(expression, input_index)) continue;
                const tensor_id = program.input_refs[node.input_start + input_index].?;
                const input = program.tensors[tensor_id].?;
                if (input.shape.rank > segment.expression_rank) return false;
                const leading = segment.expression_rank - input.shape.rank;
                const stride = if (expression_axis < leading)
                    0
                else blk: {
                    const input_axis = expression_axis - leading;
                    break :blk if (input.shape.at(input_axis) == 1 and expression_extent != 1)
                        0
                    else
                        input.layout.strides[input_axis];
                };
                if (stride != 0 and stride != 1) return false;
            }
            return true;
        }

        fn expressionUsesInput(comptime expression: anytype, comptime input_index: usize) bool {
            for (expression.instructions) |instruction| {
                for (instruction.args[0..instruction.operation.arity()]) |reference| {
                    switch (reference) {
                        .input => |index| if (index == input_index) return true,
                        .instruction, .accumulator => {},
                    }
                }
            }
            return false;
        }

        fn expressionWork(comptime expression: anytype) u128 {
            var work: u128 = 0;
            for (expression.instructions) |instruction| work += operationWork(instruction.operation);
            return @max(work, 1);
        }

        fn operationWork(comptime operation: Elementwise.Operation) u128 {
            return switch (operation) {
                .div, .reciprocal, .sqrt => 4,
                .exp, .log => 8,
                else => 1,
            };
        }

        fn memoryTransferWork(comptime traffic: MemoryTraffic) u128 {
            const vector_bytes = (std.simd.suggestVectorLength(f32) orelse 1) * @sizeOf(f32);
            return transferUnits(traffic.bytes_read, vector_bytes) +
                transferUnits(traffic.bytes_written, vector_bytes);
        }

        fn transferUnits(comptime bytes: u128, comptime vector_bytes: usize) u128 {
            return bytes / vector_bytes + @intFromBool(bytes % vector_bytes != 0);
        }

        fn vectorSteps(comptime elements: usize, comptime vector_width: usize) u128 {
            if (vector_width <= 1) return elements;
            return elements / vector_width + elements % vector_width;
        }

        fn transferWork(comptime elements: usize, comptime vector_width: usize) u128 {
            return vectorSteps(elements, @max(vector_width, 1));
        }

        fn directCopyWork(comptime program: Program, comptime node: Program.Invocation) u128 {
            const input = program.tensors[program.input_refs[node.input_start].?].?;
            const output = program.tensors[program.output_refs[node.output_start].?].?;
            if (isRowMajorContiguous(input) and isRowMajorContiguous(output)) {
                return transferWork(output.shape.elementCount(), outputVectorWidth(program, node));
            }
            return @as(u128, output.shape.elementCount()) * @max(output.shape.rank, 1);
        }

        fn directConcatWork(comptime program: Program, comptime node: Program.Invocation) u128 {
            const output = program.tensors[program.output_refs[node.output_start].?].?;
            var all_contiguous = isRowMajorContiguous(output);
            for (0..node.input_count) |input_index| {
                const input_id = program.input_refs[node.input_start + input_index].?;
                all_contiguous = all_contiguous and isRowMajorContiguous(program.tensors[input_id].?);
            }
            if (all_contiguous) {
                return transferWork(output.shape.elementCount(), outputVectorWidth(program, node));
            }

            var work: u128 = 0;
            for (0..node.input_count) |input_index| {
                const input_id = program.input_refs[node.input_start + input_index].?;
                const input = program.tensors[input_id].?;
                work += @as(u128, input.shape.elementCount()) * @max(input.shape.rank, 1);
            }
            return work;
        }

        fn segmentIsContiguous(comptime segment: Execution.MapPlan.SegmentedPlan.Segment) bool {
            return stridesAreContiguous(segment.source_strides, segment.extents, segment.rank) and
                stridesAreContiguous(segment.destination_strides, segment.extents, segment.rank);
        }

        fn stridesAreContiguous(
            comptime strides: [Execution.MapPlan.SegmentedPlan.max_rank]isize,
            comptime extents: [Execution.MapPlan.SegmentedPlan.max_rank]usize,
            comptime rank: usize,
        ) bool {
            var expected: isize = 1;
            var axis = rank;
            while (axis > 0) {
                axis -= 1;
                if (extents[axis] > 1 and strides[axis] != expected) return false;
                expected *= @intCast(extents[axis]);
            }
            return true;
        }

        fn outputVectorWidth(comptime program: Program, comptime node: Program.Invocation) usize {
            const output = program.tensors[program.output_refs[node.output_start].?].?;
            return std.simd.suggestVectorLength(output.dtype.Scalar()) orelse 1;
        }

        fn directElementwiseVectorizes(
            comptime program: Program,
            comptime node: Program.Invocation,
            comptime output: Program.TensorInfo,
        ) bool {
            if (!isDensePositive(output)) return false;

            var all_same_dense = true;
            var all_row_major = isRowMajorContiguous(output);
            for (0..node.input_count) |input_index| {
                const input_id = program.input_refs[node.input_start + input_index].?;
                const input = program.tensors[input_id].?;
                all_same_dense = all_same_dense and isDensePositive(input) and
                    sameShape(input, output) and sameStrides(input, output);
                all_row_major = all_row_major and sameShape(input, output) and
                    isRowMajorContiguous(input);
            }
            if (all_same_dense or all_row_major) return true;

            if (node.input_count != 2 or output.shape.rank != 2) return false;
            const lhs = program.tensors[program.input_refs[node.input_start].?].?;
            const rhs = program.tensors[program.input_refs[node.input_start + 1].?].?;
            return (sameShape(lhs, output) and sameStrides(lhs, output) and isDensePositive(lhs) and
                isTrailingVectorBroadcast(rhs, output)) or
                (sameShape(rhs, output) and sameStrides(rhs, output) and isDensePositive(rhs) and
                    isTrailingVectorBroadcast(lhs, output));
        }

        fn sameShape(comptime lhs: Program.TensorInfo, comptime rhs: Program.TensorInfo) bool {
            return lhs.shape.rank == rhs.shape.rank and
                std.mem.eql(usize, lhs.shape.dims[0..lhs.shape.rank], rhs.shape.dims[0..rhs.shape.rank]);
        }

        fn sameStrides(comptime lhs: Program.TensorInfo, comptime rhs: Program.TensorInfo) bool {
            return lhs.shape.rank == rhs.shape.rank and
                std.mem.eql(isize, lhs.layout.strides[0..lhs.shape.rank], rhs.layout.strides[0..rhs.shape.rank]);
        }

        fn isTrailingVectorBroadcast(
            comptime input: Program.TensorInfo,
            comptime output: Program.TensorInfo,
        ) bool {
            if (input.shape.rank > 2) return false;
            const first_stride = broadcastStride(input, output, 0);
            const second_stride = broadcastStride(input, output, 1);
            return first_stride == 0 and second_stride == 1;
        }

        fn broadcastStride(
            comptime input: Program.TensorInfo,
            comptime output: Program.TensorInfo,
            comptime output_axis: usize,
        ) isize {
            const leading = output.shape.rank - input.shape.rank;
            if (output_axis < leading) return 0;
            const input_axis = output_axis - leading;
            if (input.shape.at(input_axis) == 1 and output.shape.at(output_axis) != 1) return 0;
            return input.layout.strides[input_axis];
        }

        fn isRowMajorContiguous(comptime info: Program.TensorInfo) bool {
            var expected: isize = 1;
            var axis = info.shape.rank;
            while (axis > 0) {
                axis -= 1;
                if (info.shape.at(axis) > 1 and info.layout.strides[axis] != expected) return false;
                expected *= @intCast(info.shape.at(axis));
            }
            return true;
        }

        fn isDensePositive(comptime info: Program.TensorInfo) bool {
            if (info.layout.offset != 0) return false;
            var consumed: [capacity.max_rank]bool = @splat(false);
            var expected: isize = 1;
            var remaining = info.shape.rank;
            while (remaining > 0) : (remaining -= 1) {
                var found: ?usize = null;
                for (0..info.shape.rank) |axis| {
                    if (consumed[axis]) continue;
                    if (info.shape.at(axis) <= 1 or info.layout.strides[axis] == expected) {
                        found = axis;
                        if (info.shape.at(axis) > 1) break;
                    }
                }
                const axis = found orelse return false;
                consumed[axis] = true;
                expected *= @intCast(info.shape.at(axis));
            }
            return true;
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

        fn persistentBytes(comptime program: Program, comptime lifetimes: anytype, comptime SourcePlan: type, comptime OutputPlan: type) usize {
            var bytes: usize = 0;
            for (0..program.tensor_ct) |tensor_id| {
                if (!program.materialized[tensor_id]) continue;
                const info = program.tensors[tensor_id].?;
                if (info.storage_tensor != tensor_id or !SourcePlan.isOwned(info) or OutputPlan.isBoundRoot(tensor_id)) continue;
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
                            .reduction => |plan| {
                                units += plan.region.expressions.instructions.len + plan.region.accumulators.len;
                                for (plan.region.accumulators) |accumulator| {
                                    units += switch (accumulator.selection) {
                                        .none => 0,
                                        .masked => 2,
                                    };
                                }
                            },
                            .contraction => units += 1,
                        },
                    },
                }
            }
            return units;
        }
    };
}

test "program representations compose independent regions and reject overlaps" {
    const capacity: Graph.Capacity = .{
        .max_nodes = 3,
        .max_input_refs = 3,
        .max_tensors = 4,
        .max_outputs = 1,
        .max_sources = 1,
        .max_rank = 1,
    };

    const fusion_candidates: Analysis.FusionCandidates(capacity.max_nodes) = comptime blk: {
        var candidates: Analysis.FusionCandidates(capacity.max_nodes) = .{};
        candidates.values[0] = .{ .regime = .unfused, .regions = .{} };

        var regions: Analysis.FusionSelection(capacity.max_nodes) = .{};
        regions.map_storage[0] = .{ .root_node = 0, .nodes = .{ true, false, false }, .node_count = 1 };
        regions.map_storage[1] = .{ .root_node = 1, .nodes = .{ false, true, false }, .node_count = 1 };
        regions.node_region[0] = .{ .map = 0 };
        regions.node_region[1] = .{ .map = 1 };
        regions.map_count = 2;
        candidates.values[1] = .{ .regime = .discovered, .regions = regions };
        candidates.count = 2;
        break :blk candidates;
    };

    const remap_candidates: Analysis.RemapCandidates(capacity.max_nodes) = comptime blk: {
        var candidates: Analysis.RemapCandidates(capacity.max_nodes) = .{};
        candidates.values[0] = .{ .regime = .direct, .regions = .{} };

        var regions: Analysis.RemapSelection(capacity.max_nodes) = .{};
        regions.groups[0] = .{ .root_node = 0, .nodes = .{ true, false, false }, .node_count = 1 };
        regions.node_region[0] = 0;
        regions.region_count = 1;
        candidates.values[1] = .{ .regime = .composed, .regions = regions };
        candidates.count = 2;
        break :blk candidates;
    };

    const representations = ExecutableSearch(capacity).composeRepresentations(
        fusion_candidates,
        comptime blk: {
            var candidates: Analysis.LayoutCandidates(capacity) = .{};
            candidates.values[0] = .{ .regime = .canonical, .regions = .{} };
            candidates.count = 1;
            break :blk candidates;
        },
        remap_candidates,
    );
    try std.testing.expectEqual(@as(usize, 6), representations.count);

    var found_both_maps = false;
    var found_map_and_remap = false;
    for (representations.values[0..representations.count]) |maybe_representation| {
        const representation = maybe_representation.?;
        try std.testing.expect(!(representation.fusion_regions.node_region[0] != null and
            representation.remap_regions.node_region[0] != null));
        found_both_maps = found_both_maps or representation.fusion_regions.map_count == 2;
        found_map_and_remap = found_map_and_remap or
            (representation.fusion_regions.node_region[1] != null and
                representation.remap_regions.node_region[0] != null);
    }
    try std.testing.expect(found_both_maps);
    try std.testing.expect(found_map_and_remap);
}

test "program representations compose independent layout regions" {
    const capacity: Graph.Capacity = .{
        .max_nodes = 1,
        .max_input_refs = 1,
        .max_tensors = 2,
        .max_outputs = 1,
        .max_sources = 1,
        .max_rank = 2,
    };
    const shape = Tensor.Shape(2).init(&.{ 4, 4 });
    const packed_layout = Tensor.Layout(2).firstAxisContiguous(shape);
    const layout_candidates: Analysis.LayoutCandidates(capacity) = comptime blk: {
        var candidates: Analysis.LayoutCandidates(capacity) = .{};
        candidates.values[0] = .{ .regime = .canonical, .regions = .{} };
        var regions: Analysis.LayoutSelection(capacity) = .{};
        regions.results[0] = .{ .tensor = 0, .layout = packed_layout };
        regions.results[1] = .{ .tensor = 1, .layout = packed_layout };
        regions.tensor_region[0] = 0;
        regions.tensor_region[1] = 1;
        regions.groups[0] = .{ .anchor_tensor = 0, .tensors = .{ true, false }, .tensor_count = 1 };
        regions.groups[1] = .{ .anchor_tensor = 1, .tensors = .{ false, true }, .tensor_count = 1 };
        regions.region_count = 2;
        candidates.values[1] = .{ .regime = .propagated, .regions = regions };
        candidates.count = 2;
        break :blk candidates;
    };
    const fusion_candidates: Analysis.FusionCandidates(capacity.max_nodes) = comptime blk: {
        var candidates: Analysis.FusionCandidates(capacity.max_nodes) = .{};
        candidates.values[0] = .{ .regime = .unfused, .regions = .{} };
        candidates.count = 1;
        break :blk candidates;
    };
    const remap_candidates: Analysis.RemapCandidates(capacity.max_nodes) = comptime blk: {
        var candidates: Analysis.RemapCandidates(capacity.max_nodes) = .{};
        candidates.values[0] = .{ .regime = .direct, .regions = .{} };
        candidates.count = 1;
        break :blk candidates;
    };

    const representations = ExecutableSearch(capacity).composeRepresentations(
        fusion_candidates,
        layout_candidates,
        remap_candidates,
    );
    try std.testing.expectEqual(@as(usize, 4), representations.count);

    var found_first_only = false;
    var found_second_only = false;
    var found_both = false;
    for (representations.values[0..representations.count]) |maybe_representation| {
        const representation = maybe_representation.?;
        const first = representation.layout_regions.results[0] != null;
        const second = representation.layout_regions.results[1] != null;
        const expected_regime: Analysis.LayoutRegime = if (first and second)
            .propagated
        else if (first or second)
            .mixed
        else
            .canonical;
        try std.testing.expectEqual(expected_regime, representation.layoutRegime(2));
        found_first_only = found_first_only or first and !second;
        found_second_only = found_second_only or !first and second;
        found_both = found_both or first and second;
    }
    try std.testing.expect(found_first_only);
    try std.testing.expect(found_second_only);
    try std.testing.expect(found_both);
}
