const std = @import("std");
const Execution = @import("../execution/execution.zig");
const Executable = @import("../execution/program.zig").Executable;
const Graph = @import("../core/graph.zig");
const Source = @import("../storage/source.zig");
const Storage = @import("../storage/storage.zig");
const Tensor = @import("../core/tensor.zig");
const Elementwise = @import("../operations/elementwise.zig");
const Analysis = @import("analysis/root.zig");
const Scheduling = @import("scheduling.zig");
const validation = @import("validation.zig");
const Planning = @import("planning/root.zig");

pub fn SearchResult(comptime capacity: Graph.Capacity) type {
    return struct {
        reference: PlanCandidate(capacity),
        frontier: CandidateSet(capacity),
        generated_count: usize,
        representation_candidate_count: usize,
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

        /// Insert one completed candidate while retaining a bounded Pareto
        /// frontier. Once full, the deterministic preference order prunes the
        /// weakest incomparable candidate instead of permitting search-state
        /// growth to become a compilation failure.
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

            if (set.count == max_candidates) {
                var worst_index: usize = 0;
                for (1..set.count) |candidate_index| {
                    if (set.candidates[worst_index].?.cost.preferredTo(set.candidates[candidate_index].?.cost)) {
                        worst_index = candidate_index;
                    }
                }
                if (candidate.cost.preferredTo(set.candidates[worst_index].?.cost)) {
                    set.candidates[worst_index] = candidate;
                }
                return;
            }
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
                .mixed => 2,
                .canonical => 3,
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
        const LayoutCandidates = Analysis.LayoutCandidates(capacity);
        const RemapCandidates = Analysis.RemapCandidates(capacity.max_nodes);
        const CandidateType = PlanCandidate(capacity);
        const Result = SearchResult(capacity);
        const max_representation_candidates = 8;

        const Representation = struct {
            fusion_regions: Analysis.FusionSelection(capacity.max_nodes) = .{},
            layout_regions: Analysis.LayoutSelection(capacity) = .{},
            remap_regions: Analysis.RemapSelection(capacity.max_nodes) = .{},
            claimed_nodes: [capacity.max_nodes]bool = @splat(false),
            selected_nodes: usize = 0,
            selected_tensors: usize = 0,
            selected_regions: usize = 0,

            fn fusionRegime(representation: @This()) Analysis.FusionRegime {
                return if (representation.fusion_regions.region_count != 0 or
                    representation.fusion_regions.map_count != 0)
                    .discovered
                else
                    .unfused;
            }

            fn remapRegime(representation: @This()) Analysis.RemapRegime {
                return if (representation.remap_regions.region_count != 0) .composed else .direct;
            }

            fn layoutRegime(representation: @This(), comptime available_regions: usize) Analysis.LayoutRegime {
                if (representation.layout_regions.region_count == 0) return .canonical;
                if (representation.layout_regions.region_count == available_regions) return .propagated;
                return .mixed;
            }
        };

        const RepresentationSet = struct {
            values: [max_representation_candidates]?Representation = @splat(null),
            count: usize = 0,

            fn init() @This() {
                var result: @This() = .{};
                result.values[0] = .{};
                result.count = 1;
                return result;
            }

            fn insert(set: *@This(), candidate: Representation) void {
                for (set.values[0..set.count]) |existing| {
                    if (sameRepresentation(existing.?, candidate)) return;
                }
                if (set.count < max_representation_candidates) {
                    set.values[set.count] = candidate;
                    set.count += 1;
                    return;
                }

                // Keep the empty planned baseline and replace only a weaker
                // non-empty partial representation. This bounds search before
                // lowering while favoring candidates that eliminate more
                // materialization with fewer region boundaries.
                var worst_index: usize = 1;
                for (2..set.count) |index| {
                    if (locallyBetter(set.values[worst_index].?, set.values[index].?)) {
                        worst_index = index;
                    }
                }
                if (locallyBetter(candidate, set.values[worst_index].?)) {
                    set.values[worst_index] = candidate;
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
            comptime source_configuration: anytype,
        ) Result {
            const representations = composeRepresentations(fusion_candidates, layout_candidates, remap_candidates);
            const available_layout_regions = countLayoutRegions(layout_candidates);
            var result: Result = undefined;
            result.frontier = .{};
            result.generated_count = 0;
            result.representation_candidate_count = representations.count;
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

            inline for (0..representations.count) |representation_index| {
                const representation = representations.values[representation_index].?;
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
                );
                const schedules = Scheduling.ExecutableScheduling(capacity).enumerate(base_executable);
                inline for (schedules) |schedule| {
                    const executable = Scheduling.ExecutableScheduling(capacity).apply(base_executable, schedule);
                    result.frontier.insert(complete(
                        SourceKey,
                        executable,
                        .planned,
                        representation.fusionRegime(),
                        representation.layoutRegime(available_layout_regions),
                        representation.remapRegime(),
                        schedule.kind,
                        source_configuration,
                    ));
                    result.generated_count += 1;
                }
            }

            return result;
        }

        fn composeRepresentations(
            comptime fusion_candidates: FusionCandidates,
            comptime layout_candidates: LayoutCandidates,
            comptime remap_candidates: RemapCandidates,
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
            if (anchor.selected_regions != 0) representations.insert(anchor);

            inline for (1..layout_candidates.count) |candidate_index| {
                const selection = layout_candidates.values[candidate_index].?.regions;
                inline for (0..selection.region_count) |region_id| {
                    branchLayout(&representations, selection, region_id);
                }
            }

            inline for (1..fusion_candidates.count) |candidate_index| {
                const selection = fusion_candidates.values[candidate_index].?.regions;
                inline for (0..selection.region_count) |region_id| {
                    if (!hasReductionRegion(selection, region_id)) continue;
                    branchReduction(&representations, selection, region_id);
                }
                inline for (0..selection.map_count) |map_id| {
                    branchMap(&representations, selection, map_id);
                }
            }

            inline for (1..remap_candidates.count) |candidate_index| {
                const selection = remap_candidates.values[candidate_index].?.regions;
                inline for (0..selection.region_count) |region_id| {
                    branchRemap(&representations, selection, region_id);
                }
            }
            return representations;
        }

        fn countLayoutRegions(comptime candidates: LayoutCandidates) usize {
            var count: usize = 0;
            for (1..candidates.count) |candidate_index| {
                count += candidates.values[candidate_index].?.regions.region_count;
            }
            return count;
        }

        fn branchLayout(
            representations: *RepresentationSet,
            comptime source: Analysis.LayoutSelection(capacity),
            comptime source_region: usize,
        ) void {
            const existing = representations.*;
            var index: usize = 0;
            while (index < existing.count) : (index += 1) {
                var candidate = existing.values[index].?;
                if (!claimLayoutRegion(&candidate, source, source_region)) continue;
                representations.insert(candidate);
            }
        }

        fn branchReduction(
            representations: *RepresentationSet,
            comptime source: Analysis.FusionSelection(capacity.max_nodes),
            comptime source_region: usize,
        ) void {
            const existing = representations.*;
            var index: usize = 0;
            while (index < existing.count) : (index += 1) {
                var candidate = existing.values[index].?;
                if (!claimFusionRegion(&candidate, source, .{ .reduction = source_region })) continue;
                representations.insert(candidate);
            }
        }

        fn branchMap(
            representations: *RepresentationSet,
            comptime source: Analysis.FusionSelection(capacity.max_nodes),
            comptime source_region: usize,
        ) void {
            const existing = representations.*;
            var index: usize = 0;
            while (index < existing.count) : (index += 1) {
                var candidate = existing.values[index].?;
                if (!claimFusionRegion(&candidate, source, .{ .map = source_region })) continue;
                representations.insert(candidate);
            }
        }

        fn branchRemap(
            representations: *RepresentationSet,
            comptime source: Analysis.RemapSelection(capacity.max_nodes),
            comptime source_region: usize,
        ) void {
            const existing = representations.*;
            var index: usize = 0;
            while (index < existing.count) : (index += 1) {
                var candidate = existing.values[index].?;
                if (!claimRemapRegion(&candidate, source, source_region)) continue;
                representations.insert(candidate);
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
            return std.meta.eql(lhs.fusion_regions, rhs.fusion_regions) and
                std.meta.eql(lhs.layout_regions, rhs.layout_regions) and
                std.meta.eql(lhs.remap_regions, rhs.remap_regions);
        }

        fn locallyBetter(lhs: Representation, rhs: Representation) bool {
            if (lhs.selected_nodes != rhs.selected_nodes) return lhs.selected_nodes > rhs.selected_nodes;
            if (lhs.selected_tensors != rhs.selected_tensors) return lhs.selected_tensors > rhs.selected_tensors;
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
        ) CandidateType {
            const Validated = validation.FinalValidation(capacity).validate(executable);
            const lifetimes = LifetimeAnalysis().analyze(Validated);
            const SourcePlan = Source.Plan(SourceKey, capacity, executable, source_configuration);
            const MemoryPlan = Storage.MemoryPlan(capacity, executable, lifetimes, SourcePlan);
            const memory_traffic = estimateMemoryTraffic(executable);

            return .{
                .executable = executable,
                .origin = origin,
                .fusion_regime = fusion_regime,
                .layout_regime = layout_regime,
                .remap_regime = remap_regime,
                .schedule = schedule,
                .cost = .{
                    .estimated_runtime_work = estimateRuntime(executable) + memoryTransferWork(memory_traffic),
                    .memory_traffic = memory_traffic,
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
                    .compute => |compute| work += invocationArithmetic(program, node, compute),
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
            for (plan.region.accumulators) |_| operation_count += 1;
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
                const m = lhs.shape.at(0);
                const k = lhs.shape.at(1);
                const n = rhs.shape.at(1);
                if (strategy == .scalar) return @as(u128, m) * k * n * 2;
                const vector_width = std.simd.suggestVectorLength(lhs.dtype.Scalar()) orelse 1;
                return switch (strategy) {
                    .output_columns => @as(u128, m) * k * vectorSteps(n, vector_width) * 2,
                    .contracted_axis => @as(u128, m) * n * vectorSteps(k, vector_width) * 2,
                    .output_rows => @as(u128, n) * k * vectorSteps(m, vector_width) * 2,
                    .scalar => unreachable,
                };
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
