const std = @import("std");
const Graph = @import("../../core/graph.zig");
const Semantic = @import("../../operations/semantic.zig");
const Analysis = @import("../analysis.zig");
const RegionAccess = @import("../optimization/regions/access.zig");
const ExpressionBuilder = @import("expression.zig");
const Region = @import("../optimization/regions/root.zig");
const Expression = @import("../optimization/fusion/expression.zig");
const ExpressionProgram = Expression.Program;

/// Concrete execution choice for a logical map-family region.
pub const Plan = struct {
    region: Region.Map,
    strategy: Strategy,

    pub const Strategy = union(enum) {
        traversal: TraversalPlan,
        segmented: SegmentedPlan,
        loop: LoopPlan,
    };

    pub const TraversalPlan = struct {
        axis_order: []const u8,
        traversal: Traversal,
        vector_axis: ?u8,
        vector_width: usize,
        unroll: usize = 1,
    };

    pub const Traversal = enum { contiguous, strided };

    /// A fixed loop over one tensor axis. The kernel traverses the remaining
    /// domain once, evaluates the producer at each source position, and sends
    /// every iteration directly to its final destination.
    pub const LoopPlan = struct {
        axis: u8,
        iterations: []const Iteration,
        value: ExpressionProgram.ValueRef,
        vector_width: usize = 1,

        pub const Iteration = struct {
            offsets: []const isize,
            boundary: Boundary,
        };

        pub const Boundary = union(enum) {
            wrap,
            redirect: usize,
        };
    };

    /// Physical transfer program selected for a map region whose body is a
    /// pure data movement. Geometry is fixed during lowering.
    pub const SegmentedPlan = struct {
        segments: []const Segment,
        vector_width: usize = 1,

        pub const max_rank = 64;

        pub const Segment = struct {
            input: usize,
            rank: u8,
            extents: [max_rank]usize = @splat(0),
            source_offset: usize,
            source_strides: [max_rank]isize = @splat(0),
            destination_offset: usize,
            destination_strides: [max_rank]isize = @splat(0),
            expression_value: ?ExpressionProgram.ValueRef = null,
            expression_rank: u8 = 0,
            expression_shape: [max_rank]usize = @splat(0),
            expression_offset: usize = 0,
            expression_strides: [max_rank]isize = @splat(0),

            pub fn elementCount(comptime segment: Segment) usize {
                var count: usize = 1;
                for (segment.extents[0..segment.rank]) |extent| count *= extent;
                return count;
            }
        };
    };
};

pub fn Planner(comptime capacity: Graph.Capacity) type {
    return struct {
        const FusionRegions = Analysis.FusionSelection(capacity.max_nodes);
        const RemapRegions = Analysis.RemapSelection(capacity.max_nodes);

        pub fn canSink(comptime graph: anytype, comptime group: anytype, comptime remap_regions: RemapRegions) bool {
            const root_tensor = graph.nodes[group.root_node].?.result;
            const root = graph.tensors[root_tensor].?;
            if (root.storage_tensor != root_tensor or root.layout.offset != 0 or !isContiguous(root)) return false;

            for (0..graph.output_ct) |output_index| {
                const output_id = graph.outputs[output_index].?;
                if (graph.tensors[output_id].?.storage_tensor == root_tensor) return false;
            }

            var reaches_remap = false;
            for (0..graph.node_ct) |node_id| {
                if (group.nodes[node_id]) continue;
                const node = graph.nodes[node_id].?;
                switch (node.op) {
                    .view => continue,
                    .compute => {},
                }
                for (0..node.input_count) |input_index| {
                    const input_id = graph.input_refs[node.input_start + input_index].?;
                    if (graph.tensors[input_id].?.storage_tensor != root_tensor) continue;
                    if (remap_regions.node_region[node_id] == null) return false;
                    reaches_remap = true;
                }
            }
            return reaches_remap;
        }

        pub fn planMap(comptime graph: anytype, comptime group: anytype) type {
            const built = comptime buildMap(graph, group);
            return struct {
                pub const input_ids = built.inputs[0..built.input_count].*;
                pub const output_ids = built.outputs;
                const instructions = built.instructions[0..built.instruction_count].*;
                const stores = built.stores;
                const axis_order = built.axis_order[0..built.axis_count].*;
                const loads = mapLoads(input_ids.len, .logical);
                const domain_shape = graph.tensors[output_ids[0]].?.shape.dims[0..graph.tensors[output_ids[0]].?.shape.rank].*;

                pub const kernel_plan: Plan = .{
                    .region = .{
                        .domain = .{ .shape = &domain_shape },
                        .loads = &loads,
                        .body = .{ .expression = .{ .instructions = &instructions } },
                        .stores = &stores,
                    },
                    .strategy = .{ .traversal = .{
                        .axis_order = &axis_order,
                        .traversal = built.traversal,
                        .vector_axis = built.vector_axis,
                        .vector_width = built.vector_width,
                    } },
                };
            };
        }

        pub fn planRemap(
            comptime graph: anytype,
            comptime group: anytype,
            comptime fusion_regions: FusionRegions,
            comptime remap_regions: RemapRegions,
        ) type {
            const root = graph.nodes[group.root_node].?;
            if (root.op.compute == .slice_loop and canUseLoopPlan(graph, root)) {
                return PlannedLoopRemap(graph, group, root, fusion_regions, remap_regions);
            }
            const segment_count = countRemapSegments(graph, group, root.result);
            const built = comptime buildRemap(
                graph,
                group,
                root.result,
                segment_count,
                fusion_regions,
                remap_regions,
            );
            if (comptime inferLoopFromSegments(graph, root, built)) |loop| {
                return PlannedInferredLoopRemap(graph, root, built, loop);
            }
            return struct {
                pub const input_ids = built.inputs[0..built.input_count].*;
                pub const output_ids = [1]usize{root.result};
                const segments = built.segments;
                const instructions = built.instructions[0..built.instruction_count].*;
                const loads = mapLoads(
                    input_ids.len,
                    if (instructions.len == 0) .segmented else .composed,
                );
                const stores = [_]RegionAccess.Store{.{
                    .output = 0,
                    .access = .segmented,
                    .value = .transfer,
                }};
                const domain_shape = graph.tensors[root.result].?.shape.dims[0..graph.tensors[root.result].?.shape.rank].*;

                pub const kernel_plan: Plan = .{
                    .region = .{
                        .domain = .{ .shape = &domain_shape },
                        .loads = &loads,
                        .body = if (instructions.len == 0)
                            .transfer
                        else
                            .{ .expression_transfer = .{ .instructions = &instructions } },
                        .stores = &stores,
                    },
                    .strategy = .{ .segmented = .{
                        .segments = &segments,
                        .vector_width = std.simd.suggestVectorLength(
                            graph.tensors[root.result].?.dtype.Scalar(),
                        ) orelse 1,
                    } },
                };
            };
        }

        fn InferredLoop(comptime iteration_count: usize) type {
            return struct {
                offsets: [iteration_count][Plan.SegmentedPlan.max_rank]isize,
                boundaries: [iteration_count]Plan.LoopPlan.Boundary,
                source_tensor: usize,
            };
        }

        fn PlannedInferredLoopRemap(
            comptime graph: anytype,
            comptime root: anytype,
            comptime built: anytype,
            comptime loop: anytype,
        ) type {
            const output = graph.tensors[root.result].?;
            const iteration_count = output.shape.at(output.shape.rank - 1);
            return struct {
                const has_expression = built.instruction_count != 0;
                pub const input_ids = if (has_expression)
                    built.inputs[0..built.input_count].*
                else
                    [1]usize{loop.source_tensor};
                pub const output_ids = [1]usize{root.result};
                const instructions = built.instructions[0..built.instruction_count].*;
                const offset_storage = loop.offsets;
                const iterations = makeInferredLoopIterations(iteration_count, output.shape.rank, offset_storage, loop.boundaries);
                const loads = mapLoads(input_ids.len, .loop);
                const stores = [_]RegionAccess.Store{.{
                    .output = 0,
                    .access = .loop,
                    .value = .transfer,
                }};
                const domain_shape = output.shape.dims[0..output.shape.rank].*;

                pub const kernel_plan: Plan = .{
                    .region = .{
                        .domain = .{ .shape = &domain_shape },
                        .loads = &loads,
                        .body = if (has_expression)
                            .{ .expression_transfer = .{ .instructions = &instructions } }
                        else
                            .transfer,
                        .stores = &stores,
                    },
                    .strategy = .{ .loop = .{
                        .axis = @intCast(output.shape.rank - 1),
                        .iterations = &iterations,
                        .value = if (has_expression) built.segments[0].expression_value.? else .{ .input = 0 },
                        .vector_width = inferredLoopVectorWidth(graph, output, built),
                    } },
                };
            };
        }

        fn makeInferredLoopIterations(
            comptime iteration_count: usize,
            comptime rank: usize,
            comptime offsets: [iteration_count][Plan.SegmentedPlan.max_rank]isize,
            comptime boundaries: [iteration_count]Plan.LoopPlan.Boundary,
        ) [iteration_count]Plan.LoopPlan.Iteration {
            var iterations: [iteration_count]Plan.LoopPlan.Iteration = undefined;
            for (0..iteration_count) |index| {
                iterations[index] = .{
                    .offsets = offsets[index][0 .. rank - 1],
                    .boundary = boundaries[index],
                };
            }
            return iterations;
        }

        fn inferLoopFromSegments(
            comptime graph: anytype,
            comptime root: anytype,
            comptime built: anytype,
        ) ?InferredLoop(graph.tensors[root.result].?.shape.at(graph.tensors[root.result].?.shape.rank - 1)) {
            if (root.op.compute != .concat) return null;
            const output = graph.tensors[root.result].?;
            if (output.shape.rank < 2 or root.op.compute.concat.axis != output.shape.rank - 1 or !isContiguous(output)) return null;
            const rank = output.shape.rank;
            const loop_axis = rank - 1;
            const iteration_count = output.shape.at(loop_axis);
            if (iteration_count == 0 or built.segment_count == 0) return null;

            const has_expression = built.instruction_count != 0;
            const value: ?Expression.Program.ValueRef = if (has_expression)
                built.segments[0].expression_value orelse return null
            else
                null;
            const source_tensor = graph.tensors[built.source_tensors[0]].?.storage_tensor;
            const source_info = graph.tensors[source_tensor].?;
            if (!std.mem.eql(usize, source_info.shape.slice(), output.shape.slice())) return null;
            var pull_offsets: [iteration_count][Plan.SegmentedPlan.max_rank]isize = @splat(@splat(0));
            var best_elements: [iteration_count]usize = @splat(0);
            var pull_redirects: [iteration_count]?usize = @splat(null);
            var has_redirect = false;

            for (built.segments, built.source_tensors) |segment, segment_source| {
                if (segment.extents[loop_axis] != 1) return null;
                if (has_expression) {
                    if (segment.expression_value == null or !std.meta.eql(segment.expression_value.?, value.?) or
                        segment.expression_rank != rank)
                    {
                        return null;
                    }
                } else if (segment.expression_value != null or graph.tensors[segment_source].?.storage_tensor != source_tensor) {
                    return null;
                }
                const destination = decodeContiguousOffset(segment.destination_offset, output.shape.slice()) orelse return null;
                const source_shape = if (has_expression) segment.expression_shape[0..rank] else source_info.shape.slice();
                if (!std.mem.eql(usize, source_shape, output.shape.slice())) return null;
                const source_offset = if (has_expression) segment.expression_offset else segment.source_offset;
                const source = decodeContiguousOffset(source_offset, source_shape) orelse return null;
                for (0..rank) |axis| {
                    const stride = contiguousStride(output.shape.slice(), axis);
                    const source_stride = if (has_expression) segment.expression_strides[axis] else segment.source_strides[axis];
                    if (segment.extents[axis] > 1 and
                        (segment.destination_strides[axis] != stride or source_stride != stride))
                    {
                        return null;
                    }
                }
                const destination_channel = destination[loop_axis];
                const source_channel = source[loop_axis];
                if (destination_channel >= iteration_count or source_channel >= iteration_count) return null;
                if (source_channel != destination_channel) {
                    has_redirect = true;
                    for (0..loop_axis) |axis| if (source[axis] != destination[axis]) return null;
                    if (pull_redirects[destination_channel]) |existing| {
                        if (existing != source_channel) return null;
                    } else {
                        pull_redirects[destination_channel] = source_channel;
                    }
                    continue;
                }
                const elements = segment.elementCount();
                if (elements > best_elements[destination_channel]) {
                    best_elements[destination_channel] = elements;
                    for (0..loop_axis) |axis| {
                        pull_offsets[destination_channel][axis] = @as(isize, @intCast(source[axis])) -
                            @as(isize, @intCast(destination[axis]));
                    }
                }
            }
            for (best_elements) |elements| if (elements == 0) return null;

            var inferred: InferredLoop(iteration_count) = .{
                .offsets = @splat(@splat(0)),
                .boundaries = undefined,
                .source_tensor = source_tensor,
            };
            if (has_redirect) {
                var inverse: [iteration_count]?usize = @splat(null);
                for (0..iteration_count) |channel| {
                    const redirect = pull_redirects[channel] orelse if (allZero(pull_offsets[channel][0..loop_axis])) channel else return null;
                    if (inverse[redirect] != null) return null;
                    inverse[redirect] = channel;
                }
                for (0..iteration_count) |channel| {
                    const target = inverse[channel] orelse return null;
                    for (0..loop_axis) |axis| {
                        if (pull_offsets[target][axis] == std.math.minInt(isize)) return null;
                        inferred.offsets[channel][axis] = -pull_offsets[channel][axis];
                        if (pull_offsets[channel][axis] != -pull_offsets[target][axis]) return null;
                    }
                    inferred.boundaries[channel] = .{ .redirect = target };
                }
            } else {
                for (0..iteration_count) |channel| {
                    for (0..loop_axis) |axis| inferred.offsets[channel][axis] = -pull_offsets[channel][axis];
                    inferred.boundaries[channel] = .wrap;
                }
            }

            for (built.segments) |segment| {
                const destination = decodeContiguousOffset(segment.destination_offset, output.shape.slice()).?;
                const source_offset = if (has_expression) segment.expression_offset else segment.source_offset;
                const source = decodeContiguousOffset(source_offset, output.shape.slice()).?;
                const channel = destination[loop_axis];
                if (has_redirect and source[loop_axis] != channel) continue;
                if (source[loop_axis] != channel) return null;
                for (0..loop_axis) |axis| {
                    const delta = @as(isize, @intCast(source[axis])) - @as(isize, @intCast(destination[axis]));
                    if (has_redirect) {
                        if (delta != pull_offsets[channel][axis]) return null;
                    } else if (@mod(delta - pull_offsets[channel][axis], @as(isize, @intCast(output.shape.at(axis)))) != 0) {
                        return null;
                    }
                }
            }
            return inferred;
        }

        fn inferredLoopVectorWidth(comptime graph: anytype, comptime output: anytype, comptime built: anytype) usize {
            const suggested = std.simd.suggestVectorLength(output.dtype.Scalar()) orelse 1;
            return if (output.shape.at(output.shape.rank - 1) >= suggested and
                loopInputsSupportVectorAxis(graph, output, built, output.shape.rank - 1))
                suggested
            else
                1;
        }

        fn decodeContiguousOffset(
            comptime offset: usize,
            comptime shape: []const usize,
        ) ?[shape.len]usize {
            var elements: usize = 1;
            for (shape) |extent| elements *= extent;
            if (offset >= elements) return null;
            var coordinates: [shape.len]usize = @splat(0);
            var remaining = offset;
            var axis = shape.len;
            while (axis > 0) {
                axis -= 1;
                coordinates[axis] = remaining % shape[axis];
                remaining /= shape[axis];
            }
            return coordinates;
        }

        fn contiguousStride(comptime shape: []const usize, comptime axis: usize) isize {
            var stride: usize = 1;
            for (shape[axis + 1 ..]) |extent| stride *= extent;
            return @intCast(stride);
        }

        fn allZero(comptime values: []const isize) bool {
            for (values) |value| if (value != 0) return false;
            return true;
        }

        fn PlannedLoopRemap(
            comptime graph: anytype,
            comptime group: anytype,
            comptime root: anytype,
            comptime fusion_regions: FusionRegions,
            comptime remap_regions: RemapRegions,
        ) type {
            _ = group;
            const attrs = root.op.compute.slice_loop;
            const built = comptime buildLoopRemap(graph, root, fusion_regions, remap_regions);
            return struct {
                pub const input_ids = built.inputs[0..built.input_count].*;
                pub const output_ids = [1]usize{root.result};
                const instructions = built.instructions[0..built.instruction_count].*;
                const iterations = built.iterations;
                const loads = mapLoads(input_ids.len, .loop);
                const stores = [_]RegionAccess.Store{.{
                    .output = 0,
                    .access = .loop,
                    .value = .transfer,
                }};
                const domain_shape = graph.tensors[root.result].?.shape.dims[0..graph.tensors[root.result].?.shape.rank].*;

                pub const kernel_plan: Plan = .{
                    .region = .{
                        .domain = .{ .shape = &domain_shape },
                        .loads = &loads,
                        .body = if (instructions.len == 0)
                            .transfer
                        else
                            .{ .expression_transfer = .{ .instructions = &instructions } },
                        .stores = &stores,
                    },
                    .strategy = .{ .loop = .{
                        .axis = @intCast(attrs.axis),
                        .iterations = &iterations,
                        .value = built.value,
                        .vector_width = built.vector_width,
                    } },
                };
            };
        }

        fn LoopBuildResult(comptime graph: anytype, comptime iteration_count: usize) type {
            return struct {
                inputs: [graph.tensor_ct]usize = undefined,
                input_count: usize = 0,
                instructions: [graph.node_ct]Expression.Program.Instruction = undefined,
                instruction_count: usize = 0,
                values: [graph.tensor_ct]?Expression.Program.ValueRef = @splat(null),
                iterations: [iteration_count]Plan.LoopPlan.Iteration = undefined,
                value: Expression.Program.ValueRef = .{ .input = 0 },
                vector_width: usize = 1,
            };
        }

        fn buildLoopRemap(
            comptime graph: anytype,
            comptime root: anytype,
            comptime fusion_regions: FusionRegions,
            comptime remap_regions: RemapRegions,
        ) LoopBuildResult(graph, root.op.compute.slice_loop.iterations.len) {
            const attrs = root.op.compute.slice_loop;
            const input_id = graph.input_refs[root.input_start].?;
            var built: LoopBuildResult(graph, attrs.iterations.len) = .{};
            if (sunkMapGroupForTensor(graph, input_id, fusion_regions, remap_regions)) |map_group| {
                built.value = ExpressionBuilder.buildValue(graph, map_group.nodes, input_id, &built);
            } else {
                built.value = ExpressionBuilder.addInput(input_id, &built);
            }

            const redirect_mode = attrs.iterations[0].boundary == .redirect;
            for (attrs.iterations, 0..) |iteration, channel| {
                built.iterations[channel].offsets = iteration.offsets;
                built.iterations[channel].boundary = if (!redirect_mode)
                    .wrap
                else blk: {
                    var target: ?usize = null;
                    for (attrs.iterations, 0..) |candidate, destination_channel| {
                        if (candidate.boundary.redirect == channel) target = destination_channel;
                    }
                    break :blk .{ .redirect = target.? };
                };
            }

            const output = graph.tensors[root.result].?;
            const suggested = std.simd.suggestVectorLength(output.dtype.Scalar()) orelse 1;
            built.vector_width = if (@as(usize, @intCast(attrs.axis)) == output.shape.rank - 1 and
                attrs.iterations.len >= suggested and
                loopInputsSupportVectorAxis(graph, output, &built, @intCast(attrs.axis)))
                suggested
            else
                1;
            return built;
        }

        fn canUseLoopPlan(comptime graph: anytype, comptime root: anytype) bool {
            const attrs = root.op.compute.slice_loop;
            const output = graph.tensors[root.result].?;
            if (attrs.axis != output.shape.rank - 1 or attrs.iterations.len == 0) return false;
            const redirect_mode = attrs.iterations[0].boundary == .redirect;
            var seen: [attrs.iterations.len]bool = @splat(false);
            for (attrs.iterations) |iteration| {
                if ((iteration.boundary == .redirect) != redirect_mode) return false;
                if (iteration.boundary == .redirect) {
                    const redirect = iteration.boundary.redirect;
                    if (redirect >= seen.len or seen[redirect]) return false;
                    seen[redirect] = true;
                } else if (iteration.boundary != .wrap) return false;
            }
            if (redirect_mode) {
                for (attrs.iterations) |iteration| {
                    const opposite = attrs.iterations[iteration.boundary.redirect];
                    for (iteration.offsets, opposite.offsets) |offset, opposite_offset| {
                        if (opposite_offset == std.math.minInt(isize)) return false;
                        if (offset != -opposite_offset) return false;
                    }
                }
            }
            return true;
        }

        fn loopInputsSupportVectorAxis(
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

        fn mapLoads(comptime count: usize, comptime access: RegionAccess.Access) [count]RegionAccess.Load {
            var loads: [count]RegionAccess.Load = undefined;
            for (0..count) |index| loads[index] = .{ .input = index, .access = access };
            return loads;
        }

        fn RemapBuildResult(comptime graph: anytype, comptime segment_count: usize) type {
            return struct {
                inputs: [graph.tensor_ct]usize = undefined,
                input_count: usize = 0,
                segments: [segment_count]Plan.SegmentedPlan.Segment = undefined,
                source_tensors: [segment_count]usize = undefined,
                segment_count: usize = 0,
                instructions: [graph.node_ct]Expression.Program.Instruction = undefined,
                instruction_count: usize = 0,
                values: [graph.tensor_ct]?Expression.Program.ValueRef = @splat(null),
            };
        }

        fn buildRemap(
            comptime graph: anytype,
            comptime group: anytype,
            comptime root_tensor: usize,
            comptime segment_count: usize,
            comptime fusion_regions: FusionRegions,
            comptime remap_regions: RemapRegions,
        ) RemapBuildResult(graph, segment_count) {
            var built: RemapBuildResult(graph, segment_count) = .{};
            const output = graph.tensors[root_tensor].?;
            var destination_strides: [Plan.SegmentedPlan.max_rank]isize = @splat(0);
            for (0..output.shape.rank) |axis| destination_strides[axis] = output.layout.strides[axis];
            emitRemapTensor(
                graph,
                group,
                root_tensor,
                output.layout.offset,
                destination_strides,
                &built,
            );
            if (built.segment_count != segment_count) @compileError("remap segment counting disagrees with lowering");
            for (0..built.segment_count) |segment_index| {
                const source_tensor = built.source_tensors[segment_index];
                if (sunkMapGroupForTensor(graph, source_tensor, fusion_regions, remap_regions)) |map_group| {
                    var segment = &built.segments[segment_index];
                    const producer_tensor = graph.nodes[map_group.root_node].?.result;
                    const producer = graph.tensors[producer_tensor].?;
                    segment.expression_value = ExpressionBuilder.buildValue(
                        graph,
                        map_group.nodes,
                        producer_tensor,
                        &built,
                    );
                    segment.expression_rank = @intCast(producer.shape.rank);
                    segment.expression_offset = segment.source_offset;
                    segment.expression_strides = segment.source_strides;
                    for (0..producer.shape.rank) |axis| {
                        segment.expression_shape[axis] = producer.shape.at(axis);
                    }
                } else {
                    built.segments[segment_index].input = addRemapInput(source_tensor, &built);
                }
            }
            return built;
        }

        pub fn sunkMapGroupForTensor(
            comptime graph: anytype,
            comptime tensor_id: usize,
            comptime fusion_regions: FusionRegions,
            comptime remap_regions: RemapRegions,
        ) ?Analysis.MapGroup(capacity.max_nodes) {
            const storage_tensor = graph.tensors[tensor_id].?.storage_tensor;
            const producer_id = switch (graph.tensors[storage_tensor].?.origin) {
                .node => |node_id| node_id,
                .source, .literal => return null,
            };
            const map_id = switch (fusion_regions.node_region[producer_id] orelse return null) {
                .map => |id| id,
                .reduction => return null,
            };
            const group = fusion_regions.map_storage[map_id].?;
            if (group.root_node != producer_id or !canSink(graph, group, remap_regions)) return null;
            return group;
        }

        fn countRemapSegments(comptime graph: anytype, comptime group: anytype, comptime tensor_id: usize) usize {
            const producer_id = switch (graph.tensors[tensor_id].?.origin) {
                .node => |id| id,
                .source, .literal => return 1,
            };
            if (!group.nodes[producer_id]) return 1;
            const node = graph.nodes[producer_id].?;
            return switch (node.op.compute) {
                .concat => blk: {
                    var count: usize = 0;
                    for (0..node.input_count) |input_index| {
                        count += countRemapSegments(graph, group, graph.input_refs[node.input_start + input_index].?);
                    }
                    break :blk count;
                },
                .shift => |attrs| blk: {
                    const output = graph.tensors[node.result].?;
                    var count: usize = 1;
                    for (0..output.shape.rank) |axis| {
                        count *= axisRuns(output.shape.at(axis), attrs.offsets[axis], attrs.boundary).count;
                    }
                    break :blk count;
                },
                .slice_loop => |attrs| blk: {
                    const output = graph.tensors[node.result].?;
                    const loop_axis: usize = @intCast(attrs.axis);
                    var count: usize = 0;
                    for (attrs.iterations) |iteration| {
                        var iteration_count: usize = 1;
                        var offset_axis: usize = 0;
                        for (0..output.shape.rank) |axis| {
                            if (axis == loop_axis) continue;
                            iteration_count *= axisRuns(
                                output.shape.at(axis),
                                iteration.offsets[offset_axis],
                                sliceLoopBoundary(iteration.boundary),
                            ).count;
                            offset_axis += 1;
                        }
                        count += iteration_count;
                    }
                    break :blk count;
                },
                else => unreachable,
            };
        }

        fn emitRemapTensor(
            comptime graph: anytype,
            comptime group: anytype,
            comptime tensor_id: usize,
            comptime destination_offset: usize,
            comptime destination_strides: [Plan.SegmentedPlan.max_rank]isize,
            built: anytype,
        ) void {
            const producer_id = switch (graph.tensors[tensor_id].?.origin) {
                .node => |id| id,
                .source, .literal => {
                    emitLeaf(graph, tensor_id, destination_offset, destination_strides, built);
                    return;
                },
            };
            if (!group.nodes[producer_id]) {
                emitLeaf(graph, tensor_id, destination_offset, destination_strides, built);
                return;
            }

            const node = graph.nodes[producer_id].?;
            switch (node.op.compute) {
                .concat => |attrs| {
                    const axis: usize = @intCast(attrs.axis);
                    var axis_offset: usize = 0;
                    for (0..node.input_count) |input_index| {
                        const input_id = graph.input_refs[node.input_start + input_index].?;
                        const input = graph.tensors[input_id].?;
                        const offset: usize = @intCast(
                            @as(isize, @intCast(destination_offset)) +
                                @as(isize, @intCast(axis_offset)) * destination_strides[axis],
                        );
                        emitRemapTensor(graph, group, input_id, offset, destination_strides, built);
                        axis_offset += input.shape.at(axis);
                    }
                },
                .shift => |attrs| emitShift(
                    graph,
                    node,
                    attrs,
                    destination_offset,
                    destination_strides,
                    built,
                ),
                .slice_loop => |attrs| emitSliceLoop(
                    graph,
                    node,
                    attrs,
                    destination_offset,
                    destination_strides,
                    built,
                ),
                else => unreachable,
            }
        }

        fn emitLeaf(
            comptime graph: anytype,
            comptime tensor_id: usize,
            comptime destination_offset: usize,
            comptime destination_strides: [Plan.SegmentedPlan.max_rank]isize,
            built: anytype,
        ) void {
            const input = graph.tensors[tensor_id].?;
            var segment: Plan.SegmentedPlan.Segment = .{
                .input = 0,
                .rank = @intCast(input.shape.rank),
                .source_offset = input.layout.offset,
                .destination_offset = destination_offset,
                .destination_strides = destination_strides,
            };
            for (0..input.shape.rank) |axis| {
                segment.extents[axis] = input.shape.at(axis);
                segment.source_strides[axis] = input.layout.strides[axis];
            }
            built.source_tensors[built.segment_count] = tensor_id;
            built.segments[built.segment_count] = segment;
            built.segment_count += 1;
        }

        fn emitShift(
            comptime graph: anytype,
            comptime node: anytype,
            comptime attrs: Semantic.Op.Compute.ShiftAttrs,
            comptime destination_offset: usize,
            comptime destination_strides: [Plan.SegmentedPlan.max_rank]isize,
            built: anytype,
        ) void {
            const input_id = graph.input_refs[node.input_start].?;
            const output = graph.tensors[node.result].?;
            if (output.shape.rank == 0) {
                emitLeaf(graph, input_id, destination_offset, destination_strides, built);
                return;
            }
            const extents: [Plan.SegmentedPlan.max_rank]usize = @splat(0);
            const source_starts: [Plan.SegmentedPlan.max_rank]usize = @splat(0);
            const source_steps: [Plan.SegmentedPlan.max_rank]isize = @splat(0);
            const destination_starts: [Plan.SegmentedPlan.max_rank]usize = @splat(0);
            emitShiftAxis(
                graph,
                node,
                attrs,
                input_id,
                output,
                destination_offset,
                destination_strides,
                0,
                false,
                extents,
                source_starts,
                source_steps,
                destination_starts,
                built,
            );
        }

        fn emitSliceLoop(
            comptime graph: anytype,
            comptime node: anytype,
            comptime attrs: Semantic.Op.Compute.SliceLoopAttrs,
            comptime destination_offset: usize,
            comptime destination_strides: [Plan.SegmentedPlan.max_rank]isize,
            built: anytype,
        ) void {
            const input_id = graph.input_refs[node.input_start].?;
            const output = graph.tensors[node.result].?;
            inline for (attrs.iterations, 0..) |iteration, iteration_index| {
                emitSliceLoopAxis(
                    graph,
                    input_id,
                    output,
                    attrs,
                    iteration,
                    iteration_index,
                    destination_offset,
                    destination_strides,
                    0,
                    0,
                    false,
                    @splat(0),
                    @splat(0),
                    @splat(0),
                    @splat(0),
                    built,
                );
            }
        }

        fn emitSliceLoopAxis(
            comptime graph: anytype,
            comptime input_id: usize,
            comptime output: anytype,
            comptime attrs: Semantic.Op.Compute.SliceLoopAttrs,
            comptime iteration: Semantic.Op.Compute.SliceLoopAttrs.Iteration,
            comptime iteration_index: usize,
            comptime destination_offset: usize,
            comptime destination_strides: [Plan.SegmentedPlan.max_rank]isize,
            comptime axis: usize,
            comptime offset_axis: usize,
            comptime outside: bool,
            comptime extents: [Plan.SegmentedPlan.max_rank]usize,
            comptime source_starts: [Plan.SegmentedPlan.max_rank]usize,
            comptime source_steps: [Plan.SegmentedPlan.max_rank]isize,
            comptime destination_starts: [Plan.SegmentedPlan.max_rank]usize,
            built: anytype,
        ) void {
            const loop_axis: usize = @intCast(attrs.axis);
            if (axis == loop_axis) {
                var next_extents = extents;
                var next_source_starts = source_starts;
                var next_source_steps = source_steps;
                var next_destination_starts = destination_starts;
                next_extents[axis] = 1;
                next_source_starts[axis] = iteration_index;
                next_source_steps[axis] = 1;
                next_destination_starts[axis] = iteration_index;
                emitSliceLoopNextAxis(
                    graph,
                    input_id,
                    output,
                    attrs,
                    iteration,
                    iteration_index,
                    destination_offset,
                    destination_strides,
                    axis,
                    offset_axis,
                    outside,
                    next_extents,
                    next_source_starts,
                    next_source_steps,
                    next_destination_starts,
                    built,
                );
                return;
            }

            const runs = axisRuns(
                output.shape.at(axis),
                iteration.offsets[offset_axis],
                sliceLoopBoundary(iteration.boundary),
            );
            inline for (0..runs.count) |run_index| {
                const run = runs.values[run_index];
                var next_extents = extents;
                var next_source_starts = source_starts;
                var next_source_steps = source_steps;
                var next_destination_starts = destination_starts;
                next_extents[axis] = run.length;
                next_destination_starts[axis] = run.output_start;
                if (run.source_start) |start| {
                    next_source_starts[axis] = start;
                    next_source_steps[axis] = run.source_step;
                }
                emitSliceLoopNextAxis(
                    graph,
                    input_id,
                    output,
                    attrs,
                    iteration,
                    iteration_index,
                    destination_offset,
                    destination_strides,
                    axis,
                    offset_axis + 1,
                    outside or run.source_start == null,
                    next_extents,
                    next_source_starts,
                    next_source_steps,
                    next_destination_starts,
                    built,
                );
            }
        }

        fn emitSliceLoopNextAxis(
            comptime graph: anytype,
            comptime input_id: usize,
            comptime output: anytype,
            comptime attrs: Semantic.Op.Compute.SliceLoopAttrs,
            comptime iteration: Semantic.Op.Compute.SliceLoopAttrs.Iteration,
            comptime iteration_index: usize,
            comptime destination_offset: usize,
            comptime destination_strides: [Plan.SegmentedPlan.max_rank]isize,
            comptime axis: usize,
            comptime offset_axis: usize,
            comptime outside: bool,
            comptime extents: [Plan.SegmentedPlan.max_rank]usize,
            comptime source_starts: [Plan.SegmentedPlan.max_rank]usize,
            comptime source_steps: [Plan.SegmentedPlan.max_rank]isize,
            comptime destination_starts: [Plan.SegmentedPlan.max_rank]usize,
            built: anytype,
        ) void {
            if (axis + 1 < output.shape.rank) {
                emitSliceLoopAxis(
                    graph,
                    input_id,
                    output,
                    attrs,
                    iteration,
                    iteration_index,
                    destination_offset,
                    destination_strides,
                    axis + 1,
                    offset_axis,
                    outside,
                    extents,
                    source_starts,
                    source_steps,
                    destination_starts,
                    built,
                );
            } else {
                emitSliceLoopSegment(
                    graph,
                    input_id,
                    output.shape.rank,
                    attrs,
                    iteration,
                    destination_offset,
                    destination_strides,
                    outside,
                    extents,
                    source_starts,
                    source_steps,
                    destination_starts,
                    built,
                );
            }
        }

        fn emitSliceLoopSegment(
            comptime graph: anytype,
            comptime input_id: usize,
            comptime rank: usize,
            comptime attrs: Semantic.Op.Compute.SliceLoopAttrs,
            comptime iteration: Semantic.Op.Compute.SliceLoopAttrs.Iteration,
            comptime destination_base: usize,
            comptime destination_strides: [Plan.SegmentedPlan.max_rank]isize,
            comptime outside: bool,
            comptime extents: [Plan.SegmentedPlan.max_rank]usize,
            comptime source_starts: [Plan.SegmentedPlan.max_rank]usize,
            comptime source_steps: [Plan.SegmentedPlan.max_rank]isize,
            comptime destination_starts: [Plan.SegmentedPlan.max_rank]usize,
            built: anytype,
        ) void {
            const input = graph.tensors[input_id].?;
            const loop_axis: usize = @intCast(attrs.axis);
            const redirect = switch (iteration.boundary) {
                .redirect => |value| value,
                .wrap, .edge, .reflect => 0,
            };
            var segment: Plan.SegmentedPlan.Segment = .{
                .input = 0,
                .rank = @intCast(rank),
                .extents = extents,
                .source_offset = input.layout.offset,
                .destination_offset = destination_base,
                .destination_strides = destination_strides,
            };
            for (0..rank) |axis| {
                segment.destination_offset = @intCast(
                    @as(isize, @intCast(segment.destination_offset)) +
                        @as(isize, @intCast(destination_starts[axis])) * destination_strides[axis],
                );
                const source_start = if (outside)
                    (if (axis == loop_axis) redirect else destination_starts[axis])
                else
                    source_starts[axis];
                segment.source_offset = @intCast(
                    @as(isize, @intCast(segment.source_offset)) +
                        @as(isize, @intCast(source_start)) * input.layout.strides[axis],
                );
                segment.source_strides[axis] = input.layout.strides[axis] *
                    (if (outside) 1 else source_steps[axis]);
            }
            built.source_tensors[built.segment_count] = input_id;
            built.segments[built.segment_count] = segment;
            built.segment_count += 1;
        }

        fn sliceLoopBoundary(comptime boundary: Semantic.Op.Compute.SliceLoopAttrs.Boundary) Semantic.Op.Compute.ShiftAttrs.Boundary {
            return switch (boundary) {
                .wrap => .wrap,
                .edge => .edge,
                .reflect => .reflect,
                .redirect => .constant,
            };
        }

        fn emitShiftAxis(
            comptime graph: anytype,
            comptime node: anytype,
            comptime attrs: Semantic.Op.Compute.ShiftAttrs,
            comptime input_id: usize,
            comptime output: anytype,
            comptime destination_offset: usize,
            comptime destination_strides: [Plan.SegmentedPlan.max_rank]isize,
            comptime axis: usize,
            comptime outside: bool,
            comptime extents: [Plan.SegmentedPlan.max_rank]usize,
            comptime source_starts: [Plan.SegmentedPlan.max_rank]usize,
            comptime source_steps: [Plan.SegmentedPlan.max_rank]isize,
            comptime destination_starts: [Plan.SegmentedPlan.max_rank]usize,
            built: anytype,
        ) void {
            const runs = axisRuns(output.shape.at(axis), attrs.offsets[axis], attrs.boundary);
            inline for (0..runs.count) |run_index| {
                const run = runs.values[run_index];
                var next_extents = extents;
                var next_source_starts = source_starts;
                var next_source_steps = source_steps;
                var next_destination_starts = destination_starts;
                next_extents[axis] = run.length;
                next_destination_starts[axis] = run.output_start;
                if (run.source_start) |start| {
                    next_source_starts[axis] = start;
                    next_source_steps[axis] = run.source_step;
                }
                const next_outside = outside or run.source_start == null;
                if (axis + 1 < output.shape.rank) {
                    emitShiftAxis(
                        graph,
                        node,
                        attrs,
                        input_id,
                        output,
                        destination_offset,
                        destination_strides,
                        axis + 1,
                        next_outside,
                        next_extents,
                        next_source_starts,
                        next_source_steps,
                        next_destination_starts,
                        built,
                    );
                } else {
                    emitShiftSegment(
                        graph,
                        node,
                        input_id,
                        output.shape.rank,
                        destination_offset,
                        destination_strides,
                        next_outside,
                        next_extents,
                        next_source_starts,
                        next_source_steps,
                        next_destination_starts,
                        built,
                    );
                }
            }
        }

        fn emitShiftSegment(
            comptime graph: anytype,
            comptime node: anytype,
            comptime input_id: usize,
            comptime rank: usize,
            comptime destination_base: usize,
            comptime destination_strides: [Plan.SegmentedPlan.max_rank]isize,
            comptime outside: bool,
            comptime extents: [Plan.SegmentedPlan.max_rank]usize,
            comptime source_starts: [Plan.SegmentedPlan.max_rank]usize,
            comptime source_steps: [Plan.SegmentedPlan.max_rank]isize,
            comptime destination_starts: [Plan.SegmentedPlan.max_rank]usize,
            built: anytype,
        ) void {
            const selected_input_id = if (outside)
                graph.input_refs[node.input_start + 1].?
            else
                input_id;
            const input = graph.tensors[selected_input_id].?;
            var segment: Plan.SegmentedPlan.Segment = .{
                .input = 0,
                .rank = @intCast(rank),
                .extents = extents,
                .source_offset = input.layout.offset,
                .destination_offset = destination_base,
                .destination_strides = destination_strides,
            };
            for (0..rank) |axis| {
                segment.destination_offset = @intCast(
                    @as(isize, @intCast(segment.destination_offset)) +
                        @as(isize, @intCast(destination_starts[axis])) * destination_strides[axis],
                );
                if (!outside) {
                    segment.source_offset = @intCast(
                        @as(isize, @intCast(segment.source_offset)) +
                            @as(isize, @intCast(source_starts[axis])) * input.layout.strides[axis],
                    );
                    segment.source_strides[axis] = input.layout.strides[axis] * source_steps[axis];
                }
            }
            built.source_tensors[built.segment_count] = selected_input_id;
            built.segments[built.segment_count] = segment;
            built.segment_count += 1;
        }

        fn addRemapInput(comptime tensor_id: usize, built: anytype) usize {
            for (built.inputs[0..built.input_count], 0..) |existing, index| {
                if (existing == tensor_id) return index;
            }
            const index = built.input_count;
            built.inputs[index] = tensor_id;
            built.input_count += 1;
            return index;
        }

        const AxisRun = struct {
            output_start: usize,
            length: usize,
            source_start: ?usize,
            source_step: isize,
        };

        fn AxisRuns(comptime extent: usize) type {
            return struct {
                values: [extent]AxisRun = undefined,
                count: usize = 0,
            };
        }

        fn axisRuns(
            comptime extent: usize,
            comptime offset: isize,
            comptime boundary: Semantic.Op.Compute.ShiftAttrs.Boundary,
        ) AxisRuns(extent) {
            var result: AxisRuns(extent) = .{};
            if (boundary == .constant) {
                if (offset >= 0) {
                    const displacement: usize = @intCast(offset);
                    if (displacement >= extent) {
                        result.values[0] = .{
                            .output_start = 0,
                            .length = extent,
                            .source_start = null,
                            .source_step = 0,
                        };
                        result.count = 1;
                        return result;
                    }
                    if (displacement != 0) {
                        result.values[result.count] = .{
                            .output_start = 0,
                            .length = displacement,
                            .source_start = null,
                            .source_step = 0,
                        };
                        result.count += 1;
                    }
                    result.values[result.count] = .{
                        .output_start = displacement,
                        .length = extent - displacement,
                        .source_start = 0,
                        .source_step = 1,
                    };
                    result.count += 1;
                    return result;
                }

                const displacement: usize = @intCast(@abs(offset));
                if (displacement >= extent) {
                    result.values[0] = .{
                        .output_start = 0,
                        .length = extent,
                        .source_start = null,
                        .source_step = 0,
                    };
                    result.count = 1;
                    return result;
                }
                result.values[0] = .{
                    .output_start = 0,
                    .length = extent - displacement,
                    .source_start = displacement,
                    .source_step = 1,
                };
                result.count = 1;
                if (displacement != 0) {
                    result.values[1] = .{
                        .output_start = extent - displacement,
                        .length = displacement,
                        .source_start = null,
                        .source_step = 0,
                    };
                    result.count = 2;
                }
                return result;
            }
            var coordinate: usize = 0;
            while (coordinate < extent) {
                const first = remapCoordinate(coordinate, extent, offset, boundary);
                var length: usize = 1;
                var step: isize = 0;
                if (coordinate + 1 < extent) {
                    const second = remapCoordinate(coordinate + 1, extent, offset, boundary);
                    if (first == null and second == null) {
                        while (coordinate + length < extent and
                            remapCoordinate(coordinate + length, extent, offset, boundary) == null)
                        {
                            length += 1;
                        }
                    } else if (first != null and second != null) {
                        step = @as(isize, @intCast(second.?)) - @as(isize, @intCast(first.?));
                        if (step >= -1 and step <= 1) {
                            var previous = second.?;
                            length = 2;
                            while (coordinate + length < extent) {
                                const next = remapCoordinate(coordinate + length, extent, offset, boundary) orelse break;
                                if (@as(isize, @intCast(next)) - @as(isize, @intCast(previous)) != step) break;
                                previous = next;
                                length += 1;
                            }
                        }
                    }
                }
                result.values[result.count] = .{
                    .output_start = coordinate,
                    .length = length,
                    .source_start = first,
                    .source_step = step,
                };
                result.count += 1;
                coordinate += length;
            }
            return result;
        }

        fn remapCoordinate(
            coordinate: usize,
            extent: usize,
            comptime offset: isize,
            comptime boundary: Semantic.Op.Compute.ShiftAttrs.Boundary,
        ) ?usize {
            return switch (boundary) {
                .wrap => blk: {
                    const displacement: usize = @intCast(@mod(@as(i128, offset), @as(i128, @intCast(extent))));
                    break :blk if (coordinate >= displacement)
                        coordinate - displacement
                    else
                        extent - (displacement - coordinate);
                },
                .edge => blk: {
                    if (offset >= 0) {
                        const displacement: usize = @intCast(offset);
                        break :blk if (displacement >= extent or coordinate < displacement) 0 else coordinate - displacement;
                    }
                    const displacement: usize = @intCast(@abs(offset));
                    break :blk if (displacement >= extent or coordinate >= extent - displacement) extent - 1 else coordinate + displacement;
                },
                .reflect => blk: {
                    if (extent == 1) break :blk 0;
                    const period = @as(u128, extent - 1) * 2;
                    const displacement: u128 = @intCast(@mod(@as(i128, offset), @as(i128, @intCast(period))));
                    const phase = if (@as(u128, coordinate) >= displacement)
                        @as(u128, coordinate) - displacement
                    else
                        period - (displacement - coordinate);
                    break :blk @intCast(if (phase < extent) phase else period - phase);
                },
                .constant => blk: {
                    if (offset >= 0) {
                        const displacement: usize = @intCast(offset);
                        if (displacement > coordinate) break :blk null;
                        break :blk coordinate - displacement;
                    }
                    const displacement: usize = @intCast(@abs(offset));
                    if (displacement >= extent or coordinate >= extent - displacement) break :blk null;
                    break :blk coordinate + displacement;
                },
            };
        }

        fn MapBuildResult(comptime graph: anytype) type {
            return struct {
                inputs: [graph.tensor_ct]usize = undefined,
                input_count: usize = 0,
                outputs: [1]usize = undefined,
                instructions: [graph.node_ct]Expression.Program.Instruction = undefined,
                instruction_count: usize = 0,
                stores: [1]RegionAccess.Store = undefined,
                values: [graph.tensor_ct]?Expression.Program.ValueRef = @splat(null),
                axis_order: [capacity.max_rank]u8 = @splat(0),
                axis_count: usize = 0,
                traversal: Plan.Traversal = .strided,
                vector_axis: ?u8 = null,
                vector_width: usize = 1,
            };
        }

        fn buildMap(comptime graph: anytype, comptime group: anytype) MapBuildResult(graph) {
            var built: MapBuildResult(graph) = .{};
            const root = graph.nodes[group.root_node].?;
            const root_value = ExpressionBuilder.buildValue(graph, group.nodes, root.result, &built);
            built.outputs[0] = root.result;
            built.stores[0] = .{ .output = 0, .value = .{ .expression = root_value } };

            const output = graph.tensors[root.result].?;
            built.traversal = if (isContiguous(output)) .contiguous else .strided;
            var vector_dtype = output.dtype;
            for (built.inputs[0..built.input_count]) |input_id| {
                const dtype = graph.tensors[input_id].?.dtype;
                if (dtype.kind() != .boolean) {
                    vector_dtype = dtype;
                    break;
                }
            }
            const suggested = std.simd.suggestVectorLength(vector_dtype.Scalar()) orelse 1;
            if (suggested > 1) {
                var axis = output.shape.rank;
                while (axis > 0) {
                    axis -= 1;
                    if (output.shape.at(axis) >= suggested and
                        output.layout.strides[axis] == 1 and
                        mapInputsSupportVectorAxis(graph, output, &built, axis))
                    {
                        built.vector_axis = @intCast(axis);
                        built.vector_width = suggested;
                        break;
                    }
                }
            }
            for (0..output.shape.rank) |axis| {
                if (built.vector_axis != null and axis == built.vector_axis.?) continue;
                built.axis_order[built.axis_count] = @intCast(axis);
                built.axis_count += 1;
            }
            if (built.vector_axis) |axis| {
                built.axis_order[built.axis_count] = axis;
                built.axis_count += 1;
            }
            return built;
        }

        fn mapInputsSupportVectorAxis(
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
    };
}
