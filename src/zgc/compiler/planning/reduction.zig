const std = @import("std");
const Graph = @import("../../core/graph.zig");
const Expression = @import("../optimization/fusion/expression.zig");
const RegionAccess = @import("../optimization/regions/access.zig");
const Reduction = @import("../../operations/reduction.zig");
const ExpressionBuilder = @import("expression.zig");
const Region = @import("../optimization/regions/root.zig");

/// Concrete traversal selected for a logical reduction region.
pub const Plan = struct {
    region: Region.Reduction,
    traversal_plan: TraversalPlan,

    pub const TraversalPlan = struct {
        outer_axis_order: []const u8,
        reduction_axis_order: []const u8,
        vector_axis: ?u8,
        vector_width: usize,
        accumulator_lanes: usize,
        unroll: usize = 1,
    };
};

pub fn Planner(comptime capacity: Graph.Capacity) type {
    return struct {
        pub fn plan(comptime graph: anytype, comptime group: anytype) type {
            const built = comptime buildReduction(graph, group);
            return struct {
                pub const input_ids = built.inputs[0..built.input_count].*;
                pub const output_ids = built.outputs[0..built.output_count].*;
                const instructions = built.instructions[0..built.instruction_count].*;
                const accumulators = built.accumulators[0..built.accumulator_count].*;
                const stores = built.stores[0..built.store_count].*;
                const domain_shape = built.domain_shape[0..built.domain_rank].*;
                const outer_axes = built.outer_axes[0..built.outer_axis_count].*;
                const reduction_axes = built.reduction_axis_order[0..built.reduction_axis_count].*;
                const loads = mapLoads(input_ids.len, .logical);

                pub const kernel_plan: Plan = .{
                    .region = .{
                        .domain = .{ .shape = &domain_shape },
                        .loads = &loads,
                        .expressions = .{ .instructions = &instructions },
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

        fn ReductionBuildResult(comptime graph: anytype) type {
            return struct {
                inputs: [graph.tensor_ct]usize = undefined,
                input_count: usize = 0,
                outputs: [graph.node_ct]usize = undefined,
                output_count: usize = 0,
                instructions: [graph.node_ct]Expression.Program.Instruction = undefined,
                instruction_count: usize = 0,
                accumulators: [graph.node_ct]Region.Reduction.Accumulator = undefined,
                accumulator_count: usize = 0,
                stores: [graph.node_ct]RegionAccess.Store = undefined,
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

        fn buildReduction(comptime graph: anytype, comptime group: anytype) ReductionBuildResult(graph) {
            var built: ReductionBuildResult(graph) = .{};
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
                    .update = ExpressionBuilder.buildValue(graph, group.nodes, input_id, &built),
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

        fn mapLoads(comptime count: usize, comptime access: RegionAccess.Access) [count]RegionAccess.Load {
            var loads: [count]RegionAccess.Load = undefined;
            for (0..count) |index| loads[index] = .{ .input = index, .access = access };
            return loads;
        }
    };
}
