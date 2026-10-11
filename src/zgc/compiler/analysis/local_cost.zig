const Contraction = @import("../planning/contraction.zig");
const Graph = @import("../../core/graph.zig");
const contraction_analysis = @import("contraction.zig");

pub const Estimate = struct {
    traffic_saved_bytes: u128 = 0,
    contraction_work_delta: i128 = 0,
};

/// Cheap partial estimate used before executable lowering. It counts only
/// traffic removed by internalized tensors and the work change of an explicit
/// contraction policy. Final candidate costs still use the executable plan.
pub fn estimate(comptime capacity: Graph.Capacity, comptime graph: anytype, representation: anytype) Estimate {
    var result: Estimate = .{};
    for (0..graph.node_ct) |node_id| {
        const node = graph.nodes[node_id].?;
        if (representation.fusion_regions.node_region[node_id]) |region| {
            const internal = switch (region) {
                .map => |id| representation.fusion_regions.map_storage[id].?.root_node != node_id,
                .reduction => |id| blk: {
                    const group = representation.fusion_regions.reduction_storage[id].?;
                    for (group.reduction_nodes[0..group.reduction_count]) |reduction_id| {
                        if (reduction_id.? == node_id) break :blk false;
                    }
                    break :blk true;
                },
            };
            if (internal) result.traffic_saved_bytes += tensorBytes(graph.tensors[node.result].?) * 2;
        } else if (representation.remap_regions.node_region[node_id]) |region_id| {
            if (representation.remap_regions.groups[region_id].?.root_node != node_id) {
                result.traffic_saved_bytes += tensorBytes(graph.tensors[node.result].?) * 2;
            }
        }

        const strategy = representation.contraction_strategies[node_id] orelse continue;
        const lhs = graph.tensors[graph.input_refs[node.input_start].?].?;
        const rhs = graph.tensors[graph.input_refs[node.input_start + 1].?].?;
        const output = graph.tensors[node.result].?;
        const default_strategy = Contraction.selectStrategy(capacity, lhs, rhs, output);
        result.contraction_work_delta += @as(i128, @intCast(contraction_analysis.estimateWork(lhs, rhs, strategy))) -
            @as(i128, @intCast(contraction_analysis.estimateWork(lhs, rhs, default_strategy)));
    }
    return result;
}

fn tensorBytes(info: anytype) u128 {
    return @as(u128, info.shape.elementCount()) * info.dtype.byteSize();
}
