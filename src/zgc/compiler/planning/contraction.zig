const std = @import("std");
const Region = @import("../optimization/regions/root.zig");
const RegionAccess = @import("../optimization/regions/access.zig");
const Graph = @import("../../core/graph.zig");
const Tensor = @import("../../core/tensor.zig");

/// Concrete kernel strategy selected for a logical contraction region.
pub const Plan = struct {
    region: Region.Contraction,
    strategy: Strategy,

    pub const Strategy = enum {
        output_columns,
        contracted_axis,
        output_rows,
        scalar,
    };
};

pub fn plan(
    comptime capacity: Graph.Capacity,
    lhs: Tensor.Info(capacity.max_rank),
    rhs: Tensor.Info(capacity.max_rank),
    output: Tensor.Info(capacity.max_rank),
    region: Region.Contraction,
) Plan {
    return .{
        .region = region,
        .strategy = selectStrategy(capacity, lhs, rhs, output),
    };
}

/// Build the concrete contraction plan and its executable tensor bindings.
pub fn planned(
    comptime capacity: Graph.Capacity,
    comptime graph: anytype,
    comptime lhs_id: usize,
    comptime rhs_id: usize,
    comptime output_id: usize,
) type {
    const output = graph.tensors[output_id].?;
    const domain_shape = output.shape.dims[0..output.shape.rank].*;
    const loads = [_]RegionAccess.Load{
        .{ .input = 0 },
        .{ .input = 1 },
    };
    const stores = [_]RegionAccess.Store{.{
        .output = 0,
        .value = .contraction,
    }};
    const selected = plan(
        capacity,
        graph.tensors[lhs_id].?,
        graph.tensors[rhs_id].?,
        output,
        .{
            .domain = .{ .shape = &domain_shape },
            .loads = &loads,
            .stores = &stores,
        },
    );
    return struct {
        pub const kernel_plan: Plan = selected;
    };
}

pub fn selectOutputLayout(
    comptime capacity: Graph.Capacity,
    graph: anytype,
    comptime lhs_id: Tensor.Id,
    comptime rhs_id: Tensor.Id,
    shape: Tensor.Shape(capacity.max_rank),
    comptime analysis: anytype,
) Tensor.Layout(capacity.max_rank) {
    packWeights(graph, rhs_id, analysis);

    const vector_len = std.simd.suggestVectorLength(f32) orelse return .contiguous(shape);
    if (shape.rank != 2 or shape.at(0) < vector_len) return .contiguous(shape);

    var lhs = &graph.tensors[lhs_id].?;
    if (isBatchLayout(capacity, lhs.*)) return .firstAxisContiguous(shape);
    if (analysis.use_counts[lhs_id] != 1) return .contiguous(shape);

    const can_relayout = switch (lhs.origin) {
        .source => |source_id| graph.sources[source_id].?.kind == .input,
        .node => lhs.storage_tensor == lhs_id,
        .literal => false,
    };
    if (!can_relayout) return .contiguous(shape);

    lhs.layout = .firstAxisContiguous(lhs.shape);
    return .firstAxisContiguous(shape);
}

pub fn validate(
    comptime plan_value: Plan,
    comptime lhs: anytype,
    comptime rhs: anytype,
    comptime output: anytype,
) void {
    const compatible = switch (plan_value.strategy) {
        .output_columns => rhs.layout.strides[1] == 1 and output.layout.strides[1] == 1,
        .contracted_axis => lhs.layout.strides[1] == 1 and rhs.layout.strides[0] == 1,
        .output_rows => lhs.layout.strides[0] == 1 and output.layout.strides[0] == 1,
        .scalar => true,
    };
    if (!compatible) @compileError("optimized matmul strategy is incompatible with its layouts");
}

fn selectStrategy(
    comptime capacity: Graph.Capacity,
    lhs: Tensor.Info(capacity.max_rank),
    rhs: Tensor.Info(capacity.max_rank),
    output: Tensor.Info(capacity.max_rank),
) Plan.Strategy {
    if (rhs.layout.strides[1] == 1 and output.layout.strides[1] == 1) return .output_columns;
    if (lhs.layout.strides[1] == 1 and rhs.layout.strides[0] == 1) return .contracted_axis;
    if (lhs.layout.strides[0] == 1 and output.layout.strides[0] == 1) return .output_rows;
    return .scalar;
}

fn packWeights(
    graph: anytype,
    comptime rhs_id: Tensor.Id,
    comptime analysis: anytype,
) void {
    if (analysis.use_counts[rhs_id] != 1) return;
    var rhs = &graph.tensors[rhs_id].?;
    if (rhs.shape.rank != 2 or rhs.storage_tensor != rhs_id) return;
    const source_id = switch (rhs.origin) {
        .source => |id| id,
        .node, .literal => return,
    };
    switch (graph.sources[source_id].?.kind) {
        .parameter, .constant => rhs.layout = .firstAxisContiguous(rhs.shape),
        .input, .state => {},
    }
}

fn isBatchLayout(
    comptime capacity: Graph.Capacity,
    info: Tensor.Info(capacity.max_rank),
) bool {
    return info.shape.rank == 2 and
        info.layout.offset == 0 and
        info.layout.strides[0] == 1 and
        info.layout.strides[1] == @as(isize, @intCast(info.shape.at(0)));
}
