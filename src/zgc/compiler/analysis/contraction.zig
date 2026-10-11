const std = @import("std");
const Graph = @import("../../core/graph.zig");
const Semantic = @import("../../operations/semantic.zig");
const Contraction = @import("../planning/contraction.zig");

pub const Policy = struct {
    node: usize,
    strategy: Contraction.Plan.Strategy,
};

pub fn estimateWork(lhs: anytype, rhs: anytype, strategy: Contraction.Plan.Strategy) u128 {
    const m = lhs.shape.at(0);
    const k = lhs.shape.at(1);
    const n = rhs.shape.at(1);
    const width = std.simd.suggestVectorLength(lhs.dtype.Scalar()) orelse 1;
    return switch (strategy) {
        .output_columns => @as(u128, m) * k * vectorSteps(n, width) * 2,
        .contracted_axis => @as(u128, m) * n * vectorSteps(k, width) * 2,
        .output_rows => @as(u128, n) * k * vectorSteps(m, width) * 2,
        .scalar => @as(u128, m) * k * n * 2,
    };
}

fn vectorSteps(elements: usize, width: usize) u128 {
    if (width <= 1) return elements;
    return elements / width + elements % width;
}

pub fn Candidates(comptime capacity: Graph.Capacity) type {
    return struct {
        values: [capacity.max_nodes * 4]?Policy = @splat(null),
        count: usize = 0,

        fn add(self: *@This(), policy: Policy) void {
            self.values[self.count] = policy;
            self.count += 1;
        }
    };
}

/// Advertise contraction traversals independently of the chosen graph layout.
/// Search checks each policy against the composed physical layout before it
/// asks lowering to build the executable.
pub fn Analysis(comptime capacity: Graph.Capacity) type {
    return struct {
        pub fn analyze(comptime graph: Graph.Graph(capacity, Semantic.Op)) Candidates(capacity) {
            var candidates: Candidates(capacity) = .{};
            for (0..graph.node_ct) |node_id| {
                const node = graph.nodes[node_id].?;
                if (node.op != .compute or node.op.compute != .matmul) continue;
                inline for (.{ .output_columns, .contracted_axis, .output_rows, .scalar }) |strategy| {
                    candidates.add(.{ .node = node_id, .strategy = strategy });
                }
            }
            return candidates;
        }
    };
}
