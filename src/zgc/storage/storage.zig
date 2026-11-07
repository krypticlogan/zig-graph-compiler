const std = @import("std");
const Graph = @import("../core/graph.zig");

pub const StorageRegion = struct {
    offset: usize,
    len_bytes: usize,
    alignment: usize,
};

/// Immutable storage result retained by executable search for the winner.
pub const Snapshot = struct {
    tensor_regions: []const ?StorageRegion,
    byte_count: usize,
    alignment: usize,
};

pub fn fromSnapshot(comptime snapshot: Snapshot) type {
    return struct {
        pub const tensor_regions = snapshot.tensor_regions[0..].*;
        pub const byte_count = snapshot.byte_count;
        pub const alignment = snapshot.alignment;
    };
}

const FreeSpan = struct {
    offset: usize,
    len_bytes: usize,
};

fn ReusePlanner(comptime allocation_capacity: usize) type {
    return struct {
        const Self = @This();
        // Two spare entries cover the transient state before adjacent spans
        // are coalesced after a release or split.
        free_spans: [allocation_capacity + 2]FreeSpan = undefined,
        free_count: usize = 0,
        high_water: usize = 0,

        fn release(planner: *Self, region: StorageRegion) void {
            planner.addFree(.{ .offset = region.offset, .len_bytes = region.len_bytes });
        }

        fn allocate(
            planner: *Self,
            len_bytes: usize,
            alignment: usize,
        ) StorageRegion {
            if (len_bytes == 0) {
                return .{ .offset = 0, .len_bytes = 0, .alignment = alignment };
            }

            var best_index: ?usize = null;
            var best_offset: usize = 0;
            var best_waste: usize = std.math.maxInt(usize);
            for (planner.free_spans[0..planner.free_count], 0..) |span, span_index| {
                const aligned_offset = std.mem.alignForward(usize, span.offset, alignment);
                const span_end = span.offset + span.len_bytes;
                if (aligned_offset > span_end or len_bytes > span_end - aligned_offset) continue;
                const waste = span.len_bytes - len_bytes;
                if (waste < best_waste or
                    (waste == best_waste and aligned_offset < best_offset))
                {
                    best_index = span_index;
                    best_offset = aligned_offset;
                    best_waste = waste;
                }
            }

            if (best_index) |span_index| {
                const span = planner.removeFree(span_index);
                const allocation_end = best_offset + len_bytes;
                if (span.offset < best_offset) {
                    planner.addFree(.{
                        .offset = span.offset,
                        .len_bytes = best_offset - span.offset,
                    });
                }
                const span_end = span.offset + span.len_bytes;
                if (allocation_end < span_end) {
                    planner.addFree(.{
                        .offset = allocation_end,
                        .len_bytes = span_end - allocation_end,
                    });
                }
                return .{
                    .offset = best_offset,
                    .len_bytes = len_bytes,
                    .alignment = alignment,
                };
            }

            const offset = std.mem.alignForward(usize, planner.high_water, alignment);
            if (planner.high_water < offset) {
                planner.addFree(.{
                    .offset = planner.high_water,
                    .len_bytes = offset - planner.high_water,
                });
            }
            planner.high_water = offset + len_bytes;
            return .{
                .offset = offset,
                .len_bytes = len_bytes,
                .alignment = alignment,
            };
        }

        fn addFree(planner: *Self, span: FreeSpan) void {
            if (span.len_bytes == 0) return;
            // Locate this span by offset, then coalesce only its neighbors.
            // Existing free spans are sorted, disjoint, and nonadjacent.
            var low: usize = 0;
            var high = planner.free_count;
            while (low < high) {
                const middle = low + (high - low) / 2;
                if (planner.free_spans[middle].offset < span.offset) low = middle + 1 else high = middle;
            }
            var index = low;
            if (index > 0 and planner.free_spans[index - 1].offset + planner.free_spans[index - 1].len_bytes >= span.offset) {
                index -= 1;
                const end = @max(planner.free_spans[index].offset + planner.free_spans[index].len_bytes, span.offset + span.len_bytes);
                planner.free_spans[index].len_bytes = end - planner.free_spans[index].offset;
            } else {
                if (planner.free_count == planner.free_spans.len) @compileError("memory planner free-span capacity exceeded");
                var cursor = planner.free_count;
                while (cursor > index) : (cursor -= 1) planner.free_spans[cursor] = planner.free_spans[cursor - 1];
                planner.free_spans[index] = span;
                planner.free_count += 1;
            }
            while (index + 1 < planner.free_count) {
                const next = planner.free_spans[index + 1];
                const end = planner.free_spans[index].offset + planner.free_spans[index].len_bytes;
                if (end < next.offset) break;
                planner.free_spans[index].len_bytes = @max(end, next.offset + next.len_bytes) - planner.free_spans[index].offset;
                _ = planner.removeFree(index + 1);
            }
        }

        fn removeFree(planner: *Self, index: usize) FreeSpan {
            const result = planner.free_spans[index];
            var current = index;
            while (current + 1 < planner.free_count) : (current += 1) {
                planner.free_spans[current] = planner.free_spans[current + 1];
            }
            planner.free_count -= 1;
            return result;
        }
    };
}

pub fn MemoryPlan(
    comptime capacities: Graph.Capacity,
    comptime g: anytype,
    comptime lifetime_analysis: anytype,
    comptime SourcePlan: type,
    comptime OutputPlan: type,
) type {
    _ = capacities;
    var memory_plan: [g.tensor_ct]?StorageRegion = @splat(null);
    var planner: ReusePlanner(g.tensor_ct) = .{};

    var max_alignment: usize = 1;
    var eligible: [g.tensor_ct]bool = @splat(false);
    var starts: [g.node_ct]?usize = @splat(null);
    var ends: [g.node_ct + 1]?usize = @splat(null);
    var next_start: [g.tensor_ct]?usize = @splat(null);
    var next_end: [g.tensor_ct]?usize = @splat(null);

    // Prepend in reverse tensor order so simultaneous events preserve the old
    // ascending-ID allocation and release order, including best-fit ties.
    for (0..g.tensor_ct) |reverse| {
        const tensor_id = g.tensor_ct - 1 - reverse;
        if (comptime @hasField(@TypeOf(g), "materialized")) {
            if (!g.materialized[tensor_id]) continue;
        }
        const info = g.tensors[tensor_id].?;
        if (info.storage_tensor != tensor_id or !SourcePlan.isOwned(info) or OutputPlan.isBoundRoot(tensor_id)) continue;
        eligible[tensor_id] = true;
        if (info.origin != .node) continue;
        const lifetime = lifetime_analysis.tensor_lifetimes[tensor_id];
        if (lifetime.begin_node >= g.node_ct) @compileError("allocation begins outside the executable");
        next_start[tensor_id] = starts[lifetime.begin_node];
        starts[lifetime.begin_node] = tensor_id;
        if (lifetime.end_node_exclusive) |end| {
            if (end <= lifetime.begin_node or end > g.node_ct) @compileError("allocation has an invalid lifetime end");
            next_end[tensor_id] = ends[end];
            ends[end] = tensor_id;
        }
    }

    // Reserve owned sources before the first invocation, in tensor ID order.
    for (0..g.tensor_ct) |tensor_id| {
        if (!eligible[tensor_id]) continue;
        const tensor_info = g.tensors[tensor_id].?;
        switch (tensor_info.origin) {
            .source => {},
            .node, .literal => continue,
        }

        const lifetime = lifetime_analysis.tensor_lifetimes[tensor_id];
        if (lifetime.end_node_exclusive != null) {
            @compileError("owned source storage must remain persistent");
        }
        const dtype = tensor_info.dtype;
        const alignment = dtype.alignment();
        const len_bytes = tensor_info.shape.elementCount() * dtype.byteSize();
        const region = planner.allocate(len_bytes, alignment);
        memory_plan[tensor_id] = region;
        max_alignment = @max(max_alignment, alignment);
    }

    // Process lifetime ends before starts at the same invocation. Each owned
    // tensor is allocated once and released at most once; no global scans here.
    for (0..g.node_ct) |begin_node| {
        var release_id = ends[begin_node];
        while (release_id) |tensor_id| {
            planner.release(memory_plan[tensor_id].?);
            release_id = next_end[tensor_id];
        }
        var allocation_id = starts[begin_node];
        while (allocation_id) |tensor_id| {
            const info = g.tensors[tensor_id].?;
            const alignment = info.dtype.alignment();
            const region = planner.allocate(info.shape.elementCount() * info.dtype.byteSize(), alignment);
            memory_plan[tensor_id] = region;
            max_alignment = @max(max_alignment, alignment);
            allocation_id = next_start[tensor_id];
        }
    }

    for (0..g.tensor_ct) |tensor_id| {
        if (comptime @hasField(@TypeOf(g), "materialized")) {
            if (!g.materialized[tensor_id]) continue;
        }
        const storage_tensor = g.tensors[tensor_id].?.storage_tensor;
        if (storage_tensor != tensor_id) {
            memory_plan[tensor_id] = memory_plan[storage_tensor];
        }
    }

    const regions = memory_plan;
    const total_bytes = planner.high_water;
    const storage_alignment = max_alignment;

    return struct {
        pub const tensor_regions: [g.tensor_ct]?StorageRegion = regions;
        pub const byte_count: usize = total_bytes;
        pub const alignment: usize = storage_alignment;
        pub const snapshot: Snapshot = .{
            .tensor_regions = &regions,
            .byte_count = total_bytes,
            .alignment = storage_alignment,
        };
    };
}
