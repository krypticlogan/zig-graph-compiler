const std = @import("std");
const Graph = @import("../core/graph.zig");

pub const StorageRegion = struct {
    offset: usize,
    len_bytes: usize,
    alignment: usize,
};

const FreeSpan = struct {
    offset: usize,
    len_bytes: usize,
};

fn ReusePlanner(comptime allocation_capacity: usize) type {
    return struct {
        const Self = @This();
        const Active = struct {
            region: StorageRegion,
            end_node_exclusive: ?usize,
        };

        // Two spare entries cover the transient state before adjacent spans
        // are coalesced after a release or split.
        free_spans: [allocation_capacity + 2]FreeSpan = undefined,
        free_count: usize = 0,
        active: [allocation_capacity]?Active = @splat(null),
        high_water: usize = 0,

        fn releaseBefore(planner: *Self, begin_node: usize) void {
            for (&planner.active) |*maybe_active| {
                const active = maybe_active.* orelse continue;
                const end_node = active.end_node_exclusive orelse continue;
                if (end_node > begin_node) continue;
                planner.addFree(.{
                    .offset = active.region.offset,
                    .len_bytes = active.region.len_bytes,
                });
                maybe_active.* = null;
            }
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

        fn track(
            planner: *Self,
            tensor_id: usize,
            region: StorageRegion,
            end_node_exclusive: ?usize,
        ) void {
            if (region.len_bytes == 0) return;
            planner.active[tensor_id] = .{
                .region = region,
                .end_node_exclusive = end_node_exclusive,
            };
        }

        fn addFree(planner: *Self, span: FreeSpan) void {
            if (span.len_bytes == 0) return;
            if (planner.free_count == planner.free_spans.len) {
                @compileError("memory planner free-span capacity exceeded");
            }
            planner.free_spans[planner.free_count] = span;
            planner.free_count += 1;

            var index = planner.free_count - 1;
            while (index > 0 and
                planner.free_spans[index - 1].offset > planner.free_spans[index].offset)
            {
                std.mem.swap(FreeSpan, &planner.free_spans[index - 1], &planner.free_spans[index]);
                index -= 1;
            }

            var read_index: usize = 0;
            var write_count: usize = 0;
            while (read_index < planner.free_count) {
                const current = planner.free_spans[read_index];
                if (write_count > 0) {
                    var previous = &planner.free_spans[write_count - 1];
                    const previous_end = previous.offset + previous.len_bytes;
                    if (current.offset <= previous_end) {
                        previous.len_bytes = @max(
                            previous_end,
                            current.offset + current.len_bytes,
                        ) - previous.offset;
                        read_index += 1;
                        continue;
                    }
                }
                planner.free_spans[write_count] = current;
                write_count += 1;
                read_index += 1;
            }
            planner.free_count = write_count;
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
) type {
    _ = capacities;
    var memory_plan: [g.tensor_ct]?StorageRegion = @splat(null);
    var planner: ReusePlanner(g.tensor_ct) = .{};

    var max_alignment: usize = 1;
    for (0..g.tensor_ct) |tensor_id| {
        if (comptime @hasField(@TypeOf(g), "materialized")) {
            if (!g.materialized[tensor_id]) continue;
        }
        const tensor_info = g.tensors[tensor_id].?;
        if (tensor_info.storage_tensor != tensor_id) continue;
        if (!SourcePlan.isOwned(tensor_info)) continue;

        const lifetime = lifetime_analysis.tensor_lifetimes[tensor_id];
        planner.releaseBefore(lifetime.begin_node);
        const dtype = tensor_info.dtype;
        const alignment = dtype.alignment();
        const len_bytes = tensor_info.shape.elementCount() * dtype.byteSize();
        const region = planner.allocate(len_bytes, alignment);
        memory_plan[tensor_id] = region;
        planner.track(tensor_id, region, lifetime.end_node_exclusive);
        max_alignment = @max(max_alignment, alignment);
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
    };
}
