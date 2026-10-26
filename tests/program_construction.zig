const std = @import("std");
const internal = @import("zgc_internal");
const zgc = internal.api;

// Populated count crosses two chunk boundaries; capacity includes padding.
const capacity: @TypeOf(@import("fixtures/models.zig").BasicExecutable.semantic_capacity) = .{ .max_nodes = 135, .max_tensors = 136, .max_input_refs = 140, .max_outputs = 2, .max_sources = 3, .max_rank = 1, .max_output_refs = 140 };
const Op = enum { pair };
const Program = internal.Program(capacity, Op);
const metadata = blk: {
    var value: Program = .init();
    const shape = zgc.Tensor.Shape(1).init(&.{4});
    for (0..133) |id| {
        _ = value.insertTensor(.{
            .dtype = .f32,
            .shape = shape,
            .layout = .contiguous(shape),
            .storage_tensor = id,
            .origin = if (id == 0) .{ .source = 2 } else .{ .node = 999 },
        });
    }
    value.insertSource(2, .{ .kind = .input, .tensor = 0 });
    value.insertOutput(130);
    value.insertOutput(131);
    break :blk value;
};

test "chunked program construction matches column values across chunks and multi-output writes" {
    const pair = comptime blk: {
        @setEvalBranchQuota(100_000);
        var builder: internal.program_builder.Builder(capacity, Op) = .{};
        var expected = metadata;
        expected.materialized[0] = true;
        for (0..130) |id| {
            const start = expected.input_ref_ct;
            expected.insertInputRef(id);
            const output_start = expected.output_ref_ct;
            expected.insertOutputRef(id + 1);
            expected.tensors[id + 1].?.origin = .{ .node = id };
            expected.materialized[id + 1] = true;
            if (id == 129) {
                expected.insertOutputRef(131);
                expected.tensors[131].?.origin = .{ .node = id };
                expected.materialized[131] = true;
                builder.append(.pair, &.{id}, &.{ id + 1, 131 });
            } else builder.append(.pair, &.{id}, &.{id + 1});
            expected.insertInvocation(.{
                .op = .pair,
                .input_start = start,
                .input_count = 1,
                .output_start = output_start,
                .output_count = if (id == 129) 2 else 1,
            });
        }
        // Metadata deliberately retains an unused node tensor and empty source slots.
        var semantic_metadata = metadata;
        semantic_metadata.materialized[0] = true;
        break :blk .{ builder.finish(semantic_metadata), expected };
    };
    try std.testing.expectEqualDeep(pair[1], pair[0]);
    try std.testing.expect(!pair[0].materialized[132]);
    try std.testing.expectEqual(@as(usize, 1), pair[0].source_ct);
}

test "empty program construction preserves metadata and materialization" {
    const result = comptime blk: {
        var builder: internal.program_builder.Builder(capacity, Op) = .{};
        break :blk builder.finish(metadata);
    };
    try std.testing.expectEqualDeep(metadata, result);
}

test "executable frontier storage does not embed program arrays" {
    const cap = @import("fixtures/models.zig").MatmulExecutable.semantic_capacity;
    try std.testing.expect(@sizeOf(internal.search.CandidateSet(cap)) <= 256);
}

test "reference construction executes a chain spanning program chunks" {
    const Executable = comptime blk: {
        var builder = zgc.DefinitionBuilder.init();
        var value = builder.sources(enum { input }).input(.input, .f32, &.{4});
        for (0..130) |_| value = builder.copy(value);
        builder.output(value);
        break :blk builder.finish().executableWith(.{ .search = .{ .max_executables = 1 } });
    };
    try std.testing.expectEqual(@as(usize, 130), Executable.program.node_ct);
    var runtime = Executable.Runtime.init();
    try runtime.copyInput(.input, &.{ 1, 2, 3, 4 });
    runtime.run();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, runtime.outputView(0).contiguousSlice().?);
}

test "frontier eviction preserves target winners and immutable snapshots" {
    const cap = @import("fixtures/models.zig").BasicExecutable.semantic_capacity;
    const result = comptime blk: {
        @setEvalBranchQuota(1_000_000);
        var frontier: internal.search.CandidateSet(cap) = .{};
        for (0..20) |index| {
            var program: internal.Program(cap, @TypeOf(@import("fixtures/models.zig").BasicExecutable.program.nodes[0].?.op)) = .init();
            const shape = zgc.Tensor.Shape(cap.max_rank).init(&.{index + 1});
            _ = program.insertTensor(.{ .dtype = .f32, .shape = shape, .layout = .contiguous(shape), .storage_tensor = 0, .origin = .{ .source = 0 } });
            program.insertSource(0, .{ .kind = .input, .tensor = 0 });
            program.insertOutput(0);
            frontier.insert(.{
                .executable = program,
                .origin = .reference,
                .schedule = .semantic,
                .cost = .{ .estimated_runtime_work = 1000 - index * 10, .peak_memory_bytes = 100 + index, .persistent_memory_bytes = 100 + index, .scratch_memory_bytes = 0, .code_size_units = index + 1 },
            });
        }
        const snapshot = frontier;
        var improved = frontier.candidates[frontier.bestIndex(.fast)].?.*;
        improved.cost.estimated_runtime_work = 1;
        frontier.insert(improved); // Replace a duplicate without mutating the snapshot.
        break :blk .{ snapshot, frontier };
    };
    const previous = result[0];
    const current = result[1];
    try std.testing.expectEqual(@as(usize, 16), current.count);
    try std.testing.expectEqual(@as(u128, 810), previous.candidates[comptime previous.bestIndex(.fast)].?.cost.estimated_runtime_work);
    try std.testing.expectEqual(@as(u128, 1), current.candidates[comptime current.bestIndex(.fast)].?.cost.estimated_runtime_work);
    try std.testing.expectEqual(@as(usize, 100), current.candidates[comptime current.bestIndex(.memory)].?.cost.peak_memory_bytes);
    try std.testing.expectEqual(@as(usize, 1), current.candidates[comptime current.bestIndex(.small)].?.cost.code_size_units);
}
