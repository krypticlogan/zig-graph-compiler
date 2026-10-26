//! Rebuild completed graphs with capacities matching their populated counts.
//! IDs are preserved; source slots remain indexed by the public source enum.
const Graph = @import("../core/graph.zig");
const Tensor = @import("../core/tensor.zig");
const Semantic = @import("../operations/semantic.zig");
const Execution = @import("../execution/execution.zig");
const Program = @import("../execution/program.zig").Program;

pub fn count(comptime graph: anytype) Graph.Capacity {
    var capacity: Graph.Capacity = .{
        .max_nodes = graph.node_ct,
        .max_tensors = graph.tensor_ct,
        .max_input_refs = graph.input_ref_ct,
        .max_outputs = graph.output_ct,
    };
    for (graph.tensors[0..graph.tensor_ct]) |tensor| {
        capacity.max_rank = @max(capacity.max_rank, tensor.?.shape.rank);
    }
    for (graph.sources, 0..) |source, index| {
        if (source != null) capacity.max_sources = index + 1;
    }
    if (@hasField(@TypeOf(graph), "output_ref_ct")) {
        capacity.max_output_refs = graph.output_ref_ct;
    }
    return capacity;
}

fn tensorInfo(comptime rank: usize, comptime info: anytype) Tensor.Info(rank) {
    var result: Tensor.Info(rank) = .{
        .dtype = info.dtype,
        .shape = .init(info.shape.slice()),
        .layout = .{ .offset = info.layout.offset, .strides = @splat(0) },
        .storage_tensor = info.storage_tensor,
        .origin = info.origin,
    };
    @memcpy(result.layout.strides[0..info.shape.rank], info.layout.strides[0..info.shape.rank]);
    return result;
}

pub fn semantic(comptime graph: anytype) Graph.Graph(count(graph), Semantic.Op) {
    const capacity = count(graph);
    var result: Graph.Graph(capacity, Semantic.Op) = .init();
    for (graph.tensors[0..graph.tensor_ct]) |tensor| {
        _ = result.insertTensor(tensorInfo(capacity.max_rank, tensor.?));
    }
    for (graph.nodes[0..graph.node_ct]) |node| {
        const entry = node.?;
        result.insertNode(.{ .op = entry.op, .input_start = entry.input_start, .input_count = entry.input_count, .result = entry.result });
    }
    for (graph.input_refs[0..graph.input_ref_ct]) |reference| result.insertRef(reference.?);
    for (graph.outputs[0..graph.output_ct]) |output| result.insertOutput(output.?);
    inline for (0..capacity.max_sources) |index| {
        if (graph.sources[index]) |source| result.insertSource(index, source);
    }
    return result;
}

pub fn executable(comptime program: anytype) Program(count(program), Execution.Op) {
    const capacity = count(program);
    var result: Program(capacity, Execution.Op) = .init();
    for (program.tensors[0..program.tensor_ct], 0..) |tensor, index| {
        _ = result.insertTensor(tensorInfo(capacity.max_rank, tensor.?));
        result.materialized[index] = program.materialized[index];
    }
    for (program.nodes[0..program.node_ct]) |node| {
        const entry = node.?;
        result.insertInvocation(.{ .op = entry.op, .input_start = entry.input_start, .input_count = entry.input_count, .output_start = entry.output_start, .output_count = entry.output_count });
    }
    for (program.input_refs[0..program.input_ref_ct]) |reference| result.insertInputRef(reference.?);
    for (program.output_refs[0..program.output_ref_ct]) |reference| result.insertOutputRef(reference.?);
    for (program.outputs[0..program.output_ct]) |output| result.insertOutput(output.?);
    inline for (0..capacity.max_sources) |index| {
        if (program.sources[index]) |source| result.insertSource(index, source);
    }
    return result;
}
