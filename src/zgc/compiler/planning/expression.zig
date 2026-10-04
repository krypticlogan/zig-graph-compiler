const Expression = @import("../optimization/fusion/expression.zig");
const Elementwise = @import("../../operations/elementwise.zig");

pub fn buildValue(
    comptime graph: anytype,
    comptime included_nodes: anytype,
    comptime tensor_id: usize,
    built: anytype,
) Expression.Program.ValueRef {
    if (built.values[tensor_id]) |value| return value;
    const producer_id = switch (graph.tensors[tensor_id].?.origin) {
        .node => |id| id,
        .source, .literal => return addInput(tensor_id, built),
    };
    if (!included_nodes[producer_id]) return addInput(tensor_id, built);

    const producer = graph.nodes[producer_id].?;
    const operation = Elementwise.fromCompute(producer.op.compute).?;
    var args: [3]Expression.Program.ValueRef = @splat(.{ .input = 0 });
    for (0..producer.input_count) |input_index| {
        args[input_index] = buildValue(
            graph,
            included_nodes,
            graph.input_refs[producer.input_start + input_index].?,
            built,
        );
    }
    const value: Expression.Program.ValueRef = .{ .instruction = built.instruction_count };
    built.instructions[built.instruction_count] = .{
        .operation = operation,
        .dtype = graph.tensors[tensor_id].?.dtype,
        .args = args,
    };
    built.instruction_count += 1;
    built.values[tensor_id] = value;
    return value;
}

pub fn addInput(comptime tensor_id: usize, built: anytype) Expression.Program.ValueRef {
    for (built.inputs[0..built.input_count], 0..) |existing, index| {
        if (existing == tensor_id) return .{ .input = index };
    }
    const index = built.input_count;
    built.inputs[index] = tensor_id;
    built.input_count += 1;
    const value: Expression.Program.ValueRef = .{ .input = index };
    built.values[tensor_id] = value;
    return value;
}
