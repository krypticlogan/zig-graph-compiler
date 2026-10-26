//! Append physical invocations in bounded chunks and assemble a program once.
const Graph = @import("../core/graph.zig");
const Programs = @import("../execution/program.zig");
const Buffer = @import("../core/chunked_buffer.zig").Buffer;

pub fn Builder(comptime capacity: Graph.Capacity, comptime Operation: type) type {
    return struct {
        const Self = @This();
        const Program = Programs.Program(capacity, Operation);
        nodes: Buffer(?Program.Invocation) = .{},
        inputs: Buffer(?usize) = .{},
        outputs: Buffer(?usize) = .{},
        producers: [capacity.max_tensors]?usize = @splat(null),

        pub fn append(comptime self: *Self, comptime op: Operation, comptime inputs: anytype, comptime outputs: anytype) void {
            if (self.nodes.len >= capacity.max_nodes) @compileError("program builder exceeds node capacity");
            if (self.inputs.len + inputs.len > capacity.max_input_refs) @compileError("program builder exceeds input reference capacity");
            if (self.outputs.len + outputs.len > (capacity.max_output_refs orelse capacity.max_tensors)) @compileError("program builder exceeds output reference capacity");
            const input_start = self.inputs.len;
            const output_start = self.outputs.len;
            for (inputs) |input| self.inputs.append(tensorId(input));
            for (outputs) |output| {
                const id = tensorId(output);
                self.outputs.append(id);
                self.producers[id] = self.nodes.len;
            }
            self.nodes.append(.{
                .op = op,
                .input_start = input_start,
                .input_count = inputs.len,
                .output_start = output_start,
                .output_count = outputs.len,
            });
        }

        /// Metadata is shared throughout construction; only producer origins
        /// and materialization change when invocations are appended/reordered.
        pub fn finish(comptime self: *const Self, comptime metadata: anytype) Program {
            var nodes: [capacity.max_nodes]?Program.Invocation = @splat(null);
            var inputs: [capacity.max_input_refs]?usize = @splat(null);
            var outputs: [capacity.max_output_refs orelse capacity.max_tensors]?usize = @splat(null);
            const built_nodes = self.nodes.toArray();
            const built_inputs = self.inputs.toArray();
            const built_outputs = self.outputs.toArray();
            @memcpy(nodes[0..built_nodes.len], &built_nodes);
            @memcpy(inputs[0..built_inputs.len], &built_inputs);
            @memcpy(outputs[0..built_outputs.len], &built_outputs);

            var tensors = metadata.tensors;
            var materialized: [capacity.max_tensors]bool = @splat(false);
            for (metadata.tensors[0..metadata.tensor_ct], 0..) |tensor, id| {
                const info = tensor.?;
                materialized[id] = if (@hasField(@TypeOf(metadata), "materialized"))
                    metadata.materialized[id]
                else
                    info.origin != .node;
                if (self.producers[id]) |producer| {
                    if (info.origin != .node or info.origin.node != producer) {
                        tensors[id].?.origin = .{ .node = producer };
                    }
                    materialized[id] = true;
                }
            }
            return .{
                .nodes = nodes,
                .tensors = tensors,
                .input_refs = inputs,
                .output_refs = outputs,
                .outputs = metadata.outputs,
                .sources = metadata.sources,
                .materialized = materialized,
                .node_ct = self.nodes.len,
                .input_ref_ct = self.inputs.len,
                .output_ref_ct = self.outputs.len,
                .tensor_ct = metadata.tensor_ct,
                .output_ct = metadata.output_ct,
                .source_ct = metadata.source_ct,
            };
        }

        fn tensorId(value: anytype) usize {
            return if (@typeInfo(@TypeOf(value)) == .optional) value.? else value;
        }
    };
}
