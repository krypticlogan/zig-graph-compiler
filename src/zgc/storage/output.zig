//! Storage policy for the complete set of graph outputs.
pub const Binding = enum { owned, bound };

pub const Snapshot = struct {
    binding: Binding,
    root_output_indices: []const ?usize,
};

pub fn fromSnapshot(comptime snapshot: Snapshot) type {
    return struct {
        pub const output_binding = snapshot.binding;
        pub const root_output_indices = snapshot.root_output_indices[0..snapshot.root_output_indices.len].*;

        pub fn outputForRoot(comptime root_id: usize) ?usize {
            return root_output_indices[root_id];
        }

        pub fn isBoundRoot(root_id: usize) bool {
            return root_output_indices[root_id] != null;
        }
    };
}

pub fn Plan(comptime program: anytype, comptime binding: Binding) type {
    var root_outputs: [program.tensor_ct]?usize = @splat(null);
    if (binding == .bound) {
        for (0..program.output_ct) |output_index| {
            const output_id = program.outputs[output_index].?;
            const root_id = program.tensors[output_id].?.storage_tensor;
            const root = program.tensors[root_id].?;
            if (output_id != root_id) {
                @compileError("bound outputs must own their storage; materialize view outputs with copy or contiguous");
            }
            if (root.origin != .node) {
                @compileError("bound outputs require computed storage; materialize source or literal outputs with copy");
            }
            if (comptime @hasField(@TypeOf(program), "materialized")) {
                if (!program.materialized[root_id]) @compileError("bound output storage must be materialized");
            }
            if (root_outputs[root_id] != null) {
                @compileError("bound outputs cannot share one storage root; use copy for a distinct output buffer");
            }
            root_outputs[root_id] = output_index;
        }
    }
    const mapped_outputs = root_outputs;
    return fromSnapshot(.{ .binding = binding, .root_output_indices = &mapped_outputs });
}
