const Graph = @import("../core/graph.zig");
const Tensor = @import("../core/tensor.zig");
/// A complete executable program. Its node sequence is the literal execution
/// schedule, and one invocation may write several tensors.
pub fn Executable(comptime capacity: Graph.Capacity, Operation: type) type {
    return struct {
        const Self = @This();
        pub const max_rank = capacity.max_rank;
        pub const TensorInfo = Tensor.Info(max_rank);
        pub const Invocation = struct {
            op: Operation,
            input_start: usize,
            input_count: usize,
            output_start: usize,
            output_count: usize,
        };

        nodes: [capacity.max_nodes]?Invocation = @splat(null),
        tensors: [capacity.max_tensors]?TensorInfo = @splat(null),
        input_refs: [capacity.max_input_refs]?Tensor.Id = @splat(null),
        output_refs: [capacity.max_tensors]?Tensor.Id = @splat(null),
        outputs: [capacity.max_outputs]?Tensor.Id = @splat(null),
        sources: [capacity.max_sources]?Tensor.Source = @splat(null),
        materialized: [capacity.max_tensors]bool = @splat(false),

        node_ct: usize = 0,
        input_ref_ct: usize = 0,
        output_ref_ct: usize = 0,
        tensor_ct: usize = 0,
        output_ct: usize = 0,
        source_ct: usize = 0,

        pub fn init() Self {
            return .{};
        }

        pub fn insertSource(program: *Self, comptime source_index: usize, source: Tensor.Source) void {
            program.sources[source_index] = source;
            program.source_ct += 1;
        }

        pub fn insertInvocation(program: *Self, invocation: Invocation) void {
            program.nodes[program.node_ct] = invocation;
            program.node_ct += 1;
        }

        pub fn insertTensor(program: *Self, info: TensorInfo) Tensor.Id {
            const id = program.tensor_ct;
            program.tensors[id] = info;
            program.tensor_ct += 1;
            return id;
        }

        pub fn insertInputRef(program: *Self, tensor_id: Tensor.Id) void {
            program.input_refs[program.input_ref_ct] = tensor_id;
            program.input_ref_ct += 1;
        }

        pub fn insertOutputRef(program: *Self, tensor_id: Tensor.Id) void {
            program.output_refs[program.output_ref_ct] = tensor_id;
            program.output_ref_ct += 1;
        }

        pub fn insertOutput(program: *Self, tensor_id: Tensor.Id) void {
            program.outputs[program.output_ct] = tensor_id;
            program.output_ct += 1;
        }
    };
}
