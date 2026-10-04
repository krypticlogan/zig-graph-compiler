const Tensor = @import("tensor.zig");
const Dtype = @import("../storage/dtype.zig").Dtype;
const Shape_T = Tensor.Shape_T;
pub fn Graph(comptime capacity: Capacity, comptime Operation: type) type {
    return struct {
        const Self = @This();
        pub const max_rank = capacity.max_rank;
        pub const TensorInfo = Tensor.Info(max_rank);
        pub const Node = struct {
            op: Operation,
            input_start: usize,
            input_count: usize,
            result: Tensor.Id,
        };

        nodes: [capacity.max_nodes]?Node = @splat(null),
        tensors: [capacity.max_tensors]?TensorInfo = @splat(null),
        input_refs: [capacity.max_input_refs]?Tensor.Id = @splat(null),
        outputs: [capacity.max_outputs]?Tensor.Id = @splat(null),
        sources: [capacity.max_sources]?Tensor.Source = @splat(null),

        node_ct: usize = 0,
        input_ref_ct: usize = 0,
        tensor_ct: usize = 0,
        output_ct: usize = 0,
        source_ct: usize = 0,

        pub fn init() Self {
            return Self{};
        }

        pub fn insertSource(g: *Self, comptime source_index: usize, source: Tensor.Source) void {
            g.sources[source_index] = source;
            g.source_ct += 1;
        }

        pub fn insertNode(g: *Self, node: Node) void {
            g.nodes[g.node_ct] = node;
            g.node_ct += 1;
        }

        pub fn insertTensor(g: *Self, info: TensorInfo) Tensor.Id {
            const id = g.tensor_ct;
            g.tensors[id] = info;
            g.tensor_ct += 1;
            return id;
        }

        pub fn insertRef(g: *Self, tensor_id: Tensor.Id) void {
            g.input_refs[g.input_ref_ct] = tensor_id;
            g.input_ref_ct += 1;
        }

        pub fn insertOutput(g: *Self, tensor_id: Tensor.Id) void {
            g.outputs[g.output_ct] = tensor_id;
            g.output_ct += 1;
        }
    };
}

pub const Capacity = struct {
    max_nodes: usize = 0,
    max_input_refs: usize = 0,
    max_tensors: usize = 0,
    max_outputs: usize = 0,
    max_sources: usize = 0,
    max_rank: usize = 0,
};
