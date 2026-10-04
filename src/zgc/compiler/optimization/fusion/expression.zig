const Elementwise = @import("../../../operations/elementwise.zig");
const Dtype = @import("../../../storage/dtype.zig").Dtype;

/// Scalar expression instructions shared by logical fusion regions.
pub const Program = struct {
    instructions: []const Instruction,

    pub const ValueRef = union(enum) {
        input: usize,
        instruction: usize,
        accumulator: usize,
    };

    pub const Instruction = struct {
        operation: Elementwise.Operation,
        dtype: Dtype,
        args: [3]ValueRef,
    };
};
