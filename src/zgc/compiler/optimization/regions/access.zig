const Expression = @import("../fusion/expression.zig").Program;

/// Logical access attached to a region boundary. Physical offsets and strides
/// belong to the selected kernel plan, not to the region.
pub const Access = enum {
    logical,
    segmented,
    composed,
    loop,
};

pub const Load = struct {
    input: usize,
    access: Access = .logical,
};

pub const Store = struct {
    output: usize,
    access: Access = .logical,
    value: StoreValue,
};

/// Value exposed at a region boundary. Internal expression references remain
/// local to expression bodies; accumulator, contraction, and transfer results
/// are identified explicitly.
pub const StoreValue = union(enum) {
    expression: Expression.ValueRef,
    accumulator: usize,
    contraction,
    transfer,
};
