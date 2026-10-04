const Dtype = @import("../storage/dtype.zig").Dtype;

/// Semantic operation supported by a fused elementwise program.
pub const Operation = enum {
    relu,
    exp,
    neg,
    abs,
    sqrt,
    log,
    reciprocal,
    add,
    sub,
    mul,
    div,
    minimum,
    maximum,
    clamp,
    equal,
    not_equal,
    less_than,
    less_equal,
    greater_than,
    greater_equal,
    logical_not,
    logical_and,
    logical_or,
    select,

    pub fn arity(operation: Operation) usize {
        return switch (operation) {
            .relu, .exp, .neg, .abs, .sqrt, .log, .reciprocal, .logical_not => 1,
            .add,
            .sub,
            .mul,
            .div,
            .minimum,
            .maximum,
            .equal,
            .not_equal,
            .less_than,
            .less_equal,
            .greater_than,
            .greater_equal,
            .logical_and,
            .logical_or,
            => 2,
            .clamp, .select => 3,
        };
    }

    /// Whether this operation is valid inside a homogeneous numeric reduction
    /// body.
    pub fn isReductionCompatible(operation: Operation) bool {
        return switch (operation) {
            .relu,
            .exp,
            .neg,
            .abs,
            .sqrt,
            .log,
            .reciprocal,
            .add,
            .sub,
            .mul,
            .div,
            .minimum,
            .maximum,
            .clamp,
            => true,
            else => false,
        };
    }

    pub fn acceptsDtype(operation: Operation, dtype: Dtype) bool {
        if (!operation.isReductionCompatible()) return false;
        const dtypes: [operation.arity()]Dtype = @splat(dtype);
        return operation.acceptsOperands(&dtypes);
    }

    pub fn acceptsOperands(operation: Operation, dtypes: []const Dtype) bool {
        if (dtypes.len != operation.arity()) return false;
        return switch (operation) {
            .relu, .neg, .abs => dtypes[0].kind() != .boolean,
            .exp, .sqrt, .log, .reciprocal => dtypes[0].kind() == .float,
            .add, .sub, .mul, .minimum, .maximum => dtypes[0] == dtypes[1] and dtypes[0].kind() != .boolean,
            .div => dtypes[0] == dtypes[1] and dtypes[0].kind() == .float,
            .clamp => dtypes[0] == dtypes[1] and dtypes[0] == dtypes[2] and dtypes[0].kind() != .boolean,
            .equal, .not_equal => dtypes[0] == dtypes[1],
            .less_than, .less_equal, .greater_than, .greater_equal => dtypes[0] == dtypes[1] and dtypes[0].kind() != .boolean,
            .logical_not => dtypes[0] == .bool,
            .logical_and, .logical_or => dtypes[0] == .bool and dtypes[1] == .bool,
            .select => dtypes[0] == .bool and dtypes[1] == dtypes[2],
        };
    }

    pub fn inferDtype(operation: Operation, dtypes: []const Dtype) Dtype {
        return switch (operation) {
            .equal,
            .not_equal,
            .less_than,
            .less_equal,
            .greater_than,
            .greater_equal,
            .logical_not,
            .logical_and,
            .logical_or,
            => .bool,
            .select => dtypes[1],
            else => dtypes[0],
        };
    }
};

pub fn fromCompute(compute: anytype) ?Operation {
    return switch (compute) {
        .relu => .relu,
        .exp => .exp,
        .neg => .neg,
        .abs => .abs,
        .sqrt => .sqrt,
        .log => .log,
        .reciprocal => .reciprocal,
        .add => .add,
        .sub => .sub,
        .mul => .mul,
        .div => .div,
        .minimum => .minimum,
        .maximum => .maximum,
        .clamp => .clamp,
        .equal => .equal,
        .not_equal => .not_equal,
        .less_than => .less_than,
        .less_equal => .less_equal,
        .greater_than => .greater_than,
        .greater_equal => .greater_equal,
        .logical_not => .logical_not,
        .logical_and => .logical_and,
        .logical_or => .logical_or,
        .where => .select,
        else => null,
    };
}
