const Dtype = @import("../storage/dtype.zig").Dtype;
const Operation = @import("../operations/elementwise.zig").Operation;

/// Arithmetic semantics for one elementwise instruction. Traversal kernels
/// supply indexing, broadcasting, scheduling, and stores.
pub inline fn evaluate(
    comptime output_dtype: Dtype,
    comptime vector_len: usize,
    comptime operation: Operation,
    params: anytype,
) output_dtype.Vector(vector_len) {
    return switch (operation) {
        .relu => @max(params[0], output_dtype.vectorZero(vector_len)),
        .exp => @exp(params[0]),
        .neg => -params[0],
        .abs => @abs(params[0]),
        .sqrt => @sqrt(params[0]),
        .log => @log(params[0]),
        .reciprocal => @as(output_dtype.Vector(vector_len), @splat(1)) / params[0],
        .add => params[0] + params[1],
        .sub => params[0] - params[1],
        .mul => params[0] * params[1],
        .div => params[0] / params[1],
        .minimum => @min(params[0], params[1]),
        .maximum => @max(params[0], params[1]),
        .clamp => @min(@max(params[0], params[1]), params[2]),
        .equal => params[0] == params[1],
        .not_equal => params[0] != params[1],
        .less_than => params[0] < params[1],
        .less_equal => params[0] <= params[1],
        .greater_than => params[0] > params[1],
        .greater_equal => params[0] >= params[1],
        .logical_not => @select(bool, params[0], @as(@Vector(vector_len, bool), @splat(false)), @as(@Vector(vector_len, bool), @splat(true))),
        .logical_and => @select(bool, params[0], params[1], @as(@Vector(vector_len, bool), @splat(false))),
        .logical_or => @select(bool, params[0], @as(@Vector(vector_len, bool), @splat(true)), params[1]),
        .select => @select(output_dtype.Scalar(), params[0], params[1], params[2]),
    };
}

pub inline fn evaluateScalar(
    comptime output_dtype: Dtype,
    comptime operation: Operation,
    params: anytype,
) output_dtype.Scalar() {
    var vectors: VectorizedParams(@TypeOf(params)) = undefined;
    inline for (params, 0..) |param, index| vectors[index] = @splat(param);
    return evaluate(output_dtype, 1, operation, vectors)[0];
}

fn VectorizedParams(comptime Params: type) type {
    return switch (@typeInfo(Params)) {
        .array => |array| [array.len]@Vector(1, array.child),
        .@"struct" => |structure| blk: {
            var types: [structure.fields.len]type = undefined;
            for (structure.fields, 0..) |field, index| types[index] = @Vector(1, field.type);
            break :blk @import("std").meta.Tuple(&types);
        },
        else => @compileError("pointwise parameters must be an array or tuple"),
    };
}
