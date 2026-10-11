//! Chainable graph expressions for definition construction.
const std = @import("std");
const definition = @import("definition.zig");
const DefinitionBuilder = definition.DefinitionBuilder;
const Value = definition.Value;
const Expr = @This();

b: *DefinitionBuilder,
value: Value,

/// Apply a custom expression function to the current expression.
pub fn apply(comptime self: Expr, comptime operation: anytype, comptime args: anytype) Expr {
    assertExprFn(operation);
    const field_types = @typeInfo(@TypeOf(args)).@"struct".field_types;
    var argument_types: [field_types.len + 1]type = undefined;
    argument_types[0] = Expr;
    inline for (field_types, 1..) |field_type, index| argument_types[index] = field_type;

    var call_args: @Tuple(&argument_types) = undefined;
    call_args[0] = self;
    inline for (0..field_types.len) |index| call_args[index + 1] = args[index];
    return @call(.auto, operation, call_args);
}

pub fn relu(comptime self: Expr) Expr {
    return self.wrap(self.b.relu(self.value));
}

pub fn exp(comptime self: Expr) Expr {
    return self.wrap(self.b.exp(self.value));
}

pub fn neg(comptime self: Expr) Expr {
    return self.wrap(self.b.neg(self.value));
}

pub fn abs(comptime self: Expr) Expr {
    return self.wrap(self.b.abs(self.value));
}

pub fn sqrt(comptime self: Expr) Expr {
    return self.wrap(self.b.sqrt(self.value));
}

pub fn log(comptime self: Expr) Expr {
    return self.wrap(self.b.log(self.value));
}

pub fn reciprocal(comptime self: Expr) Expr {
    return self.wrap(self.b.reciprocal(self.value));
}

pub fn add(comptime self: Expr, comptime rhs: Expr) Expr {
    return self.wrap(self.b.add(self.value, rhs.value));
}

pub fn sub(comptime self: Expr, comptime rhs: Expr) Expr {
    return self.wrap(self.b.sub(self.value, rhs.value));
}

pub fn mul(comptime self: Expr, comptime rhs: Expr) Expr {
    return self.wrap(self.b.mul(self.value, rhs.value));
}

pub fn div(comptime self: Expr, comptime rhs: Expr) Expr {
    return self.wrap(self.b.div(self.value, rhs.value));
}

pub fn minimum(comptime self: Expr, comptime rhs: Expr) Expr {
    return self.wrap(self.b.minimum(self.value, rhs.value));
}

pub fn maximum(comptime self: Expr, comptime rhs: Expr) Expr {
    return self.wrap(self.b.maximum(self.value, rhs.value));
}

pub fn clamp(comptime self: Expr, comptime lower: Expr, comptime upper: Expr) Expr {
    return self.wrap(self.b.clamp(self.value, lower.value, upper.value));
}

pub fn equal(comptime self: Expr, comptime rhs: Expr) Expr {
    return self.wrap(self.b.equal(self.value, rhs.value));
}

pub fn notEqual(comptime self: Expr, comptime rhs: Expr) Expr {
    return self.wrap(self.b.notEqual(self.value, rhs.value));
}

pub fn lessThan(comptime self: Expr, comptime rhs: Expr) Expr {
    return self.wrap(self.b.lessThan(self.value, rhs.value));
}

pub fn lessEqual(comptime self: Expr, comptime rhs: Expr) Expr {
    return self.wrap(self.b.lessEqual(self.value, rhs.value));
}

pub fn greaterThan(comptime self: Expr, comptime rhs: Expr) Expr {
    return self.wrap(self.b.greaterThan(self.value, rhs.value));
}

pub fn greaterEqual(comptime self: Expr, comptime rhs: Expr) Expr {
    return self.wrap(self.b.greaterEqual(self.value, rhs.value));
}

pub fn logicalNot(comptime self: Expr) Expr {
    return self.wrap(self.b.logicalNot(self.value));
}

pub fn logicalAnd(comptime self: Expr, comptime rhs: Expr) Expr {
    return self.wrap(self.b.logicalAnd(self.value, rhs.value));
}

pub fn logicalOr(comptime self: Expr, comptime rhs: Expr) Expr {
    return self.wrap(self.b.logicalOr(self.value, rhs.value));
}

/// Select between two values using this boolean expression as the condition.
pub fn where(comptime self: Expr, comptime when_true: Expr, comptime when_false: Expr) Expr {
    return self.wrap(self.b.where(self.value, when_true.value, when_false.value));
}

pub fn copy(comptime self: Expr) Expr {
    return self.wrap(self.b.copy(self.value));
}

pub fn contiguous(comptime self: Expr) Expr {
    return self.wrap(self.b.contiguous(self.value));
}

pub fn pad(comptime self: Expr, comptime fill: Expr, comptime options: definition.PadOptions) Expr {
    return self.wrap(self.b.pad(self.value, fill.value, options));
}

pub fn shift(
    comptime self: Expr,
    comptime offsets: []const isize,
    comptime boundary: DefinitionBuilder.ShiftBoundary,
) Expr {
    return self.wrap(self.b.shift(self.value, offsets, boundary));
}

pub fn sliceLoop(comptime self: Expr, comptime options: DefinitionBuilder.SliceLoopOptions) Expr {
    return self.wrap(self.b.sliceLoop(self.value, options));
}

pub fn matmul(comptime self: Expr, comptime rhs: Expr) Expr {
    return self.wrap(self.b.matmul(self.value, rhs.value));
}

pub fn sum(comptime self: Expr, comptime options: definition.ReductionOptions) Expr {
    return self.wrap(self.b.sum(self.value, options));
}

pub fn mean(comptime self: Expr, comptime options: definition.ReductionOptions) Expr {
    return self.wrap(self.b.mean(self.value, options));
}

pub fn min(comptime self: Expr, comptime options: definition.ReductionOptions) Expr {
    return self.wrap(self.b.min(self.value, options));
}

pub fn max(comptime self: Expr, comptime options: definition.ReductionOptions) Expr {
    return self.wrap(self.b.max(self.value, options));
}

/// Concatenate this expression followed by `rhs` along `axis`.
pub fn concat(comptime self: Expr, comptime rhs: Expr, comptime axis: i8) Expr {
    return self.wrap(self.b.concat(&.{ self.value, rhs.value }, axis));
}

pub fn softmax(comptime self: Expr, comptime axis: i8) Expr {
    return self.wrap(self.b.softmax(self.value, axis));
}

pub fn transpose(comptime self: Expr, comptime axis_a: i8, comptime axis_b: i8) Expr {
    return self.wrap(self.b.transpose(self.value, axis_a, axis_b));
}

pub fn reshape(comptime self: Expr, comptime extents: []const usize) Expr {
    return self.wrap(self.b.reshape(self.value, extents));
}

pub fn broadcastTo(comptime self: Expr, comptime extents: []const usize) Expr {
    return self.wrap(self.b.broadcastTo(self.value, extents));
}

pub fn flatten(comptime self: Expr, comptime options: definition.FlattenOptions) Expr {
    return self.wrap(self.b.flatten(self.value, options));
}

pub fn squeeze(comptime self: Expr, comptime axis: i8) Expr {
    return self.wrap(self.b.squeeze(self.value, axis));
}

pub fn unsqueeze(comptime self: Expr, comptime axis: i8) Expr {
    return self.wrap(self.b.unsqueeze(self.value, axis));
}

pub fn permute(comptime self: Expr, comptime axes: []const i8) Expr {
    return self.wrap(self.b.permute(self.value, axes));
}

pub fn slice(comptime self: Expr, comptime options: definition.SliceOptions) Expr {
    return self.wrap(self.b.slice(self.value, options));
}

pub fn windows(comptime self: Expr, comptime options: definition.WindowOptions) Expr {
    return self.wrap(self.b.windows(self.value, options));
}

fn wrap(comptime self: Expr, comptime value: Value) Expr {
    return self.b.expr(value);
}

fn assertExprFn(comptime operation: anytype) void {
    const info = switch (@typeInfo(@TypeOf(operation))) {
        .@"fn" => |info| info,
        else => @compileError("expression operation must be a function"),
    };
    if (info.param_types.len == 0 or info.param_types[0] != Expr) {
        @compileError("expression operation must accept Expr as its first parameter");
    }
    if (info.return_type != Expr) {
        @compileError("expression operation must return Expr");
    }
}
