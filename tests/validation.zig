const std = @import("std");
const zgc = @import("zgc");

const CountedValue = struct {
    dtype: zgc.memory.Dtype,
    rank: usize,
};

fn ShapedValue(comptime max_rank: usize) type {
    return struct {
        dtype: zgc.memory.Dtype,
        shape: zgc.core.Tensor.Shape(max_rank),
    };
}

test "validation reads rank metadata from both graph passes" {
    const counted = CountedValue{ .dtype = .f32, .rank = 3 };
    const shaped = ShapedValue(4){
        .dtype = .f32,
        .shape = .init(&.{ 2, 3, 4 }),
    };

    try std.testing.expectEqual(@as(usize, 3), zgc.compiler.Validation.rankOf(counted));
    try std.testing.expectEqual(@as(usize, 3), zgc.compiler.Validation.rankOf(shaped));
}

test "validation checks arity ranks and dtypes" {
    const inputs = [_]CountedValue{
        .{ .dtype = .f32, .rank = 2 },
        .{ .dtype = .f32, .rank = 2 },
    };
    const mismatched = [_]CountedValue{
        .{ .dtype = .f32, .rank = 2 },
        .{ .dtype = .f16, .rank = 1 },
    };
    const integer = CountedValue{ .dtype = .i8, .rank = 1 };
    const boolean = CountedValue{ .dtype = .bool, .rank = 1 };

    try std.testing.expect(zgc.compiler.Validation.inputCountIs(&inputs, 2));
    try std.testing.expect(zgc.compiler.Validation.ranksAre(&inputs, &.{ 2, 2 }));
    try std.testing.expect(zgc.compiler.Validation.ranksMatch(&inputs));
    try std.testing.expect(zgc.compiler.Validation.dtypesMatch(&inputs));
    try std.testing.expect(!zgc.compiler.Validation.ranksMatch(&mismatched));
    try std.testing.expect(!zgc.compiler.Validation.dtypesMatch(&mismatched));
    try std.testing.expect(zgc.compiler.Validation.dtypeKindIs(inputs[0], .float));
    try std.testing.expect(!zgc.compiler.Validation.dtypeKindIs(integer, .float));
    try std.testing.expect(zgc.compiler.Validation.dtypeKindIs(boolean, .boolean));
    try std.testing.expect(zgc.compiler.Validation.dtypeIsNumeric(inputs[0]));
    try std.testing.expect(zgc.compiler.Validation.dtypeIsNumeric(integer));
    try std.testing.expect(!zgc.compiler.Validation.dtypeIsNumeric(boolean));
}

test "validation checks shapes extents and axes" {
    const Value = ShapedValue(2);
    const lhs = Value{ .dtype = .f32, .shape = .init(&.{ 3, 4 }) };
    const same = Value{ .dtype = .f32, .shape = .init(&.{ 3, 4 }) };
    const rhs = Value{ .dtype = .f32, .shape = .init(&.{ 4, 7 }) };

    try std.testing.expect(zgc.compiler.Validation.shapesMatch(lhs, same));
    try std.testing.expect(!zgc.compiler.Validation.shapesMatch(lhs, rhs));
    try std.testing.expect(zgc.compiler.Validation.extentsMatch(lhs, 1, rhs, 0));
    try std.testing.expect(zgc.compiler.Validation.axisIsValid(lhs, 0));
    try std.testing.expect(zgc.compiler.Validation.axisIsValid(lhs, 1));
    try std.testing.expect(!zgc.compiler.Validation.axisIsValid(lhs, -1));
    try std.testing.expect(!zgc.compiler.Validation.axisIsValid(lhs, 2));
}

test "validation checks trailing-axis broadcast compatibility" {
    const Value = ShapedValue(3);
    const matrix = Value{ .dtype = .f32, .shape = .init(&.{ 2, 3 }) };
    const vector = Value{ .dtype = .f32, .shape = .init(&.{3}) };
    const outer_lhs = Value{ .dtype = .f32, .shape = .init(&.{ 2, 1 }) };
    const outer_rhs = Value{ .dtype = .f32, .shape = .init(&.{ 1, 4 }) };
    const invalid = Value{ .dtype = .f32, .shape = .init(&.{2}) };

    try std.testing.expect(zgc.compiler.Validation.shapesBroadcast(matrix, vector));
    try std.testing.expect(zgc.compiler.Validation.shapesBroadcast(outer_lhs, outer_rhs));
    try std.testing.expect(!zgc.compiler.Validation.shapesBroadcast(matrix, invalid));
}

test "validation checks normalized reduction axis sets" {
    const Value = ShapedValue(3);
    const tensor = Value{ .dtype = .f32, .shape = .init(&.{ 2, 3, 4 }) };

    try std.testing.expect(zgc.compiler.Validation.reductionAxesAreValid(tensor, 0b001));
    try std.testing.expect(zgc.compiler.Validation.reductionAxesAreValid(tensor, 0b101));
    try std.testing.expect(!zgc.compiler.Validation.reductionAxesAreValid(tensor, 0));
    try std.testing.expect(!zgc.compiler.Validation.reductionAxesAreValid(tensor, 0b1000));
}
