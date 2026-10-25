const std = @import("std");
const zgc = @import("zgc");
const Dtype = zgc.memory.Dtype;

const NumericSources = enum(usize) { lhs, rhs };

fn numericExecutable(comptime dtype: Dtype) type {
    var builder = zgc.DefinitionBuilder.init();
    const sources = builder.sources(NumericSources);
    const lhs = sources.input(.lhs, dtype, &.{3});
    const rhs = sources.input(.rhs, dtype, &.{3});
    const condition = builder.greaterThan(lhs, rhs);
    builder.output(builder.where(condition, builder.add(lhs, rhs), lhs));
    builder.output(condition);
    builder.output(builder.sum(lhs, .{ .axes = &.{0} }));
    return builder.finish().executable();
}

test "numeric dtypes compile and execute typed graphs" {
    inline for ([_]Dtype{ .f32, .f16, .i8, .i32, .u8 }) |dtype| {
        const T = dtype.Scalar();
        const Executable = numericExecutable(dtype);
        var runtime = Executable.Runtime.init();
        try runtime.copyInput(.lhs, &[_]T{ 1, 4, 2 });
        try runtime.copyInput(.rhs, &[_]T{ 2, 1, 3 });
        runtime.run();

        const result = runtime.outputView(0);
        const condition = runtime.outputView(1);
        try std.testing.expectEqual(T, @TypeOf(result.storage[0]));
        try std.testing.expectEqual(bool, @TypeOf(condition.storage[0]));
        try std.testing.expectEqualSlices(T, &.{ 1, 5, 2 }, result.contiguousSlice().?);
        try std.testing.expectEqualSlices(bool, &.{ false, true, false }, condition.contiguousSlice().?);
        try std.testing.expectEqual(@as(T, 7), runtime.outputView(2).storage[0]);
        try std.testing.expectEqual(dtype, Dtype.fromScalar(T));
        try std.testing.expectEqual(@as(T, 1), zgc.memory.ScalarValue.init(dtype, 1).get(dtype));
    }
}

const BooleanSources = enum(usize) { lhs, rhs };

const BooleanExecutable = model: {
    var builder = zgc.DefinitionBuilder.init();
    const sources = builder.sources(BooleanSources);
    const lhs = sources.input(.lhs, .bool, &.{3});
    const rhs = sources.input(.rhs, .bool, &.{3});
    builder.output(builder.logicalAnd(lhs, rhs));
    builder.output(builder.logicalOr(lhs, rhs));
    builder.output(builder.where(lhs, rhs, builder.logicalNot(rhs)));
    break :model builder.finish().executable();
};

test "boolean graph executes logical and selection operations" {
    var runtime = BooleanExecutable.Runtime.init();
    try runtime.copyInput(.lhs, &[_]bool{ true, false, true });
    try runtime.copyInput(.rhs, &[_]bool{ true, true, false });
    runtime.run();

    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, runtime.outputView(0).contiguousSlice().?);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true }, runtime.outputView(1).contiguousSlice().?);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, runtime.outputView(2).contiguousSlice().?);
}

test "cross-dtype validation distinguishes matching values and boolean masks" {
    const Value = struct { dtype: Dtype };
    const Validation = zgc.compiler.Validation;
    try std.testing.expect(Validation.dtypesMatch(&[_]Value{ .{ .dtype = .u8 }, .{ .dtype = .u8 } }));
    try std.testing.expect(!Validation.dtypesMatch(&[_]Value{ .{ .dtype = .u8 }, .{ .dtype = .i8 } }));
    try std.testing.expect(!Validation.dtypesMatch(&[_]Value{ .{ .dtype = .f32 }, .{ .dtype = .f16 } }));
    try std.testing.expect(Validation.dtypeIsNumeric(Value{ .dtype = .u8 }));
    try std.testing.expect(!Validation.dtypeIsNumeric(Value{ .dtype = .bool }));
    try std.testing.expect(Validation.dtypeKindIs(Value{ .dtype = .u8 }, .unsigned_integer));
    try std.testing.expectEqual(Dtype.u8, zgc.Op.Compute.inferDtype(.where, &[_]Value{
        .{ .dtype = .bool }, .{ .dtype = .u8 }, .{ .dtype = .u8 },
    }));
    try std.testing.expectEqual(Dtype.bool, zgc.Op.Compute.inferDtype(.equal, &[_]Value{
        .{ .dtype = .i32 }, .{ .dtype = .i32 },
    }));
}
