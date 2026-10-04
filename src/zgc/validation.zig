const std = @import("std");
const Dtype = @import("storage/dtype.zig").Dtype;

/// Return the rank carried by either a counting-pass value (`rank`) or a
/// materialized graph value (`shape.rank`).
pub fn rankOf(value: anytype) usize {
    const Value = @TypeOf(value);
    if (@hasField(Value, "rank")) return value.rank;
    if (@hasField(Value, "shape")) return value.shape.rank;
    @compileError("graph validation requires values with rank metadata");
}

pub fn inputCountIs(inputs: anytype, comptime expected: usize) bool {
    return inputs.len == expected;
}

pub fn ranksAre(inputs: anytype, comptime expected: []const usize) bool {
    if (inputs.len != expected.len) return false;
    inline for (expected, 0..) |rank, index| {
        if (rankOf(inputs[index]) != rank) return false;
    }
    return true;
}

pub fn ranksMatch(inputs: anytype) bool {
    if (inputs.len < 2) return true;
    const first_rank = rankOf(inputs[0]);
    for (inputs[1..]) |input| {
        if (rankOf(input) != first_rank) return false;
    }
    return true;
}

pub fn dtypesMatch(inputs: anytype) bool {
    if (inputs.len < 2) return true;
    const first_dtype = inputs[0].dtype;
    for (inputs[1..]) |input| {
        if (input.dtype != first_dtype) return false;
    }
    return true;
}

pub fn dtypeIs(value: anytype, comptime expected: Dtype) bool {
    return value.dtype == expected;
}

pub fn dtypeKindIs(value: anytype, comptime expected: Dtype.Kind) bool {
    return value.dtype.kind() == expected;
}

pub fn dtypeIsNumeric(value: anytype) bool {
    return value.dtype.kind() != .boolean;
}

pub fn shapesMatch(lhs: anytype, rhs: anytype) bool {
    return std.mem.eql(usize, lhs.shape.slice(), rhs.shape.slice());
}

pub fn extentsBroadcast(lhs_extent: usize, rhs_extent: usize) bool {
    return lhs_extent == rhs_extent or lhs_extent == 1 or rhs_extent == 1;
}

pub fn shapesBroadcast(lhs: anytype, rhs: anytype) bool {
    const lhs_shape = lhs.shape.slice();
    const rhs_shape = rhs.shape.slice();
    const aligned_rank = @min(lhs_shape.len, rhs_shape.len);

    for (0..aligned_rank) |axis_from_end| {
        const lhs_extent = lhs_shape[lhs_shape.len - 1 - axis_from_end];
        const rhs_extent = rhs_shape[rhs_shape.len - 1 - axis_from_end];
        if (!extentsBroadcast(lhs_extent, rhs_extent)) return false;
    }
    return true;
}

pub fn reductionAxesAreValid(value: anytype, comptime axes: u64) bool {
    const rank = rankOf(value);
    if (axes == 0 or rank > 64) return false;
    if (rank == 64) return true;
    return axes < (@as(u64, 1) << @intCast(rank));
}

pub fn extentsMatch(
    lhs: anytype,
    comptime lhs_axis: usize,
    rhs: anytype,
    comptime rhs_axis: usize,
) bool {
    if (lhs_axis >= rankOf(lhs) or rhs_axis >= rankOf(rhs)) return false;
    return lhs.shape.at(lhs_axis) == rhs.shape.at(rhs_axis);
}

pub fn axisIsValid(value: anytype, comptime axis: i8) bool {
    if (axis < 0) return false;
    return @as(usize, @intCast(axis)) < rankOf(value);
}

pub fn requireInputCount(
    comptime operation: []const u8,
    inputs: anytype,
    comptime expected: usize,
) void {
    if (!inputCountIs(inputs, expected)) {
        @compileError(std.fmt.comptimePrint(
            "{s} requires exactly {d} input(s)",
            .{ operation, expected },
        ));
    }
}

pub fn requireRanks(
    comptime operation: []const u8,
    inputs: anytype,
    comptime expected: []const usize,
) void {
    if (!ranksAre(inputs, expected)) {
        @compileError(std.fmt.comptimePrint(operation ++ " input ranks are invalid\nExpected: {any}", .{expected}));
    }
}

pub fn requireMatchingRanks(comptime operation: []const u8, inputs: anytype) void {
    if (!ranksMatch(inputs)) {
        @compileError(operation ++ " input ranks must match");
    }
}

pub fn requireMatchingDtypes(comptime operation: []const u8, inputs: anytype) void {
    if (!dtypesMatch(inputs)) {
        @compileError(operation ++ " input dtypes must match");
    }
}

pub fn requireDtype(
    comptime operation: []const u8,
    value: anytype,
    comptime expected: Dtype,
) void {
    if (!dtypeIs(value, expected)) {
        @compileError(operation ++ " does not support the provided dtype");
    }
}

pub fn requireDtypeKind(
    comptime operation: []const u8,
    value: anytype,
    comptime expected: Dtype.Kind,
) void {
    if (!dtypeKindIs(value, expected)) {
        @compileError(operation ++ " does not support the provided dtype kind");
    }
}

pub fn requireNumericDtype(comptime operation: []const u8, value: anytype) void {
    if (!dtypeIsNumeric(value)) {
        @compileError(operation ++ " requires a numeric tensor");
    }
}

pub fn requireMatchingShapes(
    comptime operation: []const u8,
    lhs: anytype,
    rhs: anytype,
) void {
    if (!shapesMatch(lhs, rhs)) {
        @compileError(operation ++ " input shapes must match");
    }
}

pub fn requireMatchingExtents(
    comptime operation: []const u8,
    lhs: anytype,
    comptime lhs_axis: usize,
    rhs: anytype,
    comptime rhs_axis: usize,
) void {
    if (!extentsMatch(lhs, lhs_axis, rhs, rhs_axis)) {
        @compileError(operation ++ " contracted dimensions must match");
    }
}

pub fn requireAxis(
    comptime operation: []const u8,
    value: anytype,
    comptime axis: i8,
) void {
    if (!axisIsValid(value, axis)) {
        @compileError(operation ++ " axis is outside the input rank");
    }
}

pub fn requireReductionAxes(
    comptime operation: []const u8,
    value: anytype,
    comptime axes: u64,
) void {
    if (!reductionAxesAreValid(value, axes)) {
        @compileError(operation ++ " requires one or more unique axes within the input rank");
    }
}
