const std = @import("std");

pub const Dtype = enum {
    f32,
    f16,
    i8,
    bool,

    pub const Kind = enum {
        float,
        signed_integer,
        boolean,
    };

    pub fn Scalar(comptime self: Dtype) type {
        return switch (self) {
            .f32 => f32,
            .f16 => f16,
            .i8 => i8,
            .bool => bool,
        };
    }

    pub fn fromScalar(comptime T: type) Dtype {
        return if (T == f32)
            .f32
        else if (T == f16)
            .f16
        else if (T == i8)
            .i8
        else if (T == bool)
            .bool
        else
            @compileError("unsupported tensor scalar type: " ++ @typeName(T));
    }

    pub fn kind(dtype: Dtype) Kind {
        return switch (dtype) {
            .f32, .f16 => .float,
            .i8 => .signed_integer,
            .bool => .boolean,
        };
    }

    pub fn Vector(comptime dtype: Dtype, comptime len: usize) type {
        return @Vector(len, dtype.Scalar());
    }

    pub fn zero(comptime dtype: Dtype) dtype.Scalar() {
        return switch (dtype) {
            .bool => false,
            else => 0,
        };
    }

    pub fn vectorZero(comptime dtype: Dtype, comptime len: usize) dtype.Vector(len) {
        return @splat(dtype.zero());
    }

    pub fn byteSize(comptime dtype: Dtype) usize {
        return switch (dtype) {
            .f32 => @sizeOf(f32),
            .f16 => @sizeOf(f16),
            .i8 => @sizeOf(i8),
            .bool => @sizeOf(bool),
        };
    }

    pub fn alignment(comptime dtype: Dtype) usize {
        return switch (dtype) {
            .f32 => @alignOf(f32),
            .f16 => @alignOf(f16),
            .i8 => @alignOf(i8),
            .bool => @alignOf(bool),
        };
    }
};

/// A compile-time scalar literal carried by the graph without requiring a
/// named source or executable node.
pub const ScalarValue = struct {
    data_type: Dtype,
    bits: u32,

    pub fn init(comptime dtype_value: Dtype, comptime value: anytype) ScalarValue {
        return switch (dtype_value) {
            .f32 => .{ .data_type = .f32, .bits = @bitCast(@as(f32, value)) },
            .f16 => .{ .data_type = .f16, .bits = @as(u16, @bitCast(@as(f16, value))) },
            .i8 => .{ .data_type = .i8, .bits = @as(u8, @bitCast(@as(i8, value))) },
            .bool => .{ .data_type = .bool, .bits = @intFromBool(@as(bool, value)) },
        };
    }

    pub fn dtype(value: ScalarValue) Dtype {
        return value.data_type;
    }

    pub fn get(value: ScalarValue, comptime dtype_value: Dtype) dtype_value.Scalar() {
        if (dtype_value == .bool) return value.bits != 0;
        const T = dtype_value.Scalar();
        const Bits = std.meta.Int(.unsigned, @bitSizeOf(T));
        return @bitCast(@as(Bits, @truncate(value.bits)));
    }
};
