const std = @import("std");
const base = @import("executable_abi.zig");

pub const Api = base.Api;

/// Allocate aligned bytes inside the module's exported linear memory.
pub fn alloc(byte_count: usize, alignment: usize) callconv(.c) ?*anyopaque {
    if (byte_count == 0 or alignment == 0 or !std.math.isPowerOfTwo(alignment)) return null;
    const bytes = std.heap.wasm_allocator.rawAlloc(
        byte_count,
        .fromByteUnits(alignment),
        @returnAddress(),
    ) orelse return null;
    return @ptrCast(bytes);
}

/// Release a block using the size and alignment passed to alloc().
pub fn free(pointer: ?*anyopaque, byte_count: usize, alignment: usize) callconv(.c) void {
    const address = pointer orelse return;
    if (byte_count == 0 or alignment == 0 or !std.math.isPowerOfTwo(alignment)) return;
    const bytes: [*]u8 = @ptrCast(address);
    std.heap.wasm_allocator.rawFree(
        bytes[0..byte_count],
        .fromByteUnits(alignment),
        @returnAddress(),
    );
}

comptime {
    _ = base.Api;
    @export(&alloc, .{ .name = "zgc_wasm_alloc" });
    @export(&free, .{ .name = "zgc_wasm_free" });
}
