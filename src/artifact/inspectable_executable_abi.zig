pub const inspectable = true;

const base = @import("executable_abi.zig");

comptime {
    _ = base.Api;
}
