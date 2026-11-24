pub const inspectable = true;

const base = @import("wasm_executable_abi.zig");

comptime {
    _ = base.Api;
}
