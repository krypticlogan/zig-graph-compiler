pub const inspectable = true;

const executable_abi = @import("executable_abi.zig");

/// The executable exports an inspectable ABI; its entry point performs no inference.
pub fn main() void {
    executable_abi.retain();
}
