const executable_abi = @import("executable_abi.zig");

/// The executable is an artifact container. Applications interact with its
/// exported executable ABI; the entry point itself performs no inference.
pub fn main() void {
    executable_abi.retain();
}
