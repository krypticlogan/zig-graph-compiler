const model_abi = @import("model_abi.zig");

/// The executable is an artifact container. Applications interact with its
/// exported model ABI; the entry point itself performs no inference.
pub fn main() void {
    model_abi.retain();
}
