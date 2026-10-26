//! Shared append-only comptime storage used by graph and program builders.
const shared = @import("../core/chunked_buffer.zig");
pub const max_size = shared.max_size;
pub const chunk_size = shared.chunk_size;
pub const Buffer = shared.Buffer;
