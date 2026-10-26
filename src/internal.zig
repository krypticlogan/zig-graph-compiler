//! Private entry point for repository tests and compiler benchmarks.
//! Keep the public API and internals in one module so Zig types share identity.
pub const api = @import("root.zig");
pub const analysis = @import("zgc/compiler/analysis/root.zig");
pub const planning = @import("zgc/compiler/planning/root.zig");
pub const search = @import("zgc/compiler/search.zig");
pub const finalization = @import("zgc/compiler/finalization.zig");
pub const traversal = @import("zgc/compiler/traversal.zig");
pub const scheduling = @import("zgc/compiler/scheduling.zig");
pub const ready_queue = @import("zgc/compiler/ready_queue.zig");
pub const Validation = @import("zgc/validation.zig");
pub const Op = @import("zgc/operations/semantic.zig").Op;
pub const Inspect = @import("zgc_inspect_impl");

pub const node_set = @import("zgc/compiler/node_set.zig");

pub const program_builder = @import("zgc/compiler/program_builder.zig");
pub const Program = @import("zgc/execution/program.zig").Program;
