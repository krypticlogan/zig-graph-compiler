const std = @import("std");
const Graph = @import("../core/graph.zig");
const ScalarValue = @import("dtype.zig").ScalarValue;

/// Compile-time storage selection for a graph source.
pub const Binding = union(enum) {
    /// Reserve storage in the model's inline memory.
    owned,
    /// Borrow storage supplied to a model instance at runtime.
    bound,
    /// Read immutable bytes from the executable's read-only data. Logical
    /// bytes are packed into the layout selected during lowering; physical
    /// bytes already satisfy that layout contract.
    embedded: Embedded,
    /// Compiler-owned immutable scalar storage.
    literal: ScalarValue,
};

pub const Embedded = struct {
    bytes: []const u8,
    order: Order,

    pub const Order = enum {
        logical,
        physical,
    };
};

pub const owned: Binding = .owned;
pub const bound: Binding = .bound;

/// Embed values in logical row-major order. If lowering selects a different
/// physical layout, the compiler packs these bytes into that layout.
pub fn embed(comptime bytes: []const u8) Binding {
    return .{ .embedded = .{ .bytes = bytes, .order = .logical } };
}

/// Embed values which are already stored in the physical order reported by
/// the compiled model's source layout.
pub fn embedPacked(comptime bytes: []const u8) Binding {
    return .{ .embedded = .{ .bytes = bytes, .order = .physical } };
}

/// Normalize a named source configuration into the graph's enum-indexed source
/// table. Unspecified sources remain model-owned.
pub fn Plan(
    comptime SourceKey: type,
    comptime capacity: Graph.Capacity,
    comptime graph: anytype,
    comptime configuration: anytype,
) type {
    var bindings: [capacity.max_sources]Binding = @splat(.owned);
    var configured: [capacity.max_sources]bool = @splat(false);

    for (configuration) |override| {
        const key: SourceKey = override.source;
        const source_index: usize = @intCast(@intFromEnum(key));
        if (source_index >= graph.sources.len or graph.sources[source_index] == null) {
            @compileError("source configuration refers to a source that is not used by the graph: " ++ @tagName(key));
        }
        if (configured[source_index]) {
            @compileError("source configuration contains a duplicate source: " ++ @tagName(key));
        }
        configured[source_index] = true;

        const binding: Binding = override.binding;
        const source = graph.sources[source_index].?;
        const tensor = graph.tensors[source.tensor].?;
        const expected_bytes = tensor.shape.elementCount() * tensor.dtype.byteSize();

        switch (binding) {
            .owned => {},
            .bound => {
                if (source.kind != .input) {
                    @compileError("only input sources may use runtime-bound storage");
                }
            },
            .embedded => |embedded| {
                if (source.kind != .parameter and source.kind != .constant) {
                    @compileError("only parameter and constant sources may be embedded");
                }
                if (embedded.bytes.len != expected_bytes) {
                    @compileError(std.fmt.comptimePrint(
                        "embedded source '{s}' requires {d} bytes, received {d}",
                        .{ @tagName(key), expected_bytes, embedded.bytes.len },
                    ));
                }
            },
            .literal => @compileError("literal bindings are created by scalar definitions"),
        }
        bindings[source_index] = binding;
    }

    const normalized = bindings;
    return struct {
        pub const source_bindings = normalized;

        pub fn bindingForTensor(tensor_info: anytype) Binding {
            return switch (tensor_info.origin) {
                .source => |source_index| if (comptime source_bindings.len == 0)
                    unreachable
                else
                    source_bindings[source_index],
                .node => .owned,
                .literal => |value| .{ .literal = value },
            };
        }

        pub fn isOwned(comptime tensor_info: anytype) bool {
            return switch (bindingForTensor(tensor_info)) {
                .owned => true,
                .bound, .embedded, .literal => false,
            };
        }
    };
}
