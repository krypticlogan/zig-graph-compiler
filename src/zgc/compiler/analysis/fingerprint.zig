const std = @import("std");

/// Hash the active, data-only executable structure. Pointer addresses, array
/// padding, and unused graph capacity do not participate in the result.
pub fn executable(comptime program: anytype) u64 {
    @setEvalBranchQuota(500_000);
    var hash = Hash{};
    hash.value(@as(u8, 1)); // fingerprint format version
    hash.value(program.node_ct);
    hash.value(program.tensor_ct);
    hash.value(program.source_ct);
    hash.value(program.input_ref_ct);
    hash.value(program.output_ref_ct);
    hash.value(program.output_ct);
    for (program.nodes[0..program.node_ct]) |node| hash.value(node.?);
    for (program.tensors[0..program.tensor_ct]) |tensor| hash.value(tensor.?);
    hash.value(program.sources[0..program.source_ct]);
    hash.value(program.input_refs[0..program.input_ref_ct]);
    hash.value(program.output_refs[0..program.output_ref_ct]);
    hash.value(program.outputs[0..program.output_ct]);
    hash.value(program.materialized[0..program.tensor_ct]);
    return hash.state;
}

const Hash = struct {
    state: u64 = 0xcbf29ce484222325,

    fn byte(self: *Hash, octet: u8) void {
        self.state = (self.state ^ octet) *% 0x100000001b3;
    }

    fn integer(self: *Hash, number: anytype) void {
        const T = @TypeOf(number);
        const U = @Int(.unsigned, @bitSizeOf(T));
        const bits: U = @bitCast(number);
        inline for (0..@sizeOf(T)) |index| {
            self.byte(@truncate(bits >> @intCast(index * 8)));
        }
    }

    fn value(self: *Hash, item: anytype) void {
        const T = @TypeOf(item);
        switch (@typeInfo(T)) {
            .bool => self.byte(@intFromBool(item)),
            .int => self.integer(item),
            .float => {
                const U = @Int(.unsigned, @bitSizeOf(T));
                self.integer(@as(U, @bitCast(item)));
            },
            .@"enum" => self.integer(@backingInt(item)),
            .void => {},
            .optional => {
                if (item) |present| {
                    self.byte(1);
                    self.value(present);
                } else self.byte(0);
            },
            .array => {
                for (item) |element| self.value(element);
            },
            .pointer => |pointer| switch (pointer.size) {
                .slice => {
                    self.value(item.len);
                    for (item) |element| self.value(element);
                },
                .one => self.value(item.*),
                else => @compileError("unsupported pointer in executable fingerprint"),
            },
            .@"struct" => |structure| {
                inline for (structure.field_names) |name| self.value(@field(item, name));
            },
            .@"union" => |union_info| {
                if (union_info.tag_type == null) @compileError("untagged union in executable fingerprint");
                self.value(std.meta.activeTag(item));
                switch (item) {
                    inline else => |payload| self.value(payload),
                }
            },
            else => @compileError("unsupported value in executable fingerprint: " ++ @typeName(T)),
        }
    }
};
