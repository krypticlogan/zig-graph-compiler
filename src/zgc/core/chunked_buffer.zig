//! Append-only storage for comptime graph construction. Completed chunks are
//! immutable so copying a builder or finishing it does not share mutable state.
const std = @import("std");

pub const max_size: usize = std.math.maxInt(u32);
pub const chunk_size = 64;

pub fn Buffer(comptime T: type) type {
    return struct {
        const Self = @This();
        const Chunk = struct {
            values: [chunk_size]T,
            previous: ?*const @This(),
        };

        len: usize = 0,
        tail: [chunk_size]T = undefined,
        completed: ?*const Chunk = null,

        pub fn append(comptime self: *Self, comptime value: T) void {
            if (self.len >= max_size) @compileError("comptime buffer exceeds max_size");
            self.tail[self.len % chunk_size] = value;
            self.len += 1;
            if (self.len % chunk_size == 0) {
                const chunk: Chunk = .{ .values = self.tail, .previous = self.completed };
                self.completed = &chunk;
            }
        }

        /// Copy only populated entries, in insertion order, into exact storage.
        pub fn toArray(comptime self: *const Self) [self.len]T {
            var result: [self.len]T = undefined;
            const tail_len = self.len % chunk_size;
            var end = self.len - tail_len;
            @memcpy(result[end..], self.tail[0..tail_len]);
            var chunk = self.completed;
            while (chunk) |current| {
                const begin = end - chunk_size;
                @memcpy(result[begin..end], &current.values);
                end = begin;
                chunk = current.previous;
            }
            return result;
        }

        pub const ReverseIterator = struct {
            values: []const T,
            index: usize,
            chunk: ?*const Chunk,

            pub fn next(iterator: *@This()) ?T {
                if (iterator.index == 0) {
                    const chunk = iterator.chunk orelse return null;
                    iterator.values = &chunk.values;
                    iterator.index = chunk_size;
                    iterator.chunk = chunk.previous;
                }
                iterator.index -= 1;
                return iterator.values[iterator.index];
            }
        };

        pub fn reverseIterator(comptime self: *const Self) ReverseIterator {
            return .{
                .values = &self.tail,
                .index = self.len % chunk_size,
                .chunk = self.completed,
            };
        }
    };
}
