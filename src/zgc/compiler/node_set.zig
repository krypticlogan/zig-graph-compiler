//! Fixed-capacity membership mask. Iteration visits set IDs only, in ID order.
pub fn Set(comptime capacity: usize) type {
    return struct {
        const Self = @This();
        words: [(capacity + 63) / 64]u64 = @splat(0),

        pub fn init(comptime ids: []const usize) Self {
            var result: Self = .{};
            for (ids) |id| result.set(id);
            return result;
        }

        pub fn isSet(self: *const Self, id: usize) bool {
            return self.words[id / 64] & (@as(u64, 1) << @as(u6, @intCast(id % 64))) != 0;
        }

        pub fn set(self: *Self, id: usize) void {
            self.words[id / 64] |= @as(u64, 1) << @as(u6, @intCast(id % 64));
        }

        pub fn intersects(self: *const Self, other: *const Self) bool {
            for (self.words, other.words) |lhs, rhs| if (lhs & rhs != 0) return true;
            return false;
        }

        pub fn containsAll(self: *const Self, other: *const Self) bool {
            for (self.words, other.words) |lhs, rhs| if (rhs & ~lhs != 0) return false;
            return true;
        }

        pub fn include(self: *Self, other: *const Self) void {
            for (&self.words, other.words) |*lhs, rhs| lhs.* |= rhs;
        }

        pub fn eql(self: *const Self, other: *const Self) bool {
            for (self.words, other.words) |lhs, rhs| if (lhs != rhs) return false;
            return true;
        }

        pub fn iterator(self: *const Self) Iterator {
            return .{ .source = self };
        }

        pub const Iterator = struct {
            source: *const Self,
            word: usize = 0,
            pending: u64 = 0,
            pub fn next(it: *Iterator) ?usize {
                if (comptime capacity == 0) return null;
                while (it.pending == 0) {
                    if (it.word == it.source.words.len) return null;
                    it.pending = it.source.words[it.word];
                    it.word += 1;
                }
                const bit = @ctz(it.pending);
                it.pending &= it.pending - 1;
                return (it.word - 1) * 64 + bit;
            }
        };
    };
}
