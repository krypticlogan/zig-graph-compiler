//! Indexed max heap. Larger scores win, with lower node IDs breaking ties.
//! Scores can be raised for waiting nodes and for nodes already in the heap.
pub fn ReadyQueue(comptime capacity: usize, comptime Score: type) type {
    return struct {
        const Self = @This();
        nodes: [capacity]usize = undefined,
        positions: [capacity]?usize = @splat(null),
        scores: [capacity]Score,
        count: usize = 0,

        fn better(queue: *const Self, lhs: usize, rhs: usize) bool {
            return queue.scores[lhs] > queue.scores[rhs] or
                (queue.scores[lhs] == queue.scores[rhs] and lhs < rhs);
        }

        fn swap(queue: *Self, lhs: usize, rhs: usize) void {
            const node = queue.nodes[lhs];
            queue.nodes[lhs] = queue.nodes[rhs];
            queue.nodes[rhs] = node;
            queue.positions[queue.nodes[lhs]] = lhs;
            queue.positions[queue.nodes[rhs]] = rhs;
        }

        fn rise(queue: *Self, start: usize) void {
            var index = start;
            while (index > 0) {
                const parent = (index - 1) / 2;
                if (!queue.better(queue.nodes[index], queue.nodes[parent])) break;
                queue.swap(index, parent);
                index = parent;
            }
        }

        pub fn push(queue: *Self, node: usize) void {
            queue.nodes[queue.count] = node;
            queue.positions[node] = queue.count;
            queue.count += 1;
            queue.rise(queue.count - 1);
        }

        pub fn raise(queue: *Self, node: usize, amount: Score) void {
            queue.scores[node] += amount;
            if (queue.positions[node]) |position| queue.rise(position);
        }

        pub fn pop(queue: *Self) usize {
            const result = queue.nodes[0];
            queue.positions[result] = null;
            queue.count -= 1;
            if (queue.count == 0) return result;
            queue.nodes[0] = queue.nodes[queue.count];
            queue.positions[queue.nodes[0]] = 0;
            var index: usize = 0;
            while (index * 2 + 1 < queue.count) {
                var child = index * 2 + 1;
                if (child + 1 < queue.count and queue.better(queue.nodes[child + 1], queue.nodes[child])) child += 1;
                if (!queue.better(queue.nodes[child], queue.nodes[index])) break;
                queue.swap(index, child);
                index = child;
            }
            return result;
        }
    };
}
