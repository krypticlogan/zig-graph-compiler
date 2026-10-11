const builtin = @import("builtin");
pub const Expression = @import("../optimization/fusion/expression.zig").Program;

/// Count simultaneously live expression results in the unrolled kernel body.
/// Values with no instruction consumer are treated as live through the store.
pub fn peakLiveValues(comptime expression: Expression) usize {
    const count = expression.instructions.len;
    var last_use: [count]usize = undefined;
    var has_consumer: [count]bool = @splat(false);
    for (0..count) |index| last_use[index] = index;

    for (expression.instructions, 0..) |instruction, user| {
        for (instruction.args[0..instruction.operation.arity()]) |reference| {
            switch (reference) {
                .instruction => |producer| {
                    has_consumer[producer] = true;
                    last_use[producer] = @max(last_use[producer], user);
                },
                else => {},
            }
        }
    }
    for (has_consumer, 0..) |used, index| {
        if (!used) last_use[index] = count;
    }

    var ends: [count + 1]usize = @splat(0);
    for (last_use) |last| ends[last] += 1;
    var peak: usize = 0;
    var live: usize = 0;
    for (0..count) |current| {
        live += 1;
        peak = @max(peak, live);
        live -= ends[current];
    }
    return peak;
}

pub fn maxArity(comptime expression: Expression) usize {
    var result: usize = 0;
    for (expression.instructions) |instruction| result = @max(result, instruction.operation.arity());
    return result;
}

/// Conservative architectural register budget. This is a cost threshold,
/// not a promise about the register allocator's exact assignment.
pub fn registerBudget(comptime vector_width: usize) usize {
    if (vector_width <= 1) return 24;
    return switch (builtin.target.cpu.arch) {
        .aarch64 => 32,
        .x86_64 => 16,
        else => 16,
    };
}

pub fn registerPressureWork(comptime iterations: u128, comptime vector_width: usize, comptime demand: usize) u128 {
    const budget = registerBudget(vector_width);
    if (demand <= budget) return 0;
    const excess = demand - budget;
    // One spill creates both a store and a later load. Vector spills move a
    // wider value, so account for their lane width as well.
    return iterations * excess * (2 + @as(u128, @max(vector_width, 1)) / 4);
}
