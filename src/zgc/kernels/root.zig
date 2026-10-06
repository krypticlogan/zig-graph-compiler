const Op = @import("../operations/semantic.zig").Op;
const Pointwise = @import("../operations/elementwise.zig");
const concatenation = @import("concatenation.zig");
const contraction = @import("contraction.zig");
const elementwise = @import("elementwise.zig");
const materialization = @import("materialization.zig");
const padding = @import("padding.zig");
const reduction = @import("reduction.zig");
const shifting = @import("shifting.zig");
const slice_loop = @import("slice_loop.zig");
const special = @import("special.zig");

/// Execute one semantic compute operation without a specialized kernel plan.
pub inline fn execute(comptime op: Op.Compute, inputs: anytype, output: anytype) void {
    if (comptime Pointwise.fromCompute(op)) |pointwise| {
        elementwise.execute(pointwise, inputs, output);
        return;
    }
    switch (op) {
        .copy, .contiguous => materialization.copy(inputs[0], output),
        .pad => |attrs| padding.constant(inputs[0], inputs[1], output, attrs),
        .shift => |attrs| shifting.shift(inputs, output, attrs),
        .slice_loop => |attrs| slice_loop.execute(attrs, inputs[0], output),
        .matmul => contraction.matmulWithPlan(.scalar, inputs[0], inputs[1], output),
        .sum => |attrs| reduction.sum(inputs[0], output, attrs),
        .mean => |attrs| reduction.mean(inputs[0], output, attrs),
        .min => |attrs| reduction.min(inputs[0], output, attrs),
        .max => |attrs| reduction.max(inputs[0], output, attrs),
        .concat => |attrs| concatenation.concat(inputs, output, attrs.axis),
        .softmax => |attrs| special.softmax(inputs[0], output, attrs.axis),
        else => unreachable,
    }
}
