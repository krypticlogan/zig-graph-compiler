const Semantic = @import("../operations/semantic.zig");
const contraction = @import("../kernels/contraction.zig");
const direct = @import("../kernels/root.zig");
const map_kernel = @import("../kernels/map/root.zig");
const reduction_kernel = @import("../kernels/fusion/reduction.zig");

/// Operation after optimization has selected physical layouts and concrete
/// kernel strategies. Unchanged semantic operations are retained directly;
/// only specialized operations gain an execution-specific representation.
pub const ExecutableCompute = union(enum) {
    direct: Semantic.Op.Compute,
    kernel: KernelPlan,

    pub fn execute(comptime compute: ExecutableCompute, inputs: anytype, outputs: anytype) void {
        switch (compute) {
            .direct => |semantic| {
                if (outputs.len != 1) @compileError("direct compute requires exactly one output");
                direct.execute(semantic, inputs, outputs[0]);
            },
            .kernel => |plan| switch (plan) {
                .map => |map| map_kernel.execute(map, inputs, outputs),
                .reduction => |reduction_plan| reduction_kernel.execute(reduction_plan, inputs, outputs),
                .contraction => |contraction_plan| {
                    if (outputs.len != 1) @compileError("contraction requires exactly one output");
                    contraction.matmulWithPlan(
                        contraction_plan.strategy,
                        inputs[0],
                        inputs[1],
                        outputs[0],
                    );
                },
            },
        }
    }
};

pub const Op = union(enum) {
    compute: ExecutableCompute,
    view: Semantic.Op.View,

    pub const Kind = Semantic.Op.Kind;
    pub const Compute = ExecutableCompute;

    pub fn kind(op: Op) Kind {
        return switch (op) {
            .compute => .compute,
            .view => .view,
        };
    }
};

pub const MapPlan = @import("../compiler/planning/map.zig").Plan;
pub const ReductionPlan = @import("../compiler/planning/reduction.zig").Plan;
pub const ContractionPlan = @import("../compiler/planning/contraction.zig").Plan;

/// Data-only physical plan selected after fusion and layout planning.
pub const KernelPlan = union(enum) {
    map: MapPlan,
    reduction: ReductionPlan,
    contraction: ContractionPlan,
};
