const Expression = @import("../fusion/expression.zig").Program;
const ReductionOperation = @import("../../../operations/reduction.zig");
const Domain = @import("domain.zig").Domain;
const Access = @import("access.zig");

/// Logical computation assigned to one kernel invocation. Regions describe
/// what is computed together without choosing a physical execution strategy.
pub const Region = union(enum) {
    map: Map,
    reduction: Reduction,
    contraction: Contraction,
};

pub const Map = @import("map.zig").Map;

pub const Reduction = struct {
    domain: Domain,
    loads: []const Access.Load,
    expressions: Expression,
    reduction_axes: u64,
    keep_dims: bool,
    accumulators: []const Accumulator,
    stores: []const Access.Store,

    pub const Accumulator = struct {
        combine: Combine,
        update: Expression.ValueRef,
        finalize: Finalize,
    };

    pub const Combine = ReductionOperation.Combine;
    pub const Finalize = ReductionOperation.Finalize;
};

pub const Contraction = struct {
    domain: Domain,
    loads: []const Access.Load,
    epilogue: ?Expression = null,
    stores: []const Access.Store,
};
