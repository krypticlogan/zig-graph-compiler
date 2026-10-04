const Expression = @import("../fusion/expression.zig").Program;
const Domain = @import("domain.zig").Domain;
const Access = @import("access.zig");

/// A map-family region describes one logical iteration domain and the values
/// loaded and stored around it. The body may evaluate an expression or perform
/// a pure transfer; map planning chooses the physical strategy.
pub const Map = struct {
    domain: Domain,
    loads: []const Access.Load,
    body: Body,
    stores: []const Access.Store,

    pub const Body = union(enum) {
        expression: Expression,
        transfer,
        expression_transfer: Expression,
    };
};
