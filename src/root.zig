//! Public frontend ZGC API.

pub const build = struct {
    pub const DefinitionBuilder = @import("zgc/frontend/definition.zig").DefinitionBuilder;
    pub const Expr = @import("zgc/frontend/expressive.zig");
    pub const Value = @import("zgc/frontend/definition.zig").Value;
    pub const Shape = @import("zgc/frontend/definition.zig").Shape;
};

pub const compile = struct {
    pub const executableFromZgir = @import("zgc/frontend/instantiation.zig").executableFromZgir;
    pub const executableFromZgirWithOptions = @import("zgc/frontend/instantiation.zig").executableFromZgirWithOptions;
    pub const ZgirExecutableOptions = @import("zgc/frontend/instantiation.zig").ZgirExecutableOptions;
};

pub const op_options = struct {
    pub const ReductionOptions = @import("zgc/frontend/definition.zig").ReductionOptions;
    pub const FlattenOptions = @import("zgc/frontend/definition.zig").FlattenOptions;
    pub const SliceOptions = @import("zgc/frontend/definition.zig").SliceOptions;
    pub const PadOptions = @import("zgc/frontend/definition.zig").PadOptions;
    pub const WindowOptions = @import("zgc/frontend/definition.zig").WindowOptions;
};

pub const Tensor = @import("zgc/core/tensor.zig");

pub const memory = struct {
    pub const Dtype = @import("zgc/storage/dtype.zig").Dtype;
    pub const ScalarValue = @import("zgc/storage/dtype.zig").ScalarValue;
    pub const Source = @import("zgc/storage/source.zig");
    pub const Output = @import("zgc/storage/output.zig");
};

/// Runtime ABI for generated executable artifacts.
pub const artifact = struct {
    pub const ABI = @import("artifact/abi.zig");
};

pub const OptimizationTarget = @import("zgc/compiler/search.zig").OptimizationTarget; // move this type external to the compiler?
pub const DefinitionBuilder = build.DefinitionBuilder;
pub const Expr = build.Expr;
pub const Value = build.Value;
pub const Shape = build.Shape;
pub const ext = struct {
    pub const nn = @import("extensions/nn.zig");
    pub const img = @import("extensions/img.zig");
};
