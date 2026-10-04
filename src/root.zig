//! Public ZGC API.
pub const frontend = struct {
    pub const DefinitionBuilder = @import("zgc/frontend/definition.zig").DefinitionBuilder;
    pub const DefinitionLimits = @import("zgc/frontend/definition.zig").Limits;
    pub const ReductionOptions = @import("zgc/frontend/definition.zig").ReductionOptions;
    pub const FlattenOptions = @import("zgc/frontend/definition.zig").FlattenOptions;
    pub const SliceOptions = @import("zgc/frontend/definition.zig").SliceOptions;
    pub const PadOptions = @import("zgc/frontend/definition.zig").PadOptions;
    pub const WindowOptions = @import("zgc/frontend/definition.zig").WindowOptions;
};

pub const compiler = struct {
    pub const PlanCandidate = @import("zgc/compiler/search.zig").PlanCandidate;
    pub const PlanCost = @import("zgc/compiler/search.zig").PlanCost;
    pub const MemoryTraffic = @import("zgc/compiler/search.zig").MemoryTraffic;
    pub const ConversionCost = @import("zgc/compiler/search.zig").ConversionCost;
    pub const LayoutRequirement = @import("zgc/compiler/search.zig").LayoutRequirement;
    pub const LayoutResult = @import("zgc/compiler/search.zig").LayoutResult;
    pub const Validation = @import("zgc/validation.zig");
};

pub const core = struct {
    pub const GraphCapacity = @import("zgc/core/graph.zig").Capacity;
    pub const Tensor = @import("zgc/core/tensor.zig");
    pub const Model = @import("zgc/core/model.zig").Model;
};

pub const memory = struct {
    pub const Dtype = @import("zgc/storage/dtype.zig").Dtype;
    pub const ScalarValue = @import("zgc/storage/dtype.zig").ScalarValue;
    pub const Source = @import("zgc/storage/source.zig");
    pub const Storage = @import("zgc/storage/storage.zig");
};

pub const Op = @import("zgc/operations/semantic.zig").Op;

pub const execution = struct {
    pub const Executable = @import("zgc/execution/program.zig").Executable;
    pub const ExecutableCompute = @import("zgc/execution/execution.zig").ExecutableCompute;
    pub const KernelPlan = @import("zgc/execution/execution.zig").KernelPlan;
};

pub const DefinitionBuilder = frontend.DefinitionBuilder;
pub const DefinitionLimits = frontend.DefinitionLimits;
pub const ReductionOptions = frontend.ReductionOptions;
pub const FlattenOptions = frontend.FlattenOptions;
pub const SliceOptions = frontend.SliceOptions;
pub const PadOptions = frontend.PadOptions;
pub const WindowOptions = frontend.WindowOptions;
pub const Inspect = @import("zgc/core/inspect.zig");

pub const ext = struct {
    pub const nn = @import("extensions/nn.zig");
    pub const img = @import("extensions/img.zig");
};
