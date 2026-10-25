const std = @import("std");
const zgc = @import("zgc_internal").api;
const build_options = @import("build_options");

pub fn executable(
    comptime definition: anytype,
    comptime sources: []const @TypeOf(definition).SourceOverride,
) type {
    const target = std.meta.stringToEnum(
        zgc.OptimizationTarget,
        build_options.optimization_target,
    ) orelse @compileError("optimization_target must be balanced, memory, fast, or small");
    return definition.executableWith(.{
        .sources = sources,
        .target = target,
        .frontier_index = if (build_options.frontier_index == std.math.maxInt(usize))
            null
        else
            build_options.frontier_index,
    });
}
