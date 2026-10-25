const std = @import("std");
const zgc = @import("zgc_internal").api;
const embedded_parameters = @import("embed_params");

pub const BasicSources = enum(usize) { input };
const BasicDefinition = zgc.DefinitionBuilder;
pub const BasicExecutable = model: {
    var builder = BasicDefinition.init();
    const builder_sources = builder.sources(BasicSources);
    const input = builder_sources.input(.input, .f32, &.{ 2, 3 });
    builder.output(builder.relu(builder.transpose(input, 0, 1)));
    break :model builder.finish().executable();
};

const FullSources = enum(noreturn) {};
const FullDefinition = zgc.DefinitionBuilder;
pub const FullExecutable = model: {
    var builder = FullDefinition.init();
    builder.output(builder.full(.f32, &.{ 2, 3 }, 7.5));
    break :model builder.finish().executable();
};

pub const ParameterSources = enum(usize) { input, parameter };
const ParameterDefinition = zgc.DefinitionBuilder;
const parameter_definition = definition: {
    var builder = ParameterDefinition.init();
    const builder_sources = builder.sources(ParameterSources);
    const input = builder_sources.input(.input, .f32, &.{ 2, 2 });
    const parameter = builder_sources.parameter(.parameter, .f32, &.{ 2, 2 });
    builder.output(builder.add(input, parameter));
    break :definition builder.finish();
};
pub const EmbeddedParameterExecutable = parameter_definition.executableWith(.{ .sources = &.{
    .{ .source = .parameter, .binding = zgc.memory.Source.embed(embedded_parameters.weights[0]) },
} });
pub const BoundInputExecutable = parameter_definition.executableWith(.{ .sources = &.{
    .{ .source = .input, .binding = zgc.memory.Source.bound },
    .{ .source = .parameter, .binding = zgc.memory.Source.embed(embedded_parameters.weights[0]) },
} });

pub const MatmulSources = enum(usize) { input, weights };
pub const matmul_batch = std.simd.suggestVectorLength(f32) orelse 4;
const MatmulDefinition = zgc.DefinitionBuilder;
pub const matmul_definition = definition: {
    var builder = MatmulDefinition.init();
    const builder_sources = builder.sources(MatmulSources);
    const input = builder_sources.input(.input, .f32, &.{ matmul_batch, 3 });
    const weights = builder_sources.parameter(.weights, .f32, &.{ 3, 2 });
    builder.output(builder.matmul(input, weights));
    break :definition builder.finish();
};
pub const MatmulExecutable = matmul_definition.executable();
const logical_weights = [_]f32{ 1, 2, 3, 4, 5, 6 };
const packed_weights = [_]f32{ 1, 3, 5, 2, 4, 6 };
pub const EmbeddedMatmulExecutable = matmul_definition.executableWith(.{ .sources = &.{
    .{ .source = .weights, .binding = zgc.memory.Source.embed(std.mem.asBytes(&logical_weights)) },
} });
pub const PackedMatmulExecutable = matmul_definition.executableWith(.{ .sources = &.{
    .{ .source = .weights, .binding = zgc.memory.Source.embedPacked(std.mem.asBytes(&packed_weights)) },
} });

const ReuseDefinition = zgc.DefinitionBuilder;
const reuse_definition = definition: {
    var builder = ReuseDefinition.init();
    const builder_sources = builder.sources(BasicSources);
    const input = builder_sources.input(.input, .f32, &.{4});
    const first = builder.copy(input);
    const second = builder.copy(first);
    builder.output(builder.copy(second));
    break :definition builder.finish();
};
pub const ReuseExecutable = reuse_definition.executable();
pub const BoundReuseExecutable = reuse_definition.executableWith(.{ .sources = &.{
    .{ .source = .input, .binding = zgc.memory.Source.bound },
} });
