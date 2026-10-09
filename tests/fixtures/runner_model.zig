const zgc = @import("zgc");

const Sources = enum(usize) { input };
const Definition = zgc.DefinitionBuilder;

pub const Model = model: {
    var builder = Definition.init();
    const builder_sources = builder.sources(Sources);
    const input = builder_sources.input(.input, .f32, &.{4});
    builder.output(builder.relu(input));
    break :model builder.finish().model();
};
