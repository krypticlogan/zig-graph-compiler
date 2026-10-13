const zgc = @import("zgc");
const graph = @import("graph");

const options: zgc.frontend.ZgirExecutableOptions = if (@hasDecl(graph, "executable_options")) .{
    .target = graph.executable_options.target,
    .frontier_index = graph.executable_options.frontier_index,
} else .{};

pub const Executable = zgc.frontend.executableFromZgirWithOptions(graph.source, options);
