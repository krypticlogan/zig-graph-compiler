const zgc = @import("zgc");
const graph = @import("graph");

const options: zgc.compile.ZgirExecutableOptions = if (@hasDecl(graph, "executable_options")) .{
    .target = graph.executable_options.target,
    .frontier_index = graph.executable_options.frontier_index,
    .outputs = if (@hasField(@TypeOf(graph.executable_options), "outputs"))
        graph.executable_options.outputs
    else
        .owned,
} else .{};

pub const Executable = zgc.compile.executableFromZgirWithOptions(graph.source, options);
