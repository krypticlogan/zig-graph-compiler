const zgc = @import("zgc");
const graph = @import("graph");
const search_defaults: zgc.compile.SearchOptions = .{};

const options: zgc.compile.ZgirExecutableOptions = if (@hasDecl(graph, "executable_options")) .{
    .search = if (@hasField(@TypeOf(graph.executable_options), "search")) .{
        .max_executables = if (@hasField(@TypeOf(graph.executable_options.search), "max_executables")) graph.executable_options.search.max_executables else search_defaults.max_executables,
        .max_composition_attempts = if (@hasField(@TypeOf(graph.executable_options.search), "max_composition_attempts")) graph.executable_options.search.max_composition_attempts else search_defaults.max_composition_attempts,
    } else .{},
    .target = graph.executable_options.target,
    .frontier_index = graph.executable_options.frontier_index,
    .diagnostics = if (@hasField(@TypeOf(graph.executable_options), "diagnostics")) .{
        .log = if (@hasField(@TypeOf(graph.executable_options.diagnostics), "log")) graph.executable_options.diagnostics.log else false,
        .detailed = if (@hasField(@TypeOf(graph.executable_options.diagnostics), "detailed")) graph.executable_options.diagnostics.detailed else false,
        .stop_after = if (@hasField(@TypeOf(graph.executable_options.diagnostics), "stop_after")) graph.executable_options.diagnostics.stop_after else null,
    } else .{},
    .outputs = if (@hasField(@TypeOf(graph.executable_options), "outputs"))
        graph.executable_options.outputs
    else
        .owned,
} else .{};

pub const Executable = zgc.compile.executableFromZgirWithOptions(graph.source, options);
