const Construction = @import("construction.zig");
const Analysis = @import("analysis/root.zig");
const Search = @import("search.zig");
const SemanticOptimization = @import("semantic_optimization.zig");
const validation = @import("validation.zig");
const Executable = @import("../core/executable.zig").Executable;
const Source = @import("../storage/source.zig");
const Output = @import("../storage/output.zig");
const std = @import("std");
const Diagnostics = @import("diagnostics.zig");
const Finalization = @import("finalization.zig");
/// Compile a completed definition into an executable.
pub fn executable(
    comptime Definition: type,
    comptime definition: Definition,
    comptime source_configuration: anytype,
    comptime output_binding: Output.Binding,
    comptime selection: Search.SelectionOptions,
    comptime diagnostics: Diagnostics.Options,
    comptime search_options: Search.Options,
) type {
    const compile_work = 40_000 + definition.node_count *
        (definition.tensor_count + definition.input_ref_count + Definition.max_rank + 16) *
        4096 * Search.max_candidates;
    @setEvalBranchQuota(@min(compile_work, std.math.maxInt(u32)));
    const Trace = Diagnostics.Trace(diagnostics);
    Trace.end(.definition, .{ .nodes = definition.node_count, .tensors = definition.tensor_count, .input_refs = definition.input_ref_count, .outputs = definition.output_count, .rank = Definition.max_rank });
    Trace.begin(.counting);
    const raw_capacity = Construction.count(Definition, definition);
    Trace.end(.counting, raw_capacity);
    Trace.begin(.construction);
    const raw_graph = Construction.GraphConstruction(Definition, raw_capacity).build(definition);
    Trace.end(.construction, .{ .nodes = raw_graph.node_ct, .tensors = raw_graph.tensor_ct, .graph_bytes = @sizeOf(@TypeOf(raw_graph)) });
    Trace.begin(.raw_validation);
    const RawValidated = validation.Validation(raw_capacity).validate(raw_graph);
    Trace.end(.raw_validation, .{ .nodes = RawValidated.graph.node_ct });
    Trace.begin(.semantic_optimization);
    const optimization_result = SemanticOptimization.SemanticOptimization(raw_capacity).optimize(RawValidated.graph, diagnostics);
    Trace.begin(.semantic_finalization);
    const semantic_graph = Finalization.semantic(optimization_result.graph);
    const capacity = Finalization.count(semantic_graph);
    const provenance = .{
        .raw_to_optimized_node = optimization_result.provenance.raw_to_optimized_node,
        .raw_to_optimized_tensor = optimization_result.provenance.raw_to_optimized_tensor,
        .optimized_to_raw_node = optimization_result.provenance.optimized_to_raw_node[0..semantic_graph.node_ct].*,
        .optimized_to_raw_tensor = optimization_result.provenance.optimized_to_raw_tensor[0..semantic_graph.tensor_ct].*,
    };
    const semantic_optimization = .{ .graph = semantic_graph, .provenance = provenance };
    Trace.end(.semantic_finalization, capacity);
    Trace.end(.semantic_optimization, .{ .nodes = semantic_graph.node_ct, .tensors = semantic_graph.tensor_ct, .capacity = capacity, .graph_bytes = @sizeOf(@TypeOf(semantic_graph)) });
    Trace.begin(.semantic_validation);
    const SemanticValidated = validation.Validation(capacity).validate(semantic_optimization.graph);
    Trace.end(.semantic_validation, .{ .nodes = SemanticValidated.graph.node_ct });
    Trace.begin(.semantic_analysis);
    const semantic_analysis = Analysis.SemanticAnalysis(capacity).analyze(SemanticValidated);
    Trace.end(.semantic_analysis, .{ .edges = semantic_analysis.dependencies.edge_count, .facts_bytes = @sizeOf(@TypeOf(semantic_analysis)) });
    Trace.begin(.fusion_analysis);
    const fusion_candidates = Analysis.FusionAnalysis(capacity).analyze(
        SemanticValidated.graph,
        semantic_analysis,
    );
    Trace.end(.fusion_analysis, .{ .candidates = fusion_candidates.count });
    Trace.begin(.layout_analysis);
    const layout_candidates = Analysis.LayoutAnalysis(capacity).analyze(
        SemanticValidated.graph,
        semantic_analysis,
    );
    Trace.end(.layout_analysis, .{ .candidates = layout_candidates.count });
    Trace.begin(.remap_analysis);
    const remap_candidates = Analysis.RemapAnalysis(capacity).analyze(
        SemanticValidated.graph,
        semantic_analysis,
    );
    Trace.end(.remap_analysis, .{ .candidates = remap_candidates.count });
    Trace.begin(.contraction_analysis);
    const contraction_candidates = Analysis.ContractionAnalysis(capacity).analyze(SemanticValidated.graph);

    Trace.end(.contraction_analysis, .{ .candidates = contraction_candidates.count });

    Trace.begin(.executable_search);
    const executable_search = Search.ExecutableSearch(capacity).search(
        Definition.Source,
        SemanticValidated,
        semantic_analysis,
        fusion_candidates,
        layout_candidates,
        remap_candidates,
        contraction_candidates,
        source_configuration,
        output_binding,
        diagnostics,
        search_options,
    );
    Trace.end(.executable_search, .{ .representations = executable_search.representation_candidate_count, .generated = executable_search.generated_count, .retained = executable_search.frontier.count });
    Trace.begin(.selection);
    const selected_candidate = executable_search.selected(selection);
    const program = Finalization.executable(selected_candidate.executable);
    const executable_capacity = Finalization.count(program);
    Trace.end(.selection, .{ .origin = selected_candidate.origin, .schedule = selected_candidate.schedule, .nodes = program.node_ct, .cost = selected_candidate.cost });
    Trace.begin(.final_validation);
    const FinalValidated = validation.FinalValidation(executable_capacity).validate(program);
    const graph = FinalValidated.graph;

    Trace.end(.final_validation, .{ .nodes = graph.node_ct });
    Trace.begin(.lifetime_analysis);
    const lifetime_analysis = Search.LifetimeAnalysis().analyze(FinalValidated);
    Trace.end(.lifetime_analysis, .{ .tensors = lifetime_analysis.tensor_lifetimes.len });
    Trace.begin(.source_planning);
    const SourcePlan = Source.Plan(
        Definition.Source,
        executable_capacity,
        graph,
        source_configuration,
    );
    Trace.end(.source_planning, .{ .sources = capacity.max_sources, .plan_type_bytes = @sizeOf(SourcePlan) });
    Trace.begin(.output_planning);
    const OutputPlan = Output.Plan(graph, output_binding);
    Trace.end(.output_planning, .{ .outputs = graph.output_ct, .plan_type_bytes = @sizeOf(OutputPlan) });
    Trace.begin(.executable_generation);
    const Compiled = Executable(
        Definition.Source,
        executable_capacity,
        raw_graph,
        semantic_optimization,
        SemanticValidated,
        semantic_analysis,
        executable_search,
        selection,
        FinalValidated,
        lifetime_analysis,
        SourcePlan,
        OutputPlan,
        &definition.source_names,
    );
    if (diagnostics.log or diagnostics.stop_after != null) {
        Trace.end(.executable_generation, .{ .memory_bytes = Compiled.memory_plan.byte_count, .runtime_bytes = @sizeOf(Compiled.Runtime) });
    }
    return Compiled;
}
