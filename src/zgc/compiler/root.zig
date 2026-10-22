const Construction = @import("construction.zig");
const Analysis = @import("analysis/root.zig");
const Search = @import("search.zig");
const SemanticOptimization = @import("semantic_optimization.zig");
const validation = @import("validation.zig");
const Executable = @import("../core/executable.zig").Executable;
const Source = @import("../storage/source.zig");
const Output = @import("../storage/output.zig");

/// Compile a completed definition into an executable.
pub fn executable(
    comptime Definition: type,
    comptime definition: Definition,
    comptime source_configuration: anytype,
    comptime output_binding: Output.Binding,
    comptime selection: Search.SelectionOptions,
) type {
    const compile_work = 40_000 + definition.node_count *
        (definition.tensor_count + definition.input_ref_count + Definition.max_rank + 16) *
        4096 * Search.max_candidates;
    @setEvalBranchQuota(compile_work);
    const capacity = Construction.count(Definition, definition);
    const raw_graph = Construction.GraphConstruction(Definition, capacity).build(definition);
    const RawValidated = validation.Validation(capacity).validate(raw_graph);
    const semantic_optimization = SemanticOptimization.SemanticOptimization(capacity).optimize(RawValidated.graph);
    const SemanticValidated = validation.Validation(capacity).validate(semantic_optimization.graph);
    const semantic_analysis = Analysis.SemanticAnalysis(capacity).analyze(SemanticValidated);
    const fusion_candidates = Analysis.FusionAnalysis(capacity).analyze(
        SemanticValidated.graph,
        semantic_analysis,
    );
    const layout_candidates = Analysis.LayoutAnalysis(capacity).analyze(
        SemanticValidated.graph,
        semantic_analysis,
    );
    const remap_candidates = Analysis.RemapAnalysis(capacity).analyze(
        SemanticValidated.graph,
        semantic_analysis,
    );
    const contraction_candidates = Analysis.ContractionAnalysis(capacity).analyze(SemanticValidated.graph);

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
    );
    const selected_candidate = executable_search.selected(selection);
    const program = selected_candidate.executable;
    const FinalValidated = validation.FinalValidation(capacity).validate(program);
    const graph = FinalValidated.graph;

    const lifetime_analysis = Search.LifetimeAnalysis().analyze(FinalValidated);
    const SourcePlan = Source.Plan(
        Definition.Source,
        capacity,
        graph,
        source_configuration,
    );
    const OutputPlan = Output.Plan(graph, output_binding);
    return Executable(
        Definition.Source,
        capacity,
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
}
