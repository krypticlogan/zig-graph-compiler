const Construction = @import("construction.zig");
const Analysis = @import("analysis/root.zig");
const Search = @import("search.zig");
const SemanticOptimization = @import("semantic_optimization.zig");
const validation = @import("validation.zig");
const Model = @import("../core/model.zig").Model;
const Source = @import("../storage/source.zig");

/// Run the compilation steps from a completed definition to a generated model.
pub fn model(
    comptime Definition: type,
    comptime definition: Definition,
    comptime source_configuration: anytype,
) type {
    const compile_work = 10_000 + definition.node_count *
        (definition.tensor_count + definition.input_ref_count + Definition.max_rank + 16) * 1024;
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

    const executable_search = Search.ExecutableSearch(capacity).search(
        Definition.Source,
        SemanticValidated,
        semantic_analysis,
        fusion_candidates,
        layout_candidates,
        remap_candidates,
        source_configuration,
    );
    const selected_candidate = executable_search.selected();
    const executable = selected_candidate.executable;
    const FinalValidated = validation.FinalValidation(capacity).validate(executable);
    const graph = FinalValidated.graph;

    const lifetime_analysis = Search.LifetimeAnalysis().analyze(FinalValidated);
    const SourcePlan = Source.Plan(
        Definition.Source,
        capacity,
        graph,
        source_configuration,
    );
    return Model(
        Definition.Source,
        capacity,
        raw_graph,
        semantic_optimization,
        SemanticValidated,
        semantic_analysis,
        executable_search,
        FinalValidated,
        lifetime_analysis,
        SourcePlan,
        &definition.source_names,
    );
}
