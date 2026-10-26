//! Opt-in comptime diagnostics. Compile logs intentionally prevent artifact
//! emission. A stop checkpoint flushes diagnostics before a later expensive stage.
pub const Stage = enum {
    definition,
    counting,
    construction,
    raw_validation,
    semantic_optimization,
    semantic_finalization,
    semantic_canonicalization,
    canonical_use_counts,
    canonical_outputs,
    canonical_concat_analysis,
    canonical_tensor_order,
    canonical_rewrite,
    semantic_dead_elimination,
    semantic_provenance,
    semantic_validation,
    semantic_analysis,
    fusion_analysis,
    layout_analysis,
    remap_analysis,
    contraction_analysis,
    executable_search,
    search_composition,
    search_reference,
    reference_scheduling,
    reference_construction,
    search_layout,
    search_validation,
    search_lowering,
    search_scheduling,
    candidate_validation,
    candidate_lifetimes,
    candidate_sources,
    candidate_outputs,
    candidate_storage,
    candidate_cost,
    selection,
    final_validation,
    lifetime_analysis,
    source_planning,
    output_planning,
    executable_generation,

    fn detailed(stage: Stage) bool {
        return switch (stage) {
            .semantic_canonicalization,
            .canonical_use_counts,
            .canonical_outputs,
            .canonical_concat_analysis,
            .canonical_tensor_order,
            .canonical_rewrite,
            .semantic_dead_elimination,
            .semantic_provenance,
            .search_composition,
            .search_reference,
            .reference_scheduling,
            .reference_construction,
            .search_layout,
            .search_validation,
            .search_lowering,
            .search_scheduling,
            .candidate_validation,
            .candidate_lifetimes,
            .candidate_sources,
            .candidate_outputs,
            .candidate_storage,
            .candidate_cost,
            => true,
            else => false,
        };
    }
};

pub const Options = struct {
    log: bool = false,
    /// Include optimizer substeps, search representations, and candidate costs.
    detailed: bool = false,
    /// Intentionally fail immediately after this stage, before later work.
    stop_after: ?Stage = null,
};

pub fn Trace(comptime options: Options) type {
    return struct {
        fn enabled(comptime stage: Stage) bool {
            return options.stop_after == stage or
                ((options.log or options.stop_after != null) and (!stage.detailed() or options.detailed));
        }

        pub fn begin(comptime stage: Stage) void {
            if (enabled(stage)) @compileLog("zgc.compile", @tagName(stage), "begin");
        }

        pub fn end(comptime stage: Stage, comptime summary: anytype) void {
            if (enabled(stage)) @compileLog("zgc.compile", @tagName(stage), "done", summary);
            if (options.stop_after == stage) @compileError("ZGC diagnostic stop after " ++ @tagName(stage));
        }
    };
}
