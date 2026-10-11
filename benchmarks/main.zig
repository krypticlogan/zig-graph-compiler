const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const Workload = @import("benchmark_case");

const run_count = build_options.runs;

comptime {
    if (run_count == 0) @compileError("benchmark runs must be greater than zero");
    if (build_options.iterations == 0 and build_options.sample_ms == 0) {
        @compileError("sample_ms must be greater than zero when iterations are calibrated");
    }
    if (build_options.warmup_iterations == 0 and build_options.warmup_ms == 0) {
        @compileError("warmup_ms must be greater than zero when warmup iterations are automatic");
    }
}

pub fn main(init: std.process.Init) !void {
    const zgc_result = try runBenchmark(Workload.ZgcBenchmark, init);
    const direct_result = try runBenchmark(Workload.DirectBenchmark, init);
    const speed_ratio = direct_result.average_ns / zgc_result.average_ns;
    std.debug.print(
        \\Comparison
        \\  average latency  ZGC {d:.3} ns | direct {d:.3} ns
        \\  direct / ZGC    {d:.3}x (below 1.0 means direct is faster)
        \\
    ,
        .{ zgc_result.average_ns, direct_result.average_ns, speed_ratio },
    );
    if (build_options.results_file.len != 0) {
        try appendResultIndex(init, .{
            .schema_version = 2,
            .workload = Workload.ZgcBenchmark.name,
            .optimization_mode = @tagName(builtin.mode),
            .environment = .{
                .operating_system = @tagName(builtin.target.os.tag),
                .architecture = @tagName(builtin.target.cpu.arch),
                .cpu_model = builtin.target.cpu.model.name,
                .zig_version = builtin.zig_version_string,
            },
            .selection = selectionRecord(Workload.ZgcBenchmark.CompiledModel),
            .configuration = .{
                .runs = run_count,
                .iterations_per_sample = zgc_result.iterations,
                .warmup_iterations = zgc_result.warmup_iterations,
                .sample_ms_target = if (build_options.iterations == 0) build_options.sample_ms else 0,
                .explicit_iterations = build_options.iterations,
            },
            .zgc = benchmarkRecord(Workload.ZgcBenchmark, zgc_result),
            .direct = benchmarkRecord(Workload.DirectBenchmark, direct_result),
            .calibration = calibrationRecord(Workload.ZgcBenchmark.CompiledModel, zgc_result.average_ns),
            .direct_over_zgc = speed_ratio,
        });
        std.debug.print("  JSON record    {s}\n", .{build_options.results_file});
    }
}

const BenchmarkResult = struct {
    minimum_ns: f64,
    average_ns: f64,
    maximum_ns: f64,
    median_ns: f64,
    percentile_95_ns: f64,
    standard_deviation_ns: f64,
    coefficient_of_variation_percent: f64,
    iterations: usize,
    warmup_iterations: usize,
    timed_elapsed_ns: f64,
    samples_ns: [run_count]f64,
};

const TrafficRecord = struct {
    estimated_bytes_read: u128,
    estimated_bytes_written: u128,
};

const MemoryRecord = struct {
    planned_peak_bytes: usize,
    persistent_bytes: usize,
    scratch_bytes: usize,
    model_state_bytes: usize,
};

const CostRecord = struct {
    estimated_runtime_work: u128,
    arithmetic_work: u128,
    access_work: u128,
    register_pressure_work: u128,
    transfer_work: u128,
    traffic: TrafficRecord,
    memory: MemoryRecord,
    code_size_units: usize,
};

const SelectionRecord = struct {
    target: []const u8,
    frontier_index: usize,
    automatic_frontier_index: usize,
    frontier_count: usize,
    forced: bool,
    origin: []const u8,
    schedule: []const u8,
    fusion: []const u8,
    layout: []const u8,
    remap: []const u8,
    node_count: usize,
    cost: CostRecord,
    structure: StructureRecord,
};

const StructureRecord = struct {
    fingerprint: []const u8,
    nodes: []const NodeRecord,
    tensors: []const TensorRecord,
};

const NodeRecord = struct {
    kind: []const u8,
    operation: []const u8,
    strategy: []const u8 = "none",
    input_count: usize,
    output_count: usize,
    inputs: []const usize = &.{},
    outputs: []const usize = &.{},
    vector_axis: ?u8 = null,
    vector_width: ?usize = null,
    unroll: ?usize = null,
    axis_order: []const usize = &.{},
    reduction_axis_order: []const usize = &.{},
    accumulator_lanes: ?usize = null,
    segment_count: ?usize = null,
    domain_shape: []const usize = &.{},
    expression_operations: []const []const u8 = &.{},
    load_count: ?usize = null,
    store_count: ?usize = null,
    accumulator_count: ?usize = null,
};

const TensorRecord = struct {
    dtype: []const u8,
    shape: []const usize,
    strides: []const isize,
    offset: usize,
    storage_tensor: usize,
    origin: []const u8,
    origin_id: ?usize,
    materialized: bool,
};

const BenchmarkRecord = struct {
    name: []const u8,
    average_latency_ns: f64,
    minimum_latency_ns: f64,
    maximum_latency_ns: f64,
    median_latency_ns: f64,
    p95_latency_ns: f64,
    standard_deviation_ns: f64,
    coefficient_of_variation_percent: f64,
    samples_ns: [run_count]f64,
    iterations_per_sample: usize,
    work_items_per_second: f64,
    logical_gib_per_second: f64,
};

const CalibrationRecord = struct {
    observed_ns_per_work_unit: ?f64,
    estimated_traffic_gib_per_second: f64,
};

fn runBenchmark(comptime Selected: type, init: std.process.Init) !BenchmarkResult {
    const clock = std.Io.Clock.awake;
    var selected = Selected.init();
    if (comptime @hasDecl(Selected, "prepare")) try selected.prepare();
    if (comptime @hasDecl(Selected, "validate")) try selected.validate();

    const warmup = warmUp(Selected, &selected, clock, init);
    const iterations = calibrateIterations(Selected, &selected, clock, init);

    var samples_ns: [run_count]f64 = undefined;
    var timed_elapsed_ns: f64 = 0;
    for (&samples_ns) |*sample| {
        const start = clock.now(init.io).nanoseconds;
        selected.run(iterations);
        const end = clock.now(init.io).nanoseconds;
        const elapsed_ns: f64 = @floatFromInt(end - start);
        timed_elapsed_ns += elapsed_ns;
        sample.* = elapsed_ns / @as(f64, @floatFromInt(iterations));
    }

    var minimum_ns = std.math.inf(f64);
    var maximum_ns: f64 = 0;
    var total_ns: f64 = 0;
    for (samples_ns) |sample| {
        minimum_ns = @min(minimum_ns, sample);
        maximum_ns = @max(maximum_ns, sample);
        total_ns += sample;
    }
    const average_ns = total_ns / @as(f64, @floatFromInt(run_count));

    var squared_deviation: f64 = 0;
    for (samples_ns) |sample| {
        const difference = sample - average_ns;
        squared_deviation += difference * difference;
    }
    const standard_deviation_ns = @sqrt(
        squared_deviation / @as(f64, @floatFromInt(run_count)),
    );
    var sorted_samples = samples_ns;
    std.mem.sort(f64, &sorted_samples, {}, std.sort.asc(f64));
    const median_ns = if (run_count % 2 == 0)
        (sorted_samples[run_count / 2 - 1] + sorted_samples[run_count / 2]) / 2
    else
        sorted_samples[run_count / 2];
    const percentile_95_index = @min((95 * run_count + 99) / 100 - 1, run_count - 1);
    const percentile_95_ns = sorted_samples[percentile_95_index];

    const invocations_per_second = @as(f64, std.time.ns_per_s) / average_ns;
    const work_items_per_second = invocations_per_second * Selected.work_items_per_invocation;
    const bytes_per_second = invocations_per_second * Selected.bytes_per_invocation;
    const coefficient_of_variation = standard_deviation_ns / average_ns * 100;

    std.debug.print(
        \\Benchmark: {s} ({s})
        \\  configuration  {d} warmup iterations / {d:.3} s | {d} samples x {d} iterations
        \\  sample timing  {d} ms target | {d:.3} s total timed
        \\  latency (ns)   min {d:>10.3} | avg {d:>10.3} | max {d:>10.3}
        \\  distribution   median {d:.3} ns | p95 {d:.3} ns
        \\  variability    stddev {d:.3} ns | CV {d:.2}%
        \\  throughput     {d:.3} invocations/s | {d:.3} {s}/s | {d:.3} GiB/s
        \\
    , .{
        Selected.name,
        @tagName(builtin.mode),
        warmup.iterations,
        @as(f64, @floatFromInt(warmup.elapsed_ns)) / std.time.ns_per_s,
        run_count,
        iterations,
        if (build_options.iterations == 0) build_options.sample_ms else 0,
        timed_elapsed_ns / std.time.ns_per_s,
        minimum_ns,
        average_ns,
        maximum_ns,
        median_ns,
        percentile_95_ns,
        standard_deviation_ns,
        coefficient_of_variation,
        invocations_per_second,
        work_items_per_second,
        Selected.work_unit,
        bytes_per_second / (1024 * 1024 * 1024),
    });

    if (comptime @hasDecl(Selected, "latency_divisor")) {
        const divisor = Selected.latency_divisor;
        std.debug.print(
            "  normalized     min {d:>10.3} | avg {d:>10.3} | max {d:>10.3} ns/{s}\n",
            .{
                minimum_ns / divisor,
                average_ns / divisor,
                maximum_ns / divisor,
                Selected.latency_unit,
            },
        );
    }
    if (comptime @hasDecl(Selected, "parameter_count")) {
        std.debug.print(
            "  model          {d} parameters | batch {d}\n",
            .{ Selected.parameter_count, Selected.batch },
        );
    }
    if (comptime @hasDecl(Selected, "CompiledModel")) {
        const Model = Selected.CompiledModel;
        const choice = Model.selected_executable_candidate;
        const cost = choice.cost;
        const estimated_traffic_bytes = cost.memory_traffic.bytes_read + cost.memory_traffic.bytes_written;
        const implied_traffic_gib_per_second = @as(f64, @floatFromInt(estimated_traffic_bytes)) /
            average_ns * std.time.ns_per_s / (1024 * 1024 * 1024);
        std.debug.print(
            "  executable     {s} | schedule {s} | fusion {s} | layout {s} | remap {s} | {d} nodes\n",
            .{
                @tagName(choice.origin),
                @tagName(choice.schedule),
                regimeName(choice.fusion_regime),
                regimeName(choice.layout_regime),
                regimeName(choice.remap_regime),
                Model.executable.node_ct,
            },
        );
        std.debug.print(
            "  selection      {s} | frontier index {d}/{d}{s}\n",
            .{
                @tagName(Model.selection_options.target),
                Model.selected_executable_candidate_index,
                Model.executable_candidate_frontier.count,
                if (Model.selection_options.frontier_index != null) " (forced)" else "",
            },
        );
        std.debug.print("  fingerprint    exec-v1-{x:0>16}\n", .{Model.structural_fingerprint});
        if (comptime build_options.show_frontier) {
            inline for (Model.executable_candidate_frontier.candidates[0..Model.executable_candidate_frontier.count], 0..) |entry, index| {
                const candidate = entry.?;
                std.debug.print(
                    "    [{d}] {s} | schedule {s} | fusion {s} | layout {s} | remap {s} | work {d} | peak {d} B | code {d} | exec-v1-{x:0>16}\n",
                    .{
                        index,
                        @tagName(candidate.origin),
                        @tagName(candidate.schedule),
                        regimeName(candidate.fusion_regime),
                        regimeName(candidate.layout_regime),
                        regimeName(candidate.remap_regime),
                        candidate.cost.estimated_runtime_work,
                        candidate.cost.peak_memory_bytes,
                        candidate.cost.code_size_units,
                        @import("zgc").compiler.analysis.fingerprint.executable(candidate.executable),
                    },
                );
            }
        }
        std.debug.print(
            \\  cost estimate  {d} work units | arithmetic {d} | access {d} | registers {d} | transfer {d}
            \\  traffic est.   read {d} B | write {d} B | {d:.3} GiB/s implied by latency
            \\  storage plan   peak {d} B | persistent {d} B | scratch {d} B | model state {d} B
            \\  code estimate  {d} units
            \\
        , .{
            cost.estimated_runtime_work,
            cost.arithmetic_work,
            cost.access_work,
            cost.register_pressure_work,
            cost.transfer_work,
            cost.memory_traffic.bytes_read,
            cost.memory_traffic.bytes_written,
            implied_traffic_gib_per_second,
            cost.peak_memory_bytes,
            cost.persistent_memory_bytes,
            cost.scratch_memory_bytes,
            @sizeOf(Model),
            cost.code_size_units,
        });
        if (cost.estimated_runtime_work != 0) {
            std.debug.print(
                "  calibration    {d:.3} ns observed / {d} work units = {d:.6} ns/unit\n",
                .{
                    average_ns,
                    cost.estimated_runtime_work,
                    average_ns / @as(f64, @floatFromInt(cost.estimated_runtime_work)),
                },
            );
        }
    }
    std.debug.print("\n", .{});
    return .{
        .minimum_ns = minimum_ns,
        .average_ns = average_ns,
        .maximum_ns = maximum_ns,
        .median_ns = median_ns,
        .percentile_95_ns = percentile_95_ns,
        .standard_deviation_ns = standard_deviation_ns,
        .coefficient_of_variation_percent = coefficient_of_variation,
        .iterations = iterations,
        .warmup_iterations = warmup.iterations,
        .timed_elapsed_ns = timed_elapsed_ns,
        .samples_ns = samples_ns,
    };
}

fn selectionRecord(comptime Model: type) SelectionRecord {
    const choice = Model.selected_executable_candidate;
    const cost = choice.cost;
    return .{
        .target = @tagName(Model.selection_options.target),
        .frontier_index = Model.selected_executable_candidate_index,
        .automatic_frontier_index = Model.executable_candidate_frontier.bestIndex(Model.selection_options.target),
        .frontier_count = Model.executable_candidate_frontier.count,
        .forced = Model.selection_options.frontier_index != null,
        .origin = @tagName(choice.origin),
        .schedule = @tagName(choice.schedule),
        .fusion = regimeName(choice.fusion_regime),
        .layout = regimeName(choice.layout_regime),
        .remap = regimeName(choice.remap_regime),
        .node_count = Model.executable.node_ct,
        .cost = .{
            .estimated_runtime_work = cost.estimated_runtime_work,
            .arithmetic_work = cost.arithmetic_work,
            .access_work = cost.access_work,
            .register_pressure_work = cost.register_pressure_work,
            .transfer_work = cost.transfer_work,
            .traffic = .{
                .estimated_bytes_read = cost.memory_traffic.bytes_read,
                .estimated_bytes_written = cost.memory_traffic.bytes_written,
            },
            .memory = .{
                .planned_peak_bytes = cost.peak_memory_bytes,
                .persistent_bytes = cost.persistent_memory_bytes,
                .scratch_bytes = cost.scratch_memory_bytes,
                .model_state_bytes = @sizeOf(Model),
            },
            .code_size_units = cost.code_size_units,
        },
        .structure = structureRecord(Model),
    };
}

fn structureRecord(comptime Model: type) StructureRecord {
    const graph = Model.executable;
    const nodes = comptime blk: {
        var records: [graph.node_ct]NodeRecord = undefined;
        for (graph.nodes[0..graph.node_ct], 0..) |node, index| {
            const invocation = node.?;
            records[index] = nodeRecord(invocation);
            records[index].inputs = refIndices(graph.input_refs[invocation.input_start..][0..invocation.input_count]);
            records[index].outputs = refIndices(graph.output_refs[invocation.output_start..][0..invocation.output_count]);
        }
        break :blk records;
    };
    const tensors = comptime blk: {
        var records: [graph.tensor_ct]TensorRecord = undefined;
        for (graph.tensors[0..graph.tensor_ct], 0..) |maybe_tensor, index| {
            const tensor = maybe_tensor.?;
            records[index] = .{
                .dtype = @tagName(tensor.dtype),
                .shape = tensor.shape.dims[0..tensor.shape.rank],
                .strides = tensor.layout.strides[0..tensor.shape.rank],
                .offset = tensor.layout.offset,
                .storage_tensor = tensor.storage_tensor,
                .origin = @tagName(tensor.origin),
                .origin_id = switch (tensor.origin) {
                    .source => |source_id| source_id,
                    .node => |node_id| node_id,
                    .literal => null,
                },
                .materialized = graph.materialized[index],
            };
        }
        break :blk records;
    };
    return .{
        .fingerprint = std.fmt.comptimePrint("exec-v1-{x:0>16}", .{Model.structural_fingerprint}),
        .nodes = &nodes,
        .tensors = &tensors,
    };
}

fn nodeRecord(comptime node: anytype) NodeRecord {
    var record: NodeRecord = .{
        .kind = @tagName(node.op),
        .operation = "none",
        .input_count = node.input_count,
        .output_count = node.output_count,
    };
    switch (node.op) {
        .view => |view| record.operation = @tagName(view),
        .compute => |compute| switch (compute) {
            .direct => |direct| record.operation = @tagName(direct),
            .kernel => |kernel| switch (kernel) {
                .contraction => |plan| {
                    record.operation = "contraction";
                    record.strategy = @tagName(plan.strategy);
                    record.domain_shape = plan.region.domain.shape;
                    record.load_count = plan.region.loads.len;
                    record.store_count = plan.region.stores.len;
                    if (plan.region.epilogue) |expression| {
                        record.expression_operations = expressionOperations(expression);
                    }
                },
                .map => |plan| {
                    record.operation = "map";
                    record.domain_shape = plan.region.domain.shape;
                    record.load_count = plan.region.loads.len;
                    record.store_count = plan.region.stores.len;
                    switch (plan.region.body) {
                        .expression, .expression_transfer => |expression| {
                            record.expression_operations = expressionOperations(expression);
                        },
                        .transfer => {},
                    }
                    switch (plan.strategy) {
                        .traversal => |traversal| {
                            record.strategy = @tagName(traversal.traversal);
                            record.vector_axis = traversal.vector_axis;
                            record.vector_width = traversal.vector_width;
                            record.unroll = traversal.unroll;
                            record.axis_order = axisOrder(traversal.axis_order);
                        },
                        .segmented => |segmented| {
                            record.strategy = "segmented";
                            record.vector_width = segmented.vector_width;
                            record.segment_count = segmented.segments.len;
                        },
                        .loop => |loop| {
                            record.strategy = "loop";
                            record.vector_axis = loop.axis;
                            record.vector_width = loop.vector_width;
                            record.segment_count = loop.iterations.len;
                        },
                    }
                },
                .reduction => |plan| {
                    record.operation = "reduction";
                    record.strategy = "traversal";
                    record.domain_shape = plan.region.domain.shape;
                    record.expression_operations = expressionOperations(plan.region.expressions);
                    record.load_count = plan.region.loads.len;
                    record.store_count = plan.region.stores.len;
                    record.accumulator_count = plan.region.accumulators.len;
                    record.vector_axis = plan.traversal_plan.vector_axis;
                    record.vector_width = plan.traversal_plan.vector_width;
                    record.unroll = plan.traversal_plan.unroll;
                    record.axis_order = axisOrder(plan.traversal_plan.outer_axis_order);
                    record.reduction_axis_order = axisOrder(plan.traversal_plan.reduction_axis_order);
                    record.accumulator_lanes = plan.traversal_plan.accumulator_lanes;
                },
            },
        },
    }
    return record;
}

fn axisOrder(comptime axes: []const u8) []const usize {
    const values = comptime blk: {
        var result: [axes.len]usize = undefined;
        for (axes, 0..) |axis, index| result[index] = axis;
        break :blk result;
    };
    return &values;
}

fn refIndices(comptime refs: anytype) []const usize {
    const values = comptime blk: {
        var result: [refs.len]usize = undefined;
        for (refs, 0..) |reference, index| result[index] = reference.?;
        break :blk result;
    };
    return &values;
}

fn expressionOperations(comptime expression: anytype) []const []const u8 {
    const operations = comptime blk: {
        var result: [expression.instructions.len][]const u8 = undefined;
        for (expression.instructions, 0..) |instruction, index| {
            result[index] = @tagName(instruction.operation);
        }
        break :blk result;
    };
    return &operations;
}

fn benchmarkRecord(comptime Selected: type, result: BenchmarkResult) BenchmarkRecord {
    return .{
        .name = Selected.name,
        .average_latency_ns = result.average_ns,
        .minimum_latency_ns = result.minimum_ns,
        .maximum_latency_ns = result.maximum_ns,
        .median_latency_ns = result.median_ns,
        .p95_latency_ns = result.percentile_95_ns,
        .standard_deviation_ns = result.standard_deviation_ns,
        .coefficient_of_variation_percent = result.coefficient_of_variation_percent,
        .samples_ns = result.samples_ns,
        .iterations_per_sample = result.iterations,
        .work_items_per_second = @as(f64, std.time.ns_per_s) / result.average_ns * Selected.work_items_per_invocation,
        .logical_gib_per_second = @as(f64, std.time.ns_per_s) / result.average_ns * Selected.bytes_per_invocation / (1024 * 1024 * 1024),
    };
}

fn calibrationRecord(comptime Model: type, average_ns: f64) CalibrationRecord {
    const cost = Model.selected_executable_candidate.cost;
    const estimated_traffic = cost.memory_traffic.bytes_read + cost.memory_traffic.bytes_written;
    return .{
        .observed_ns_per_work_unit = if (cost.estimated_runtime_work == 0)
            null
        else
            average_ns / @as(f64, @floatFromInt(cost.estimated_runtime_work)),
        .estimated_traffic_gib_per_second = @as(f64, @floatFromInt(estimated_traffic)) /
            average_ns * std.time.ns_per_s / (1024 * 1024 * 1024),
    };
}

fn appendResultIndex(init: std.process.Init, record: anytype) !void {
    const allocator = init.gpa;
    const record_json = try std.json.Stringify.valueAlloc(allocator, record, .{});
    defer allocator.free(record_json);

    const directory = std.Io.Dir.cwd();
    const existing = directory.readFileAlloc(init.io, build_options.results_file, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return writeNewResultIndex(init, record_json),
        else => return err,
    };
    defer allocator.free(existing);

    const trimmed = std.mem.trimEnd(u8, existing, " \t\r\n");
    if (trimmed.len < 2 or trimmed[0] != '[' or trimmed[trimmed.len - 1] != ']') {
        return error.InvalidBenchmarkResultIndex;
    }
    const entries = std.mem.trimEnd(u8, trimmed[1 .. trimmed.len - 1], " \t\r\n");
    var output = std.Io.Writer.Allocating.init(allocator);
    defer output.deinit();
    try output.writer.writeAll(if (entries.len == 0) "[\n" else trimmed[0 .. trimmed.len - 1]);
    if (entries.len != 0) try output.writer.writeAll(",\n");
    try output.writer.writeAll(record_json);
    try output.writer.writeAll("\n]\n");
    try directory.writeFile(init.io, .{
        .sub_path = build_options.results_file,
        .data = output.written(),
    });
}

fn writeNewResultIndex(init: std.process.Init, record_json: []const u8) !void {
    const allocator = init.gpa;
    var output = std.Io.Writer.Allocating.init(allocator);
    defer output.deinit();
    try output.writer.writeAll("[\n");
    try output.writer.writeAll(record_json);
    try output.writer.writeAll("\n]\n");
    try std.Io.Dir.cwd().writeFile(init.io, .{
        .sub_path = build_options.results_file,
        .data = output.written(),
    });
}

fn regimeName(comptime regime: anytype) []const u8 {
    return if (regime) |value| @tagName(value) else "none";
}

fn calibrateIterations(
    comptime Benchmark: type,
    selected: *Benchmark,
    clock: std.Io.Clock,
    init: std.process.Init,
) usize {
    if (build_options.iterations != 0) return build_options.iterations;

    const target_ns: i96 = @intCast(build_options.sample_ms * std.time.ns_per_ms);
    var iterations: usize = Benchmark.default_iterations;
    for (0..4) |_| {
        const start = clock.now(init.io).nanoseconds;
        selected.run(iterations);
        const elapsed_ns = @max(clock.now(init.io).nanoseconds - start, 1);
        if (elapsed_ns >= target_ns) return iterations;

        const scale: usize = @intCast(@divTrunc(target_ns + elapsed_ns - 1, elapsed_ns));
        iterations = std.math.mul(usize, iterations, scale) catch
            @panic("benchmark calibration exceeded the iteration range");
    }
    return iterations;
}

const WarmupResult = struct {
    iterations: usize,
    elapsed_ns: i96,
};

fn warmUp(
    comptime Benchmark: type,
    selected: *Benchmark,
    clock: std.Io.Clock,
    init: std.process.Init,
) WarmupResult {
    const start = clock.now(init.io).nanoseconds;
    if (build_options.warmup_iterations != 0) {
        selected.run(build_options.warmup_iterations);
        return .{
            .iterations = build_options.warmup_iterations,
            .elapsed_ns = clock.now(init.io).nanoseconds - start,
        };
    }

    const target_ns: i96 = @intCast(build_options.warmup_ms * std.time.ns_per_ms);
    const chunk_iterations = Benchmark.default_warmup_iterations;
    var completed_iterations: usize = 0;
    var elapsed_ns: i96 = 0;
    while (elapsed_ns < target_ns) {
        selected.run(chunk_iterations);
        completed_iterations = std.math.add(usize, completed_iterations, chunk_iterations) catch
            @panic("benchmark warmup exceeded the iteration range");
        elapsed_ns = clock.now(init.io).nanoseconds - start;
    }
    return .{ .iterations = completed_iterations, .elapsed_ns = elapsed_ns };
}
