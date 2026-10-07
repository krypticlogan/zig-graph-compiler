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
    const zgc_ns = try runBenchmark(Workload.ZgcBenchmark, init);
    const direct_ns = try runBenchmark(Workload.DirectBenchmark, init);
    std.debug.print(
        \\Comparison
        \\  average latency  ZGC {d:.3} ns | direct {d:.3} ns
        \\  direct / ZGC    {d:.3}x (below 1.0 means direct is faster)
        \\
    ,
        .{ zgc_ns, direct_ns, direct_ns / zgc_ns },
    );
}

fn runBenchmark(comptime Selected: type, init: std.process.Init) !f64 {
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
        const choice = Selected.CompiledModel.selected_executable_candidate;
        std.debug.print(
            "  executable     {s} | schedule {s} | fusion {s} | layout {s} | remap {s} | {d} nodes\n",
            .{
                @tagName(choice.origin),
                @tagName(choice.schedule),
                regimeName(choice.fusion_regime),
                regimeName(choice.layout_regime),
                regimeName(choice.remap_regime),
                Selected.CompiledModel.executable.node_ct,
            },
        );
    }
    std.debug.print("\n", .{});
    return average_ns;
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
