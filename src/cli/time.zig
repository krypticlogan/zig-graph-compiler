const std = @import("std");
const Abi = @import("zgc").artifact.ABI;

pub fn run(init: std.process.Init, input_path: []const u8, options: Options, expected_abi: u32, json_path: ?[]const u8) !void {
    const allocator = init.arena.allocator();
    const cwd = try std.process.currentPathAlloc(init.io, allocator);
    const path = try std.fs.path.resolve(allocator, &.{ cwd, input_path });
    var library = try std.DynLib.open(path);
    defer library.close();

    const abi_version = library.lookup(*const fn () callconv(.c) u32, "zgc_abi_version") orelse return error.NotZgcExecutable;
    if (abi_version() != expected_abi) return error.IncompatibleAbiVersion;
    const runtime_size = library.lookup(*const fn () callconv(.c) usize, "zgc_runtime_size") orelse return error.NotZgcExecutable;
    const runtime_alignment = library.lookup(*const fn () callconv(.c) usize, "zgc_runtime_alignment") orelse return error.NotZgcExecutable;
    const runtime_init = library.lookup(*const fn (?*anyopaque, usize) callconv(.c) Abi.Status, "zgc_runtime_init") orelse return error.NotZgcExecutable;
    const runtime_deinit = library.lookup(*const fn (?*anyopaque) callconv(.c) void, "zgc_runtime_deinit") orelse return error.NotZgcExecutable;
    const runtime_run = library.lookup(*const fn (?*anyopaque) callconv(.c) Abi.Status, "zgc_runtime_run") orelse return error.NotZgcExecutable;
    const source_count = library.lookup(*const fn () callconv(.c) usize, "zgc_executable_source_count") orelse return error.NotZgcExecutable;
    const source_descriptor = library.lookup(*const fn (usize, ?*Abi.SourceDescriptor) callconv(.c) Abi.Status, "zgc_executable_get_source_descriptor") orelse return error.NotZgcExecutable;
    const copy_source = library.lookup(*const fn (?*anyopaque, u32, ?*const anyopaque, usize) callconv(.c) Abi.Status, "zgc_runtime_copy_source") orelse return error.NotZgcExecutable;
    const bind_source = library.lookup(*const fn (?*anyopaque, u32, ?*const anyopaque, usize) callconv(.c) Abi.Status, "zgc_runtime_bind_source") orelse return error.NotZgcExecutable;

    const alignment = runtime_alignment();
    if (alignment == 0 or !std.math.isPowerOfTwo(alignment)) return error.InvalidRuntimeAlignment;
    const size = runtime_size();
    const storage = try allocator.alloc(u8, try std.math.add(usize, size, alignment - 1));
    const runtime: *anyopaque = @ptrFromInt(std.mem.alignForward(usize, @intFromPtr(storage.ptr), alignment));
    if (runtime_init(runtime, size) != .ok) return error.RuntimeInitializationFailed;
    defer runtime_deinit(runtime);

    for (0..source_count()) |source_index| {
        var descriptor: Abi.SourceDescriptor = undefined;
        if (source_descriptor(source_index, &descriptor) != .ok) return error.SourceDescriptionFailed;
        if (descriptor.binding == .embedded) continue;
        const byte_count = descriptor.tensor.logical_byte_count;
        const source_storage = try allocator.alloc(u8, try std.math.add(usize, byte_count, 15));
        const source_address = std.mem.alignForward(usize, @intFromPtr(source_storage.ptr), 16);
        const source_bytes: [*]u8 = @ptrFromInt(source_address);
        @memset(source_bytes[0..byte_count], 0);
        const source_pointer: *const anyopaque = @ptrFromInt(source_address);
        const status = switch (descriptor.binding) {
            .owned => copy_source(runtime, descriptor.key, source_pointer, byte_count),
            .bound => bind_source(runtime, descriptor.key, source_pointer, byte_count),
            .embedded => unreachable,
        };
        if (status != .ok) return error.SourceInitializationFailed;
    }
    if (runtime_run(runtime) != .ok) return error.RuntimeExecutionFailed;

    const Runner = struct {
        runtime_run: *const fn (?*anyopaque) callconv(.c) Abi.Status,
        runtime: *anyopaque,

        pub fn run(self: *@This(), iterations: usize) !void {
            for (0..iterations) |_| {
                if (self.runtime_run(self.runtime) != .ok) return error.RuntimeExecutionFailed;
            }
        }
    };
    var runner: Runner = .{ .runtime_run = runtime_run, .runtime = runtime };
    const result = try measure(init, options, &runner);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.print(
        \\Executable: {s}
        \\  warmup       {d} iterations / {d:.3} s
        \\  timing       {d} samples x {d} iterations / {d:.3} s
        \\  latency (ns) min {d:.3} | avg {d:.3} | median {d:.3} | p95 {d:.3} | max {d:.3}
        \\  variability  stddev {d:.3} ns | CV {d:.2}%
        \\  throughput   {d:.3} invocations/s
        \\
    , .{
        path,
        result.warmup_iterations,
        @as(f64, @floatFromInt(result.warmup_elapsed_ns)) / std.time.ns_per_s,
        options.runs,
        result.iterations,
        result.timed_elapsed_ns / std.time.ns_per_s,
        result.minimum_ns,
        result.average_ns,
        result.median_ns,
        result.percentile_95_ns,
        result.maximum_ns,
        result.standard_deviation_ns,
        result.coefficient_of_variation_percent,
        if (result.average_ns == 0) 0 else @as(f64, std.time.ns_per_s) / result.average_ns,
    });
    try stdout.flush();
    if (json_path) |destination| {
        try appendJson(init, destination, .{
            .schema_version = 2,
            .input = input_path,
            .executable = path,
            .configuration = .{
                .runs = options.runs,
                .iterations_per_sample = result.iterations,
                .warmup_iterations = result.warmup_iterations,
                .warmup_ms_target = if (options.warmup_iterations == 0) options.warmup_ms else 0,
                .sample_ms_target = if (options.iterations == 0) options.sample_ms else 0,
            },
            .timing = .{
                .minimum_ns = result.minimum_ns,
                .average_ns = result.average_ns,
                .median_ns = result.median_ns,
                .percentile_95_ns = result.percentile_95_ns,
                .maximum_ns = result.maximum_ns,
                .standard_deviation_ns = result.standard_deviation_ns,
                .coefficient_of_variation_percent = result.coefficient_of_variation_percent,
                .samples_ns = result.samples_ns,
            },
        });
    }
}

fn appendJson(init: std.process.Init, path: []const u8, record: anytype) !void {
    const allocator = init.arena.allocator();
    const serialized = try std.json.Stringify.valueAlloc(allocator, record, .{});
    const directory = std.Io.Dir.cwd();
    const existing = directory.readFileAlloc(init.io, path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    var output = std.Io.Writer.Allocating.init(allocator);
    defer output.deinit();
    if (existing) |contents| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, contents, .{});
        if (parsed.value != .array) return error.InvalidBenchmarkResultIndex;
        const trimmed = std.mem.trimEnd(u8, contents, " \t\r\n");
        const entries = std.mem.trim(u8, trimmed[1 .. trimmed.len - 1], " \t\r\n");
        try output.writer.writeAll(trimmed[0 .. trimmed.len - 1]);
        if (entries.len != 0) try output.writer.writeAll(",");
        try output.writer.writeAll("\n");
    } else {
        try output.writer.writeAll("[\n");
    }
    try output.writer.writeAll(serialized);
    try output.writer.writeAll("\n]\n");
    try directory.writeFile(init.io, .{ .sub_path = path, .data = output.written() });
}

pub const Options = struct {
    runs: usize = 30,
    iterations: usize = 0,
    initial_iterations: usize = 1,
    sample_ms: usize = 250,
    warmup_iterations: usize = 0,
    warmup_ms: usize = 2_000,
    warmup_chunk_iterations: usize = 1,
};

pub const Result = struct {
    minimum_ns: f64,
    average_ns: f64,
    maximum_ns: f64,
    median_ns: f64,
    percentile_95_ns: f64,
    standard_deviation_ns: f64,
    coefficient_of_variation_percent: f64,
    iterations: usize,
    warmup_iterations: usize,
    warmup_elapsed_ns: i96,
    timed_elapsed_ns: f64,
    samples_ns: []f64,
};

/// Measure an already-initialized runner. Its run method executes the requested
/// number of invocations and returns an error if execution fails.
pub fn measure(init: std.process.Init, options: Options, runner: anytype) !Result {
    if (options.runs == 0 or options.initial_iterations == 0 or options.warmup_chunk_iterations == 0 or
        (options.iterations == 0 and options.sample_ms == 0)) return error.InvalidTimingConfiguration;

    const clock = std.Io.Clock.awake;
    const warmup_start = clock.now(init.io).nanoseconds;
    var warmup_count: usize = 0;
    if (options.warmup_iterations != 0) {
        try runner.run(options.warmup_iterations);
        warmup_count = options.warmup_iterations;
    } else if (options.warmup_ms != 0) {
        const target_ns: i96 = @intCast(try std.math.mul(usize, options.warmup_ms, std.time.ns_per_ms));
        while (clock.now(init.io).nanoseconds - warmup_start < target_ns) {
            try runner.run(options.warmup_chunk_iterations);
            warmup_count = try std.math.add(usize, warmup_count, options.warmup_chunk_iterations);
        }
    }
    const warmup_elapsed = clock.now(init.io).nanoseconds - warmup_start;

    var iterations = options.iterations;
    if (iterations == 0) {
        iterations = options.initial_iterations;
        const target_ns: i96 = @intCast(try std.math.mul(usize, options.sample_ms, std.time.ns_per_ms));
        for (0..4) |_| {
            const start = clock.now(init.io).nanoseconds;
            try runner.run(iterations);
            const elapsed = @max(clock.now(init.io).nanoseconds - start, 1);
            if (elapsed >= target_ns) break;
            const scale: usize = @intCast(@divTrunc(target_ns + elapsed - 1, elapsed));
            iterations = try std.math.mul(usize, iterations, scale);
        }
    }

    const samples = try init.arena.allocator().alloc(f64, options.runs);
    var timed_elapsed_ns: f64 = 0;
    for (samples) |*sample| {
        const start = clock.now(init.io).nanoseconds;
        try runner.run(iterations);
        const elapsed: f64 = @floatFromInt(clock.now(init.io).nanoseconds - start);
        timed_elapsed_ns += elapsed;
        sample.* = elapsed / @as(f64, @floatFromInt(iterations));
    }

    var minimum = std.math.inf(f64);
    var maximum: f64 = 0;
    var total: f64 = 0;
    for (samples) |sample| {
        minimum = @min(minimum, sample);
        maximum = @max(maximum, sample);
        total += sample;
    }
    const average = total / @as(f64, @floatFromInt(options.runs));
    var squared_deviation: f64 = 0;
    for (samples) |sample| {
        const difference = sample - average;
        squared_deviation += difference * difference;
    }
    const standard_deviation = @sqrt(squared_deviation / @as(f64, @floatFromInt(options.runs)));
    const sorted_samples = try init.arena.allocator().dupe(f64, samples);
    std.mem.sort(f64, sorted_samples, {}, std.sort.asc(f64));
    const median = if (options.runs % 2 == 0)
        (sorted_samples[options.runs / 2 - 1] + sorted_samples[options.runs / 2]) / 2
    else
        sorted_samples[options.runs / 2];
    const p95 = sorted_samples[@min((95 * options.runs + 99) / 100 - 1, options.runs - 1)];

    return .{
        .minimum_ns = minimum,
        .average_ns = average,
        .maximum_ns = maximum,
        .median_ns = median,
        .percentile_95_ns = p95,
        .standard_deviation_ns = standard_deviation,
        .coefficient_of_variation_percent = if (average == 0) 0 else standard_deviation / average * 100,
        .iterations = iterations,
        .warmup_iterations = warmup_count,
        .warmup_elapsed_ns = warmup_elapsed,
        .timed_elapsed_ns = timed_elapsed_ns,
        .samples_ns = samples,
    };
}
