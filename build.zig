const std = @import("std");
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // library
    const zgc_mod = b.addModule("zgc", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const cli_options = b.addOptions();
    cli_options.addOption([]const u8, "source_root", b.root.joinString(b.allocator, ".") catch @panic("OOM"));
    cli_options.addOption(u32, "abi_version", @import("src/artifact/abi.zig").version);
    const cli = b.addExecutable(.{
        .name = "zgc",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "build_options", .module = cli_options.createModule() }},
        }),
    });
    b.installArtifact(cli);
    _ = b.addModule("zgc_inspect_cli", .{
        .root_source_file = b.path("src/cli/inspect.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zgc", .module = zgc_mod }},
    });
    _ = b.addModule("zgc_executable_runner", .{
        .root_source_file = b.path("src/artifact/executable_runner.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zgc", .module = zgc_mod }},
    });
    _ = b.addModule("zgc_executable_abi", .{
        .root_source_file = b.path("src/artifact/executable_abi.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zgc", .module = zgc_mod }},
    });
    _ = b.addModule("zgc_inspectable_executable_abi", .{
        .root_source_file = b.path("src/artifact/inspectable_executable_abi.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zgc", .module = zgc_mod }},
    });
    _ = b.addModule("zgc_zgir_executable", .{
        .root_source_file = b.path("src/artifact/zgir_executable.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zgc", .module = zgc_mod }},
    });
    b.installFile("bindings/c/zgc_executable.h", "include/zgc_executable.h");

    // tests
    const test_embed_params = b.createModule(.{
        .root_source_file = b.path("tests/fixtures/embed_params.zig"),
        .target = target,
        .optimize = optimize,
    });
    const test_root_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const test_mod = b.addModule("zgc_tests", .{
        .root_source_file = b.path("tests/tests.zig"),
        .target = target,
        .optimize = optimize,
    });

    test_mod.addImport("zgc", test_root_mod);
    test_mod.addImport("embed_params", test_embed_params);

    const tests = b.addTest(.{
        .root_module = test_mod,
    });

    const runner_test_executable = b.createModule(.{
        .root_source_file = b.path("tests/fixtures/runner_model.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zgc", .module = zgc_mod }},
    });
    const runner_test_module = b.createModule(.{
        .root_source_file = b.path("src/artifact/executable_runner.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "executable", .module = runner_test_executable },
            .{ .name = "zgc", .module = zgc_mod },
        },
    });
    const runner_test_exe = b.addExecutable(.{
        .name = "zgc-executable-runner-test",
        .root_module = runner_test_module,
    });
    runner_test_exe.forceUndefinedSymbol(if (target.result.os.tag == .macos)
        "_zgc_runtime_run"
    else
        "zgc_runtime_run");

    const check_step = b.step("check", "Compile tests without running them");
    check_step.dependOn(&tests.step);
    check_step.dependOn(&runner_test_exe.step);

    const run_mod_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&runner_test_exe.step);

    const python = b.option(
        []const u8,
        "python",
        "Python interpreter for bindings tests",
    ) orelse "python3";
    
    const bindings_smoke = b.addSystemCommand(
        &.{ python, "bindings/tests/smoke.py" },
    );
    bindings_smoke.addArtifactArg(cli);
    const bindings_test_step = b.step("test-bindings", "Compile and exercise the C ABI and Python runtime");
    bindings_test_step.dependOn(&bindings_smoke.step);

    // benchmarks
    const benchmark_name = b.option([]const u8, "benchmark", "Specialized graph benchmark to run") orelse "dense";
    const benchmark_iterations = b.option(usize, "iterations", "Benchmark invocations per timed sample (0 calibrates to sample_ms)") orelse 0;
    const benchmark_runs = b.option(usize, "runs", "Number of timed samples") orelse 30;
    const benchmark_warmup = b.option(usize, "warmup_iterations", "Untimed warmup invocations (0 warms for warmup_ms)") orelse 0;
    const benchmark_sample_ms = b.option(usize, "sample_ms", "Minimum duration of each automatically calibrated timed sample") orelse 250;
    const benchmark_warmup_ms = b.option(usize, "warmup_ms", "Duration of the automatic untimed warmup") orelse 2_000;
    const benchmark_target = b.option([]const u8, "optimization_target", "Executable selection: balanced, memory, fast, or small") orelse "balanced";
    const benchmark_frontier_index = b.option(usize, "frontier_index", "Force this zero-based retained executable index") orelse std.math.maxInt(usize);
    const benchmark_show_frontier = b.option(bool, "show_frontier", "Print costs for every retained executable") orelse false;
    const benchmark_results_file = b.option([]const u8, "results_file", "Append benchmark records to a JSON index file") orelse "";

    const benchmark_options = b.addOptions();
    benchmark_options.addOption(usize, "iterations", benchmark_iterations);
    benchmark_options.addOption(usize, "runs", benchmark_runs);
    benchmark_options.addOption(usize, "warmup_iterations", benchmark_warmup);
    benchmark_options.addOption(usize, "sample_ms", benchmark_sample_ms);
    benchmark_options.addOption(usize, "warmup_ms", benchmark_warmup_ms);
    benchmark_options.addOption([]const u8, "optimization_target", benchmark_target);
    benchmark_options.addOption(usize, "frontier_index", benchmark_frontier_index);
    benchmark_options.addOption(bool, "show_frontier", benchmark_show_frontier);
    benchmark_options.addOption([]const u8, "results_file", benchmark_results_file);
    const benchmark_options_module = benchmark_options.createModule();
    const benchmark_selection = b.createModule(.{
        .root_source_file = b.path("benchmarks/selection.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zgc", .module = zgc_mod },
            .{ .name = "build_options", .module = benchmark_options_module },
        },
    });

    const benchmark_source = if (std.mem.eql(u8, benchmark_name, "contraction"))
        "benchmarks/workloads/contraction.zig"
    else if (std.mem.eql(u8, benchmark_name, "reduction"))
        "benchmarks/workloads/reduction.zig"
    else if (std.mem.eql(u8, benchmark_name, "fusion"))
        "benchmarks/workloads/fusion.zig"
    else if (std.mem.eql(u8, benchmark_name, "dense"))
        "benchmarks/workloads/dense.zig"
    else if (std.mem.eql(u8, benchmark_name, "lbm"))
        "benchmarks/workloads/lbm.zig"
    else if (std.mem.eql(u8, benchmark_name, "conway"))
        "benchmarks/workloads/conway.zig"
    else
        @panic("unknown benchmark; expected contraction, reduction, fusion, dense, lbm, or conway");
    const benchmark_case = b.createModule(.{
        .root_source_file = b.path(benchmark_source),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zgc", .module = zgc_mod },
            .{ .name = "benchmark_selection", .module = benchmark_selection },
        },
    });

    const benchmark_exe = b.addExecutable(.{
        .name = "zgc-benchmark",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zgc", .module = zgc_mod },
                .{ .name = "build_options", .module = benchmark_options_module },
                .{ .name = "benchmark_case", .module = benchmark_case },
            },
        }),
    });
    const run_benchmark = b.addRunArtifact(benchmark_exe);
    const benchmark_step = b.step("benchmark", "Run the graph selected by -Dbenchmark");
    benchmark_step.dependOn(&run_benchmark.step);
}
