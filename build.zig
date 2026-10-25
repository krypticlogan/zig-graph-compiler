const std = @import("std");
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Shared private renderer used by the inspection entry points and ABI.
    const inspect_mod = b.createModule(.{
        .root_source_file = b.path("src/zgc/core/inspect.zig"),
        .target = target,
        .optimize = optimize,
    });

    // library
    const zgc_mod = b.addModule("zgc", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    zgc_mod.addImport("zgc_inspect_impl", inspect_mod);
    const translate_abi = b.addTranslateC(.{
        .root_source_file = b.path("bindings/c/zgc_executable.h"),
        .target = target,
        .optimize = optimize,
        .link_libc = false,
    });
    translate_abi.addIncludePath(b.path("bindings/c"));
    const runtime_mod = b.addModule("zgc_runtime", .{
        .root_source_file = b.path("bindings/zig/runtime.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zgc_abi_c", .module = translate_abi.createModule() }},
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
            .link_libc = target.result.os.tag == .linux,
            .imports = &.{
                .{ .name = "build_options", .module = cli_options.createModule() },
                .{ .name = "zgc", .module = zgc_mod },
            },
        }),
    });
    b.installArtifact(cli);
    _ = b.addModule("zgc_inspect_cli", .{
        .root_source_file = b.path("src/cli/inspect.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zgc_inspect_impl", .module = inspect_mod }},
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
    // Private module: tests and benchmarks share the public API
    // and its implementation types without exporting internals to consumers.
    const internal_mod = b.createModule(.{
        .root_source_file = b.path("src/internal.zig"),
        .target = target,
        .optimize = optimize,
    });

    internal_mod.addImport("zgc_inspect_impl", inspect_mod);

    const test_mod = b.addModule("zgc_tests", .{
        .root_source_file = b.path("tests/tests.zig"),
        .target = target,
        .optimize = optimize,
    });

    test_mod.addImport("zgc_internal", internal_mod);
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

    const check_zgc = b.step("check-zgc", "Compile the public ZGC module");
    const zgc_object = b.addObject(.{ .name = "zgc-check", .root_module = zgc_mod });
    check_zgc.dependOn(&zgc_object.step);

    const check_runtime = b.step("check-runtime", "Compile the Zig artifact runtime module");
    const runtime_object = b.addObject(.{ .name = "zgc-runtime-check", .root_module = runtime_mod });
    check_runtime.dependOn(&runtime_object.step);

    const docs = b.step("docs", "Generate the ZGC Zig API documentation in zig-out/docs/api");
    const docs_object = b.addObject(.{ .name = "zgc-docs", .root_module = zgc_mod });
    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs_object.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "web/docs",
    });
    docs.dependOn(&install_docs.step);

    const check_inspect_cli = b.step("check-inspect-cli", "Compile the inspection CLI module");
    const inspect_check_mod = b.createModule(.{
        .root_source_file = b.path("src/cli/inspect.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zgc_inspect_impl", .module = inspect_mod },
            .{ .name = "executable", .module = runner_test_executable },
        },
    });
    const inspect_check_exe = b.addExecutable(.{ .name = "zgc-inspect-check", .root_module = inspect_check_mod });
    check_inspect_cli.dependOn(&inspect_check_exe.step);

    const check_runner = b.step("check-executable-runner", "Compile the executable runner module");
    check_runner.dependOn(&runner_test_exe.step);

    const check_abi = b.step("check-executable-abi", "Compile the executable ABI module");
    const abi_check_mod = b.createModule(.{
        .root_source_file = b.path("src/artifact/executable_abi.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zgc", .module = zgc_mod },
            .{ .name = "executable", .module = runner_test_executable },
        },
    });
    const abi_check_object = b.addObject(.{ .name = "zgc-abi-check", .root_module = abi_check_mod });
    check_abi.dependOn(&abi_check_object.step);

    const check_inspectable_abi = b.step("check-inspectable-executable-abi", "Compile the inspectable ABI module");
    const inspectable_abi_check_mod = b.createModule(.{
        .root_source_file = b.path("src/artifact/inspectable_executable_abi.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zgc", .module = zgc_mod },
            .{ .name = "executable", .module = runner_test_executable },
        },
    });
    const inspectable_abi_check_object = b.addObject(.{ .name = "zgc-inspectable-abi-check", .root_module = inspectable_abi_check_mod });
    check_inspectable_abi.dependOn(&inspectable_abi_check_object.step);

    const check_zgir_executable = b.step("check-zgir-executable", "Compile the serialized-graph executable module");
    const check_graph = b.addOptions();
    check_graph.addOption([]const u8, "source", @embedFile("bindings/tests/fixtures/smoke.zgir"));
    const zgir_check_mod = b.createModule(.{
        .root_source_file = b.path("src/artifact/zgir_executable.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zgc", .module = zgc_mod },
            .{ .name = "graph", .module = check_graph.createModule() },
        },
    });
    const zgir_check_object = b.addObject(.{ .name = "zgc-zgir-check", .root_module = zgir_check_mod });
    check_zgir_executable.dependOn(&zgir_check_object.step);

    const check_cli_step = b.step("check-cli", "Compile the CLI without installing it");
    check_cli_step.dependOn(&cli.step);

    const check_tests_step = b.step("check-tests", "Compile tests without running them");
    check_tests_step.dependOn(&tests.step);
    check_tests_step.dependOn(&runner_test_exe.step);

    const check_step = b.step("check", "Compile library modules, CLI, and tests without running them");
    check_step.dependOn(check_cli_step);
    check_step.dependOn(check_tests_step);
    check_step.dependOn(check_zgc);
    check_step.dependOn(check_runtime);
    check_step.dependOn(check_inspect_cli);
    check_step.dependOn(check_runner);
    check_step.dependOn(check_abi);
    check_step.dependOn(check_inspectable_abi);
    check_step.dependOn(check_zgir_executable);

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
    const benchmark_results_file = b.option([]const u8, "out", "Append benchmark records to a JSON index file") orelse "";

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
            .{ .name = "zgc_internal", .module = internal_mod },
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
            .{ .name = "zgc_internal", .module = internal_mod },
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
                .{ .name = "zgc_internal", .module = internal_mod },
                .{ .name = "build_options", .module = benchmark_options_module },
                .{ .name = "benchmark_case", .module = benchmark_case },
            },
        }),
    });
    const run_benchmark = b.addRunArtifact(benchmark_exe);
    const benchmark_step = b.step("benchmark", "Run the graph selected by -Dbenchmark");
    benchmark_step.dependOn(&run_benchmark.step);

    const check_workloads = b.step("check-workloads", "Compile every benchmark workload without running it");
    const benchmark_suite = b.step("benchmark-suite", "Build and run all benchmark workloads sequentially");
    var previous_workload_run: ?*std.Build.Step = null;
    inline for (.{ "contraction", "reduction", "fusion", "dense", "lbm", "conway" }) |name| {
        const workload = b.createModule(.{
            .root_source_file = b.path("benchmarks/workloads/" ++ name ++ ".zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zgc_internal", .module = internal_mod },
                .{ .name = "benchmark_selection", .module = benchmark_selection },
            },
        });
        const workload_exe = b.addExecutable(.{
            .name = "zgc-benchmark-check-" ++ name,
            .root_module = b.createModule(.{
                .root_source_file = b.path("benchmarks/main.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "zgc_internal", .module = internal_mod },
                    .{ .name = "build_options", .module = benchmark_options_module },
                    .{ .name = "benchmark_case", .module = workload },
                },
            }),
        });
        check_workloads.dependOn(&workload_exe.step);

        const run_workload = b.addRunArtifact(workload_exe);
        if (previous_workload_run) |previous| run_workload.step.dependOn(previous);
        previous_workload_run = &run_workload.step;
    }
    benchmark_suite.dependOn(previous_workload_run.?);
    check_step.dependOn(check_workloads);
}
