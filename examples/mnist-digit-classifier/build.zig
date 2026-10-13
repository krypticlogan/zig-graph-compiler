const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zgc_dep = b.dependency("zgc", .{
        .target = target,
        .optimize = optimize,
    });
    const zgc_mod = zgc_dep.module("zgc");
    const model_params_mod = b.createModule(.{
        .root_source_file = b.path("model_params/params.zig"),
        .target = target,
        .optimize = optimize,
    });
    const classifier_model_mod = b.createModule(.{
        .root_source_file = b.path("src/digit-classifier.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zgc", .module = zgc_mod },
            .{ .name = "model_params", .module = model_params_mod },
        },
    });
    const raylib_dep = b.dependency("raylib_zig", .{
        .target = target,
        .optimize = optimize,
    });
    if (b.graph.needed_lazy_dependencies.count() != 0)
        return error.LazyDependencyNeeded;
    const raylib_mod = raylib_dep.module("raylib");
    const raylib_artifact = raylib_dep.artifact("raylib");

    const inspect_cli_mod = zgc_dep.module("zgc_inspect_cli");
    inspect_cli_mod.addImport("executable", classifier_model_mod);
    const model_inspector = b.addExecutable(.{
        .name = "zgc-inspect",
        .root_module = inspect_cli_mod,
    });
    b.installArtifact(model_inspector);

    const inspect_step = b.step("inspect", "Inspect the digit-classifier model");
    const inspect_cmd = b.addRunArtifact(model_inspector);
    inspect_step.dependOn(&inspect_cmd.step);
    inspect_cmd.addPassthruArgs();

    const demo_exe = b.addExecutable(.{
        .name = "demo",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/demo.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zgc", .module = zgc_mod },
                .{ .name = "model_params", .module = model_params_mod },
                .{ .name = "raylib", .module = raylib_mod },
            },
            .link_libc = true,
        }),
    });
    demo_exe.root_module.linkLibrary(raylib_artifact);
    b.installArtifact(demo_exe);

    const demo_step = b.step("demo", "Run the interactive digit-classification demo");
    const demo_cmd = b.addRunArtifact(demo_exe);
    demo_step.dependOn(&demo_cmd.step);

    const executable_runner_mod = zgc_dep.module("zgc_executable_runner");
    executable_runner_mod.addImport("executable", classifier_model_mod);
    const executable_binary = b.addExecutable(.{
        .name = "zgc-executable",
        .root_module = executable_runner_mod,
    });
    executable_binary.forceUndefinedSymbol(if (target.result.os.tag == .macos)
        "_zgc_runtime_run"
    else
        "zgc_runtime_run");

    const install_executable = b.addInstallArtifact(executable_binary, .{});
    const build_executable_step = b.step("build-executable", "Build the lean generated executable");
    build_executable_step.dependOn(&install_executable.step);

    const executable_abi_mod = zgc_dep.module("zgc_executable_abi");
    executable_abi_mod.addImport("executable", classifier_model_mod);
    const executable_library = b.addLibrary(.{
        .name = "zgc-executable",
        .linkage = .dynamic,
        .root_module = executable_abi_mod,
    });
    executable_library.forceUndefinedSymbol(if (target.result.os.tag == .macos)
        "_zgc_runtime_run"
    else
        "zgc_runtime_run");
    const install_executable_library = b.addInstallArtifact(executable_library, .{});
    const build_executable_library_step = b.step("build-executable-library", "Build the generated executable C ABI library");
    build_executable_library_step.dependOn(&install_executable_library.step);

    const disassembler = if (target.result.os.tag == .macos)
        b.addSystemCommand(&.{ "xcrun", "llvm-objdump", "--disassemble-symbols=_zgc_runtime_run" })
    else
        b.addSystemCommand(&.{ "objdump", "--disassemble=zgc_runtime_run" });
    disassembler.addArtifactArg(executable_binary);
    const disassemble_executable_step = b.step("disassemble-executable", "Disassemble the generated runtime execution symbol");
    disassemble_executable_step.dependOn(&disassembler.step);
}
