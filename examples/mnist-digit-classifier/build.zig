const std = @import("std");

pub fn build(b: *std.Build) void {
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
    const raylib_mod = raylib_dep.module("raylib");
    const raylib_artifact = raylib_dep.artifact("raylib");

    const inspect_cli_mod = zgc_dep.module("zgc_inspect_cli");
    inspect_cli_mod.addImport("model", classifier_model_mod);
    const model_inspector = b.addExecutable(.{
        .name = "zgc-inspect",
        .root_module = inspect_cli_mod,
    });
    b.installArtifact(model_inspector);

    const inspect_step = b.step("inspect", "Inspect the digit-classifier model");
    const inspect_cmd = b.addRunArtifact(model_inspector);
    inspect_step.dependOn(&inspect_cmd.step);
    if (b.args) |args| inspect_cmd.addArgs(args);

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

    const model_runner_mod = zgc_dep.module("zgc_model_runner");
    model_runner_mod.addImport("model", classifier_model_mod);
    const model_exe = b.addExecutable(.{
        .name = "zgc-model",
        .root_module = model_runner_mod,
    });
    model_exe.forceUndefinedSymbol(if (target.result.os.tag == .macos)
        "_zgc_model_run"
    else
        "zgc_model_run");

    const install_model = b.addInstallArtifact(model_exe, .{});
    const build_model_step = b.step("build-model", "Build the lean generated-model binary");
    build_model_step.dependOn(&install_model.step);

    const model_abi_mod = zgc_dep.module("zgc_model_abi");
    model_abi_mod.addImport("model", classifier_model_mod);
    const model_library = b.addLibrary(.{
        .name = "zgc-model",
        .linkage = .dynamic,
        .root_module = model_abi_mod,
    });
    model_library.forceUndefinedSymbol(if (target.result.os.tag == .macos)
        "_zgc_model_run"
    else
        "zgc_model_run");
    const install_model_library = b.addInstallArtifact(model_library, .{});
    const build_model_library_step = b.step("build-model-library", "Build the generated-model C ABI library");
    build_model_library_step.dependOn(&install_model_library.step);

    const disassembler = if (target.result.os.tag == .macos)
        b.addSystemCommand(&.{ "xcrun", "llvm-objdump", "--disassemble-symbols=_zgc_model_run" })
    else
        b.addSystemCommand(&.{ "objdump", "--disassemble=zgc_model_run" });
    disassembler.addArtifactArg(model_exe);
    const disassemble_model_step = b.step("disassemble-model", "Disassemble the generated model execution symbol");
    disassemble_model_step.dependOn(&disassembler.step);
}
