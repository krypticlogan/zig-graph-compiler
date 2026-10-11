const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zgc_dep = b.dependency("zgc", .{
        .target = target,
        .optimize = optimize,
    });
    const zgc_mod = zgc_dep.module("zgc");
    const raylib_dep = b.dependency("raylib_zig", .{
        .target = target,
        .optimize = optimize,
    });
    if (b.graph.needed_lazy_dependencies.count() != 0)
        return error.LazyDependencyNeeded;

    const raylib_mod = raylib_dep.module("raylib");
    const raylib_artifact = raylib_dep.artifact("raylib");

    const fluid_model_mod = b.createModule(.{
        .root_source_file = b.path("src/lbm-model.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zgc", .module = zgc_mod },
        },
    });

    const cli_module = zgc_dep.module("zgc_inspect_cli");
    cli_module.addImport("model", fluid_model_mod);
    const inspector = b.addExecutable(.{
        .name = "zgc-inspect",
        .root_module = cli_module,
    });
    b.installArtifact(inspector);

    const exe = b.addExecutable(.{
        .name = "fluid_dynamics",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "fluid_model", .module = fluid_model_mod },
                .{ .name = "zgc", .module = zgc_mod },
                .{ .name = "raylib", .module = raylib_mod },
            },
            .link_libc = true,
        }),
    });
    exe.root_module.linkLibrary(raylib_artifact);

    b.installArtifact(exe);
    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    run_cmd.addPassthruArgs();
}
