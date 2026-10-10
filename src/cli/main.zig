const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);

    var stderr_buffer: [4096]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(init.io, &stderr_buffer);
    const stderr = &stderr_writer.interface;
    defer stderr.flush() catch {};

    if (args.len < 2 or isHelp(args[1])) {
        try usage(stderr);
        return;
    }
    if (!std.mem.eql(u8, args[1], "compile")) {
        try stderr.print("unknown command: {s}\n\n", .{args[1]});
        try usage(stderr);
        std.process.exit(2);
    }
    compile(init, args[2..], stderr) catch |err| {
        try stderr.print("zgc compile failed: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

fn compile(init: std.process.Init, args: []const []const u8, stderr: *std.Io.Writer) !void {
    const allocator = init.arena.allocator();
    var input: ?[]const u8 = null;
    var output: ?[]const u8 = null;
    var optimize: []const u8 = "ReleaseFast";
    var target: ?[]const u8 = null;
    var zig: []const u8 = "zig";
    var source_root: []const u8 = build_options.source_root;

    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (isHelp(arg)) {
            try compileUsage(stderr);
            return;
        } else if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
            index += 1;
            if (index == args.len) return error.MissingOutputPath;
            output = args[index];
        } else if (std.mem.eql(u8, arg, "-O") or std.mem.eql(u8, arg, "--optimize")) {
            index += 1;
            if (index == args.len) return error.MissingOptimizeMode;
            optimize = args[index];
        } else if (std.mem.eql(u8, arg, "--target")) {
            index += 1;
            if (index == args.len) return error.MissingTarget;
            target = args[index];
        } else if (std.mem.eql(u8, arg, "--zig")) {
            index += 1;
            if (index == args.len) return error.MissingZigPath;
            zig = args[index];
        } else if (std.mem.eql(u8, arg, "--zgc-root")) {
            index += 1;
            if (index == args.len) return error.MissingSourceRoot;
            source_root = args[index];
        } else if (arg.len != 0 and arg[0] == '-') {
            try stderr.print("unknown compile option: {s}\n", .{arg});
            return error.InvalidOption;
        } else if (input == null) {
            input = arg;
        } else return error.MultipleInputs;
    }

    const input_path = input orelse return error.MissingInput;
    if (!std.mem.eql(u8, std.fs.path.extension(input_path), ".zgir")) return error.ExpectedZgir;
    if (!validOptimize(optimize)) return error.InvalidOptimizeMode;

    const cwd = try std.process.currentPathAlloc(init.io, allocator);
    const absolute_input = try std.fs.path.resolve(allocator, &.{ cwd, input_path });
    const absolute_root = try std.fs.path.resolve(allocator, &.{ cwd, source_root });
    const output_path = output orelse try defaultOutput(allocator, input_path);
    const absolute_output = try std.fs.path.resolve(allocator, &.{ cwd, output_path });
    const graph = try std.Io.Dir.readFileAlloc(.cwd(), init.io, absolute_input, allocator, .limited(1024 * 1024 * 1024));
    const adapter_path = try std.fs.path.join(allocator, &.{ absolute_root, "src", "artifact", "zgir_model.zig" });
    const adapter_source = try std.Io.Dir.readFileAlloc(.cwd(), init.io, adapter_path, allocator, .limited(1024 * 1024));
    const source_adapter_path = try std.fs.path.join(allocator, &.{ absolute_root, "src", "artifact", "zgir_source.zig" });
    const source_adapter = try std.Io.Dir.readFileAlloc(.cwd(), init.io, source_adapter_path, allocator, .limited(1024 * 1024));

    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(graph, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    const work_path = try std.fs.path.join(allocator, &.{ cwd, ".zig-cache", "zgc-cli", &hex });
    var work = try std.Io.Dir.createDirPathOpen(.cwd(), init.io, work_path, .{});
    defer work.close(init.io);
    try work.writeFile(init.io, .{ .sub_path = "graph.zgir", .data = graph });
    try work.writeFile(init.io, .{ .sub_path = "model.zig", .data = adapter_source });
    try work.writeFile(init.io, .{ .sub_path = "graph.zig", .data = source_adapter });

    const model_path = try std.fs.path.join(allocator, &.{ work_path, "model.zig" });
    const graph_path = try std.fs.path.join(allocator, &.{ work_path, "graph.zig" });
    const abi_path = try std.fs.path.join(allocator, &.{ absolute_root, "src", "artifact", "model_abi.zig" });
    const root_path = try std.fs.path.join(allocator, &.{ absolute_root, "src", "root.zig" });
    const optimize_arg = try std.fmt.allocPrint(allocator, "-O{s}", .{optimize});
    const root_arg = try std.fmt.allocPrint(allocator, "-Mroot={s}", .{abi_path});
    const model_arg = try std.fmt.allocPrint(allocator, "-Mmodel={s}", .{model_path});
    const zgc_arg = try std.fmt.allocPrint(allocator, "-Mzgc={s}", .{root_path});
    const graph_arg = try std.fmt.allocPrint(allocator, "-Mgraph={s}", .{graph_path});
    const output_arg = try std.fmt.allocPrint(allocator, "-femit-bin={s}", .{absolute_output});

    var command: [24][]const u8 = undefined;
    var count: usize = 0;
    for ([_][]const u8{ zig, "build-lib", "-dynamic", optimize_arg }) |arg| {
        command[count] = arg;
        count += 1;
    }
    if (target) |value| {
        command[count] = "-target";
        command[count + 1] = value;
        count += 2;
    }
    for ([_][]const u8{ "--dep", "model", "--dep", "zgc", root_arg, "--dep", "zgc", "--dep", "graph", model_arg, zgc_arg, graph_arg, output_arg }) |arg| {
        command[count] = arg;
        count += 1;
    }

    const result = try std.process.run(allocator, init.io, .{
        .argv = command[0..count],
        .environ_map = init.environ_map,
        .stdout_limit = .limited(16 * 1024 * 1024),
        .stderr_limit = .limited(16 * 1024 * 1024),
    });
    if (result.stdout.len != 0) try stderr.writeAll(result.stdout);
    if (result.stderr.len != 0) try stderr.writeAll(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) return error.CompilerFailed,
        else => return error.CompilerFailed,
    }

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    try stdout_writer.interface.print("{s}\n", .{absolute_output});
    try stdout_writer.interface.flush();
}

fn defaultOutput(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    const basename = std.fs.path.basename(input);
    const extension = std.fs.path.extension(basename);
    const stem = basename[0 .. basename.len - extension.len];
    const suffix = switch (builtin.os.tag) {
        .windows => ".dll",
        .macos, .ios, .tvos, .watchos, .visionos => ".dylib",
        else => ".so",
    };
    return std.fmt.allocPrint(allocator, "{s}{s}", .{ stem, suffix });
}

fn validOptimize(value: []const u8) bool {
    return std.mem.eql(u8, value, "Debug") or
        std.mem.eql(u8, value, "ReleaseSafe") or
        std.mem.eql(u8, value, "ReleaseFast") or
        std.mem.eql(u8, value, "ReleaseSmall");
}

fn isHelp(arg: []const u8) bool {
    return std.mem.eql(u8, arg, "help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help");
}

fn usage(writer: *std.Io.Writer) !void {
    try writer.writeAll(
        \\usage: zgc <command> [options]
        \\
        \\commands:
        \\  compile    specialize a .zgir graph and emit a native shared library
        \\
        \\run 'zgc compile --help' for compile options
        \\
    );
}

fn compileUsage(writer: *std.Io.Writer) !void {
    try writer.writeAll(
        \\usage: zgc compile <graph.zgir> [-o <library>] [options]
        \\
        \\options:
        \\  -o, --output <path>      output shared library path
        \\  -O, --optimize <mode>   Debug, ReleaseSafe, ReleaseFast, or ReleaseSmall
        \\  --target <triple>        Zig compilation target
        \\  --zig <path>             Zig executable (default: zig from PATH)
        \\  --zgc-root <path>        ZGC source checkout
        \\
    );
}
