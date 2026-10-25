const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const timing = @import("time.zig");
const Library = @import("library.zig").Library;

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
    if (std.mem.eql(u8, args[1], "inspect")) {
        inspect(init, args[2..], stderr) catch |err| {
            try stderr.print("zgc inspect failed: {s}\n", .{@errorName(err)});
            try stderr.flush();
            std.process.exit(1);
        };
        return;
    }
    if (std.mem.eql(u8, args[1], "cache")) {
        cache(init, args[2..], stderr) catch |err| {
            try stderr.print("zgc cache failed: {s}\n", .{@errorName(err)});
            try stderr.flush();
            std.process.exit(1);
        };
        return;
    }
    if (std.mem.eql(u8, args[1], "compile")) {
        compile(init, args[2..], stderr) catch |err| {
            try stderr.print("zgc compile failed: {s}\n", .{@errorName(err)});
            try stderr.flush();
            std.process.exit(1);
        };
        return;
    }
    if (std.mem.eql(u8, args[1], "time")) {
        timeCommand(init, args[2..], stderr) catch |err| {
            try stderr.print("zgc time failed: {s}\n", .{@errorName(err)});
            try stderr.flush();
            std.process.exit(1);
        };
        return;
    }
    try stderr.print("unknown command: {s}\n\n", .{args[1]});
    try usage(stderr);
    try stderr.flush();
    std.process.exit(2);
}

fn cachePath(init: std.process.Init, configured: ?[]const u8) ![]const u8 {
    const allocator = init.arena.allocator();
    const cwd = try std.process.currentPathAlloc(init.io, allocator);
    const path = configured orelse init.environ_map.get("ZGC_CACHE_DIR") orelse ".zgc-cache";
    return std.fs.path.resolve(allocator, &.{ cwd, path });
}

fn cache(init: std.process.Init, args: []const []const u8, stderr: *std.Io.Writer) !void {
    if (args.len == 1 and isHelp(args[0])) {
        try cacheUsage(stderr);
        return;
    }
    if (args.len == 0) {
        try cacheUsage(stderr);
        return error.MissingCacheAction;
    }
    var configured: ?[]const u8 = null;
    var index: usize = 1;
    while (index < args.len) : (index += 1) {
        if (!std.mem.eql(u8, args[index], "--cache-dir")) return error.InvalidCacheOption;
        index += 1;
        if (index == args.len) return error.MissingCachePath;
        configured = args[index];
    }
    const path = try cachePath(init, configured);
    if (std.mem.eql(u8, args[0], "path")) {
        var stdout_buffer: [4096]u8 = undefined;
        var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
        try stdout_writer.interface.print("{s}\n", .{path});
        try stdout_writer.interface.flush();
    } else if (std.mem.eql(u8, args[0], "clear")) {
        const cwd = try std.process.currentPathAlloc(init.io, init.arena.allocator());
        if (std.mem.eql(u8, path, cwd) or std.mem.eql(u8, path, std.fs.path.sep_str)) return error.UnsafeCachePath;
        if (init.environ_map.get("HOME")) |home| {
            if (std.mem.eql(u8, path, home)) return error.UnsafeCachePath;
        }
        try std.Io.Dir.deleteTree(.cwd(), init.io, path);
    } else {
        try cacheUsage(stderr);
        return error.InvalidCacheAction;
    }
}

fn inspect(init: std.process.Init, args: []const []const u8, stderr: *std.Io.Writer) !void {
    if (args.len == 1 and isHelp(args[0])) {
        try inspectUsage(stderr);
        return;
    }
    if (args.len == 0 or args.len > 2) {
        try inspectUsage(stderr);
        return error.InvalidArguments;
    }
    const section = if (args.len == 2) inspectSection(args[1]) orelse {
        try inspectUsage(stderr);
        return error.InvalidInspectionSection;
    } else 0;
    const allocator = init.arena.allocator();
    const cwd = try std.process.currentPathAlloc(init.io, allocator);
    const library_path = try std.fs.path.resolve(allocator, &.{ cwd, args[0] });
    var library = try Library.open(allocator, library_path);
    defer library.close();
    const abi_version = library.lookup(*const fn () callconv(.c) u32, "zgc_abi_version") orelse return error.NotZgcExecutable;
    if (abi_version() != build_options.abi_version) return error.IncompatibleAbiVersion;
    const render = library.lookup(*const fn (u32, ?[*]u8, usize, ?*usize) callconv(.c) c_int, "zgc_executable_inspect") orelse {
        try stderr.writeAll("this executable is not inspectable; compile it with --inspectable\n");
        return error.InspectionUnavailable;
    };
    var required: usize = 0;
    if (render(section, null, 0, &required) != 0) return error.InspectionFailed;
    const bytes = try allocator.alloc(u8, required);
    if (render(section, bytes.ptr, bytes.len, &required) != 0) return error.InspectionFailed;
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    try stdout_writer.interface.writeAll(bytes[0..required]);
    try stdout_writer.interface.flush();
}

fn inspectSection(name: []const u8) ?u32 {
    const sections = [_][]const u8{ "all", "summary", "raw-graph", "semantic-graph", "executable", "representations", "tree", "memory-plan" };
    for (sections, 0..) |section, index| {
        if (std.mem.eql(u8, name, section)) return @intCast(index);
    }
    return null;
}

fn timeCommand(init: std.process.Init, args: []const []const u8, stderr: *std.Io.Writer) !void {
    var options: timing.Options = .{};
    var input: ?[]const u8 = null;
    var json_path: ?[]const u8 = null;

    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (isHelp(arg)) {
            try timeUsage(stderr);
            return;
        }
        const timing_value = std.mem.eql(u8, arg, "--runs") or
            std.mem.eql(u8, arg, "--warmup-ms") or
            std.mem.eql(u8, arg, "--warmup-iterations") or
            std.mem.eql(u8, arg, "--sample-ms") or
            std.mem.eql(u8, arg, "--iterations");
        if (std.mem.eql(u8, arg, "--out")) {
            index += 1;
            if (index == args.len) return error.MissingOutputPath;
            json_path = args[index];
            continue;
        }
        if (timing_value) {
            index += 1;
            if (index == args.len) return error.MissingOptionValue;
            const value = args[index];
            const number = try std.fmt.parseInt(usize, value, 10);
            if (std.mem.eql(u8, arg, "--runs")) options.runs = number else if (std.mem.eql(u8, arg, "--warmup-ms")) options.warmup_ms = number else if (std.mem.eql(u8, arg, "--warmup-iterations")) options.warmup_iterations = number else if (std.mem.eql(u8, arg, "--sample-ms")) options.sample_ms = number else options.iterations = number;
        } else if (arg.len != 0 and arg[0] == '-') {
            return error.InvalidTimeOption;
        } else if (input == null) {
            input = arg;
        } else {
            return error.MultipleInputs;
        }
    }

    const source = input orelse {
        try timeUsage(stderr);
        return error.MissingInput;
    };
    if (std.mem.eql(u8, std.fs.path.extension(source), ".zgir")) return error.ExpectedExecutableLibrary;
    if (options.runs == 0 or (options.iterations == 0 and options.sample_ms == 0)) return error.InvalidTimingConfiguration;
    try timing.run(init, source, options, build_options.abi_version, json_path);
}

const DefinitionImport = struct { name: []const u8, path: []const u8 };

fn compile(init: std.process.Init, args: []const []const u8, stderr: *std.Io.Writer) !void {
    const allocator = init.arena.allocator();
    var filtered: std.ArrayList([]const u8) = .empty;
    var show_time = false;
    for (args) |arg| {
        if (std.mem.eql(u8, arg, "--time")) {
            show_time = true;
        } else {
            try filtered.append(allocator, arg);
        }
    }
    const start = std.Io.Clock.awake.now(init.io).nanoseconds;
    const output = try compileArtifact(init, filtered.items, stderr) orelse return;
    const elapsed = std.Io.Clock.awake.now(init.io).nanoseconds - start;
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    try stdout_writer.interface.print("{s}\n", .{output});
    try stdout_writer.interface.flush();
    if (show_time) {
        try stderr.print("compilation: {d:.3} s\n", .{@as(f64, @floatFromInt(elapsed)) / std.time.ns_per_s});
        try stderr.flush();
    }
}

const ArtifactKind = enum {
    shared,
    static,
    executable,
};

/// Compiles a .zgir or Zig definition into a native library or executable.
fn compileArtifact(init: std.process.Init, args: []const []const u8, stderr: *std.Io.Writer) !?[]const u8 {
    const allocator = init.arena.allocator();
    var input: ?[]const u8 = null;
    var output: ?[]const u8 = null;
    var optimize: []const u8 = "ReleaseFast";
    var target: ?[]const u8 = null;
    var cpu: ?[]const u8 = null;
    var selection_target: []const u8 = "balanced";
    var frontier_index: ?usize = null;
    var zig: []const u8 = "zig";
    var source_root: []const u8 = build_options.source_root;
    var cache_dir: ?[]const u8 = null;
    var cached_output = false;
    var inspectable = false;
    var bind_outputs = false;
    var requested_name: ?[]const u8 = null;
    var artifact_kind: ArtifactKind = .shared;
    var imports: std.ArrayList(DefinitionImport) = .empty;

    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (isHelp(arg)) {
            try compileUsage(stderr);
            return null;
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
        } else if (std.mem.eql(u8, arg, "--cpu")) {
            index += 1;
            if (index == args.len) return error.MissingCpu;
            cpu = args[index];
        } else if (std.mem.eql(u8, arg, "--selection-target")) {
            index += 1;
            if (index == args.len) return error.MissingSelectionTarget;
            selection_target = args[index];
        } else if (std.mem.eql(u8, arg, "--frontier-index")) {
            index += 1;
            if (index == args.len) return error.MissingFrontierIndex;
            frontier_index = try std.fmt.parseInt(usize, args[index], 10);
        } else if (std.mem.eql(u8, arg, "--zig")) {
            index += 1;
            if (index == args.len) return error.MissingZigPath;
            zig = args[index];
        } else if (std.mem.eql(u8, arg, "--zgc-root")) {
            index += 1;
            if (index == args.len) return error.MissingSourceRoot;
            source_root = args[index];
        } else if (std.mem.eql(u8, arg, "--cache-dir")) {
            index += 1;
            if (index == args.len) return error.MissingCachePath;
            cache_dir = args[index];
        } else if (std.mem.eql(u8, arg, "--cache-output")) {
            cached_output = true;
        } else if (std.mem.eql(u8, arg, "--inspectable")) {
            inspectable = true;
        } else if (std.mem.eql(u8, arg, "--bind-outputs")) {
            bind_outputs = true;
        } else if (std.mem.eql(u8, arg, "--name")) {
            index += 1;
            if (index == args.len) return error.MissingArtifactName;
            requested_name = args[index];
        } else if (std.mem.eql(u8, arg, "--artifact")) {
            index += 1;
            if (index == args.len) return error.MissingArtifactKind;
            artifact_kind = std.meta.stringToEnum(ArtifactKind, args[index]) orelse return error.InvalidArtifactKind;
        } else if (std.mem.eql(u8, arg, "--import")) {
            index += 1;
            if (index == args.len) return error.MissingImport;
            const equal = std.mem.indexOfScalar(u8, args[index], '=') orelse return error.InvalidImport;
            if (equal == 0 or equal + 1 == args[index].len) return error.InvalidImport;
            try imports.append(allocator, .{ .name = args[index][0..equal], .path = args[index][equal + 1 ..] });
        } else if (arg.len != 0 and arg[0] == '-') {
            try stderr.print("unknown compile option: {s}\n", .{arg});
            return error.InvalidOption;
        } else if (input == null) {
            input = arg;
        } else return error.MultipleInputs;
    }

    const input_path = input orelse return error.MissingInput;
    const is_zgir = std.mem.eql(u8, std.fs.path.extension(input_path), ".zgir");
    if (!is_zgir and !std.mem.eql(u8, std.fs.path.extension(input_path), ".zig")) return error.ExpectedDefinition;
    if (is_zgir and imports.items.len != 0) return error.GraphDoesNotAcceptImports;
    if (!is_zgir and bind_outputs) return error.OutputBindingRequiresZgir;
    if (!is_zgir and (!std.mem.eql(u8, selection_target, "balanced") or frontier_index != null)) return error.SelectionRequiresZgir;
    if (!validOptimize(optimize)) return error.InvalidOptimizeMode;
    if (!validSelectionTarget(selection_target)) return error.InvalidSelectionTarget;
    if (cached_output and output != null) return error.ConflictingOutputOptions;
    if (requested_name) |name| {
        if (!validArtifactName(name)) return error.InvalidArtifactName;
    }

    const cwd = try std.process.currentPathAlloc(init.io, allocator);
    const cache_root = try cachePath(init, cache_dir);
    const zig_cache = try std.fs.path.join(allocator, &.{ cache_root, "zig" });
    var cache_directory = try std.Io.Dir.createDirPathOpen(.cwd(), init.io, zig_cache, .{});
    cache_directory.close(init.io);
    const absolute_input = try std.fs.path.resolve(allocator, &.{ cwd, input_path });
    const absolute_root = try std.fs.path.resolve(allocator, &.{ cwd, source_root });
    const target_os = if (target) |triple|
        (try std.Target.Query.parse(.{ .arch_os_abi = triple })).os_tag orelse builtin.os.tag
    else
        builtin.os.tag;
    const output_path = if (cached_output) blk: {
        var digest: [32]u8 = undefined;
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        hasher.update(try std.Io.Dir.readFileAlloc(.cwd(), init.io, absolute_input, allocator, .limited(1024 * 1024 * 1024)));
        if (!is_zgir) hasher.update(absolute_input);
        hasher.update(absolute_root);
        hasher.update(optimize);
        hasher.update(target orelse "native");
        hasher.update(cpu orelse "default-cpu");
        hasher.update(selection_target);
        hasher.update(zig);
        hasher.update(if (inspectable) "inspection:on" else "inspection:off");
        hasher.update(if (bind_outputs) "outputs:bound" else "outputs:owned");
        hasher.update(@tagName(artifact_kind));
        if (frontier_index) |value| hasher.update(try std.fmt.allocPrint(allocator, "{d}", .{value}));
        for (imports.items) |item| {
            const path = try std.fs.path.resolve(allocator, &.{ cwd, item.path });
            hasher.update(item.name);
            hasher.update(path);
            hasher.update(try std.Io.Dir.readFileAlloc(.cwd(), init.io, path, allocator, .limited(1024 * 1024 * 1024)));
        }
        hasher.final(&digest);
        const hex = std.fmt.bytesToHex(digest, .lower);
        const filename = try defaultOutput(allocator, requested_name orelse input_path, target_os, artifact_kind);
        break :blk try std.fs.path.join(allocator, &.{ cache_root, "executables", &hex, filename });
    } else output orelse try defaultOutput(allocator, requested_name orelse input_path, target_os, artifact_kind);
    const absolute_output = try std.fs.path.resolve(allocator, &.{ cwd, output_path });
    if (cached_output) {
        var output_directory = try std.Io.Dir.createDirPathOpen(.cwd(), init.io, std.fs.path.dirname(absolute_output).?, .{});
        output_directory.close(init.io);
    }
    var definition_path = absolute_input;
    var graph_path: ?[]const u8 = null;
    if (is_zgir) {
        const graph = try std.Io.Dir.readFileAlloc(.cwd(), init.io, absolute_input, allocator, .limited(1024 * 1024 * 1024));
        const adapter_path = try std.fs.path.join(allocator, &.{ absolute_root, "src", "artifact", "zgir_executable.zig" });
        const adapter_source = try std.Io.Dir.readFileAlloc(.cwd(), init.io, adapter_path, allocator, .limited(1024 * 1024));
        const source_adapter_path = try std.fs.path.join(allocator, &.{ absolute_root, "src", "artifact", "zgir_source.zig" });
        const source_adapter = try std.Io.Dir.readFileAlloc(.cwd(), init.io, source_adapter_path, allocator, .limited(1024 * 1024));
        const frontier_text = if (frontier_index) |value| try std.fmt.allocPrint(allocator, "{d}", .{value}) else "null";
        var digest: [32]u8 = undefined;
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        hasher.update(graph);
        hasher.update(selection_target);
        hasher.update(frontier_text);
        hasher.update(if (bind_outputs) "outputs:bound" else "outputs:owned");
        hasher.final(&digest);
        const hex = std.fmt.bytesToHex(digest, .lower);
        const work_path = try std.fs.path.join(allocator, &.{ cache_root, "graphs", &hex });
        var work = try std.Io.Dir.createDirPathOpen(.cwd(), init.io, work_path, .{});
        defer work.close(init.io);
        try work.writeFile(init.io, .{ .sub_path = "graph.zgir", .data = graph });
        try work.writeFile(init.io, .{ .sub_path = "executable.zig", .data = adapter_source });
        const configured_source_adapter = try std.fmt.allocPrint(
            allocator,
            "{s}\npub const executable_options = .{{ .target = .{s}, .frontier_index = {s}, .outputs = .{s} }};\n",
            .{ source_adapter, selection_target, frontier_text, if (bind_outputs) "bound" else "owned" },
        );
        try work.writeFile(init.io, .{ .sub_path = "graph.zig", .data = configured_source_adapter });
        definition_path = try std.fs.path.join(allocator, &.{ work_path, "executable.zig" });
        graph_path = try std.fs.path.join(allocator, &.{ work_path, "graph.zig" });
    }
    const artifact_root = switch (artifact_kind) {
        .shared, .static => if (inspectable) "inspectable_executable_abi.zig" else "executable_abi.zig",
        .executable => if (inspectable) "inspectable_executable_runner.zig" else "executable_runner.zig",
    };
    const abi_path = try std.fs.path.join(allocator, &.{ absolute_root, "src", "artifact", artifact_root });
    const root_path = try std.fs.path.join(allocator, &.{ absolute_root, "src", "root.zig" });
    const optimize_arg = try std.fmt.allocPrint(allocator, "-O{s}", .{optimize});
    const root_arg = try std.fmt.allocPrint(allocator, "-Mroot={s}", .{abi_path});
    const executable_arg = try std.fmt.allocPrint(allocator, "-Mexecutable={s}", .{definition_path});
    const zgc_arg = try std.fmt.allocPrint(allocator, "-Mzgc={s}", .{root_path});
    const output_arg = try std.fmt.allocPrint(allocator, "-femit-bin={s}", .{absolute_output});
    var command: std.ArrayList([]const u8) = .empty;
    try command.appendSlice(allocator, &.{ zig, if (artifact_kind == .executable) "build-exe" else "build-lib", optimize_arg });
    if (artifact_kind == .shared) try command.append(allocator, "-dynamic");
    if (artifact_kind == .static) try command.append(allocator, "-static");
    if (artifact_kind == .shared and target_os == .linux) {
        try command.append(allocator, try std.fmt.allocPrint(
            allocator,
            "-fsoname={s}",
            .{std.fs.path.basename(absolute_output)},
        ));
    }
    if (target) |value| {
        try command.appendSlice(allocator, &.{ "-target", value });
    }
    if (cpu) |value| {
        try command.appendSlice(allocator, &.{ "-mcpu", value });
    }
    try command.appendSlice(allocator, &.{ "--dep", "executable", "--dep", "zgc", root_arg, "--dep", "zgc" });
    if (graph_path != null) try command.appendSlice(allocator, &.{ "--dep", "graph" });
    for (imports.items) |item| try command.appendSlice(allocator, &.{ "--dep", item.name });
    try command.appendSlice(allocator, &.{ executable_arg, zgc_arg });
    if (graph_path) |path| {
        try command.append(allocator, try std.fmt.allocPrint(allocator, "-Mgraph={s}", .{path}));
    }
    for (imports.items) |item| {
        const path = try std.fs.path.resolve(allocator, &.{ cwd, item.path });
        try command.append(allocator, try std.fmt.allocPrint(allocator, "-M{s}={s}", .{ item.name, path }));
    }
    try command.append(allocator, output_arg);

    var environment = try init.environ_map.clone(allocator);
    defer environment.deinit();
    try environment.put("ZIG_LOCAL_CACHE_DIR", zig_cache);
    const result = try std.process.run(allocator, init.io, .{
        .argv = command.items,
        .environ_map = &environment,
        .stdout_limit = .limited(16 * 1024 * 1024),
        .stderr_limit = .limited(16 * 1024 * 1024),
    });
    if (result.stdout.len != 0) try stderr.writeAll(result.stdout);
    if (result.stderr.len != 0) try stderr.writeAll(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) return error.CompilerFailed,
        else => return error.CompilerFailed,
    }

    return absolute_output;
}

fn defaultOutput(allocator: std.mem.Allocator, input: []const u8, target_os: std.Target.Os.Tag, kind: ArtifactKind) ![]const u8 {
    const basename = std.fs.path.basename(input);
    const extension = std.fs.path.extension(basename);
    const stem = basename[0 .. basename.len - extension.len];
    const suffix = switch (kind) {
        .shared => switch (target_os) {
            .windows => ".dll",
            .macos, .ios, .tvos, .watchos, .visionos => ".dylib",
            else => ".so",
        },
        .static => if (target_os == .windows) ".lib" else ".a",
        .executable => if (target_os == .windows) ".exe" else "",
    };
    return std.fmt.allocPrint(allocator, "{s}{s}{s}", .{ if (kind == .static and target_os != .windows) "lib" else "", stem, suffix });
}

fn validOptimize(value: []const u8) bool {
    return std.mem.eql(u8, value, "Debug") or
        std.mem.eql(u8, value, "ReleaseSafe") or
        std.mem.eql(u8, value, "ReleaseFast") or
        std.mem.eql(u8, value, "ReleaseSmall");
}

fn validArtifactName(name: []const u8) bool {
    if (name.len == 0 or name[0] == '.') return false;
    for (name) |byte| {
        if (!std.ascii.isAlphanumeric(byte) and byte != '-' and byte != '_' and byte != '.') return false;
    }
    return true;
}

fn validSelectionTarget(value: []const u8) bool {
    return std.mem.eql(u8, value, "balanced") or
        std.mem.eql(u8, value, "memory") or
        std.mem.eql(u8, value, "fast") or
        std.mem.eql(u8, value, "small");
}

fn isHelp(arg: []const u8) bool {
    return std.mem.eql(u8, arg, "help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help");
}

fn usage(writer: *std.Io.Writer) !void {
    try writer.writeAll(
        \\usage: zgc <command> [options]
        \\
        \\commands:
        \\  compile    specialize a .zgir or Zig definition and emit a native artifact
        \\  inspect    show metadata and graph representations from a shared library
        \\  time       measure a compiled executable runtime
        \\  cache      show or clear the native compiler cache
        \\
        \\run 'zgc <command> --help' for command options
        \\
    );
}

fn cacheUsage(writer: *std.Io.Writer) !void {
    try writer.writeAll(
        \\usage: zgc cache <path|clear> [--cache-dir <path>]
        \\
    );
}

fn inspectUsage(writer: *std.Io.Writer) !void {
    try writer.writeAll(
        \\usage: zgc inspect <executable-library> [all|summary|raw-graph|semantic-graph|executable|representations|tree|memory-plan]
        \\
        \\The default section is all. Inspection loads the library but does not initialize or run the runtime.
        \\
    );
}

fn compileUsage(writer: *std.Io.Writer) !void {
    try writer.writeAll(
        \\usage: zgc compile <graph.zgir|definition.zig> [-o <artifact>] [options]
        \\
        \\options:
        \\  -o, --output <path>      output artifact path
        \\  --artifact <kind>        shared (default), static, or executable
        \\  -O, --optimize <mode>   Debug, ReleaseSafe, ReleaseFast, or ReleaseSmall
        \\  --target <triple>        Zig compilation target
        \\  --cpu <model>            Zig CPU model and features (for example, cortex_m0plus)
        \\  --selection-target <name>  balanced, memory, fast, or small
        \\  --frontier-index <n>   select a retained executable by index
        \\  --zig <path>             Zig executable (default: zig from PATH)
        \\  --zgc-root <path>        ZGC source checkout
        \\  --cache-dir <path>       local Zig and graph cache (default: .zgc-cache)
        \\  --cache-output           emit under the native executable cache
        \\  --inspectable            embed inspection data and export zgc_executable_inspect
        \\  --bind-outputs           require caller storage for every output (.zgir only)
        \\  --name <name>            cached artifact filename stem
        \\  --import <name=path>     Zig module import (repeatable for .zig definitions)
        \\  --time                   report total compilation time after success
        \\
    );
}

fn timeUsage(writer: *std.Io.Writer) !void {
    try writer.writeAll(
        \\usage: zgc time <executable-library> [options]
        \\
        \\timing options:
        \\  --runs <n>                 timed samples (default: 30)
        \\  --warmup-ms <n>            warmup duration (default: 2000)
        \\  --warmup-iterations <n>    fixed warmup iterations instead
        \\  --sample-ms <n>            target duration per sample (default: 250)
        \\  --iterations <n>           fixed iterations per sample instead
        \\  --out <path>               append a timing record to a JSON array
        \\
        \\Sources are initialized to zero. Setup is excluded from runtime samples.
        \\Compile .zgir files first with 'zgc compile'; use its --time flag to time compilation.
        \\
    );
}
