const std = @import("std");
const Inspect = @import("zgc_internal").Inspect;
const models = @import("fixtures/models.zig");
const Executable = models.BasicExecutable;

test "inspection renders a generated model through a writer" {
    var buffer: [8192]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    try Inspect.writeExecutable(Executable, &writer, .{});
    const output = writer.buffered();

    try std.testing.expect(std.mem.indexOf(u8, output, "Capacity(nodes=2") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "transpose(axes=0,1)") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "relu(t1) -> t2") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Executable structure:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "MemoryPlan(bytes=48") != null);
}

test "inspection CLI selects individual representations" {
    var buffer: [4096]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    try std.testing.expect(try Inspect.runCli(Executable, &.{"semantic-graph"}, &writer));
    const output = writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "== Optimized semantic graph ==") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "== Raw graph ==") == null);
    try std.testing.expect(std.mem.indexOf(u8, output, "== Executable ==") == null);
}

test "inspection CLI infers a uniquely exported executable declaration" {
    const SingleExecutableModule = struct {
        pub const FluidStep = Executable;
    };
    var buffer: [2048]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);

    try std.testing.expect(try Inspect.runModuleCli(SingleExecutableModule, &.{"summary"}, &writer));
    try std.testing.expect(std.mem.indexOf(u8, writer.buffered(), "== Capacity ==") != null);
}

test "inspection CLI selects an executable declaration from a module" {
    var buffer: [4096]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);

    try std.testing.expect(try Inspect.runModuleCli(
        models,
        &.{ "--executable", "FullExecutable", "summary" },
        &writer,
    ));
    try std.testing.expect(std.mem.indexOf(u8, writer.buffered(), "Capacity(nodes=1") != null);
}

test "inspection CLI requires selection when a module exports multiple executables" {
    var buffer: [4096]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);

    try std.testing.expect(!try Inspect.runModuleCli(models, &.{"summary"}, &writer));
    const output = writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "select one with --executable") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "BasicExecutable") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "FullExecutable") != null);
}

test "inspection renders bounded model memory" {
    var model = Executable.Runtime.init();
    var buffer: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    try Inspect.writeRuntimeMemory(&model, &writer, 8);
    const output = writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "showing=8") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "40 bytes omitted") != null);
}

test "inspection renders reused regions in tensor order" {
    var buffer: [2048]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);

    try Inspect.writeMemoryPlan(models.BoundReuseExecutable, &writer);
    const output = writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "MemoryPlan(bytes=32") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "t1: [0..16)") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "t3: [0..16)") != null);
}
