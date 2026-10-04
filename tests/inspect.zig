const std = @import("std");
const zgc = @import("zgc");
const models = @import("fixtures/models.zig");
const Model = models.BasicModel;

test "inspection renders a generated model through a writer" {
    var buffer: [8192]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    try zgc.Inspect.writeModel(Model, &writer, .{});
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
    try std.testing.expect(try zgc.Inspect.runCli(Model, &.{"semantic-graph"}, &writer));
    const output = writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "== Optimized semantic graph ==") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "== Raw graph ==") == null);
    try std.testing.expect(std.mem.indexOf(u8, output, "== Executable ==") == null);
}

test "inspection CLI reports invalid commands" {
    var buffer: [2048]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    try std.testing.expect(!try zgc.Inspect.runCli(Model, &.{"unknown"}, &writer));
    try std.testing.expect(std.mem.indexOf(u8, writer.buffered(), "usage: zgc-inspect") != null);
}

test "inspection CLI infers a uniquely exported model declaration" {
    const SingleModelModule = struct {
        pub const FluidStep = Model;
    };
    var buffer: [2048]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);

    try std.testing.expect(try zgc.Inspect.runModuleCli(SingleModelModule, &.{"summary"}, &writer));
    try std.testing.expect(std.mem.indexOf(u8, writer.buffered(), "== Capacity ==") != null);
}

test "inspection CLI selects a model declaration from a module" {
    var buffer: [4096]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);

    try std.testing.expect(try zgc.Inspect.runModuleCli(
        models,
        &.{ "--model", "FullModel", "summary" },
        &writer,
    ));
    try std.testing.expect(std.mem.indexOf(u8, writer.buffered(), "Capacity(nodes=1") != null);
}

test "inspection CLI requires selection when a module exports multiple models" {
    var buffer: [4096]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);

    try std.testing.expect(!try zgc.Inspect.runModuleCli(models, &.{"summary"}, &writer));
    const output = writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "select one with --model") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "BasicModel") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "FullModel") != null);
}

test "inspection renders bounded model memory" {
    var model = Model.init();
    var buffer: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    try zgc.Inspect.writeModelMemory(&model, &writer, 8);
    const output = writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "showing=8") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "40 bytes omitted") != null);
}

test "inspection renders reused regions in tensor order" {
    var buffer: [2048]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);

    try zgc.Inspect.writeMemoryPlan(models.BoundReuseModel, &writer);
    const output = writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "MemoryPlan(bytes=32") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "t1: [0..16)") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "t3: [0..16)") != null);
}
