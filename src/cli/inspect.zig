const std = @import("std");
const zgc = @import("zgc");
const inspected_executable = @import("executable");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    const success = try zgc.Inspect.runModuleCli(inspected_executable, args[1..], stdout);
    try stdout.flush();
    if (!success) std.process.exit(2);
}
