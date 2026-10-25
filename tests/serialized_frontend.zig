const std = @import("std");
const zgc = @import("zgc");

test "serialized frontend instantiates a definition" {
    const source =
        \\zgir 1
        \\
        \\%0 = input(name="x", dtype=f32, shape=[2], binding=bound)
        \\%1 = parameter(name="bias", dtype=f32, shape=[2], binding=owned)
        \\%2 = add(%0, %1)
        \\%3 = relu(%2)
        \\
        \\outputs = [%3]
        \\
    ;
    const Executable = zgc.frontend.executableFromZgir(source);
    const Memory = zgc.frontend.executableFromZgirWithOptions(source, .{ .target = .memory });
    try std.testing.expectEqual(@as(usize, 2), Executable.source_names.len);
    try std.testing.expectEqualStrings("x", Executable.source_names[0]);
    try std.testing.expectEqualStrings("bias", Executable.source_names[1]);
    try std.testing.expectEqual(.memory, Memory.selection_options.target);
    try std.testing.expectEqual(.bound, Memory.source_plan.source_bindings[0]);
}

test "u8 ZGIR executes with matching ABI dtype" {
    const source =
        \\zgir 1
        \\
        \\%0 = input(name="pixels", dtype=u8, shape=[3], binding=owned)
        \\%1 = scalar(dtype=u8, value=5)
        \\%2 = add(%0, %1)
        \\
        \\outputs = [%2]
        \\
    ;
    const Executable = zgc.frontend.executableFromZgir(source);
    const Abi = zgc.artifact.ABI.Adapter(Executable);
    var descriptor: zgc.artifact.ABI.SourceDescriptor = undefined;
    try std.testing.expectEqual(.ok, Abi.sourceDescriptor(0, &descriptor));
    try std.testing.expectEqual(.u8, descriptor.tensor.dtype);
    var runtime = Executable.Runtime.init();
    try runtime.copyInput(@fromBackingInt(0), &[_]u8{ 0, 4, 5 });
    runtime.run();
    try std.testing.expectEqualSlices(u8, &.{ 5, 9, 10 }, runtime.outputView(0).contiguousSlice().?);
}
