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
    const Model = zgc.frontend.modelFromZgir(source);
    try std.testing.expectEqual(@as(usize, 2), Model.source_names.len);
    try std.testing.expectEqualStrings("x", Model.source_names[0]);
    try std.testing.expectEqualStrings("bias", Model.source_names[1]);
}
