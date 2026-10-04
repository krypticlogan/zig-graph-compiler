const std = @import("std");
const zgc = @import("zgc");
const nn = zgc.ext.nn;
const Sources = enum(usize) { input, w1, b1, w2, b2 };
const Definition = zgc.DefinitionBuilder(Sources, .{
    .max_rank = 2,
    .max_nodes = 6,
    .max_tensors = 11,
    .max_input_refs = 10,
    .max_outputs = 1,
});
const Dense = nn.Dense(Sources);
const Classifier = nn.Sequential(&[_]Dense{
    .{
        .weights = .w1,
        .bias = .b1,
        .output_size = 2,
        .activation = .relu,
    },
    .{
        .weights = .w2,
        .bias = .b2,
        .output_size = 2,
        .activation = .softmax,
    },
});

const definition = blk: {
    var builder = Definition.init();
    const input = builder.input(.input, .f32, &.{ 2, 2 });
    builder.output(Classifier.apply(&builder, input));
    break :blk builder.finish();
};
const Model = definition.model();

test "dense layers build a sequential core graph" {
    const graph = Model.executable;

    try std.testing.expectEqual(@as(usize, 6), Model.semantic_graph.node_ct);
    try std.testing.expectEqual(@as(usize, 5), graph.node_ct);
    switch (graph.nodes[1].?.op.compute.kernel) {
        .map => |plan| try std.testing.expectEqual(
            @as(usize, 2),
            plan.region.body.expression.instructions.len,
        ),
        else => return error.TestUnexpectedResult,
    }
    try std.testing.expectEqualSlices(
        usize,
        &.{ 2, 2 },
        graph.tensors[graph.outputs[0].?].?.shape.slice(),
    );
    try std.testing.expectEqual(@as(usize, 2), Classifier.layer_count);
    try std.testing.expectEqual(nn.Activation.relu, Classifier.layer_definitions[0].activation);
    try std.testing.expectEqual(nn.Activation.softmax, Classifier.layer_definitions[1].activation);
}

test "dense sequential model executes through core kernels" {
    var model = Model.init();
    const input = [_]f32{ 4, 5, -2, 1 };
    const identity = [_]f32{ 1, 0, 0, 1 };
    const first_bias = [_]f32{ 1, -2 };
    const zero_bias = [_]f32{ 0, 0 };

    try model.copyInput(.input, &input);
    try model.copySource(.w1, &identity);
    try model.copySource(.b1, &first_bias);
    try model.copySource(.w2, &identity);
    try model.copySource(.b2, &zero_bias);
    model.run();

    const output = model.outputView(0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.880797), output.get(.{ 0, 0 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.11920292), output.get(.{ 0, 1 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output.get(.{ 1, 0 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output.get(.{ 1, 1 }), 1e-6);
}

test "dense output-input weights create an aliasing transpose" {
    const LayoutSources = enum(usize) { input, weights, bias };
    const LayoutDefinition = zgc.DefinitionBuilder(LayoutSources, .{
        .max_rank = 2,
        .max_nodes = 3,
        .max_tensors = 6,
        .max_input_refs = 5,
        .max_outputs = 1,
    });
    const OutputMajorDense = nn.Dense(LayoutSources);
    const layout_definition = comptime blk: {
        var builder = LayoutDefinition.init();
        const input = builder.input(.input, .f32, &.{ 1, 3 });
        const layer: OutputMajorDense = .{
            .weights = .weights,
            .bias = .bias,
            .output_size = 2,
            .weight_layout = .output_input,
        };
        builder.output(layer.apply(&builder, input));
        break :blk builder.finish();
    };
    const graph = layout_definition.model().executable;

    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, graph.tensors[1].?.shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, graph.tensors[2].?.shape.slice());
    try std.testing.expectEqual(graph.tensors[1].?.storage_tensor, graph.tensors[2].?.storage_tensor);
}


const img = zgc.ext.img;
test "image helpers declare channel-aware core inputs" {
    try std.testing.expectEqual(
        [4]usize{ 2, 28, 28, 3 },
        (img.Dimensions{ .batch = 2, .height = 28, .width = 28, .channels = 3 }).shape(),
    );
    try std.testing.expectEqual(
        [4]usize{ 2, 3, 28, 28 },
        (img.Dimensions{
            .batch = 2,
            .height = 28,
            .width = 28,
            .channels = 3,
            .layout = .channels_first,
        }).shape(),
    );

    const ImageSources = enum(usize) { image };
    const ImageDefinition = zgc.DefinitionBuilder(ImageSources, .{
        .max_rank = 4,
        .max_outputs = 1,
    });
    const image_definition = comptime blk: {
        var builder = ImageDefinition.init();
        const image = img.input(
            &builder,
            .image,
            .f32,
            .{ .batch = 2, .height = 28, .width = 28, .channels = 3 },
        );
        builder.output(image);
        break :blk builder.finish();
    };
    try std.testing.expectEqualSlices(
        usize,
        &.{ 2, 28, 28, 3 },
        image_definition.tensors[0].value.shape.slice(),
    );
}
