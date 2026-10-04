const zgc = @import("zgc");
const params = @import("model_params");

pub const batch_size = 1;
pub const input_size = 784;
pub const hidden_size_1 = 128;
pub const hidden_size_2 = 64;
pub const output_size = 10;

// Source keys form the typed boundary between the graph and storage supplied
// either by the caller or by the model's parameter package.
pub const Sources = enum(usize) {
    input,
    w1,
    b1,
    w2,
    b2,
    w3,
    b3,
};

pub const Definition = zgc.DefinitionBuilder(Sources, .{
    .max_rank = 2,
    .max_nodes = 12,
    .max_tensors = 19,
    .max_input_refs = 18,
    .max_outputs = 1,
});

const Dense = zgc.ext.nn.Dense(Sources);

// The layer description captures architecture independently of storage. The
// parameter files use output-major weights, which Dense adapts for contraction.
const Network = zgc.ext.nn.Sequential(&[_]Dense{
    .{
        .weights = .w1,
        .bias = .b1,
        .output_size = hidden_size_1,
        .activation = .relu,
        .weight_layout = .output_input,
    },
    .{
        .weights = .w2,
        .bias = .b2,
        .output_size = hidden_size_2,
        .activation = .relu,
        .weight_layout = .output_input,
    },
    .{
        .weights = .w3,
        .bias = .b3,
        .output_size = output_size,
        .activation = .softmax,
        .weight_layout = .output_input,
    },
});

fn defineGraph(builder: *Definition) void {
    // A runtime image flows through the composed layers to one probability
    // vector; Sequential contributes the intermediate graph nodes.
    const input = builder.input(.input, .f32, &.{ batch_size, input_size });
    builder.output(Network.apply(builder, input));
}

// Finishing freezes and validates graph geometry before any source is assigned
// a concrete storage policy.
pub const definition = blk: {
    var builder = Definition.init();
    defineGraph(&builder);
    break :blk builder.finish();
};

// Inputs remain caller-owned and replaceable, while immutable parameters are
// embedded into the executable and available for compile-time source packing.
pub const Model = definition.modelWith(&.{
    .{ .source = .input, .binding = zgc.memory.Source.bound },
    .{ .source = .w1, .binding = zgc.memory.Source.embed(params.w1) },
    .{ .source = .b1, .binding = zgc.memory.Source.embed(params.b1) },
    .{ .source = .w2, .binding = zgc.memory.Source.embed(params.w2) },
    .{ .source = .b2, .binding = zgc.memory.Source.embed(params.b2) },
    .{ .source = .w3, .binding = zgc.memory.Source.embed(params.w3) },
    .{ .source = .b3, .binding = zgc.memory.Source.embed(params.b3) },
});

pub fn bindInput(model: *Model, input: *const [input_size]f32) void {
    // Binding supplies only runtime storage; its dtype and geometry were fixed
    // by the definition above.
    model.bindInput(.input, input) catch unreachable;
}
