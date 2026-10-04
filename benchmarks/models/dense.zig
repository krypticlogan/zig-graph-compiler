const std = @import("std");
const zgc = @import("zgc");

const nn = zgc.ext.nn;

const Sources = enum(usize) {
    input,
    w1,
    b1,
    w2,
    b2,
    w3,
    b3,
    w4,
    b4,
};

const Activation = nn.Activation;
const weight_keys = [_]Sources{ .w1, .w2, .w3, .w4 };
const bias_keys = [_]Sources{ .b1, .b2, .b3, .b4 };

fn DenseBenchmark(
    comptime sizes: []const usize,
    comptime activations: []const Activation,
    comptime batch_size: usize,
    comptime benchmark_name: []const u8,
    comptime iterations: usize,
) type {
    @setEvalBranchQuota(100_000);
    if (sizes.len < 2) @compileError("a dense benchmark requires at least two layer sizes");
    const layer_count = sizes.len - 1;
    if (activations.len != layer_count) @compileError("each dense layer requires an activation");
    if (layer_count > weight_keys.len) @compileError("dense benchmark exceeds the source-key capacity");

    const Definition = zgc.DefinitionBuilder(Sources, .{
        .max_rank = 2,
        .max_nodes = layer_count * 3,
        .max_tensors = 1 + layer_count * 5,
        .max_input_refs = layer_count * 5,
        .max_outputs = 1,
    });
    const Dense = nn.Dense(Sources);
    const Network = comptime network: {
        var layers: [layer_count]Dense = undefined;
        for (0..layer_count) |layer| {
            layers[layer] = .{
                .weights = weight_keys[layer],
                .bias = bias_keys[layer],
                .output_size = sizes[layer + 1],
                .activation = activations[layer],
            };
        }
        break :network nn.Sequential(layers);
    };
    const definition = definition: {
        var builder = Definition.init();
        const input = builder.input(.input, .f32, &.{ batch_size, sizes[0] });
        builder.output(Network.apply(&builder, input));
        break :definition builder.finish();
    };
    const Model = definition.modelWith(&.{.{ .source = .input, .binding = zgc.memory.Source.bound }});
    const input_element_count = batch_size * sizes[0];
    const output_element_count = batch_size * sizes[sizes.len - 1];
    const parameters = parameterCount(sizes);

    return struct {
        const Self = @This();

        pub const name = benchmark_name;
        pub const default_iterations = iterations;
        pub const default_warmup_iterations = @max(iterations / 10, 1);
        pub const work_items_per_invocation: f64 = batch_size;
        pub const work_unit = "inferences";
        pub const bytes_per_invocation: f64 = @sizeOf(f32) *
            (parameters + input_element_count + output_element_count);
        pub const latency_divisor: f64 = batch_size;
        pub const latency_unit = "inference";
        pub const parameter_count = parameters;
        pub const batch = batch_size;

        model: Model,
        input: [input_element_count]f32,

        pub fn init() Self {
            var result: Self = .{
                .model = Model.init(),
                .input = undefined,
            };
            const input_layout = Model.sourceLayout(.input);
            for (0..batch_size) |batch_index| {
                for (0..sizes[0]) |input_index| {
                    const storage_index: usize = @intCast(
                        @as(isize, @intCast(batch_index)) * input_layout.strides[0] +
                            @as(isize, @intCast(input_index)) * input_layout.strides[1],
                    );
                    result.input[storage_index] =
                        sampleValue(batch_index * sizes[0] + input_index, 17);
                }
            }
            inline for (0..layer_count) |layer| {
                var weights: [sizes[layer] * sizes[layer + 1]]f32 = undefined;
                var bias: [sizes[layer + 1]]f32 = undefined;
                for (&weights, 0..) |*value, index| {
                    value.* = sampleValue(index, 31 + layer * 13);
                }
                for (&bias, 0..) |*value, index| {
                    value.* = sampleValue(index, 7 + layer * 19);
                }
                result.model.copySource(weight_keys[layer], &weights) catch unreachable;
                result.model.copySource(bias_keys[layer], &bias) catch unreachable;
            }
            return result;
        }

        pub fn prepare(self: *Self) void {
            self.model.bindInput(.input, &self.input) catch unreachable;
        }

        pub fn run(self: *Self, run_iterations: usize) void {
            for (0..run_iterations) |_| {
                self.model.run();
                std.mem.doNotOptimizeAway(self.model.outputView(0).storage);
            }
        }
    };
}

fn parameterCount(comptime sizes: []const usize) usize {
    var count: usize = 0;
    for (0..sizes.len - 1) |layer| {
        count += sizes[layer] * sizes[layer + 1] + sizes[layer + 1];
    }
    return count;
}

fn sampleValue(index: usize, salt: usize) f32 {
    const centered = @as(f32, @floatFromInt((index * 37 + salt) % 257)) - 128.0;
    return centered / 2048.0;
}

const small_sizes = &[_]usize{ 32, 16, 8 };
const medium_sizes = &[_]usize{ 128, 64, 10 };
const large_sizes = &[_]usize{ 1024, 256, 128, 64, 2 };
const two_layer_activations = &[_]Activation{ .relu, .softmax };
const four_layer_activations = &[_]Activation{ .relu, .relu, .relu, .softmax };

pub fn select(comptime model_name: []const u8, comptime batch_size: usize) type {
    if (batch_size == 0) @compileError("model benchmark batch size must be greater than zero");

    if (std.mem.eql(u8, model_name, "small")) {
        return DenseBenchmark(
            small_sizes,
            two_layer_activations,
            batch_size,
            std.fmt.comptimePrint(
                "model/dense/32-16-8/params664/batch{d}",
                .{batch_size},
            ),
            @max(100_000 / batch_size, 1),
        );
    }
    if (std.mem.eql(u8, model_name, "medium")) {
        return DenseBenchmark(
            medium_sizes,
            two_layer_activations,
            batch_size,
            std.fmt.comptimePrint(
                "model/dense/128-64-10/params8906/batch{d}",
                .{batch_size},
            ),
            @max(10_000 / batch_size, 1),
        );
    }
    if (std.mem.eql(u8, model_name, "large")) {
        return DenseBenchmark(
            large_sizes,
            four_layer_activations,
            batch_size,
            std.fmt.comptimePrint(
                "model/dense/1024-256-128-64-2/params303682/batch{d}",
                .{batch_size},
            ),
            @max(100 / batch_size, 1),
        );
    }
    @compileError("unknown dense model benchmark: " ++ model_name);
}

pub const SmallBatch1 = select("small", 1);
pub const MediumBatch1 = select("medium", 1);
pub const LargeBatch1 = select("large", 1);
pub const SmallBatch32 = select("small", 32);
pub const MediumBatch32 = select("medium", 32);
pub const LargeBatch32 = select("large", 32);
