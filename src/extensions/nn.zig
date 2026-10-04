pub const Activation = enum {
    none,
    relu,
    softmax,

    pub fn apply(
        comptime activation: Activation,
        builder: anytype,
        comptime value: @TypeOf(builder.*).TensorValue,
    ) @TypeOf(builder.*).TensorValue {
        return switch (activation) {
            .none => value,
            .relu => builder.relu(value),
            .softmax => builder.softmax(value, value.shape.rank - 1),
        };
    }
};

pub const WeightLayout = enum {
    /// Logical and stored shape `[input, output]`.
    input_output,
    /// Stored shape `[output, input]`, exposed to matmul through a transpose.
    output_input,
};

/// Configuration for a fully connected graph layer. Applying the layer adds
/// its parameter sources and computation to a `DefinitionBuilder`.
pub fn Dense(comptime SourceKey: type) type {
    return struct {
        const Self = @This();

        weights: SourceKey,
        bias: SourceKey,
        output_size: usize,
        activation: Activation = .none,
        weight_layout: WeightLayout = .input_output,

        pub fn apply(
            comptime layer: Self,
            builder: anytype,
            comptime input: @TypeOf(builder.*).TensorValue,
        ) @TypeOf(builder.*).TensorValue {
            if (input.shape.rank != 2) {
                @compileError("zgc.ext.nn.Dense requires a rank-2 [batch, features] input");
            }
            if (layer.output_size == 0) {
                @compileError("zgc.ext.nn.Dense output_size must be greater than zero");
            }
            if (input.dtype != .f32) {
                @compileError("zgc.ext.nn.Dense currently supports only f32 tensors");
            }

            const input_size = input.shape.at(1);
            const stored_shape = switch (layer.weight_layout) {
                .input_output => &.{ input_size, layer.output_size },
                .output_input => &.{ layer.output_size, input_size },
            };
            const stored_weights = builder.parameter(
                layer.weights,
                input.dtype,
                stored_shape,
            );
            const weights = switch (layer.weight_layout) {
                .input_output => stored_weights,
                .output_input => builder.transpose(stored_weights, 0, 1),
            };
            const bias = builder.parameter(
                layer.bias,
                input.dtype,
                &.{layer.output_size},
            );
            const affine = builder.add(builder.matmul(input, weights), bias);
            return layer.activation.apply(builder, affine);
        }
    };
}

/// Composes graph-layer values from left to right. Every layer must expose an
/// `apply(builder, value)` function returning the builder's tensor value type.
pub fn Sequential(comptime layers: anytype) type {
    if (layers.len == 0) {
        @compileError("zgc.ext.nn.Sequential requires at least one layer");
    }

    return struct {
        pub const layer_definitions = layers;
        pub const layer_count = layers.len;

        pub fn apply(
            builder: anytype,
            comptime input: @TypeOf(builder.*).TensorValue,
        ) @TypeOf(builder.*).TensorValue {
            var value = input;
            inline for (layers) |layer| {
                value = layer.apply(builder, value);
            }
            return value;
        }
    };
}
