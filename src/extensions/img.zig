const Dtype = @import("../zgc/storage/dtype.zig").Dtype;

pub const Layout = enum {
    channels_last,
    channels_first,
};

pub const Dimensions = struct {
    batch: usize,
    height: usize,
    width: usize,
    channels: usize,
    layout: Layout = .channels_last,

    pub fn shape(comptime dimensions: Dimensions) [4]usize {
        if (dimensions.batch == 0 or
            dimensions.height == 0 or
            dimensions.width == 0 or
            dimensions.channels == 0)
        {
            @compileError("zgc.ext.img dimensions must be greater than zero");
        }
        return switch (dimensions.layout) {
            .channels_last => .{
                dimensions.batch,
                dimensions.height,
                dimensions.width,
                dimensions.channels,
            },
            .channels_first => .{
                dimensions.batch,
                dimensions.channels,
                dimensions.height,
                dimensions.width,
            },
        };
    }
};

/// Declares a rank-4 image input using the selected channel convention.
pub fn input(
    builder: anytype,
    comptime source_key: @TypeOf(builder.*).Source,
    comptime dtype: Dtype,
    comptime dimensions: Dimensions,
) @TypeOf(builder.*).TensorValue {
    const image_shape = dimensions.shape();
    return builder.input(source_key, dtype, &image_shape);
}
