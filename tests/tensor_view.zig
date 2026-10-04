const std = @import("std");
const zgc = @import("zgc");

test "contiguous layout computes row-major strides" {
    const Shape = zgc.core.Tensor.Shape(3);
    const Layout = zgc.core.Tensor.Layout(3);
    const layout = Layout.contiguous(Shape.init(&.{ 2, 3, 4 }));

    try std.testing.expectEqual(@as(usize, 0), layout.offset);
    try std.testing.expectEqual([3]isize{ 12, 4, 1 }, layout.strides);
}

test "view maps logical indices through offset and strides" {
    var storage = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    var view: zgc.core.Tensor.View(i32, 2) = .{
        .storage = &storage,
        .shape = .{ 2, 2 },
        .strides = .{ 3, 1 },
        .offset = 1,
    };

    try std.testing.expectEqual(@as(usize, 4), view.len());
    try std.testing.expectEqual(@as(usize, 1), view.elementOffset(.{ 0, 0 }));
    try std.testing.expectEqual(@as(usize, 5), view.elementOffset(.{ 1, 1 }));
    try std.testing.expectEqual(@as(i32, 5), view.get(.{ 1, 1 }));

    view.set(.{ 1, 0 }, 40);
    try std.testing.expectEqual(@as(i32, 40), storage[4]);
}

test "views identify row-major contiguity" {
    var storage: [8]f32 = @splat(0);
    const contiguous: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &storage,
        .shape = .{ 2, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };
    const transposed: zgc.core.Tensor.View(f32, 2) = .{
        .storage = &storage,
        .shape = .{ 3, 2 },
        .strides = .{ 1, 3 },
        .offset = 0,
    };
    const singleton_axis: zgc.core.Tensor.View(f32, 3) = .{
        .storage = &storage,
        .shape = .{ 2, 1, 4 },
        .strides = .{ 4, 99, 1 },
        .offset = 0,
    };

    try std.testing.expect(contiguous.isContiguous());
    try std.testing.expect(!transposed.isContiguous());
    try std.testing.expect(singleton_axis.isContiguous());
}

test "contiguous slices respect logical offset and length" {
    var storage = [_]i32{ -1, -1, 10, 20, 30, 40, -1 };
    var view: zgc.core.Tensor.View(i32, 2) = .{
        .storage = &storage,
        .shape = .{ 2, 2 },
        .strides = .{ 2, 1 },
        .offset = 2,
    };
    const const_view: zgc.core.Tensor.ConstView(i32, 2) = .{
        .storage = &storage,
        .shape = .{ 2, 2 },
        .strides = .{ 2, 1 },
        .offset = 2,
    };
    const strided: zgc.core.Tensor.ConstView(i32, 2) = .{
        .storage = &storage,
        .shape = .{ 2, 2 },
        .strides = .{ 3, 1 },
        .offset = 1,
    };

    const mutable_slice = view.contiguousSlice().?;
    try std.testing.expectEqualSlices(i32, &.{ 10, 20, 30, 40 }, mutable_slice);
    mutable_slice[1] = 200;
    try std.testing.expectEqual(@as(i32, 200), storage[3]);
    try std.testing.expectEqualSlices(
        i32,
        &.{ 10, 200, 30, 40 },
        const_view.contiguousSlice().?,
    );
    try std.testing.expect(strided.contiguousSlice() == null);
}

test "logical linear offsets support negative strides" {
    var storage = [_]i32{ 0, 1, 2, 3, 4, 5 };
    const reversed_columns: zgc.core.Tensor.ConstView(i32, 2) = .{
        .storage = &storage,
        .shape = .{ 2, 3 },
        .strides = .{ 3, -1 },
        .offset = 2,
    };

    try std.testing.expectEqual(@as(usize, 2), reversed_columns.elementOffsetFromLinear(0));
    try std.testing.expectEqual(@as(usize, 0), reversed_columns.elementOffsetFromLinear(2));
    try std.testing.expectEqual(@as(usize, 5), reversed_columns.elementOffsetFromLinear(3));
    try std.testing.expectEqual(@as(usize, 3), reversed_columns.elementOffsetFromLinear(5));
}

test "dense slices expose a physically contiguous axis permutation" {
    var storage = [_]i32{ 1, 4, 2, 5, 3, 6 };
    const view: zgc.core.Tensor.View(i32, 2) = .{
        .storage = &storage,
        .shape = .{ 2, 3 },
        .strides = .{ 1, 2 },
        .offset = 0,
    };

    try std.testing.expect(view.contiguousSlice() == null);
    try std.testing.expectEqualSlices(i32, &storage, view.denseSlice().?);
}

test "axis slices preserve storage, offsets, and selected strides" {
    var storage = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    var view: zgc.core.Tensor.View(i32, 3) = .{
        .storage = &storage,
        .shape = .{ 2, 2, 3 },
        .strides = .{ 6, 3, 1 },
        .offset = 0,
    };

    const last_axis = view.axisSlice(2, 2);
    try std.testing.expectEqual([1]usize{3}, last_axis.shape);
    try std.testing.expectEqual([1]isize{1}, last_axis.strides);
    try std.testing.expectEqual(@as(usize, 6), last_axis.offset);
    try std.testing.expectEqualSlices(i32, &.{ 6, 7, 8 }, last_axis.contiguousSlice().?);

    const middle_axis = view.axisSlice(1, 5);
    try std.testing.expectEqual([1]usize{2}, middle_axis.shape);
    try std.testing.expectEqual([1]isize{3}, middle_axis.strides);
    try std.testing.expectEqual(@as(usize, 8), middle_axis.offset);
    try std.testing.expectEqual(@as(i32, 8), middle_axis.get(.{0}));
    try std.testing.expectEqual(@as(i32, 11), middle_axis.get(.{1}));

    middle_axis.set(.{1}, 111);
    try std.testing.expectEqual(@as(i32, 111), storage[11]);
}

test "const axis slices support offsets and negative selected strides" {
    const storage = [_]i32{ 0, 1, 2, 3, 4, 5 };
    const view: zgc.core.Tensor.ConstView(i32, 2) = .{
        .storage = &storage,
        .shape = .{ 2, 3 },
        .strides = .{ 3, -1 },
        .offset = 2,
    };

    const second_row = view.axisSlice(1, 1);
    try std.testing.expectEqual(@as(usize, 5), second_row.offset);
    try std.testing.expectEqual([1]isize{-1}, second_row.strides);
    try std.testing.expectEqual(@as(i32, 5), second_row.get(.{0}));
    try std.testing.expectEqual(@as(i32, 3), second_row.get(.{2}));
    try std.testing.expect(second_row.contiguousSlice() == null);
}

test "broadcast views introduce zero strides without allocating storage" {
    var storage = [_]i32{ 10, 20, 30 };
    const vector: zgc.core.Tensor.View(i32, 1) = .{
        .storage = &storage,
        .shape = .{3},
        .strides = .{1},
        .offset = 0,
    };
    const matrix = vector.broadcastTo(2, .{ 2, 3 });

    try std.testing.expectEqual([2]usize{ 2, 3 }, matrix.shape);
    try std.testing.expectEqual([2]isize{ 0, 1 }, matrix.strides);
    try std.testing.expectEqual(@as(i32, 20), matrix.get(.{ 0, 1 }));
    try std.testing.expectEqual(@as(i32, 20), matrix.get(.{ 1, 1 }));

    matrix.set(.{ 1, 2 }, 300);
    try std.testing.expectEqual(@as(i32, 300), storage[2]);
}

test "broadcast views expand singleton axes and rank-zero scalars" {
    const matrix_storage = [_]i32{ 1, 2 };
    const scalar_storage = [_]i32{7};
    const matrix: zgc.core.Tensor.ConstView(i32, 2) = .{
        .storage = &matrix_storage,
        .shape = .{ 2, 1 },
        .strides = .{ 1, 1 },
        .offset = 0,
    };
    const scalar: zgc.core.Tensor.ConstView(i32, 0) = .{
        .storage = &scalar_storage,
        .shape = .{},
        .strides = .{},
        .offset = 0,
    };

    const expanded_matrix = matrix.broadcastTo(3, .{ 4, 2, 3 });
    const expanded_scalar = scalar.broadcastTo(3, .{ 4, 2, 3 });
    try std.testing.expectEqual([3]isize{ 0, 1, 0 }, expanded_matrix.strides);
    try std.testing.expectEqual([3]isize{ 0, 0, 0 }, expanded_scalar.strides);
    try std.testing.expectEqual(@as(i32, 2), expanded_matrix.get(.{ 3, 1, 2 }));
    try std.testing.expectEqual(@as(i32, 7), expanded_scalar.get(.{ 3, 1, 2 }));
}
