const std = @import("std");
const Graph = @import("../core/graph.zig");
const Op = @import("../operations/semantic.zig").Op;
const Tensor = @import("../core/tensor.zig");
const layout_ops = @import("../kernels/layout.zig");
const Plan = @import("../execution/execution.zig");
const fusion = @import("optimization/fusion/expression.zig");
const matmul = @import("planning/contraction.zig");

/// Validates the constructed semantic graph before analysis.
pub fn Validation(comptime capacity: Graph.Capacity) type {
    return struct {
        pub fn validate(comptime lowered_graph: Graph.Graph(capacity, Op)) type {
            inline for (0..lowered_graph.node_ct) |node_id| {
                const node = lowered_graph.nodes[node_id].?;
                const output = lowered_graph.tensors[node.result].?;
                const Inputs = [node.input_count]Graph.Graph(capacity, Op).TensorInfo;
                var inputs: Inputs = undefined;
                inline for (0..node.input_count) |input_index| {
                    const tensor_id = lowered_graph.input_refs[node.input_start + input_index].?;
                    inputs[input_index] = lowered_graph.tensors[tensor_id].?;
                }

                switch (node.op) {
                    .view => |view| {
                        const expected = layout_ops.infer(view, &inputs, output.shape, capacity.max_rank);
                        if (!std.mem.eql(usize, expected.shape.slice(), output.shape.slice()) or
                            expected.layout.offset != output.layout.offset or
                            !std.mem.eql(
                                isize,
                                expected.layout.strides[0..output.shape.rank],
                                output.layout.strides[0..output.shape.rank],
                            ) or expected.storage_tensor != output.storage_tensor)
                        {
                            @compileError("lowered view metadata does not match its inferred alias");
                        }
                        if (output.dtype != inputs[0].dtype) {
                            @compileError("view output dtype does not match its input dtype");
                        }
                    },
                    .compute => |compute| {
                        const expected_shape = compute.inferShape(&inputs, capacity.max_rank);
                        if (!std.mem.eql(usize, expected_shape.slice(), output.shape.slice())) {
                            @compileError("lowered operation output shape does not match its inferred shape");
                        }
                        const expected_dtype = compute.inferDtype(&inputs);
                        if (output.dtype != expected_dtype) {
                            @compileError("lowered operation output dtype does not match its inferred dtype");
                        }
                    },
                }
            }
            return struct {
                pub const graph = lowered_graph;

                pub fn View(comptime tensor_id: Tensor.Id) type {
                    const info = tensorInfo(tensor_id);
                    return Tensor.StaticView(
                        info.dtype.Scalar(),
                        info.shape.dims[0..info.shape.rank].*,
                        info.layout.strides[0..info.shape.rank].*,
                        info.layout.offset,
                    );
                }

                pub fn ConstView(comptime tensor_id: Tensor.Id) type {
                    const info = tensorInfo(tensor_id);
                    return Tensor.StaticConstView(
                        info.dtype.Scalar(),
                        info.shape.dims[0..info.shape.rank].*,
                        info.layout.strides[0..info.shape.rank].*,
                        info.layout.offset,
                    );
                }

                fn tensorInfo(comptime tensor_id: Tensor.Id) Graph.Graph(capacity, Op).TensorInfo {
                    if (tensor_id >= graph.tensor_ct) {
                        @compileError("tensor id is outside the validated graph");
                    }
                    return graph.tensors[tensor_id].?;
                }
            };
        }
    };
}

/// Validates optimizer output before lifetime analysis and model generation.
/// Failures here indicate an invalid compiler rewrite or execution plan.
pub fn FinalValidation(comptime capacity: Graph.Capacity) type {
    return struct {
        pub fn validate(comptime executable: anytype) type {
            const Program = @TypeOf(executable);
            inline for (0..executable.node_ct) |node_id| {
                const node = executable.nodes[node_id].?;
                const Inputs = [node.input_count]Program.TensorInfo;
                var inputs: Inputs = undefined;
                inline for (0..node.input_count) |input_index| {
                    const tensor_id = executable.input_refs[node.input_start + input_index].?;
                    inputs[input_index] = executable.tensors[tensor_id].?;
                }
                const Outputs = [node.output_count]Program.TensorInfo;
                var outputs: Outputs = undefined;
                inline for (0..node.output_count) |output_index| {
                    const tensor_id = executable.output_refs[node.output_start + output_index].?;
                    outputs[output_index] = executable.tensors[tensor_id].?;
                }

                switch (node.op) {
                    .view => |view| {
                        requireOutputCount(node.output_count, 1);
                        validateView(view, &inputs, outputs[0]);
                    },
                    .compute => |compute| switch (compute) {
                        .direct => |semantic| {
                            requireOutputCount(node.output_count, 1);
                            validateSemantic(semantic, &inputs, outputs[0]);
                        },
                        .kernel => |kernel_plan| switch (kernel_plan) {
                            .map => |plan| validateMapPlan(plan, &inputs, &outputs),
                            .reduction => |plan| validateReductionPlan(plan, &inputs, &outputs),
                            .contraction => |plan| {
                                validateContractionPlan(plan, &inputs, &outputs);
                            },
                        },
                    },
                }
            }

            return struct {
                pub const graph = executable;

                pub fn View(comptime tensor_id: Tensor.Id) type {
                    const info = tensorInfo(tensor_id);
                    return Tensor.StaticView(
                        info.dtype.Scalar(),
                        info.shape.dims[0..info.shape.rank].*,
                        info.layout.strides[0..info.shape.rank].*,
                        info.layout.offset,
                    );
                }

                pub fn ConstView(comptime tensor_id: Tensor.Id) type {
                    const info = tensorInfo(tensor_id);
                    return Tensor.StaticConstView(
                        info.dtype.Scalar(),
                        info.shape.dims[0..info.shape.rank].*,
                        info.layout.strides[0..info.shape.rank].*,
                        info.layout.offset,
                    );
                }

                fn tensorInfo(comptime tensor_id: Tensor.Id) Program.TensorInfo {
                    if (tensor_id >= graph.tensor_ct) @compileError("tensor id is outside the final validated graph");
                    return graph.tensors[tensor_id].?;
                }
            };
        }

        fn validateView(comptime view: Op.View, comptime inputs: anytype, comptime output: anytype) void {
            const expected = layout_ops.infer(view, inputs, output.shape, capacity.max_rank);
            if (!std.mem.eql(usize, expected.shape.slice(), output.shape.slice()) or
                expected.layout.offset != output.layout.offset or
                !std.mem.eql(isize, expected.layout.strides[0..output.shape.rank], output.layout.strides[0..output.shape.rank]) or
                expected.storage_tensor != output.storage_tensor)
            {
                @compileError("optimized view metadata does not match its inferred alias");
            }
            if (output.dtype != inputs[0].dtype) @compileError("optimized view output dtype does not match its input");
        }

        fn validateSemantic(comptime compute: Op.Compute, comptime inputs: anytype, comptime output: anytype) void {
            const expected_shape = compute.inferShape(inputs, capacity.max_rank);
            if (!std.mem.eql(usize, expected_shape.slice(), output.shape.slice())) {
                @compileError("optimized operation output shape does not match semantic inference");
            }
            if (output.dtype != compute.inferDtype(inputs)) {
                @compileError("optimized operation output dtype does not match semantic inference");
            }
        }

        fn requireOutputCount(comptime actual: usize, comptime expected: usize) void {
            if (actual != expected) @compileError("executable operation has an invalid output count");
        }

        fn validateMapPlan(comptime plan: Plan.MapPlan, comptime inputs: anytype, comptime outputs: anytype) void {
            if (plan.region.stores.len != outputs.len) @compileError("map stores must match invocation outputs");
            if (outputs.len != 1) @compileError("multi-store map execution is not implemented");
            const store = plan.region.stores[0];
            if (store.output != 0) @compileError("single-output map store must target output zero");
            if (!std.mem.eql(usize, plan.region.domain.shape, outputs[0].shape.slice())) {
                @compileError("map domain must match its output shape");
            }
            if (plan.region.loads.len != inputs.len) @compileError("map loads must match invocation inputs");
            for (plan.region.loads, 0..) |load, index| {
                if (load.input != index) @compileError("map loads must identify invocation inputs in order");
            }
            switch (plan.strategy) {
                .traversal => |traversal| {
                    const expression = switch (plan.region.body) {
                        .expression => |expression| expression,
                        .transfer, .expression_transfer => @compileError("map traversal requires an expression body"),
                    };
                    if (store.access != .logical) @compileError("expression map stores require logical access");
                    validateElementwiseProgram(expression, inputs, outputs[0]);
                    switch (store.value) {
                        .expression => |value| switch (value) {
                            .instruction => |index| if (index != expression.instructions.len - 1) {
                                @compileError("map executor requires its store to reference the final instruction");
                            },
                            .input, .accumulator => @compileError("map store must reference an expression instruction"),
                        },
                        .accumulator, .contraction, .transfer => {
                            @compileError("map executor requires its store to reference the final instruction");
                        },
                    }
                    if (traversal.vector_width == 0 or traversal.unroll == 0) {
                        @compileError("map traversal factors must be nonzero");
                    }
                    if (traversal.axis_order.len != outputs[0].shape.rank) {
                        @compileError("map traversal plan must order every output axis");
                    }
                    var seen: [capacity.max_rank]bool = @splat(false);
                    for (traversal.axis_order) |axis| {
                        if (axis >= outputs[0].shape.rank or seen[axis]) @compileError("map axis order is invalid");
                        seen[axis] = true;
                    }
                    if (traversal.vector_axis) |axis| {
                        if (axis >= outputs[0].shape.rank) @compileError("map vector axis is outside the output rank");
                        if (traversal.axis_order[traversal.axis_order.len - 1] != axis) {
                            @compileError("map vector axis must be the innermost planned loop");
                        }
                        if (outputs[0].shape.at(axis) < traversal.vector_width or outputs[0].layout.strides[axis] != 1) {
                            @compileError("map vector axis must be contiguous and at least one vector wide");
                        }
                        for (inputs) |input| {
                            const leading_axes = outputs[0].shape.rank - input.shape.rank;
                            if (axis < leading_axes) continue;
                            const input_axis = axis - leading_axes;
                            if (input.shape.at(input_axis) == 1 and outputs[0].shape.at(axis) != 1) continue;
                            if (input.layout.strides[input_axis] != 1) {
                                @compileError("map vector inputs must be contiguous or broadcast on the vector axis");
                            }
                        }
                    } else if (traversal.vector_width != 1) {
                        @compileError("scalar map traversal plans must have vector width one");
                    }
                },
                .segmented => |segmented| {
                    const expression = switch (plan.region.body) {
                        .transfer => null,
                        .expression_transfer => |program| program,
                        .expression => @compileError("segmented maps require a transfer body"),
                    };
                    if (store.access != .segmented or store.value != .transfer) {
                        @compileError("segmented map stores are defined by their transfer segments");
                    }
                    validateSegmentedMap(segmented, expression, inputs, outputs);
                },
                .loop => |loop_plan| {
                    const expression: ?fusion.Program = switch (plan.region.body) {
                        .transfer => null,
                        .expression_transfer => |program| program,
                        .expression => @compileError("loop maps require a transfer body"),
                    };
                    if (store.access != .loop or store.value != .transfer) {
                        @compileError("loop map stores are defined by their loop plan");
                    }
                    if (loop_plan.axis >= outputs[0].shape.rank) @compileError("loop axis is outside the output rank");
                    if (loop_plan.iterations.len != outputs[0].shape.at(loop_plan.axis)) {
                        @compileError("loop iterations must match the loop-axis extent");
                    }
                    if (loop_plan.vector_width == 0) @compileError("loop vector width must be nonzero");
                    if (loop_plan.vector_width > 1 and loop_plan.axis != outputs[0].shape.rank - 1) {
                        @compileError("vector loop execution requires the innermost axis");
                    }
                    for (loop_plan.iterations) |iteration| {
                        if (iteration.offsets.len + 1 != outputs[0].shape.rank) {
                            @compileError("loop offsets must cover every non-loop axis");
                        }
                        switch (iteration.boundary) {
                            .wrap => {},
                            .redirect => |redirect| if (redirect >= loop_plan.iterations.len) {
                                @compileError("loop redirect is outside the loop-axis extent");
                            },
                        }
                    }
                    if (expression) |program| {
                        validateElementwiseProgramForShape(program, inputs, plan.region.domain.shape, outputs[0].dtype);
                        validateExpressionResult(program, loop_plan.value, inputs, outputs[0].dtype);
                    } else switch (loop_plan.value) {
                        .input => |index| {
                            if (index >= inputs.len or inputs[index].dtype != outputs[0].dtype) {
                                @compileError("loop transfer input must match its output dtype");
                            }
                        },
                        .instruction, .accumulator => @compileError("loop transfer must reference an input"),
                    }
                },
            }
        }

        fn validateSegmentedMap(
            comptime plan: Plan.MapPlan.SegmentedPlan,
            comptime expression: ?fusion.Program,
            comptime inputs: anytype,
            comptime outputs: anytype,
        ) void {
            requireOutputCount(outputs.len, 1);
            if (plan.segments.len == 0) @compileError("segmented map plan requires at least one segment");
            if (plan.vector_width == 0) @compileError("segmented map vector width must be nonzero");
            const output = outputs[0];
            var written_elements: usize = 0;
            for (plan.segments) |segment| {
                if (segment.rank != output.shape.rank) @compileError("remap segment rank must match its output");
                var destination_max: isize = @intCast(segment.destination_offset);
                for (0..segment.rank) |axis| {
                    const extent = segment.extents[axis];
                    if (extent == 0) @compileError("remap segment extents must be nonzero");
                    const distance: isize = @intCast(extent - 1);
                    const destination_delta = distance * segment.destination_strides[axis];
                    if (destination_delta < 0 and destination_max + destination_delta < 0) @compileError("remap destination geometry is outside storage");
                    destination_max += @max(destination_delta, 0);
                }
                const destination_bounds = viewBounds(output);
                if (segment.destination_offset < destination_bounds.minimum or destination_max > destination_bounds.maximum) {
                    @compileError("remap destination geometry is outside its output view");
                }
                if (segment.expression_value) |value| {
                    const program = expression orelse @compileError("expression segment requires an expression body");
                    const shape = segment.expression_shape[0..segment.expression_rank];
                    validateElementwiseProgramForShape(program, inputs, shape, output.dtype);
                    validateExpressionResult(program, value, inputs, output.dtype);
                    var expression_max: isize = @intCast(segment.expression_offset);
                    for (0..segment.rank) |axis| {
                        const delta = @as(isize, @intCast(segment.extents[axis] - 1)) * segment.expression_strides[axis];
                        if (delta < 0 and expression_max + delta < 0) @compileError("composed expression access is outside its domain");
                        expression_max += @max(delta, 0);
                    }
                    var expression_elements: usize = 1;
                    for (shape) |extent| expression_elements *= extent;
                    if (expression_max >= expression_elements) @compileError("composed expression access is outside its domain");
                } else {
                    if (segment.input >= inputs.len) @compileError("remap segment refers to an unknown input");
                    if (inputs[segment.input].dtype != output.dtype) @compileError("remap input and output dtypes must match");
                    var source_max: isize = @intCast(segment.source_offset);
                    for (0..segment.rank) |axis| {
                        const delta = @as(isize, @intCast(segment.extents[axis] - 1)) * segment.source_strides[axis];
                        if (delta < 0 and source_max + delta < 0) @compileError("remap source geometry is outside storage");
                        source_max += @max(delta, 0);
                    }
                    const source_bounds = viewBounds(inputs[segment.input]);
                    if (segment.source_offset < source_bounds.minimum or source_max > source_bounds.maximum) {
                        @compileError("remap source geometry is outside its input view");
                    }
                }
                written_elements += segment.elementCount();
            }
            if (written_elements != output.shape.elementCount()) {
                @compileError("remap segments must cover the output exactly once");
            }
        }

        fn validateContractionPlan(comptime plan: Plan.ContractionPlan, comptime inputs: anytype, comptime outputs: anytype) void {
            requireOutputCount(outputs.len, 1);
            if (inputs.len != 2) @compileError("contraction plan requires two inputs");
            if (plan.region.loads.len != inputs.len) @compileError("contraction loads must match invocation inputs");
            for (plan.region.loads, 0..) |load, index| {
                if (load.input != index or load.access != .logical) {
                    @compileError("contraction loads must identify logical invocation inputs in order");
                }
            }
            if (plan.region.stores.len != 1) @compileError("contraction plan requires one store");
            const store = plan.region.stores[0];
            if (store.output != 0 or store.access != .logical or store.value != .contraction) {
                @compileError("contraction result must store directly to output zero");
            }
            if (!std.mem.eql(usize, plan.region.domain.shape, outputs[0].shape.slice())) {
                @compileError("contraction domain must match its output shape");
            }
            validateSemantic(.matmul, inputs, outputs[0]);
            matmul.validate(plan, inputs[0], inputs[1], outputs[0]);
        }

        fn viewBounds(comptime info: anytype) struct { minimum: usize, maximum: isize } {
            var minimum: isize = @intCast(info.layout.offset);
            var maximum: isize = @intCast(info.layout.offset);
            for (info.shape.slice(), info.layout.strides[0..info.shape.rank]) |extent, stride| {
                const delta = @as(isize, @intCast(extent - 1)) * stride;
                minimum += @min(delta, 0);
                maximum += @max(delta, 0);
            }
            if (minimum < 0) @compileError("tensor view addresses storage before its origin");
            return .{ .minimum = @intCast(minimum), .maximum = maximum };
        }

        fn validateReductionPlan(comptime plan: Plan.ReductionPlan, comptime inputs: anytype, comptime outputs: anytype) void {
            const region = plan.region;
            const traversal_plan = plan.traversal_plan;
            const rank = region.domain.shape.len;

            if (outputs.len == 0) @compileError("reduction plan requires an output");
            if (plan.region.stores.len != outputs.len) @compileError("reduction stores must match invocation outputs");
            if (plan.region.accumulators.len == 0) @compileError("reduction plan requires an accumulator");
            if (region.loads.len != inputs.len) @compileError("reduction loads must match invocation inputs");
            for (region.loads, 0..) |load, index| {
                if (load.input != index or load.access != .logical) {
                    @compileError("reduction loads must identify logical invocation inputs in order");
                }
            }
            if (rank > 64 or region.reduction_axes == 0 or
                (rank < 64 and region.reduction_axes >= (@as(u64, 1) << @intCast(rank))))
            {
                @compileError("reduction plan axes are outside its domain rank");
            }
            for (region.domain.shape) |extent| {
                if (extent == 0) @compileError("reduction domain extents must be nonzero");
            }

            const dtype = outputs[0].dtype;
            if (dtype.kind() == .boolean) @compileError("reduction plans require a numeric dtype");
            for (inputs) |input| {
                if (input.dtype != dtype) @compileError("reduction expression inputs must match the output dtype");
                if (!broadcastsToShape(input, region.domain.shape)) {
                    @compileError("reduction expression input does not broadcast to the reduction domain");
                }
            }
            for (outputs) |output| {
                if (output.dtype != dtype) @compileError("reduction outputs must have matching dtypes");
                if (!isReductionOutputShape(region.domain.shape, region.reduction_axes, region.keep_dims, output)) {
                    @compileError("reduction output shape does not match its domain and axes");
                }
            }

            validateReductionExpressions(region.expressions, inputs.len, dtype);

            for (region.accumulators) |accumulator| {
                validateReductionValueRef(
                    accumulator.update,
                    inputs.len,
                    region.expressions.instructions.len,
                    0,
                );
                if (accumulator.finalize == .mean and dtype.kind() != .float) {
                    @compileError("mean reduction finalization requires a floating-point dtype");
                }
            }

            var stored_outputs: [outputs.len]bool = @splat(false);
            var stored_accumulators: [region.accumulators.len]bool = @splat(false);
            for (region.stores) |store| {
                if (store.output >= outputs.len) @compileError("reduction store refers to an unknown output");
                if (stored_outputs[store.output]) @compileError("reduction output has more than one store");
                stored_outputs[store.output] = true;
                if (store.access != .logical) @compileError("reduction stores require logical access");
                const accumulator_index = switch (store.value) {
                    .accumulator => |index| index,
                    .expression, .contraction, .transfer => @compileError("reduction stores must refer to accumulators"),
                };
                if (accumulator_index >= region.accumulators.len) {
                    @compileError("reduction store refers to an unknown accumulator");
                }
                stored_accumulators[accumulator_index] = true;
            }
            for (stored_outputs) |stored| {
                if (!stored) @compileError("reduction output is missing a store");
            }
            for (stored_accumulators) |stored| {
                if (!stored) @compileError("reduction plan contains an unused accumulator");
            }

            if (traversal_plan.vector_width == 0 or traversal_plan.accumulator_lanes == 0 or traversal_plan.unroll == 0) {
                @compileError("reduction traversal factors must be nonzero");
            }
            if (traversal_plan.vector_axis == null and traversal_plan.vector_width != 1) {
                @compileError("a scalar reduction traversal plan must have vector width one");
            }
            if (traversal_plan.vector_axis) |axis| {
                if (axis >= rank) @compileError("reduction vector axis is outside the domain rank");
                if (region.reduction_axes & (@as(u64, 1) << @intCast(axis)) == 0) {
                    @compileError("reduction vector axis must be a reduced axis");
                }
                if (region.domain.shape[axis] < traversal_plan.vector_width) {
                    @compileError("reduction vector axis is shorter than its vector width");
                }
                for (inputs) |input| {
                    if (!supportsReductionVectorAxis(input, region.domain.shape, axis)) {
                        @compileError("reduction input is neither contiguous nor broadcast on its vector axis");
                    }
                }
            }
            validateReductionAxisOrder(
                rank,
                region.reduction_axes,
                traversal_plan.outer_axis_order,
                traversal_plan.reduction_axis_order,
            );
        }

        fn supportsReductionVectorAxis(comptime input: anytype, comptime domain_shape: []const usize, comptime axis: usize) bool {
            const leading_axes = domain_shape.len - input.shape.rank;
            if (axis < leading_axes) return true;
            const input_axis = axis - leading_axes;
            if (input.shape.at(input_axis) == 1 and domain_shape[axis] != 1) return true;
            return input.layout.strides[input_axis] == 1;
        }

        fn validateReductionExpressions(
            comptime program: fusion.Program,
            comptime input_count: usize,
            comptime dtype: @import("../storage/dtype.zig").Dtype,
        ) void {
            for (program.instructions, 0..) |instruction, instruction_index| {
                if (instruction.dtype != dtype) {
                    @compileError("reduction expression instruction dtype must match the reduction dtype");
                }
                if (!instruction.operation.acceptsDtype(dtype)) {
                    @compileError("reduction expression instruction does not accept the reduction dtype");
                }
                for (instruction.args[0..instruction.operation.arity()]) |reference| {
                    validateReductionValueRef(reference, input_count, instruction_index, 0);
                }
            }
        }

        fn validateReductionValueRef(
            comptime reference: fusion.Program.ValueRef,
            comptime input_count: usize,
            comptime instruction_limit: usize,
            comptime accumulator_count: usize,
        ) void {
            switch (reference) {
                .input => |index| if (index >= input_count) {
                    @compileError("reduction expression refers to an unknown input");
                },
                .instruction => |index| if (index >= instruction_limit) {
                    @compileError("reduction expression must refer only to prior instructions");
                },
                .accumulator => |index| if (index >= accumulator_count) {
                    @compileError("reduction expression cannot read an accumulator at this stage");
                },
            }
        }

        fn validateReductionAxisOrder(
            comptime rank: usize,
            comptime reduction_axes: u64,
            comptime outer_order: []const u8,
            comptime reduction_order: []const u8,
        ) void {
            if (outer_order.len + reduction_order.len != rank) {
                @compileError("reduction traversal axes must partition the domain");
            }
            var seen: [rank]bool = @splat(false);
            for (outer_order) |axis| {
                if (axis >= rank or seen[axis]) @compileError("reduction outer-axis order is invalid");
                if (reduction_axes & (@as(u64, 1) << @intCast(axis)) != 0) {
                    @compileError("reduction outer-axis order contains a reduced axis");
                }
                seen[axis] = true;
            }
            for (reduction_order) |axis| {
                if (axis >= rank or seen[axis]) @compileError("reduction axis order is invalid");
                if (reduction_axes & (@as(u64, 1) << @intCast(axis)) == 0) {
                    @compileError("reduction axis order contains a retained axis");
                }
                seen[axis] = true;
            }
        }

        fn isReductionOutputShape(
            comptime domain_shape: []const usize,
            comptime reduction_axes: u64,
            comptime keep_dims: bool,
            comptime output: anytype,
        ) bool {
            const expected_rank = if (keep_dims) domain_shape.len else domain_shape.len - @popCount(reduction_axes);
            if (output.shape.rank != expected_rank) return false;
            var output_axis: usize = 0;
            for (domain_shape, 0..) |extent, domain_axis| {
                const reduced = reduction_axes & (@as(u64, 1) << @intCast(domain_axis)) != 0;
                if (reduced and !keep_dims) continue;
                const expected_extent: usize = if (reduced) 1 else extent;
                if (output.shape.at(output_axis) != expected_extent) return false;
                output_axis += 1;
            }
            return true;
        }

        fn broadcastsToShape(comptime input: anytype, comptime shape: []const usize) bool {
            if (input.shape.rank > shape.len) return false;
            for (0..input.shape.rank) |axis_from_end| {
                const input_extent = input.shape.at(input.shape.rank - 1 - axis_from_end);
                const output_extent = shape[shape.len - 1 - axis_from_end];
                if (input_extent != 1 and input_extent != output_extent) return false;
            }
            return true;
        }

        fn validateElementwiseProgram(comptime program: fusion.Program, comptime inputs: anytype, comptime output: anytype) void {
            validateElementwiseProgramForShape(program, inputs, output.shape.slice(), output.dtype);
        }

        fn validateElementwiseProgramForShape(
            comptime program: fusion.Program,
            comptime inputs: anytype,
            comptime shape: []const usize,
            comptime output_dtype: @import("../storage/dtype.zig").Dtype,
        ) void {
            if (program.instructions.len == 0) @compileError("fused elementwise program must contain an instruction");
            for (inputs, 0..) |input, input_index| {
                if (!programUsesInput(program, input_index)) continue;
                if (!broadcastsToShape(input, shape)) @compileError("fused elementwise input does not broadcast to its output shape");
            }
            for (program.instructions, 0..) |instruction, instruction_index| {
                var operand_dtypes: [instruction.operation.arity()]@import("../storage/dtype.zig").Dtype = undefined;
                for (instruction.args[0..instruction.operation.arity()], 0..) |reference, operand_index| {
                    switch (reference) {
                        .input => |input_index| {
                            if (input_index >= inputs.len) {
                                @compileError("fused elementwise instruction refers to an unknown input");
                            }
                            operand_dtypes[operand_index] = inputs[input_index].dtype;
                        },
                        .instruction => |prior| {
                            if (prior >= instruction_index) {
                                @compileError("fused elementwise instruction must refer only to prior instructions");
                            }
                            operand_dtypes[operand_index] = program.instructions[prior].dtype;
                        },
                        .accumulator => @compileError("map expressions cannot refer to accumulators"),
                    }
                }
                if (!instruction.operation.acceptsOperands(&operand_dtypes)) {
                    @compileError("fused pointwise instruction has invalid operand dtypes");
                }
                if (instruction.dtype != instruction.operation.inferDtype(&operand_dtypes)) {
                    @compileError("fused pointwise instruction result dtype is invalid");
                }
            }
            if (program.instructions[program.instructions.len - 1].dtype != output_dtype) {
                @compileError("fused pointwise program result dtype does not match its output");
            }
        }

        fn validateExpressionResult(
            comptime program: fusion.Program,
            comptime value: fusion.Program.ValueRef,
            comptime inputs: anytype,
            comptime output_dtype: @import("../storage/dtype.zig").Dtype,
        ) void {
            const dtype = switch (value) {
                .input => |index| if (index < inputs.len) inputs[index].dtype else @compileError("expression result refers to an unknown input"),
                .instruction => |index| if (index < program.instructions.len) program.instructions[index].dtype else @compileError("expression result refers to an unknown instruction"),
                .accumulator => @compileError("map expressions cannot refer to accumulators"),
            };
            if (dtype != output_dtype) @compileError("composed expression result dtype does not match its output");
        }

        fn programUsesInput(comptime program: fusion.Program, comptime input_index: usize) bool {
            for (program.instructions) |instruction| {
                for (instruction.args[0..instruction.operation.arity()]) |reference| {
                    switch (reference) {
                        .input => |index| if (index == input_index) return true,
                        .instruction, .accumulator => {},
                    }
                }
            }
            return false;
        }
    };
}
