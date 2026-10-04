const std = @import("std");
const Execution = @import("../execution/execution.zig");
const Op = @import("../operations/semantic.zig").Op;
const Tensor = @import("tensor.zig");

const Writer = std.Io.Writer;

pub const Sections = struct {
    capacity: bool = true,
    raw_graph: bool = true,
    semantic_graph: bool = true,
    executable: bool = true,
    structure: bool = true,
    memory_plan: bool = true,
};

/// Render the compile-time structure and storage plan of a generated model.
pub fn writeModel(
    comptime Model: type,
    writer: *Writer,
    sections: Sections,
) Writer.Error!void {
    if (sections.capacity) {
        try writer.writeAll("== Capacity ==\n");
        try writeCapacity(writer, Model.internal_capacity);
    }
    if (sections.raw_graph) {
        if (sections.capacity) try writer.writeByte('\n');
        try writer.writeAll("== Raw graph ==\n");
        try writeGraph(writer, Model.raw_graph);
    }
    if (sections.semantic_graph) {
        if (sections.capacity or sections.raw_graph) try writer.writeByte('\n');
        try writer.writeAll("== Optimized semantic graph ==\n");
        try writeGraph(writer, Model.semantic_graph);
    }
    if (sections.executable) {
        if (sections.capacity or sections.raw_graph or sections.semantic_graph) try writer.writeByte('\n');
        try writer.writeAll("== Executable ==\n");
        try writeExecutable(writer, Model.executable);
    }
    if (sections.structure) {
        if (sections.capacity or sections.raw_graph or sections.semantic_graph or sections.executable) try writer.writeByte('\n');
        try writeExecutableStructure(writer, Model.executable);
    }
    if (sections.memory_plan) {
        if (sections.capacity or sections.raw_graph or sections.semantic_graph or sections.executable or sections.structure) {
            try writer.writeByte('\n');
        }
        try writer.writeAll("== Memory plan ==\n");
        try writeMemoryPlan(Model, writer);
    }
}

pub fn writeCapacity(writer: *Writer, capacity: anytype) Writer.Error!void {
    try writer.print(
        "Capacity(nodes={d}, input_refs={d}, tensors={d}, outputs={d}, rank={d}, sources={d})\n",
        .{
            capacity.max_nodes,
            capacity.max_input_refs,
            capacity.max_tensors,
            capacity.max_outputs,
            capacity.max_rank,
            capacity.max_sources,
        },
    );
}

pub fn writeGraph(writer: *Writer, comptime graph: anytype) Writer.Error!void {
    return writeRepresentation(writer, graph, "Graph");
}

pub fn writeExecutable(writer: *Writer, comptime executable: anytype) Writer.Error!void {
    return writeRepresentation(writer, executable, "Executable");
}

fn writeRepresentation(writer: *Writer, comptime graph: anytype, comptime label: []const u8) Writer.Error!void {
    try writer.print(
        "{s}(nodes={d}/{d}, input_refs={d}/{d}, tensors={d}/{d}, outputs={d}/{d})\n",
        .{
            label,
            graph.node_ct,
            graph.nodes.len,
            graph.input_ref_ct,
            graph.input_refs.len,
            graph.tensor_ct,
            graph.tensors.len,
            graph.output_ct,
            graph.outputs.len,
        },
    );

    try writer.writeAll("Tensors:\n");
    for (graph.tensors[0..graph.tensor_ct], 0..) |maybe_info, id| {
        try writeTensorInfo(writer, id, maybe_info.?);
    }

    try writer.writeAll("Nodes:\n");
    for (graph.nodes[0..graph.node_ct], 0..) |maybe_node, id| {
        const node = maybe_node.?;
        try writer.print("  n{d} [{s}]: ", .{ id, @tagName(node.op.kind()) });
        try writeOp(writer, node.op);
        try writer.writeByte('(');
        for (0..node.input_count) |input_index| {
            if (input_index != 0) try writer.writeAll(", ");
            const ref = graph.input_refs[node.input_start + input_index].?;
            try writer.print("t{d}", .{ref});
        }
        try writer.writeAll(") -> ");
        if (comptime @hasField(@TypeOf(node), "result")) {
            try writer.print("t{d}\n", .{node.result});
        } else {
            try writer.writeByte('[');
            for (0..node.output_count) |output_index| {
                if (output_index != 0) try writer.writeAll(", ");
                const ref = graph.output_refs[node.output_start + output_index].?;
                try writer.print("t{d}", .{ref});
            }
            try writer.writeAll("]\n");
        }
    }

    try writer.writeAll("Outputs: [");
    for (graph.outputs[0..graph.output_ct], 0..) |maybe_output, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.print("t{d}", .{maybe_output.?});
    }
    try writer.writeAll("]\n");
}

pub fn writeExecutableStructure(writer: *Writer, comptime graph: anytype) Writer.Error!void {
    try writer.writeAll("Executable structure:\n");
    if (graph.output_ct == 0) {
        try writer.writeAll("  (no graph outputs)\n");
        return;
    }

    for (graph.outputs[0..graph.output_ct], 0..) |maybe_output, output_index| {
        try writer.print("output[{d}]\n", .{output_index});
        var expanded_nodes: [graph.nodes.len]bool = @splat(false);
        var ancestor_is_last: [graph.nodes.len * 2 + 2]bool = @splat(true);
        try writeTensorTree(
            writer,
            graph,
            maybe_output.?,
            0,
            true,
            &ancestor_is_last,
            &expanded_nodes,
        );
    }
}

pub fn writeMemoryPlan(comptime Model: type, writer: *Writer) Writer.Error!void {
    const graph = Model.executable;
    const plan = Model.memory_plan;
    const SourcePlan = Model.source_plan;

    try writer.print(
        "MemoryPlan(bytes={d}, alignment={d}, tensors={d})\n",
        .{ plan.byte_count, plan.alignment, plan.tensor_regions.len },
    );

    for (plan.tensor_regions, 0..) |maybe_region, tensor_id| {
        const info = graph.tensors[tensor_id].?;
        const region = maybe_region orelse {
            const binding = SourcePlan.bindingForTensor(graph.tensors[info.storage_tensor].?);
            try writer.print(
                "  t{d}: external storage={s} storage=t{d} dtype={s} shape=",
                .{ tensor_id, @tagName(binding), info.storage_tensor, @tagName(info.dtype) },
            );
            try writeShape(writer, &info.shape);
            try writer.writeByte('\n');
            continue;
        };
        const end = region.offset + region.len_bytes;
        try writer.print(
            "  t{d}: [{d}..{d}) bytes={d} align={d} storage=t{d} dtype={s} shape=",
            .{
                tensor_id,
                region.offset,
                end,
                region.len_bytes,
                region.alignment,
                info.storage_tensor,
                @tagName(info.dtype),
            },
        );
        try writeShape(writer, &info.shape);
        try writer.writeByte('\n');
    }
}

/// Render the mutable inline storage of a particular model instance.
pub fn writeModelMemory(model: anytype, writer: *Writer, byte_limit: usize) Writer.Error!void {
    const Model = @TypeOf(model.*);
    const plan = Model.memory_plan;
    const displayed_bytes = @min(byte_limit, plan.byte_count);
    try writer.print(
        "ModelMemory(bytes={d}, alignment={d}, showing={d})\n",
        .{ plan.byte_count, plan.alignment, displayed_bytes },
    );

    for (model.memory[0..displayed_bytes], 0..) |byte, offset| {
        if (offset % 16 == 0) try writer.print("  {x:0>6}: ", .{offset});
        try writer.print("{x:0>2} ", .{byte});
        if (offset % 16 == 15 or offset + 1 == displayed_bytes) {
            try writer.writeByte('\n');
        }
    }

    if (displayed_bytes < plan.byte_count) {
        try writer.print("  ... {d} bytes omitted\n", .{plan.byte_count - displayed_bytes});
    }
}

/// Parse and render one static model-inspection command. Returns false after
/// writing usage information for an invalid command.
pub fn runModuleCli(
    comptime ModelModule: type,
    args: []const []const u8,
    writer: *Writer,
) Writer.Error!bool {
    const model_count = comptime inspectableModelCount(ModelModule);
    var command_args = args;
    var requested_model: ?[]const u8 = null;

    if (args.len > 0 and std.mem.eql(u8, args[0], "--model")) {
        if (args.len < 2) {
            try writer.writeAll("--model requires an exported model declaration\n\n");
            try writeModuleCliUsage(ModelModule, writer);
            return false;
        }
        requested_model = args[1];
        command_args = args[2..];
    }

    if (requested_model) |name| {
        inline for (@typeInfo(ModelModule).@"struct".decls) |declaration| {
            const Candidate = @field(ModelModule, declaration.name);
            if (comptime isInspectableModel(Candidate)) {
                if (std.mem.eql(u8, name, declaration.name)) {
                    return runCli(Candidate, command_args, writer);
                }
            }
        }

        try writer.print("unknown model declaration: {s}\n\n", .{name});
        try writeModuleCliUsage(ModelModule, writer);
        return false;
    }

    if (model_count == 1) {
        inline for (@typeInfo(ModelModule).@"struct".decls) |declaration| {
            const Candidate = @field(ModelModule, declaration.name);
            if (comptime isInspectableModel(Candidate)) {
                return runCli(Candidate, command_args, writer);
            }
        }
    }

    if (model_count == 0) {
        try writer.writeAll("the imported model module exports no inspectable models\n");
        return false;
    }

    try writer.writeAll("the imported module exports multiple models; select one with --model\n\n");
    try writeModuleCliUsage(ModelModule, writer);
    return false;
}

/// Parse and render one static model-inspection command. Returns false after
/// writing usage information for an invalid command.
pub fn runCli(
    comptime Model: type,
    args: []const []const u8,
    writer: *Writer,
) Writer.Error!bool {
    if (args.len == 0 or std.mem.eql(u8, args[0], "all")) {
        if (args.len > 1) return invalidCommand(writer, args[1]);
        try writeModel(Model, writer, .{});
        return true;
    }
    if (args.len > 1) return invalidCommand(writer, args[1]);

    const command = args[0];
    if (std.mem.eql(u8, command, "summary")) {
        try writeModel(Model, writer, .{
            .executable = false,
            .raw_graph = false,
            .semantic_graph = false,
            .structure = false,
            .memory_plan = false,
        });
    } else if (std.mem.eql(u8, command, "executable")) {
        try writeModel(Model, writer, .{
            .capacity = false,
            .raw_graph = false,
            .semantic_graph = false,
            .structure = false,
            .memory_plan = false,
        });
    } else if (std.mem.eql(u8, command, "raw-graph")) {
        try writeModel(Model, writer, .{
            .capacity = false,
            .semantic_graph = false,
            .executable = false,
            .structure = false,
            .memory_plan = false,
        });
    } else if (std.mem.eql(u8, command, "semantic-graph")) {
        try writeModel(Model, writer, .{
            .capacity = false,
            .raw_graph = false,
            .executable = false,
            .structure = false,
            .memory_plan = false,
        });
    } else if (std.mem.eql(u8, command, "representations")) {
        try writeModel(Model, writer, .{
            .capacity = false,
            .structure = false,
            .memory_plan = false,
        });
    } else if (std.mem.eql(u8, command, "tree")) {
        try writeModel(Model, writer, .{
            .capacity = false,
            .raw_graph = false,
            .semantic_graph = false,
            .executable = false,
            .memory_plan = false,
        });
    } else if (std.mem.eql(u8, command, "memory-plan")) {
        try writeModel(Model, writer, .{
            .capacity = false,
            .raw_graph = false,
            .semantic_graph = false,
            .executable = false,
            .structure = false,
        });
    } else if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help")) {
        try writeCliUsage(writer);
    } else {
        return invalidCommand(writer, command);
    }
    return true;
}

pub fn writeCliUsage(writer: *Writer) Writer.Error!void {
    try writer.writeAll(
        \\usage: zgc-inspect [--model <declaration>] [all|summary|raw-graph|semantic-graph|executable|representations|tree|memory-plan|help]
        \\
        \\  all          capacity, graph, executable, tree, and memory plan (default)
        \\  summary      exact graph capacity
        \\  raw-graph    semantic graph before optimization
        \\  semantic-graph optimized semantic graph
        \\  executable   selected executable
        \\  representations raw graph, optimized semantic graph, and selected executable
        \\  tree         output-oriented executable structure
        \\  memory-plan  owned, bound, and embedded tensor storage
        \\  help         show this message
        \\
    );
}

pub fn writeModuleCliUsage(comptime ModelModule: type, writer: *Writer) Writer.Error!void {
    try writeCliUsage(writer);
    try writer.writeAll("\ninspectable model declarations:\n");
    inline for (@typeInfo(ModelModule).@"struct".decls) |declaration| {
        const Candidate = @field(ModelModule, declaration.name);
        if (comptime isInspectableModel(Candidate)) {
            try writer.print("  {s}\n", .{declaration.name});
        }
    }
}

fn inspectableModelCount(comptime ModelModule: type) usize {
    var count: usize = 0;
    for (@typeInfo(ModelModule).@"struct".decls) |declaration| {
        if (isInspectableModel(@field(ModelModule, declaration.name))) count += 1;
    }
    return count;
}

fn isInspectableModel(comptime Candidate: anytype) bool {
    if (@TypeOf(Candidate) != type) return false;
    return @hasDecl(Candidate, "internal_capacity") and
        @hasDecl(Candidate, "raw_graph") and
        @hasDecl(Candidate, "executable") and
        @hasDecl(Candidate, "memory_plan") and
        @hasDecl(Candidate, "source_plan");
}

fn invalidCommand(writer: *Writer, command: []const u8) Writer.Error!bool {
    try writer.print("unknown inspection command: {s}\n\n", .{command});
    try writeCliUsage(writer);
    return false;
}

fn writeTensorInfo(writer: *Writer, id: Tensor.Id, info: anytype) Writer.Error!void {
    try writer.print("  t{d}: {s} shape=", .{ id, @tagName(info.dtype) });
    try writeShape(writer, &info.shape);
    switch (info.origin) {
        .node => |node| try writer.print(" producer=n{d}\n", .{node}),
        .source => |source| try writer.print(" source={d}\n", .{source}),
        .literal => |value| {
            try writer.writeAll(" literal=");
            try writeScalarValue(writer, value);
            try writer.writeByte('\n');
        },
    }
}

fn writeScalarValue(writer: *Writer, value: @import("../storage/dtype.zig").ScalarValue) Writer.Error!void {
    switch (value.data_type) {
        .f32 => try writer.print("f32({d})", .{value.get(.f32)}),
        .f16 => try writer.print("f16({d})", .{value.get(.f16)}),
        .i8 => try writer.print("i8({d})", .{value.get(.i8)}),
        .bool => try writer.print("bool({any})", .{value.get(.bool)}),
    }
}

fn writeShape(writer: *Writer, shape: anytype) Writer.Error!void {
    try writer.writeByte('[');
    for (shape.slice(), 0..) |extent, axis| {
        if (axis != 0) try writer.writeAll(", ");
        try writer.print("{d}", .{extent});
    }
    try writer.writeByte(']');
}

fn writeOp(writer: *Writer, op: anytype) Writer.Error!void {
    if (comptime @TypeOf(op) == Op) return writeSemanticOp(writer, op);
    if (comptime @TypeOf(op) == Execution.Op) {
        return switch (op) {
            .view => |view| writeSemanticOp(writer, .{ .view = view }),
            .compute => |compute| switch (compute) {
                .direct => |semantic| writeSemanticOp(writer, .{ .compute = semantic }),
                .kernel => |plan| switch (plan) {
                    .map => |map| switch (map.strategy) {
                        .traversal => switch (map.region.body) {
                            .expression => |expression| writeElementwiseProgram(writer, expression),
                            .transfer, .expression_transfer => unreachable,
                        },
                        .segmented => |segmented| switch (map.region.body) {
                            .transfer => writer.print(
                                "map(segmented, segments={d})",
                                .{segmented.segments.len},
                            ),
                            .expression_transfer => |expression| writer.print(
                                "map(composed, instructions={d}, segments={d}, vector={d})",
                                .{ expression.instructions.len, segmented.segments.len, segmented.vector_width },
                            ),
                            .expression => unreachable,
                        },
                        .loop => |loop_plan| switch (map.region.body) {
                            .transfer => writer.print(
                                "map(loop, iterations={d}, vector={d})",
                                .{ loop_plan.iterations.len, loop_plan.vector_width },
                            ),
                            .expression_transfer => |expression| writer.print(
                                "map(loop-composed, instructions={d}, iterations={d}, vector={d})",
                                .{ expression.instructions.len, loop_plan.iterations.len, loop_plan.vector_width },
                            ),
                            .expression => unreachable,
                        },
                    },
                    .reduction => |reduction| writer.print(
                        "reduction(accumulators={d}, stores={d})",
                        .{ reduction.region.accumulators.len, reduction.region.stores.len },
                    ),
                    .contraction => |contraction| writer.print(
                        "matmul({s})",
                        .{@tagName(contraction.strategy)},
                    ),
                },
            },
        };
    }
    @compileError("inspection does not support operation type " ++ @typeName(@TypeOf(op)));
}

fn writeSemanticOp(writer: *Writer, op: Op) Writer.Error!void {
    switch (op) {
        .compute => |compute| switch (compute) {
            .relu => try writer.writeAll("relu"),
            .exp => try writer.writeAll("exp"),
            .neg => try writer.writeAll("neg"),
            .abs => try writer.writeAll("abs"),
            .sqrt => try writer.writeAll("sqrt"),
            .log => try writer.writeAll("log"),
            .reciprocal => try writer.writeAll("reciprocal"),
            .add => try writer.writeAll("add"),
            .sub => try writer.writeAll("sub"),
            .mul => try writer.writeAll("mul"),
            .div => try writer.writeAll("div"),
            .minimum => try writer.writeAll("minimum"),
            .maximum => try writer.writeAll("maximum"),
            .clamp => try writer.writeAll("clamp"),
            .equal => try writer.writeAll("equal"),
            .not_equal => try writer.writeAll("not_equal"),
            .less_than => try writer.writeAll("less_than"),
            .less_equal => try writer.writeAll("less_equal"),
            .greater_than => try writer.writeAll("greater_than"),
            .greater_equal => try writer.writeAll("greater_equal"),
            .logical_not => try writer.writeAll("logical_not"),
            .logical_and => try writer.writeAll("logical_and"),
            .logical_or => try writer.writeAll("logical_or"),
            .where => try writer.writeAll("where"),
            .copy => try writer.writeAll("copy"),
            .contiguous => try writer.writeAll("contiguous"),
            .pad => |attrs| {
                try writer.writeAll("pad(before=");
                try writeDimensions(writer, attrs.before);
                try writer.writeAll(", after=");
                try writeDimensions(writer, attrs.after);
                try writer.writeByte(')');
            },
            .shift => |attrs| {
                try writer.writeAll("shift(offsets=[");
                for (attrs.offsets, 0..) |offset, axis| {
                    if (axis != 0) try writer.writeByte(',');
                    try writer.print("{d}", .{offset});
                }
                try writer.print("], boundary={s})", .{@tagName(attrs.boundary)});
            },
            .slice_loop => |attrs| try writer.print(
                "slice_loop(axis={d}, iterations={d})",
                .{ attrs.axis, attrs.iterations.len },
            ),
            .matmul => try writer.writeAll("matmul"),
            .sum => |attrs| try writeReduction(writer, "sum", attrs),
            .mean => |attrs| try writeReduction(writer, "mean", attrs),
            .min => |attrs| try writeReduction(writer, "min", attrs),
            .max => |attrs| try writeReduction(writer, "max", attrs),
            .concat => |attrs| try writer.print("concat(axis={d})", .{attrs.axis}),
            .softmax => |attrs| try writer.print("softmax(axis={d})", .{attrs.axis}),
        },
        .view => |view| switch (view) {
            .transpose => |attrs| try writer.print(
                "transpose(axes={d},{d})",
                .{ attrs.axis_a, attrs.axis_b },
            ),
            .reshape => try writer.writeAll("reshape"),
            .flatten => |attrs| try writer.print(
                "flatten(axes={d}..{d})",
                .{ attrs.start_axis, attrs.end_axis },
            ),
            .squeeze => |attrs| try writer.print("squeeze(axis={d})", .{attrs.axis}),
            .unsqueeze => |attrs| try writer.print("unsqueeze(axis={d})", .{attrs.axis}),
            .slice => |attrs| try writer.print(
                "slice(axis={d}, start={d}, length={d}, step={d})",
                .{ attrs.axis, attrs.start, attrs.length, attrs.step },
            ),
            .broadcast => try writer.writeAll("broadcast"),
            .windows => |attrs| {
                try writer.writeAll("windows(sizes=");
                try writeDimensions(writer, attrs.sizes);
                if (attrs.strides) |strides| {
                    try writer.writeAll(", strides=");
                    try writeDimensions(writer, strides);
                }
                if (attrs.dilations) |dilations| {
                    try writer.writeAll(", dilations=");
                    try writeDimensions(writer, dilations);
                }
                try writer.writeByte(')');
            },
        },
    }
}

fn writeElementwiseProgram(writer: *Writer, program: @import("../compiler/optimization/fusion/expression.zig").Program) Writer.Error!void {
    try writer.writeAll("fused_elementwise[");
    for (program.instructions, 0..) |instruction, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.print("{s}", .{@tagName(instruction.operation)});
    }
    try writer.writeByte(']');
}

fn writeDimensions(writer: *Writer, dimensions: []const usize) Writer.Error!void {
    try writer.writeByte('[');
    for (dimensions, 0..) |dimension, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.print("{d}", .{dimension});
    }
    try writer.writeByte(']');
}

fn writeReduction(writer: *Writer, name: []const u8, attrs: Op.Compute.ReductionAttrs) Writer.Error!void {
    try writer.print("{s}(axes=[", .{name});
    var first = true;
    for (0..64) |axis| {
        if (attrs.axes & (@as(u64, 1) << @intCast(axis)) == 0) continue;
        if (!first) try writer.writeByte(',');
        try writer.print("{d}", .{axis});
        first = false;
    }
    try writer.print("], keep_dims={any})", .{attrs.keep_dims});
}

fn writeTensorTree(
    writer: *Writer,
    comptime graph: anytype,
    tensor_id: Tensor.Id,
    depth: usize,
    is_last: bool,
    ancestor_is_last: []bool,
    expanded_nodes: *[graph.nodes.len]bool,
) Writer.Error!void {
    try writeTreePrefix(writer, depth, is_last, ancestor_is_last);
    if (tensor_id >= graph.tensor_ct or graph.tensors[tensor_id] == null) {
        try writer.print("t{d} (missing tensor metadata)\n", .{tensor_id});
        return;
    }

    const info = graph.tensors[tensor_id].?;
    try writer.print("t{d} [{s} ", .{ tensor_id, @tagName(info.dtype) });
    try writeShape(writer, &info.shape);
    try writer.writeByte(']');

    const producer_id = switch (info.origin) {
        .node => |node| node,
        .source => |source_index| {
            if (comptime graph.sources.len == 0) {
                try writer.writeAll(" (invalid source)\n");
                return;
            }
            const source = graph.sources[source_index].?;
            try writer.print(" (source[{d}])={s}\n", .{ source_index, @tagName(source.kind) });
            return;
        },
        .literal => |value| {
            try writer.writeAll(" (literal=");
            try writeScalarValue(writer, value);
            try writer.writeAll(")\n");
            return;
        },
    };

    if (producer_id >= graph.node_ct or graph.nodes[producer_id] == null) {
        try writer.print(" (missing producer n{d})\n", .{producer_id});
        return;
    }
    if (expanded_nodes[producer_id]) {
        try writer.print(" (from n{d}, already shown)\n", .{producer_id});
        return;
    }

    try writer.writeByte('\n');
    expanded_nodes[producer_id] = true;
    ancestor_is_last[depth] = is_last;

    const node = graph.nodes[producer_id].?;
    try writeTreePrefix(writer, depth + 1, true, ancestor_is_last);
    try writer.print("n{d} [{s}] ", .{ producer_id, @tagName(node.op.kind()) });
    try writeOp(writer, node.op);
    try writer.writeByte('\n');

    ancestor_is_last[depth + 1] = true;
    for (0..node.input_count) |input_index| {
        const input_id = graph.input_refs[node.input_start + input_index].?;
        try writeTensorTree(
            writer,
            graph,
            input_id,
            depth + 2,
            input_index == node.input_count - 1,
            ancestor_is_last,
            expanded_nodes,
        );
    }
}

fn writeTreePrefix(
    writer: *Writer,
    depth: usize,
    is_last: bool,
    ancestor_is_last: []const bool,
) Writer.Error!void {
    for (0..depth) |ancestor_depth| {
        try writer.writeAll(if (ancestor_is_last[ancestor_depth]) "   " else "│  ");
    }
    try writer.writeAll(if (is_last) "└─ " else "├─ ");
}
